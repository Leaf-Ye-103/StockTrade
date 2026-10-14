from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from stocktrade.model import FeatureScaler, PolicyAgent, PolicyConfig, PolicySample
from stocktrade.execution import MatchResult, OrderIntent, PortfolioLedger
from stocktrade.policy_contracts import PortfolioState
from stocktrade.schemas import FeatureFrame


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_frame(rows=72, symbol="AAA"):
    rng = np.random.default_rng(12)
    returns = rng.normal(0.0002, 0.002, rows)
    flow = rng.normal(0, 1, rows)
    values = np.column_stack((returns, flow, np.sin(np.arange(rows) / 20))).astype(np.float32)
    observed = np.ones_like(values, dtype=bool)
    observed[::4, 1] = False
    observed[0, 0] = False
    values[~observed] = np.nan
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    return FeatureFrame(
        timestamps=[start + timedelta(seconds=i) for i in range(rows)],
        values=values, observed=observed,
        feature_names=["market.log_return", "orderbook.imbalance", "macro.rate"],
        groups={"market": [0], "orderbook": [1], "macro": [2]},
        prices=100 * np.exp(np.cumsum(returns)), symbol=symbol,
    )


def make_agent(frames=None, **config):
    frames = frames or {"AAA": make_frame(), "BBB": make_frame(symbol="BBB")}
    agent = PolicyAgent.create(frames, tuple(frames), 44, 10_000,
                               PolicyConfig(lookback=8, hidden_dim=12, **config))
    state = PortfolioState(10_000, 10_000, {}, {symbol: 100.0 for symbol in frames})
    return agent, frames, state


def platform_frame(*, observed_in_training=True):
    frame = make_frame()
    rows = len(frame.values)
    platform = np.column_stack((np.sin(np.arange(rows) / 3), np.arange(rows) % 7)).astype(np.float32)
    available = np.ones_like(platform, dtype=bool)
    if not observed_in_training:
        available[:44] = False
    return FeatureFrame(
        frame.timestamps, np.column_stack((frame.values, platform)),
        np.column_stack((frame.observed, available)),
        frame.feature_names + ["platform.sentiment", "platform.age_seconds"],
        dict(frame.groups, platform=[3, 4]), frame.prices, frame.symbol,
    )


def test_direct_cash_actions_condition_on_entire_allowed_universe_and_funding():
    agent, frames, state = make_agent()
    probabilities, actions = agent.distribution(frames, 60, state)
    assert len(actions) == 1 + len(frames) * 2 * len(agent.config.budget_fractions)
    assert actions[0].side == "hold" and actions[0].symbol is None
    assert actions[1].symbol == "AAA" and actions[1].side == "buy" and actions[1].value == 100
    assert actions[9].symbol == "BBB" and actions[9].side == "buy"
    assert all(probabilities.probs[a.action_id] == 0 for a in actions if a.side == "sell")
    assert torch.isfinite(probabilities.probs).all()
    assert probabilities.probs.sum().item() == pytest.approx(1)
    changed_funding = PortfolioState(20_000, 20_000, {}, state.prices)
    altered, alternate_actions = agent.distribution(frames, 60, changed_funding)
    assert alternate_actions[1].value == 200
    assert not torch.equal(probabilities.probs, altered.probs)
    sample = agent.act(frames, 60, state, deterministic=True)
    assert sample.action == actions[int(probabilities.probs.argmax())]
    assert not hasattr(sample, "forecast") and not hasattr(sample, "value")
    assert not hasattr(agent, "predict")
    assert set(agent.model(*agent._inputs(frames, 60, state))) == {"logits"}
    assert not any(name in key for key in agent.model.state_dict()
                   for name in ("next_mean", "next_log_scale", "reconstruction", "critic", "allocator"))


@pytest.mark.parametrize("reward", [0.1, -0.1])
def test_terminal_wealth_feedback_trains_policy_and_perception(reward):
    agent, frames, state = make_agent(entropy_weight=0, normalize_advantages=False)
    distribution, actions = agent.distribution(frames, 60, state)
    action_id = torch.tensor(1)
    before_probability = distribution.probs[1].item()
    before_encoder = agent.model.encoders[0].encoders[0][0].weight.detach().clone()
    sample = PolicySample(actions[1], distribution.log_prob(action_id), distribution.entropy())
    metrics = agent.update([sample], [reward], learning_rate=0.01)
    after, _ = agent.distribution(frames, 60, state)
    if reward > 0:
        assert after.probs[1].item() > before_probability
    else:
        assert after.probs[1].item() < before_probability
    assert not torch.equal(before_encoder, agent.model.encoders[0].encoders[0][0].weight)
    assert metrics["baseline"] == 0
    assert metrics["episode_wealth_growth"] == pytest.approx(reward)
    assert agent.wealth_mean == pytest.approx(reward) and agent.wealth_episode_count == 1
    assert "value_loss" not in metrics and "prediction_loss" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.model.parameters())


def test_small_episode_wealth_growth_is_not_erased_by_centering_on_itself():
    agent, frames, state = make_agent(entropy_weight=0)
    distribution, actions = agent.distribution(frames, 60, state)
    before = distribution.probs[1].item()
    sample = PolicySample(actions[1], distribution.log_prob(torch.tensor(1)), distribution.entropy())
    metrics = agent.update([sample], [0.001])
    after, _ = agent.distribution(frames, 60, state)
    assert after.probs[1].item() > before
    assert metrics["advantage_scale"] == pytest.approx(0.001)
    assert metrics["policy_loss"] > 0


def test_loss_making_episode_has_one_negative_wealth_signal_despite_later_collapse():
    agent, frames, buy_state = make_agent(normalize_advantages=False)
    buy_distribution, buy_actions = agent.distribution(frames, 60, buy_state)
    sale_state = PortfolioState(10_000, 9900, {"AAA": 1}, {"AAA": 95.0, "BBB": 100.0})
    sale_distribution, sale_actions = agent.distribution(frames, 61, sale_state)
    buy_id, sale_id = 1, 5
    samples = [
        PolicySample(buy_actions[buy_id], buy_distribution.log_prob(torch.tensor(buy_id)),
                     buy_distribution.entropy()),
        PolicySample(sale_actions[sale_id], sale_distribution.log_prob(torch.tensor(sale_id)),
                     sale_distribution.entropy()),
    ]
    ledger = PortfolioLedger(10_000)
    for sample, index, price in ((samples[0], 60, 100.0), (samples[1], 61, 95.0)):
        decision = frames["AAA"].timestamps[index]
        execution = decision + timedelta(milliseconds=100)
        observed_trade = MatchResult("matched", "compatible", 1, (f"observed-{index}",),
                                     price, 1, execution)
        intent = OrderIntent(decision, "AAA", sample.action.side, 1, price, price)
        assert ledger.apply_match(intent, observed_trade).status == "matched"
    ledger.mark(frames["AAA"].timestamps[62], "AAA", 10.0)
    wealth_growth = (ledger.equity - ledger.initial_cash) / ledger.initial_cash
    assert wealth_growth == pytest.approx(-5 / 10_000)
    before_log_probability = sum(sample.log_probability.item() for sample in samples)
    metrics = agent.update(samples, [wealth_growth] * len(samples), learning_rate=0.01)
    after_buy, _ = agent.distribution(frames, 60, buy_state)
    after_sale, _ = agent.distribution(frames, 61, sale_state)
    after_log_probability = (after_buy.log_prob(torch.tensor(buy_id))
                             + after_sale.log_prob(torch.tensor(sale_id))).item()
    assert after_log_probability < before_log_probability
    assert metrics["episode_wealth_growth"] < 0
    assert agent.wealth_episode_count == 1
    assert agent.wealth_mean == wealth_growth
    assert "sale_reward" not in metrics and "hold_advantage" not in metrics


@pytest.mark.parametrize("wealth_growth", [0.01, -0.01])
def test_complete_trajectory_receives_one_terminal_objective(wealth_growth):
    agent, frames, state = make_agent(normalize_advantages=False)
    samples = []
    for index in (58, 59, 60):
        distribution, actions = agent.distribution(frames, index, state)
        samples.append(PolicySample(actions[1], distribution.log_prob(torch.tensor(1)), distribution.entropy()))
    before = sum(sample.log_probability.item() for sample in samples)
    metrics = agent.update(samples, [wealth_growth] * len(samples))
    after = sum(agent.distribution(frames, index, state)[0].log_prob(torch.tensor(1)).item()
                for index in (58, 59, 60))
    assert metrics["policy_loss"] == pytest.approx(-before * wealth_growth)
    assert metrics["total_loss"] == metrics["policy_loss"]
    assert metrics["samples"] == 3 and agent.wealth_episode_count == 1
    if wealth_growth > 0:
        assert after > before
    else:
        assert after < before


def test_different_action_grades_are_rejected_and_zero_wealth_has_no_auxiliary_loss():
    agent, frames, state = make_agent()
    samples = [agent.act(frames, 60, state), agent.act(frames, 61, state)]
    with pytest.raises(ValueError, match="one terminal wealth"):
        agent.update(samples, [-0.01, 0.02])
    before = {name: value.detach().clone() for name, value in agent.model.state_dict().items()}
    metrics = agent.update(samples, [0.0, 0.0])
    assert metrics["policy_loss"] == 0 and metrics["total_loss"] == 0
    assert all(torch.equal(value, agent.model.state_dict()[name]) for name, value in before.items())
    assert agent.wealth_episode_count == 1


def test_platform_information_importance_is_learned_from_wealth_without_fixed_weights():
    frames = {"AAA": platform_frame()}
    agent, _, state = make_agent(frames)
    assert not hasattr(agent.config, "platform_weight")
    before = agent.model.encoders[0].encoders[-1][0].weight.detach().clone()
    distribution, actions = agent.distribution(frames, 60, state)
    sample = PolicySample(actions[1], distribution.log_prob(torch.tensor(1)), distribution.entropy())
    agent.update([sample], [0.01])
    assert not torch.equal(before, agent.model.encoders[0].encoders[-1][0].weight)
    previous, _ = agent.distribution(frames, 60, state)
    frames["AAA"].values[53:61, 3] += 20
    altered, _ = agent.distribution(frames, 60, state)
    assert not torch.equal(previous.probs, altered.probs)


def test_never_trained_platform_information_stays_masked_and_future_inputs_are_inaccessible():
    frames = {"AAA": platform_frame(observed_in_training=False)}
    agent, _, state = make_agent(frames)
    before, _ = agent.distribution(frames, 60, state)
    frames["AAA"].values[44:, 3:] = 1e20
    frames["AAA"].timestamps[61:] = [frames["AAA"].timestamps[0]] * 11
    altered, _ = agent.distribution(frames, 60, state)
    assert torch.equal(before.probs, altered.probs)


def test_cash_inventory_and_runtime_universe_masks_keep_actions_feasible():
    agent, frames, state = make_agent()
    limited = PortfolioState(10_000, 0, {"AAA": 1}, state.prices)
    distribution, actions = agent.distribution(frames, 60, limited)
    assert all(distribution.probs[a.action_id] == 0 for a in actions if a.side == "buy")
    assert all(distribution.probs[a.action_id] == 0 for a in actions if a.side == "sell" and a.symbol == "BBB")
    sells = [a for a in actions if a.side == "sell" and a.symbol == "AAA"]
    assert all(a.value == 100 for a in sells)
    restricted, actions = agent.distribution(frames, 60, state, allowed_symbols=("BBB",))
    assert all(restricted.probs[a.action_id] == 0 for a in actions if a.symbol == "AAA")
    for _ in range(10):
        sample = agent.act(frames, 60, state, allowed_symbols=("BBB",))
        assert sample.action.symbol in {None, "BBB"}
    for invalid in ((), ("AAA", "AAA"), ("NEW",)):
        with pytest.raises(ValueError, match="subset"):
            agent.distribution(frames, 60, state, allowed_symbols=invalid)


def test_scalers_and_initial_policy_ignore_held_out_data_completely():
    frames = {"AAA": make_frame()}
    agent, _, state = make_agent(frames)
    poisoned = {"AAA": make_frame()}
    poisoned["AAA"].values[44:] = 100000
    poisoned["AAA"].timestamps[44:] = [poisoned["AAA"].timestamps[0]] * 28
    alternate, _, _ = make_agent(poisoned)
    assert np.array_equal(agent.scalers["AAA"].mean, alternate.scalers["AAA"].mean)
    assert np.array_equal(agent.scalers["AAA"].scale, alternate.scalers["AAA"].scale)
    assert all(torch.equal(value, alternate.model.state_dict()[name])
               for name, value in agent.model.state_dict().items())
    assert agent.scalers["AAA"].mean[2] == pytest.approx(np.mean(frames["AAA"].values[:44, 2]))
    assert agent.training_last_timestamp == frames["AAA"].timestamps[43].isoformat()
    assert agent.initial_funding == state.initial_funding


def test_inference_transforms_only_past_window_and_ignores_future_and_missing_sentinels(monkeypatch):
    agent, frames, state = make_agent({"AAA": make_frame()})
    original = agent.scalers["AAA"].transform
    sizes = []

    def track(window):
        sizes.append(len(window.timestamps))
        return original(window)

    monkeypatch.setattr(agent.scalers["AAA"], "transform", track)
    before, _ = agent.distribution(frames, 60, state)
    frames["AAA"].values[61:] = 1e20
    frames["AAA"].timestamps[61:] = [frames["AAA"].timestamps[0]] * 11
    frames["AAA"].values[~frames["AAA"].observed] = 1e30
    after, _ = agent.distribution(frames, 60, state)
    assert torch.equal(before.probs, after.probs)
    assert sizes == [8, 8]
    with pytest.raises(ValueError, match="complete historical window"):
        agent.act(frames, 2, state)


def test_features_never_observed_in_training_remain_masked_at_inference():
    frame = make_frame()
    frame.observed[:44, 2] = False
    frame.values[:44, 2] = np.nan
    agent, frames, state = make_agent({"AAA": frame})
    values, observed = agent.scalers["AAA"].transform(frame)
    assert agent.scalers["AAA"].counts[2] == 0
    assert not observed[:, 2].any() and not values[:, 2].any()
    before, _ = agent.distribution(frames, 60, state)
    frame.values[44:, 2] = 1e20
    after, _ = agent.distribution(frames, 60, state)
    assert torch.equal(before.probs, after.probs)


def test_multistock_context_and_existing_holdings_change_policy_probabilities():
    agent, frames, state = make_agent()
    before, _ = agent.distribution(frames, 60, state)
    frames["BBB"].values[53:61, 1] += 20
    after, _ = agent.distribution(frames, 60, state)
    assert not torch.equal(before.probs[1:5], after.probs[1:5])
    held = PortfolioState(10_000, 5000, {"AAA": 50}, state.prices, 11_000, 10_200)
    held_probabilities, held_actions = agent.distribution(frames, 60, held)
    assert any(held_probabilities.probs[a.action_id] > 0
               for a in held_actions if a.side == "sell" and a.symbol == "AAA")
    assert not torch.equal(after.probs, held_probabilities.probs)


def test_safe_checkpoint_roundtrip_preserves_policy_and_historical_baseline(tmp_path):
    agent, frames, state = make_agent()
    agent.risk_config = {"max_position_fraction": 0.20, "max_drawdown": 0.10}
    samples = [agent.act(frames, 60, state), agent.act(frames, 61, state)]
    agent.update(samples, [0.01, 0.01])
    agent.update([agent.act(frames, 62, state)], [-0.005])
    path = tmp_path / "checkpoint.pt"
    agent.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["model_kind"] == "direct_wealth_policy"
    assert payload["format_version"] == 3
    assert payload["training_objective"] == "terminal_net_wealth_growth"
    assert "optimizer" not in payload
    restored = PolicyAgent.load(path)
    original, _ = agent.distribution(frames, 60, state)
    loaded, _ = restored.distribution(frames, 60, state)
    assert torch.equal(original.probs, loaded.probs)
    assert agent.act(frames, 60, state, deterministic=True).action == restored.act(frames, 60, state, deterministic=True).action
    assert restored.universe == agent.universe and restored.config == agent.config
    assert restored.wealth_episode_count == 2
    assert restored.wealth_mean == agent.wealth_mean and restored.wealth_m2 == agent.wealth_m2
    assert restored.training_last_timestamp == agent.training_last_timestamp
    assert restored.risk_config == agent.risk_config
    torch.save({"format_version": 2, "model_kind": "direct_action_policy"}, path)
    with pytest.raises(ValueError, match="direct-action"):
        PolicyAgent.load(path)


def test_rejects_invalid_universe_schema_alignment_and_feedback():
    frame = make_frame()
    with pytest.raises(ValueError, match="distinct"):
        PolicyAgent.create({"AAA": frame}, ("AAA", "AAA"), 44, 10_000)
    with pytest.raises(ValueError, match="supplied"):
        PolicyAgent.create({"AAA": frame}, ("BBB",), 44, 10_000)
    with pytest.raises(ValueError, match="lookback"):
        PolicyAgent.create({"AAA": frame}, ("AAA",), 8, 10_000, PolicyConfig(lookback=8))
    alternate = make_frame(symbol="BBB")
    alternate.timestamps[1] += timedelta(microseconds=1)
    with pytest.raises(ValueError, match="aligned"):
        PolicyAgent.create({"AAA": frame, "BBB": alternate}, ("AAA", "BBB"), 44, 10_000)
    agent, frames, state = make_agent()
    frames["AAA"].feature_names.reverse()
    with pytest.raises(ValueError, match="names and order"):
        agent.act(frames, 60, state)
    agent, frames, state = make_agent()
    frames["BBB"].timestamps[60] += timedelta(microseconds=1)
    with pytest.raises(ValueError, match="aligned"):
        agent.act(frames, 60, state)
    with pytest.raises(ValueError, match="nonempty"):
        agent.update([], [])
    clean = {"AAA": make_frame(), "BBB": make_frame(symbol="BBB")}
    with pytest.raises(ValueError, match="finite"):
        agent.update([agent.act(clean, 60, state)], [float("nan")])


@pytest.mark.parametrize("kwargs", [
    {"lookback": 1}, {"hidden_dim": 0}, {"budget_fractions": ()},
    {"budget_fractions": (0.1, 0.1)}, {"budget_fractions": (float("nan"),)},
    {"learning_rate": 0}, {"entropy_weight": -1}, {"advantage_scale_floor": 0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PolicyConfig(**kwargs)


def test_feature_scaler_rejects_feature_order_mismatch():
    frame = make_frame()
    scaler = FeatureScaler.fit(frame, 40)
    frame.feature_names.reverse()
    with pytest.raises(ValueError, match="names and order"):
        scaler.transform(frame)
