from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from stocktrade.model import (
    CausalMarketModel, FeatureScaler, TrainingConfig, TrainingResult, _Windows,
    _return_index, train_model,
)
from stocktrade.schemas import FeatureFrame


def make_frame(rows=72):
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
        prices=100 * np.exp(np.cumsum(returns)), symbol="TEST",
    )


def test_trainable_finite_allocator_and_safe_checkpoint(tmp_path):
    frame = make_frame()
    config = TrainingConfig(lookback=8, hidden_dim=12, epochs=2, batch_size=16)
    torch.manual_seed(config.seed)
    initial = CausalMarketModel(3, frame.groups, config)
    before = initial.allocator[0].weight.detach().clone()
    result = train_model(frame, 44, 56, config)
    assert all(np.isfinite(list(row.values())).all() for row in result.history)
    assert not torch.equal(before, result.model.allocator[0].weight)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in result.model.parameters())
    forecast = result.predict(frame)
    assert np.isfinite(list(forecast.__dict__.values())).all()
    assert 0 <= forecast.allocation_fraction <= config.max_allocation
    assert forecast.volatility > 0
    assert 0 <= forecast.confidence <= 1
    path = tmp_path / "checkpoint.pt"
    result.save(path)
    restored = TrainingResult.load(path)
    assert restored.predict(frame).__dict__ == pytest.approx(forecast.__dict__, abs=1e-8)
    assert restored.splits == result.splits


def test_scaler_and_training_ignore_test_rows_and_targets_stay_in_partition():
    frame = make_frame()
    config = TrainingConfig(lookback=8, hidden_dim=8, epochs=1)
    result = train_model(frame, 44, 56, config)
    poisoned = make_frame()
    poisoned.values[56:] = 100000
    alternate = train_model(poisoned, 44, 56, config)
    assert np.array_equal(result.scaler.mean, alternate.scaler.mean)
    assert np.array_equal(result.scaler.scale, alternate.scaler.scale)
    assert all(torch.equal(result.model.state_dict()[key], alternate.model.state_dict()[key]) for key in result.model.state_dict())
    values, observed = result.scaler.transform(frame)
    train = _Windows(values, observed, range(8, 44), 8)
    assert torch.equal(train[-1][2], torch.from_numpy(values[43]))
    assert train[-1][0].shape == (8, 3)
    assert result.splits == {"train": (8, 44), "validation": (44, 56), "test": (56, 72)}
    assert result.scaler.mean[2] == pytest.approx(np.mean(frame.values[:44, 2]))


def test_gru_is_causal_and_missing_sentinels_do_not_affect_predictions():
    torch.set_num_threads(1)
    frame = make_frame()
    config = TrainingConfig(lookback=8, hidden_dim=8, epochs=1)
    model = CausalMarketModel(3, frame.groups, config).eval()
    values = torch.randn(1, 12, 3)
    observed = torch.ones_like(values)
    original = model(values, observed)
    changed = values.clone()
    changed[:, 7:] += 1000
    altered = model(changed, observed)
    for name in original:
        assert torch.equal(original[name][:, :7], altered[name][:, :7])
    result = train_model(frame, 44, 56, config)
    poisoned = make_frame()
    poisoned.values[~poisoned.observed] = 1e12
    assert result.predict(frame).__dict__ == pytest.approx(result.predict(poisoned).__dict__)
    with pytest.raises(ValueError, match="complete historical window"):
        result.predict(frame, index=2)


def test_rejects_feature_order_mismatch_and_invalid_splits():
    frame = make_frame()
    scaler = FeatureScaler.fit(frame, 40)
    frame.feature_names.reverse()
    with pytest.raises(ValueError, match="names and order"):
        scaler.transform(frame)
    frame = make_frame()
    with pytest.raises(ValueError, match="lookback"):
        train_model(frame, 8, 40, TrainingConfig(lookback=8))
    frame.timestamps[1] = frame.timestamps[0]
    with pytest.raises(ValueError, match="increase strictly"):
        train_model(frame, 40, 55)


def test_inference_transforms_only_causal_window_and_ignores_future(monkeypatch):
    frame = make_frame()
    config = TrainingConfig(lookback=8, hidden_dim=8, epochs=1)
    result = train_model(frame, 44, 56, config)
    original = result.scaler.transform
    transformed_rows = []

    def track_transform(window):
        transformed_rows.append(len(window.timestamps))
        return original(window)

    monkeypatch.setattr(result.scaler, "transform", track_transform)
    forecast = result.predict(frame, index=60)
    frame.values[61:] = 1e20
    frame.timestamps[61:] = [frame.timestamps[0]] * 11
    assert result.predict(frame, index=60).__dict__ == pytest.approx(forecast.__dict__)
    assert transformed_rows == [8, 8]


def test_return_target_distinguishes_company_return_from_market_context():
    assert _return_index(["macro.market.log_return", "market.log_return"]) == 1
    assert _return_index(["macro.market.log_return", "log_return"]) == 1
    with pytest.raises(ValueError, match="market.log_return"):
        _return_index(["macro.market.log_return"])


def test_features_never_observed_in_training_remain_masked_at_inference():
    frame = make_frame()
    frame.observed[:44, 2] = False
    frame.values[:44, 2] = np.nan
    result = train_model(frame, 44, 56, TrainingConfig(lookback=8, hidden_dim=8, epochs=1))
    values, observed = result.scaler.transform(frame)
    assert result.scaler.counts[2] == 0
    assert not observed[:, 2].any()
    assert not values[:, 2].any()
    forecast = result.predict(frame)
    frame.values[44:, 2] = 1e20
    assert result.predict(frame).__dict__ == pytest.approx(forecast.__dict__)
