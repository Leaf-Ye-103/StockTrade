"""Causal multi-asset action policy trained from historical evaluator feedback.

The actor emits HOLD or a symbol, side and monetary budget. There is no price,
return, reconstructed-input or allocation-forecast training objective. The only
loss is a policy gradient from observed evaluator feedback and an exploration
regularizer. A running historical reward mean supplies a non-neural baseline.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .policy_contracts import PortfolioState, TradingAction
from .schemas import FeatureFrame


@dataclass(frozen=True)
class PolicyConfig:
    lookback: int = 16
    hidden_dim: int = 32
    budget_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    learning_rate: float = 0.001
    entropy_weight: float = 0.0001
    normalize_advantages: bool = True
    advantage_scale_floor: float = 0.001
    gradient_clip: float = 1.0
    seed: int = 7
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 2 or self.hidden_dim < 1 or self.num_threads < 1:
            raise ValueError("Lookback must be >= 2; dimensions and threads must be positive")
        fractions = tuple(self.budget_fractions)
        if (not fractions or len(set(fractions)) != len(fractions)
                or any(not math.isfinite(f) or not 0 < f <= 1 for f in fractions)):
            raise ValueError("Budget fractions must be distinct finite values in (0,1]")
        if any(not math.isfinite(v) or v <= 0
               for v in (self.learning_rate, self.gradient_clip, self.advantage_scale_floor)):
            raise ValueError("Learning rate, gradient clip and scale floor must be finite and positive")
        if not math.isfinite(self.entropy_weight) or self.entropy_weight < 0:
            raise ValueError("Entropy weight must be finite and nonnegative")
        object.__setattr__(self, "budget_fractions", fractions)


@dataclass
class FeatureScaler:
    """Statistics fitted exclusively on observed training-prefix rows."""

    mean: np.ndarray
    scale: np.ndarray
    counts: np.ndarray
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: FeatureFrame, end: int) -> "FeatureScaler":
        if not 1 <= end <= len(frame.values):
            raise ValueError("Scaler training prefix must contain at least one row")
        values = np.asarray(frame.values[:end], dtype=np.float64)
        observed = np.asarray(frame.observed[:end], dtype=bool) & np.isfinite(values)
        counts = observed.sum(axis=0)
        means = np.divide(
            np.where(observed, values, 0).sum(axis=0), counts,
            out=np.zeros(values.shape[1]), where=counts > 0,
        )
        variance = np.divide(
            np.where(observed, (values - means) ** 2, 0).sum(axis=0), counts,
            out=np.ones(values.shape[1]), where=counts > 0,
        )
        scales = np.sqrt(variance)
        scales[scales < 1e-8] = 1.0
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("Training feature statistics must be finite")
        return cls(means.astype(np.float32), scales.astype(np.float32), counts, list(frame.feature_names))

    def transform(self, frame: FeatureFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.feature_names) != self.feature_names:
            raise ValueError("Feature names and order differ from the trained model")
        values = np.asarray(frame.values, dtype=np.float64)
        observed = np.asarray(frame.observed, dtype=bool) & np.isfinite(values) & (self.counts > 0)
        # Unseen modalities remain masked until chronological retraining.
        safe = np.where(observed, values, self.mean)
        transformed = ((safe - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Scaled feature values must be finite")
        return transformed, observed


def _validate_frame(frame: FeatureFrame) -> dict[str, list[int]]:
    values = np.asarray(frame.values)
    if values.ndim != 2 or values.shape != np.asarray(frame.observed).shape:
        raise ValueError("Values and observed masks must have the same [time,feature] shape")
    if len(frame.timestamps) != len(values) or len(frame.prices) != len(values):
        raise ValueError("Timestamps, prices and feature rows must align")
    if len(frame.feature_names) != values.shape[1] or len(set(frame.feature_names)) != values.shape[1]:
        raise ValueError("Feature names must be unique and match feature columns")
    if any(t.tzinfo is None or t.utcoffset() is None for t in frame.timestamps):
        raise ValueError("Feature timestamps must be timezone-aware")
    if any(a >= b for a, b in zip(frame.timestamps, frame.timestamps[1:])):
        raise ValueError("Feature timestamps must increase strictly")
    groups = {name: list(indices) for name, indices in frame.groups.items() if indices}
    assigned = [i for indices in groups.values() for i in indices]
    if any(not isinstance(i, int) or i < 0 or i >= values.shape[1] for i in assigned):
        raise ValueError("Feature group indices are invalid")
    if len(assigned) != len(set(assigned)):
        raise ValueError("Each feature may occur in only one modality group")
    missing = sorted(set(range(values.shape[1])) - set(assigned))
    if missing:
        groups["ungrouped"] = missing
    if not groups:
        raise ValueError("At least one feature is required")
    return groups


class _AssetEncoder(nn.Module):
    """Encode modality masks and a strictly historical temporal window."""

    def __init__(self, groups: dict[str, list[int]], hidden_dim: int):
        super().__init__()
        self.groups = groups
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * len(indices), hidden_dim), nn.GELU(),
                          nn.Linear(hidden_dim, hidden_dim))
            for indices in groups.values()
        ])
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * len(groups), hidden_dim), nn.GELU())

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        encoded = [
            encoder(torch.cat((values[..., indices] * observed[..., indices], observed[..., indices]), dim=-1))
            for indices, encoder in zip(self.groups.values(), self.encoders)
        ]
        return self.fusion(torch.cat(encoded, dim=-1))


class PolicyModel(nn.Module):
    """Account-conditioned categorical actions across the allowed stocks.

    Per-asset modality encoders share a past-only GRU. Cross-asset attention
    jointly considers competing stock opportunities and current inventory.
    Its only outputs are action logits. No forecast or reward-prediction head
    exists: perception and decisions learn directly from evaluator rewards.
    """

    def __init__(self, groups: dict[str, dict[str, list[int]]], config: PolicyConfig):
        super().__init__()
        self.universe = tuple(groups)
        self.groups = groups
        self.config = config
        width = config.hidden_dim
        self.encoders = nn.ModuleList([_AssetEncoder(group, width) for group in groups.values()])
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.symbol_embedding = nn.Parameter(torch.randn(len(groups), width) * 0.02)
        self.account_encoder = nn.Sequential(nn.Linear(5, width), nn.GELU())
        self.inventory_encoder = nn.Sequential(nn.Linear(3, width), nn.GELU())
        self.attention = nn.MultiheadAttention(width, num_heads=1, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 2 * len(config.budget_fractions)),
        )
        self.hold_head = nn.Linear(2 * width, 1)

    def forward(
        self, values: list[torch.Tensor], observed: list[torch.Tensor],
        account: torch.Tensor, inventory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embeddings = []
        for encoder, x, mask in zip(self.encoders, values, observed):
            _, hidden = self.temporal(encoder(x, mask))
            embeddings.append(hidden[-1])
        assets = torch.stack(embeddings, dim=1) + self.symbol_embedding.unsqueeze(0)
        account_hidden = self.account_encoder(account)
        assets = assets + self.inventory_encoder(inventory) + account_hidden.unsqueeze(1)
        attended, _ = self.attention(assets, assets, assets, need_weights=False)
        assets = assets + attended
        context = torch.cat((assets.mean(dim=1), account_hidden), dim=-1)
        per_asset = self.action_head(torch.cat((assets, account_hidden.unsqueeze(1).expand_as(assets)), dim=-1))
        logits = torch.cat((self.hold_head(context), per_asset.flatten(start_dim=1)), dim=-1)
        return {"logits": logits}


@dataclass
class PolicySample:
    action: TradingAction
    log_probability: torch.Tensor
    entropy: torch.Tensor


@dataclass
class PolicyAgent:
    model: PolicyModel
    scalers: dict[str, FeatureScaler]
    config: PolicyConfig
    universe: tuple[str, ...]
    initial_funding: float
    train_end: int
    training_last_timestamp: str
    validation_end: int | None = None
    validation_last_timestamp: str | None = None
    _optimizer: torch.optim.Optimizer | None = None
    reward_count: int = 0
    reward_mean: float = 0.0
    reward_m2: float = 0.0

    @classmethod
    def create(
        cls, frames: dict[str, FeatureFrame], universe: tuple[str, ...],
        train_end: int, initial_funding: float, config: PolicyConfig | None = None,
    ) -> "PolicyAgent":
        config = config or PolicyConfig()
        universe = tuple(universe)
        if not universe or len(set(universe)) != len(universe) or any(not s for s in universe):
            raise ValueError("The trading universe must contain distinct nonempty symbols")
        if any(symbol not in frames for symbol in universe):
            raise ValueError("Every allowed stock must have a supplied feature frame")
        if not math.isfinite(initial_funding) or initial_funding <= 0:
            raise ValueError("Initial funding must be finite and positive")
        groups = {}
        scalers = {}
        timestamps = None
        for symbol in universe:
            frame = frames[symbol]
            if frame.symbol != symbol:
                raise ValueError("Feature-frame symbol must match the allowed stock")
            if not config.lookback < train_end <= len(frame.values):
                raise ValueError("Require lookback < train_end <= row count")
            # Training setup validates only the fitted prefix, never held-out rows.
            prefix = FeatureFrame(
                frame.timestamps[:train_end], frame.values[:train_end], frame.observed[:train_end],
                frame.feature_names, frame.groups, frame.prices[:train_end], symbol,
            )
            groups[symbol] = _validate_frame(prefix)
            if timestamps is not None and prefix.timestamps != timestamps:
                raise ValueError("Allowed stocks must have aligned training timestamps")
            timestamps = prefix.timestamps
            scalers[symbol] = FeatureScaler.fit(prefix, train_end)
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            model = PolicyModel(groups, config)
        return cls(model, scalers, config, universe, initial_funding, train_end, timestamps[-1].isoformat())

    def _inputs(
        self, frames: dict[str, FeatureFrame], index: int, state: PortfolioState,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or index < self.config.lookback - 1:
            raise ValueError("Decision index must have a complete historical window")
        if not math.isfinite(state.initial_funding) or state.initial_funding <= 0:
            raise ValueError("Initial funding must be finite and positive")
        if not math.isfinite(state.cash) or state.cash < 0:
            raise ValueError("Available cash must be finite and nonnegative")
        if any(symbol not in self.universe and amount != 0 for symbol, amount in state.holdings.items()):
            raise ValueError("Portfolio positions must belong to the allowed stock universe")
        values, masks = [], []
        timestamps = None
        inventory = []
        equity = state.equity
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Account equity must be finite and positive")
        for symbol in self.universe:
            if symbol not in frames or symbol not in state.prices:
                raise ValueError("Every allowed stock requires historical features and a current price")
            frame = frames[symbol]
            if frame.symbol != symbol or not index < len(frame.values):
                raise ValueError("Decision frame and index must match the allowed stock")
            start = index + 1 - self.config.lookback
            window = FeatureFrame(
                frame.timestamps[start:index + 1], frame.values[start:index + 1],
                frame.observed[start:index + 1], frame.feature_names, frame.groups,
                frame.prices[start:index + 1], symbol,
            )
            groups = _validate_frame(window)
            if groups != self.model.groups[symbol]:
                raise ValueError("Feature modality groups differ from the trained model")
            if timestamps is not None and window.timestamps != timestamps:
                raise ValueError("Allowed stocks must have aligned decision timestamps")
            timestamps = window.timestamps
            x, mask = self.scalers[symbol].transform(window)
            values.append(torch.from_numpy(x).unsqueeze(0))
            masks.append(torch.from_numpy(mask.astype(np.float32)).unsqueeze(0))
            price, quantity = state.prices[symbol], state.holdings.get(symbol, 0)
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Current prices must be finite and positive")
            if not isinstance(quantity, int) or quantity < 0:
                raise ValueError("Portfolio shares must be nonnegative integers")
            inventory.append([math.log1p(price) / 10, quantity * price / equity, math.log1p(quantity) / 10])
        peak = equity if state.peak_equity is None else state.peak_equity
        day_start = equity if state.day_start_equity is None else state.day_start_equity
        if any(not math.isfinite(v) or v <= 0 for v in (peak, day_start)):
            raise ValueError("Peak equity and day-start equity must be finite and positive")
        account = [math.log1p(state.initial_funding) / 10, equity / state.initial_funding,
                   state.cash / equity, max(0.0, 1 - equity / peak), equity / day_start - 1]
        return values, masks, torch.tensor([account], dtype=torch.float32), torch.tensor([inventory], dtype=torch.float32)

    def _actions(
        self, state: PortfolioState, allowed_symbols: tuple[str, ...] | None = None,
    ) -> tuple[list[TradingAction], torch.Tensor]:
        allowed = self.universe if allowed_symbols is None else tuple(allowed_symbols)
        if (not allowed or len(set(allowed)) != len(allowed)
                or any(symbol not in self.universe for symbol in allowed)):
            raise ValueError("Runtime allowed symbols must be a distinct nonempty subset of the trained universe")
        actions = [TradingAction(0, None, "hold", 0.0, 0.0)]
        mask = [True]
        for symbol in self.universe:
            price = state.prices[symbol]
            held_value = state.holdings.get(symbol, 0) * price
            for side in ("buy", "sell"):
                for fraction in self.config.budget_fractions:
                    requested = state.initial_funding * fraction
                    amount = (requested if side == "buy" or held_value == 0
                              else min(requested, held_value))
                    actions.append(TradingAction(len(actions), symbol, side, amount, fraction))
                    mask.append(symbol in allowed and amount >= price
                                and (amount <= state.cash if side == "buy" else held_value > 0))
        return actions, torch.tensor(mask, dtype=torch.bool)

    def distribution(
        self, frames: dict[str, FeatureFrame], index: int, state: PortfolioState,
        allowed_symbols: tuple[str, ...] | None = None,
    ) -> tuple[Categorical, list[TradingAction]]:
        """Return masked direct-action probabilities and cash-valued actions."""
        inputs = self._inputs(frames, index, state)
        actions, available = self._actions(state, allowed_symbols)
        output = self.model(*inputs)
        logits = output["logits"].squeeze(0).masked_fill(~available, -torch.inf)
        if not torch.isfinite(output["logits"]).all():
            raise ValueError("Nonfinite policy output")
        return Categorical(logits=logits), actions

    def act(
        self, frames: dict[str, FeatureFrame], index: int, state: PortfolioState,
        deterministic: bool = False,
        allowed_symbols: tuple[str, ...] | None = None,
    ) -> PolicySample:
        """Select a direct action; preserve its graph for delayed feedback."""
        distribution, actions = self.distribution(frames, index, state, allowed_symbols)
        action_id = distribution.probs.argmax() if deterministic else distribution.sample()
        return PolicySample(actions[int(action_id)], distribution.log_prob(action_id), distribution.entropy())

    def update(
        self, samples: list[PolicySample], feedback: list[float], learning_rate: float | None = None,
    ) -> dict[str, float]:
        """REINFORCE update from matured, observed evaluator rewards only.

        ``feedback`` contains each action's own fixed-horizon observed advantage
        over its no-action baseline. These overlapping, non-additive advantages
        must never be summed into reward-to-go: credit for a later protective
        sale must not turn a preceding harmful purchase into a rewarded action.
        There is no price or return target. Above-baseline feedback increases
        an action's likelihood; below-baseline feedback decreases it. Samples
        must precede any intervening optimizer step because they retain the
        rollout computation graphs.
        """
        if not samples or len(samples) != len(feedback):
            raise ValueError("Feedback requires matching nonempty samples and action advantages")
        if any(not math.isfinite(r) for r in feedback):
            raise ValueError("Evaluator feedback must be finite")
        rate = self.config.learning_rate if learning_rate is None else learning_rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=rate)
        else:
            for group in self._optimizer.param_groups:
                group["lr"] = rate
        targets = torch.tensor(feedback, dtype=torch.float32)
        # Prior episodes supply the mean/variance of individual action feedback,
        # never cumulative rewards. Centering this batch on its own mean would
        # erase all learning from a single evaluated action.
        baseline = self.reward_mean
        scale = 1.0
        if self.config.normalize_advantages:
            variance = self.reward_m2 / max(1, self.reward_count - 1)
            scale = max(math.sqrt(max(0.0, variance)), float(targets.std(unbiased=False)),
                        self.config.advantage_scale_floor)
        advantages = ((targets - baseline) / scale).detach()
        log_probabilities = torch.stack([sample.log_probability for sample in samples])
        entropy = torch.stack([sample.entropy for sample in samples]).mean()
        policy_loss = -(log_probabilities * advantages).mean()
        total = policy_loss - self.config.entropy_weight * entropy
        if not torch.isfinite(total):
            raise ValueError("Nonfinite evaluator-feedback objective")
        self._optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self._optimizer.step()
        for reward in feedback:
            self.reward_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            self.reward_m2 += delta * (reward - self.reward_mean)
        return {"policy_loss": float(policy_loss.detach()), "entropy": float(entropy.detach()),
                "total_loss": float(total.detach()), "mean_feedback": float(targets.mean()),
                "baseline": baseline, "advantage_scale": scale, "samples": float(len(samples))}

    def save(self, path: str | Path) -> None:
        """Save tensors and plain metadata for safe weights-only loading."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2, "model_kind": "direct_action_policy",
            "state_dict": self.model.state_dict(), "config": asdict(self.config),
            "universe": self.universe, "groups": self.model.groups,
            "initial_funding": self.initial_funding, "train_end": self.train_end,
            "training_last_timestamp": self.training_last_timestamp,
            "validation_end": self.validation_end,
            "validation_last_timestamp": self.validation_last_timestamp,
            "reward_count": self.reward_count, "reward_mean": self.reward_mean, "reward_m2": self.reward_m2,
            "scalers": {symbol: {"mean": torch.from_numpy(scaler.mean),
                                 "scale": torch.from_numpy(scaler.scale),
                                 "counts": torch.from_numpy(scaler.counts),
                                 "feature_names": scaler.feature_names}
                        for symbol, scaler in self.scalers.items()},
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyAgent":
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 2 or artifact.get("model_kind") != "direct_action_policy":
            raise ValueError("Unsupported artifact: a direct-action policy checkpoint is required")
        config = PolicyConfig(**artifact["config"])
        universe = tuple(artifact["universe"])
        scalers = {symbol: FeatureScaler(payload["mean"].numpy(), payload["scale"].numpy(),
                                         payload["counts"].numpy(), list(payload["feature_names"]))
                   for symbol, payload in artifact["scalers"].items()}
        model = PolicyModel(artifact["groups"], config)
        if model.universe != universe or set(scalers) != set(universe):
            raise ValueError("Checkpoint stock universe and feature schemas disagree")
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        return cls(model, scalers, config, universe, artifact["initial_funding"],
                   artifact["train_end"], artifact["training_last_timestamp"],
                   validation_end=artifact.get("validation_end"),
                   validation_last_timestamp=artifact.get("validation_last_timestamp"),
                   reward_count=artifact["reward_count"], reward_mean=artifact["reward_mean"],
                   reward_m2=artifact["reward_m2"])
