"""Small, causal multimodal self-supervised model for research.

No human buy/sell labels are used. A Gaussian next-observation objective and
causal masked reconstruction learn representations. A separate trainable
allocator maximizes a constrained one-step forecast utility. This is an
initial model, not evidence of profitable trading or calibrated uncertainty.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .risk import Forecast
from .schemas import FeatureFrame


@dataclass(frozen=True)
class TrainingConfig:
    lookback: int = 16
    hidden_dim: int = 32
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.001
    mask_probability: float = 0.15
    reconstruction_weight: float = 0.25
    allocation_weight: float = 1.0
    risk_aversion: float = 10.0
    tail_aversion: float = 1.0
    transaction_cost_bps: float = 5.0
    max_allocation: float = 0.20
    seed: int = 7
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 2 or min(self.hidden_dim, self.epochs, self.batch_size, self.num_threads) < 1:
            raise ValueError("Lookback must be >= 2; dimensions, epochs, batches and threads positive")
        if not 0 < self.mask_probability < 1 or not 0 < self.max_allocation <= 1:
            raise ValueError("Mask probability and maximum allocation must lie in (0,1]")
        positive = (self.learning_rate,)
        nonnegative = (
            self.reconstruction_weight, self.allocation_weight,
            self.risk_aversion, self.tail_aversion, self.transaction_cost_bps,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Learning rate must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("Objective weights and costs must be finite and nonnegative")


@dataclass
class FeatureScaler:
    """Statistics fitted exclusively on observed training rows."""

    mean: np.ndarray
    scale: np.ndarray
    counts: np.ndarray
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: FeatureFrame, end: int) -> "FeatureScaler":
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
        return cls(means.astype(np.float32), scales.astype(np.float32), counts, list(frame.feature_names))

    def transform(self, frame: FeatureFrame) -> tuple[np.ndarray, np.ndarray]:
        if list(frame.feature_names) != self.feature_names:
            raise ValueError("Feature names and order differ from the trained model")
        values = np.asarray(frame.values, dtype=np.float32)
        # Features with no training observations have no learned scaler or
        # encoder behavior. New held-out values remain unavailable until a
        # subsequent chronological retraining includes them.
        observed = np.asarray(frame.observed, dtype=bool) & np.isfinite(values) & (self.counts > 0)
        transformed = np.where(observed, (values - self.mean) / self.scale, 0)
        return transformed.astype(np.float32), observed


class CausalMarketModel(nn.Module):
    """Group encoders and a unidirectional GRU; future inputs are inaccessible."""

    def __init__(self, feature_count: int, groups: dict[str, list[int]], config: TrainingConfig):
        super().__init__()
        self.feature_count = feature_count
        self.groups = {name: list(indices) for name, indices in groups.items() if indices}
        self.group_names = list(self.groups)
        self.config = config
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * len(self.groups[name]), config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            ) for name in self.group_names
        ])
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self.group_names), config.hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.reconstruction = nn.Linear(config.hidden_dim, feature_count)
        self.next_mean = nn.Linear(config.hidden_dim, feature_count)
        self.next_log_scale = nn.Linear(config.hidden_dim, feature_count)
        self.allocator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = []
        for name, encoder in zip(self.group_names, self.encoders):
            indices = self.groups[name]
            # Masks distinguish a real numeric zero from an unavailable value.
            inputs = torch.cat((values[..., indices] * observed[..., indices], observed[..., indices]), dim=-1)
            encoded.append(encoder(inputs))
        fused = self.fusion(torch.cat(encoded, dim=-1))
        hidden, _ = self.temporal(fused)
        return {
            "reconstruction": self.reconstruction(hidden),
            "next_mean": self.next_mean(hidden),
            "next_log_scale": self.next_log_scale(hidden).clamp(-5, 2),
            # Avoid incentives to inflate return forecasts to improve utility.
            "allocation": torch.sigmoid(self.allocator(hidden.detach())).squeeze(-1) * self.config.max_allocation,
        }


class _Windows(Dataset):
    def __init__(self, values: np.ndarray, observed: np.ndarray, targets: range, lookback: int):
        self.values = torch.from_numpy(values)
        self.observed = torch.from_numpy(observed.astype(np.float32))
        self.targets = targets
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        target = self.targets[index]
        start = target - self.lookback
        return (
            self.values[start:target], self.observed[start:target],
            self.values[target], self.observed[target],
        )


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


def _return_index(names: list[str]) -> int:
    if "market.log_return" in names:
        return names.index("market.log_return")
    if "log_return" in names:
        return names.index("log_return")
    raise ValueError("A market.log_return (or plain log_return) feature is required")


def _allocation_loss(
    output: dict[str, torch.Tensor], return_index: int, scaler: FeatureScaler,
    config: TrainingConfig, return_observed: torch.Tensor,
) -> torch.Tensor:
    # The return head is fitted by next-observation likelihood, not utility.
    mu = output["next_mean"][:, -1, return_index].detach() * float(scaler.scale[return_index]) + float(scaler.mean[return_index])
    sigma = output["next_log_scale"][:, -1, return_index].detach().exp() * float(scaler.scale[return_index])
    allocation = output["allocation"][:, -1]
    previous = output["allocation"][:, -2].detach()
    expected_simple_return = torch.expm1((mu + sigma.square() / 2).clamp(-20, 5))
    variance = (torch.expm1(sigma.square().clamp(max=10)) * (2 * mu + sigma.square()).clamp(-20, 10).exp())
    # Gaussian expected shortfall below the 5th percentile: mu - 2.063 sigma.
    lower_tail_loss = (-torch.expm1((mu - 2.063 * sigma).clamp(-20, 5))).clamp_min(0)
    turnover = (allocation - previous).abs()
    transaction_cost = config.transaction_cost_bps / 10000 * (turnover + allocation)
    utility = (
        allocation * expected_simple_return - transaction_cost
        - 0.5 * config.risk_aversion * allocation.square() * variance
        - config.tail_aversion * allocation * lower_tail_loss
    )
    return -(utility * return_observed).sum() / return_observed.sum().clamp_min(1)


@dataclass
class TrainingResult:
    model: CausalMarketModel
    scaler: FeatureScaler
    config: TrainingConfig
    history: list[dict[str, float]]
    splits: dict[str, tuple[int, int]]
    symbol: str
    training_last_timestamp: str
    validation_last_timestamp: str

    def predict(self, frame: FeatureFrame, index: int | None = None) -> Forecast:
        """Predict the next interval from a full past window ending at index."""
        if frame.symbol != self.symbol:
            raise ValueError("This checkpoint was trained for a different symbol")
        if list(frame.feature_names) != self.scaler.feature_names:
            raise ValueError("Feature names and order differ from the trained model")
        shape = np.asarray(frame.values).shape
        if (len(shape) != 2 or shape != np.asarray(frame.observed).shape
                or shape[1] != len(frame.feature_names)
                or shape[0] != len(frame.timestamps) or shape[0] != len(frame.prices)):
            raise ValueError("Timestamps, prices, features and observed masks must align")
        index = shape[0] - 1 if index is None else index
        if not self.config.lookback - 1 <= index < shape[0]:
            raise ValueError("Prediction index must have a complete historical window")
        start = index + 1 - self.config.lookback
        # Transform only past context. Repeated evaluation costs O(T*lookback)
        # and neither reads nor validates observations beyond decision time.
        window = FeatureFrame(
            frame.timestamps[start:index + 1], frame.values[start:index + 1],
            frame.observed[start:index + 1], frame.feature_names, frame.groups,
            frame.prices[start:index + 1], frame.symbol,
        )
        _validate_frame(window)
        values, observed = self.scaler.transform(window)
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(values).unsqueeze(0),
                torch.from_numpy(observed.astype(np.float32)).unsqueeze(0),
            )
        return_index = _return_index(self.scaler.feature_names)
        mu = float(output["next_mean"][0, -1, return_index]) * float(self.scaler.scale[return_index]) + float(self.scaler.mean[return_index])
        sigma = float(output["next_log_scale"][0, -1, return_index].exp()) * float(self.scaler.scale[return_index])
        sigma2 = min(sigma * sigma, 10.0)
        mean = math.expm1(min(max(mu + sigma2 / 2, -20), 5))
        variance = math.expm1(sigma2) * math.exp(min(max(2 * mu + sigma2, -20), 10))
        downside = math.expm1(min(max(mu - 1.644853627 * sigma, -20), 5))
        trained = self.scaler.counts > 0
        coverage = float(observed[:, trained].mean()) if trained.any() else 0.0
        # This is a data/uncertainty gate score, not calibrated prediction accuracy.
        confidence = coverage * math.exp(-sigma / 0.02)
        return Forecast(mean, math.sqrt(variance), downside, confidence, float(output["allocation"][0, -1]))

    def save(self, path: str | Path) -> None:
        """Write only tensors and plain metadata; load with weights_only=True."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1,
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "feature_names": self.scaler.feature_names,
            "groups": self.model.groups,
            "mean": torch.from_numpy(self.scaler.mean),
            "scale": torch.from_numpy(self.scaler.scale),
            "counts": torch.from_numpy(self.scaler.counts),
            "history": self.history,
            "splits": self.splits,
            "symbol": self.symbol,
            "training_last_timestamp": self.training_last_timestamp,
            "validation_last_timestamp": self.validation_last_timestamp,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingResult":
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 1:
            raise ValueError("Unsupported model artifact format")
        config = TrainingConfig(**artifact["config"])
        scaler = FeatureScaler(
            artifact["mean"].numpy(), artifact["scale"].numpy(),
            artifact["counts"].numpy(), list(artifact["feature_names"]),
        )
        model = CausalMarketModel(len(scaler.feature_names), artifact["groups"], config)
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        return cls(
            model, scaler, config, artifact["history"], artifact["splits"],
            artifact["symbol"], artifact["training_last_timestamp"],
            artifact["validation_last_timestamp"],
        )


def train_model(
    frame: FeatureFrame,
    train_end: int,
    validation_end: int,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train with half-open chronological row boundaries.

    Target rows [lookback,train_end) train the network. Targets
    [train_end,validation_end) select the checkpoint; subsequent rows are
    held out. Context can come from an earlier partition; target labels never
    cross its end. No statistic or gradient is fitted on held-out rows.
    """
    config = config or TrainingConfig()
    groups = _validate_frame(frame)
    if not config.lookback + 1 <= train_end < validation_end <= len(frame.values):
        raise ValueError("Require lookback < train_end < validation_end <= row count")
    return_index = _return_index(frame.feature_names)
    scaler = FeatureScaler.fit(frame, train_end)
    if scaler.counts[return_index] < 2:
        raise ValueError("At least two observed training returns are required")
    values, observed = scaler.transform(frame)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.num_threads)
    try:
        model = CausalMarketModel(values.shape[1], groups, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        train = DataLoader(_Windows(values, observed, range(config.lookback, train_end), config.lookback), batch_size=config.batch_size, shuffle=False)
        validation = DataLoader(_Windows(values, observed, range(train_end, validation_end), config.lookback), batch_size=config.batch_size, shuffle=False)
        history: list[dict[str, float]] = []
        best_score = float("inf")
        best_state: dict[str, Any] | None = None
        for epoch in range(config.epochs):
            model.train()
            totals = np.zeros(4, dtype=float)
            count = 0
            for x, available, target, target_available in train:
                masked = (torch.rand_like(available) < config.mask_probability) & available.bool()
                presented = available * ~masked
                output = model(x * presented, presented)
                reconstruction = (((output["reconstruction"] - x).square()) * masked).sum() / masked.sum().clamp_min(1)
                mean = output["next_mean"][:, -1]
                log_scale = output["next_log_scale"][:, -1]
                nll = ((0.5 * (target - mean).square() * (-2 * log_scale).exp() + log_scale) * target_available).sum() / target_available.sum().clamp_min(1)
                allocation_loss = _allocation_loss(output, return_index, scaler, config, target_available[:, return_index])
                loss = nll + config.reconstruction_weight * reconstruction + config.allocation_weight * allocation_loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training objective; inspect feature magnitudes and observations")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                batch_size = len(x)
                totals += np.array([float(loss.detach()), float(nll.detach()), float(reconstruction.detach()), float(allocation_loss.detach())]) * batch_size
                count += batch_size
            model.eval()
            validation_sum = 0.0
            validation_observations = 0.0
            with torch.no_grad():
                for x, available, target, target_available in validation:
                    output = model(x, available)
                    mean, log_scale = output["next_mean"][:, -1], output["next_log_scale"][:, -1]
                    losses = 0.5 * (target - mean).square() * (-2 * log_scale).exp() + log_scale
                    validation_sum += float((losses * target_available).sum())
                    validation_observations += float(target_available.sum())
            if validation_observations == 0:
                raise ValueError("Validation partition contains no observed target features")
            score = validation_sum / validation_observations
            if not math.isfinite(score):
                raise ValueError("Nonfinite validation objective")
            history.append(dict(zip(("train_loss", "next_nll", "reconstruction", "allocation_loss"), (totals / count).tolist()), epoch=float(epoch + 1), validation_nll=score))
            if score < best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        return TrainingResult(
            model, scaler, config, history,
            {"train": (config.lookback, train_end), "validation": (train_end, validation_end), "test": (validation_end, len(values))},
            frame.symbol, frame.timestamps[train_end - 1].isoformat(), frame.timestamps[validation_end - 1].isoformat(),
        )
    finally:
        torch.set_num_threads(previous_threads)
