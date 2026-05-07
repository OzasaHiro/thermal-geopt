"""Normalization helpers for Thermal GeoPT training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ChannelStatsAccumulator:
    """Streaming per-channel mean/std accumulator."""

    total: int = 0
    sum: np.ndarray | None = None
    sumsq: np.ndarray | None = None

    def update(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        else:
            array = array.reshape(-1, array.shape[-1])
        if array.size == 0:
            return
        if self.sum is None:
            self.sum = np.zeros(array.shape[-1], dtype=np.float64)
            self.sumsq = np.zeros(array.shape[-1], dtype=np.float64)
        self.total += int(array.shape[0])
        self.sum += array.sum(axis=0)
        self.sumsq += np.square(array).sum(axis=0)

    def finalize(self, *, min_std: float = 1e-6) -> dict[str, list[float] | int]:
        if self.total <= 0 or self.sum is None or self.sumsq is None:
            raise ValueError("Cannot finalize empty normalization statistics.")
        mean = self.sum / float(self.total)
        variance = np.maximum(self.sumsq / float(self.total) - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), min_std)
        return {
            "count": int(self.total),
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        }


def _sample_array(sample: dict[str, Any], key: str) -> np.ndarray:
    value = sample[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def estimate_dataset_normalization(
    dataset: Dataset,
    *,
    max_items: int = 0,
    include_target: bool = True,
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Estimate x/fx/y per-channel normalization stats from a dataset."""

    dataset_len = len(dataset)
    item_count = dataset_len if max_items <= 0 else min(dataset_len, int(max_items))
    if item_count <= 0:
        raise ValueError("Cannot estimate normalization from an empty dataset.")
    if item_count >= dataset_len:
        indices = range(dataset_len)
    else:
        indices = np.linspace(0, dataset_len - 1, item_count, dtype=np.int64).tolist()

    x_stats = ChannelStatsAccumulator()
    feature_stats = ChannelStatsAccumulator()
    target_stats = ChannelStatsAccumulator()
    for index in indices:
        sample = dataset[index]  # type: ignore[index]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dataset sample to be a dict, got {type(sample)!r}")
        x_stats.update(_sample_array(sample, "x"))
        feature_stats.update(_sample_array(sample, "fx"))
        if include_target:
            target_stats.update(_sample_array(sample, "y"))

    payload: dict[str, Any] = {
        "coordinate": x_stats.finalize(min_std=min_std),
        "feature": feature_stats.finalize(min_std=min_std),
        "estimated_items": int(item_count),
    }
    if include_target:
        payload["target"] = target_stats.finalize(min_std=min_std)
    return payload


def vector_from_stats(
    normalization: dict[str, Any],
    group: str,
    field: str,
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    stats = normalization.get(group)
    if not isinstance(stats, dict) or field not in stats:
        if field == "mean":
            values = [0.0] * width
        elif field == "std":
            values = [1.0] * width
        else:
            raise KeyError(f"Unknown stats field: {field}")
    else:
        values = stats[field]
    tensor = torch.as_tensor(values, device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 1 and width != 1:
        tensor = tensor.repeat(width)
    if tensor.numel() != width:
        raise ValueError(f"Normalization stats for {group}.{field} have width {tensor.numel()}, expected {width}.")
    if field == "std":
        tensor = torch.clamp(tensor, min=1e-8)
    return tensor


def normalize_tensor(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean.view(*([1] * (values.ndim - 1)), -1)) / std.view(*([1] * (values.ndim - 1)), -1)


def denormalize_tensor(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * std.view(*([1] * (values.ndim - 1)), -1) + mean.view(*([1] * (values.ndim - 1)), -1)


def legacy_finetune_normalization(
    *,
    target_mean: float,
    target_std: float,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, Any]:
    """Build a normalization payload matching the old downstream behavior."""

    return {
        "mode": "legacy_downstream",
        "coordinate": {
            "count": 0,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
        "feature": {
            "count": int(feature_mean.numel()),
            "mean": [float(value) for value in feature_mean.detach().cpu().reshape(-1).tolist()],
            "std": [float(value) for value in torch.clamp(feature_std.detach().cpu().reshape(-1), min=1e-6).tolist()],
        },
        "target": {
            "count": 1,
            "mean": [float(target_mean)],
            "std": [float(max(target_std, 1e-6))],
        },
    }
