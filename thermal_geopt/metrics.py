"""Metrics used by Thermal GeoPT experiments."""

from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(pred_arr - target_arr) / max(np.linalg.norm(target_arr), eps))


def centered_relative_l2(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    target_mean = float(np.mean(target_arr))
    return float(
        np.linalg.norm((pred_arr - target_mean) - (target_arr - target_mean))
        / max(np.linalg.norm(target_arr - target_mean), eps)
    )


def normalized_rmse_range(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    error = pred_arr - target_arr
    target_range = float(np.max(target_arr) - np.min(target_arr))
    return float(np.sqrt(np.mean(error * error)) / max(target_range, eps))


def max_abs_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def max_value_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.asarray(pred, dtype=np.float64)) - np.max(np.asarray(target, dtype=np.float64)))


def max_temperature_abs_error(pred: np.ndarray, target: np.ndarray) -> float:
    return abs(max_value_error(pred, target))


def hotspot_abs_error(pred: np.ndarray, target: np.ndarray) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    target_hotspot = int(np.argmax(target_arr))
    return float(abs(pred_arr[target_hotspot] - target_arr[target_hotspot]))
