"""Metrics used by Thermal GeoPT experiments."""

from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, target: np.ndarray, *, eps: float = 1e-8) -> float:
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(pred_arr - target_arr) / max(np.linalg.norm(target_arr), eps))


def max_abs_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def max_value_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.max(np.asarray(pred, dtype=np.float64)) - np.max(np.asarray(target, dtype=np.float64)))
