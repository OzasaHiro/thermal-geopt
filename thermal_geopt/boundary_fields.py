"""Synthetic thermal boundary field helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_geopt.geometry import as_points, nearest_boundary


@dataclass(frozen=True)
class RBFBoundaryField:
    centers: np.ndarray
    amplitudes: np.ndarray
    sigmas: np.ndarray
    base: float = 0.0

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        pts = as_points(points)
        diff = pts[:, None, :] - self.centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        basis = np.exp(-dist2 / (2.0 * np.maximum(self.sigmas[None, :] ** 2, 1e-12)))
        values = self.base + basis @ self.amplitudes
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        if max_abs > 1e-12:
            values = values / max_abs
        return values.astype(np.float32)


def sample_rbf_boundary_field(
    boundary_points: np.ndarray,
    *,
    num_patches: int = 8,
    amplitude_range: tuple[float, float] = (-1.0, 1.0),
    sigma_fraction_range: tuple[float, float] = (0.05, 0.25),
    length_scale: float = 1.0,
    seed: int = 42,
) -> RBFBoundaryField:
    boundary = as_points(boundary_points, name="boundary_points")
    if num_patches <= 0:
        raise ValueError("num_patches must be positive")
    rng = np.random.default_rng(seed)
    ids = rng.choice(boundary.shape[0], size=min(num_patches, boundary.shape[0]), replace=False)
    centers = boundary[ids]
    amplitudes = rng.uniform(amplitude_range[0], amplitude_range[1], size=centers.shape[0])
    sigmas = rng.uniform(sigma_fraction_range[0], sigma_fraction_range[1], size=centers.shape[0]) * length_scale
    return RBFBoundaryField(
        centers=centers.astype(np.float32),
        amplitudes=amplitudes.astype(np.float32),
        sigmas=sigmas.astype(np.float32),
    )


def nearest_boundary_field_values(
    points: np.ndarray,
    boundary_points: np.ndarray,
    field: RBFBoundaryField,
) -> np.ndarray:
    nearest = nearest_boundary(points, boundary_points)
    return field.evaluate(nearest.points)
