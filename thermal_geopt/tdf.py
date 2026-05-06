"""Thermal diffusion feature construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_geopt.geometry import as_points, nearest_boundary


@dataclass(frozen=True)
class ThermalFeatureConfig:
    alpha: float = 1.0
    conductivity: float = 1.0
    length_scale: float = 1.0
    time_scales: tuple[float, ...] = (0.01, 0.05, 0.1)
    eps: float = 1e-8


def heat_kernel_proximity(
    distances: np.ndarray,
    *,
    alpha: float,
    time_scales: tuple[float, ...],
    eps: float = 1e-8,
) -> np.ndarray:
    d2 = np.asarray(distances, dtype=np.float64)[:, None] ** 2
    times = np.asarray(time_scales, dtype=np.float64)[None, :]
    return np.exp(-d2 / (4.0 * max(alpha, eps) * np.maximum(times, eps))).astype(np.float32)


def radial_proximity(points: np.ndarray, centers: np.ndarray | None, *, sigma: float) -> np.ndarray:
    if centers is None or centers.size == 0:
        return np.empty((points.shape[0], 0), dtype=np.float32)
    pts = as_points(points)
    ctr = as_points(centers, name="centers")
    dist2 = np.sum((pts[:, None, :] - ctr[None, :, :]) ** 2, axis=-1)
    nearest = np.min(dist2, axis=1, keepdims=True)
    return np.exp(-nearest / max(sigma * sigma, 1e-12)).astype(np.float32)


def thermal_diffusion_features(
    points: np.ndarray,
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray | None = None,
    *,
    config: ThermalFeatureConfig | None = None,
    source_centers: np.ndarray | None = None,
    sink_centers: np.ndarray | None = None,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Build multi-channel TDF-style features for query points.

    The output is a self-supervised feature target, not a solved temperature
    field. It encodes geometry, approximate diffusion scale, boundary normal,
    and optional source/sink proximity.
    """
    cfg = config or ThermalFeatureConfig()
    length_scale = max(float(cfg.length_scale), cfg.eps)
    alpha = max(float(cfg.alpha), cfg.eps)
    conductivity = max(float(cfg.conductivity), cfg.eps)

    nearest = nearest_boundary(points, boundary_points, boundary_normals)
    distance = nearest.distances.astype(np.float32)
    vectors = nearest.vectors.astype(np.float32) / length_scale
    normalized_distance = (distance / length_scale)[:, None]
    diffusion_time = ((distance.astype(np.float64) ** 2) / alpha)[:, None].astype(np.float32)
    heat_kernel = heat_kernel_proximity(distance, alpha=alpha, time_scales=cfg.time_scales, eps=cfg.eps)
    resistance_distance = (distance / conductivity)[:, None].astype(np.float32)

    parts = [vectors, normalized_distance, diffusion_time, heat_kernel, resistance_distance]
    names = [
        "vdf_x",
        "vdf_y",
        "vdf_z",
        "distance",
        "diffusion_time",
        *[f"heat_kernel_t{idx}" for idx, _ in enumerate(cfg.time_scales)],
        "resistance_distance",
    ]

    if nearest.normals is not None:
        parts.append(nearest.normals.astype(np.float32))
        names.extend(["normal_x", "normal_y", "normal_z"])

    sigma = 0.1 * length_scale
    source = radial_proximity(np.asarray(points), source_centers, sigma=sigma)
    if source.shape[1]:
        parts.append(source)
        names.append("source_proximity")

    sink = radial_proximity(np.asarray(points), sink_centers, sigma=sigma)
    if sink.shape[1]:
        parts.append(sink)
        names.append("sink_proximity")

    features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise ValueError("thermal diffusion features contain non-finite values")
    return (features, names) if return_names else features
