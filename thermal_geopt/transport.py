"""GeoPT-style synthetic transport trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_geopt.geometry import as_points, estimate_surface_spacing, nearest_boundary


@dataclass(frozen=True)
class TransportConfig:
    steps: int = 3
    min_step: float = 0.0
    max_step: float = 0.12
    hit_radius: float | None = None
    seed: int = 42


@dataclass(frozen=True)
class TransportResult:
    positions: np.ndarray
    hit_mask: np.ndarray
    hit_step: np.ndarray
    condition: np.ndarray


def _random_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=(count, 3))
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return (vec / np.maximum(norms, 1e-12)).astype(np.float32)


def simulate_vector_transport(
    start_points: np.ndarray,
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray | None = None,
    *,
    config: TransportConfig | None = None,
) -> TransportResult:
    """Generate GeoPT-style trajectories under a fixed synthetic vector field.

    This mirrors GeoPT's lifted pretext more closely than Brownian motion: each
    query point receives a fixed direction and step length as the dynamics
    prompt, then moves under that prompt until it reaches the sampled boundary.
    Exact ray/triangle intersection can replace the nearest-boundary clamp later
    without changing the stored schema.
    """
    cfg = config or TransportConfig()
    if cfg.steps < 1:
        raise ValueError("steps must be positive")
    if cfg.min_step < 0.0 or cfg.max_step < cfg.min_step:
        raise ValueError("Expected 0 <= min_step <= max_step")

    current = as_points(start_points, name="start_points").astype(np.float64)
    boundary = as_points(boundary_points, name="boundary_points")
    normals = as_points(boundary_normals, name="boundary_normals") if boundary_normals is not None else None
    if normals is not None and normals.shape[0] != boundary.shape[0]:
        raise ValueError("boundary_normals must have the same length as boundary_points")

    hit_radius = cfg.hit_radius
    if hit_radius is None:
        spacing = estimate_surface_spacing(boundary)
        hit_radius = max(1.5 * spacing, 1e-4)

    rng = np.random.default_rng(cfg.seed)
    count = current.shape[0]
    directions = _random_unit_vectors(count, rng).astype(np.float64)
    step_lengths = rng.uniform(cfg.min_step, cfg.max_step, size=count).astype(np.float64)
    condition = np.concatenate([directions.astype(np.float32), step_lengths[:, None].astype(np.float32)], axis=1)

    positions = np.empty((cfg.steps + 1, count, 3), dtype=np.float32)
    positions[0] = current.astype(np.float32)
    active = np.ones(count, dtype=bool)
    hit_mask = np.zeros(count, dtype=bool)
    hit_step = np.full(count, -1, dtype=np.int32)

    for step in range(1, cfg.steps + 1):
        proposal = current.copy()
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            local_dirs = directions[active_ids]
            local_steps = step_lengths[active_ids]
            local_current = current[active_ids]
            intended = local_current + local_dirs * local_steps[:, None]

            nearest_now = nearest_boundary(local_current, boundary, normals)
            projected_distance = np.sum(nearest_now.vectors.astype(np.float64) * local_dirs, axis=1)
            ray_like_hit = (projected_distance > 0.0) & (projected_distance <= local_steps)

            nearest_intended = nearest_boundary(intended, boundary, normals)
            near_hit = nearest_intended.distances <= hit_radius
            local_hit = ray_like_hit | near_hit
            local_proposal = intended
            if np.any(ray_like_hit):
                ray_ids = np.flatnonzero(ray_like_hit)
                actual_steps = np.maximum(projected_distance[ray_ids] * 0.99, 0.0)
                local_proposal[ray_ids] = local_current[ray_ids] + local_dirs[ray_ids] * actual_steps[:, None]
            if np.any(near_hit):
                near_ids = np.flatnonzero(near_hit)
                local_proposal[near_ids] = nearest_intended.points[near_ids]

            hit_ids = active_ids[local_hit]
            if hit_ids.size:
                hit_mask[hit_ids] = True
                hit_step[hit_ids] = step
                active[hit_ids] = False
            proposal[active_ids] = local_proposal

        current = proposal
        positions[step] = current.astype(np.float32)

    return TransportResult(
        positions=positions,
        hit_mask=hit_mask,
        hit_step=hit_step,
        condition=condition.astype(np.float32),
    )
