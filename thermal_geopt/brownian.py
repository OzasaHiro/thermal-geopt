"""Brownian trajectory generation for diffusion-lifted pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thermal_geopt.geometry import as_points, estimate_surface_spacing, nearest_boundary


@dataclass(frozen=True)
class BrownianConfig:
    steps: int = 2
    dt: float = 0.02
    alpha: float = 1.0
    boundary_mode: str = "absorbing"
    partial_absorb_prob: float = 0.5
    hit_radius: float | None = None
    seed: int = 42


@dataclass(frozen=True)
class BrownianResult:
    positions: np.ndarray
    hit_mask: np.ndarray
    hit_step: np.ndarray


def _reflect_across_tangent(proposal: np.ndarray, boundary_points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    offset = proposal - boundary_points
    signed = np.sum(offset * normals, axis=1, keepdims=True)
    return proposal - 2.0 * signed * normals


def simulate_brownian_walk(
    start_points: np.ndarray,
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray | None = None,
    *,
    config: BrownianConfig | None = None,
) -> BrownianResult:
    """Generate approximate Brownian trajectories with sampled-boundary handling.

    This is a solver-free pretext generator. Boundary interaction is based on
    nearest sampled boundary distance, so exact segment-triangle intersection
    can replace this later without changing the result schema.
    """
    cfg = config or BrownianConfig()
    if cfg.steps < 0:
        raise ValueError("steps must be non-negative")
    if cfg.dt <= 0.0 or cfg.alpha <= 0.0:
        raise ValueError("dt and alpha must be positive")
    if cfg.boundary_mode not in {"absorbing", "reflecting", "partial_absorbing"}:
        raise ValueError(f"Unsupported boundary_mode: {cfg.boundary_mode}")

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
    positions = np.empty((cfg.steps + 1, current.shape[0], 3), dtype=np.float32)
    positions[0] = current.astype(np.float32)
    active = np.ones(current.shape[0], dtype=bool)
    hit_mask = np.zeros(current.shape[0], dtype=bool)
    hit_step = np.full(current.shape[0], -1, dtype=np.int32)
    step_scale = float(np.sqrt(2.0 * cfg.alpha * cfg.dt))

    for step in range(1, cfg.steps + 1):
        proposal = current.copy()
        active_ids = np.flatnonzero(active)
        if active_ids.size:
            proposal[active_ids] += step_scale * rng.normal(size=(active_ids.size, 3))
            nearest = nearest_boundary(proposal[active_ids], boundary, normals)
            local_hit = nearest.distances <= hit_radius
            hit_ids = active_ids[local_hit]

            if hit_ids.size:
                hit_mask[hit_ids] = True
                hit_step[hit_ids] = step
                if cfg.boundary_mode == "absorbing":
                    proposal[hit_ids] = nearest.points[local_hit]
                    active[hit_ids] = False
                elif cfg.boundary_mode == "reflecting":
                    if nearest.normals is not None:
                        proposal[hit_ids] = _reflect_across_tangent(
                            proposal[hit_ids],
                            nearest.points[local_hit],
                            nearest.normals[local_hit],
                        )
                    else:
                        proposal[hit_ids] = current[hit_ids]
                else:
                    absorb = rng.random(hit_ids.size) < cfg.partial_absorb_prob
                    absorb_ids = hit_ids[absorb]
                    reflect_ids = hit_ids[~absorb]
                    if absorb_ids.size:
                        proposal[absorb_ids] = nearest.points[local_hit][absorb]
                        active[absorb_ids] = False
                    if reflect_ids.size:
                        if nearest.normals is not None:
                            proposal[reflect_ids] = _reflect_across_tangent(
                                proposal[reflect_ids],
                                nearest.points[local_hit][~absorb],
                                nearest.normals[local_hit][~absorb],
                            )
                        else:
                            proposal[reflect_ids] = current[reflect_ids]

        current = proposal
        positions[step] = current.astype(np.float32)

    return BrownianResult(positions=positions, hit_mask=hit_mask, hit_step=hit_step)
