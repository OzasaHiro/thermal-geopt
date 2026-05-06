"""Geometry helpers for Thermal GeoPT preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback is covered by behavior.
    cKDTree = None


ArrayLike = Any


@dataclass(frozen=True)
class NormalizeResult:
    points: np.ndarray
    center: np.ndarray
    scale: float


@dataclass(frozen=True)
class NearestBoundaryResult:
    indices: np.ndarray
    points: np.ndarray
    normals: np.ndarray | None
    vectors: np.ndarray
    distances: np.ndarray


def as_points(points: ArrayLike, *, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def normalize_points(points: ArrayLike, *, target_extent: float = 1.0) -> NormalizeResult:
    """Center points and scale the largest axis-aligned extent to target_extent."""
    pts = as_points(points)
    bounds_min = pts.min(axis=0)
    bounds_max = pts.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    if extent <= 1e-12:
        raise ValueError("Cannot normalize a degenerate point cloud")
    scale = float(target_extent / extent)
    return NormalizeResult(points=((pts - center) * scale).astype(np.float32), center=center, scale=scale)


def sphere_surface_points(
    *,
    radius: float = 1.0,
    num_lat: int = 32,
    num_lon: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic sphere surface point cloud and outward normals."""
    if num_lat < 3 or num_lon < 4:
        raise ValueError("num_lat >= 3 and num_lon >= 4 are required")
    theta = np.linspace(0.0, np.pi, num_lat, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, num_lon, endpoint=False, dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    normals = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    points = radius * normals
    return points.astype(np.float32), normals.astype(np.float32)


def _nearest_bruteforce(
    points: np.ndarray,
    boundary_points: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.empty(points.shape[0], dtype=np.int64)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        diff = points[start:stop, None, :] - boundary_points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        local_indices = np.argmin(dist2, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(dist2[np.arange(stop - start), local_indices])
    return distances, indices


def nearest_boundary(
    points: ArrayLike,
    boundary_points: ArrayLike,
    boundary_normals: ArrayLike | None = None,
) -> NearestBoundaryResult:
    """Project each query point to the nearest sampled boundary point."""
    query = as_points(points, name="points")
    boundary = as_points(boundary_points, name="boundary_points")
    normals = None
    if boundary_normals is not None:
        normals = as_points(boundary_normals, name="boundary_normals")
        if normals.shape[0] != boundary.shape[0]:
            raise ValueError("boundary_normals must have the same length as boundary_points")

    if cKDTree is not None:
        distances, indices = cKDTree(boundary).query(query, k=1)
    else:
        distances, indices = _nearest_bruteforce(query, boundary)

    nearest = boundary[indices]
    nearest_normals = normals[indices] if normals is not None else None
    vectors = nearest - query
    return NearestBoundaryResult(
        indices=indices.astype(np.int64, copy=False),
        points=nearest.astype(np.float32, copy=False),
        normals=nearest_normals.astype(np.float32, copy=False) if nearest_normals is not None else None,
        vectors=vectors.astype(np.float32, copy=False),
        distances=np.asarray(distances, dtype=np.float32),
    )


def estimate_surface_spacing(boundary_points: ArrayLike, *, sample_size: int = 8192) -> float:
    """Estimate a robust nearest-neighbor spacing for a sampled surface."""
    boundary = as_points(boundary_points, name="boundary_points")
    if boundary.shape[0] < 2:
        return 0.0
    if boundary.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        ids = np.sort(rng.choice(boundary.shape[0], size=sample_size, replace=False))
        boundary = boundary[ids]

    if cKDTree is not None:
        distances, _ = cKDTree(boundary).query(boundary, k=2)
        nn = distances[:, 1]
    else:
        diff = boundary[:, None, :] - boundary[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(dist2, np.inf)
        nn = np.sqrt(np.min(dist2, axis=1))
    return float(np.median(nn))
