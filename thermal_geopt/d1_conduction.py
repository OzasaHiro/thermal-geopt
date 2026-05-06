"""OpenFOAM-free D1 solid-conduction proxy case generation.

This module intentionally does not run a full FEM/FVM conduction solve. It
creates deterministic, smooth source/sink influence fields on points sampled
from already processed mesh NPZ files. The output is meant for tiny smoke runs
and data plumbing checks, not for validating physical accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thermal_geopt.geometry import as_points, estimate_surface_spacing, nearest_boundary


@dataclass(frozen=True)
class ProcessedMesh:
    name: str
    path: Path
    surface_points: np.ndarray
    surface_normals: np.ndarray
    vertices: np.ndarray | None
    faces: np.ndarray | None

    @property
    def bounds_min(self) -> np.ndarray:
        return self.surface_points.min(axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.surface_points.max(axis=0)

    @property
    def extent(self) -> float:
        return float(np.max(self.bounds_max - self.bounds_min))


@dataclass(frozen=True)
class D1ProxyCase:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def load_processed_mesh(path: Path) -> ProcessedMesh:
    """Load the sampled surface arrays produced by scripts/preprocess_meshes.py."""
    with np.load(path) as data:
        if "surface_points" not in data or "surface_normals" not in data:
            raise ValueError(f"{path} is missing surface_points/surface_normals")
        surface_points = as_points(data["surface_points"], name="surface_points").astype(np.float32)
        surface_normals = as_points(data["surface_normals"], name="surface_normals").astype(np.float32)
        if surface_points.shape[0] != surface_normals.shape[0]:
            raise ValueError(f"{path} has mismatched surface point and normal counts")
        if surface_points.shape[0] < 2:
            raise ValueError(f"{path} needs at least two surface samples for source/sink patches")
        vertices = data["vertices"].astype(np.float32) if "vertices" in data else None
        faces = data["faces"].astype(np.int64) if "faces" in data else None

    normal_norm = np.linalg.norm(surface_normals, axis=1, keepdims=True)
    surface_normals = surface_normals / np.maximum(normal_norm, 1e-12)
    return ProcessedMesh(
        name=path.stem,
        path=path,
        surface_points=surface_points,
        surface_normals=surface_normals.astype(np.float32),
        vertices=vertices,
        faces=faces,
    )


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_solid_proxy_points(
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface and shallow inward-shell points from a processed mesh.

    Processed NPZ files only contain sampled surface geometry. For this proxy we
    keep sampling cheap and deterministic by taking a small surface subset plus
    points offset along the negative sampled normals. This is sufficient for
    smoke testing downstream D1 data loaders without OpenFOAM.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    boundary = as_points(boundary_points, name="boundary_points").astype(np.float32)
    normals = as_points(boundary_normals, name="boundary_normals").astype(np.float32)
    if boundary.shape[0] != normals.shape[0]:
        raise ValueError("boundary_points and boundary_normals must have the same length")

    extent = float(np.max(boundary.max(axis=0) - boundary.min(axis=0)))
    if extent <= 1e-12:
        raise ValueError("Cannot sample from a degenerate boundary point cloud")
    spacing = estimate_surface_spacing(boundary)
    min_depth = max(0.5 * spacing, 0.005 * extent)
    max_depth = max(4.0 * spacing, 0.16 * extent)

    surface_count = min(count, max(1, count // 4))
    shell_count = count - surface_count

    surface_ids = rng.integers(0, boundary.shape[0], size=surface_count)
    surface = boundary[surface_ids]
    is_boundary = np.ones(surface_count, dtype=bool)

    if shell_count > 0:
        shell_ids = rng.integers(0, boundary.shape[0], size=shell_count)
        depths = rng.uniform(min_depth, max_depth, size=(shell_count, 1)).astype(np.float32)
        jitter = rng.normal(0.0, 0.01 * extent, size=(shell_count, 3)).astype(np.float32)
        shell = boundary[shell_ids] - depths * normals[shell_ids] + jitter
        points = np.concatenate([surface, shell], axis=0).astype(np.float32)
        is_boundary = np.concatenate([is_boundary, np.zeros(shell_count, dtype=bool)], axis=0)
    else:
        points = surface.astype(np.float32)

    order = rng.permutation(count)
    return points[order], is_boundary[order]


def choose_source_sink(
    boundary_points: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Choose source and sink boundary samples with a useful separation."""
    boundary = as_points(boundary_points, name="boundary_points")
    source_index = int(rng.integers(0, boundary.shape[0]))
    distances = np.linalg.norm(boundary - boundary[source_index], axis=1)
    threshold = float(np.quantile(distances, 0.65))
    candidates = np.flatnonzero(distances >= threshold)
    if candidates.size == 0:
        sink_index = int(np.argmax(distances))
    else:
        sink_index = int(rng.choice(candidates))
    if sink_index == source_index:
        sink_index = int(np.argmax(distances))
    return source_index, sink_index


def gaussian_patch(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    pts = as_points(points)
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    radius = max(float(radius), 1e-6)
    dist2 = np.sum((pts - c) ** 2, axis=1)
    return np.exp(-dist2 / (2.0 * radius * radius)).astype(np.float32)


def influence_temperature(
    points: np.ndarray,
    *,
    source_center: np.ndarray,
    sink_center: np.ndarray,
    source_temperature: float,
    sink_temperature: float,
    source_strength: float,
    sink_strength: float,
    conductivity: float,
    core_radius: float,
    power: float = 1.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a bounded inverse-distance source/sink proxy and heat flux."""
    pts = as_points(points).astype(np.float64)
    source_vec = pts - np.asarray(source_center, dtype=np.float64).reshape(1, 3)
    sink_vec = pts - np.asarray(sink_center, dtype=np.float64).reshape(1, 3)
    core2 = max(float(core_radius) ** 2, 1e-12)

    source_r2 = np.sum(source_vec * source_vec, axis=1, keepdims=True) + core2
    sink_r2 = np.sum(sink_vec * sink_vec, axis=1, keepdims=True) + core2
    source_weight = float(source_strength) * source_r2 ** (-0.5 * power)
    sink_weight = float(sink_strength) * sink_r2 ** (-0.5 * power)
    denom = np.maximum(source_weight + sink_weight, 1e-12)
    theta = source_weight / denom

    grad_source = -power * source_weight * source_vec / source_r2
    grad_sink = -power * sink_weight * sink_vec / sink_r2
    grad_theta = (grad_source * sink_weight - source_weight * grad_sink) / np.maximum(denom * denom, 1e-12)

    delta_t = float(source_temperature - sink_temperature)
    temperature = float(sink_temperature) + delta_t * theta
    gradient = delta_t * grad_theta
    heat_flux = -float(conductivity) * gradient
    return (
        temperature.astype(np.float32),
        gradient.astype(np.float32),
        heat_flux.astype(np.float32),
    )


def generate_d1_proxy_case(
    mesh: ProcessedMesh,
    *,
    points_per_case: int,
    seed: int,
    case_index: int,
) -> D1ProxyCase:
    """Generate one deterministic source/sink D1 conduction proxy case."""
    rng = np.random.default_rng(seed)
    extent = max(mesh.extent, 1e-6)
    points, is_boundary = sample_solid_proxy_points(
        mesh.surface_points,
        mesh.surface_normals,
        count=points_per_case,
        rng=rng,
    )
    nearest = nearest_boundary(points, mesh.surface_points, mesh.surface_normals)

    source_index, sink_index = choose_source_sink(mesh.surface_points, rng=rng)
    source_center = mesh.surface_points[source_index]
    sink_center = mesh.surface_points[sink_index]
    source_temperature = float(rng.uniform(360.0, 430.0))
    sink_temperature = float(rng.uniform(275.0, 310.0))
    if source_temperature < sink_temperature + 35.0:
        source_temperature = sink_temperature + float(rng.uniform(35.0, 85.0))
    conductivity = log_uniform(rng, 0.2, 25.0)
    source_strength = float(rng.uniform(0.85, 1.15))
    sink_strength = float(rng.uniform(0.85, 1.15))
    patch_radius = float(rng.uniform(0.08, 0.18) * extent)
    core_radius = float(rng.uniform(0.035, 0.075) * extent)

    temperature, temperature_gradient, heat_flux = influence_temperature(
        points,
        source_center=source_center,
        sink_center=sink_center,
        source_temperature=source_temperature,
        sink_temperature=sink_temperature,
        source_strength=source_strength,
        sink_strength=sink_strength,
        conductivity=conductivity,
        core_radius=core_radius,
    )

    source_patch = gaussian_patch(nearest.points, source_center, patch_radius)[:, None]
    sink_patch = gaussian_patch(nearest.points, sink_center, patch_radius)[:, None]
    boundary_source_patch = gaussian_patch(mesh.surface_points, source_center, patch_radius)[:, None]
    boundary_sink_patch = gaussian_patch(mesh.surface_points, sink_center, patch_radius)[:, None]
    dirichlet_mask = ((source_patch > 0.2) | (sink_patch > 0.2)).astype(np.bool_)
    raw_dirichlet_temperature = (
        source_patch * source_temperature + sink_patch * sink_temperature
    ) / np.maximum(source_patch + sink_patch, 1e-8)
    dirichlet_temperature = np.where(dirichlet_mask, raw_dirichlet_temperature, 0.0)

    condition_names = np.asarray(
        [
            "conductivity",
            "source_temperature",
            "sink_temperature",
            "source_patch",
            "sink_patch",
            "nearest_boundary_distance",
        ]
    )
    conditions = np.concatenate(
        [
            np.full((points_per_case, 1), conductivity, dtype=np.float32),
            np.full((points_per_case, 1), source_temperature, dtype=np.float32),
            np.full((points_per_case, 1), sink_temperature, dtype=np.float32),
            source_patch.astype(np.float32),
            sink_patch.astype(np.float32),
            nearest.distances[:, None].astype(np.float32),
        ],
        axis=1,
    )
    case_param_names = np.asarray(
        [
            "source_temperature",
            "sink_temperature",
            "conductivity",
            "source_strength",
            "sink_strength",
            "patch_radius",
            "core_radius",
            "source_index",
            "sink_index",
        ]
    )
    case_params = np.asarray(
        [
            source_temperature,
            sink_temperature,
            conductivity,
            source_strength,
            sink_strength,
            patch_radius,
            core_radius,
            float(source_index),
            float(sink_index),
        ],
        dtype=np.float32,
    )

    arrays = {
        "points": points.astype(np.float32),
        "is_boundary": is_boundary.astype(np.bool_),
        "temperature": temperature.astype(np.float32),
        "temperature_gradient": temperature_gradient.astype(np.float32),
        "heat_flux": heat_flux.astype(np.float32),
        "conditions": conditions.astype(np.float32),
        "condition_names": condition_names,
        "dirichlet_temperature": dirichlet_temperature.astype(np.float32),
        "dirichlet_mask": dirichlet_mask,
        "source_patch": source_patch.astype(np.float32),
        "sink_patch": sink_patch.astype(np.float32),
        "nearest_boundary_index": nearest.indices.astype(np.int64),
        "nearest_boundary_distance": nearest.distances.astype(np.float32),
        "nearest_boundary_normal": nearest.normals.astype(np.float32),
        "boundary_points": mesh.surface_points.astype(np.float32),
        "boundary_normals": mesh.surface_normals.astype(np.float32),
        "boundary_source_patch": boundary_source_patch.astype(np.float32),
        "boundary_sink_patch": boundary_sink_patch.astype(np.float32),
        "source_center": source_center.astype(np.float32),
        "sink_center": sink_center.astype(np.float32),
        "case_params": case_params,
        "case_param_names": case_param_names,
        "mesh_name": np.asarray(mesh.name),
        "generator": np.asarray("d1_conduction_proxy_v1"),
    }
    if mesh.vertices is not None:
        arrays["vertices"] = mesh.vertices.astype(np.float32)
    if mesh.faces is not None:
        arrays["faces"] = mesh.faces.astype(np.int64)

    metadata: dict[str, Any] = {
        "mesh_name": mesh.name,
        "processed_path": str(mesh.path),
        "case_index": int(case_index),
        "seed": int(seed),
        "points": int(points_per_case),
        "boundary_points": int(mesh.surface_points.shape[0]),
        "source_index": int(source_index),
        "sink_index": int(sink_index),
        "source_temperature": source_temperature,
        "sink_temperature": sink_temperature,
        "conductivity": conductivity,
        "source_strength": source_strength,
        "sink_strength": sink_strength,
        "patch_radius": patch_radius,
        "core_radius": core_radius,
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "proxy_note": "Deterministic source/sink inverse-distance smoke proxy; not a FEM/FVM/OpenFOAM solve.",
    }
    return D1ProxyCase(arrays=arrays, metadata=metadata)


def save_d1_proxy_case(case: D1ProxyCase, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **case.arrays)
