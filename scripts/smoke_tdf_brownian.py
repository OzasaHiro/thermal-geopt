#!/usr/bin/env python3
"""Smoke test thermal diffusion features and Brownian trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.boundary_fields import nearest_boundary_field_values, sample_rbf_boundary_field
from thermal_geopt.brownian import BrownianConfig, simulate_brownian_walk
from thermal_geopt.geometry import sphere_surface_points
from thermal_geopt.tdf import ThermalFeatureConfig, thermal_diffusion_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--sphere-lat", type=int, default=32)
    parser.add_argument("--sphere-lon", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--conductivity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    boundary_points, boundary_normals = sphere_surface_points(num_lat=args.sphere_lat, num_lon=args.sphere_lon)
    query_points = rng.uniform(-0.75, 0.75, size=(args.num_points, 3)).astype(np.float32)
    source_centers = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    sink_centers = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32)

    features, names = thermal_diffusion_features(
        query_points,
        boundary_points,
        boundary_normals,
        config=ThermalFeatureConfig(alpha=args.alpha, conductivity=args.conductivity),
        source_centers=source_centers,
        sink_centers=sink_centers,
        return_names=True,
    )
    boundary_field = sample_rbf_boundary_field(boundary_points, num_patches=8, seed=args.seed)
    q_near = nearest_boundary_field_values(query_points, boundary_points, boundary_field)
    result = simulate_brownian_walk(
        query_points,
        boundary_points,
        boundary_normals,
        config=BrownianConfig(steps=args.steps, dt=args.dt, alpha=args.alpha, seed=args.seed),
    )

    if not np.all(np.isfinite(features)):
        raise SystemExit("Non-finite thermal features")
    if not np.all(np.isfinite(result.positions)):
        raise SystemExit("Non-finite Brownian positions")

    report = {
        "boundary_points": int(boundary_points.shape[0]),
        "query_points": int(query_points.shape[0]),
        "feature_shape": list(features.shape),
        "feature_names": names,
        "distance_min": float(features[:, 3].min()),
        "distance_max": float(features[:, 3].max()),
        "q_near_min": float(q_near.min()),
        "q_near_max": float(q_near.max()),
        "trajectory_shape": list(result.positions.shape),
        "hit_count": int(result.hit_mask.sum()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
