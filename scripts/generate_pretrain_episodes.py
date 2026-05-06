#!/usr/bin/env python3
"""Generate tiny Thermal GeoPT pretraining shards from processed meshes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.boundary_fields import nearest_boundary_field_values, sample_rbf_boundary_field
from thermal_geopt.brownian import BrownianConfig, simulate_brownian_walk
from thermal_geopt.tdf import ThermalFeatureConfig, thermal_diffusion_features


def sample_query_points(
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    surface_count = count // 2
    volume_count = count - surface_count

    ids = rng.integers(0, boundary_points.shape[0], size=surface_count)
    offsets = rng.uniform(-0.12, 0.12, size=(surface_count, 1)).astype(np.float32)
    shell_points = boundary_points[ids] + offsets * boundary_normals[ids]

    bounds_min = boundary_points.min(axis=0) - 0.08
    bounds_max = boundary_points.max(axis=0) + 0.08
    volume_points = rng.uniform(bounds_min, bounds_max, size=(volume_count, 3)).astype(np.float32)
    points = np.concatenate([shell_points, volume_points], axis=0).astype(np.float32)
    rng.shuffle(points)
    return points


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/meshes_processed/cadquery"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/pretrain_zarr/tiny_smoke"))
    parser.add_argument("--episodes-per-shape", type=int, default=2)
    parser.add_argument("--points-per-episode", type=int, default=2048)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_paths = sorted(args.processed_dir.glob("*.npz"))
    if args.max_shapes > 0:
        processed_paths = processed_paths[: args.max_shapes]
    if not processed_paths:
        raise SystemExit(f"No processed mesh npz files found in {args.processed_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    shard_records = []
    feature_names: list[str] | None = None
    shard_index = 0
    for mesh_index, processed_path in enumerate(processed_paths):
        mesh = np.load(processed_path)
        boundary_points = mesh["surface_points"].astype(np.float32)
        boundary_normals = mesh["surface_normals"].astype(np.float32)

        xs = []
        conds = []
        ys = []
        trajectories = []
        hit_masks = []
        hit_steps = []
        episode_meta = []
        for episode_index in range(args.episodes_per_shape):
            alpha = log_uniform(rng, 0.05, 2.0)
            conductivity = log_uniform(rng, 0.1, 10.0)
            query_points = sample_query_points(
                boundary_points,
                boundary_normals,
                count=args.points_per_episode,
                rng=rng,
            )
            source_center = boundary_points[rng.integers(0, boundary_points.shape[0])][None, :]
            sink_center = boundary_points[rng.integers(0, boundary_points.shape[0])][None, :]
            features, names = thermal_diffusion_features(
                query_points,
                boundary_points,
                boundary_normals,
                config=ThermalFeatureConfig(alpha=alpha, conductivity=conductivity),
                source_centers=source_center,
                sink_centers=sink_center,
                return_names=True,
            )
            feature_names = names
            boundary_field = sample_rbf_boundary_field(
                boundary_points,
                num_patches=8,
                seed=args.seed + mesh_index * 1000 + episode_index,
            )
            q_near = nearest_boundary_field_values(query_points, boundary_points, boundary_field)[:, None]
            alpha_col = np.full((args.points_per_episode, 1), alpha, dtype=np.float32)
            conductivity_col = np.full((args.points_per_episode, 1), conductivity, dtype=np.float32)
            cond = np.concatenate([alpha_col, conductivity_col, q_near], axis=1)

            trajectory = simulate_brownian_walk(
                query_points,
                boundary_points,
                boundary_normals,
                config=BrownianConfig(
                    steps=args.steps,
                    dt=float(rng.uniform(0.01, 0.05)),
                    alpha=alpha,
                    boundary_mode="partial_absorbing",
                    seed=args.seed + mesh_index * 1000 + episode_index,
                ),
            )
            xs.append(query_points)
            conds.append(cond)
            ys.append(features)
            trajectories.append(np.moveaxis(trajectory.positions, 0, 1))
            hit_masks.append(trajectory.hit_mask)
            hit_steps.append(trajectory.hit_step)
            episode_meta.append(
                {
                    "sample": processed_path.stem,
                    "episode_index": episode_index,
                    "alpha": alpha,
                    "conductivity": conductivity,
                }
            )

        shard_path = args.output_dir / f"shard_{shard_index:06d}.zarr"
        if shard_path.exists():
            if not args.overwrite:
                raise SystemExit(f"Shard already exists: {shard_path}")
            shutil.rmtree(shard_path)
        zarr.save_group(
            str(shard_path),
            x=np.stack(xs).astype(np.float16),
            cond=np.stack(conds).astype(np.float16),
            y_tdf=np.stack(ys).astype(np.float16),
            trajectory=np.stack(trajectories).astype(np.float16),
            hit_mask=np.stack(hit_masks),
            hit_step=np.stack(hit_steps),
        )
        meta = {
            "processed_path": str(processed_path),
            "feature_names": feature_names,
            "condition_names": ["alpha", "conductivity", "q_near"],
            "episodes": episode_meta,
        }
        (shard_path / "meta.json").write_text(json.dumps(meta, indent=2))
        shard_records.append(
            {
                "shard": str(shard_path),
                "sample": processed_path.stem,
                "episodes": args.episodes_per_shape,
                "points_per_episode": args.points_per_episode,
                "feature_dim": len(feature_names or []),
            }
        )
        print(shard_records[-1])
        shard_index += 1

    manifest = {
        "description": "Tiny Thermal GeoPT pretraining shards.",
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "episodes_per_shape": args.episodes_per_shape,
        "points_per_episode": args.points_per_episode,
        "steps": args.steps,
        "seed": args.seed,
        "shards": shard_records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({"output_dir": str(args.output_dir), "shards": len(shard_records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
