#!/usr/bin/env python3
"""Preprocess STL meshes into normalized sampled point arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Mesh is empty: {path}")
    loaded.remove_unreferenced_vertices()
    loaded.merge_vertices()
    loaded.fix_normals()
    return loaded


def normalize_mesh(mesh: trimesh.Trimesh, *, target_extent: float) -> tuple[trimesh.Trimesh, dict[str, object]]:
    bounds = mesh.bounds.astype(np.float64)
    center = 0.5 * (bounds[0] + bounds[1])
    extent = float(np.max(bounds[1] - bounds[0]))
    if extent <= 1e-12:
        raise ValueError("Cannot normalize degenerate mesh")
    scale = float(target_extent / extent)
    normalized = mesh.copy()
    normalized.vertices = (normalized.vertices - center) * scale
    normalized.fix_normals()
    return normalized, {"center": center.tolist(), "scale": scale, "source_extent": extent}


def sample_surface(mesh: trimesh.Trimesh, *, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    points, face_ids = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_ids]
    return points.astype(np.float32), normals.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/meshes_raw/cadquery"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/meshes_processed/cadquery"))
    parser.add_argument("--surface-points", type=int, default=8192)
    parser.add_argument("--target-extent", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stl_paths = sorted(args.input_dir.glob("*.stl"))
    if not stl_paths:
        raise SystemExit(f"No STL files found in {args.input_dir}")

    records = []
    for index, stl_path in enumerate(stl_paths):
        sample_name = stl_path.stem
        output_path = args.output_dir / f"{sample_name}.npz"
        if output_path.exists() and not args.overwrite:
            records.append({"sample": sample_name, "status": "exists", "output_path": str(output_path)})
            continue

        mesh = load_mesh(stl_path)
        normalized, transform = normalize_mesh(mesh, target_extent=args.target_extent)
        surface_points, surface_normals = sample_surface(
            normalized,
            count=args.surface_points,
            seed=args.seed + index,
        )
        np.savez_compressed(
            output_path,
            surface_points=surface_points,
            surface_normals=surface_normals,
            vertices=normalized.vertices.astype(np.float32),
            faces=normalized.faces.astype(np.int64),
            center=np.asarray(transform["center"], dtype=np.float32),
            scale=np.asarray(transform["scale"], dtype=np.float32),
        )
        record = {
            "sample": sample_name,
            "status": "written",
            "input_path": str(stl_path),
            "output_path": str(output_path),
            "vertices": int(normalized.vertices.shape[0]),
            "faces": int(normalized.faces.shape[0]),
            "surface_points": int(surface_points.shape[0]),
            "watertight": bool(normalized.is_watertight),
            "euler_number": int(normalized.euler_number),
            **transform,
        }
        records.append(record)
        print(record)

    manifest = {
        "description": "Normalized sampled mesh arrays for Thermal GeoPT.",
        "input_dir": str(args.input_dir),
        "surface_points": args.surface_points,
        "target_extent": args.target_extent,
        "seed": args.seed,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({"output_dir": str(args.output_dir), "records": len(records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
