#!/usr/bin/env python3
"""Generate simple parametric thermal CAD shapes as STL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np
from cadquery import exporters


def make_plate_fin_heat_sink(rng: np.random.Generator) -> tuple[cq.Workplane, dict[str, object]]:
    width = float(rng.uniform(0.8, 1.4))
    depth = float(rng.uniform(0.8, 1.4))
    base_height = float(rng.uniform(0.08, 0.18))
    fin_height = float(rng.uniform(0.35, 0.9))
    fin_count = int(rng.integers(4, 11))
    fin_thickness = float(rng.uniform(0.025, 0.07))
    margin = 0.08 * width
    pitch = (width - 2.0 * margin) / max(fin_count - 1, 1)

    shape = cq.Workplane("XY").box(width, depth, base_height)
    for idx in range(fin_count):
        x = -0.5 * width + margin + idx * pitch
        fin = cq.Workplane("XY").box(fin_thickness, depth * 0.92, fin_height).translate(
            (x, 0.0, 0.5 * base_height + 0.5 * fin_height)
        )
        shape = shape.union(fin)

    params = {
        "family": "plate_fin",
        "width": width,
        "depth": depth,
        "base_height": base_height,
        "fin_height": fin_height,
        "fin_count": fin_count,
        "fin_thickness": fin_thickness,
    }
    return shape, params


def make_pin_fin_heat_sink(rng: np.random.Generator) -> tuple[cq.Workplane, dict[str, object]]:
    width = float(rng.uniform(0.8, 1.3))
    depth = float(rng.uniform(0.8, 1.3))
    base_height = float(rng.uniform(0.08, 0.18))
    pin_height = float(rng.uniform(0.25, 0.75))
    nx = int(rng.integers(3, 7))
    ny = int(rng.integers(3, 7))
    radius = float(rng.uniform(0.025, 0.06))

    shape = cq.Workplane("XY").box(width, depth, base_height)
    xs = np.linspace(-0.36 * width, 0.36 * width, nx)
    ys = np.linspace(-0.36 * depth, 0.36 * depth, ny)
    for x in xs:
        for y in ys:
            pin = cq.Workplane("XY").center(float(x), float(y)).circle(radius).extrude(pin_height).translate(
                (0.0, 0.0, 0.5 * base_height)
            )
            shape = shape.union(pin)

    params = {
        "family": "pin_fin",
        "width": width,
        "depth": depth,
        "base_height": base_height,
        "pin_height": pin_height,
        "nx": nx,
        "ny": ny,
        "pin_radius": radius,
    }
    return shape, params


def make_channel_block(rng: np.random.Generator) -> tuple[cq.Workplane, dict[str, object]]:
    width = float(rng.uniform(0.8, 1.4))
    depth = float(rng.uniform(0.6, 1.1))
    height = float(rng.uniform(0.18, 0.38))
    slot_count = int(rng.integers(2, 5))
    slot_width = float(rng.uniform(0.04, 0.09))
    slot_depth = depth * float(rng.uniform(0.55, 0.82))

    shape = cq.Workplane("XY").box(width, depth, height)
    xs = np.linspace(-0.32 * width, 0.32 * width, slot_count)
    for x in xs:
        cutter = cq.Workplane("XY").box(slot_width, slot_depth, height * 1.4).translate((float(x), 0.0, 0.08 * height))
        shape = shape.cut(cutter)

    params = {
        "family": "channel_block",
        "width": width,
        "depth": depth,
        "height": height,
        "slot_count": slot_count,
        "slot_width": slot_width,
        "slot_depth": slot_depth,
    }
    return shape, params


FAMILIES = {
    "plate_fin": make_plate_fin_heat_sink,
    "pin_fin": make_pin_fin_heat_sink,
    "channel_block": make_channel_block,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/meshes_raw/cadquery"))
    parser.add_argument("--num-per-family", type=int, default=4)
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for family in args.families:
        for index in range(args.num_per_family):
            shape, params = FAMILIES[family](rng)
            sample_name = f"{family}_{index:04d}"
            stl_path = args.output_dir / f"{sample_name}.stl"
            if stl_path.exists() and not args.overwrite:
                status = "exists"
            else:
                exporters.export(shape, str(stl_path), tolerance=0.01, angularTolerance=0.1)
                status = "written"
            records.append({"sample": sample_name, "status": status, "stl_path": str(stl_path), "params": params})
            print({"sample": sample_name, "status": status, "family": family})

    manifest = {
        "description": "CadQuery-generated thermal geometry STL files.",
        "seed": args.seed,
        "num_per_family": args.num_per_family,
        "families": args.families,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({"output_dir": str(args.output_dir), "records": len(records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
