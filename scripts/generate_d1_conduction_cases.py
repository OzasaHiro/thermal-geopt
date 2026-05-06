#!/usr/bin/env python3
"""Generate tiny D1 solid-conduction proxy NPZ cases.

The generated fields are deterministic source/sink influence proxies for smoke
runs. They do not require OpenFOAM and are not full FEM/FVM solutions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.d1_conduction import generate_d1_proxy_case, load_processed_mesh, save_d1_proxy_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/meshes_processed/cadquery"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/downstream_npz/d1_proxy"))
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--cases-per-shape", type=int, default=2)
    parser.add_argument("--points-per-case", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_seed(base_seed: int, mesh_index: int, case_index: int) -> int:
    seed_seq = np.random.SeedSequence([base_seed, mesh_index, case_index])
    return int(seed_seq.generate_state(1, dtype="uint32")[0])


def main() -> int:
    args = parse_args()
    if args.max_shapes < 0:
        raise SystemExit("--max-shapes must be non-negative")
    if args.cases_per_shape <= 0:
        raise SystemExit("--cases-per-shape must be positive")
    if args.points_per_case <= 0:
        raise SystemExit("--points-per-case must be positive")

    processed_paths = sorted(args.processed_dir.glob("*.npz"))
    if args.max_shapes > 0:
        processed_paths = processed_paths[: args.max_shapes]
    if not processed_paths:
        raise SystemExit(f"No processed mesh npz files found in {args.processed_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Manifest already exists: {manifest_path}. Use --overwrite to replace it.")

    records = []
    for mesh_index, processed_path in enumerate(processed_paths):
        mesh = load_processed_mesh(processed_path)
        for local_case_index in range(args.cases_per_shape):
            output_name = f"{mesh.name}_case_{local_case_index:04d}.npz"
            output_path = args.output_dir / output_name
            if output_path.exists() and not args.overwrite:
                raise SystemExit(f"Case already exists: {output_path}. Use --overwrite to replace it.")

            seed = case_seed(args.seed, mesh_index, local_case_index)
            case = generate_d1_proxy_case(
                mesh,
                points_per_case=args.points_per_case,
                seed=seed,
                case_index=local_case_index,
            )
            save_d1_proxy_case(case, output_path)
            record = {
                "case": output_name,
                "path": str(output_path),
                **case.metadata,
            }
            records.append(record)
            print(record)

    manifest = {
        "description": "D1 solid-conduction smoke/proxy cases generated without OpenFOAM.",
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "max_shapes": args.max_shapes,
        "cases_per_shape": args.cases_per_shape,
        "points_per_case": args.points_per_case,
        "seed": args.seed,
        "case_count": len(records),
        "records": records,
        "proxy_note": "Temperature and heat flux are deterministic source/sink influence fields, not FEM/FVM/OpenFOAM solutions.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print({"output_dir": str(args.output_dir), "cases": len(records), "manifest": str(manifest_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
