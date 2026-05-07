#!/usr/bin/env python3
"""Generate solver-backed D1 solid-conduction NPZ cases with OpenFOAM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.openfoam_d1 import (
    convert_case_to_npz,
    run_openfoam_case,
    sample_block_params,
    write_openfoam_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/downstream_raw/d1_openfoam_block_smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/downstream_npz/d1_openfoam_block_smoke"))
    parser.add_argument("--case-count", type=int, default=1)
    parser.add_argument("--cells", type=int, nargs=3, default=(8, 8, 8), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-prefix", default="d1_of_block")
    parser.add_argument("--openfoam-bash", type=Path, default=Path("/opt/openfoam13/etc/bashrc"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-only", action="store_true", help="Only write OpenFOAM case directories; do not run or convert.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    if any(value <= 0 for value in args.cells):
        raise SystemExit("--cells values must be positive")
    if not args.openfoam_bash.exists():
        raise SystemExit(f"OpenFOAM bashrc not found: {args.openfoam_bash}")


def case_seed(base_seed: int, case_index: int) -> int:
    seed_seq = np.random.SeedSequence([base_seed, case_index])
    return int(seed_seq.generate_state(1, dtype="uint32")[0])


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    written_cases = []
    for case_index in range(args.case_count):
        case_id = f"{args.case_prefix}_{case_index:05d}"
        rng = np.random.default_rng(case_seed(args.seed, case_index))
        params = sample_block_params(case_id=case_id, rng=rng, cells=tuple(args.cells))
        case_dir = args.raw_dir / case_id
        npz_path = args.output_dir / f"{case_id}.npz"

        write_openfoam_case(case_dir, params, overwrite=args.overwrite)
        written_cases.append({"case": case_id, "raw_case_dir": str(case_dir), **params.__dict__})
        print({"case": case_id, "stage": "written", "raw_case_dir": str(case_dir)})

        if args.write_only:
            continue

        run_openfoam_case(case_dir, openfoam_bash=args.openfoam_bash)
        record = convert_case_to_npz(case_dir, params, npz_path)
        records.append(record)
        print({"case": case_id, "stage": "converted", "path": str(npz_path), "points": record["points"]})

    if args.write_only:
        manifest = {
            "description": "OpenFOAM D1 solid-conduction case directories written but not solved.",
            "raw_dir": str(args.raw_dir),
            "output_dir": str(args.output_dir),
            "case_count": len(written_cases),
            "cells": list(args.cells),
            "seed": args.seed,
            "records": written_cases,
            "note": "Run again without --write-only to solve with blockMesh/laplacianFoam and emit NPZ files.",
        }
        manifest_path = args.raw_dir / "manifest_write_only.json"
    else:
        manifest = {
            "description": "Solver-backed D1 solid-conduction block cases generated with OpenFOAM Foundation v13 laplacianFoam.",
            "raw_dir": str(args.raw_dir),
            "output_dir": str(args.output_dir),
            "case_count": len(records),
            "cells": list(args.cells),
            "seed": args.seed,
            "records": records,
            "schema": {
                "training_compatibility_keys": ["points", "conditions", "temperature"],
                "detailed_plan_keys": ["points", "normals", "region", "material", "bc_features", "T", "case_params_json"],
            },
            "m1_scope_note": "M1 minimal blockMesh case with fixed-temperature source/sink patches and insulated sides. Use for solver-backed pipeline validation before plate-fin/pin-fin/snappyHexMesh scale-up.",
        }
        manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print({"manifest": str(manifest_path), "cases": manifest["case_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
