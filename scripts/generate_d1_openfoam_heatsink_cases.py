#!/usr/bin/env python3
"""Generate solver-backed D1 heat-sink solid-conduction NPZ cases."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.openfoam_d1 import (
    convert_heatsink_case_to_npz,
    run_openfoam_case,
    sample_heatsink_params,
    write_openfoam_heatsink_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/downstream_raw/d1_openfoam_heatsink_smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/downstream_npz/d1_openfoam_heatsink_smoke"))
    parser.add_argument("--case-count", type=int, default=2)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("plate_fin", "pin_fin", "staggered_pin_fin"),
        default=("plate_fin", "pin_fin"),
    )
    parser.add_argument("--cells-x", type=int, default=24)
    parser.add_argument("--cells-y", type=int, default=24)
    parser.add_argument("--base-cells-z", type=int, default=4)
    parser.add_argument("--feature-cells-z", type=int, default=12)
    parser.add_argument("--source-temperature-min", type=float, default=360.0)
    parser.add_argument("--source-temperature-max", type=float, default=430.0)
    parser.add_argument("--sink-temperature-min", type=float, default=280.0)
    parser.add_argument("--sink-temperature-max", type=float, default=310.0)
    parser.add_argument(
        "--sink-value-fraction",
        type=float,
        default=1.0,
        help="OpenFOAM mixed boundary valueFraction for exterior cooling. 1.0 is fixed cold; lower values are weaker cooling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-prefix", default="d1_of_heatsink")
    parser.add_argument("--openfoam-bash", type=Path, default=Path("/opt/openfoam13/etc/bashrc"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-only", action="store_true", help="Only write OpenFOAM case directories; do not run or convert.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    for name in ("cells_x", "cells_y", "base_cells_z", "feature_cells_z"):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.source_temperature_min >= args.source_temperature_max:
        raise SystemExit("--source-temperature-min must be less than --source-temperature-max")
    if args.sink_temperature_min >= args.sink_temperature_max:
        raise SystemExit("--sink-temperature-min must be less than --sink-temperature-max")
    if not 0.0 <= args.sink_value_fraction <= 1.0:
        raise SystemExit("--sink-value-fraction must be between 0 and 1")
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
    families = list(args.families)
    for case_index in range(args.case_count):
        family = families[case_index % len(families)]
        case_id = f"{args.case_prefix}_{family}_{case_index:05d}"
        rng = np.random.default_rng(case_seed(args.seed, case_index))
        params = sample_heatsink_params(
            case_id=case_id,
            family=family,
            rng=rng,
            cells_xy=(args.cells_x, args.cells_y),
            base_cells_z=args.base_cells_z,
            feature_cells_z=args.feature_cells_z,
            source_temperature_range=(args.source_temperature_min, args.source_temperature_max),
            sink_temperature_range=(args.sink_temperature_min, args.sink_temperature_max),
            sink_value_fraction=args.sink_value_fraction,
        )
        case_dir = args.raw_dir / case_id
        npz_path = args.output_dir / f"{case_id}.npz"

        write_openfoam_heatsink_case(case_dir, params, overwrite=args.overwrite)
        written_cases.append({"case": case_id, "raw_case_dir": str(case_dir), **asdict(params)})
        print({"case": case_id, "family": family, "stage": "written", "raw_case_dir": str(case_dir)})

        if args.write_only:
            continue

        run_openfoam_case(case_dir, openfoam_bash=args.openfoam_bash)
        record = convert_heatsink_case_to_npz(case_dir, params, npz_path)
        records.append(record)
        print(
            {
                "case": case_id,
                "family": family,
                "stage": "converted",
                "path": str(npz_path),
                "points": record["points"],
            }
        )

    if args.write_only:
        manifest = {
            "description": "OpenFOAM D1 heat-sink case directories written but not solved.",
            "raw_dir": str(args.raw_dir),
            "output_dir": str(args.output_dir),
            "case_count": len(written_cases),
            "families": families,
            "cells_x": args.cells_x,
            "cells_y": args.cells_y,
            "base_cells_z": args.base_cells_z,
            "feature_cells_z": args.feature_cells_z,
            "source_temperature_min": args.source_temperature_min,
            "source_temperature_max": args.source_temperature_max,
            "sink_temperature_min": args.sink_temperature_min,
            "sink_temperature_max": args.sink_temperature_max,
            "sink_value_fraction": args.sink_value_fraction,
            "seed": args.seed,
            "records": written_cases,
            "note": "Run again without --write-only to solve with blockMesh/laplacianFoam and emit NPZ files.",
        }
        manifest_path = args.raw_dir / "manifest_write_only.json"
    else:
        manifest = {
            "description": "Solver-backed D1 heat-sink solid-conduction cases generated with OpenFOAM Foundation v13 laplacianFoam.",
            "raw_dir": str(args.raw_dir),
            "output_dir": str(args.output_dir),
            "case_count": len(records),
            "families": families,
            "cells_x": args.cells_x,
            "cells_y": args.cells_y,
            "base_cells_z": args.base_cells_z,
            "feature_cells_z": args.feature_cells_z,
            "source_temperature_min": args.source_temperature_min,
            "source_temperature_max": args.source_temperature_max,
            "sink_temperature_min": args.sink_temperature_min,
            "sink_temperature_max": args.sink_temperature_max,
            "sink_value_fraction": args.sink_value_fraction,
            "seed": args.seed,
            "records": records,
            "schema": {
                "training_compatibility_keys": ["points", "conditions", "temperature"],
                "detailed_plan_keys": ["points", "normals", "region", "material", "bc_features", "T", "case_params_json"],
            },
            "scope_note": (
                "M4 heat-sink D1 uses plate-fin and rectangularized pin-fin blockMesh geometries. "
                "It is solver-backed and geometry-varied, but still simpler than STL/snappyHexMesh CHT."
            ),
        }
        manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print({"manifest": str(manifest_path), "cases": manifest["case_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
