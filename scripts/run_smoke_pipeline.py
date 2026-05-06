#!/usr/bin/env python3
"""Run the lightweight Thermal GeoPT preparation smoke pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT.parents[1] / ".venv" / "bin" / "python"


def run_step(name: str, command: list[str]) -> dict[str, object]:
    print(json.dumps({"step": name, "command": command}))
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return {
        "name": name,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--num-per-family", type=int, default=1)
    parser.add_argument("--surface-points", type=int, default=1024)
    parser.add_argument("--episodes-per-shape", type=int, default=1)
    parser.add_argument("--points-per-episode", type=int, default=256)
    parser.add_argument("--d1-points-per-case", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-transolver", action="store_true")
    parser.add_argument("--skip-d1", action="store_true")
    parser.add_argument("--skip-inspect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py = str(args.python)
    overwrite = ["--overwrite"] if args.overwrite else []
    steps = [
        [
            "tdf_brownian",
            [
                py,
                "scripts/smoke_tdf_brownian.py",
                "--num-points",
                "1024",
                "--steps",
                "2",
            ],
        ],
        [
            "cadquery_shapes",
            [
                py,
                "scripts/generate_cadquery_shapes.py",
                "--num-per-family",
                str(args.num_per_family),
                *overwrite,
            ],
        ],
        [
            "preprocess_meshes",
            [
                py,
                "scripts/preprocess_meshes.py",
                "--surface-points",
                str(args.surface_points),
                *overwrite,
            ],
        ],
        [
            "pretrain_episodes",
            [
                py,
                "scripts/generate_pretrain_episodes.py",
                "--episodes-per-shape",
                str(args.episodes_per_shape),
                "--points-per-episode",
                str(args.points_per_episode),
                "--steps",
                "2",
                *overwrite,
            ],
        ],
    ]
    if not args.skip_d1:
        steps.append(
            [
                "d1_proxy_cases",
                [
                    py,
                    "scripts/generate_d1_conduction_cases.py",
                    "--max-shapes",
                    "2",
                    "--cases-per-shape",
                    "1",
                    "--points-per-case",
                    str(args.d1_points_per_case),
                    *overwrite,
                ],
            ]
        )
    if not args.skip_inspect:
        inspect_paths = ["data/pretrain_zarr/tiny_smoke/manifest.json"]
        if not args.skip_d1:
            inspect_paths.insert(0, "data/downstream_npz/d1_proxy/manifest.json")
        steps.append(
            [
                "inspect_artifacts",
                [
                    py,
                    "scripts/inspect_artifacts.py",
                    *inspect_paths,
                    "--max-check-elements",
                    "10000",
                ],
            ]
        )
    if not args.skip_transolver:
        steps.insert(
            0,
            [
                "environment",
                [
                    py,
                    "scripts/check_environment.py",
                    "--transolver-smoke",
                    "--points",
                    "512",
                    "--amp",
                    "--amp-dtype",
                    "bfloat16",
                ],
            ],
        )

    results = [run_step(name, command) for name, command in steps]
    print(json.dumps({"results": results}, indent=2))
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
