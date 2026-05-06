#!/usr/bin/env python3
"""Validate label-scarcity gate inputs before running heavier training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.datasets import D1ProxyDataset, PretrainZarrDataset
from thermal_geopt.manifest import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=Path("data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json"))
    parser.add_argument("--split-path", type=Path, default=Path("configs/d1_proxy_pilot_300_c5_n8192_label_scarcity_split.json"))
    parser.add_argument("--pretrain-manifest", type=Path, default=Path("data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json"))
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[10, 25, 50, 100])
    return parser.parse_args()


def require_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"{key!r} must be a list in split file.")
    return [str(item) for item in value]


def manifest_case_ids(manifest_path: Path) -> set[str]:
    manifest = read_json(manifest_path)
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise SystemExit(f"{manifest_path} must contain a records list.")
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = str(record.get("case") or Path(str(record.get("path"))).name)
        ids.add(case_id)
        ids.add(Path(case_id).stem)
    return ids


def validate_split(case_manifest: Path, split_path: Path, train_sizes: list[int]) -> dict[str, Any]:
    split = read_json(split_path)
    if not isinstance(split, dict):
        raise SystemExit("Split file must be a JSON object.")
    known_ids = manifest_case_ids(case_manifest)
    val = require_list(split, "val")
    test = require_list(split, "test")
    val_set = set(val)
    test_set = set(test)
    if val_set & test_set:
        raise SystemExit("val and test overlap.")
    missing = sorted((val_set | test_set) - known_ids)
    if missing:
        raise SystemExit(f"Split contains IDs absent from manifest: {missing[:10]}")

    previous_train: set[str] | None = None
    counts = {"val": len(val), "test": len(test)}
    for train_size in train_sizes:
        key = f"train_{train_size}"
        train = require_list(split, key)
        train_set = set(train)
        if len(train) != train_size:
            raise SystemExit(f"{key} has {len(train)} items; expected {train_size}.")
        if train_set & val_set or train_set & test_set:
            raise SystemExit(f"{key} overlaps val/test.")
        missing = sorted(train_set - known_ids)
        if missing:
            raise SystemExit(f"{key} contains IDs absent from manifest: {missing[:10]}")
        if previous_train is not None and not previous_train.issubset(train_set):
            raise SystemExit(f"{key} is not a superset of the previous train subset.")
        previous_train = train_set
        counts[key] = len(train)

        # Ensure the dataset selection path accepts the split key.
        dataset = D1ProxyDataset(case_manifest, split_path=split_path, split=key, point_budget=8, max_cases=1)
        if len(dataset) != 1:
            raise SystemExit(f"Unexpected dataset length for {key}: {len(dataset)}")

    return counts


def validate_pretrain(pretrain_manifest: Path) -> dict[str, Any]:
    ablations = ["full", "no_boundary_field", "static_tdf_only"]
    report: dict[str, Any] = {}
    for ablation in ablations:
        dataset = PretrainZarrDataset(
            pretrain_manifest,
            point_budget=8,
            max_episodes=1,
            ablation=ablation,
            seed=42,
        )
        sample = dataset[0]
        report[ablation] = {
            "fun_dim": int(sample["fx"].shape[-1]),
            "out_dim": int(sample["y"].shape[-1]),
            "condition_mode": dataset.condition_mode,
            "target_names": dataset.target_names,
        }
    return report


def main() -> int:
    args = parse_args()
    payload = {
        "case_manifest": str(args.case_manifest),
        "split_path": str(args.split_path),
        "pretrain_manifest": str(args.pretrain_manifest),
        "split_counts": validate_split(args.case_manifest, args.split_path, args.train_sizes),
        "pretrain_ablations": validate_pretrain(args.pretrain_manifest),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
