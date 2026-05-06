#!/usr/bin/env python3
"""Build deterministic train/val/test split plans from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.manifest import read_json, stable_split, write_json

ITEM_LIST_KEYS = ("records", "cases", "samples", "items", "shards")
ID_KEYS = ("sample", "case", "case_id", "id", "name")
PATH_KEYS = ("output_path", "path", "npz_path", "processed_path", "case_dir", "sample_dir", "shard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def manifest_items(manifest: Any) -> tuple[list[Any], str]:
    if isinstance(manifest, list):
        return manifest, "root"
    if not isinstance(manifest, dict):
        raise SystemExit("Input manifest must be a JSON object or list.")

    for key in ITEM_LIST_KEYS:
        value = manifest.get(key)
        if isinstance(value, list):
            return value, key
    raise SystemExit(f"Input manifest does not contain one of: {', '.join(ITEM_LIST_KEYS)}")


def item_id(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)

    for key in ID_KEYS:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    for key in PATH_KEYS:
        value = item.get(key)
        if value not in (None, ""):
            return Path(str(value)).stem
    raise SystemExit(f"Manifest item has no usable id or path field: {item}")


def unique_ids(items: list[Any]) -> list[str]:
    ids = [item_id(item) for item in items]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate manifest ids: {duplicates[:10]}")
    return sorted(ids)


def main() -> int:
    args = parse_args()
    manifest = read_json(args.input_manifest)
    items, source_key = manifest_items(manifest)
    ids = unique_ids(items)
    try:
        split = stable_split(ids, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    split_role = {sample_id: role for role, sample_ids in split.items() for sample_id in sample_ids}
    payload = {
        "description": "Deterministic train/val/test split plan.",
        "source_manifest": str(args.input_manifest),
        "source_key": source_key,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(1.0 - args.train_frac - args.val_frac, 12),
        "counts": {
            "total": len(ids),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "split_role": dict(sorted(split_role.items())),
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
