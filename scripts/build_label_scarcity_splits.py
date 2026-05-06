#!/usr/bin/env python3
"""Build deterministic label-scarcity train subsets from an existing split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.manifest import read_json, write_json

REQUIRED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="train")
    return parser.parse_args()


def _stable_token(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)


def _id_aliases(item: Any) -> set[str]:
    token = _stable_token(item)
    aliases = {token}
    path = Path(token)
    if path.stem:
        aliases.add(path.stem)
    return aliases


def _split_list(base_split: dict[str, Any], split_name: str) -> list[Any]:
    value = base_split.get(split_name)
    if not isinstance(value, list):
        raise SystemExit(f"Base split must contain a list field named '{split_name}'.")
    return list(value)


def _validate_unique_ids(split_items: dict[str, list[Any]]) -> None:
    all_aliases = []
    for items in split_items.values():
        for item in items:
            all_aliases.extend(_id_aliases(item))
    duplicates = sorted(sample_id for sample_id, count in Counter(all_aliases).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Base split contains duplicate ids/aliases across train/val/test: {duplicates[:10]}")


def _validate_train_sizes(train_sizes: list[int], base_train_count: int) -> list[int]:
    if any(size <= 0 for size in train_sizes):
        raise SystemExit(f"Train sizes must be positive, got: {train_sizes}")
    duplicates = sorted(size for size, count in Counter(train_sizes).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Train sizes must be unique, duplicates: {duplicates}")
    too_large = [size for size in train_sizes if size > base_train_count]
    if too_large:
        raise SystemExit(
            "Train size exceeds base train count: "
            f"requested {max(too_large)}, base train has {base_train_count}"
        )
    return train_sizes


def _stable_shuffle(items: list[Any], *, seed: int) -> list[Any]:
    ranked = []
    for index, item in enumerate(items):
        token = _stable_token(item)
        digest = hashlib.sha256(f"{seed}\0{token}\0{index}".encode("utf-8")).hexdigest()
        ranked.append((digest, token, index, item))
    return [item for _, _, _, item in sorted(ranked)]


def _split_role(train_key: str, train_items: list[Any], val: list[Any], test: list[Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    roles.update((_stable_token(item), train_key) for item in train_items)
    roles.update((_stable_token(item), "val") for item in val)
    roles.update((_stable_token(item), "test") for item in test)
    return dict(sorted(roles.items()))


def build_label_scarcity_split(
    base_split: dict[str, Any],
    *,
    base_split_path: Path,
    train_sizes: list[int],
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    split_items = {split_name: _split_list(base_split, split_name) for split_name in REQUIRED_SPLITS}
    _validate_unique_ids(split_items)
    train_sizes = _validate_train_sizes(train_sizes, len(split_items["train"]))

    shuffled_train = _stable_shuffle(split_items["train"], seed=seed)
    counts: dict[str, int] = {
        "base_train": len(split_items["train"]),
        "val": len(split_items["val"]),
        "test": len(split_items["test"]),
    }
    payload: dict[str, Any] = {
        "description": "Deterministic label-scarcity split plan derived from an existing split.",
        "base_split": str(base_split_path),
        "seed": seed,
        "prefix": prefix,
        "train_sizes": train_sizes,
        "counts": counts,
        "train": split_items["train"],
        "val": split_items["val"],
        "test": split_items["test"],
    }

    for train_size in train_sizes:
        train_key = f"{prefix}_{train_size}"
        role_key = f"split_role_{train_size}"
        train_subset = shuffled_train[:train_size]
        counts[train_key] = len(train_subset)
        payload[train_key] = train_subset
        payload[role_key] = _split_role(train_key, train_subset, split_items["val"], split_items["test"])

    return payload


def main() -> int:
    args = parse_args()
    base_split = read_json(args.base_split)
    if not isinstance(base_split, dict):
        raise SystemExit("Base split JSON must be an object containing train, val, and test lists.")

    payload = build_label_scarcity_split(
        base_split,
        base_split_path=args.base_split,
        train_sizes=args.train_sizes,
        seed=args.seed,
        prefix=args.prefix,
    )
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
