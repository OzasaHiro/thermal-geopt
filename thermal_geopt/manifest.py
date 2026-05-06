"""Small manifest helpers for Thermal GeoPT experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path | str) -> Any:
    """Read a JSON document from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any) -> None:
    """Write a deterministic, human-readable JSON document."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _stable_token(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)


def _split_sizes(total: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    if not 0.0 <= train_frac <= 1.0:
        raise ValueError(f"train_frac must be in [0, 1], got {train_frac}")
    if not 0.0 <= val_frac <= 1.0:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")
    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac + val_frac must be <= 1, got {train_frac + val_frac}")
    train_count = int(total * train_frac)
    val_count = int(total * val_frac)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def stable_split(
    items: list[Any],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[Any]]:
    """Split items by a stable seed hash, then sort each split for readable diffs."""
    train_count, val_count, _ = _split_sizes(len(items), train_frac, val_frac)
    ranked = []
    for index, item in enumerate(items):
        token = _stable_token(item)
        digest = hashlib.sha256(f"{seed}\0{token}".encode("utf-8")).hexdigest()
        ranked.append((digest, token, index, item))
    ordered = [item for _, _, _, item in sorted(ranked)]

    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count :]
    return {
        "train": sorted(train, key=_stable_token),
        "val": sorted(val, key=_stable_token),
        "test": sorted(test, key=_stable_token),
    }
