#!/usr/bin/env python3
"""Validate Thermal GeoPT pretraining shards before heavy training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.datasets import PretrainZarrDataset, resolve_existing_path


REQUIRED_ARRAYS = ("x", "cond", "y_tdf", "trajectory", "hit_mask", "hit_step")
PHASES = {
    "P0": {"shapes": 100, "episodes": 500},
    "P1": {"shapes": 500, "episodes": 5_000},
    "P2": {"shapes": 2_000, "episodes": 40_000},
    "P3": {"shapes": 8_000, "episodes": 160_000},
    "P4": {"shapes": 10_000, "episodes": 500_000},
}
POINT_TIERS = (
    ("base+", 14_336),
    ("base", 9_216),
    ("pilot", 5_120),
    ("tiny", 2_560),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Pretraining Zarr manifest.json")
    parser.add_argument("--max-shards-check", type=int, default=5)
    parser.add_argument("--sample-points-check", type=int, default=1024)
    parser.add_argument(
        "--ablation",
        action="append",
        choices=["full", "no_boundary_field", "static_tdf_only", "dynamics_lifted"],
        default=None,
        help="Dataset ablation to instantiate. Repeatable. Defaults to dynamics_lifted plus key ablations.",
    )
    parser.add_argument(
        "--require-phase",
        choices=sorted(PHASES),
        default=None,
        help="Fail if the manifest is below this planned pretraining scale.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def total_episodes(manifest: dict[str, Any]) -> int:
    return int(sum(int(shard.get("episodes", 0)) for shard in manifest.get("shards", [])))


def classify_phase(*, shapes: int, episodes: int) -> str:
    best = "below_P0"
    for phase, threshold in PHASES.items():
        if shapes >= threshold["shapes"] and episodes >= threshold["episodes"]:
            best = phase
    return best


def phase_rank(phase: str) -> int:
    order = ["below_P0", *sorted(PHASES)]
    return order.index(phase)


def classify_points(points_per_episode: int) -> str:
    for tier, threshold in POINT_TIERS:
        if points_per_episode >= threshold:
            return tier
    return "below_tiny"


def family_name_from_sample(sample: str) -> str:
    prefix, separator, suffix = sample.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return sample


def family_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard in manifest.get("shards", []):
        if not isinstance(shard, dict):
            continue
        family = str(shard.get("family") or family_name_from_sample(str(shard.get("sample") or Path(str(shard.get("shard", ""))).stem)))
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def require_array(group: Any, name: str) -> Any:
    try:
        return group[name]
    except KeyError as exc:
        raise ValueError(f"Missing required Zarr array {name!r}") from exc


def finite_check(array: Any, selection: tuple[slice, ...]) -> dict[str, Any]:
    values = np.asarray(array[selection])
    if values.dtype.kind not in {"b", "i", "u", "f"}:
        return {"status": "skipped", "dtype": str(values.dtype), "reason": "non_numeric"}
    finite = np.isfinite(values) if values.dtype.kind == "f" else np.ones(values.shape, dtype=bool)
    return {
        "status": "ok",
        "checked_elements": int(values.size),
        "all_finite": bool(finite.all()),
        "min": values.min().item() if values.size else None,
        "max": values.max().item() if values.size else None,
    }


def inspect_shard(shard_path: Path, *, sample_points_check: int) -> dict[str, Any]:
    import zarr

    group = zarr.open_group(str(shard_path), mode="r")
    arrays = {name: require_array(group, name) for name in REQUIRED_ARRAYS}
    x_shape = tuple(int(dim) for dim in arrays["x"].shape)
    cond_shape = tuple(int(dim) for dim in arrays["cond"].shape)
    y_shape = tuple(int(dim) for dim in arrays["y_tdf"].shape)
    traj_shape = tuple(int(dim) for dim in arrays["trajectory"].shape)
    hit_mask_shape = tuple(int(dim) for dim in arrays["hit_mask"].shape)
    hit_step_shape = tuple(int(dim) for dim in arrays["hit_step"].shape)

    if len(x_shape) != 3 or x_shape[-1] != 3:
        raise ValueError(f"{shard_path}: x must have shape (episodes, points, 3), got {x_shape}")
    episodes, points, _ = x_shape
    if len(cond_shape) != 3 or cond_shape[:2] != (episodes, points):
        raise ValueError(f"{shard_path}: cond shape {cond_shape} is inconsistent with x {x_shape}")
    if len(y_shape) != 3 or y_shape[:2] != (episodes, points):
        raise ValueError(f"{shard_path}: y_tdf shape {y_shape} is inconsistent with x {x_shape}")
    if len(traj_shape) != 4 or traj_shape[0] != episodes or traj_shape[1] != points or traj_shape[-1] != 3:
        raise ValueError(f"{shard_path}: trajectory must have shape (episodes, points, steps+1, 3), got {traj_shape}")
    if hit_mask_shape != (episodes, points):
        raise ValueError(f"{shard_path}: hit_mask shape {hit_mask_shape} is inconsistent with x {x_shape}")
    if hit_step_shape != (episodes, points):
        raise ValueError(f"{shard_path}: hit_step shape {hit_step_shape} is inconsistent with x {x_shape}")

    point_slice = slice(0, min(points, sample_points_check))
    checks = {
        "x": finite_check(arrays["x"], (slice(0, 1), point_slice, slice(None))),
        "cond": finite_check(arrays["cond"], (slice(0, 1), point_slice, slice(None))),
        "y_tdf": finite_check(arrays["y_tdf"], (slice(0, 1), point_slice, slice(None))),
        "trajectory": finite_check(arrays["trajectory"], (slice(0, 1), point_slice, slice(None), slice(None))),
        "hit_mask": finite_check(arrays["hit_mask"], (slice(0, 1), point_slice)),
        "hit_step": finite_check(arrays["hit_step"], (slice(0, 1), point_slice)),
    }
    failed = [name for name, check in checks.items() if check.get("all_finite") is False]
    if failed:
        raise ValueError(f"{shard_path}: non-finite values found in {failed}")

    meta_path = shard_path / "meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return {
        "shard": str(shard_path),
        "episodes": episodes,
        "points_per_episode": points,
        "condition_dim": int(cond_shape[-1]),
        "feature_dim": int(y_shape[-1]),
        "trajectory_steps_plus_one": int(traj_shape[2]),
        "condition_schema": meta.get("condition_schema"),
        "condition_names": meta.get("condition_names"),
        "feature_names": meta.get("feature_names"),
        "finite_checks": checks,
    }


def validate_dataset_modes(manifest_path: Path, ablations: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for ablation in ablations:
        dataset = PretrainZarrDataset(
            manifest_path,
            point_budget=8,
            max_episodes=1,
            seed=42,
            ablation=ablation,
        )
        sample = dataset[0]
        report[ablation] = {
            "condition_mode": dataset.condition_mode,
            "fun_dim": int(sample["fx"].shape[-1]),
            "out_dim": int(sample["y"].shape[-1]),
            "target_slices": {key: list(value) for key, value in dataset.target_slices.items()},
            "condition_names": dataset.condition_names,
        }
    return report


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest)
    if not isinstance(manifest, dict):
        raise SystemExit("Manifest must be a JSON object.")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise SystemExit("Manifest must contain a non-empty shards list.")

    shard_count = len(shards)
    episode_count = total_episodes(manifest)
    points_per_episode = int(manifest.get("points_per_episode", 0))
    family_distribution = family_counts(manifest)
    phase = classify_phase(shapes=shard_count, episodes=episode_count)
    point_tier = classify_points(points_per_episode)

    checked_shards = []
    schema_anchor: dict[str, Any] | None = None
    for shard in shards[: max(1, args.max_shards_check)]:
        shard_path = resolve_existing_path(str(shard["shard"]), base_dir=args.manifest.parent)
        report = inspect_shard(shard_path, sample_points_check=args.sample_points_check)
        schema = {
            "condition_schema": report.get("condition_schema"),
            "condition_names": report.get("condition_names"),
            "feature_names": report.get("feature_names"),
            "condition_dim": report.get("condition_dim"),
            "feature_dim": report.get("feature_dim"),
            "points_per_episode": report.get("points_per_episode"),
        }
        if schema_anchor is None:
            schema_anchor = schema
        elif schema != schema_anchor:
            raise SystemExit(f"Shard schema mismatch: {report['shard']} has {schema}, expected {schema_anchor}")
        checked_shards.append(report)

    ablations = args.ablation or ["dynamics_lifted", "static_tdf_only", "no_boundary_field"]
    dataset_modes = validate_dataset_modes(args.manifest, ablations)
    warnings = []
    if phase_rank(phase) < phase_rank("P2"):
        warnings.append(
            "Scale is below P2 first-result tier. Use this for schema/loss-speed checks only, not GeoPT efficacy claims."
        )
    if point_tier in {"below_tiny", "tiny"}:
        warnings.append("Point count is below pilot tier; it is not representative of the planned pretraining regime.")
    if len(family_distribution) < 3:
        warnings.append("Fewer than three geometry families detected; shape diversity is weak for GeoPT-style pretraining.")
    elif family_distribution:
        counts = list(family_distribution.values())
        if min(counts) > 0 and max(counts) / min(counts) > 2.0:
            warnings.append("Geometry family distribution is imbalanced; use --selection balanced when truncating shapes.")
    if schema_anchor and schema_anchor.get("condition_schema") != manifest.get("condition_schema"):
        warnings.append("Manifest condition_schema and shard meta condition_schema differ.")

    output = {
        "manifest": str(args.manifest),
        "condition_schema": manifest.get("condition_schema"),
        "shards": shard_count,
        "total_episodes": episode_count,
        "points_per_episode": points_per_episode,
        "family_counts": family_distribution,
        "scale_phase": phase,
        "point_tier": point_tier,
        "checked_shards": checked_shards,
        "dataset_modes": dataset_modes,
        "warnings": warnings,
    }
    print(json.dumps(output, indent=2, default=json_default))

    if args.require_phase and phase_rank(phase) < phase_rank(args.require_phase):
        raise SystemExit(f"Manifest is {phase}, below required phase {args.require_phase}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
