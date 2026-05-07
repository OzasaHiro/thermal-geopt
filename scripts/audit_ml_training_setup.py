#!/usr/bin/env python3
"""Audit whether pretraining and downstream fine-tuning form a valid ML transfer experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.datasets import resolve_existing_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrain-manifest",
        type=Path,
        default=Path("data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json"),
    )
    parser.add_argument(
        "--pretrain-run",
        type=Path,
        default=Path("outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20"),
    )
    parser.add_argument(
        "--downstream-manifest",
        type=Path,
        default=Path("data/downstream_npz/d1_openfoam_block_m3_300/manifest.json"),
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("configs/d1_openfoam_block_m3_300_split_seed42.json"),
    )
    parser.add_argument(
        "--scratch-run",
        type=Path,
        default=Path("outputs/checkpoints/m3_openfoam_p2_ft_tuned_oclr_scratch_split42_trainseed42_train50_ep100"),
    )
    parser.add_argument(
        "--pretrained-run",
        type=Path,
        default=Path("outputs/checkpoints/m3_openfoam_p2_ft_tuned_oclr_dynamics_lifted_split42_trainseed42_train50_ep100"),
    )
    parser.add_argument("--max-pretrain-shards", type=int, default=6)
    parser.add_argument("--max-pretrain-points", type=int, default=4096)
    parser.add_argument("--max-downstream-cases", type=int, default=60)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/logs/ml_training_setup_audit_2026-05-07.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/ml_training_setup_audit_2026-05-07.md"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
    return {
        "count": int(flat.shape[0]),
        "dim": int(flat.shape[1]),
        "mean": [float(value) for value in flat.mean(axis=0)],
        "std": [float(value) for value in flat.std(axis=0)],
        "min": [float(value) for value in flat.min(axis=0)],
        "max": [float(value) for value in flat.max(axis=0)],
    }


def compact_vector(values: Iterable[float], digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}g}" for value in values) + "]"


def item_id(record: dict[str, Any]) -> str:
    for key in ("case", "case_id", "sample", "id", "name"):
        if record.get(key):
            return str(record[key])
    if record.get("path"):
        return Path(str(record["path"])).stem
    raise ValueError(f"Manifest record has no id: {record}")


def item_path(record: dict[str, Any], manifest_path: Path) -> Path:
    raw = Path(str(record["path"]))
    return resolve_existing_path(raw, base_dir=manifest_path.parent)


def split_overlap_report(split_path: Path) -> dict[str, Any]:
    split = read_json(split_path)
    report: dict[str, Any] = {"path": str(split_path)}
    names = [name for name in ("train", "val", "test") if isinstance(split.get(name), list)]
    for name in names:
        report[f"{name}_count"] = len(split[name])
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        left_items = {str(item) for item in split[left]}
        for right in names[left_index + 1 :]:
            right_items = {str(item) for item in split[right]}
            overlaps[f"{left}_{right}"] = len(left_items & right_items)
    report["overlaps"] = overlaps
    return report


def collect_pretrain_stats(manifest_path: Path, *, max_shards: int, max_points: int) -> dict[str, Any]:
    import zarr

    manifest = read_json(manifest_path)
    shards = manifest.get("shards") or []
    x_rows: list[np.ndarray] = []
    cond_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    delta_1_rows: list[np.ndarray] = []
    delta_final_rows: list[np.ndarray] = []
    hit_mask_rows: list[np.ndarray] = []
    hit_step_rows: list[np.ndarray] = []
    condition_names: list[str] | None = None
    feature_names: list[str] | None = None

    for shard_record in shards[:max_shards]:
        shard_path = resolve_existing_path(str(shard_record["shard"]), base_dir=manifest_path.parent)
        group = zarr.open_group(str(shard_path), mode="r")
        meta_path = shard_path / "meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        condition_names = condition_names or list(meta.get("condition_names") or [])
        feature_names = feature_names or list(meta.get("feature_names") or [])
        count = min(int(group["x"].shape[1]), max_points)
        ids = np.linspace(0, int(group["x"].shape[1]) - 1, count, dtype=np.int64)
        x_rows.append(np.asarray(group["x"][0], dtype=np.float32)[ids])
        cond_rows.append(np.asarray(group["cond"][0], dtype=np.float32)[ids])
        y_rows.append(np.asarray(group["y_tdf"][0], dtype=np.float32)[ids])
        trajectory = np.asarray(group["trajectory"][0], dtype=np.float32)[ids]
        start = trajectory[:, 0, :]
        delta_1_rows.append(trajectory[:, 1, :] - start)
        delta_final_rows.append(trajectory[:, -1, :] - start)
        hit_mask_rows.append(np.asarray(group["hit_mask"][0], dtype=np.float32)[ids, None])
        hit_step = np.asarray(group["hit_step"][0], dtype=np.float32)[ids, None]
        miss_step = float(max(trajectory.shape[1], 1))
        hit_step_rows.append(np.where(hit_step >= 0.0, hit_step, miss_step) / miss_step)

    return {
        "manifest": str(manifest_path),
        "condition_schema": manifest.get("condition_schema"),
        "shards": len(shards),
        "episodes": int(sum(int(shard.get("episodes", 0)) for shard in shards)),
        "points_per_episode": int(manifest.get("points_per_episode", 0)),
        "condition_names": condition_names or [],
        "feature_names": feature_names or [],
        "x": stats(np.concatenate(x_rows, axis=0)),
        "cond": stats(np.concatenate(cond_rows, axis=0)),
        "y_tdf": stats(np.concatenate(y_rows, axis=0)),
        "delta_1": stats(np.concatenate(delta_1_rows, axis=0)),
        "delta_final": stats(np.concatenate(delta_final_rows, axis=0)),
        "hit_mask": stats(np.concatenate(hit_mask_rows, axis=0)),
        "hit_step_norm": stats(np.concatenate(hit_step_rows, axis=0)),
    }


def collect_downstream_stats(manifest_path: Path, split_path: Path, *, max_cases: int) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    records = manifest.get("records") or []
    split = read_json(split_path)
    train_ids = {str(item) for item in split.get("train", [])}
    selected = []
    for record in records:
        case_id = item_id(record)
        if not train_ids or case_id in train_ids or Path(case_id).stem in train_ids:
            selected.append(record)
        if len(selected) >= max_cases:
            break

    points: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    condition_names: list[str] | None = None
    for record in selected:
        path = item_path(record, manifest_path)
        with np.load(path) as data:
            points.append(np.asarray(data["points"], dtype=np.float32))
            conditions.append(np.asarray(data["conditions"], dtype=np.float32))
            temp = np.asarray(data["temperature"], dtype=np.float32)
            temperatures.append(temp[:, None] if temp.ndim == 1 else temp[:, :1])
            if condition_names is None and "condition_names" in data:
                condition_names = [str(value) for value in data["condition_names"].tolist()]

    temp_all = np.concatenate(temperatures, axis=0)
    return {
        "manifest": str(manifest_path),
        "sampled_train_cases": len(selected),
        "condition_names": condition_names or [],
        "points": stats(np.concatenate(points, axis=0)),
        "conditions": stats(np.concatenate(conditions, axis=0)),
        "temperature": stats(temp_all),
        "temperature_offset_ratio": float(abs(temp_all.mean()) / max(float(temp_all.std()), 1e-8)),
    }


def run_summary(run_dir: Path) -> dict[str, Any]:
    if not run_dir.exists():
        return {"path": str(run_dir), "exists": False}
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    config = read_json(config_path) if config_path.exists() else {}
    history_payload = read_json(history_path) if history_path.exists() else {}
    history = history_payload.get("history") if isinstance(history_payload, dict) else []
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    best = metrics.get("best") if isinstance(metrics, dict) else None
    if not isinstance(best, dict) and history:
        best = min(history, key=lambda row: float(row.get("relative_l2_mean", math.inf)))
    return {
        "path": str(run_dir),
        "exists": True,
        "config": config,
        "history_len": len(history) if isinstance(history, list) else 0,
        "first": history[0] if isinstance(history, list) and history else None,
        "final": history[-1] if isinstance(history, list) and history else None,
        "best": best,
    }


def pretrain_summary(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    config = read_json(config_path) if config_path.exists() else {}
    history_payload = read_json(history_path) if history_path.exists() else {}
    history = history_payload.get("history") if isinstance(history_payload, dict) else []
    best = min(history, key=lambda row: float(row.get("train_mse", math.inf))) if history else None
    return {
        "path": str(run_dir),
        "config": config,
        "history_len": len(history) if isinstance(history, list) else 0,
        "first": history[0] if isinstance(history, list) and history else None,
        "final": history[-1] if isinstance(history, list) and history else None,
        "best_train": best,
        "has_validation": any("val" in str(key).lower() for row in history for key in row) if isinstance(history, list) else False,
        "has_normalization": "normalization" in config or "normalizers" in config,
    }


def ratio_issue(left_std: list[float], right_std: list[float], *, low: float, high: float) -> bool:
    if not left_std or not right_std:
        return False
    for left, right in zip(left_std, right_std):
        denom = max(abs(float(right)), 1e-12)
        ratio = abs(float(left)) / denom
        if ratio < low or ratio > high:
            return True
    return False


def build_findings(report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    pretrain = report["pretrain_run"]
    downstream = report["downstream_data"]
    pretrain_data = report["pretrain_data"]
    pretrained_run = report["pretrained_run"]
    scratch_run = report["scratch_run"]

    if not pretrain.get("has_validation"):
        findings.append(
            {
                "severity": "critical",
                "title": "Pretraining has no validation/held-out selection",
                "detail": "best_model.pt is selected by training loss, so representation quality and overfitting are not monitored.",
            }
        )
    if not pretrain.get("has_normalization"):
        findings.append(
            {
                "severity": "critical",
                "title": "Pretraining input/target normalization is not recorded",
                "detail": "Downstream fine-tuning normalizes condition channels from downstream train statistics, while pretraining used raw condition channels.",
            }
        )
    if ratio_issue(
        downstream["points"].get("std", []),
        pretrain_data["x"].get("std", []),
        low=0.5,
        high=2.0,
    ):
        findings.append(
            {
                "severity": "critical",
                "title": "Coordinate scales between pretraining and downstream are incompatible",
                "detail": "The loaded positional/geometry representation is reused across domains with substantially different coordinate ranges.",
            }
        )
    if ratio_issue(
        downstream["conditions"].get("std", []),
        pretrain_data["cond"].get("std", []),
        low=0.5,
        high=2.0,
    ):
        findings.append(
            {
                "severity": "critical",
                "title": "Condition-channel distributions do not match",
                "detail": "Even when condition names match, train-set downstream standardization and source/sink feature semantics change the input distribution.",
            }
        )
    if downstream.get("temperature_offset_ratio", 0.0) > 5.0:
        findings.append(
            {
                "severity": "major",
                "title": "Relative L2 on absolute Kelvin temperature is too forgiving",
                "detail": "The denominator is dominated by the 300K offset. Report centered or nondimensional temperature errors as primary metrics.",
            }
        )
    final_loss = (pretrain.get("final") or {}).get("train_mse")
    best_loss = (pretrain.get("best_train") or {}).get("train_mse")
    if isinstance(final_loss, (int, float)) and isinstance(best_loss, (int, float)) and final_loss > best_loss * 1.05:
        findings.append(
            {
                "severity": "major",
                "title": "Pretraining does not improve monotonically and degrades after best epoch",
                "detail": f"Final train loss {final_loss:.4g} is worse than best train loss {best_loss:.4g}; no validation signal exists to interpret this.",
            }
        )
    scratch_best = (scratch_run.get("best") or {}).get("relative_l2_mean")
    pretrained_best = (pretrained_run.get("best") or {}).get("relative_l2_mean")
    if isinstance(scratch_best, (int, float)) and isinstance(pretrained_best, (int, float)) and pretrained_best > scratch_best * 1.2:
        findings.append(
            {
                "severity": "major",
                "title": "Current pretrained initialization is a harmful prior on M3",
                "detail": f"Representative best validation relL2: scratch={scratch_best:.4g}, pretrained={pretrained_best:.4g}.",
            }
        )
    overlaps = report["split"].get("overlaps", {})
    if any(value for value in overlaps.values()):
        findings.append(
            {
                "severity": "critical",
                "title": "Split leakage detected",
                "detail": f"Overlap counts: {overlaps}",
            }
        )
    return findings


def markdown_report(report: dict[str, Any]) -> str:
    pretrain_data = report["pretrain_data"]
    downstream = report["downstream_data"]
    pretrain_run = report["pretrain_run"]
    scratch = report["scratch_run"]
    pretrained = report["pretrained_run"]
    findings = report["findings"]

    def metric(run: dict[str, Any], key: str) -> str:
        value = (run.get("best") or {}).get(key)
        return "n/a" if value is None else f"{float(value):.6g}"

    lines = [
        "# ML Training Setup Audit 2026-05-07",
        "",
        "## Verdict",
        "",
        "The current P2-to-M3 result is not a valid efficacy test of Thermal GeoPT pretraining.",
        "The downstream training loop can optimize scratch models, but the pretraining and fine-tuning setup violates basic transfer-learning assumptions.",
        "",
        "## Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- **{finding['severity'].upper()}**: {finding['title']}. {finding['detail']}")
    lines.extend(
        [
            "",
            "## Data Interface",
            "",
            f"- Pretrain condition names: `{pretrain_data['condition_names']}`",
            f"- Downstream condition names: `{downstream['condition_names']}`",
            f"- Pretrain coordinate mean/std: `{compact_vector(pretrain_data['x']['mean'])}` / `{compact_vector(pretrain_data['x']['std'])}`",
            f"- Downstream coordinate mean/std: `{compact_vector(downstream['points']['mean'])}` / `{compact_vector(downstream['points']['std'])}`",
            f"- Pretrain condition mean/std: `{compact_vector(pretrain_data['cond']['mean'])}` / `{compact_vector(pretrain_data['cond']['std'])}`",
            f"- Downstream condition mean/std: `{compact_vector(downstream['conditions']['mean'])}` / `{compact_vector(downstream['conditions']['std'])}`",
            "",
            "## Pretraining Run",
            "",
            f"- Epochs logged: `{pretrain_run['history_len']}`",
            f"- Has validation metrics: `{pretrain_run['has_validation']}`",
            f"- Has normalization metadata: `{pretrain_run['has_normalization']}`",
            f"- First train loss: `{(pretrain_run.get('first') or {}).get('train_mse')}`",
            f"- Best train loss: `{(pretrain_run.get('best_train') or {}).get('train_mse')}` at epoch `{(pretrain_run.get('best_train') or {}).get('epoch')}`",
            f"- Final train loss: `{(pretrain_run.get('final') or {}).get('train_mse')}`",
            "",
            "## Representative Downstream Runs",
            "",
            "| run | best val relL2 | best val RMSE | final train MSE |",
            "|---|---:|---:|---:|",
            f"| scratch | {metric(scratch, 'relative_l2_mean')} | {metric(scratch, 'rmse_mean')} | {(scratch.get('final') or {}).get('train_mse')} |",
            f"| pretrained | {metric(pretrained, 'relative_l2_mean')} | {metric(pretrained, 'rmse_mean')} | {(pretrained.get('final') or {}).get('train_mse')} |",
            "",
            "## Required Standard Before More Heavy Runs",
            "",
            "1. Record and reuse one explicit pretraining input/target normalization contract.",
            "2. Apply the same coordinate convention to pretraining and downstream.",
            "3. Add pretraining validation shards and select checkpoints by validation loss or transfer-proxy validation.",
            "4. Standardize heterogeneous pretraining targets per channel or per target group before weighting losses.",
            "5. Make nondimensional or centered thermal errors primary; keep absolute-Kelvin relative L2 only as a secondary metric.",
            "6. Treat current block D1 as pipeline smoke only, not as the main GeoPT transfer benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "pretrain_data": collect_pretrain_stats(
            args.pretrain_manifest,
            max_shards=args.max_pretrain_shards,
            max_points=args.max_pretrain_points,
        ),
        "downstream_data": collect_downstream_stats(
            args.downstream_manifest,
            args.split_path,
            max_cases=args.max_downstream_cases,
        ),
        "split": split_overlap_report(args.split_path),
        "pretrain_run": pretrain_summary(args.pretrain_run),
        "scratch_run": run_summary(args.scratch_run),
        "pretrained_run": run_summary(args.pretrained_run),
    }
    report["findings"] = build_findings(report)
    write_json(args.output_json, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "findings": len(report["findings"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
