#!/usr/bin/env python3
"""Summarize label-scarcity gate evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.training import write_json

DEFAULT_GROUPS = ["scratch", "full", "static_tdf_only", "no_boundary_field"]
DEFAULT_TRAIN_SIZES = [10, 25, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument(
        "--eval-pattern",
        default="outputs/logs/d1_gate_{group}_train{train_size}_test_eval.json",
        help="Pattern with {group} and {train_size} placeholders.",
    )
    parser.add_argument(
        "--run-pattern",
        default="outputs/checkpoints/d1_gate_{group}_train{train_size}_ep20",
        help="Run directory pattern with {group} and {train_size} placeholders.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Write an incomplete summary instead of failing.")
    parser.add_argument("--expected-case-count", type=int, default=150)
    parser.add_argument("--expected-point-budget", type=int, default=4096)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/logs/label_scarcity_gate_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/label_scarcity_gate_results.md"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_path(pattern: str, *, group: str, train_size: int) -> Path:
    return Path(pattern.format(group=group, train_size=train_size))


def run_path(pattern: str, *, group: str, train_size: int) -> Path:
    return Path(pattern.format(group=group, train_size=train_size))


def metric_row(path: Path, *, allow_missing: bool, expected_case_count: int, expected_point_budget: int) -> dict[str, Any]:
    if not path.exists():
        if not allow_missing:
            raise SystemExit(f"Missing expected evaluation JSON: {path}")
        return {"path": str(path), "exists": False}
    payload = read_json(path)
    temperature = payload.get("temperature") or {}
    case_count = payload.get("case_count")
    point_budget = payload.get("point_budget")
    if expected_case_count > 0 and case_count != expected_case_count:
        raise SystemExit(f"{path} has case_count={case_count}; expected {expected_case_count}")
    if expected_point_budget > 0 and point_budget != expected_point_budget:
        raise SystemExit(f"{path} has point_budget={point_budget}; expected {expected_point_budget}")
    return {
        "path": str(path),
        "exists": True,
        "split": payload.get("split"),
        "point_budget": point_budget,
        "case_count": case_count,
        "total_points": payload.get("total_points"),
        "relative_l2_mean": temperature.get("relative_l2_mean"),
        "rmse_mean": temperature.get("rmse_mean"),
        "max_value_error_mean": temperature.get("max_value_error_mean"),
    }


def run_metadata(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    metrics_path = path / "metrics.json"
    if not config_path.exists():
        return {"path": str(path), "exists": False}
    config = read_json(config_path)
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    pretrained_load = config.get("pretrained_load") if isinstance(config, dict) else None
    model_config = config.get("model") if isinstance(config, dict) else {}
    return {
        "path": str(path),
        "exists": True,
        "metrics": metrics,
        "pretrained_model_dir": config.get("pretrained_model_dir") if isinstance(config, dict) else None,
        "pretrained_load": pretrained_load,
        "fun_dim": model_config.get("fun_dim") if isinstance(model_config, dict) else None,
        "out_dim": model_config.get("out_dim") if isinstance(model_config, dict) else None,
    }


def pct_improvement(reference: Any, candidate: Any) -> float | None:
    if not isinstance(reference, (int, float)) or not isinstance(candidate, (int, float)):
        return None
    if float(reference) == 0.0:
        return None
    return 100.0 * (float(reference) - float(candidate)) / float(reference)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for train_size in args.train_sizes:
        size_key = str(train_size)
        results[size_key] = {}
        runs[size_key] = {}
        for group in args.groups:
            results[size_key][group] = metric_row(
                eval_path(args.eval_pattern, group=group, train_size=train_size),
                allow_missing=args.allow_missing,
                expected_case_count=args.expected_case_count,
                expected_point_budget=args.expected_point_budget,
            )
            runs[size_key][group] = run_metadata(run_path(args.run_pattern, group=group, train_size=train_size))

        scratch = results[size_key].get("scratch", {})
        scratch_rel = scratch.get("relative_l2_mean")
        comparisons[size_key] = {}
        for group in args.groups:
            if group == "scratch":
                continue
            candidate_rel = results[size_key].get(group, {}).get("relative_l2_mean")
            comparisons[size_key][f"{group}_vs_scratch_relative_l2_pct"] = pct_improvement(
                scratch_rel,
                candidate_rel,
            )

    return {
        "description": "Label-scarcity Thermal GeoPT gate summary.",
        "eval_pattern": args.eval_pattern,
        "run_pattern": args.run_pattern,
        "train_sizes": args.train_sizes,
        "groups": args.groups,
        "results": results,
        "runs": runs,
        "comparisons": comparisons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_report(summary: dict[str, Any]) -> str:
    groups = list(summary["groups"])
    gate_hits = []
    best_entry = None
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        for group, result in summary["results"][size_key].items():
            rel_l2 = result.get("relative_l2_mean")
            if isinstance(rel_l2, (int, float)):
                entry = (float(rel_l2), train_size, group)
                best_entry = entry if best_entry is None or entry < best_entry else best_entry
        if train_size not in (25, 50):
            continue
        for key, value in summary["comparisons"][size_key].items():
            if isinstance(value, (int, float)) and value >= 10.0:
                gate_hits.append((train_size, key.removesuffix("_vs_scratch_relative_l2_pct"), value))
    lines = [
        "# Label-Scarcity Gate Results",
        "",
        "This report is generated from evaluation JSON files. Missing runs are left blank.",
        "",
        "## Test Relative L2",
        "",
        "| train_size | " + " | ".join(groups) + " |",
        "|---:|" + "|".join(["---:"] * len(groups)) + "|",
    ]
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        row = [str(train_size)]
        for group in groups:
            row.append(fmt(summary["results"][size_key].get(group, {}).get("relative_l2_mean")))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Improvement vs Scratch", "", "| train_size | group | relative_l2_improvement_pct |", "|---:|---|---:|"])
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        for key, value in summary["comparisons"][size_key].items():
            group = key.removesuffix("_vs_scratch_relative_l2_pct")
            lines.append(f"| {train_size} | {group} | {fmt(value)} |")

    lines.extend(["", "## Gate Interpretation", ""])
    if gate_hits:
        lines.append("Gate status: positive under the predefined 25/50-label rule.")
        lines.append("")
        for train_size, group, value in gate_hits:
            lines.append(f"- `{group}` improves scratch by {value:.2f}% at {train_size} labels.")
    else:
        lines.append("Gate status: not positive under the predefined 25/50-label rule.")
        lines.append("")
        lines.append("No pretrained group improves scratch by 10% relative L2 at 25 or 50 labels.")
    if best_entry is not None:
        best_rel_l2, best_train_size, best_group = best_entry
        lines.append("")
        lines.append(f"Best observed test relative L2 is `{best_rel_l2:.6f}` from `{best_group}` at {best_train_size} labels.")

    lines.extend(
        [
            "",
            "## Pretrained Load",
            "",
            "| train_size | group | loaded_tensors | skipped_tensors | missing_tensors | source |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        for group in groups:
            run = summary["runs"][size_key].get(group, {})
            load = run.get("pretrained_load") or {}
            if not load:
                continue
            lines.append(
                "| {train_size} | {group} | {loaded} | {skipped} | {missing} | `{source}` |".format(
                    train_size=train_size,
                    group=group,
                    loaded=fmt(load.get("loaded_tensors")),
                    skipped=fmt(load.get("skipped_tensors")),
                    missing=fmt(load.get("missing_tensors")),
                    source=load.get("checkpoint_path"),
                )
            )

    lines.extend(
        [
            "",
            "## Gate Rule",
            "",
            "A positive gate means Thermal GeoPT improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches the same error with materially fewer downstream epochs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    write_json(args.output_json, summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
