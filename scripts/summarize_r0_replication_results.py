#!/usr/bin/env python3
"""Summarize multi-seed R0 label-scarcity replication results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.training import write_json

DEFAULT_GROUPS = ["scratch", "no_boundary_field"]
DEFAULT_TRAIN_SIZES = [50, 75, 100, 125]
DEFAULT_SPLIT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TRAIN_SEEDS = [42]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument(
        "--eval-pattern",
        default=(
            "outputs/logs/d1_r0_v2_{group}_split{split_seed}_trainseed{train_seed}_"
            "train{train_size}_test_eval.json"
        ),
        help="Pattern with {group}, {split_seed}, {train_seed}, and {train_size} placeholders.",
    )
    parser.add_argument(
        "--run-pattern",
        default=(
            "outputs/checkpoints/d1_r0_v2_{group}_split{split_seed}_trainseed{train_seed}_"
            "train{train_size}_ep20"
        ),
        help="Run directory pattern with {group}, {split_seed}, {train_seed}, and {train_size} placeholders.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--expected-case-count", type=int, default=150)
    parser.add_argument("--expected-point-budget", type=int, default=4096)
    parser.add_argument("--title", default="R0 Replication Results")
    parser.add_argument("--description", default="R0 multi-seed label-scarcity replication summary.")
    parser.add_argument(
        "--interpretation-rule",
        default=(
            "Treat the 100-label no-boundary signal as replicated only if the paired run-level improvement "
            "is consistently positive, preferably above 10%, across multiple split seeds and remains visible "
            "at 125 labels.\n\n"
            "If the improvement appears only at one split seed or collapses at 75/125 labels, keep the original "
            "gate negative and move the emphasis to R1 dynamics-lifted pretraining redesign."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/logs/r0_replication_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/r0_replication_results.md"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_path(pattern: str, *, group: str, split_seed: int, train_seed: int, train_size: int) -> Path:
    return Path(
        pattern.format(group=group, split_seed=split_seed, train_seed=train_seed, train_size=train_size)
    )


def run_path(pattern: str, *, group: str, split_seed: int, train_seed: int, train_size: int) -> Path:
    return Path(
        pattern.format(group=group, split_seed=split_seed, train_seed=train_seed, train_size=train_size)
    )


def stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "min": None, "max": None}
    mean = float(sum(values) / n)
    if n == 1:
        return {"n": n, "mean": mean, "std": 0.0, "ci95": 0.0, "min": values[0], "max": values[0]}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(variance)
    ci95 = 1.96 * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": float(std),
        "ci95": float(ci95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def pct_improvement(reference: Any, candidate: Any) -> float | None:
    if not isinstance(reference, (int, float)) or not isinstance(candidate, (int, float)):
        return None
    if float(reference) == 0.0:
        return None
    return 100.0 * (float(reference) - float(candidate)) / float(reference)


def case_metric_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("case_metrics")
    if not isinstance(rows, list):
        return {}
    result = {}
    for row in rows:
        if isinstance(row, dict) and row.get("id") is not None:
            result[str(row["id"])] = row
    return result


def metric_row(
    path: Path,
    *,
    allow_missing: bool,
    expected_case_count: int,
    expected_point_budget: int,
) -> dict[str, Any]:
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
        "case_metrics": payload.get("case_metrics", []),
    }


def run_metadata(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    metrics_path = path / "metrics.json"
    if not config_path.exists():
        return {"path": str(path), "exists": False}
    config = read_json(config_path)
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    return {
        "path": str(path),
        "exists": True,
        "metrics": metrics,
        "split_path": config.get("split_path") if isinstance(config, dict) else None,
        "train_split": config.get("train_split") if isinstance(config, dict) else None,
        "seed": config.get("seed") if isinstance(config, dict) else None,
        "pretrained_model_dir": config.get("pretrained_model_dir") if isinstance(config, dict) else None,
        "pretrained_load": config.get("pretrained_load") if isinstance(config, dict) else None,
    }


def integrity_warnings(
    args: argparse.Namespace,
    per_run: dict[str, dict[str, dict[str, dict[str, Any]]]],
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[str]:
    warnings: set[str] = set()
    for train_size in args.train_sizes:
        size_key = str(train_size)
        for group in args.groups:
            for train_seed in args.train_seeds:
                split_paths = []
                rel_l2_values = []
                for split_seed in args.split_seeds:
                    run_key = f"split{split_seed}_trainseed{train_seed}"
                    metadata = runs.get(size_key, {}).get(group, {}).get(run_key, {})
                    split_path = metadata.get("split_path")
                    if isinstance(split_path, str) and split_path:
                        split_paths.append(split_path)
                        if "{" in split_path or "}" in split_path:
                            warnings.add(
                                f"Unresolved placeholder in split_path for {group} train_size={train_size}: {split_path}"
                            )
                    row = per_run.get(size_key, {}).get(group, {}).get(run_key, {})
                    rel_l2 = row.get("relative_l2_mean")
                    if isinstance(rel_l2, (int, float)):
                        rel_l2_values.append(float(rel_l2))

                if len(args.split_seeds) > 1 and split_paths and len(set(split_paths)) == 1:
                    warnings.add(
                        "All split seeds used the same split_path for "
                        f"group={group}, train_size={train_size}, train_seed={train_seed}: {split_paths[0]}"
                    )
                if len(rel_l2_values) > 1 and len({f"{value:.12g}" for value in rel_l2_values}) == 1:
                    warnings.add(
                        "Relative L2 is identical across split seeds for "
                        f"group={group}, train_size={train_size}, train_seed={train_seed}; "
                        "treat run-level CI as non-informative until split independence is verified."
                    )
    return sorted(warnings)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    per_run: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}

    for train_size in args.train_sizes:
        size_key = str(train_size)
        per_run[size_key] = {}
        runs[size_key] = {}
        aggregate[size_key] = {}
        for group in args.groups:
            per_run[size_key][group] = {}
            runs[size_key][group] = {}
            rel_l2_values = []
            for split_seed in args.split_seeds:
                for train_seed in args.train_seeds:
                    run_key = f"split{split_seed}_trainseed{train_seed}"
                    row = metric_row(
                        eval_path(
                            args.eval_pattern,
                            group=group,
                            split_seed=split_seed,
                            train_seed=train_seed,
                            train_size=train_size,
                        ),
                        allow_missing=args.allow_missing,
                        expected_case_count=args.expected_case_count,
                        expected_point_budget=args.expected_point_budget,
                    )
                    per_run[size_key][group][run_key] = row
                    runs[size_key][group][run_key] = run_metadata(
                        run_path(
                            args.run_pattern,
                            group=group,
                            split_seed=split_seed,
                            train_seed=train_seed,
                            train_size=train_size,
                        )
                    )
                    rel_l2 = row.get("relative_l2_mean")
                    if isinstance(rel_l2, (int, float)):
                        rel_l2_values.append(float(rel_l2))
            aggregate[size_key][group] = {
                "relative_l2_mean": stats(rel_l2_values),
                "rmse_mean": stats(
                    [
                        float(row["rmse_mean"])
                        for row in per_run[size_key][group].values()
                        if isinstance(row.get("rmse_mean"), (int, float))
                    ]
                ),
            }

        scratch_runs = per_run[size_key].get("scratch", {})
        comparisons[size_key] = {}
        for group in args.groups:
            if group == "scratch":
                continue
            improvements = []
            case_improvements = []
            case_wins = 0
            case_pairs = 0
            for run_key, candidate in per_run[size_key].get(group, {}).items():
                scratch = scratch_runs.get(run_key, {})
                improvement = pct_improvement(scratch.get("relative_l2_mean"), candidate.get("relative_l2_mean"))
                if improvement is not None:
                    improvements.append(float(improvement))

                scratch_cases = case_metric_by_id(scratch)
                candidate_cases = case_metric_by_id(candidate)
                for case_id in sorted(set(scratch_cases).intersection(candidate_cases)):
                    scratch_rel = scratch_cases[case_id].get("relative_l2")
                    candidate_rel = candidate_cases[case_id].get("relative_l2")
                    case_improvement = pct_improvement(scratch_rel, candidate_rel)
                    if case_improvement is None:
                        continue
                    case_improvements.append(float(case_improvement))
                    case_pairs += 1
                    if float(candidate_rel) < float(scratch_rel):
                        case_wins += 1

            comparisons[size_key][f"{group}_vs_scratch"] = {
                "relative_l2_improvement_pct": stats(improvements),
                "case_relative_l2_improvement_pct": stats(case_improvements),
                "case_win_rate": (case_wins / case_pairs) if case_pairs else None,
                "case_pairs": case_pairs,
            }

    return {
        "title": args.title,
        "description": args.description,
        "interpretation_rule": args.interpretation_rule,
        "eval_pattern": args.eval_pattern,
        "run_pattern": args.run_pattern,
        "train_sizes": args.train_sizes,
        "groups": args.groups,
        "split_seeds": args.split_seeds,
        "train_seeds": args.train_seeds,
        "per_run": per_run,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "integrity_warnings": integrity_warnings(args, per_run, runs),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def stat_cell(payload: dict[str, Any], key: str = "relative_l2_mean") -> str:
    stat = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(stat, dict) or stat.get("mean") is None:
        return ""
    return f"{float(stat['mean']):.6f} +/- {float(stat.get('ci95') or 0.0):.6f} (n={stat.get('n')})"


def markdown_report(summary: dict[str, Any]) -> str:
    groups = list(summary["groups"])
    lines = [
        f"# {summary.get('title') or 'D1 Transfer Results'}",
        "",
        str(summary.get("description") or "This report summarizes D1 label-scarcity transfer runs."),
        "",
        f"- split seeds: `{summary['split_seeds']}`",
        f"- train seeds: `{summary['train_seeds']}`",
        f"- groups: `{groups}`",
        "",
    ]
    warnings = summary.get("integrity_warnings") or []
    if warnings:
        lines.extend(["## Integrity Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Test Relative L2 Across Runs",
            "",
            "| train_size | " + " | ".join(groups) + " |",
            "|---:|" + "|".join(["---:"] * len(groups)) + "|",
        ]
    )
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        row = [str(train_size)]
        for group in groups:
            row.append(stat_cell(summary["aggregate"][size_key].get(group, {})))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Paired Improvement Vs Scratch",
            "",
            "| train_size | group | run_improvement_pct | case_improvement_pct | case_win_rate |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for train_size in summary["train_sizes"]:
        size_key = str(train_size)
        for key, payload in summary["comparisons"][size_key].items():
            group = key.removesuffix("_vs_scratch")
            run_stat = payload.get("relative_l2_improvement_pct", {})
            case_stat = payload.get("case_relative_l2_improvement_pct", {})
            run_cell = ""
            if run_stat.get("mean") is not None:
                run_cell = f"{float(run_stat['mean']):.2f} +/- {float(run_stat.get('ci95') or 0.0):.2f} (n={run_stat.get('n')})"
            case_cell = ""
            if case_stat.get("mean") is not None:
                case_cell = f"{float(case_stat['mean']):.2f} +/- {float(case_stat.get('ci95') or 0.0):.2f}"
            lines.append(
                "| {train_size} | {group} | {run_cell} | {case_cell} | {win_rate} |".format(
                    train_size=train_size,
                    group=group,
                    run_cell=run_cell,
                    case_cell=case_cell,
                    win_rate=fmt(payload.get("case_win_rate"), digits=3),
                )
            )

    interpretation = str(summary.get("interpretation_rule") or "").strip()
    if interpretation:
        lines.extend(["", "## Interpretation Rule", "", interpretation, ""])
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
