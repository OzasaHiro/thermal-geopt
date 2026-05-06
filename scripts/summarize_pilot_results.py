#!/usr/bin/env python3
"""Summarize Thermal GeoPT pilot training/evaluation outputs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-run", type=Path, default=Path("outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2"))
    parser.add_argument("--scratch-run", type=Path, default=Path("outputs/checkpoints/d1_scratch_pilot_300_c5_n8192_ep3"))
    parser.add_argument("--pretrained-run", type=Path, default=Path("outputs/checkpoints/d1_pretrained_pilot_300_c5_n8192_ep3"))
    parser.add_argument("--baseline-eval", type=Path, default=Path("outputs/logs/d1_baseline_pilot_test_eval.json"))
    parser.add_argument("--scratch-eval", type=Path, default=Path("outputs/logs/d1_scratch_pilot_test_eval.json"))
    parser.add_argument("--pretrained-eval", type=Path, default=Path("outputs/logs/d1_pretrained_pilot_test_eval.json"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/logs/pilot_result_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/pilot_results_2026-05-06.md"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"Missing expected file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(run_dir: Path) -> dict[str, Any]:
    return {
        "config": read_json(run_dir / "config.json"),
        "metrics": read_json(run_dir / "metrics.json"),
        "history": read_json(run_dir / "history.json").get("history", []),
    }


def eval_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "path": str(path),
        "mode": payload.get("mode"),
        "split": payload.get("split"),
        "case_count": payload.get("case_count"),
        "total_points": payload.get("total_points"),
        "temperature": payload.get("temperature", {}),
    }


def pct_delta(reference: float, candidate: float) -> float:
    if reference == 0.0:
        return 0.0
    return 100.0 * (reference - candidate) / reference


def fmt_float(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return ""


def history_table(history: list[dict[str, Any]]) -> str:
    lines = ["| epoch | train_mse | val_relative_l2 | val_rmse | elapsed_sec |", "|---:|---:|---:|---:|---:|"]
    for item in history:
        lines.append(
            "| {epoch} | {train_mse} | {rel} | {rmse} | {elapsed} |".format(
                epoch=item.get("epoch", ""),
                train_mse=fmt_float(item.get("train_mse")),
                rel=fmt_float(item.get("relative_l2_mean")),
                rmse=fmt_float(item.get("rmse_mean")),
                elapsed=fmt_float(item.get("elapsed_sec"), digits=1),
            )
        )
    return "\n".join(lines)


def markdown_report(summary: dict[str, Any]) -> str:
    baseline = summary["test_eval"]["baseline"]["temperature"]
    scratch = summary["test_eval"]["scratch"]["temperature"]
    pretrained = summary["test_eval"]["pretrained"]["temperature"]
    scratch_gain = summary["comparisons"]["scratch_vs_baseline_relative_l2_pct"]
    pretrained_gain = summary["comparisons"]["pretrained_vs_baseline_relative_l2_pct"]
    pretrained_vs_scratch = summary["comparisons"]["pretrained_vs_scratch_relative_l2_pct"]
    pretrain_metrics = summary["runs"]["pretrain"]["metrics"]
    pretrained_load = summary["runs"]["pretrained"]["config"].get("pretrained_load") or {}

    return f"""# Pilot Results 2026-05-06

## Scope

- Pretraining: `cadquery_pilot_300_e20_n4096`, 2 epochs.
- Downstream: `d1_proxy_pilot_300_c5_n8192`, train/val/test = 1200/150/150 cases.
- Evaluation: test split, 4096 points/case, 150 cases.
- Data and checkpoints are generated artifacts and remain outside Git.

## Result Summary

| run | test_relative_l2 | test_rmse | test_max_value_error |
|---|---:|---:|---:|
| baseline_mean_temperature | {baseline["relative_l2_mean"]:.6f} | {baseline["rmse_mean"]:.6f} | {baseline["max_value_error_mean"]:.6f} |
| d1_scratch_ep3 | {scratch["relative_l2_mean"]:.6f} | {scratch["rmse_mean"]:.6f} | {scratch["max_value_error_mean"]:.6f} |
| d1_pretrained_ep3 | {pretrained["relative_l2_mean"]:.6f} | {pretrained["rmse_mean"]:.6f} | {pretrained["max_value_error_mean"]:.6f} |

Relative L2 deltas:

- Scratch vs baseline: {scratch_gain:.2f}% improvement.
- Pretrained vs baseline: {pretrained_gain:.2f}% improvement.
- Pretrained vs scratch: {pretrained_vs_scratch:.2f}% improvement. Negative means pretrained is worse than scratch.

## Pretraining

Final pretraining train MSE: `{pretrain_metrics["train_mse"]:.6f}`.

{history_table(summary["runs"]["pretrain"]["history"])}

## D1 Scratch

{history_table(summary["runs"]["scratch"]["history"])}

## D1 Pretrained

{history_table(summary["runs"]["pretrained"]["history"])}

Pretrained load report:

- loaded tensors: {pretrained_load.get("loaded_tensors")}
- skipped tensors: {pretrained_load.get("skipped_tensors")}
- missing tensors: {pretrained_load.get("missing_tensors")}
- checkpoint: `{pretrained_load.get("checkpoint_path")}`

## Interpretation

The pilot confirms the training and evaluation pipeline works, and the scratch Transolver improves over the mean-temperature baseline. The current TDF pretraining does not help this D1 proxy after 3 fine-tuning epochs; it underperforms scratch on both validation and test relative L2.

Most likely reasons:

- The D1 proxy target is simple and directly encoded by source/sink condition features, so scratch learns it quickly.
- The pretraining target has 14 TDF/diffusion channels, while D1 predicts 1 temperature channel; shape-incompatible input/output projection tensors are intentionally skipped.
- Only 2 pretraining epochs and 3 fine-tuning epochs were run, so this is a pipeline pilot, not a final pretraining efficacy result.

## Next Runs

1. Extend D1 fine-tuning to 10-20 epochs for scratch and pretrained with the same split.
2. Add a label-scarcity matrix, for example 50/100/300 train cases, because pretraining should matter most when labels are limited.
3. Move from D1 proxy to a harder D1/D2 target before judging Thermal GeoPT value.
4. For full pretraining candidates, increase geometry diversity and pretraining epochs only after the label-scarcity comparison is in place.
"""


def main() -> int:
    args = parse_args()
    baseline_eval = eval_summary(args.baseline_eval)
    scratch_eval = eval_summary(args.scratch_eval)
    pretrained_eval = eval_summary(args.pretrained_eval)
    baseline_rel = float(baseline_eval["temperature"]["relative_l2_mean"])
    scratch_rel = float(scratch_eval["temperature"]["relative_l2_mean"])
    pretrained_rel = float(pretrained_eval["temperature"]["relative_l2_mean"])

    summary = {
        "runs": {
            "pretrain": run_metrics(args.pretrain_run),
            "scratch": run_metrics(args.scratch_run),
            "pretrained": run_metrics(args.pretrained_run),
        },
        "test_eval": {
            "baseline": baseline_eval,
            "scratch": scratch_eval,
            "pretrained": pretrained_eval,
        },
        "comparisons": {
            "scratch_vs_baseline_relative_l2_pct": pct_delta(baseline_rel, scratch_rel),
            "pretrained_vs_baseline_relative_l2_pct": pct_delta(baseline_rel, pretrained_rel),
            "pretrained_vs_scratch_relative_l2_pct": pct_delta(scratch_rel, pretrained_rel),
        },
    }
    write_json(args.output_json, summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
