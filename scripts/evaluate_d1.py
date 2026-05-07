#!/usr/bin/env python3
"""Evaluate D1 NPZ cases with a Transolver checkpoint or tiny baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.metrics import (
    centered_relative_l2,
    hotspot_abs_error,
    max_temperature_abs_error,
    max_value_error,
    normalized_rmse_range,
    relative_l2,
)

ITEM_LIST_KEYS = ("records", "cases", "samples", "items", "shards")
ID_KEYS = ("sample", "case", "case_id", "id", "name")
PATH_KEYS = ("path", "output_path", "npz_path", "case_path", "processed_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--split", default="all")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint-file", default="best_model.pt")
    parser.add_argument("--baseline", choices=("mean_temperature",))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--point-budget", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--geopt-vendor", type=Path, default=Path("../GeoPT/vendor/GeoPT"))
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def manifest_items(manifest: Any) -> list[Any]:
    if isinstance(manifest, list):
        return manifest
    if not isinstance(manifest, dict):
        raise SystemExit("Case manifest must be a JSON object or list.")
    for key in ITEM_LIST_KEYS:
        value = manifest.get(key)
        if isinstance(value, list):
            return value
    raise SystemExit(f"Case manifest does not contain one of: {', '.join(ITEM_LIST_KEYS)}")


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
            return Path(str(value)).name
    raise SystemExit(f"Manifest item has no usable id or path field: {item}")


def item_path(item: Any, manifest_path: Path) -> Path:
    if isinstance(item, str):
        raw_path = Path(item)
    elif isinstance(item, dict):
        for key in PATH_KEYS:
            value = item.get(key)
            if value not in (None, ""):
                raw_path = Path(str(value))
                break
        else:
            case_name = item.get("case") or item.get("sample") or item.get("id")
            if case_name in (None, ""):
                raise SystemExit(f"Manifest item has no usable case path: {item}")
            raw_path = Path(str(case_name))
    else:
        raw_path = Path(str(item))

    if raw_path.is_absolute():
        return raw_path

    cwd_path = (Path.cwd() / raw_path).resolve()
    if cwd_path.exists():
        return cwd_path
    manifest_relative = (manifest_path.parent / raw_path).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return cwd_path


def split_ids(split_path: Path, split: str) -> set[str]:
    payload = read_json(split_path)
    if not isinstance(payload, dict):
        raise SystemExit("Split file must be a JSON object.")
    ids = payload.get(split)
    if not isinstance(ids, list):
        raise SystemExit(f"Split file does not contain a list for split {split!r}.")
    return {str(sample_id) for sample_id in ids}


def selected_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest = read_json(args.case_manifest)
    items = manifest_items(manifest)
    allowed_ids = None
    if args.split != "all":
        if args.split_path is None:
            raise SystemExit("--split-path is required when --split is train, val, or test.")
        allowed_ids = split_ids(args.split_path, args.split)

    cases = []
    for item in items:
        case_id = item_id(item)
        case_path = item_path(item, args.case_manifest)
        aliases = {case_id, case_path.name, case_path.stem}
        if allowed_ids is not None and aliases.isdisjoint(allowed_ids):
            continue
        cases.append({"id": case_id, "path": str(case_path)})

    if not cases:
        raise SystemExit("No cases selected for evaluation.")
    return cases


def resolve_device(device_arg: str) -> Any:
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    for source_key in ("model", "model_config", "transolver"):
        source = config.get(source_key)
        if isinstance(source, dict) and key in source:
            return source[key]
    return config.get(key, default)


def require_config_int(config: dict[str, Any], key: str) -> int:
    value = config_value(config, key)
    if value is None:
        raise SystemExit(f"model-dir/config.json must define {key!r}.")
    return int(value)


def transolver_args(config: dict[str, Any]) -> SimpleNamespace:
    fun_dim = require_config_int(config, "fun_dim")
    out_dim = require_config_int(config, "out_dim")
    return SimpleNamespace(
        fun_dim=fun_dim,
        space_dim=int(config_value(config, "space_dim", 3)),
        out_dim=out_dim,
        n_hidden=int(config_value(config, "n_hidden", 256)),
        n_layers=int(config_value(config, "n_layers", 8)),
        n_heads=int(config_value(config, "n_heads", 8)),
        mlp_ratio=int(config_value(config, "mlp_ratio", 2)),
        slice_num=int(config_value(config, "slice_num", 32)),
        dropout=float(config_value(config, "dropout", 0.0)),
        checkpoint=int(config_value(config, "checkpoint", 0)),
        geotype=str(config_value(config, "geotype", "unstructured")),
        shapelist=config_value(config, "shapelist", None),
        unified_pos=bool(config_value(config, "unified_pos", False)),
        act=str(config_value(config, "act", "gelu")),
    )


def looks_like_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(isinstance(key, str) for key in value)


def checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if looks_like_state_dict(checkpoint) and any(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            value = checkpoint.get(key)
            if looks_like_state_dict(value):
                return value
    raise SystemExit("Checkpoint does not contain a recognizable model state dict.")


def strip_state_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(state)
    for prefix in ("module.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def load_model(model_dir: Path, checkpoint_file: str, device: Any, geopt_vendor: Path) -> tuple[Any, dict[str, Any], int, int]:
    import torch

    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / checkpoint_file
    if not config_path.exists():
        raise SystemExit(f"Missing model config: {config_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint_path}")

    config = read_json(config_path)
    if not isinstance(config, dict):
        raise SystemExit("model-dir/config.json must be a JSON object.")

    vendor = geopt_vendor.resolve()
    if not vendor.exists():
        raise SystemExit(f"GeoPT vendor path not found: {vendor}")
    sys.path.insert(0, str(vendor))
    try:
        from models.Transolver import Model
    except Exception as exc:
        raise SystemExit(f"Could not import vendor Transolver: {type(exc).__name__}: {exc}") from exc

    model_args = transolver_args(config)
    model = Model(model_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = strip_state_prefixes(checkpoint_state_dict(checkpoint))
    model.load_state_dict(state)
    model.eval()
    return model, config, int(model_args.fun_dim), int(model_args.out_dim)


def array_from_config(config: dict[str, Any], keys: Iterable[str], width: int) -> np.ndarray | None:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            return np.full((width,), float(arr[0]), dtype=np.float32)
        if arr.size == width:
            return arr.astype(np.float32)
    return None


def normalize_features(features: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stats = config.get("normalization") or config.get("normalizers") or config
    if not isinstance(stats, dict):
        return features
    mean = array_from_config(stats, ("feature_mean", "features_mean", "fx_mean", "input_mean"), features.shape[1])
    std = array_from_config(stats, ("feature_std", "features_std", "fx_std", "input_std"), features.shape[1])
    if mean is None or std is None:
        return features
    return (features - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-8)


def normalize_points(points: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stats = config.get("normalization") or config.get("normalizers") or config
    if not isinstance(stats, dict):
        return points
    mean = array_from_config(stats, ("coordinate_mean", "x_mean"), points.shape[1])
    std = array_from_config(stats, ("coordinate_std", "x_std"), points.shape[1])
    if mean is None or std is None:
        return points
    return (points - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-8)


def denormalize_temperature(pred: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    stats = config.get("normalization") or config.get("normalizers") or config
    if not isinstance(stats, dict):
        return pred
    mean = array_from_config(stats, ("temperature_mean", "target_mean", "output_mean", "y_mean"), 1)
    std = array_from_config(stats, ("temperature_std", "target_std", "output_std", "y_std"), 1)
    if mean is None or std is None:
        return pred
    return pred * max(float(std[0]), 1e-8) + float(mean[0])


def budget_indices(count: int, point_budget: int | None) -> np.ndarray:
    if point_budget is None or point_budget <= 0 or point_budget >= count:
        return np.arange(count)
    return np.linspace(0, count - 1, point_budget, dtype=np.int64)


def load_case(path: Path, point_budget: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Case NPZ not found: {path}")
    with np.load(path) as data:
        for key in ("points", "conditions", "temperature"):
            if key not in data:
                raise SystemExit(f"{path} is missing required array {key!r}.")
        points = np.asarray(data["points"], dtype=np.float32)
        conditions = np.asarray(data["conditions"], dtype=np.float32)
        temperature = np.asarray(data["temperature"], dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise SystemExit(f"{path} has invalid points shape {points.shape}; expected (N, 3).")
    if conditions.ndim != 2 or conditions.shape[0] != points.shape[0]:
        raise SystemExit(f"{path} has invalid conditions shape {conditions.shape}.")
    if temperature.ndim == 1:
        temperature = temperature[:, None]
    if temperature.ndim != 2 or temperature.shape[0] != points.shape[0]:
        raise SystemExit(f"{path} has invalid temperature shape {temperature.shape}.")

    index = budget_indices(points.shape[0], point_budget)
    return points[index], conditions[index], temperature[index, :1]


def predict_baseline(conditions: np.ndarray, baseline: str) -> np.ndarray:
    if baseline != "mean_temperature":
        raise SystemExit(f"Unsupported baseline: {baseline}")
    if conditions.shape[1] < 3:
        raise SystemExit("mean_temperature baseline requires source/sink temperature in condition columns 1 and 2.")
    source_temperature = conditions[:, 1:2]
    sink_temperature = conditions[:, 2:3]
    return 0.5 * (source_temperature + sink_temperature)


def predict_model(
    model: Any,
    config: dict[str, Any],
    device: Any,
    points: np.ndarray,
    conditions: np.ndarray,
    fun_dim: int,
) -> np.ndarray:
    import torch

    if conditions.shape[1] != fun_dim:
        raise SystemExit(
            f"Case conditions have fun_dim={conditions.shape[1]}, but model config expects fun_dim={fun_dim}."
        )
    features = normalize_features(conditions, config)
    model_points = normalize_points(points, config)
    with torch.no_grad():
        x = torch.from_numpy(model_points).unsqueeze(0).to(device=device, dtype=torch.float32)
        fx = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=torch.float32)
        pred = model(x, fx)
    pred_np = pred.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    if pred_np.ndim == 1:
        pred_np = pred_np[:, None]
    if pred_np.ndim != 2 or pred_np.shape[0] != points.shape[0]:
        raise SystemExit(f"Model returned invalid prediction shape {pred_np.shape}.")
    return denormalize_temperature(pred_np[:, :1].astype(np.float32), config)


def case_metrics(pred: np.ndarray, target: np.ndarray, points: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - target.astype(np.float64)
    pred_hotspot = int(np.argmax(pred.reshape(-1)))
    target_hotspot = int(np.argmax(target.reshape(-1)))
    hotspot_distance = float(
        np.linalg.norm(points[pred_hotspot].astype(np.float64) - points[target_hotspot].astype(np.float64))
    )
    max_temperature_error = max_value_error(pred, target)
    return {
        "relative_l2": relative_l2(pred, target),
        "centered_relative_l2": centered_relative_l2(pred, target),
        "rmse": float(math.sqrt(np.mean(error * error))),
        "normalized_rmse_range": normalized_rmse_range(pred, target),
        "max_value_error": max_temperature_error,
        "max_temperature_error": max_temperature_error,
        "max_temperature_abs_error": max_temperature_abs_error(pred, target),
        "hotspot_abs_error": hotspot_abs_error(pred, target),
        "hotspot_distance": hotspot_distance,
        "points": int(target.shape[0]),
    }


def summarize(metrics: list[dict[str, float]]) -> dict[str, float]:
    total_points = sum(int(item["points"]) for item in metrics)
    if total_points <= 0:
        raise SystemExit("No points evaluated.")
    return {
        "relative_l2_mean": float(np.mean([item["relative_l2"] for item in metrics])),
        "centered_relative_l2_mean": float(np.mean([item["centered_relative_l2"] for item in metrics])),
        "rmse_mean": float(np.mean([item["rmse"] for item in metrics])),
        "normalized_rmse_range_mean": float(np.mean([item["normalized_rmse_range"] for item in metrics])),
        "max_value_error_mean": float(np.mean([item["max_value_error"] for item in metrics])),
        "max_temperature_abs_error_mean": float(np.mean([item["max_temperature_abs_error"] for item in metrics])),
        "hotspot_abs_error_mean": float(np.mean([item["hotspot_abs_error"] for item in metrics])),
        "hotspot_distance_mean": float(np.mean([item["hotspot_distance"] for item in metrics])),
    }


def main() -> int:
    args = parse_args()
    if args.point_budget is not None and args.point_budget <= 0:
        raise SystemExit("--point-budget must be positive when provided.")
    if args.model_dir is None and args.baseline is None:
        raise SystemExit("Provide --model-dir or use --baseline mean_temperature.")
    if args.model_dir is not None and args.baseline is not None:
        raise SystemExit("Use either --model-dir or --baseline, not both.")

    cases = selected_cases(args)
    device = None
    model = None
    config: dict[str, Any] = {}
    fun_dim = 0
    out_dim = 0
    mode = "baseline"

    if args.model_dir is not None:
        device = resolve_device(args.device)
        model, config, fun_dim, out_dim = load_model(args.model_dir, args.checkpoint_file, device, args.geopt_vendor)
        mode = "model"

    per_case = []
    for case in cases:
        points, conditions, target = load_case(Path(case["path"]), args.point_budget)
        if model is None:
            pred = predict_baseline(conditions, args.baseline)
        else:
            pred = predict_model(model, config, device, points, conditions, fun_dim)
        metrics = case_metrics(pred, target, points)
        per_case.append({"id": case["id"], "path": case["path"], **metrics})

    payload = {
        "case_manifest": str(args.case_manifest),
        "split_path": str(args.split_path) if args.split_path else None,
        "split": args.split,
        "mode": mode,
        "baseline": args.baseline if model is None else None,
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "checkpoint_file": args.checkpoint_file if args.model_dir else None,
        "device": str(device) if device is not None else "cpu",
        "point_budget": args.point_budget,
        "case_count": len(per_case),
        "total_points": sum(int(item["points"]) for item in per_case),
        "fun_dim": fun_dim if model is not None else None,
        "out_dim": out_dim if model is not None else None,
        "temperature": summarize(per_case),
        "case_metrics": per_case,
    }

    if args.output_json:
        write_json(args.output_json, payload)
    print(json.dumps({key: payload[key] for key in ("mode", "split", "case_count", "total_points", "temperature")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
