#!/usr/bin/env python3
"""Render D1 temperature contour slices from solver-backed NPZ cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="D1 manifest containing NPZ records.")
    parser.add_argument("--case", type=Path, action="append", help="Specific NPZ case path. Can be repeated.")
    parser.add_argument("--case-id", action="append", help="Case id to select from --manifest. Can be repeated.")
    parser.add_argument("--family", help="Optional family filter when selecting from --manifest.")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/d1_contours"))
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--levels", type=int, default=32)
    parser.add_argument("--slice-tolerance-frac", type=float, default=0.015)
    parser.add_argument("--min-slice-points", type=int, default=80)
    parser.add_argument("--mask-distance-scale", type=float, default=0.8)
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (manifest_path.parent / path).resolve()


def selected_cases(args: argparse.Namespace) -> list[tuple[str, Path, dict[str, Any]]]:
    cases: list[tuple[str, Path, dict[str, Any]]] = []
    if args.case:
        for path in args.case:
            cases.append((path.stem, path, {}))
        return cases[: max(args.max_cases, 1)]

    if args.manifest is None:
        raise SystemExit("Provide --manifest or --case.")

    manifest = read_json(args.manifest)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise SystemExit("Manifest must contain a records list.")
    wanted_ids = set(args.case_id or [])
    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = str(record.get("case") or record.get("case_id") or record.get("sample") or "")
        if wanted_ids and case_id not in wanted_ids and Path(case_id).stem not in wanted_ids:
            continue
        if args.family and str(record.get("family")) != args.family:
            continue
        raw_path = record.get("path")
        if raw_path in (None, ""):
            continue
        cases.append((case_id or Path(str(raw_path)).stem, resolve_record_path(str(raw_path), args.manifest), record))
        if len(cases) >= max(args.max_cases, 1):
            break
    if not cases:
        raise SystemExit("No cases selected.")
    return cases


def load_case(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path) as data:
        points = np.asarray(data["points"], dtype=np.float64)
        temperature = np.asarray(data["temperature"], dtype=np.float64).reshape(-1)
        metadata = {
            "generator": str(data["generator"]) if "generator" in data else "",
            "case_params_json": str(data["case_params_json"]) if "case_params_json" in data else "",
        }
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path} points must have shape [N, 3], got {points.shape}")
    if points.shape[0] != temperature.shape[0]:
        raise ValueError(f"{path} point/temperature mismatch: {points.shape[0]} vs {temperature.shape[0]}")
    return points, temperature, metadata


def choose_slice(points: np.ndarray, axis: str, value: float | None, tolerance_frac: float, min_points: int) -> np.ndarray:
    axis_idx = AXIS_INDEX[axis]
    coord = points[:, axis_idx]
    if value is None:
        value = 0.5 * (float(coord.min()) + float(coord.max()))
    span = max(float(coord.max() - coord.min()), 1e-12)
    tolerance = max(span * tolerance_frac, 1e-12)
    for scale in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        mask = np.abs(coord - value) <= tolerance * scale
        if int(mask.sum()) >= min_points:
            return mask
    distances = np.abs(coord - value)
    count = min(max(min_points, 8), points.shape[0])
    ids = np.argsort(distances)[:count]
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[ids] = True
    return mask


def slice_panel(
    ax: plt.Axes,
    *,
    points: np.ndarray,
    temperature: np.ndarray,
    fixed_axis: str,
    fixed_value: float | None,
    title: str,
    args: argparse.Namespace,
    vmin: float,
    vmax: float,
) -> Any:
    mask = choose_slice(points, fixed_axis, fixed_value, args.slice_tolerance_frac, args.min_slice_points)
    fixed_idx = AXIS_INDEX[fixed_axis]
    free_axes = [idx for idx in range(3) if idx != fixed_idx]
    free_names = ["x", "y", "z"]
    xs = points[mask, free_axes[0]]
    ys = points[mask, free_axes[1]]
    values = temperature[mask]

    x_span = max(float(xs.max() - xs.min()), 1e-12)
    y_span = max(float(ys.max() - ys.min()), 1e-12)
    gx = np.linspace(float(xs.min()), float(xs.max()), args.grid_size)
    gy = np.linspace(float(ys.min()), float(ys.max()), args.grid_size)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_t = griddata(np.column_stack([xs, ys]), values, (grid_x, grid_y), method="linear")
    if np.isnan(grid_t).all():
        grid_t = griddata(np.column_stack([xs, ys]), values, (grid_x, grid_y), method="nearest")
    nearest_distance, _ = cKDTree(np.column_stack([xs, ys])).query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)
    sorted_x = np.unique(np.round(xs, 10))
    sorted_y = np.unique(np.round(ys, 10))
    dx = np.median(np.diff(sorted_x)) if sorted_x.size > 1 else x_span / max(args.grid_size, 1)
    dy = np.median(np.diff(sorted_y)) if sorted_y.size > 1 else y_span / max(args.grid_size, 1)
    mask_distance = args.mask_distance_scale * max(float(dx), float(dy), 1e-12)
    outside = nearest_distance.reshape(grid_x.shape) > mask_distance
    grid_t = np.ma.array(grid_t, mask=np.isnan(grid_t) | outside)

    contour = ax.contourf(grid_x, grid_y, grid_t, levels=args.levels, cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax.scatter(xs, ys, s=1.2, c="white", alpha=0.16, linewidths=0)
    ax.set_title(f"{title} ({int(mask.sum())} cells)", fontsize=9)
    ax.set_xlabel(free_names[free_axes[0]])
    ax.set_ylabel(free_names[free_axes[1]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(xs.min()) - 0.02 * x_span, float(xs.max()) + 0.02 * x_span)
    ax.set_ylim(float(ys.min()) - 0.02 * y_span, float(ys.max()) + 0.02 * y_span)
    ax.tick_params(labelsize=7)
    return contour


def render_case(case_id: str, path: Path, record: dict[str, Any], args: argparse.Namespace) -> Path:
    points, temperature, _ = load_case(path)
    family = str(record.get("family") or "unknown")
    z_span = float(points[:, 2].max() - points[:, 2].min())
    xy_top_z = float(points[:, 2].min() + 0.55 * z_span)
    vmin = float(np.percentile(temperature, 1.0))
    vmax = float(np.percentile(temperature, 99.0))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    contour = slice_panel(
        axes[0],
        points=points,
        temperature=temperature,
        fixed_axis="y",
        fixed_value=None,
        title="mid-y side slice",
        args=args,
        vmin=vmin,
        vmax=vmax,
    )
    slice_panel(
        axes[1],
        points=points,
        temperature=temperature,
        fixed_axis="x",
        fixed_value=None,
        title="mid-x side slice",
        args=args,
        vmin=vmin,
        vmax=vmax,
    )
    slice_panel(
        axes[2],
        points=points,
        temperature=temperature,
        fixed_axis="z",
        fixed_value=xy_top_z,
        title="upper plan slice",
        args=args,
        vmin=vmin,
        vmax=vmax,
    )
    fig.suptitle(
        f"{case_id}  family={family}  T=[{temperature.min():.1f}, {temperature.max():.1f}] K",
        fontsize=11,
    )
    cbar = fig.colorbar(contour, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label("temperature [K]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{case_id}_temperature_contours.png"
    fig.savefig(output_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    cases = selected_cases(args)
    outputs = []
    for case_id, path, record in cases:
        output_path = render_case(case_id, path, record, args)
        outputs.append(str(output_path))
        print({"case": case_id, "input": str(path), "output": str(output_path)})
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
