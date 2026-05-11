#!/usr/bin/env python3
"""Render 3D surface temperature figures from OpenFOAM-backed D1 cases."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="D1 manifest containing raw_case_dir records.")
    parser.add_argument("--raw-case-dir", type=Path, action="append", help="OpenFOAM case directory. Can be repeated.")
    parser.add_argument("--case-id", action="append", help="Case id to select from --manifest. Can be repeated.")
    parser.add_argument("--family", help="Optional family filter when selecting from --manifest.")
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/d1_surface_temperature"))
    parser.add_argument("--openfoam-bash", type=Path, default=Path("/opt/openfoam13/etc/bashrc"))
    parser.add_argument("--skip-foam-to-vtk", action="store_true")
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument("--window-width", type=int, default=2100)
    parser.add_argument("--window-height", type=int, default=760)
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--show-edges", action="store_true")
    parser.add_argument("--no-feature-edges", dest="feature_edges", action="store_false")
    parser.add_argument("--camera-zoom", type=float, default=0.98)
    parser.set_defaults(feature_edges=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (manifest_path.parent / path).resolve()


def selected_cases(args: argparse.Namespace) -> list[tuple[str, Path, dict[str, Any]]]:
    if args.raw_case_dir:
        return [(path.name, path, {}) for path in args.raw_case_dir][: max(args.max_cases, 1)]
    if args.manifest is None:
        raise SystemExit("Provide --manifest or --raw-case-dir.")

    manifest = read_json(args.manifest)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise SystemExit("Manifest must contain a records list.")
    wanted_ids = set(args.case_id or [])
    cases: list[tuple[str, Path, dict[str, Any]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = str(record.get("case") or record.get("case_id") or record.get("sample") or "")
        if wanted_ids and case_id not in wanted_ids and Path(case_id).stem not in wanted_ids:
            continue
        if args.family and str(record.get("family")) != args.family:
            continue
        raw_case_dir = record.get("raw_case_dir")
        if raw_case_dir in (None, ""):
            continue
        cases.append((case_id or Path(str(raw_case_dir)).name, resolve_path(str(raw_case_dir), args.manifest), record))
        if len(cases) >= max(args.max_cases, 1):
            break
    if not cases:
        raise SystemExit("No cases selected.")
    return cases


def run_foam_to_vtk(case_dir: Path, openfoam_bash: Path) -> None:
    if not openfoam_bash.exists():
        raise FileNotFoundError(f"OpenFOAM bashrc not found: {openfoam_bash}")
    command = (
        f"source {shlex.quote(str(openfoam_bash))} && "
        f"foamToVTK -case {shlex.quote(str(case_dir))} -latestTime -fields '(T)' -nearCellValue -useTimeName"
    )
    completed = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "foamToVTK_surface_temperature.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"foamToVTK failed for {case_dir}; see {log_dir / 'foamToVTK_surface_temperature.log'}")


def latest_internal_vtk(case_dir: Path) -> Path:
    vtk_dir = case_dir / "VTK"
    candidates = sorted(path for path in vtk_dir.glob("*.vtk") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No internal VTK file found in {vtk_dir}")

    def key(path: Path) -> tuple[float, str]:
        stem = path.stem
        suffix = stem.rsplit("_", 1)[-1]
        try:
            return (float(suffix), path.name)
        except ValueError:
            return (-1.0, path.name)

    return sorted(candidates, key=key)[-1]


def load_mesh_and_surface(vtk_path: Path) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    mesh = pv.read(vtk_path)
    if "T" not in mesh.point_data and "T" in mesh.cell_data:
        mesh = mesh.cell_data_to_point_data()
    surface = mesh.extract_surface(algorithm="dataset_surface")
    if "T" not in surface.point_data and "T" in surface.cell_data:
        surface = surface.cell_data_to_point_data()
    if "T" not in mesh.point_data:
        raise ValueError(f"{vtk_path} mesh has no point temperature field 'T'")
    if "T" not in surface.point_data:
        raise ValueError(f"{vtk_path} surface has no point temperature field 'T'")
    surface = surface.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    return mesh, surface


def camera_positions(surface: pv.PolyData) -> dict[str, Any]:
    bounds = surface.bounds
    center = np.asarray(surface.center, dtype=float)
    x_span = max(bounds[1] - bounds[0], 1e-9)
    y_span = max(bounds[3] - bounds[2], 1e-9)
    z_span = max(bounds[5] - bounds[4], 1e-9)
    scale = max(x_span, y_span, z_span)
    return {
        "iso": [
            tuple(center + np.asarray([1.25 * scale, -1.55 * scale, 0.95 * scale])),
            tuple(center),
            (0.0, 0.0, 1.0),
        ],
        "side": [
            tuple(center + np.asarray([0.0, -2.25 * scale, 0.35 * scale])),
            tuple(center),
            (0.0, 0.0, 1.0),
        ],
        "top": [
            tuple(center + np.asarray([0.0, 0.0, 2.35 * scale])),
            tuple(center),
            (0.0, 1.0, 0.0),
        ],
    }


def internal_slice(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    center = np.asarray(mesh.center, dtype=float)
    for normal in ("y", "x", "z"):
        section = mesh.slice(normal=normal, origin=center)
        if section.n_points > 0:
            return section
    raise ValueError("Failed to extract a non-empty internal temperature slice")


def feature_edge_mesh(surface: pv.PolyData) -> pv.PolyData:
    edges = surface.extract_feature_edges(
        feature_edges=True,
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
        feature_angle=24.0,
    )
    return edges


def render_case(case_id: str, case_dir: Path, record: dict[str, Any], args: argparse.Namespace) -> Path:
    if not args.skip_foam_to_vtk:
        run_foam_to_vtk(case_dir, args.openfoam_bash)
    vtk_path = latest_internal_vtk(case_dir)
    mesh, surface = load_mesh_and_surface(vtk_path)
    section = internal_slice(mesh)
    edges = feature_edge_mesh(surface) if args.feature_edges else None
    temperature = np.asarray(surface.point_data["T"], dtype=float)
    mesh_temperature = np.asarray(mesh.point_data["T"], dtype=float)
    clim = (float(np.percentile(mesh_temperature, 1.0)), float(np.percentile(mesh_temperature, 99.0)))
    family = str(record.get("family") or "unknown")

    pv.global_theme.font.family = "arial"
    plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 3),
        window_size=(args.window_width, args.window_height),
    )
    plotter.set_background("white")

    cameras = camera_positions(surface)
    panels = [
        ("3D surface temperature", "surface", cameras["iso"]),
        ("internal heat path slice", "slice", cameras["side"]),
        ("top fin layout", "surface", cameras["top"]),
    ]
    for idx, (title, mode, camera_position) in enumerate(panels):
        plotter.subplot(0, idx)
        if mode == "slice":
            plotter.add_mesh(
                surface,
                scalars="T",
                cmap=args.cmap,
                clim=clim,
                opacity=0.18,
                smooth_shading=True,
                lighting=False,
                show_scalar_bar=False,
            )
            plotter.add_mesh(
                section,
                scalars="T",
                cmap=args.cmap,
                clim=clim,
                show_edges=False,
                lighting=False,
                scalar_bar_args={
                    "title": "T [K]",
                    "vertical": True,
                    "title_font_size": 13,
                    "label_font_size": 11,
                    "height": 0.72,
                    "position_x": 0.84,
                    "position_y": 0.14,
                },
                show_scalar_bar=True,
            )
            if edges is not None:
                plotter.add_mesh(
                    edges,
                    color="#111111",
                    line_width=0.8,
                    opacity=0.55,
                    render_lines_as_tubes=True,
                    show_scalar_bar=False,
                )
        else:
            plotter.add_mesh(
                surface,
                scalars="T",
                cmap=args.cmap,
                clim=clim,
                smooth_shading=True,
                show_edges=args.show_edges or idx == 2,
                edge_color="#1f2933",
                line_width=0.35,
                lighting=False,
                show_scalar_bar=False,
            )
        if edges is not None and mode != "slice":
            plotter.add_mesh(edges, color="#111111", line_width=1.0, render_lines_as_tubes=True, show_scalar_bar=False)
        plotter.add_text(title, position="upper_left", font_size=11, color="black")
        plotter.camera_position = camera_position
        plotter.camera.zoom(args.camera_zoom)
        plotter.enable_parallel_projection()
        plotter.add_axes(line_width=2, labels_off=False)

    plotter.subplot(0, 0)
    plotter.add_text(
        f"{case_id}\n{family} | OpenFOAM steady solid conduction",
        position="lower_right",
        font_size=10,
        color="black",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{case_id}_surface_temperature.png"
    plotter.screenshot(str(output_path), transparent_background=args.transparent_background)
    plotter.close()
    return output_path


def main() -> int:
    args = parse_args()
    outputs: list[str] = []
    for case_id, case_dir, record in selected_cases(args):
        output_path = render_case(case_id, case_dir, record, args)
        outputs.append(str(output_path))
        print({"case": case_id, "raw_case_dir": str(case_dir), "output": str(output_path)})
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
