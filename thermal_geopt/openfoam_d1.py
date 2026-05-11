"""OpenFOAM-backed D1 solid conduction case utilities.

The first M1 target is deliberately small: generate a blockMesh solid domain,
solve a Laplace/steady diffusion temperature field with OpenFOAM
``laplacianFoam``, and convert the cell-centred field to the NPZ schema used by
the downstream Transolver pipeline.  Proxy generation remains separate in
``thermal_geopt.d1_conduction``.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class D1OpenFOAMCaseParams:
    case_id: str
    length_x: float
    length_y: float
    length_z: float
    cells_x: int
    cells_y: int
    cells_z: int
    diffusivity: float
    conductivity: float
    source_temperature: float
    sink_temperature: float
    initial_temperature: float
    bc_model: str = "fixed_temperature_source_sink_insulated_sides"

    @property
    def cell_count(self) -> int:
        return int(self.cells_x * self.cells_y * self.cells_z)


@dataclass(frozen=True)
class D1OpenFOAMHeatSinkParams:
    case_id: str
    family: str
    width: float
    depth: float
    base_height: float
    feature_height: float
    feature_count_x: int
    feature_count_y: int
    feature_width: float
    feature_depth: float
    cells_x: int
    cells_y: int
    base_cells_z: int
    feature_cells_z: int
    diffusivity: float
    conductivity: float
    source_temperature: float
    sink_temperature: float
    initial_temperature: float
    sink_value_fraction: float = 1.0
    bc_model: str = "fixed_hot_base_cold_exterior"


@dataclass(frozen=True)
class _MeshBlock:
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    cells_x: int
    cells_y: int
    cells_z: int


@dataclass(frozen=True)
class _BoundaryFace:
    patch: str
    axis: int
    value: float
    ranges: tuple[tuple[float, float], tuple[float, float]]
    normal: tuple[float, float, float]


def foam_header(*, class_name: str, object_name: str, location: str | None = None) -> str:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        f"    class       {class_name};",
    ]
    if location is not None:
        lines.append(f'    location    "{location}";')
    lines.extend([f"    object      {object_name};", "}", ""])
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def block_mesh_dict(params: D1OpenFOAMCaseParams) -> str:
    lx, ly, lz = params.length_x, params.length_y, params.length_z
    return f"""
{foam_header(class_name="dictionary", object_name="blockMeshDict")}
convertToMeters 1;

vertices
(
    (0 0 0)
    ({lx:.9g} 0 0)
    ({lx:.9g} {ly:.9g} 0)
    (0 {ly:.9g} 0)
    (0 0 {lz:.9g})
    ({lx:.9g} 0 {lz:.9g})
    ({lx:.9g} {ly:.9g} {lz:.9g})
    (0 {ly:.9g} {lz:.9g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({params.cells_x} {params.cells_y} {params.cells_z}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    source
    {{
        type patch;
        faces
        (
            (0 3 2 1)
        );
    }}
    sink
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
    xmin
    {{
        type wall;
        faces
        (
            (0 4 7 3)
        );
    }}
    xmax
    {{
        type wall;
        faces
        (
            (1 2 6 5)
        );
    }}
    ymin
    {{
        type wall;
        faces
        (
            (0 1 5 4)
        );
    }}
    ymax
    {{
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }}
);

mergePatchPairs
(
);
"""


def temperature_field(params: D1OpenFOAMCaseParams) -> str:
    return f"""
{foam_header(class_name="volScalarField", object_name="T", location="0")}
dimensions      [0 0 0 1 0 0 0];

internalField   uniform {params.initial_temperature:.9g};

boundaryField
{{
    source
    {{
        type            fixedValue;
        value           uniform {params.source_temperature:.9g};
    }}

    sink
    {{
        type            fixedValue;
        value           uniform {params.sink_temperature:.9g};
    }}

    xmin
    {{
        type            zeroGradient;
    }}

    xmax
    {{
        type            zeroGradient;
    }}

    ymin
    {{
        type            zeroGradient;
    }}

    ymax
    {{
        type            zeroGradient;
    }}
}}
"""


def physical_properties(params: D1OpenFOAMCaseParams) -> str:
    return f"""
{foam_header(class_name="dictionary", object_name="physicalProperties", location="constant")}
DT              DT [0 2 -1 0 0 0 0] {params.diffusivity:.9g};
"""


def control_dict() -> str:
    return f"""
{foam_header(class_name="dictionary", object_name="controlDict", location="system")}
application     laplacianFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;

writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  9;
writeCompression off;
timeFormat      general;
timePrecision   9;
runTimeModifiable false;
"""


def fv_schemes() -> str:
    return f"""
{foam_header(class_name="dictionary", object_name="fvSchemes", location="system")}
ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
    grad(T)         Gauss linear;
}}

divSchemes
{{
    default         none;
}}

laplacianSchemes
{{
    default         none;
    laplacian(DT,T) Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""


def fv_solution() -> str:
    return f"""
{foam_header(class_name="dictionary", object_name="fvSolution", location="system")}
solvers
{{
    T
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-10;
        relTol          0;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl
    {{
        T               1e-10;
    }}
}}
"""


def write_openfoam_case(case_dir: Path, params: D1OpenFOAMCaseParams, *, overwrite: bool = False) -> None:
    if case_dir.exists() and overwrite:
        shutil.rmtree(case_dir)
    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError(f"OpenFOAM case already exists: {case_dir}")

    write_text(case_dir / "system" / "blockMeshDict", block_mesh_dict(params))
    write_text(case_dir / "system" / "controlDict", control_dict())
    write_text(case_dir / "system" / "fvSchemes", fv_schemes())
    write_text(case_dir / "system" / "fvSolution", fv_solution())
    write_text(case_dir / "constant" / "physicalProperties", physical_properties(params))
    write_text(case_dir / "0" / "T", temperature_field(params))
    write_text(case_dir / "case_params.json", json.dumps(asdict(params), indent=2))


def run_command(command: str, *, case_dir: Path, log_name: str, openfoam_bash: Path) -> None:
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    wrapped = f"source {shlex.quote(str(openfoam_bash))} && {command}"
    completed = subprocess.run(
        ["bash", "-lc", wrapped],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (log_dir / log_name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command}. See {log_dir / log_name}")


def run_openfoam_case(case_dir: Path, *, openfoam_bash: Path = Path("/opt/openfoam13/etc/bashrc")) -> None:
    quoted_case = shlex.quote(str(case_dir.resolve()))
    run_command(f"blockMesh -case {quoted_case}", case_dir=case_dir, log_name="blockMesh.log", openfoam_bash=openfoam_bash)
    run_command(
        f"laplacianFoam -case {quoted_case}",
        case_dir=case_dir,
        log_name="laplacianFoam.log",
        openfoam_bash=openfoam_bash,
    )
    run_command(
        f"postProcess -case {quoted_case} -latestTime -func writeCellCentres",
        case_dir=case_dir,
        log_name="writeCellCentres.log",
        openfoam_bash=openfoam_bash,
    )


def numeric_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    found = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        found.append((value, child))
    return sorted(found, key=lambda item: item[0])


def latest_time_dir(case_dir: Path) -> Path:
    times = numeric_time_dirs(case_dir)
    if not times:
        raise FileNotFoundError(f"No numeric OpenFOAM time directories found in {case_dir}")
    return times[-1][1]


def parse_internal_field(path: Path, *, components: int, expected_count: int | None = None) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    start_match = re.search(r"\binternalField\b", text)
    if start_match is None:
        raise ValueError(f"{path} has no internalField")
    field_text = text[start_match.end() :]

    uniform_match = re.search(r"\buniform\s+([^;]+);", field_text)
    nonuniform_match = re.search(r"\bnonuniform\s+List<[^>]+>\s+(\d+)\s*\(", field_text)
    if uniform_match is not None and (
        nonuniform_match is None or uniform_match.start() < nonuniform_match.start()
    ):
        if expected_count is None:
            raise ValueError(f"{path} uses uniform internalField but expected_count was not supplied")
        values = np.fromstring(uniform_match.group(1).replace("(", " ").replace(")", " "), sep=" ", dtype=np.float64)
        if components == 1:
            if values.size != 1:
                raise ValueError(f"{path} uniform scalar field has {values.size} values")
            return np.full((expected_count, 1), float(values[0]), dtype=np.float32)
        if values.size != components:
            raise ValueError(f"{path} uniform vector field has {values.size} values; expected {components}")
        return np.tile(values.reshape(1, components), (expected_count, 1)).astype(np.float32)

    if nonuniform_match is None:
        raise ValueError(f"{path} has no supported uniform/nonuniform internalField")

    count = int(nonuniform_match.group(1))
    body_start = nonuniform_match.end()
    body_end = field_text.find("\n)", body_start)
    if body_end < 0:
        body_end = field_text.find(");", body_start)
    if body_end < 0:
        raise ValueError(f"{path} has an unterminated nonuniform field")
    body = field_text[body_start:body_end]

    if components == 1:
        values = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ", dtype=np.float64)
        if values.size != count:
            raise ValueError(f"{path} parsed {values.size} scalar values; expected {count}")
        return values.reshape(count, 1).astype(np.float32)

    vector_items = re.findall(r"\(([^()]+)\)", body)
    if len(vector_items) != count:
        raise ValueError(f"{path} parsed {len(vector_items)} vector values; expected {count}")
    values = np.asarray(
        [np.fromstring(item, sep=" ", dtype=np.float64) for item in vector_items],
        dtype=np.float64,
    )
    if values.shape != (count, components):
        raise ValueError(f"{path} parsed vector shape {values.shape}; expected {(count, components)}")
    return values.astype(np.float32)


def nearest_block_boundary(points: np.ndarray, params: D1OpenFOAMCaseParams) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    distances = np.stack(
        [
            pts[:, 0],
            params.length_x - pts[:, 0],
            pts[:, 1],
            params.length_y - pts[:, 1],
            pts[:, 2],
            params.length_z - pts[:, 2],
        ],
        axis=1,
    )
    nearest = np.argmin(distances, axis=1)
    normals = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return distances[np.arange(pts.shape[0]), nearest].astype(np.float32), normals[nearest]


def block_source_sink_features(points: np.ndarray, params: D1OpenFOAMCaseParams) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    dz = max(params.length_z / max(params.cells_z, 1), 1e-12)
    source = np.exp(-pts[:, 2:3] / max(2.0 * dz, 1e-12))
    sink = np.exp(-(params.length_z - pts[:, 2:3]) / max(2.0 * dz, 1e-12))
    return source.astype(np.float32), sink.astype(np.float32)


def convert_case_to_npz(case_dir: Path, params: D1OpenFOAMCaseParams, output_path: Path) -> dict[str, Any]:
    latest = latest_time_dir(case_dir)
    centers_path = latest / "C"
    temperature_path = latest / "T"
    if not centers_path.exists():
        raise FileNotFoundError(f"Missing cell centres field: {centers_path}")
    if not temperature_path.exists():
        raise FileNotFoundError(f"Missing temperature field: {temperature_path}")

    points = parse_internal_field(centers_path, components=3)
    temperature = parse_internal_field(temperature_path, components=1, expected_count=points.shape[0])
    if points.shape[0] != temperature.shape[0]:
        raise ValueError(f"Point/temperature count mismatch: {points.shape[0]} vs {temperature.shape[0]}")

    nearest_distance, nearest_normal = nearest_block_boundary(points, params)
    source_patch, sink_patch = block_source_sink_features(points, params)
    conditions = np.concatenate(
        [
            np.full((points.shape[0], 1), params.conductivity, dtype=np.float32),
            np.full((points.shape[0], 1), params.source_temperature, dtype=np.float32),
            np.full((points.shape[0], 1), params.sink_temperature, dtype=np.float32),
            source_patch,
            sink_patch,
            nearest_distance[:, None].astype(np.float32),
        ],
        axis=1,
    )
    material = np.concatenate(
        [
            np.full((points.shape[0], 1), params.conductivity, dtype=np.float32),
            np.full((points.shape[0], 1), params.diffusivity, dtype=np.float32),
        ],
        axis=1,
    )
    bc_features = np.concatenate(
        [
            source_patch,
            sink_patch,
            np.full((points.shape[0], 1), params.source_temperature, dtype=np.float32),
            np.full((points.shape[0], 1), params.sink_temperature, dtype=np.float32),
            nearest_distance[:, None].astype(np.float32),
        ],
        axis=1,
    )
    condition_names = np.asarray(
        [
            "conductivity",
            "source_temperature",
            "sink_temperature",
            "source_patch",
            "sink_patch",
            "nearest_boundary_distance",
        ]
    )
    case_params = asdict(params)
    case_params_json = json.dumps(case_params, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        normals=nearest_normal.astype(np.float32),
        region=np.zeros(points.shape[0], dtype=np.int32),
        material=material.astype(np.float32),
        bc_features=bc_features.astype(np.float32),
        T=temperature.astype(np.float32),
        temperature=temperature.astype(np.float32),
        conditions=conditions.astype(np.float32),
        condition_names=condition_names,
        source_patch=source_patch.astype(np.float32),
        sink_patch=sink_patch.astype(np.float32),
        nearest_boundary_distance=nearest_distance.astype(np.float32),
        nearest_boundary_normal=nearest_normal.astype(np.float32),
        case_params_json=np.asarray(case_params_json),
        case_params=np.asarray(
            [
                params.length_x,
                params.length_y,
                params.length_z,
                params.cells_x,
                params.cells_y,
                params.cells_z,
                params.diffusivity,
                params.conductivity,
                params.source_temperature,
                params.sink_temperature,
            ],
            dtype=np.float32,
        ),
        case_param_names=np.asarray(
            [
                "length_x",
                "length_y",
                "length_z",
                "cells_x",
                "cells_y",
                "cells_z",
                "diffusivity",
                "conductivity",
                "source_temperature",
                "sink_temperature",
            ]
        ),
        solver=np.asarray("OpenFOAM-13 laplacianFoam"),
        generator=np.asarray("d1_openfoam_block_v1"),
    )
    return {
        "case": params.case_id,
        "path": str(output_path),
        "raw_case_dir": str(case_dir),
        "solver": "OpenFOAM-13 laplacianFoam",
        "generator": "d1_openfoam_block_v1",
        "bc_model": params.bc_model,
        "points": int(points.shape[0]),
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "temperature_mean": float(np.mean(temperature)),
        **case_params,
    }


def sample_block_params(
    *,
    case_id: str,
    rng: np.random.Generator,
    cells: tuple[int, int, int],
) -> D1OpenFOAMCaseParams:
    length_x = float(rng.uniform(0.04, 0.12))
    length_y = float(rng.uniform(0.04, 0.12))
    length_z = float(rng.uniform(0.008, 0.03))
    source_temperature = float(rng.uniform(360.0, 430.0))
    sink_temperature = float(rng.uniform(280.0, 310.0))
    if source_temperature < sink_temperature + 40.0:
        source_temperature = sink_temperature + float(rng.uniform(40.0, 90.0))
    conductivity = float(np.exp(rng.uniform(np.log(0.2), np.log(25.0))))
    diffusivity = float(np.exp(rng.uniform(np.log(1.0e-6), np.log(1.0e-4))))
    initial_temperature = 0.5 * (source_temperature + sink_temperature)
    return D1OpenFOAMCaseParams(
        case_id=case_id,
        length_x=length_x,
        length_y=length_y,
        length_z=length_z,
        cells_x=int(cells[0]),
        cells_y=int(cells[1]),
        cells_z=int(cells[2]),
        diffusivity=diffusivity,
        conductivity=conductivity,
        source_temperature=source_temperature,
        sink_temperature=sink_temperature,
        initial_temperature=float(initial_temperature),
    )


def _clip_interval(lo: float, hi: float, *, lower: float, upper: float) -> tuple[float, float]:
    lo = max(lower, min(upper, float(lo)))
    hi = max(lower, min(upper, float(hi)))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _unique_edges(values: list[float], *, lower: float, upper: float) -> list[float]:
    clipped = [max(lower, min(upper, float(value))) for value in values]
    rounded = sorted({round(value, 12) for value in clipped})
    edges: list[float] = []
    for value in rounded:
        if not edges or abs(value - edges[-1]) > 1e-10:
            edges.append(value)
    if not edges or abs(edges[0] - lower) > 1e-10:
        edges.insert(0, float(lower))
    if abs(edges[-1] - upper) > 1e-10:
        edges.append(float(upper))
    return edges


def _interval_cells(edges: list[float], total_cells: int) -> dict[tuple[float, float], int]:
    widths = [max(edges[idx + 1] - edges[idx], 0.0) for idx in range(len(edges) - 1)]
    total_width = max(sum(widths), 1e-12)
    result: dict[tuple[float, float], int] = {}
    for idx, width in enumerate(widths):
        lo, hi = edges[idx], edges[idx + 1]
        if hi <= lo + 1e-10:
            continue
        cells = max(1, int(round(total_cells * width / total_width)))
        result[(round(lo, 12), round(hi, 12))] = cells
    return result


def _cell_count_for_span(
    cell_map: dict[tuple[float, float], int],
    lo: float,
    hi: float,
    *,
    fallback_total_cells: int,
    fallback_total_width: float,
) -> int:
    lo_key = round(float(lo), 12)
    hi_key = round(float(hi), 12)
    direct = cell_map.get((lo_key, hi_key))
    if direct is not None:
        return direct
    cells = 0
    for (edge_lo, edge_hi), count in cell_map.items():
        if edge_lo >= lo_key - 1e-10 and edge_hi <= hi_key + 1e-10:
            cells += count
    if cells > 0:
        return cells
    width = max(float(hi - lo), 0.0)
    return max(1, int(round(fallback_total_cells * width / max(fallback_total_width, 1e-12))))


def _feature_footprints(params: D1OpenFOAMHeatSinkParams) -> list[tuple[float, float, float, float]]:
    footprints: list[tuple[float, float, float, float]] = []
    if params.family == "plate_fin":
        count = max(1, int(params.feature_count_x))
        margin = 0.08 * params.width
        if count == 1:
            centers = np.asarray([0.5 * params.width], dtype=np.float64)
        else:
            centers = np.linspace(margin, params.width - margin, count)
        y0 = 0.5 * (params.depth - params.feature_depth)
        y1 = y0 + params.feature_depth
        for center in centers:
            x0, x1 = _clip_interval(
                float(center) - 0.5 * params.feature_width,
                float(center) + 0.5 * params.feature_width,
                lower=0.0,
                upper=params.width,
            )
            footprints.append((x0, x1, y0, y1))
        return footprints

    if params.family in {"pin_fin", "staggered_pin_fin"}:
        nx = max(1, int(params.feature_count_x))
        ny = max(1, int(params.feature_count_y))
        base_xs = np.linspace(0.14 * params.width, 0.86 * params.width, nx)
        ys = np.linspace(0.14 * params.depth, 0.86 * params.depth, ny)
        pitch = 0.72 * params.width / max(nx - 1, 1)
        for row_index, y in enumerate(ys):
            xs = base_xs
            if params.family == "staggered_pin_fin" and nx > 1 and row_index % 2 == 1:
                xs = base_xs + 0.5 * pitch
                xs = xs[(xs >= 0.12 * params.width) & (xs <= 0.88 * params.width)]
            for x in xs:
                x0, x1 = _clip_interval(
                    float(x) - 0.5 * params.feature_width,
                    float(x) + 0.5 * params.feature_width,
                    lower=0.0,
                    upper=params.width,
                )
                y0, y1 = _clip_interval(
                    float(y) - 0.5 * params.feature_depth,
                    float(y) + 0.5 * params.feature_depth,
                    lower=0.0,
                    upper=params.depth,
                )
                footprints.append((x0, x1, y0, y1))
        return footprints

    raise ValueError(f"Unsupported heat-sink family: {params.family}")


def _heatsink_blocks(params: D1OpenFOAMHeatSinkParams) -> list[_MeshBlock]:
    footprints = _feature_footprints(params)
    x_edges = _unique_edges(
        [0.0, params.width] + [value for footprint in footprints for value in footprint[:2]],
        lower=0.0,
        upper=params.width,
    )
    y_edges = _unique_edges(
        [0.0, params.depth] + [value for footprint in footprints for value in footprint[2:]],
        lower=0.0,
        upper=params.depth,
    )
    x_cells = _interval_cells(x_edges, params.cells_x)
    y_cells = _interval_cells(y_edges, params.cells_y)
    blocks: list[_MeshBlock] = []

    for xi in range(len(x_edges) - 1):
        x0, x1 = x_edges[xi], x_edges[xi + 1]
        if x1 <= x0 + 1e-10:
            continue
        for yi in range(len(y_edges) - 1):
            y0, y1 = y_edges[yi], y_edges[yi + 1]
            if y1 <= y0 + 1e-10:
                continue
            key_x = (round(x0, 12), round(x1, 12))
            key_y = (round(y0, 12), round(y1, 12))
            blocks.append(
                _MeshBlock(
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    z0=0.0,
                    z1=params.base_height,
                    cells_x=x_cells[key_x],
                    cells_y=y_cells[key_y],
                    cells_z=max(1, int(params.base_cells_z)),
                )
            )

    footprint_set = {
        (round(x0, 12), round(x1, 12), round(y0, 12), round(y1, 12)) for x0, x1, y0, y1 in footprints
    }
    for x0, x1, y0, y1 in sorted(footprint_set):
        blocks.append(
            _MeshBlock(
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                z0=params.base_height,
                z1=params.base_height + params.feature_height,
                cells_x=_cell_count_for_span(
                    x_cells,
                    x0,
                    x1,
                    fallback_total_cells=params.cells_x,
                    fallback_total_width=params.width,
                ),
                cells_y=_cell_count_for_span(
                    y_cells,
                    y0,
                    y1,
                    fallback_total_cells=params.cells_y,
                    fallback_total_width=params.depth,
                ),
                cells_z=max(1, int(params.feature_cells_z)),
            )
        )
    return blocks


def _coord_key(coord: tuple[float, float, float]) -> tuple[float, float, float]:
    return (round(coord[0], 12), round(coord[1], 12), round(coord[2], 12))


def _block_coords(block: _MeshBlock) -> list[tuple[float, float, float]]:
    return [
        (block.x0, block.y0, block.z0),
        (block.x1, block.y0, block.z0),
        (block.x1, block.y1, block.z0),
        (block.x0, block.y1, block.z0),
        (block.x0, block.y0, block.z1),
        (block.x1, block.y0, block.z1),
        (block.x1, block.y1, block.z1),
        (block.x0, block.y1, block.z1),
    ]


def _face_specs(block: _MeshBlock) -> list[tuple[str, tuple[int, int, int, int], int, float, tuple[tuple[float, float], tuple[float, float]], tuple[float, float, float]]]:
    return [
        ("zmin", (0, 3, 2, 1), 2, block.z0, ((block.x0, block.x1), (block.y0, block.y1)), (0.0, 0.0, -1.0)),
        ("zmax", (4, 5, 6, 7), 2, block.z1, ((block.x0, block.x1), (block.y0, block.y1)), (0.0, 0.0, 1.0)),
        ("xmin", (0, 4, 7, 3), 0, block.x0, ((block.y0, block.y1), (block.z0, block.z1)), (-1.0, 0.0, 0.0)),
        ("xmax", (1, 2, 6, 5), 0, block.x1, ((block.y0, block.y1), (block.z0, block.z1)), (1.0, 0.0, 0.0)),
        ("ymin", (0, 1, 5, 4), 1, block.y0, ((block.x0, block.x1), (block.z0, block.z1)), (0.0, -1.0, 0.0)),
        ("ymax", (3, 7, 6, 2), 1, block.y1, ((block.x0, block.x1), (block.z0, block.z1)), (0.0, 1.0, 0.0)),
    ]


def _heatsink_mesh(
    params: D1OpenFOAMHeatSinkParams,
) -> tuple[list[tuple[float, float, float]], list[tuple[tuple[int, ...], tuple[int, int, int]]], list[_BoundaryFace], dict[str, list[tuple[int, int, int, int]]]]:
    vertices: list[tuple[float, float, float]] = []
    vertex_ids: dict[tuple[float, float, float], int] = {}
    blocks = _heatsink_blocks(params)
    block_defs: list[tuple[tuple[int, ...], tuple[int, int, int]]] = []

    def add_vertex(coord: tuple[float, float, float]) -> int:
        key = _coord_key(coord)
        existing = vertex_ids.get(key)
        if existing is not None:
            return existing
        vertex_ids[key] = len(vertices)
        vertices.append(key)
        return vertex_ids[key]

    block_vertex_ids: list[tuple[_MeshBlock, tuple[int, ...]]] = []
    for block in blocks:
        ids = tuple(add_vertex(coord) for coord in _block_coords(block))
        block_vertex_ids.append((block, ids))
        block_defs.append((ids, (block.cells_x, block.cells_y, block.cells_z)))

    face_counts: dict[tuple[int, int, int, int], int] = {}
    for block, ids in block_vertex_ids:
        for _, local_face, *_ in _face_specs(block):
            face = tuple(ids[index] for index in local_face)
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1

    patch_faces: dict[str, list[tuple[int, int, int, int]]] = {"source": [], "sink": []}
    external_faces: list[_BoundaryFace] = []
    for block, ids in block_vertex_ids:
        for name, local_face, axis, value, ranges, normal in _face_specs(block):
            face = tuple(ids[index] for index in local_face)
            if face_counts[tuple(sorted(face))] != 1:
                continue
            patch = "source" if name == "zmin" and abs(value) <= 1e-10 else "sink"
            patch_faces[patch].append(face)
            external_faces.append(_BoundaryFace(patch=patch, axis=axis, value=value, ranges=ranges, normal=normal))

    return vertices, block_defs, external_faces, patch_faces


def heatsink_block_mesh_dict(params: D1OpenFOAMHeatSinkParams) -> str:
    vertices, block_defs, _, patch_faces = _heatsink_mesh(params)
    vertex_lines = "\n".join(f"    ({x:.9g} {y:.9g} {z:.9g})" for x, y, z in vertices)
    block_lines = "\n".join(
        "    hex ({}) ({} {} {}) simpleGrading (1 1 1)".format(
            " ".join(str(index) for index in ids),
            cells[0],
            cells[1],
            cells[2],
        )
        for ids, cells in block_defs
    )

    def patch_block(name: str) -> str:
        faces = patch_faces.get(name, [])
        face_lines = "\n".join("            ({})".format(" ".join(str(index) for index in face)) for face in faces)
        return f"""
    {name}
    {{
        type patch;
        faces
        (
{face_lines}
        );
    }}"""

    return f"""
{foam_header(class_name="dictionary", object_name="blockMeshDict")}
convertToMeters 1;

vertices
(
{vertex_lines}
);

blocks
(
{block_lines}
);

edges
(
);

boundary
(
{patch_block("source")}
{patch_block("sink")}
);

mergePatchPairs
(
);
"""


def heatsink_temperature_field(params: D1OpenFOAMHeatSinkParams) -> str:
    sink_value_fraction = float(np.clip(params.sink_value_fraction, 0.0, 1.0))
    if sink_value_fraction >= 0.999:
        sink_bc = f"""
    sink
    {{
        type            fixedValue;
        value           uniform {params.sink_temperature:.9g};
    }}"""
    else:
        sink_bc = f"""
    sink
    {{
        type            mixed;
        refValue        uniform {params.sink_temperature:.9g};
        refGradient     uniform 0;
        valueFraction   uniform {sink_value_fraction:.9g};
        value           uniform {params.sink_temperature:.9g};
    }}"""
    return f"""
{foam_header(class_name="volScalarField", object_name="T", location="0")}
dimensions      [0 0 0 1 0 0 0];

internalField   uniform {params.initial_temperature:.9g};

boundaryField
{{
    source
    {{
        type            fixedValue;
        value           uniform {params.source_temperature:.9g};
    }}
{sink_bc}
}}
"""


def write_openfoam_heatsink_case(
    case_dir: Path,
    params: D1OpenFOAMHeatSinkParams,
    *,
    overwrite: bool = False,
) -> None:
    if case_dir.exists() and overwrite:
        shutil.rmtree(case_dir)
    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError(f"OpenFOAM case already exists: {case_dir}")

    write_text(case_dir / "system" / "blockMeshDict", heatsink_block_mesh_dict(params))
    write_text(case_dir / "system" / "controlDict", control_dict())
    write_text(case_dir / "system" / "fvSchemes", fv_schemes())
    write_text(case_dir / "system" / "fvSolution", fv_solution())
    write_text(case_dir / "constant" / "physicalProperties", physical_properties(params))  # type: ignore[arg-type]
    write_text(case_dir / "0" / "T", heatsink_temperature_field(params))
    write_text(case_dir / "case_params.json", json.dumps(asdict(params), indent=2))


def _distance_to_face(points: np.ndarray, face: _BoundaryFace) -> np.ndarray:
    axis = face.axis
    other_axes = [idx for idx in range(3) if idx != axis]
    d2 = (points[:, axis] - face.value) ** 2
    for range_index, other_axis in enumerate(other_axes):
        lo, hi = face.ranges[range_index]
        below = np.maximum(lo - points[:, other_axis], 0.0)
        above = np.maximum(points[:, other_axis] - hi, 0.0)
        d2 = d2 + below**2 + above**2
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def heatsink_boundary_features(
    points: np.ndarray,
    params: D1OpenFOAMHeatSinkParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    _, _, faces, _ = _heatsink_mesh(params)
    nearest = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    source_distance = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    sink_distance = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    nearest_normal = np.zeros((pts.shape[0], 3), dtype=np.float32)

    for face in faces:
        distance = _distance_to_face(pts, face)
        update = distance < nearest
        nearest[update] = distance[update]
        nearest_normal[update] = np.asarray(face.normal, dtype=np.float32)
        if face.patch == "source":
            source_distance = np.minimum(source_distance, distance)
        else:
            sink_distance = np.minimum(sink_distance, distance)

    span = max(params.width, params.depth, params.base_height + params.feature_height)
    scale = max(span / max(max(params.cells_x, params.cells_y), 8), 1e-6)
    source_patch = np.exp(-source_distance[:, None] / (2.0 * scale)).astype(np.float32)
    sink_patch = np.exp(-sink_distance[:, None] / (2.0 * scale)).astype(np.float32)
    return nearest.astype(np.float32), nearest_normal, source_patch, sink_patch


def convert_heatsink_case_to_npz(
    case_dir: Path,
    params: D1OpenFOAMHeatSinkParams,
    output_path: Path,
) -> dict[str, Any]:
    latest = latest_time_dir(case_dir)
    centers_path = latest / "C"
    temperature_path = latest / "T"
    if not centers_path.exists():
        raise FileNotFoundError(f"Missing cell centres field: {centers_path}")
    if not temperature_path.exists():
        raise FileNotFoundError(f"Missing temperature field: {temperature_path}")

    points = parse_internal_field(centers_path, components=3)
    temperature = parse_internal_field(temperature_path, components=1, expected_count=points.shape[0])
    if points.shape[0] != temperature.shape[0]:
        raise ValueError(f"Point/temperature count mismatch: {points.shape[0]} vs {temperature.shape[0]}")

    nearest_distance, nearest_normal, source_patch, sink_patch = heatsink_boundary_features(points, params)
    conditions = np.concatenate(
        [
            np.full((points.shape[0], 1), params.conductivity, dtype=np.float32),
            np.full((points.shape[0], 1), params.source_temperature, dtype=np.float32),
            np.full((points.shape[0], 1), params.sink_temperature, dtype=np.float32),
            source_patch,
            sink_patch,
            nearest_distance[:, None].astype(np.float32),
        ],
        axis=1,
    )
    material = np.concatenate(
        [
            np.full((points.shape[0], 1), params.conductivity, dtype=np.float32),
            np.full((points.shape[0], 1), params.diffusivity, dtype=np.float32),
        ],
        axis=1,
    )
    bc_features = np.concatenate(
        [
            source_patch,
            sink_patch,
            np.full((points.shape[0], 1), params.source_temperature, dtype=np.float32),
            np.full((points.shape[0], 1), params.sink_temperature, dtype=np.float32),
            nearest_distance[:, None].astype(np.float32),
        ],
        axis=1,
    )
    condition_names = np.asarray(
        [
            "conductivity",
            "source_temperature",
            "sink_temperature",
            "source_patch",
            "sink_patch",
            "nearest_boundary_distance",
        ]
    )
    case_params = asdict(params)
    case_params_json = json.dumps(case_params, sort_keys=True)
    source_center = np.asarray([0.5 * params.width, 0.5 * params.depth, 0.0], dtype=np.float32)
    sink_center = np.asarray(
        [0.5 * params.width, 0.5 * params.depth, params.base_height + params.feature_height],
        dtype=np.float32,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        normals=nearest_normal.astype(np.float32),
        region=np.zeros(points.shape[0], dtype=np.int32),
        material=material.astype(np.float32),
        bc_features=bc_features.astype(np.float32),
        T=temperature.astype(np.float32),
        temperature=temperature.astype(np.float32),
        conditions=conditions.astype(np.float32),
        condition_names=condition_names,
        source_patch=source_patch.astype(np.float32),
        sink_patch=sink_patch.astype(np.float32),
        nearest_boundary_distance=nearest_distance.astype(np.float32),
        nearest_boundary_normal=nearest_normal.astype(np.float32),
        source_center=source_center,
        sink_center=sink_center,
        case_params_json=np.asarray(case_params_json),
        solver=np.asarray("OpenFOAM-13 laplacianFoam"),
        generator=np.asarray("d1_openfoam_heatsink_blockmesh_v1"),
    )
    return {
        "case": params.case_id,
        "path": str(output_path),
        "raw_case_dir": str(case_dir),
        "solver": "OpenFOAM-13 laplacianFoam",
        "generator": "d1_openfoam_heatsink_blockmesh_v1",
        "family": params.family,
        "bc_model": params.bc_model,
        "points": int(points.shape[0]),
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "temperature_mean": float(np.mean(temperature)),
        **case_params,
    }


def sample_heatsink_params(
    *,
    case_id: str,
    family: str,
    rng: np.random.Generator,
    cells_xy: tuple[int, int],
    base_cells_z: int,
    feature_cells_z: int,
    source_temperature_range: tuple[float, float] = (360.0, 430.0),
    sink_temperature_range: tuple[float, float] = (280.0, 310.0),
    sink_value_fraction: float = 1.0,
) -> D1OpenFOAMHeatSinkParams:
    if family not in {"plate_fin", "pin_fin", "staggered_pin_fin"}:
        raise ValueError("family must be one of: plate_fin, pin_fin, staggered_pin_fin")

    width = float(rng.uniform(0.8, 1.4))
    depth = float(rng.uniform(0.8, 1.4))
    base_height = float(rng.uniform(0.08, 0.18))
    feature_height = float(rng.uniform(0.25, 0.85))
    if family == "plate_fin":
        feature_count_x = int(rng.integers(4, 11))
        feature_count_y = 1
        feature_width = float(rng.uniform(0.025, 0.07))
        feature_depth = depth * float(rng.uniform(0.82, 0.96))
    else:
        if family == "staggered_pin_fin":
            feature_count_x = int(rng.integers(5, 10))
            feature_count_y = int(rng.integers(4, 8))
        else:
            feature_count_x = int(rng.integers(3, 7))
            feature_count_y = int(rng.integers(3, 7))
        pitch_x = 0.72 * width / max(feature_count_x - 1, 1)
        pitch_y = 0.72 * depth / max(feature_count_y - 1, 1)
        diameter = float(rng.uniform(0.04, 0.10) if family == "staggered_pin_fin" else rng.uniform(0.05, 0.12))
        diameter = min(diameter, 0.55 * pitch_x, 0.55 * pitch_y)
        feature_width = diameter
        feature_depth = diameter

    source_temperature = float(rng.uniform(*source_temperature_range))
    sink_temperature = float(rng.uniform(*sink_temperature_range))
    if source_temperature < sink_temperature + 40.0:
        source_temperature = sink_temperature + float(rng.uniform(40.0, 90.0))
    conductivity = float(np.exp(rng.uniform(np.log(0.2), np.log(25.0))))
    diffusivity = float(np.exp(rng.uniform(np.log(1.0e-6), np.log(1.0e-4))))
    initial_temperature = 0.5 * (source_temperature + sink_temperature)
    return D1OpenFOAMHeatSinkParams(
        case_id=case_id,
        family=family,
        width=width,
        depth=depth,
        base_height=base_height,
        feature_height=feature_height,
        feature_count_x=feature_count_x,
        feature_count_y=feature_count_y,
        feature_width=feature_width,
        feature_depth=feature_depth,
        cells_x=int(cells_xy[0]),
        cells_y=int(cells_xy[1]),
        base_cells_z=int(base_cells_z),
        feature_cells_z=int(feature_cells_z),
        diffusivity=diffusivity,
        conductivity=conductivity,
        source_temperature=source_temperature,
        sink_temperature=sink_temperature,
        initial_temperature=float(initial_temperature),
        sink_value_fraction=float(sink_value_fraction),
        bc_model="fixed_hot_base_mixed_cooling_exterior"
        if float(sink_value_fraction) < 0.999
        else "fixed_hot_base_cold_exterior",
    )
