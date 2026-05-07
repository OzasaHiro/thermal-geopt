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
