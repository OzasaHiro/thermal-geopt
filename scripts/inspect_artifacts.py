#!/usr/bin/env python3
"""Inspect generated NPZ, manifest, and Zarr artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format


DEFAULT_MAX_CHECK_ELEMENTS = 1_000_000
READ_CHUNK_BYTES = 8 * 1024 * 1024


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def element_count(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(math.prod(shape))


def finite_summary(values: np.ndarray, *, total_elements: int, checked_all: bool) -> dict[str, Any]:
    checked_elements = int(values.size)
    summary: dict[str, Any] = {
        "checked_elements": checked_elements,
        "total_elements": int(total_elements),
        "checked_all": bool(checked_all),
    }

    if values.dtype.hasobject or values.dtype.fields is not None:
        summary.update({"status": "skipped", "reason": "non_numeric_dtype"})
        return summary

    kind = values.dtype.kind
    if kind in {"b", "i", "u"}:
        summary.update(
            {
                "status": "ok",
                "all_finite": True,
                "nan_count": 0,
                "inf_count": 0,
            }
        )
        if checked_elements:
            summary["min"] = values.min().item()
            summary["max"] = values.max().item()
        return summary

    if kind not in {"f", "c"}:
        summary.update({"status": "skipped", "reason": f"unsupported_dtype_kind:{kind}"})
        return summary

    finite_mask = np.isfinite(values)
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    summary.update(
        {
            "status": "ok",
            "all_finite": bool(finite_mask.all()) if checked_elements else True,
            "nan_count": int(nan_mask.sum()),
            "inf_count": int(inf_mask.sum()),
        }
    )
    if checked_elements and kind == "f" and bool(finite_mask.any()):
        finite_values = values[finite_mask]
        summary["finite_min"] = finite_values.min().item()
        summary["finite_max"] = finite_values.max().item()
    if checked_elements and kind == "c":
        magnitudes = np.abs(values[finite_mask])
        if magnitudes.size:
            summary["finite_abs_min"] = magnitudes.min().item()
            summary["finite_abs_max"] = magnitudes.max().item()
    return summary


def read_npy_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = npy_format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported .npy version in NPZ member: {version}")
    return tuple(int(dim) for dim in shape), bool(fortran_order), np.dtype(dtype)


def stream_npz_finite_check(
    archive: zipfile.ZipFile,
    member_name: str,
    *,
    dtype: np.dtype[Any],
    total_elements: int,
    max_check_elements: int,
) -> dict[str, Any]:
    if dtype.hasobject or dtype.fields is not None:
        return {
            "status": "skipped",
            "reason": "non_numeric_dtype",
            "checked_elements": 0,
            "total_elements": int(total_elements),
            "checked_all": False,
        }
    if dtype.kind not in {"b", "i", "u", "f", "c"}:
        return {
            "status": "skipped",
            "reason": f"unsupported_dtype_kind:{dtype.kind}",
            "checked_elements": 0,
            "total_elements": int(total_elements),
            "checked_all": False,
        }

    target_elements = min(total_elements, max(0, max_check_elements))
    if target_elements == 0:
        return finite_summary(np.empty((0,), dtype=dtype), total_elements=total_elements, checked_all=total_elements == 0)

    bytes_per_element = int(dtype.itemsize)
    target_bytes = target_elements * bytes_per_element
    chunks: list[np.ndarray[Any, Any]] = []
    bytes_read = 0

    with archive.open(member_name, "r") as handle:
        read_npy_header(handle)
        while bytes_read < target_bytes:
            next_size = min(READ_CHUNK_BYTES, target_bytes - bytes_read)
            next_size -= next_size % bytes_per_element
            if next_size <= 0:
                break
            data = handle.read(next_size)
            if not data:
                break
            usable = len(data) - (len(data) % bytes_per_element)
            if usable:
                chunks.append(np.frombuffer(data[:usable], dtype=dtype))
                bytes_read += usable
            if usable < len(data):
                break

    if chunks:
        checked = np.concatenate(chunks)
    else:
        checked = np.empty((0,), dtype=dtype)
    if checked.size > target_elements:
        checked = checked[:target_elements]
    return finite_summary(
        checked,
        total_elements=total_elements,
        checked_all=int(checked.size) == total_elements,
    )


def inspect_npz(path: Path, *, max_check_elements: int) -> dict[str, Any]:
    arrays: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as archive:
        members = [name for name in archive.namelist() if name.endswith(".npy")]
        for member_name in sorted(members):
            array_name = member_name[:-4]
            with archive.open(member_name, "r") as handle:
                shape, fortran_order, dtype = read_npy_header(handle)
            total_elements = element_count(shape)
            arrays.append(
                {
                    "name": array_name,
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "fortran_order": fortran_order,
                    "finite_check": stream_npz_finite_check(
                        archive,
                        member_name,
                        dtype=dtype,
                        total_elements=total_elements,
                        max_check_elements=max_check_elements,
                    ),
                }
            )

    return {
        "path": str(path),
        "type": "npz",
        "array_count": len(arrays),
        "arrays": arrays,
    }


def sample_shape(shape: tuple[int, ...], *, max_check_elements: int) -> tuple[int, ...]:
    if not shape:
        return ()
    if element_count(shape) <= max_check_elements:
        return shape

    selected_reversed: list[int] = []
    selected_count = 1
    for dim in reversed(shape):
        remaining = max(1, max_check_elements // selected_count)
        take = max(1, min(int(dim), remaining))
        selected_reversed.append(take)
        selected_count *= take
    return tuple(reversed(selected_reversed))


def inspect_zarr_array(array: Any, *, max_check_elements: int) -> dict[str, Any]:
    shape = tuple(int(dim) for dim in array.shape)
    dtype = np.dtype(array.dtype)
    total_elements = element_count(shape)
    selected_shape = sample_shape(shape, max_check_elements=max_check_elements)
    if not shape:
        values = np.asarray(array[()])
    else:
        selection = tuple(slice(0, dim) for dim in selected_shape)
        values = np.asarray(array[selection])
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "chunks": list(array.chunks) if getattr(array, "chunks", None) is not None else None,
        "finite_check": finite_summary(
            values,
            total_elements=total_elements,
            checked_all=int(values.size) == total_elements,
        ),
    }


def inspect_zarr(path: Path, *, max_check_elements: int) -> dict[str, Any]:
    try:
        import zarr
        from zarr.errors import ZarrUserWarning
    except ImportError as exc:
        raise RuntimeError("zarr is required to inspect .zarr directories") from exc

    group = zarr.open_group(str(path), mode="r")
    arrays: list[dict[str, Any]] = []

    def walk(current: Any, prefix: str = "") -> None:
        for name, child in sorted(current.members(), key=lambda item: item[0]):
            child_name = f"{prefix}/{name}" if prefix else str(name)
            if hasattr(child, "shape") and hasattr(child, "dtype"):
                record = {"name": child_name}
                record.update(inspect_zarr_array(child, max_check_elements=max_check_elements))
                arrays.append(record)
            else:
                walk(child, child_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        walk(group)
    return {
        "path": str(path),
        "type": "zarr",
        "array_count": len(arrays),
        "arrays": arrays,
    }


def manifest_references(manifest: dict[str, Any]) -> list[str]:
    references: list[str] = []
    reference_keys = ("output_path", "path", "npz_path", "processed_path", "shard")
    for record in manifest.get("records", []):
        if isinstance(record, dict):
            for key in reference_keys:
                value = record.get(key)
                if isinstance(value, str):
                    references.append(value)
                    break
    for shard in manifest.get("shards", []):
        if isinstance(shard, dict):
            for key in reference_keys:
                value = shard.get(key)
                if isinstance(value, str):
                    references.append(value)
                    break
    return references


def resolve_reference(reference: str, *, manifest_path: Path) -> Path:
    reference_path = Path(reference)
    if reference_path.is_absolute():
        return reference_path

    cwd_candidate = Path.cwd() / reference_path
    if cwd_candidate.exists():
        return cwd_candidate

    manifest_candidate = manifest_path.parent / reference_path
    if manifest_candidate.exists():
        return manifest_candidate

    return cwd_candidate


def has_error(summary: dict[str, Any]) -> bool:
    if summary.get("status") == "error":
        return True
    references = summary.get("references", [])
    if isinstance(references, list):
        return any(isinstance(reference, dict) and has_error(reference) for reference in references)
    return False


def inspect_manifest(
    path: Path,
    *,
    max_check_elements: int,
    seen_paths: set[Path],
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    references = manifest_references(manifest)
    inspected_references = []
    for reference in references:
        reference_path = resolve_reference(reference, manifest_path=path)
        inspected_references.append(
            inspect_path(
                reference_path,
                max_check_elements=max_check_elements,
                seen_paths=seen_paths,
                source_reference=reference,
            )
        )

    summary: dict[str, Any] = {
        "path": str(path),
        "type": "manifest",
        "top_level_keys": sorted(str(key) for key in manifest.keys()),
        "reference_count": len(references),
        "reference_error_count": sum(1 for reference in inspected_references if has_error(reference)),
        "references": inspected_references,
    }
    if isinstance(manifest.get("records"), list):
        summary["record_count"] = len(manifest["records"])
    if isinstance(manifest.get("shards"), list):
        summary["shard_count"] = len(manifest["shards"])
    return summary


def inspect_path(
    path: Path,
    *,
    max_check_elements: int,
    seen_paths: set[Path],
    source_reference: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen_paths:
        return {
            "path": str(path),
            "source_reference": source_reference,
            "status": "skipped",
            "reason": "already_inspected",
        }
    seen_paths.add(resolved)

    try:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.is_file() and path.suffix == ".npz":
            summary = inspect_npz(path, max_check_elements=max_check_elements)
        elif path.is_file() and path.name == "manifest.json":
            summary = inspect_manifest(path, max_check_elements=max_check_elements, seen_paths=seen_paths)
        elif path.is_dir() and path.name.endswith(".zarr"):
            summary = inspect_zarr(path, max_check_elements=max_check_elements)
        else:
            raise ValueError("supported inputs are .npz files, manifest.json files, and .zarr directories")
        summary["status"] = "ok"
        if summary.get("reference_error_count", 0):
            summary["status"] = "error"
            summary["error"] = f"manifest_reference_errors:{summary['reference_error_count']}"
    except Exception as exc:  # noqa: BLE001 - inspection should report all per-path failures as JSON.
        summary = {
            "path": str(path),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if source_reference is not None:
        summary["source_reference"] = source_reference
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="NPZ files, manifest.json files, or .zarr directories.")
    parser.add_argument(
        "--max-check-elements",
        type=int,
        default=DEFAULT_MAX_CHECK_ELEMENTS,
        help="Maximum values to sample per array for finite checks.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [
        inspect_path(path, max_check_elements=args.max_check_elements, seen_paths=set())
        for path in args.paths
    ]
    print(json.dumps({"artifacts": summaries}, indent=args.indent, sort_keys=True, default=json_default))
    return 1 if any(summary.get("status") == "error" for summary in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
