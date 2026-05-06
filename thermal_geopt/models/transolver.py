"""GeoPT Transolver model helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


def import_geopt_transolver(vendor_root: Path):
    vendor_path = str(vendor_root.resolve())
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    from models.Transolver import Model  # noqa: PLC0415

    return Model


def build_transolver_args(
    *,
    fun_dim: int,
    out_dim: int,
    n_hidden: int = 256,
    n_layers: int = 8,
    n_heads: int = 8,
    mlp_ratio: int = 2,
    slice_num: int = 32,
    dropout: float = 0.0,
    checkpoint: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        fun_dim=fun_dim,
        space_dim=3,
        out_dim=out_dim,
        n_hidden=n_hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        slice_num=slice_num,
        dropout=dropout,
        checkpoint=checkpoint,
        geotype="unstructured",
        shapelist=None,
        unified_pos=False,
        act="gelu",
    )


def create_transolver_model(*, vendor_root: Path, model_config: dict[str, Any]) -> torch.nn.Module:
    model_cls = import_geopt_transolver(vendor_root)
    args = build_transolver_args(**model_config)
    return model_cls(args)


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise ValueError("Could not find a model state dict in checkpoint payload")


def load_matching_state(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint_state_dict(payload)
    target_state = model.state_dict()
    matched = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    skipped = sorted(key for key in source_state if key not in matched)
    missing = sorted(key for key in target_state if key not in matched)
    model.load_state_dict(matched, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded_tensors": len(matched),
        "skipped_tensors": len(skipped),
        "missing_tensors": len(missing),
        "skipped_tensor_names": skipped[:50],
        "missing_tensor_names": missing[:50],
    }
