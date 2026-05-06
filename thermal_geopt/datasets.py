"""Datasets for Thermal GeoPT pilot training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    if base_dir is not None:
        joined = base_dir / candidate
        if joined.exists():
            return joined
    return candidate


def sample_indices(num_points: int, point_budget: int, *, seed: int) -> np.ndarray:
    if point_budget <= 0 or point_budget >= num_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=point_budget, replace=False).astype(np.int64))


DEFAULT_PRETRAIN_FEATURE_NAMES = [
    "vdf_x",
    "vdf_y",
    "vdf_z",
    "distance",
    "diffusion_time",
    "heat_kernel_t0",
    "heat_kernel_t1",
    "heat_kernel_t2",
    "resistance_distance",
    "normal_x",
    "normal_y",
    "normal_z",
    "source_proximity",
    "sink_proximity",
]
DEFAULT_PRETRAIN_CONDITION_NAMES = ["alpha", "conductivity", "q_near"]
STATIC_TDF_FEATURE_NAMES = [
    "vdf_x",
    "vdf_y",
    "vdf_z",
    "distance",
    "normal_x",
    "normal_y",
    "normal_z",
]
PRETRAIN_ABLATIONS = ("full", "no_boundary_field", "static_tdf_only")
PRETRAIN_CONDITION_MODES = ("full", "zero_boundary_field", "zero_all")


def _read_shard_meta(shard_path: Path) -> dict[str, Any]:
    meta_path = shard_path / "meta.json"
    if not meta_path.exists():
        return {}
    return read_json(meta_path)


def _indices_for_names(available: list[str], selected: list[str]) -> list[int]:
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in selected if name not in lookup]
    if missing:
        raise ValueError(f"Missing expected feature names: {missing}")
    return [lookup[name] for name in selected]


@dataclass(frozen=True)
class PretrainEpisodeRef:
    shard_path: Path
    episode_index: int


class PretrainZarrDataset(Dataset):
    """Episode dataset over generated Thermal GeoPT Zarr shards."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        point_budget: int = 0,
        max_episodes: int = 0,
        seed: int = 42,
        ablation: str = "full",
        condition_mode: str | None = None,
        validate_schema: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = read_json(manifest_path)
        self.point_budget = point_budget
        self.seed = seed
        if ablation not in PRETRAIN_ABLATIONS:
            raise ValueError(f"Unknown pretrain ablation {ablation!r}; expected one of {PRETRAIN_ABLATIONS}")
        if condition_mode is None:
            condition_mode = {
                "full": "full",
                "no_boundary_field": "zero_boundary_field",
                "static_tdf_only": "zero_all",
            }[ablation]
        if condition_mode not in PRETRAIN_CONDITION_MODES:
            raise ValueError(
                f"Unknown pretrain condition mode {condition_mode!r}; expected one of {PRETRAIN_CONDITION_MODES}"
            )
        self.ablation = ablation
        self.condition_mode = condition_mode
        refs: list[PretrainEpisodeRef] = []
        shard_paths: list[Path] = []
        for shard in self.manifest.get("shards", []):
            shard_path = resolve_existing_path(shard["shard"], base_dir=manifest_path.parent)
            shard_paths.append(shard_path)
            episodes = int(shard.get("episodes", 0))
            for episode_index in range(episodes):
                refs.append(PretrainEpisodeRef(shard_path=shard_path, episode_index=episode_index))
        if max_episodes > 0:
            refs = refs[:max_episodes]
        if not refs:
            raise ValueError(f"No pretraining episodes found in {manifest_path}")
        self.refs = refs
        meta = _read_shard_meta(refs[0].shard_path)
        self.feature_names = list(meta.get("feature_names") or DEFAULT_PRETRAIN_FEATURE_NAMES)
        self.condition_names = list(meta.get("condition_names") or DEFAULT_PRETRAIN_CONDITION_NAMES)
        if validate_schema:
            for shard_path in shard_paths[1:]:
                shard_meta = _read_shard_meta(shard_path)
                feature_names = list(shard_meta.get("feature_names") or DEFAULT_PRETRAIN_FEATURE_NAMES)
                condition_names = list(shard_meta.get("condition_names") or DEFAULT_PRETRAIN_CONDITION_NAMES)
                if feature_names != self.feature_names or condition_names != self.condition_names:
                    raise ValueError(
                        "Pretraining shard schema mismatch: "
                        f"{shard_path} has feature_names={feature_names}, condition_names={condition_names}; "
                        f"expected feature_names={self.feature_names}, condition_names={self.condition_names}"
                    )
        target_names = self.feature_names if ablation != "static_tdf_only" else STATIC_TDF_FEATURE_NAMES
        self.target_indices = _indices_for_names(self.feature_names, target_names)
        self.target_names = [self.feature_names[index] for index in self.target_indices]

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import zarr

        ref = self.refs[index]
        group = zarr.open_group(str(ref.shard_path), mode="r")
        episode = ref.episode_index
        x = np.asarray(group["x"][episode], dtype=np.float32)
        cond = np.asarray(group["cond"][episode], dtype=np.float32)
        y = np.asarray(group["y_tdf"][episode], dtype=np.float32)[:, self.target_indices]
        if self.condition_mode == "zero_boundary_field":
            for field_name in ("q_near", "boundary_field", "heat_flux_near"):
                if field_name in self.condition_names:
                    cond[:, self.condition_names.index(field_name)] = 0.0
        elif self.condition_mode == "zero_all":
            cond = np.zeros_like(cond)
        ids = sample_indices(x.shape[0], self.point_budget, seed=self.seed + index)
        return {
            "x": torch.from_numpy(x[ids]),
            "fx": torch.from_numpy(cond[ids]),
            "y": torch.from_numpy(y[ids]),
        }

    @property
    def fun_dim(self) -> int:
        sample = self[0]
        return int(sample["fx"].shape[-1])

    @property
    def out_dim(self) -> int:
        sample = self[0]
        return int(sample["y"].shape[-1])


@dataclass(frozen=True)
class D1CaseRef:
    case_id: str
    path: Path


def _record_id(record: dict[str, Any]) -> str:
    if record.get("case"):
        return str(record["case"])
    if record.get("path"):
        return Path(str(record["path"])).name
    raise ValueError(f"D1 record has no case/path field: {record}")


class D1ProxyDataset(Dataset):
    """D1 source/sink proxy case dataset."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split_path: Path | None = None,
        split: str = "all",
        point_budget: int = 0,
        max_cases: int = 0,
        seed: int = 42,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = read_json(manifest_path)
        self.point_budget = point_budget
        self.seed = seed
        allowed: set[str] | None = None
        if split_path is not None and split != "all":
            split_payload = read_json(split_path)
            split_items = split_payload.get(split)
            if not isinstance(split_items, list):
                raise ValueError(f"Split file {split_path} does not contain a list for split={split!r}")
            allowed = set(str(item) for item in split_items)

        refs: list[D1CaseRef] = []
        for record in self.manifest.get("records", []):
            case_id = _record_id(record)
            if allowed is not None and case_id not in allowed and Path(case_id).stem not in allowed:
                continue
            path = resolve_existing_path(record["path"], base_dir=manifest_path.parent)
            refs.append(D1CaseRef(case_id=case_id, path=path))

        if max_cases > 0:
            refs = refs[:max_cases]
        if not refs:
            raise ValueError(f"No D1 cases found for split={split} in {manifest_path}")
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ref = self.refs[index]
        with np.load(ref.path) as data:
            x = np.asarray(data["points"], dtype=np.float32)
            fx = np.asarray(data["conditions"], dtype=np.float32)
            y = np.asarray(data["temperature"], dtype=np.float32)
        ids = sample_indices(x.shape[0], self.point_budget, seed=self.seed + index)
        return {
            "case_id": ref.case_id,
            "x": torch.from_numpy(x[ids]),
            "fx": torch.from_numpy(fx[ids]),
            "y": torch.from_numpy(y[ids]),
        }

    @property
    def fun_dim(self) -> int:
        sample = self[0]
        return int(sample["fx"].shape[-1])  # type: ignore[index]

    @property
    def out_dim(self) -> int:
        sample = self[0]
        return int(sample["y"].shape[-1])  # type: ignore[index]
