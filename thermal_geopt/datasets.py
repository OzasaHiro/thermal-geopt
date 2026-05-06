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
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = read_json(manifest_path)
        self.point_budget = point_budget
        self.seed = seed
        refs: list[PretrainEpisodeRef] = []
        for shard in self.manifest.get("shards", []):
            shard_path = resolve_existing_path(shard["shard"], base_dir=manifest_path.parent)
            episodes = int(shard.get("episodes", 0))
            for episode_index in range(episodes):
                refs.append(PretrainEpisodeRef(shard_path=shard_path, episode_index=episode_index))
        if max_episodes > 0:
            refs = refs[:max_episodes]
        if not refs:
            raise ValueError(f"No pretraining episodes found in {manifest_path}")
        self.refs = refs

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import zarr

        ref = self.refs[index]
        group = zarr.open_group(str(ref.shard_path), mode="r")
        episode = ref.episode_index
        x = np.asarray(group["x"][episode], dtype=np.float32)
        cond = np.asarray(group["cond"][episode], dtype=np.float32)
        y = np.asarray(group["y_tdf"][episode], dtype=np.float32)
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
            allowed = set(str(item) for item in split_payload.get(split, []))

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
