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
PRETRAIN_DYNAMICS_FEATURE_NAMES = [
    "brownian_delta_1_x",
    "brownian_delta_1_y",
    "brownian_delta_1_z",
    "brownian_delta_final_x",
    "brownian_delta_final_y",
    "brownian_delta_final_z",
    "boundary_hit",
    "hit_step_norm",
]
PRETRAIN_ABLATIONS = ("full", "no_boundary_field", "static_tdf_only", "dynamics_lifted")
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
        self.target_slices: dict[str, tuple[int, int]] = {"tdf": (0, len(self.target_names))}
        if ablation == "dynamics_lifted":
            start = len(self.target_names)
            self.target_names = [*self.target_names, *PRETRAIN_DYNAMICS_FEATURE_NAMES]
            self.target_slices.update(
                {
                    "brownian_delta_1": (start, start + 3),
                    "brownian_delta_final": (start + 3, start + 6),
                    "boundary_hit": (start + 6, start + 7),
                    "hit_step_norm": (start + 7, start + 8),
                }
            )

    @staticmethod
    def _dynamics_targets(group: Any, episode: int, ids: np.ndarray) -> np.ndarray:
        if "trajectory" not in group or "hit_mask" not in group or "hit_step" not in group:
            raise KeyError(
                "dynamics_lifted pretraining requires trajectory, hit_mask, and hit_step arrays in each shard."
            )
        trajectory = np.asarray(group["trajectory"][episode], dtype=np.float32)
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory shape (points, steps+1, 3), got {trajectory.shape}")
        if trajectory.shape[1] < 2:
            raise ValueError("dynamics_lifted pretraining requires at least one Brownian step.")

        start = trajectory[:, 0, :]
        delta_1 = trajectory[:, 1, :] - start
        delta_final = trajectory[:, -1, :] - start
        hit_mask = np.asarray(group["hit_mask"][episode], dtype=np.float32).reshape(-1, 1)
        hit_step = np.asarray(group["hit_step"][episode], dtype=np.float32).reshape(-1, 1)
        max_step = float(max(trajectory.shape[1] - 1, 1))
        miss_step = max_step + 1.0
        hit_step_norm = np.where(hit_step >= 0.0, hit_step, miss_step) / miss_step
        target = np.concatenate([delta_1, delta_final, hit_mask, hit_step_norm.astype(np.float32)], axis=1)
        return target[ids].astype(np.float32)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import zarr

        ref = self.refs[index]
        group = zarr.open_group(str(ref.shard_path), mode="r")
        episode = ref.episode_index
        x = np.asarray(group["x"][episode], dtype=np.float32)
        cond = np.asarray(group["cond"][episode], dtype=np.float32)
        if self.condition_mode == "zero_boundary_field":
            for field_name in ("q_near", "boundary_field", "heat_flux_near"):
                if field_name in self.condition_names:
                    cond[:, self.condition_names.index(field_name)] = 0.0
        elif self.condition_mode == "zero_all":
            cond = np.zeros_like(cond)
        ids = sample_indices(x.shape[0], self.point_budget, seed=self.seed + index)
        y = np.asarray(group["y_tdf"][episode], dtype=np.float32)[:, self.target_indices][ids]
        if self.ablation == "dynamics_lifted":
            dynamics = self._dynamics_targets(group, episode, ids)
            y = np.concatenate([y, dynamics], axis=1)
        return {
            "x": torch.from_numpy(x[ids]),
            "fx": torch.from_numpy(cond[ids]),
            "y": torch.from_numpy(y),
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
    """D1 NPZ case dataset.

    The class name is kept for compatibility with earlier proxy runs, but the
    loader only requires the common downstream keys used by both proxy and
    solver-backed D1 cases: ``points``, ``conditions``, and ``temperature``.
    """

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
