#!/usr/bin/env python3
"""Pilot pretraining for Thermal GeoPT self-supervised targets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.datasets import PretrainZarrDataset
from thermal_geopt.models.transolver import create_transolver_model
from thermal_geopt.normalization import estimate_dataset_normalization, normalize_tensor, vector_from_stats
from thermal_geopt.training import resolve_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--point-budget", type=int, default=2048)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--normalization", choices=["none", "standardize"], default="none")
    parser.add_argument("--normalization-max-episodes", type=int, default=512)
    parser.add_argument(
        "--target-min-std",
        type=float,
        default=0.05,
        help="Minimum target std when --normalization standardize is used; prevents rare binary targets from exploding.",
    )
    parser.add_argument("--best-metric", choices=["auto", "train_loss", "val_loss"], default="auto")
    parser.add_argument(
        "--pretext-ablation",
        choices=["full", "no_boundary_field", "static_tdf_only", "dynamics_lifted"],
        default="full",
        help="Select the pretraining target/prompt ablation without regenerating Zarr shards.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=["auto", "full", "zero_boundary_field", "zero_all"],
        default="auto",
        help="Override prompt handling. auto maps from --pretext-ablation.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--tdf-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=1.0)
    parser.add_argument("--hit-mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--hit-step-loss-weight", type=float, default=1.0)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geopt-vendor", type=Path, default=Path("../GeoPT/vendor/GeoPT"))
    return parser.parse_args()


def split_pretrain_dataset(
    dataset: PretrainZarrDataset,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset | None, dict[str, object]]:
    if val_fraction <= 0.0:
        indices = list(range(len(dataset)))
        return Subset(dataset, indices), None, {
            "mode": "none",
            "train_count": len(indices),
            "val_count": 0,
        }
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1).")

    refs = list(dataset.refs)
    shard_keys = sorted({str(ref.shard_path) for ref in refs})
    rng = torch.Generator().manual_seed(seed)
    if len(shard_keys) > 1:
        order = torch.randperm(len(shard_keys), generator=rng).tolist()
        val_shard_count = max(1, min(len(shard_keys) - 1, round(len(shard_keys) * val_fraction)))
        val_shards = {shard_keys[index] for index in order[:val_shard_count]}
        train_indices = [index for index, ref in enumerate(refs) if str(ref.shard_path) not in val_shards]
        val_indices = [index for index, ref in enumerate(refs) if str(ref.shard_path) in val_shards]
        mode = "shard_holdout"
    else:
        order = torch.randperm(len(dataset), generator=rng).tolist()
        val_count = max(1, min(len(dataset) - 1, round(len(dataset) * val_fraction)))
        val_indices = sorted(order[:val_count])
        train_indices = sorted(order[val_count:])
        val_shards = set(shard_keys)
        mode = "episode_holdout"

    if not train_indices or not val_indices:
        raise ValueError(
            f"Invalid pretrain validation split: train_count={len(train_indices)}, val_count={len(val_indices)}"
        )
    return Subset(dataset, train_indices), Subset(dataset, val_indices), {
        "mode": mode,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "val_shards": sorted(val_shards)[:20],
        "val_shard_count": len(val_shards),
    }


def _slice_loss(pred: torch.Tensor, target: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return F.mse_loss(pred[..., start:end], target[..., start:end])


def pretrain_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    target_slices: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if "brownian_delta_1" not in target_slices:
        loss = F.mse_loss(pred, target)
        return loss, {"mse": float(loss.detach().cpu())}

    tdf_start, tdf_end = target_slices["tdf"]
    delta_1_start, delta_1_end = target_slices["brownian_delta_1"]
    delta_final_start, delta_final_end = target_slices["brownian_delta_final"]
    hit_start, hit_end = target_slices["boundary_hit"]
    hit_step_start, hit_step_end = target_slices["hit_step_norm"]

    tdf = _slice_loss(pred, target, tdf_start, tdf_end)
    delta_1 = _slice_loss(pred, target, delta_1_start, delta_1_end)
    delta_final = _slice_loss(pred, target, delta_final_start, delta_final_end)
    hit = _slice_loss(pred, target, hit_start, hit_end)
    hit_step = _slice_loss(pred, target, hit_step_start, hit_step_end)
    trajectory = 0.5 * (delta_1 + delta_final)
    loss = (
        args.tdf_loss_weight * tdf
        + args.trajectory_loss_weight * trajectory
        + args.hit_mask_loss_weight * hit
        + args.hit_step_loss_weight * hit_step
    )
    return loss, {
        "loss_total": float(loss.detach().cpu()),
        "loss_tdf": float(tdf.detach().cpu()),
        "loss_trajectory": float(trajectory.detach().cpu()),
        "loss_boundary_hit": float(hit.detach().cpu()),
        "loss_hit_step_norm": float(hit_step.detach().cpu()),
    }


def prepare_normalization(
    dataset: Subset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if args.normalization == "none":
        sample = dataset[0]
        x_width = int(sample["x"].shape[-1])  # type: ignore[index]
        fx_width = int(sample["fx"].shape[-1])  # type: ignore[index]
        y_width = int(sample["y"].shape[-1])  # type: ignore[index]
        payload: dict[str, object] = {
            "mode": "none",
            "coordinate": {"count": 0, "mean": [0.0] * x_width, "std": [1.0] * x_width},
            "feature": {"count": 0, "mean": [0.0] * fx_width, "std": [1.0] * fx_width},
            "target": {"count": 0, "mean": [0.0] * y_width, "std": [1.0] * y_width},
        }
    else:
        payload = estimate_dataset_normalization(
            dataset,
            max_items=args.normalization_max_episodes,
            include_target=True,
        )
        payload["mode"] = "standardize"
        target = payload["target"]  # type: ignore[index]
        target["std"] = [float(max(value, args.target_min_std)) for value in target["std"]]  # type: ignore[index]

    coordinate = payload["coordinate"]  # type: ignore[index]
    feature = payload["feature"]  # type: ignore[index]
    target = payload["target"]  # type: ignore[index]
    payload.update(
        {
            "coordinate_mean": coordinate["mean"],  # type: ignore[index]
            "coordinate_std": coordinate["std"],  # type: ignore[index]
            "x_mean": coordinate["mean"],  # type: ignore[index]
            "x_std": coordinate["std"],  # type: ignore[index]
            "feature_mean": feature["mean"],  # type: ignore[index]
            "feature_std": feature["std"],  # type: ignore[index]
            "target_mean": target["mean"],  # type: ignore[index]
            "target_std": target["std"],  # type: ignore[index]
        }
    )
    vectors = {
        "x_mean": vector_from_stats(payload, "coordinate", "mean", width=3, device=device),
        "x_std": vector_from_stats(payload, "coordinate", "std", width=3, device=device),
        "fx_mean": vector_from_stats(payload, "feature", "mean", width=len(feature["mean"]), device=device),  # type: ignore[arg-type,index]
        "fx_std": vector_from_stats(payload, "feature", "std", width=len(feature["std"]), device=device),  # type: ignore[arg-type,index]
        "y_mean": vector_from_stats(payload, "target", "mean", width=len(target["mean"]), device=device),  # type: ignore[arg-type,index]
        "y_std": vector_from_stats(payload, "target", "std", width=len(target["std"]), device=device),  # type: ignore[arg-type,index]
    }
    return payload, vectors


def normalize_batch(
    batch: dict[str, torch.Tensor],
    *,
    vectors: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    fx = batch["fx"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    x_norm = normalize_tensor(x, vectors["x_mean"], vectors["x_std"])
    fx_norm = normalize_tensor(fx, vectors["fx_mean"], vectors["fx_std"])
    y_norm = normalize_tensor(y, vectors["y_mean"], vectors["y_std"])
    return x_norm, fx_norm, y_norm


@torch.no_grad()
def evaluate_pretrain(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    dataset: PretrainZarrDataset,
    args: argparse.Namespace,
    vectors: dict[str, torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses = []
    component_sums: dict[str, float] = {}
    for batch in loader:
        x, fx, y = normalize_batch(batch, vectors=vectors, device=device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(x, fx)
            loss, components = pretrain_loss(pred, y, target_slices=dataset.target_slices, args=args)
        losses.append(float(loss.detach().cpu()))
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value
    metrics = {"val_loss": float(sum(losses) / max(len(losses), 1))}
    for key, value in sorted(component_sums.items()):
        metrics[f"val_{key}"] = float(value / max(len(losses), 1))
    return metrics


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = PretrainZarrDataset(
        args.manifest,
        point_budget=args.point_budget,
        max_episodes=args.max_episodes,
        seed=args.seed,
        ablation=args.pretext_ablation,
        condition_mode=None if args.condition_mode == "auto" else args.condition_mode,
    )
    train_dataset, val_dataset, split_report = split_pretrain_dataset(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )
    normalization, normalization_vectors = prepare_normalization(train_dataset, args=args, device=device)
    model_config = {
        "fun_dim": dataset.fun_dim,
        "out_dim": dataset.out_dim,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "mlp_ratio": 2,
        "slice_num": args.slice_num,
        "dropout": 0.0,
        "checkpoint": 0,
    }
    model = create_transolver_model(vendor_root=args.geopt_vendor, model_config=model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    best_metric_config = "val_loss" if (args.best_metric == "auto" and val_loader is not None) else args.best_metric
    if best_metric_config == "auto":
        best_metric_config = "train_loss"

    config = {
        "task": "thermal_geopt_pretrain",
        "manifest": str(args.manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "point_budget": args.point_budget,
        "max_episodes": args.max_episodes,
        "val_fraction": args.val_fraction,
        "pretrain_split": split_report,
        "pretext_ablation": args.pretext_ablation,
        "condition_mode": dataset.condition_mode,
        "condition_names": dataset.condition_names,
        "target_names": dataset.target_names,
        "target_slices": {key: list(value) for key, value in dataset.target_slices.items()},
        "normalization": normalization,
        "best_metric": best_metric_config,
        "loss_weights": {
            "tdf": args.tdf_loss_weight,
            "trajectory": args.trajectory_loss_weight,
            "hit_mask": args.hit_mask_loss_weight,
            "hit_step": args.hit_step_loss_weight,
        },
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "model": model_config,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config)

    history = []
    best_metric_name = str(config["best_metric"])
    if best_metric_name == "val_loss" and val_loader is None:
        raise ValueError("--best-metric val_loss requires --val-fraction > 0.")
    best_score = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        component_sums: dict[str, float] = {}
        for batch in train_loader:
            x, fx, y = normalize_batch(batch, vectors=normalization_vectors, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(x, fx)
                loss, components = pretrain_loss(pred, y, target_slices=dataset.target_slices, args=args)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite pretrain loss at epoch {epoch}: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
        epoch_loss = float(sum(losses) / max(len(losses), 1))
        metrics = {
            "epoch": epoch,
            "train_mse": epoch_loss,
            "train_loss": epoch_loss,
            "elapsed_sec": time.time() - start_time,
        }
        for key, value in sorted(component_sums.items()):
            metrics[key] = float(value / max(len(losses), 1))
        if val_loader is not None:
            metrics.update(
                evaluate_pretrain(
                    model,
                    val_loader,
                    dataset=dataset,
                    args=args,
                    vectors=normalization_vectors,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            )
        history.append(metrics)
        print(metrics)
        save_checkpoint(args.output_dir / "model.pt", model=model, optimizer=optimizer, epoch=epoch, config=config, metrics=metrics)
        score = float(metrics[str(best_metric_name)])
        if score < best_score:
            best_score = score
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics,
            )

    best_metrics = min(history, key=lambda item: float(item[str(best_metric_name)]))
    write_json(args.output_dir / "history.json", {"history": history})
    write_json(args.output_dir / "metrics.json", {"final": history[-1], "best": best_metrics, "best_metric": best_metric_name})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
