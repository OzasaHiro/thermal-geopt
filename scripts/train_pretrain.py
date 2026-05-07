#!/usr/bin/env python3
"""Pilot pretraining for Thermal GeoPT self-supervised targets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thermal_geopt.datasets import PretrainZarrDataset
from thermal_geopt.models.transolver import create_transolver_model
from thermal_geopt.training import resolve_device, save_checkpoint, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--point-budget", type=int, default=2048)
    parser.add_argument("--max-episodes", type=int, default=64)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
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

    config = {
        "task": "thermal_geopt_pretrain",
        "manifest": str(args.manifest),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "point_budget": args.point_budget,
        "max_episodes": args.max_episodes,
        "pretext_ablation": args.pretext_ablation,
        "condition_mode": dataset.condition_mode,
        "condition_names": dataset.condition_names,
        "target_names": dataset.target_names,
        "target_slices": {key: list(value) for key, value in dataset.target_slices.items()},
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
    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        component_sums: dict[str, float] = {}
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            fx = batch["fx"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
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
        metrics = {"epoch": epoch, "train_mse": epoch_loss, "elapsed_sec": time.time() - start_time}
        for key, value in sorted(component_sums.items()):
            metrics[key] = float(value / max(len(losses), 1))
        history.append(metrics)
        print(metrics)
        save_checkpoint(args.output_dir / "model.pt", model=model, optimizer=optimizer, epoch=epoch, config=config, metrics=metrics)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics,
            )

    write_json(args.output_dir / "history.json", {"history": history})
    write_json(args.output_dir / "metrics.json", history[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
