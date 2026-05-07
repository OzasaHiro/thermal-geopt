#!/usr/bin/env python3
"""Fine-tune on D1 solid-conduction NPZ cases."""

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

from thermal_geopt.datasets import D1ProxyDataset
from thermal_geopt.models.transolver import create_transolver_model, load_matching_state
from thermal_geopt.training import (
    max_value_error_torch,
    relative_l2_torch,
    resolve_device,
    rmse_torch,
    save_checkpoint,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=Path("data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json"))
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-model-dir", type=Path)
    parser.add_argument("--pretrained-checkpoint-file", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--point-budget", type=int, default=2048)
    parser.add_argument("--eval-point-budget", type=int, default=2048)
    parser.add_argument("--max-train-cases", type=int, default=64)
    parser.add_argument("--max-val-cases", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
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


def train_stats(dataset: D1ProxyDataset) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    target_values = []
    feature_values = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target_values.append(sample["y"].reshape(-1))  # type: ignore[index,union-attr]
        feature_values.append(sample["fx"].reshape(-1, sample["fx"].shape[-1]))  # type: ignore[index,union-attr]
    y = torch.cat(target_values).float()
    fx = torch.cat(feature_values).float()
    mean = float(y.mean())
    std = float(torch.clamp(y.std(unbiased=False), min=1e-6))
    feature_mean = fx.mean(dim=0)
    feature_std = torch.clamp(fx.std(dim=0, unbiased=False), min=1e-6)
    return mean, std, feature_mean, feature_std


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    target_mean: float,
    target_std: float,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    rel_l2_values = []
    rmse_values = []
    max_error_values = []
    for batch in loader:
        x = batch["x"].to(device=device, dtype=torch.float32)
        fx = batch["fx"].to(device=device, dtype=torch.float32)
        y = batch["y"].to(device=device, dtype=torch.float32)
        fx_norm = (fx - feature_mean.view(1, 1, -1)) / feature_std.view(1, 1, -1)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred_norm = model(x, fx_norm)
        pred = pred_norm.float() * target_std + target_mean
        rel_l2_values.append(float(relative_l2_torch(pred, y).cpu()))
        rmse_values.append(float(rmse_torch(pred, y).cpu()))
        max_error_values.append(float(max_value_error_torch(pred, y).cpu()))
    return {
        "relative_l2_mean": float(sum(rel_l2_values) / max(len(rel_l2_values), 1)),
        "rmse_mean": float(sum(rmse_values) / max(len(rmse_values), 1)),
        "max_value_error_mean": float(sum(max_error_values) / max(len(max_error_values), 1)),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_dataset = D1ProxyDataset(
        args.case_manifest,
        split_path=args.split_path,
        split=args.train_split if args.split_path else "all",
        point_budget=args.point_budget,
        max_cases=args.max_train_cases,
        seed=args.seed,
    )
    val_dataset = D1ProxyDataset(
        args.case_manifest,
        split_path=args.split_path,
        split=args.val_split if args.split_path else "all",
        point_budget=args.eval_point_budget,
        max_cases=args.max_val_cases,
        seed=args.seed + 10_000,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    target_mean, target_std, feature_mean_cpu, feature_std_cpu = train_stats(train_dataset)
    feature_mean = feature_mean_cpu.to(device=device, dtype=torch.float32)
    feature_std = feature_std_cpu.to(device=device, dtype=torch.float32)
    model_config = {
        "fun_dim": train_dataset.fun_dim,
        "out_dim": train_dataset.out_dim,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "mlp_ratio": 2,
        "slice_num": args.slice_num,
        "dropout": 0.0,
        "checkpoint": 0,
    }
    model = create_transolver_model(vendor_root=args.geopt_vendor, model_config=model_config).to(device)
    init_report = None
    if args.pretrained_model_dir is not None:
        init_report = load_matching_state(
            model,
            args.pretrained_model_dir / args.pretrained_checkpoint_file,
            device="cpu",
        )
        print({"pretrained_load": init_report})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    config = {
        "task": "d1_finetune",
        "case_manifest": str(args.case_manifest),
        "split_path": str(args.split_path) if args.split_path else None,
        "train_split": args.train_split if args.split_path else "all",
        "val_split": args.val_split if args.split_path else "all",
        "pretrained_model_dir": str(args.pretrained_model_dir) if args.pretrained_model_dir else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "point_budget": args.point_budget,
        "eval_point_budget": args.eval_point_budget,
        "max_train_cases": args.max_train_cases,
        "max_val_cases": args.max_val_cases,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "target_mean": target_mean,
        "target_std": target_std,
        "normalization": {
            "target_mean": target_mean,
            "target_std": target_std,
            "feature_mean": [float(value) for value in feature_mean_cpu.tolist()],
            "feature_std": [float(value) for value in feature_std_cpu.tolist()],
        },
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "model": model_config,
        "pretrained_load": init_report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config)

    history = []
    best_score = float("inf")
    best_metrics = None
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            fx = batch["fx"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            fx_norm = (fx - feature_mean.view(1, 1, -1)) / feature_std.view(1, 1, -1)
            y_norm = (y - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model(x, fx_norm)
                loss = F.mse_loss(pred_norm, y_norm)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite D1 loss at epoch {epoch}: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_mse = float(sum(losses) / max(len(losses), 1))
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            target_mean=target_mean,
            target_std=target_std,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        metrics = {"epoch": epoch, "train_mse": train_mse, "elapsed_sec": time.time() - start_time, **val_metrics}
        history.append(metrics)
        print(metrics)
        save_checkpoint(args.output_dir / "model.pt", model=model, optimizer=optimizer, epoch=epoch, config=config, metrics=metrics)
        score = val_metrics["relative_l2_mean"]
        if score < best_score:
            best_score = score
            best_metrics = metrics
            save_checkpoint(
                args.output_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=metrics,
            )

    write_json(args.output_dir / "history.json", {"history": history})
    write_json(args.output_dir / "metrics.json", {"final": history[-1], "best": best_metrics or history[-1]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
