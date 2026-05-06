#!/usr/bin/env python3
"""Check the local Thermal GeoPT runtime environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def package_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(name)
        except Exception:
            version = "unknown"
    return {"available": True, "version": str(version)}


def transolver_smoke(args: argparse.Namespace) -> dict[str, object]:
    import torch

    vendor = args.geopt_vendor.resolve()
    if not vendor.exists():
        return {"ok": False, "error": f"GeoPT vendor path not found: {vendor}"}
    sys.path.insert(0, str(vendor))
    try:
        from models.Transolver import Model
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else args.device)
    model_args = SimpleNamespace(
        fun_dim=args.fun_dim,
        space_dim=3,
        out_dim=args.out_dim,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=2,
        slice_num=args.slice_num,
        dropout=0.0,
        checkpoint=0,
        geotype="unstructured",
        shapelist=None,
        unified_pos=False,
        act="gelu",
    )
    model = Model(model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x = torch.randn(1, args.points, 3, device=device)
    fx = torch.randn(1, args.points, args.fun_dim, device=device)
    target = torch.randn(1, args.points, args.out_dim, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast_enabled = args.amp and device.type == "cuda"
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        pred = model(x, fx)
        loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    peak_bytes = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    return {
        "ok": True,
        "device": str(device),
        "amp": autocast_enabled,
        "amp_dtype": args.amp_dtype if autocast_enabled else None,
        "points": args.points,
        "output_shape": list(pred.shape),
        "loss": float(loss.detach().cpu()),
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "cuda_peak_memory_reserved_mb": round(peak_bytes / 1024**2, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transolver-smoke", action="store_true")
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--fun-dim", type=int, default=11)
    parser.add_argument("--out-dim", type=int, default=8)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--geopt-vendor",
        type=Path,
        default=Path("../GeoPT/vendor/GeoPT"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "packages": {
            name: package_status(name)
            for name in ("numpy", "scipy", "torch", "pyvista", "trimesh", "open3d", "zarr", "cadquery")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
    }
    if torch.cuda.is_available():
        report["torch"]["device_name"] = torch.cuda.get_device_name(0)
        report["torch"]["bf16_supported"] = torch.cuda.is_bf16_supported()
    if args.transolver_smoke:
        report["transolver_smoke"] = transolver_smoke(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    smoke = report.get("transolver_smoke")
    if isinstance(smoke, dict) and not smoke.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
