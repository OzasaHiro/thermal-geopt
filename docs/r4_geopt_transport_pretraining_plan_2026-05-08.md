# R4 GeoPT-Transport Pretraining Plan

Date: 2026-05-08

## Decision

R4は、元GeoPTに最も近いThermal GeoPT候補として、solver-freeなsynthetic transport fieldをdynamics promptにする。

重要な制約:

- 真の温度場から計算した `grad(T)` やheat fluxは使わない。これは下流ラベル漏洩になる。
- promptは元GeoPTと同じく、geometryだけから生成できるランダム固定ベクトル場にする。
- pretrainingでは `direction_x/y/z + step_length` を各点のconditionへ追加する。
- targetはその固定場に沿ったTDF/VDF trajectoryにする。

この設計は、元GeoPTの「random velocity condition + boundary-constrained VDF trajectory」を熱伝達向けTDFへ置き換えるものであり、Brownian random walk版よりGeoPT本来の形に近い。

## Positioning

R3 `diffusion_lifted` は熱拡散らしさを持つ対照群として残す。一方、論文主張に近い本命候補はR4 `geopt_transport_lifted` とする。

比較すべき群:

- scratch
- original GeoPT pretrained model
- R3 diffusion_lifted
- R4 geopt_transport_lifted

解釈:

- R4がR3より良い: GeoPT型の固定dynamics promptがThermalでも重要。
- R3がR4より良い: 放物型拡散の確率過程的priorがより有効。
- original GeoPTがR4と同等以上: Thermal独自pretrainingより、元GeoPT汎用priorで十分な可能性。
- すべてscratch以下: prompt alignment、normalization、D1 task設計を優先して疑う。

## Downstream Alignment

元GeoPTはfine-tuning時もsimulation conditionからdirection/speed promptを作って入力へ連結している。したがって、R4のfine-tuningではD1側にも同種のpromptを追加する。

Thermal D1では、solver-freeに作れる近似として次を使う。

- direction: hot sourceからcold sinkへ向かう単位ベクトル
- step_length: temperature differenceとconductivityから作る正規化スカラー

これは正解温度場を使わないためラベル漏洩ではない。

## Commands

Tiny smoke:

```bash
PY=../../.venv/bin/python

$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_pilot_300 \
  --output-dir data/pretrain_zarr/geopt_transport_smoke_codex \
  --max-shapes 1 \
  --selection first \
  --episodes-per-shape 1 \
  --points-per-episode 128 \
  --steps 3 \
  --condition-schema d1_thermal \
  --trajectory-mode geopt_transport \
  --transport-max-step 0.12 \
  --save-trajectory-tdf \
  --trajectory-tdf-feature-set vdf_distance \
  --seed 42 \
  --overwrite

$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/geopt_transport_smoke_codex/manifest.json \
  --ablation geopt_transport_lifted

$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/geopt_transport_smoke_codex/manifest.json \
  --output-dir outputs/checkpoints/pretrain_geopt_transport_smoke_codex \
  --pretext-ablation geopt_transport_lifted \
  --epochs 1 \
  --batch-size 1 \
  --point-budget 64 \
  --max-episodes 1 \
  --normalization standardize \
  --normalization-max-episodes 1 \
  --target-min-std 0.05 \
  --tdf-loss-weight 0.1 \
  --diffusion-loss-weight 1.0 \
  --trajectory-tdf-loss-weight 1.0 \
  --device cpu
```

P2 data generation:

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_p2_2100 \
  --output-dir data/pretrain_zarr/cadquery_p2_d1_thermal_geopt_transport_2000_e20_n8192 \
  --max-shapes 2000 \
  --selection balanced \
  --episodes-per-shape 20 \
  --points-per-episode 8192 \
  --steps 3 \
  --condition-schema d1_thermal \
  --trajectory-mode geopt_transport \
  --transport-max-step 0.12 \
  --save-trajectory-tdf \
  --trajectory-tdf-feature-set vdf_distance \
  --seed 42 \
  --overwrite
```

P2 pretraining:

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_geopt_transport_2000_e20_n8192/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r4_geopt_transport_p2_norm_val_ep100_wcos \
  --epochs 100 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --val-fraction 0.05 \
  --normalization standardize \
  --normalization-max-episodes 2048 \
  --target-min-std 0.05 \
  --pretext-ablation geopt_transport_lifted \
  --tdf-loss-weight 0.1 \
  --diffusion-loss-weight 1.0 \
  --trajectory-tdf-loss-weight 1.0 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --scheduler warmup_cosine \
  --warmup-ratio 0.05 \
  --min-lr-scale 0.01 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda
```

Aligned M3 transfer gate:

```bash
PY=../../.venv/bin/python \
PRETRAIN_GEOPT_TRANSPORT=outputs/checkpoints/pretrain_r4_geopt_transport_p2_norm_val_ep100_wcos \
NORMALIZATION_PROTOCOL=pretrained \
NORMALIZATION_CONFIG=outputs/checkpoints/pretrain_r4_geopt_transport_p2_norm_val_ep100_wcos/config.json \
CONDITION_AUGMENTATION=thermal_transport \
RUN_PREFIX=m3_openfoam_p2_r4_geopt_transport_oclr \
EPOCHS=100 \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
TRAIN_SEEDS="42" \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
GATE_GROUPS="scratch geopt_transport_lifted" \
FINETUNE_SCHEDULER=onecycle \
PRETRAINED_BACKBONE_LR=3e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
MAX_GRAD_NORM=1.0 \
MODE=all \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```
