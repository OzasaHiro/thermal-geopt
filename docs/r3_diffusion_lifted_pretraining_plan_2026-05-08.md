# R3 Diffusion-Lifted Pretraining Plan

Date: 2026-05-08

## Decision

R3は、D1固定温度タスクに寄せた複雑なsource/sink温度代理ターゲットではなく、GeoPTの本質に近いシンプルなdiffusion-lifted pretrainingを本命にする。

採用する最小ターゲットは次の通り。

- TDF auxiliary: VDF, distance, diffusion time, heat-kernel proximity, resistance distance, normal, source/sink proximity
- boundary_hit: Brownian diffusionが境界へ到達したか
- boundary_survival: まだ内部に残っているか
- hit_step_norm: 到達時刻の正規化値
- optional trajectory TDF: Brownian後のVDF/distance trajectory。GeoPTのVDF trajectoryに最も近い教師信号。

R2で使った個別Brownian displacement回帰は、本命から外す。乱数増分が入力にないため、モデルから見ると個別変位は決定不能で、GeoPTの「入力されたdynamicsからtrajectory featureを予測する」構造と合わない。

## Why This Is More GeoPT-Faithful

GeoPTのpretrainingは、random velocity fieldとsticking boundaryからVDF trajectoryを作り、形状とdynamicsの相互作用を学習させる。論文上は、この過程がtransport equation with sticking boundaryに対応し、質量保存と境界相互作用を学ぶことが重要な説明になっている。

Thermal GeoPTで対応させるべき基礎原理は、熱源温度の手作り補間ではなく、Brownian diffusion under boundary interactionである。つまり、拡散質量が内部に残るか、境界へ移るか、どの時間スケールで境界へ到達するかを学ばせる。

この設計なら、熱伝導D1だけでなく、将来のCHT、Robin/Neumann境界、構造・光輸送のような別物理への一般priorとしても主張しやすい。

## Relation To Detailed Plan

詳細計画書のsource/sink absorption、heat influence、thermal resistanceは有効な追加候補だが、初期R3本命に全部入れると下流D1へ寄せすぎる。R3では次の順に扱う。

1. R3a: 既存P2を使うdiffusion_lifted。境界到達・未到達・到達時刻を主にする。
2. R3b: GeoPTにより近いtrajectory TDF版。Brownian後のVDF/distance trajectoryも保存して学習する。
3. R3c: solver-backed heat sink D1が整った後、source/sink absorptionやRobin/heat influenceを追加アブレーションにする。

## Heavy Commands

R3aは既存P2をそのまま使える。再生成は不要。

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
```

Readiness check:

```bash
$PY scripts/check_pretrain_readiness.py \
  data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --require-phase P2 \
  --ablation diffusion_lifted
```

R3a P2 diagnostic pretraining:

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep20 \
  --epochs 20 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --val-fraction 0.05 \
  --normalization standardize \
  --normalization-max-episodes 2048 \
  --target-min-std 0.05 \
  --pretext-ablation diffusion_lifted \
  --tdf-loss-weight 0.2 \
  --diffusion-loss-weight 1.0 \
  --trajectory-tdf-loss-weight 1.0 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda
```

R3a P2 serious pretraining should move closer to GeoPT's training recipe: AdamW, cosine-style schedule, and substantially more epochs than the 20-epoch diagnostic run. Start with 100 epochs locally; extend to 200 if validation components are still improving.

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep100_wcos \
  --epochs 100 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --val-fraction 0.05 \
  --normalization standardize \
  --normalization-max-episodes 2048 \
  --target-min-std 0.05 \
  --pretext-ablation diffusion_lifted \
  --tdf-loss-weight 0.2 \
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

R3bとしてGeoPTのVDF trajectoryにより近づける場合は、P2 shardをtrajectory TDF付きで再生成する。これはデータ生成を伴うため、R3aがnoisyまたは弱い場合に実行する。

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_p2_2100 \
  --output-dir data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192_trajtdf \
  --max-shapes 2000 \
  --selection balanced \
  --episodes-per-shape 20 \
  --points-per-episode 8192 \
  --steps 3 \
  --condition-schema d1_thermal \
  --save-trajectory-tdf \
  --trajectory-tdf-feature-set vdf_distance \
  --seed 42 \
  --overwrite
```

R3b pretraining commandはR3aと同じで、manifest/outputだけ差し替える。`diffusion_lifted` は `trajectory_tdf` arrayを検出すると自動でtrajectory targetを追加する。

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_p2_d1_thermal_2000_e20_n8192_trajtdf/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r3b_diffusion_trajtdf_p2_norm_val_ep100_wcos \
  --epochs 100 \
  --batch-size 1 \
  --point-budget 8192 \
  --max-episodes 0 \
  --val-fraction 0.05 \
  --normalization standardize \
  --normalization-max-episodes 2048 \
  --target-min-std 0.05 \
  --pretext-ablation diffusion_lifted \
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
PRETRAIN_DIFFUSION=outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep20 \
NORMALIZATION_PROTOCOL=pretrained \
NORMALIZATION_CONFIG=outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep20/config.json \
RUN_PREFIX=m3_openfoam_p2_r3_diffusion_oclr \
EPOCHS=100 \
TRAIN_SIZES="10 25 50 100" \
SPLIT_SEEDS="42 43 44" \
TRAIN_SEEDS="42" \
POINT_BUDGET=3072 \
EVAL_POINT_BUDGET=3072 \
GATE_GROUPS="scratch diffusion_lifted" \
FINETUNE_SCHEDULER=onecycle \
PRETRAINED_BACKBONE_LR=3e-4 \
PRETRAINED_HEAD_LR=1e-3 \
FREEZE_PRETRAINED_BACKBONE_EPOCHS=5 \
MAX_GRAD_NORM=1.0 \
MODE=all \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
```

For the serious 100-epoch checkpoint, replace the two pretrain paths above with:

```bash
PRETRAIN_DIFFUSION=outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep100_wcos
NORMALIZATION_CONFIG=outputs/checkpoints/pretrain_r3_diffusion_lifted_p2_norm_val_ep100_wcos/config.json
```

## Interpretation

Positive signal:

- scratchより25/50 labelsでRelative L2が約10%以上良い。
- max-temperature/hotspot系の指標を悪化させない。
- 10 labelsで不安定でも、25/50 labelsで改善すればR3aは継続候補。

No-Go for R3a:

- diffusion_liftedのvalidationでboundary_hit/survival/hit_stepがほぼ改善しない。
- M3で全train sizeでscratchより悪い。

No-Goの場合でも、Thermal GeoPT自体の棄却ではない。R3b trajectory TDFでも弱い場合に、複数walk集約の到達確率・期待到達時刻へ進む。
