# Heavy Run Commands

このメモは、時間がかかる可能性がある処理を手元で実行するためのコマンド集である。Codex側では重い生成を走らせず、短いsmokeだけを実行する。

前提:

```bash
cd /home/hiroaki-ozasa/Desktop/Development/PINN_CFD/EXPERIMENT/Thermal_GeoPT
PY=../../.venv/bin/python
```

## 0. 軽量smoke

まずこれは短時間で通す。

```bash
$PY scripts/run_smoke_pipeline.py --overwrite
```

## 1. CadQuery形状生成

初回pilot:

```bash
$PY scripts/generate_cadquery_shapes.py \
  --output-dir data/meshes_raw/cadquery_pilot_300 \
  --num-per-family 100 \
  --seed 42 \
  --overwrite
```

本格pretraining前の候補:

```bash
$PY scripts/generate_cadquery_shapes.py \
  --output-dir data/meshes_raw/cadquery_pretrain_3000 \
  --num-per-family 1000 \
  --seed 42 \
  --overwrite
```

## 2. メッシュ前処理

Pilot:

```bash
$PY scripts/preprocess_meshes.py \
  --input-dir data/meshes_raw/cadquery_pilot_300 \
  --output-dir data/meshes_processed/cadquery_pilot_300 \
  --surface-points 8192 \
  --seed 42 \
  --overwrite
```

本格pretraining候補:

```bash
$PY scripts/preprocess_meshes.py \
  --input-dir data/meshes_raw/cadquery_pretrain_3000 \
  --output-dir data/meshes_processed/cadquery_pretrain_3000 \
  --surface-points 8192 \
  --seed 42 \
  --overwrite
```

## 3. Pretraining episode生成

まず小さめ:

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_pilot_300 \
  --output-dir data/pretrain_zarr/cadquery_pilot_300_e20_n4096 \
  --episodes-per-shape 20 \
  --points-per-episode 4096 \
  --steps 2 \
  --seed 42 \
  --overwrite
```

RTX 5070 Ti向けの初期候補:

```bash
$PY scripts/generate_pretrain_episodes.py \
  --processed-dir data/meshes_processed/cadquery_pretrain_3000 \
  --output-dir data/pretrain_zarr/cadquery_pretrain_3000_e20_n8192 \
  --episodes-per-shape 20 \
  --points-per-episode 8192 \
  --steps 2 \
  --seed 42 \
  --overwrite
```

## 4. 実行後の確認

D1 proxyを軽く作る場合:

```bash
$PY scripts/generate_d1_conduction_cases.py \
  --processed-dir data/meshes_processed/cadquery_pilot_300 \
  --output-dir data/downstream_npz/d1_proxy_pilot_300_c5_n8192 \
  --cases-per-shape 5 \
  --points-per-case 8192 \
  --seed 42 \
  --overwrite
```

split作成:

```bash
$PY scripts/build_splits.py \
  --input-manifest data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json \
  --output configs/d1_proxy_pilot_300_c5_n8192_split.json \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --seed 42
```

`scripts/inspect_artifacts.py` が追加済みの場合:

```bash
$PY scripts/inspect_artifacts.py \
  data/meshes_raw/cadquery_pilot_300/manifest.json \
  data/meshes_processed/cadquery_pilot_300/manifest.json \
  data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json \
  data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json
```

未追加の場合は、まず以下でファイル数を確認する。

```bash
find data/meshes_raw/cadquery_pilot_300 -name '*.stl' | wc -l
find data/meshes_processed/cadquery_pilot_300 -name '*.npz' | wc -l
find data/pretrain_zarr/cadquery_pilot_300_e20_n4096 -name 'shard_*.zarr' -maxdepth 1 | wc -l
```

## 5. Pilot学習

Artifact確認まで終わったら、次は `docs/pilot_training_commands.md` の順に進める。

推奨順:

1. Pretrain smoke
2. Pretrain pilot
3. D1 scratch pilot
4. D1 pretrained pilot
5. D1 test評価

## 6. Post-gate R0/R1

今回の目的は GeoPT の熱伝達サロゲート転用の有効性検証なので、単一seedの pilot gate から次へ進む場合は以下を使う。

R0: `100 labels + no_boundary_field` の再現性確認

```bash
bash scripts/run_r0_replication_matrix.sh
bash scripts/run_r0_test_eval.sh
```

詳細と smoke は `docs/r0_replication_commands.md` を参照。

R1: GeoPT 本来の dynamics-lifted pretraining への戻し

```bash
$PY scripts/train_pretrain.py \
  --manifest data/pretrain_zarr/cadquery_pilot_300_e20_n4096/manifest.json \
  --output-dir outputs/checkpoints/pretrain_r1_dynamics_lifted_no_boundary_ep2 \
  --pretext-ablation dynamics_lifted \
  --condition-mode zero_boundary_field \
  --epochs 2 \
  --batch-size 1 \
  --point-budget 4096 \
  --max-episodes 0 \
  --amp \
  --amp-dtype bfloat16 \
  --device cuda \
  --seed 42
```

詳細は `docs/r1_dynamics_lifted_pretraining.md` を参照。

## 7. Git管理

生成データは `.gitignore` 対象なのでGitに入れない。コード・設定・軽量Markdownだけをコミットする。
