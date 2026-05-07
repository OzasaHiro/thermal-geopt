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

## 7. M1 solver-backed D1 OpenFOAM pilot

OpenFOAM Foundation v13 `laplacianFoam` でD1 solid-conduction block pilotを生成する。
現在のblockMesh版50ケースは通常「数分程度」の想定だが、OpenFOAMをケース数分起動するため手元実行に回す。

```bash
OVERWRITE=1 bash scripts/run_m1_openfoam_pilot.sh
```

短い確認だけ先に行う場合:

```bash
CASE_COUNT=10 CELLS_X=8 CELLS_Y=8 CELLS_Z=8 \
RAW_DIR=data/downstream_raw/d1_openfoam_block_pilot_10_smoke \
NPZ_DIR=data/downstream_npz/d1_openfoam_block_pilot_10_smoke \
SPLIT_PATH=configs/d1_openfoam_block_pilot_10_smoke_split_seed42.json \
BASELINE_JSON=outputs/logs/d1_openfoam_block_pilot_10_smoke_baseline_test.json \
OVERWRITE=1 bash scripts/run_m1_openfoam_pilot.sh
```

このrunnerは内部で `/opt/openfoam13/etc/bashrc` をsourceする。

solver-backed D1 scratch smoke:

```bash
bash scripts/run_m1_openfoam_downstream_smoke.sh
```

solver-backed D1 scratch pilot:

```bash
MAX_TRAIN_CASES=0 MAX_VAL_CASES=0 EPOCHS=20 DEVICE=cuda \
OUTPUT_DIR=outputs/checkpoints/d1_openfoam_block_scratch_pilot_ep20 \
EVAL_JSON=outputs/logs/d1_openfoam_block_scratch_pilot_ep20_test.json \
bash scripts/run_m1_openfoam_downstream_smoke.sh
```

solver-backed D1 initialization comparison:

```bash
bash scripts/run_m1_openfoam_init_compare.sh
```

この比較runnerのデフォルトは、既存pretraining checkpointとarchitectureを揃えるため
`N_HIDDEN=256 N_LAYERS=8 N_HEADS=8 SLICE_NUM=32` にしている。

If the R1 dynamics-lifted pretraining checkpoint exists, include it with:

```bash
INIT_GROUPS="scratch full static_tdf_only no_boundary_field dynamics_lifted" \
bash scripts/run_m1_openfoam_init_compare.sh
```

## 8. Git管理

生成データは `.gitignore` 対象なのでGitに入れない。コード・設定・軽量Markdownだけをコミットする。
