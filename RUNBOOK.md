# Thermal GeoPT Runbook

作成日: 2026-05-06

## 1. 計画書からの実行方針

詳細計画では、Thermal GeoPTを単純な「GeoPTの熱版」ではなく、熱拡散に合わせた `Diffusion-Lifted Geometric Pre-Training` として進める。

初期方針:

- 原案の `TDF = 境界距離 * 熱伝導率` は査読上弱いため、距離、境界方向、拡散時間スケール、heat-kernel proximity、近似熱抵抗、境界法線、source/sink proximity の多チャネル特徴にする。
- 事前学習は Brownian random walk、境界到達、熱源/放熱境界の合成場を使う。
- 下流評価は CHT 直行ではなく、D1 solid conduction、D2 simplified CHT、D3 OOD geometry transfer の順に段階化する。
- 初期論文化に必要な検証は、Baseモデル、0.16M-0.60M episodes、下流ラベル 50-300 ケース規模を現実ラインとする。

## 2. 最初の実装順序

1. 環境確認: RTX 5070 Tiで Transolver-Base の forward/backward、bf16 AMP、9k-12k points/sample を確認する。
2. Geometry preprocessing: STL/STEP読み込み、watertight check、法線修正、座標正規化、surface/volume sampling、nearest boundary projection。
3. TDF/Thermal diffusion feature検証: sphere/cube/cylinder/plate-fin で距離・法線・inside/outside判定を確認する。
4. Brownian trajectory generator: absorbing、reflecting、partial absorbing の境界処理を最小実装する。
5. Boundary heat field generator: surface RBF patchで `q_b`、`h_b`、source/sink codeを作る。
6. Pretrain dataset writer: fp16/Zarr shardまたはオンザフライ生成の両対応にする。
7. D1 fine-tuning dataset writer: OpenFOAM/FEM結果を `case_*.npz` へ統一する。

## 3. GeoPT PoCからの流用方針

詳細は `docs/geopt_asset_reuse_inventory.md` にまとめた。

優先的に流用するもの:

- 公式GeoPT Transolverコードと checkpoint load 方針。
- VTP読み込み、point normals、field key解決、`params.json` flatten、condition vector化。
- split/audit生成、preprocessed `.npy` manifest、best checkpoint評価、VTP/PNG可視化、summary report生成。
- bf16 AMP前提。GeoPT PoCでは fp16 AMP が初期forwardからNaN化したため、Thermal側でも最初は bf16 を標準にする。

## 4. Git/GitHub方針

通常Gitで管理する:

- `thermal_geopt/`
- `scripts/`
- `configs/`
- `docs/`
- `README.md`
- `RUNBOOK.md`
- 軽量な評価JSON/CSV/Markdownのうち、論文化に必要なもの

通常Gitに入れない:

- raw dataset
- STL/VTP/VTU/FOAM outputs
- `.npy` / `.npz`
- Zarr shard
- `.pt` / checkpoint
- 大量のPNG/動画/ログ

GitHub repo名の初期案:

```text
OzasaHiro/thermal-geopt
```

`gh` は導入済みだが未ログイン。認証とGitHub repo作成手順は `docs/github_setup.md` を参照。

## 5. 2026-05-06 実験準備メモ

Git/GitHub:

- `main` は `origin/main` と同期済み。
- remote: `https://github.com/OzasaHiro/thermal-geopt.git`
- 初回コミット: `609aa16 Initialize Thermal GeoPT experiment`

環境確認:

```bash
../../.venv/bin/python scripts/check_environment.py \
  --transolver-smoke \
  --points 1024 \
  --amp \
  --amp-dtype bfloat16
```

確認結果:

- Python 3.12.3
- PyTorch 2.10.0+cu128
- CUDA 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti
- bf16 supported: true
- GeoPT Transolver smoke: forward/backward成功
- smoke条件: 1024 points、out_dim 8、bf16 AMP
- reserved peak memory: 約148 MB

導入済み依存:

- cadquery 2.7.0
- trimesh 4.12.2
- zarr 3.2.1
- pyvista 0.47.1
- scipy 1.17.1

未導入だが現時点では任意:

- open3d

初期パイプラインsmoke:

```bash
# 1. TDF/Brownian unit smoke on a sphere
../../.venv/bin/python scripts/smoke_tdf_brownian.py \
  --num-points 2048 \
  --steps 2

# 2. Generate tiny CadQuery thermal shapes
../../.venv/bin/python scripts/generate_cadquery_shapes.py \
  --num-per-family 1 \
  --overwrite

# 3. Preprocess STL meshes into sampled normalized arrays
../../.venv/bin/python scripts/preprocess_meshes.py \
  --surface-points 2048 \
  --overwrite

# 4. Generate tiny pretraining Zarr shards
../../.venv/bin/python scripts/generate_pretrain_episodes.py \
  --episodes-per-shape 1 \
  --points-per-episode 512 \
  --steps 2 \
  --overwrite
```

確認済み成果:

- CadQueryで `channel_block`、`pin_fin`、`plate_fin` の3形状をSTL生成できた。
- STLを正規化し、各2048 surface pointsの `.npz` へ変換できた。
- 前処理済み形状3件から、各1 episode、512 points/episode、14 feature channelsのZarr shardを生成できた。
- 生成データは `.gitignore` 対象で、Git管理対象には入れない。

次の実装候補:

1. `generate_cadquery_shapes.py` を100-300形状規模へ拡張し、manifestにtrain/val/test split候補を出す。
2. `preprocess_meshes.py` にinside/outside判定、volume/shell sampling、nearest-boundary validationを追加する。
3. `generate_pretrain_episodes.py` をshard size指定、fp16/Zarr圧縮、checksum保存に対応させる。
4. D1 solid conduction用のOpenFOAM Foundation v13 `laplacianFoam` writerを作る。portable FEM/FVMは明示承認がある場合だけの退避策にする。

軽量smokeをまとめて実行する場合:

```bash
../../.venv/bin/python scripts/run_smoke_pipeline.py --overwrite
```

時間がかかるデータ生成は `docs/heavy_run_commands.md` のコマンドを使って手元で実行する。

追加で整備済み:

- `scripts/generate_d1_conduction_cases.py`: OpenFOAMなしでD1 downstream配管を確認するsource/sink proxy case生成。物理解ではなくsmoke/proxy用途。
- `scripts/build_splits.py`: manifestから決定的なtrain/val/test split JSONを生成。
- `scripts/inspect_artifacts.py`: `.npz`、`manifest.json`、`.zarr` のshape/dtype/finite check。
- `scripts/train_pretrain.py`: Thermal diffusion feature/TDFを教師にしたTransolver pilot pretraining。
- `scripts/train_finetune_d1.py`: D1 source/sink proxy temperature predictionのscratch/pretrained fine-tuning。
- `scripts/evaluate_d1.py`: D1 checkpointまたはmean-temperature baselineの評価JSON生成。

学習smokeで確認済み:

- `train_pretrain.py`: tiny Zarr、128 points、2 episodes、1 epoch、bf16 AMPでcheckpoint生成まで成功。
- `train_finetune_d1.py`: tiny D1 proxy、scratch/pretrained初期化の両方で1 epoch、checkpoint生成とvalidation評価まで成功。
- `evaluate_d1.py`: baseline評価とcheckpoint評価の両方でJSON出力まで成功。
- pilot生成済みartifactでも、pretrain manifest、D1 manifest/split、D1 checkpoint evalを最小設定で読み込み確認済み。

Pilot artifact確認後の次手順は `docs/pilot_training_commands.md` を使う。

GeoPT本来のデータ効率検証へ進む場合は、full-label D1 pilotの延長ではなく `docs/geopt_gate_commands.md` のlabel-scarcity gateを使う。
