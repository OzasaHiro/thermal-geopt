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
