# 予備レポート: Original GeoPT の Solver-Backed Heat-Sink Conduction への転移

日付: 2026-05-11
リポジトリ: `OzasaHiro/thermal-geopt`
ステータス: 査読済み論文ではなく、予備的な技術レポート

## 概要

本実験は、GeoPT のアイデアを Thermal GeoPT pretraining によって熱伝達サロゲートモデリングへ適用する試みとして開始した。これまでで最も興味深い結果は、当初計画とは少し異なるものである。

> 流体向けの幾何 pretraining によって学習された original GeoPT pretrained checkpoint は、solver-backed heat-sink solid conduction surrogate task において label efficiency を改善した。

2つの OpenFOAM-backed heat-sink D1 benchmark において、original GeoPT backbone は 25-label および 50-label の条件で scratch training を一貫して上回った。一方、現在の Thermal-specific R4 pretraining は有効ではなく、negative transfer を引き起こした。

この結果は、より限定的だが有用な仮説を支持している。

> GeoPT は、流体予測に限定されない transferable geometry-boundary-dynamics prior を学習している可能性がある。

## 検証内容

### Downstream Task

下流タスクは steady solid conduction であり、OpenFOAM Foundation v13 の `laplacianFoam` で解いた。

各ケースでは以下を行う。

- 3D heat-sink-like solid geometry を生成する。
- base を heat source または hot boundary とする。
- exterior を cooling boundary とする。
- OpenFOAM が scalar temperature field `T` を解く。
- 結果を、point coordinates、condition features、cell-centered temperature labels を持つ NPZ に変換する。
- Transolver backbone が point-wise input から temperature を予測する。

これは solver-backed data である。初期の synthetic D1 proxy ではない。

### Benchmarks

幾何形状に変化を持つ2つの D1 benchmark を用いた。

| Benchmark | Cases | Families | Test cases | Purpose |
|---|---:|---|---:|---|
| M4 heat-sink D1 | 300 | `plate_fin`, `pin_fin` | 45 | 最初の solver-backed heat-sink gate |
| M5 complex heat-sink D1 | 300 | `plate_fin`, `pin_fin`, `staggered_pin_fin` | 45 | より複雑な geometry gate および visualization target |

どちらも 25 および 50 training cases の label-scarcity split を用い、split seeds は `42`, `43`, `44` とした。

### 比較群

| Group | Meaning |
|---|---|
| `scratch` | 同一の downstream architecture を random initialization から学習 |
| `geopt_original` | original GeoPT pretrained checkpoint, `../GeoPT/checkpoints/GeoPT_8layers.pt` |
| `geopt_transport_lifted` | 現在の Thermal GeoPT R4 transport-lifted pretraining |

original GeoPT checkpoint は、shape-compatible な backbone tensor を D1 Transolver に読み込む。Input/output tensor のうち shape が合わないものは skip する。

## 主な結果

### M4 Heat-Sink D1

Test Relative L2:

| Train labels | Scratch | Thermal R4 | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.013122 +/- 0.000685 | 0.037595 +/- 0.001770 | 0.011370 +/- 0.001009 |
| 50 | 0.010674 +/- 0.000470 | 0.022637 +/- 0.000570 | 0.008840 +/- 0.000345 |

Paired improvement vs scratch:

| Train labels | Thermal R4 | Original GeoPT |
|---:|---:|---:|
| 25 | -187.1% | +13.4% |
| 50 | -112.4% | +17.2% |

Temperature extrema and hotspot metrics:

| Train labels | Group | maxT abs error [K] | hotspot abs error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.64 | 2.74 |
| 25 | original GeoPT | 1.98 | 2.00 |
| 50 | scratch | 1.14 | 1.24 |
| 50 | original GeoPT | 0.99 | 1.00 |

### M5 Complex Heat-Sink D1

Test Relative L2:

| Train labels | Scratch | Thermal R4 | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.012548 +/- 0.000259 | 0.032063 +/- 0.001846 | 0.011186 +/- 0.000006 |
| 50 | 0.010613 +/- 0.000060 | 0.022536 +/- 0.002766 | 0.008950 +/- 0.000194 |

Paired improvement vs scratch:

| Train labels | Thermal R4 | Original GeoPT |
|---:|---:|---:|
| 25 | -155.5% | +10.8% |
| 50 | -112.3% | +15.7% |

Temperature extrema and hotspot metrics:

| Train labels | Group | maxT abs error [K] | hotspot abs error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.88 | 2.93 |
| 25 | original GeoPT | 1.44 | 1.44 |
| 50 | scratch | 1.12 | 1.14 |
| 50 | original GeoPT | 0.93 | 0.91 |

## 解釈

M4 および M5 の結果は同じ方向を示している。

- Original GeoPT は、label scarcity 条件における solver-backed heat-sink conduction prediction を改善した。
- 改善は Relative L2 だけでなく、max-temperature および hotspot metrics にも現れている。
- 現在の Thermal-specific R4 pretraining は、この downstream task に対して有用な initialization ではなかった。

これは重要である。GeoPT の transferable component が、単なる fluid-specific prior ではない可能性を示しているためである。あり得る解釈として、original checkpoint は以下のような有用な geometry-boundary-dynamics structure を符号化している。

- 3D point set の spatial organization
- boundary-aware representation
- geometry constraints の下での field propagation
- PDE-like surrogate task に再利用可能な inductive bias

Thermal R4 の negative result も有用である。heat-themed な pretraining target を作るだけでは不十分であることを示しているためである。Thermal-specific extension は、poorly aligned synthetic task によって有用な GeoPT prior を上書きするのではなく、それを保持する必要がある。

### Original GeoPT と Thermal R4 Pretraining の違い

この比較は、Thermal GeoPT のコンセプトそのものを否定する結果として読むべきではない。2つの pretrained initialization は、規模も構成も大きく異なる。

Original GeoPT は、はるかに大規模に学習されている。原論文では、100万を超える pretraining samples を、多様な off-the-shelf geometries と dynamics-lifted geometric self-supervision によって用いたことが報告されている。その target は thermal conduction そのものではなく、generic geometry-boundary-dynamics prior である。

一方、現在の Thermal R4 pretraining は、この project 内で作成した、はるかに小規模な thermal-motivated prototype である。geometry families は限定的であり、synthetic target もより specific である。したがって、この negative transfer が示しているのは、この R4 design と scale が downstream task にまだ整合していないということである。thermal-specific GeoPT pretraining が不可能、あるいは本質的に無益であることを示すものではない。

より重要な signal は、original GeoPT からの positive transfer である。ここで用いた checkpoint は thermal labels で学習されていないにもかかわらず、solver-backed heat conduction を改善した。Original GeoPT 論文では、同じ lifted geometric pretraining が fluid mechanics だけでなく solid mechanics benchmark にも有効であることが報告されている。この結果と合わせると、single dynamics-lifted geometric pretraining が複数の simulation families に対する reusable initialization になり得る、というより広い仮説を補強している。

次の Thermal GeoPT iteration では、この general GeoPT prior を保持しながら thermal relevance を追加することが重要である。考えられる方向性としては、original GeoPT からの continued pretraining、lightweight thermal adapters、あるいはより大規模で多様な thermal-lifted pretraining dataset がある。いずれの場合も、original GeoPT を strong control として比較し続ける必要がある。

## 可視化例

コミュニケーション用途として、現時点で最も適した図は、3D heat-sink surface と internal temperature cut plane を組み合わせたものである。weak-cooling case は M4/M5 benchmark data とは別であり、説明用の visualization のみを意図している。

![Weak-cooling staggered-pin heat-sink surface temperature](assets/figures/m5_weak_cooling_surface_temperature.png)

Visualization case の生成:

```bash
../../.venv/bin/python scripts/generate_d1_openfoam_heatsink_cases.py \
  --case-count 1 \
  --families staggered_pin_fin \
  --cells-x 20 \
  --cells-y 20 \
  --base-cells-z 4 \
  --feature-cells-z 12 \
  --source-temperature-min 520 \
  --source-temperature-max 560 \
  --sink-temperature-min 295 \
  --sink-temperature-max 305 \
  --sink-value-fraction 0.025 \
  --raw-dir data/downstream_raw/d1_openfoam_visual_staggered_pin_weak_cooling \
  --output-dir data/downstream_npz/d1_openfoam_visual_staggered_pin_weak_cooling \
  --overwrite
```

Figure の render:

```bash
../../.venv/bin/python scripts/render_d1_surface_temperature.py \
  --manifest data/downstream_npz/d1_openfoam_visual_staggered_pin_weak_cooling/manifest.json \
  --family staggered_pin_fin \
  --max-cases 1 \
  --output-dir outputs/figures/m5_visual_weak_cooling_surface_temperature \
  --camera-zoom 0.9
```

## 再現コマンド

### M4 Data の生成

```bash
OVERWRITE=1 CASE_COUNT=300 bash scripts/run_m4_openfoam_heatsink_d1.sh
```

### M4 Transfer Gate の実行

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m4_heatsink_transfer_gate.sh
```

### M5 Data の生成

```bash
OVERWRITE=1 bash scripts/run_m5_openfoam_complex_heatsink_d1.sh
```

### M5 Transfer Gate の実行

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m5_complex_heatsink_transfer_gate.sh
```

## Artifacts

主な output files は以下である。

- `docs/m4_heatsink_transfer_results.md`
- `docs/results/m4_heatsink_transfer_summary.json`
- `docs/m5_complex_heatsink_transfer_results.md`
- `docs/results/m5_complex_heatsink_transfer_summary.json`
- `docs/assets/figures/m5_weak_cooling_surface_temperature.png`

OpenFOAM case directories、NPZ datasets、checkpoints、full logs、大量の generated figures などの large artifacts は、デフォルトで Git から除外している。この repository には、選択した軽量な result summaries と communication figure を1つだけ保存している。

## 限界

これは最終的な論文レベルの主張ではない。

- split seeds は3つのみである。
- downstream task は solid conduction であり、conjugate heat transfer ではない。
- pin-fin geometry は blockMesh-friendly で rectangularized されたものであり、fully resolved industrial heat sink ではない。
- original GeoPT checkpoint は external pretrained control として使用しており、この report では再配布していない。
- original GeoPT と Thermal R4 pretraining は、data scale、geometry diversity、pretraining target が大きく異なる。そのため、R4 の negative result は Thermal GeoPT concept に対する controlled ablation ではない。
- Thermal GeoPT R4 の negative result は、現在の R4 pretraining design に対する結果であり、あらゆる thermal-specific GeoPT extension を否定するものではない。

## 公開時の推奨表現

慎重な表現を用いる。

> We provide preliminary evidence that the original GeoPT pretrained backbone
> transfers to OpenFOAM-backed heat-sink solid-conduction surrogate modeling
> under label scarcity.  Across two D1 heat-sink benchmarks, original GeoPT
> improves scratch training by about 10-17% Relative L2 and improves hotspot
> metrics.  A first thermal-specific pretraining attempt caused negative
> transfer, suggesting that thermal extensions should preserve the original
> GeoPT geometry-boundary prior.

日本語では以下のように表現できる。

> Original GeoPT の pretrained backbone が、label scarcity 条件下の OpenFOAM-backed heat-sink solid-conduction surrogate modeling に転移することを示す予備的証拠を提示する。2つの D1 heat-sink benchmark において、original GeoPT は scratch training に対し Relative L2 を約10-17%改善し、hotspot metrics も改善した。一方、最初の thermal-specific pretraining attempt は negative transfer を引き起こした。この結果は、thermal extension では original GeoPT の geometry-boundary prior を保持する必要があることを示唆している。

これは、repository を tag し archive した後で、GitHub technical report または短い arXiv-style note として公開するのに適している。
