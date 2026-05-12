# 予備レポート: OpenFOAMで解いたヒートシンク熱伝導に対する Original GeoPT checkpoint の転移

日付: 2026-05-11
リポジトリ: `OzasaHiro/thermal-geopt`
ステータス: 査読済み論文ではなく、予備的な技術レポート

## 概要

本実験は、GeoPT の dynamics-lifted geometric pre-training idea を熱伝達サロゲートモデリングへ適用する試みとして開始した。これまでで最も興味深い結果は、当初計画とは少し異なるものである。

> このリポジトリで行った予備的な OpenFOAM-solved heat-sink conduction test では、original GeoPT checkpoint の shape-compatible tensors で初期化した Transolver が、25ケースおよび50ケースの low-data setting において、scratch training より低い test error を示した。

2つの生成された OpenFOAM-solved heat-sink solid-conduction benchmark において、3つの data split seed の各集計で、partially loaded original GeoPT checkpoint は 25 および 50 downstream training cases の条件で mean Relative L2 を改善した。ただし、すべての individual test cases が改善したわけではない。現在の project-local thermal-specific pretraining prototype、内部ID `R4` は有効ではなく、negative transfer を示した。

original checkpoint は、off-the-shelf の car、airplane、watercraft geometry subsets に対し、dynamics-lifted self-supervision によって pre-training されたものである。thermal labels で pre-training されたものではない。

この結果は、original GeoPT paper の broader cross-regime framing と整合しており、このリポジトリ内でのより限定的な仮説を支持する。

> Partially transferred original GeoPT initialization は、original paper では評価されていない OpenFOAM-solved solid-conduction setting でも有用である可能性がある。

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

これは solver-generated data であり、初期の synthetic proxy dataset ではない。このタスクは、controlled solid-conduction surrogate benchmark として解釈すべきであり、物理的に完全な heat-sink cooling model ではない。boundary condition types、source/sink value ranges、material assumptions、feature encodings は generation scripts によって定義されている。

### Benchmarks

幾何形状に変化を持つ2つの solid-conduction benchmark を用いた。

| Benchmark | Cases | Families | Test cases | Purpose |
|---|---:|---|---:|---|
| Benchmark A, internal ID `M4` | 300 | `plate_fin`, `pin_fin` | 45 | 最初の OpenFOAM-solved heat-sink benchmark |
| Benchmark B, internal ID `M5` | 300 | `plate_fin`, `pin_fin`, `staggered_pin_fin` | 45 | より複雑な heat-sink benchmark および visualization target |

どちらも 25 および 50 training cases の low-data split を用い、data split seeds は `42`, `43`, `44` とした。

内部データセットラベル `D1` は、このプロジェクトにおける最初の OpenFOAM-solved solid-conduction dataset family を指す。すなわち、steady scalar-conduction labels を持つ、生成された heat-sink-like blockMesh geometries である。一次元の物理問題という意味ではない。

### 比較群

| Group | Meaning |
|---|---|
| `scratch` | 同一の downstream architecture を random initialization から学習 |
| `geopt_original` | original GeoPT pretrained checkpoint, `../GeoPT/checkpoints/GeoPT_8layers.pt` |
| `geopt_transport_lifted` | project-local thermal-specific pretraining prototype, internal ID `R4` |

Checkpoint transfer は partial かつ shape-matched である。内部ID `M4` および `M5` の runs では、original GeoPT は 166 個の compatible tensors を読み込み、`preprocess.linear_pre.0.weight` と final prediction head `blocks.7.mlp2.{weight,bias}` を skipped/missed とした。Thermal-specific prototype `R4` は 167 個の tensors を読み込み、`blocks.7.mlp2.{weight,bias}` のみを skipped/missed とした。したがって、どちらの pretrained group でも最終的な temperature prediction head は downstream で学習されており、original GeoPT の input projection は部分的にのみ転移されている。

## 主な結果

値は、特記がない限り data split seeds `42`, `43`, `44` に対する mean +/- standard deviation である。すべての runs で downstream training seed は `42` のみを用いているため、このばらつきは split-level variability であり、optimizer stochasticity 全体や paper-level confidence interval を表すものではない。

### Benchmark A: Heat-Sink Solid Conduction, Internal ID `M4`

Test Relative L2:

| Training cases | Scratch | Thermal prototype (`R4`) | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.013122 +/- 0.000685 | 0.037595 +/- 0.001770 | 0.011370 +/- 0.001009 |
| 50 | 0.010674 +/- 0.000470 | 0.022637 +/- 0.000570 | 0.008840 +/- 0.000345 |

Paired improvement vs scratch:

Paired improvement は、同じ train/test split を用いた scratch run に対して、data split seed ごとに `(scratch Relative L2 - group Relative L2) / scratch Relative L2` として計算した。負の値は scratch より error が大きいことを意味する。

| Training cases | Thermal prototype (`R4`) | Original GeoPT |
|---:|---:|---:|
| 25 | -187.1% | +13.4% |
| 50 | -112.4% | +17.2% |

Temperature extrema and hotspot-temperature metrics:

| Training cases | Group | max-temperature absolute error [K] | hotspot-temperature absolute error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.64 | 2.74 |
| 25 | original GeoPT | 1.98 | 2.00 |
| 50 | scratch | 1.14 | 1.24 |
| 50 | original GeoPT | 0.99 | 1.00 |

### Benchmark B: Complex Heat-Sink Solid Conduction, Internal ID `M5`

Test Relative L2:

| Training cases | Scratch | Thermal prototype (`R4`) | Original GeoPT |
|---:|---:|---:|---:|
| 25 | 0.012548 +/- 0.000259 | 0.032063 +/- 0.001846 | 0.011186 +/- 0.000006 |
| 50 | 0.010613 +/- 0.000060 | 0.022536 +/- 0.002766 | 0.008950 +/- 0.000194 |

Paired improvement vs scratch:

| Training cases | Thermal prototype (`R4`) | Original GeoPT |
|---:|---:|---:|
| 25 | -155.5% | +10.8% |
| 50 | -112.3% | +15.7% |

Temperature extrema and hotspot-temperature metrics:

| Training cases | Group | max-temperature absolute error [K] | hotspot-temperature absolute error [K] |
|---:|---|---:|---:|
| 25 | scratch | 2.88 | 2.93 |
| 25 | original GeoPT | 1.44 | 1.44 |
| 50 | scratch | 1.12 | 1.14 |
| 50 | original GeoPT | 0.93 | 0.91 |

## 解釈

2つの benchmark results は同じ方向を示している。

- Original GeoPT は、low-data fine-tuning における OpenFOAM-solved heat-sink conduction prediction を改善した。
- 改善は Relative L2 だけでなく、max-temperature および hotspot-temperature metrics にも現れている。
- 現在の project-local thermal-specific pretraining prototype、内部ID `R4` は、この downstream task に対して有用な initialization ではなかった。

これは、GeoPT が fluid-only prediction に限られないという original paper の broader cross-regime framing と整合する。あり得る解釈として、original checkpoint は GeoPT pretext task から、boundary-aware spatial organization および transport-like geometry-dynamics correlations を学習しており、それが今回の transfer に寄与した可能性がある。

- 3D point set の spatial organization
- boundary-aware representation
- synthetic-velocity pretext task から得られる transport-like geometry-dynamics correlations
- neural PDE surrogate tasks に対する reusable initialization

ただし、この実験は checkpoint のどの部分が transfer を引き起こしたかを特定していない。boundary-aware geometry-dynamics representation という解釈は plausible explanation であり、ablation-proven mechanism ではない。

Thermal-specific prototype の negative result も有用だが、狭く解釈すべきである。この結果は、この heat-themed pretext task が downstream task に自動的に整合するわけではないことを示している。ただし、original GeoPT checkpoint の learned representation に対する controlled ablation として解釈すべきではない。

### Original GeoPT と Thermal-Specific Prototype Pretraining の違い

この比較は、Thermal GeoPT のコンセプトそのものを否定する結果として読むべきではない。2つの pretrained initialization は、規模も構成も大きく異なる。

Original GeoPT は、はるかに大規模に学習されている。原論文では、100万を超える solver-free pretraining samples を、多様な off-the-shelf geometries、具体的には car、airplane、watercraft subsets と dynamics-lifted geometric self-supervision によって用いたことが報告されている。その target は thermal conduction そのものではなく、generic geometry-dynamics pretext task である。

一方、現在の project-local thermal-specific pretraining prototype、内部ID `R4` は、はるかに小規模であり、geometry families も限定的で、synthetic target もより specific である。したがって、この negative transfer が示しているのは、この design と scale が downstream task にまだ整合していないということである。thermal-specific GeoPT pretraining が不可能、あるいは本質的に無益であることを示すものではない。

本予備レポートにおいて最も実用的な観察結果は、original GeoPT からの positive checkpoint-transfer result である。一方、`R4` result は、小規模な thermal-specific prototype 1件に対する negative result として扱うのが適切である。ここで用いた checkpoint は thermal labels で学習されていないにもかかわらず、OpenFOAM-solved heat conduction を改善した。Original GeoPT paper では、同じ lifted geometric pretraining が fluid mechanics だけでなく solid mechanics benchmarks にも有効であることが報告されている。この結果は、broader GeoPT hypothesis と整合する。ただし、小規模な conduction benchmark suite から得られた preliminary evidence として扱うべきである。

将来の thermal extension は、今回の evidence とは別に記述すべきである。本レポートで主張するのは、観測された checkpoint-transfer result に限る。

## 可視化例

コミュニケーション用途として、現時点で最も適した図は、3D heat-sink surface と internal temperature cut plane を組み合わせたものである。weak-cooling case は quantitative benchmark datasets、内部ID `M4` および `M5` とは別であり、説明用の visualization のみを意図している。

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

### Benchmark A Data の生成, Internal ID `M4`

```bash
OVERWRITE=1 CASE_COUNT=300 bash scripts/run_m4_openfoam_heatsink_d1.sh
```

### Benchmark A Transfer Evaluation の実行

```bash
TRAIN_SIZES="25 50" SPLIT_SEEDS="42 43 44" EPOCHS=50 MODE=all \
  bash scripts/run_m4_heatsink_transfer_gate.sh
```

### Benchmark B Data の生成, Internal ID `M5`

```bash
OVERWRITE=1 bash scripts/run_m5_openfoam_complex_heatsink_d1.sh
```

### Benchmark B Transfer Evaluation の実行

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

OpenFOAM case directories、NPZ datasets、checkpoints、full logs、大量の generated figures などの large artifacts は、デフォルトで Git から除外している。この repository には、選択した軽量な result summaries と communication figure を1つだけ保存している。軽量な JSON summaries が、報告した tables に対する主要な audit trail である。Full OpenFOAM cases、NPZ datasets、logs、checkpoints は Git から意図的に除外している。厳密な numerical auditability が必要な archival release では、これらを別途添付すべきである。

## 限界

これは最終的な論文レベルの主張ではない。

- data split seeds は3つのみである。
- downstream training seed は `42` の1つのみである。報告したばらつきは split-level variation であり、optimizer stochasticity を表すものではない。
- downstream task は solid conduction であり、conjugate heat transfer ではない。
- pin-fin geometry は blockMesh-friendly で rectangularized されたものであり、fully resolved industrial heat sink ではない。
- original GeoPT checkpoint は external pretrained control として使用しており、この report では再配布していない。
- original GeoPT と project-local thermal-specific pretraining runs は、data scale、geometry diversity、pretraining target が大きく異なる。そのため、`R4` の negative result は Thermal GeoPT concept に対する controlled ablation ではない。
- `R4` negative transfer は、pretraining scale、target alignment、normalization、optimizer schedule、checkpoint selection、downstream fine-tuning recipe などを反映している可能性がある。これは thermal-specific pretraining に対する反証ではなく、この `R4` transfer recipe の失敗として解釈すべきである。
- 内部ID `R4` の negative result は、現在の pretraining design に対する結果であり、あらゆる thermal-specific GeoPT extension を否定するものではない。
- 再現には OpenFOAM Foundation v13、この project の Python environment、外部の original GeoPT checkpoint `../GeoPT/checkpoints/GeoPT_8layers.pt` が必要である。checkpoint と generated datasets は再配布していないため、この repository は script-level reproduction を支援するが、完全に self-contained な artifact reproduction ではない。

## 公開時の推奨表現

慎重な表現を用いる。

> Preliminary OpenFOAM-solved heat-sink solid-conduction benchmarks において、original GeoPT checkpoint の shape-compatible tensors で初期化した Transolver は、25 および 50 training cases の条件で scratch training に対し mean Relative L2 を約10-17%改善し、max-temperature および hotspot-temperature errors も改善した。小規模な project-local thermal-specific pretraining prototype、内部ID `R4` は、同じ evaluation protocol で negative transfer を示した。これらの結果は preliminary checkpoint-transfer evidence であり、Thermal GeoPT concept 全体に対する controlled ablation ではない。

これは、clear preliminary-status caveats を付けた GitHub technical report として公開するのに適している。
