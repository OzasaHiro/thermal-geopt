# GeoPT PoC Asset Reuse Inventory

参照元: `../GeoPT`

## PoCの確認結果

GeoPT PoCは、SHIFT-Wing sample 100件を使って GeoPT pretrained Transolver と scratch Transolver/GeoTransolver を比較している。

主要結果:

| Model | Test overall rel L2 | Macro field rel L2 | Pressure rel L2 | WSS vector rel L2 |
|---|---:|---:|---:|---:|
| A balanced: GeoPT pretrained + normalized MSE | 0.006949 | 0.105802 | 0.006949 | 0.107160 |
| B practical: scratch Transolver | 0.008472 | 0.112725 | 0.008471 | 0.111682 |
| D GeoTransolver scratch | 0.007152 | 0.106877 | 0.007152 | 0.107099 |

解釈:

- `A balanced` が同一Transolver構造のscratch baselineに対して最良。
- 公式raw relative L2 lossはPressure支配になりやすく、WSSが悪化した。Thermal側でも物理量スケール差を考慮したbalanced lossが必要。
- 強いscratch baselineであるGeoTransolverはAに接近したため、Thermal側でも「pretraining効果」と「architecture効果」を分ける実験設計が必要。

## 流用優先度A

| Asset | Path | Thermal側での用途 |
|---|---|---|
| 公式GeoPTモデル | `../GeoPT/vendor/GeoPT/models/Transolver.py` | Transolver-Base backbone、checkpoint key構造の参照 |
| GeoPT checkpoint | `../GeoPT/checkpoints/GeoPT_8layers.pt` | 流体GeoPT転用baseline G用 |
| 共通処理 | `../GeoPT/scripts/wing_geopt_common.py` | JSON flatten、field key解決、正規化、device/seed、relative L2 |
| 学習ループ | `../GeoPT/scripts/train_wing_geopt.py` | bf16 AMP、scheduler、checkpoint保存、balanced lossの実装参考 |
| 評価 | `../GeoPT/scripts/evaluate_wing_geopt.py` | best checkpoint評価、split別metrics保存 |
| 可視化 | `../GeoPT/scripts/visualize_pressure_predictions.py` | VTP/PNG出力の作法を温度場へ移植 |
| summary生成 | `../GeoPT/scripts/create_experiment_summary_docx.py` | 論文化向けの表・図のまとめ方 |

## 流用優先度B

| Asset | Path | Thermal側での用途 |
|---|---|---|
| split/audit | `../GeoPT/scripts/build_wing_splits.py` | D1/D2 dataset split、OOD split作成の参考 |
| VTP to NPY変換 | `../GeoPT/scripts/prepare_wing_geopt_npys.py` | OpenFOAM/FEM結果を点群配列へ変換する雛形 |
| GeoTransolver training | `../GeoPT/scripts/train_wing_geotransolver.py` | 強いscratch baseline用 |
| error hotspot analysis | `../GeoPT/scripts/analyze_pressure_error_hotspots.py` | 最高温度/hotspot誤差分析へ置換 |
| OOD distance analysis | `../GeoPT/scripts/analyze_ood_distance_vs_error.py` | 形状・条件距離と温度誤差の関係分析 |

## Thermal側で変えるべき点

- `Pressure/WSS` targetを `T`、必要なら `heat flux`、`region/material` targetへ置換する。
- `Mach/AoA/beta` conditionを `q`, `h`, `T_inf`, `k`, `alpha`, `source/sink patch` へ置換する。
- wing固有のleading-edge/wingtip hotspot logicは使わず、最高温度、source/sink近傍、fin base/fin tipなどの熱設計hotspotへ置換する。
- lossは raw relative L2 だけにしない。温度場Relative L2、最高温度補助損失、hotspot weighted lossを分ける。
- downstream mainは最初からCHTにしない。D1 solid conductionでpretextの有効性を先に切り分ける。

## 注意

`../GeoPT` の raw data、`.npy`、`.vtp`、`.pt`、PNG類はThermal_GeoPTの通常Gitにはコピーしない。必要な軽量コードだけを、出典が追える形で移植する。
