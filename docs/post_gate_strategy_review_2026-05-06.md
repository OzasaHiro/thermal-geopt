# Post-Gate Strategy Review 2026-05-06

## Executive Decision

今回の label-scarcity gate は、事前定義した成功条件では **未達** と判断する。

- 25 labels: pretrained 系は scratch と同等または悪化。
- 50 labels: 最大改善は `no_boundary_field` の `0.31%` 程度で、実用差とは言いにくい。
- 100 labels: `no_boundary_field` のみ `22.59%` 改善し、validation/test ともに強いシグナル。

したがって、現時点で言えることは次の通り。

1. 現行 `full` pretraining をこのまま P3/P4 へ大規模化しない。
2. ただし Thermal GeoPT 仮説を棄却しない。
3. `100 labels + no_boundary_field` は要追試の有望シグナルとして扱う。
4. 次は、再現性確認と GeoPT 本来の dynamics-lifted pretext への再設計を優先する。

## Why This Is Not A GeoPT Rejection

GeoPT の本質は、単なる checkpoint 初期化ではない。

- large-scale STL-only self-supervision
- dynamics-lifted geometric pretraining
- downstream physics prompt への転移
- label efficiency improvement

今回の pretraining pipeline は Brownian trajectory / hit mask を生成・保存しているが、学習では `y_tdf` の静的特徴回帰だけを使っている。つまり、現行 gate はまだ「GeoPT 本来の dynamics-lifted pretraining」の評価ではなく、「TDF/近傍境界幾何特徴による backbone 初期化」の評価に近い。

このため、gate 未達をもって Thermal GeoPT の研究仮説を否定するのは早い。

## Technical Interpretation

### Full Pretext

`full` が 25/50/100 labels で scratch を上回らない点は重要である。

現在の pretraining prompt は `alpha, conductivity, q_near` で、`q_near` はランダム RBF 境界場の nearest-boundary 値である。一方、downstream D1 proxy の condition は `conductivity, source_temperature, sink_temperature, source_patch, sink_patch, nearest_boundary_distance` である。

この schema mismatch により、`q_near` は downstream transfer では情報ではなくノイズまたは不要な依存になっている可能性が高い。

### No-Boundary-Field

`no_boundary_field` は 10/25/50 labels ではほぼ差がないが、100 labels では大きく勝った。

これは次のどちらかである。

- 本物のシグナル: 境界場 prompt を消した拡散幾何 prior が、100 labels 程度で効き始める。
- 偶然/単一 split 効果: 100 labels subset, seed 42, checkpoint selection, evaluation sampling に依存した外れ値。

現時点では後者を除外できない。よって、主張ではなく replication candidate として扱う。

### Static TDF

`static_tdf_only` が伸びないことは、単なる VDF/distance/normal の初期化では D1 source/sink temperature prediction に十分でないことを示唆する。

Thermal GeoPT の主張を立てるには、静的 geometry-only ではなく、source/sink reachability, boundary interaction, diffusion time, thermal resistance を dynamics-lifted に学習させる必要がある。

### Downstream Proxy

現 D1 proxy は smoke/gate としては有用だが、FEM/FVM/OpenFOAM の熱伝導解ではない。

scratch が 10/25/50 labels で relative L2 約 `0.072` に張り付き、事前学習差がほとんど出ていない。これは downstream target が簡単すぎる、または入力 condition が target を直接支配している可能性を示す。

## Next Experiment Plan

### Phase R0: Replicate The 100-Label Signal

目的: `100 labels + no_boundary_field` の `22.59%` 改善が再現するか確認する。

最小設計:

| Item | Setting |
|---|---|
| groups | scratch, no_boundary_field |
| train sizes | 50, 75, 100, 125 |
| split seeds | 5 seeds |
| downstream epochs | 20, 40 |
| eval | fixed test, fixed point budget; if possible 8192/all points check |

判断:

- 100/125 labels で複数 seed にわたり `>10%` 改善するなら、pretraining signal あり。
- 100 labels seed 42 だけなら、現結果は偶然扱い。
- 50/75 から滑らかに改善が立ち上がるなら、label-efficiency claim に近づく。

### Phase R1: Restore Dynamics-Lifted Pretraining

目的: GeoPT 本来の仮説を検証可能にする。

現状は `trajectory`, `hit_mask`, `hit_step` を生成しているが学習に使っていない。次の multi-task target を導入する。

- static TDF auxiliary
- Brownian next-step displacement or final displacement
- boundary hit probability
- hit step / survival time
- source/sink reachability or proximity transition

この phase では、TDF は主目的ではなく auxiliary に落とし、boundary interaction / diffusion trajectory を主目的に置く。

### Phase R2: Align Pretraining Prompts With Downstream Conditions

目的: `q_near` のような downstream と不整合な prompt を避ける。

比較候補:

| Prompt mode | Meaning |
|---|---|
| zero_all | prompt なし |
| alpha_k_only | alpha, conductivity only |
| source_sink_only | source/sink prompt only |
| aligned_boundary_conditions | downstream と同型の source/sink/temperature/patch |
| random_q_near | 現行 full の対照 |

特に、現行 `full` は「境界条件を使う pretraining」とは言いにくい。次は downstream と対応する thermal boundary prompt に揃える。

### Phase R3: Replace D1 Proxy With A Harder Thermal Target

目的: proxy が簡単すぎて pretraining 差が見えない問題を解消する。

段階:

1. 現 D1 proxy は smoke/gate 継続用として残す。
2. steady conduction / Poisson solve の数値解 target を追加する。
3. 可能なら `laplacianFoam` または軽量 FEM/FVM で D1 solid conduction を作る。
4. その後に D2 simplified CHT へ進む。

## Go / No-Go Criteria

### Go To Scale-Up

次のどちらかを満たす場合のみ、500-2,000 shapes / 5k-40k episodes 以上へ進む。

- 25/50 labels で scratch に対して relative L2 が約 10% 改善。
- 同等 error に到達する downstream epoch が明確に少ない。

### Hold / Redesign

- 100 labels だけで勝つ。
- seed によって符号が変わる。
- full pretext が no-boundary より常に悪い。

この場合は scale-up ではなく pretext/prompt/downstream を再設計する。

### Stop Current Pretext

redesigned pretext でも 25/50/100 labels のすべてで scratch と同等以下なら、現行 Thermal TDF pretraining line は止める。

## Immediate Implementation Tasks

1. Multi-seed label-scarcity split generator.
2. Matrix runner with separate `split_seed` and `train_seed`.
3. Paired case-wise summary with mean/std/95% CI.
4. Fixed-eval sampling or full-point evaluation mode.
5. Dynamics-lifted pretraining dataset mode using `trajectory`, `hit_mask`, `hit_step`.
6. Prompt schema redesign for downstream-aligned thermal boundary conditions.

## Bottom Line

今回の gate は negative だが、研究停止ではない。

より正確には、現行 full pretext は GeoPT らしい有効性を示しておらず、むしろ boundary prompt 設計の不整合を示唆している。一方で、`no_boundary_field` の 100-label 改善は、拡散幾何 prior が効き得る可能性を示している。

次は大規模化ではなく、再現性確認、dynamics-lifted pretext 復元、prompt/downstream 整合化を進める。
