# Thermal GeoPT 詳細研究計画書

## Thermal GeoPT: Diffusion-Lifted Geometric Pre-Training for Heat Transfer Surrogate Models

作成日: 2026-05-06  
想定実行環境: NVIDIA GeForce RTX 5070 Ti / ローカルストレージ約1TB  
対象成果物: 論文投稿を見据えた実験計画、実装要領、評価設計、リスク管理

---

# 0. エグゼクティブサマリー

本計画の結論は、Thermal GeoPT は実行可能であり、研究としての魅力も十分にある。ただし、添付計画案のまま「TDF = 境界までの距離 × 熱伝導率」「ランダム熱流束 q を入力して TDF 時系列を予測」と定義すると、熱方程式との対応が弱く、査読で突かれやすい。したがって、本計画では Thermal GeoPT を単なる「GeoPTの熱版」ではなく、放物型PDEである熱拡散に合わせた「Diffusion-Lifted Geometric Pre-Training」として再定義する。

中心仮説は次の通りである。

1. STLなどの形状のみから、Brownian random walk、境界到達、熱源・放熱境界の合成場を使った自己教師ありタスクを作ることで、温度場サロゲートの初期値として有効な幾何・境界相互作用の表現が学習できる。
2. 特に学習データが 10-100 ケース程度に限られる熱解析では、scratch の Transolver よりも Relative L2、最高温度誤差、収束速度、データ効率で優位になる可能性が高い。
3. 研究として最も強い主張は、「熱伝導・CHTに特化した拡散リフト事前学習は、静的TDF/SDF/VDF事前学習より有効であり、さらに流体GeoPT重みの単純転用より熱タスクに適合する」である。

RTX 5070 Ti は 16GB VRAM クラスであるため、元GeoPTと同じ 5TB 事前学習データ、A100 40GB 前提の規模をそのまま再現するのは不適切である。本計画では、事前学習データを fp16/Zarr 化し、点数を 9k-12k points/sample に抑え、事前生成とオンザフライ生成を併用する。最初の論文化に必要な検証は、Baseモデル中心、事前学習 0.16M-0.60M episodes、下流ラベル 50-300 ケース規模で成立させる。

最初の投稿先は、トップML本会議よりも、NeurIPS/ICLR/ICML workshop、ML4Science系、または Applied Thermal Engineering / IJHMT 方面を第一候補とするのが現実的である。本会議級を狙う場合は、少なくとも2種類以上の熱タスク、強いアブレーション、複数seed、コード公開、外部データまたは再現可能なベンチマーク化が必要になる。

---

# 1. 添付資料から読み取れる前提

## 1.1 GeoPT論文の要点

添付PDFの GeoPT は、幾何形状だけを使った通常の geometry-only pre-training が物理タスクに必ずしも有効でなく、むしろ negative transfer を起こし得る点を出発点としている。その対策として、形状 G にランダム速度場 V を加え、幾何特徴量の軌跡を自己教師として予測する「dynamics-lifted geometric pre-training」を提案している。

重要な設計要素は次の通りである。

| 観点 | GeoPTの設計 | Thermal GeoPTでの含意 |
|---|---|---|
| 事前学習データ | ShapeNetの車・航空機・船舶など、1万以上の形状 | 熱ではヒートシンク、電子機器、ポンプ、翼、CAD生成形状を混ぜる |
| 合成ダイナミクス | ランダム速度場による粒子軌跡 | Brownian random walk、境界到達、熱源/放熱境界への到達過程 |
| 教師信号 | Vector Distance Field の時系列 | 熱拡散特徴、境界到達特徴、熱源影響特徴の時系列 |
| 下流条件の入力 | 流速・方向などを velocity prompt として与える | 熱源、熱流束、h、T_inf、k、alpha、流入速度を thermal prompt として与える |
| 主張 | 20-60%のラベル削減、収束加速、geometry-onlyより有効 | 熱タスクで同様のデータ効率改善を検証する |

GeoPTの実装上の前提は、本研究の規模設計で特に重要である。元論文では、Base 3M、Large 7M、Huge 15M の Transolver 系バックボーンを用い、A100 40GB GPUで実験している。事前学習データは約5TBであり、元設定をローカル 1TB / RTX 5070 Ti にそのまま移植するのではなく、データ表現・点数・episode数を削る必要がある。

## 1.2 添付マークダウン計画案の要点

添付計画案は、GeoPTの STL-only self-supervision を熱解析サロゲートへ転用し、TDF、ランダム熱流束、拡散トラジェクトリ、Transolverベースのファインチューニング、OpenFOAM CHTデータによる評価を構想している。研究の方向性は妥当であるが、次の点を修正する必要がある。

| 原案の要素 | 評価 | 修正方針 |
|---|---|---|
| TDF = 距離 × 熱伝導率 | 物理次元と意味が曖昧。kが大きいほど熱抵抗は小さいため、単純積は不自然 | d/k、d^2/alpha、heat-kernel proximity などを含む多チャネル特徴へ拡張 |
| ランダム熱流束 q を直接入力 | qは境界上の場であり、点ごとの入力表現が必要 | 境界RBF場、source/sink patch、nearest-boundary projection、global condition tokenで表現 |
| ランダムウォークが境界到達で停止 | 吸収境界に対応し、断熱境界では反射が自然 | 境界タイプを synthetic prompt として吸収・反射・Robin近似を切替える |
| CHTのみで評価 | CPUデータ生成が重く、失敗時の原因分解が難しい | まず solid conduction、次に simplified CHT、最後にOOD形状へ段階化 |
| 500件CFDを想定 | 個人環境ではCPU時間・メッシュ品質がボトルネックになり得る | 10/25/50/100/200のデータ効率曲線を主軸にする |

---

# 2. 研究としての有効性・魅力

## 2.1 研究の魅力

Thermal GeoPT の魅力は、熱解析サロゲートにおける「ラベル生成の重さ」と「形状データの豊富さ」の非対称性を利用できる点にある。熱設計では、温度場や最高温度を得るために CFD/CHT/FEM を回す必要があり、形状変更のたびにメッシュ生成とソルバー実行が必要になる。一方で、ヒートシンク、筐体、フィン、ポンプ、翼、電子機器などの形状は CAD/STL として比較的容易に集められる。

GeoPTの流儀を熱へ移す意義は、単なる転用ではなく「双曲型・輸送型の軌跡学習」から「放物型・拡散型の境界相互作用学習」へ一般化する点にある。これは自己教師あり学習としても、物理サロゲートとしても主張しやすい。特に熱拡散では、温度場が境界条件、熱源、材料、距離、熱抵抗、拡散時間スケールの影響を強く受けるため、形状のみの静的距離場より、境界到達と拡散スケールを含んだ事前学習の方が下流タスクに適合するという仮説が立てやすい。

## 2.2 論文上の中心主張

本研究の中心主張は、次の3段階に整理する。

1. Static geometry pre-training is not enough for heat transfer surrogate models.  
   SDF/VDF/TDFの静的予測だけでは、熱源・放熱境界・材料・拡散時間スケールとの結合が学べず、下流の温度場予測には限定的または逆効果である。

2. Diffusion-lifted pre-training provides a heat-aware geometric prior.  
   Brownian random walk、境界到達、熱源/放熱境界の合成場を使うことで、幾何と熱駆動条件の結合表現を、物理ソルバーなしで学習できる。

3. The prior improves data efficiency and hotspot reliability.  
   少数の下流ラベルで Relative L2 と最高温度誤差を改善し、同等精度に必要なOpenFOAM/FEMケース数を削減できる。

## 2.3 想定される査読上の強み

| 強み | 内容 |
|---|---|
| 明確な未開拓性 | GeoPTのリフト事前学習を熱拡散・CHTに合わせて再設計する点が新しい |
| 工学的価値 | 熱設計で重要な最高温度、hotspot、放熱境界の予測効率に直結する |
| 個人環境でも成立 | 大規模HPCなしで、STL-only pretraining + 小規模CHT検証ができる |
| アブレーションしやすい | static TDF、Brownianなし、boundary heatなし、GeoPT流体重み転用との比較が可能 |
| 負の結果も意味がある | static geometry pretrainingが熱でも弱いことを示せれば、GeoPT文脈に接続できる |

## 2.4 想定される弱点

| リスク | 詳細 | 対策 |
|---|---|---|
| 「熱流束を入れたランダムウォーク」が本当に熱方程式か | Neumann/Robin境界の厳密な確率表現は複雑 | Dirichlet/absorbing proxy、反射境界、local-time proxyを分け、pretextであることを明示する |
| CHTデータ生成が重い | OpenFOAMのメッシュ生成・収束が詰まる | 最初はsolid conductionを主実験、CHTは小規模副実験にする |
| 1TB制約 | 元GeoPTの5TBデータは保存不可 | fp16/Zarr、点数削減、episode削減、オンザフライ生成を採用 |
| トップML会議には実験規模が弱い | RTX 5070 Ti単体ではスケール実験が限られる | Workshop/熱工学誌を第一目標、本会議は追加タスク後に狙う |
| Thermal-specific pretrainingが流体GeoPT転用に負ける可能性 | 熱タスクでも幾何一般表現が効く可能性 | 流体GeoPT重みを必須ベースラインにし、負けた場合も分析論文にする |

---

# 3. 改訂版 Thermal GeoPT の技術定義

## 3.1 名称

推奨する正式名称は次のいずれかである。

- Thermal GeoPT: Diffusion-Lifted Geometric Pre-Training for Heat Transfer Surrogates
- Thermal GeoPT: Brownian-Lifted Pre-Training for Neural Heat Transfer Simulation
- Diffusion-GeoPT: Solver-Free Geometric Pre-Training for Thermal Surrogates

本計画書では Thermal GeoPT を採用するが、論文タイトルでは Diffusion-Lifted という語を前面に出す方が良い。

## 3.2 熱版TDFの再定義

原案の TDF = distance × conductivity は単一スカラーで、熱拡散との対応が弱い。改訂版では、TDFを「Thermal Diffusion Feature」として多チャネル化する。

各点 x に対して、最寄り境界点を pi_G(x)、最寄り境界までのベクトルを r(x)=pi_G(x)-x、距離を d(x)=||r(x)|| とする。材料熱拡散率 alpha、熱伝導率 k、代表時間 t_j を synthetic prompt として与える。

推奨TDFチャネル:

| チャネル | 定義の例 | 物理的意味 |
|---|---|---|
| VDF | r(x)/L | 境界への方向と距離 |
| normalized distance | d(x)/L | 幾何近接性 |
| diffusion time scale | d(x)^2 / (alpha + eps) | 熱が届く時間スケール |
| heat-kernel proximity | exp(-d(x)^2 / (4 alpha t_j + eps)) | 時間 t_j での熱的近接性 |
| resistance distance | d(x)/(k + eps) | 均質材料での近似熱抵抗 |
| boundary normal | n(pi_G(x)) | 境界条件・放熱方向の手掛かり |
| source/sink proximity | exp(-d_source^2/sigma^2), exp(-d_sink^2/sigma^2) | 熱源・放熱境界との結合 |

この設計により、TDFは単なる距離場ではなく、熱拡散のスケール、材料、境界影響を含む特徴になる。論文では「TDFは物理解ではなく、熱拡散に整合する自己教師信号である」と明確に書く。

## 3.3 合成熱ダイナミクス

熱方程式の拡散過程は、確率過程として Brownian motion と対応付けられる。したがって、GeoPTの直線的な粒子移動 x_{t+1}=x_t+v に対し、Thermal GeoPTでは次のようなランダムウォークを使う。

```
x_{t+1} = x_t + sqrt(2 * alpha * Delta_t) * epsilon_t
where epsilon_t ~ N(0, I)
```

境界に到達した場合は、synthetic boundary type b に応じて処理を変える。

| 境界タイプ | 処理 | 熱境界条件との対応 |
|---|---|---|
| absorbing | 境界点で停止し、hit pointを記録 | Dirichlet境界・熱浴の近似 |
| reflecting | 法線方向成分を反射 | Neumann断熱境界の近似 |
| partially absorbing | 確率 p_absorb で停止、残りは反射 | Robin/対流境界の近似 |
| source boundary | hit pointの q_b を記録 | 熱流束・発熱面の影響 |

重要なのは、事前学習で厳密な熱解析を解くのではなく、幾何と境界条件の相互作用に関する表現を作ることである。下流のOpenFOAM/FEMラベルで最終的な物理写像へ適応する。

## 3.4 ランダム熱源・熱流束場の表現

ランダム熱流束 q を単一の数値として与えるのではなく、境界上の関数 q_b(s) として定義する。実装では、境界面に RBF または patch basis を置き、低ランクな場として表現する。

```
q_b(s) = q0 + sum_{m=1}^{M} a_m * exp(-||s-c_m||^2 / (2 sigma_m^2))
```

推奨設定:

| 項目 | 初期値 |
|---|---:|
| RBF数 M | 8-32 |
| q0 | 0 または正値ベース |
| a_m | Uniform(-1,1) または LogUniform強度 |
| sigma_m | 0.05L-0.25L |
| 正規化 | shapeごとに q_b を [-1,1] へ |

各query pointには、次の condition features を与える。

| point-wise condition | 内容 |
|---|---|
| q_near | pi_G(x) 上の q_b |
| h_near | 合成対流係数 h_b |
| source distance | 熱源patchまでの距離 |
| sink distance | 放熱patchまでの距離 |
| alpha, k | 材料・拡散率 |
| boundary type code | absorbing/reflecting/Robin/source |
| optional global token | RBF係数列、source/sink patch ID |

## 3.5 事前学習タスク

事前学習は一つに絞らず、軽量なmulti-taskにする。これにより、どの成分が効いたかをアブレーションしやすくする。

### Task A: Diffusive TDF trajectory prediction

入力: x0, geometry tokens, alpha, boundary condition features, random seed/step embedding  
教師: random walk の各時刻 x_t における TDF(x_t)

```
Loss_A = MSE( F_theta(x0, G, z_T)_traj, {TDF(x_t)}_{t=0..tau} )
```

このタスクはGeoPTのVDF trajectoryに最も近く、まず実装すべき最小構成である。

### Task B: Boundary hitting prediction

入力: x0, geometry, alpha, boundary type  
教師: first hitting time、hit pointのVDF、hit boundary type、q_b(hit point)

```
y_hit = [tau_hit, r_hit, n_hit, q_b(s_hit), h_b(s_hit)]
```

これにより、熱源や放熱境界がどこにあり、どの点がどの境界から影響を受けやすいかを学習させる。

### Task C: Synthetic heat influence prediction

入力: x0, geometry, q_b, h_b, alpha  
教師: 1本または少数本のrandom walkから得た境界影響量

```
I(x0) = q_b(s_hit) * exp(-lambda * tau_hit)
```

これは厳密解ではないが、熱源境界からの影響が距離・拡散率・境界到達時間で弱まるという構造を与える。

### 推奨する初期損失

```
Loss_pre = 1.0 * Loss_A + 0.3 * Loss_B + 0.3 * Loss_C
```

初期論文化では Task A のみでも成立するが、熱らしさと査読耐性を高めるには Task B までは入れるべきである。Task C は余裕があれば導入する。

---

# 4. 下流タスク設計

## 4.1 全体方針

下流評価は、いきなり複雑な CHT だけで行わない。失敗時に「事前学習が悪い」のか「メッシュ・ソルバー・境界条件・データ不足が悪い」のか切り分けられないためである。以下の3段階にする。

| 段階 | タスク | 目的 | 論文化での役割 |
|---|---|---|---|
| D1 | Solid conduction on parametric heat sinks | 熱伝導タスクでの基本有効性を確認 | メイン実験 |
| D2 | Simplified CHT in channel + heat sink | 流体・固体連成に近い条件で確認 | 強い副実験 |
| D3 | Out-of-domain geometry transfer | 形状一般化を確認 | 論文の魅力を増す実験 |

## 4.2 D1: Solid conduction benchmark

最初の主実験。OpenFOAMの laplacianFoam 相当、FEM、または自作有限体積で、固体内の定常熱伝導を解く。

支配方程式の例:

```
-div(k grad T) = Q in Omega_s
Boundary:
  -k grad T dot n = q on heat-source patches
  -k grad T dot n = h(T - T_inf) on cooling patches
  grad T dot n = 0 on insulated patches
```

幾何:

| 形状族 | 件数目安 | パラメータ |
|---|---:|---|
| plate-fin heat sink | 200-400 | fin数、fin厚、fin高さ、base厚、gap |
| pin-fin heat sink | 200-400 | pin数、径、高さ、配置 |
| stepped block / bracket | 100-200 | 段差、穴、リブ |
| pump-like/simple casing | 50-100 | 曲率、肉厚、入口/出口径 |

推奨ケース数:

- Pilot: 50 train / 20 test
- Main: 300-800 total, split train candidates 10/25/50/100/200, test 50-100
- Mesh points per case: 20k-100k points after downsample

出力:

- T(x): volumeまたはsurface+volume温度
- T_max: 最高温度
- hotspot position: argmax T
- optional: grad T, boundary heat flux

このD1は、研究の「通る/通らない」を最も速く判定できる。CHTより物理は単純だが、熱拡散事前学習の効果を見るには十分である。

## 4.3 D2: Simplified CHT benchmark

D1で有望な結果が出た後、簡易CHTへ進む。PhysicsNeMoやOpenFOAMのヒートシンク例に近い、チャネル流れ中の3Dヒートシンクを対象にする。

設定:

| 項目 | 推奨 |
|---|---|
| 流れ | laminarまたは低Re RANSから開始 |
| 形状 | plate-fin / pin-fin / 3-fin |
| ケース数 | 80-200 total |
| train/test | train 20/50/100、test 20-50 |
| 出力 | solid temperature、surface temperature、fluid temperature optional |
| 条件 | inlet velocity, inlet temperature, heat flux, material k, h相当 |

CHTでは、温度場だけでなく流れ場も関係する。Thermal GeoPT単体の主張を明確にするには、最初は「流れ条件は固定または少数」「熱条件と形状変化に注目」とする。流速やReを大きく変えると、流体サロゲートの問題になり、Thermal pretrainingの効果が見えにくくなる。

## 4.4 D3: OOD geometry transfer

研究の魅力を高めるため、同じ温度場タスクで形状族を跨いだ評価を行う。

例:

| 学習 | 評価 | 目的 |
|---|---|---|
| plate-fin | pin-fin | フィン形状の外挿 |
| low fin count | high fin count | 複雑度外挿 |
| CadQuery heat sink | ShapeNet electronics/lamp | web形状への一般化 |
| solid conduction | simplified CHT | 物理条件の転移 |

OODで少しでも scratch より安定すれば、Thermal GeoPT の価値は大きく見える。

---

# 5. 事前学習データ計画

## 5.1 データソース

| ソース | 優先度 | 目的 | 注意点 |
|---|---:|---|---|
| CadQuery自動生成 | 最優先 | 熱タスクに近い形状を大量生成 | 再現性が高く、ライセンス問題が少ない |
| ShapeNet selected categories | 高 | 形状多様性 | ライセンス確認、mesh品質処理が必要 |
| ヒートシンク/電子機器CAD | 中 | 熱設計らしさ | GrabCAD等はライセンス確認必須 |
| Open-source benchmark shapes | 中 | 外部妥当性 | 形状数は限定的 |
| 3D生成モデル由来 | 低-中 | 将来拡張 | 品質管理が重要 |

実装上は、まず CadQuery で完全再現可能な 1,000-3,000 形状を作り、ShapeNetや外部形状は Phase 2 で混ぜる。これは、初期実験の失敗原因を減らすためである。

## 5.2 形状生成仕様

CadQueryで生成する形状族:

| family | number | parameters |
|---|---:|---|
| plate_fin | 800 | n_fins, fin_height, fin_thickness, base_thickness, gap, length, width |
| pin_fin | 800 | n_pins_x/y, diameter, height, stagger, base size |
| louver_fin_simple | 300 | louver angle, slot count, fin pitch |
| block_with_channels | 500 | channel count, hole radius, wall thickness |
| pump_like_casing | 300 | volute radius, inlet/outlet diameter, shell thickness |
| wing/extruded airfoil | 200 | chord, thickness, sweep, extrusion |

Total initial CAD shapes: 2,900  
Phase 2 with ShapeNet/electronics: +3,000-8,000  
Full feasible range: 5,000-12,000 shapes

## 5.3 点サンプリング

元GeoPTは 32,768 volume points + 4,096 surface points を使うが、RTX 5070 Tiでは重い。以下を標準とする。

| 設定 | volume points | surface points | 合計 | 用途 |
|---|---:|---:|---:|---|
| tiny | 2,048 | 512 | 2,560 | CI、debug |
| pilot | 4,096 | 1,024 | 5,120 | 100-500 shapes |
| base | 8,192 | 1,024 | 9,216 | 主実験 |
| base+ | 12,288 | 2,048 | 14,336 | 余裕がある場合 |

## 5.4 episode数

| Phase | shapes | dynamics/shape | episodes | 目的 |
|---|---:|---:|---:|---|
| P0 debug | 100 | 5 | 500 | 実装確認 |
| P1 pilot | 500 | 10 | 5,000 | 損失・速度確認 |
| P2 first result | 2,000 | 20 | 40,000 | 下流で有効性の初判定 |
| P3 main | 8,000 | 20 | 160,000 | 論文最低ライン |
| P4 expanded | 10,000-12,000 | 50 | 500k-600k | ML投稿向け強化 |

元GeoPTの 1.3M episodes には届かなくても、熱に近いCadQuery形状を混ぜれば、初期論文の有効性検証には十分である。特にGeoPT論文でも、動的trajectory数より base geometry diversity の影響が大きい傾向が示されているため、まず形状数を優先する。

## 5.5 1TBストレージ設計

推奨データ形式:

- Mesh: compressed npz or ply/stl + metadata json
- Pretrain: Zarr / HDF5, fp16, chunked by shard
- Downstream: npz or zarr, raw OpenFOAM結果は必要分だけ保存
- Checkpoints: last, best, selected epoch only

概算:

| 項目 | 推奨上限 |
|---|---:|
| raw/preprocessed meshes | 30-80 GB |
| pretrain shards | 100-350 GB |
| downstream solver raw | 150-250 GB |
| downstream processed npz | 30-100 GB |
| checkpoints/logs | 50-100 GB |
| temporary/safety margin | 150-250 GB |
| 合計 | 510-1,130 GB |

したがって、raw solver dumpを残し続けると1TBを超える。原則として、OpenFOAMの時系列・中間場は、NPZ/VTPへ変換した後に削除または外部退避する。

推奨保存ポリシー:

1. 事前学習は 160k episodes までは保存型、600k episodes 以上は一部オンザフライ生成。
2. OpenFOAM raw は latest 20 cases のみ保持し、処理済みNPZを正本にする。
3. 各実験の checkpoint は best と final のみ保存。
4. 乱数seed、geometry parameter、solver config は必ず保存し、raw meshやraw solver結果を再生成可能にする。

---

# 6. モデル・学習設計

## 6.1 バックボーン

第一候補は Transolver Base である。GeoPTがTransolverを標準バックボーンとして使っており、形状上の不規則点群を扱いやすいためである。

| モデル | パラメータ目安 | RTX 5070 Tiでの位置づけ |
|---|---:|---|
| Transolver-Base | 3M | 主実験。最優先 |
| Transolver-Large | 7M | P3以降、良い結果が出たら追加 |
| Transolver-Huge | 15M | 16GBでは非推奨。必要なら勾配checkpoint + 小点数 |
| GNO/GNOT/FNO系 | 変動 | 論文強化用の補助baseline |

## 6.2 入力特徴

### 事前学習時

各点 token の入力:

| group | channels |
|---|---|
| geometry | x, normal, surface/volume flag, distance-to-boundary optional |
| thermal material | alpha, k, rho_cp optional |
| synthetic boundary | q_near, h_near, T_inf_near, boundary type code |
| diffusion prompt | Delta_t, step scale, random walk scale |
| global condition | RBF coefficients or source/sink patch embedding |

### ファインチューニング時

| group | D1 solid conduction | D2 CHT |
|---|---|---|
| geometry | x, normal, region id | x, normal, solid/fluid region id |
| material | k, alpha | k_s, alpha_s, fluid properties optional |
| heat input | q, Q, source patch distance | heat source q/Q |
| cooling | h, T_inf, sink patch distance | inlet T, wall h/adiabatic flag |
| flow | none or dummy | inlet velocity magnitude/direction |
| output | T | T_solid, T_fluid optional |

## 6.3 学習条件

### 事前学習

| 項目 | 推奨初期値 |
|---|---:|
| precision | bf16またはfp16 AMP |
| batch size | 1-2 physical batch |
| grad accumulation | 8-16 |
| effective batch | 8-32 |
| optimizer | AdamW |
| lr | 1e-3から開始、必要に応じて3e-4 |
| schedule | cosine + warmup 5-10 epochs |
| epochs | pilot 50、main 100-150 |
| points/sample | 9,216 |
| tau | 2を標準、3はアブレーション |
| gradient checkpointing | Large以上で必須 |

### ファインチューニング

| 項目 | 推奨初期値 |
|---|---:|
| optimizer | AdamW |
| lr | 1e-3、難しければ3e-4 |
| schedule | OneCycleLR or cosine |
| epochs | 200 |
| batch size | 1 |
| points/case train | 20k-50kへdownsample |
| inference | full meshが無理ならchunk inference |
| loss | Relative L2 + maxT auxiliary + optional gradient loss |

## 6.4 損失関数

### 下流メイン損失

```
Loss_T = ||T_pred - T_true||_2 / ||T_true||_2
```

### 最高温度補助損失

```
Loss_max = |max(T_pred) - max(T_true)| / (|max(T_true)| + eps)
```

### hotspot重み付き損失

高温領域を重視する。

```
w(x) = 1 + beta * sigmoid((T_true(x) - percentile_90(T_true)) / s)
Loss_hot = mean( w(x) * |T_pred - T_true|^2 )
```

推奨初期設定:

```
Loss_fine = Loss_T + 0.2 * Loss_max + 0.1 * Loss_hot
```

評価指標としての最高温度誤差は必須だが、Loss_maxを入れすぎると場全体が悪化する可能性があるため、最初は0.2以下にする。

---

# 7. 実験設計

## 7.1 検証仮説

| ID | 仮説 | 判定方法 |
|---|---|---|
| H1 | Thermal GeoPT は scratch より少数データで高精度 | train cases 10/25/50/100/200 の曲線比較 |
| H2 | dynamics-lifted は static TDF より有効 | A vs C/Dの比較 |
| H3 | thermal-specific pretraining は流体GeoPT転用より熱に強い | A vs GeoPT-fluid checkpoint |
| H4 | 境界熱源/放熱promptが効く | q/hなしアブレーション |
| H5 | hotspot信頼性が上がる | maxT error, hotspot localization |

## 7.2 実験グループ

| Group | 内容 | 必須度 |
|---|---|---|
| A | Thermal GeoPT full: Brownian + TDF + boundary hit + q/h prompt | 必須 |
| B | Transolver from scratch, same downstream inputs | 必須 |
| C | Static TDF pretraining only | 必須 |
| D | VDF/SDF geometry-only pretraining | 必須 |
| E | Brownian trajectory without q/h boundary fields | 必須 |
| F | q/h boundary field without random walk | 推奨 |
| G | Fluid GeoPT checkpoint -> thermal fine-tune | 推奨 |
| H | Thermal GeoPT with tau=0/1/2/3 | 推奨 |
| I | Thermal GeoPT with fewer shapes/dynamics | 推奨 |
| J | FNO/GNO/GNOT baseline | 余裕があれば |

実験の最小セットは A/B/C/D/E である。論文投稿を見据えるなら G/H/I を追加する。

## 7.3 データ効率曲線

各タスクで、次のtrain sizeを固定する。

```
D1: 10 / 25 / 50 / 100 / 200 train cases, test 50-100
D2: 10 / 25 / 50 / 100 train cases, test 20-50
```

各点で 3 seeds を推奨する。計算量が厳しい場合は、まず seed=0 の全体傾向を出し、有望な箇所だけ 3 seeds に増やす。

## 7.4 評価指標

| 指標 | 定義 | 目的 |
|---|---|---|
| Relative L2 | ||T_pred-T_true||/||T_true|| | メイン精度 |
| MAE / RMSE | normalized temperatureで計算 | 直感的比較 |
| Max temperature error | |Tmax_pred-Tmax_true| | 熱設計上最重要 |
| Hotspot localization error | argmax位置の距離 | 最高温部位の信頼性 |
| Boundary/interface L2 | 熱源面・放熱面・固体流体界面で計算 | 境界近傍の品質 |
| Energy balance residual | 入熱・放熱・蓄熱の整合 | 物理妥当性 |
| Time-to-target | scratchの最終精度に達するepoch | 収束加速 |
| Data saving | 同等精度に必要なtrain cases削減率 | GeoPTとの比較主張 |

## 7.5 成功基準

論文化のGo判定:

| レベル | 条件 |
|---|---|
| Minimum publishable | D1でAがBより Relative L2 10%以上改善、または同精度に必要なデータを30%以上削減 |
| Strong workshop | D1とD2の両方でAがB/C/D/Eに勝ち、maxT errorも改善 |
| Strong journal | CHTで最高温度・hotspot・境界誤差が明確に改善し、工学的考察ができる |
| Top ML challenging | 複数タスク、複数形状族、外部データ、スケール則、理論接続、コード公開が揃う |

失敗判定:

- AがBに対して5%未満の改善しかなく、maxTも改善しない。
- AとCが同等で、Brownian/diffusion要素の寄与が見えない。
- 事前学習の損失は下がるが下流に転移しない。

この場合は、Task B/Cを強化する、boundary prompt表現を変える、下流D1を境界条件多様な設定へ変更する。

---

# 8. 実装要領

## 8.1 リポジトリ構成

```
thermal-geopt/
  configs/
    pretrain_tiny.yaml
    pretrain_base.yaml
    finetune_d1.yaml
    finetune_d2_cht.yaml
  data/
    meshes_raw/
    meshes_processed/
    pretrain_zarr/
    downstream_raw/
    downstream_npz/
  scripts/
    generate_cadquery_shapes.py
    preprocess_meshes.py
    sample_points.py
    generate_pretrain_episodes.py
    run_openfoam_d1.py
    run_openfoam_d2_cht.py
    convert_openfoam_to_npz.py
    train_pretrain.py
    train_finetune.py
    evaluate.py
    plot_data_efficiency.py
  thermal_geopt/
    geometry.py
    brownian.py
    tdf.py
    boundary_fields.py
    datasets.py
    models/
    losses.py
    metrics.py
  outputs/
    checkpoints/
    logs/
    figures/
```

## 8.2 実装順序

### Step 0: 環境確認

- CUDA/PyTorchがRTX 5070 Tiで動くことを確認。
- AMP bf16/fp16の動作確認。
- 9,216点、Transolver-Base、batch=1でforward/backwardが通ることを確認。
- 1ケースあたりのGPUメモリをログ保存。

### Step 1: Geometry preprocessing

処理:

1. STL/STEPを読み込み。
2. watertight check、法線修正、重複頂点削除。
3. 座標正規化: center、scale to unit/length=1、向き揃え。
4. surface point sampling と volume/shell point sampling。
5. nearest boundary projection、distance、normal計算。

推奨ライブラリ:

- trimesh
- open3d
- scipy.spatial.cKDTree
- pyembree or fcpw if available
- pymeshfix / pymeshlab optional

### Step 2: TDF計算の検証

単純形状で解析解と比較する。

| shape | 検証項目 |
|---|---|
| sphere | distance = |r|-R、normal = x/|x| |
| cube | face distance、corner/edge近傍の挙動 |
| cylinder | radial/axial distance |
| plate-fin | nearest surfaceが妥当か可視化 |

合格基準:

- sphere/cubeでdistance誤差が1e-3以下、またはmesh解像度に応じた妥当範囲。
- 法線の向きが一貫。
- inside/outside判定が95%以上安定。

### Step 3: Brownian trajectory generator

実装仕様:

```
for each point x0:
  x = x0
  for t in 0..tau:
    record TDF(x)
    eps = normal(0, I)
    x_next = x + sqrt(2*alpha*dt) * eps
    if segment intersects boundary:
       process by boundary type
    else:
       x = x_next
```

初期値:

| parameter | value |
|---|---:|
| tau | 2 |
| dt | 0.01-0.05 in normalized coordinates |
| alpha | logUniform(0.05, 2.0) after normalization |
| boundary mode | absorbing 40%, reflecting 40%, partial 20% |

### Step 4: Boundary heat field generator

実装仕様:

1. surface pointsから RBF centers をM個選ぶ。
2. q_b, h_b, source/sink codeを生成。
3. 各query pointに nearest boundary の q_near/h_near/source distance を付与。
4. global conditionとしてRBF coefficientsを保存するか、point-wise featureだけで開始する。

最初はpoint-wise featureだけでよい。余裕があればglobal tokensを実装する。

### Step 5: Pretrain dataset writer

Zarr shard例:

```
shard_00001.zarr/
  x:        fp16 [E, N, 3]
  normal:   fp16 [E, N, 3]
  cond:     fp16 [E, N, C_cond]
  y_tdf:    fp16 [E, N, C_y]
  mask:     bool [E, N]
  meta:     json
```

推奨shardサイズ:

- 128-512 episodes/shard
- 1 shard 50-250MB程度
- checksumを保存

### Step 6: Fine-tuning dataset writer

OpenFOAM/FEM結果を次の形式へ統一する。

```
case_000123.npz
  points:       float32 [N,3]
  normals:      float32 [N,3]
  region:       int [N]
  material:     float32 [N,Cm]
  bc_features:  float32 [N,Cb]
  T:            float32 [N,1]
  case_params:  json
```

下流では保存時float32、学習時にAMPで混合精度にする。

---

# 9. 具体的な実験スケジュール

## 9.1 12週間プラン

| 週 | マイルストーン | 完了条件 |
|---:|---|---|
| 1 | 環境構築・GeoPT/Transolverコード確認 | 5070 TiでTransolver-Base forward/backward成功 |
| 2 | CadQuery形状生成・mesh preprocessing | 100形状のTDF可視化、品質チェック完了 |
| 3 | Brownian/TDF episode生成 | P0 500 episodes生成、lossが下がる |
| 4 | D1 solver pipeline | solid conduction 50 cases生成、NPZ変換完了 |
| 5 | P1/P2事前学習 | 40k episodesでBase pretrain完了 |
| 6 | D1 pilot fine-tune | A/B/C/D/Eをtrain 10/25/50で比較 |
| 7 | Go/No-Go判定・手法修正 | Aがscratchより有望、または失敗原因を特定 |
| 8 | P3 main pretraining | 160k episodes、100-150 epochs実行 |
| 9 | D1 main experiments | 10/25/50/100/200、主要グループ完了 |
| 10 | D2 simplified CHT data | 80-120 cases生成、pilot完了 |
| 11 | D2 fine-tune / OOD | CHTまたはOODで追加結果 |
| 12 | 図表・論文骨子 | データ効率曲線、アブレーション表、可視化作成 |

## 9.2 Go/No-Goゲート

### Gate 1: Week 3

条件:

- TDFとBrownian trajectoryの可視化が直感に合う。
- pretraining lossが安定して低下。
- 9k points/sampleでVRAM OOMしない。

失敗時:

- pointsを5kへ削減。
- targetをTask Aのみにする。
- Large/Hugeは完全に捨てる。

### Gate 2: Week 6

条件:

- D1 pilotでAがBより10%以上改善、または収束が明確に速い。
- AがC/Dより良い傾向。

失敗時:

- boundary hit taskを強化。
- q/h promptをglobal token化。
- D1の境界条件多様性を増やす。

### Gate 3: Week 10

条件:

- D1 mainで統計的に改善。
- D2またはOODで少なくとも1つ補助結果がある。

失敗時:

- 投稿先をworkshop/short paperへ切替。
- 「熱拡散でのnegative/conditional transfer分析」に論点変更。

---

# 10. 実験図表計画

## 10.1 必須図

| Figure | 内容 |
|---|---|
| Fig. 1 | Thermal GeoPT概念図: STL -> Brownian lifted pretraining -> heat fine-tuning |
| Fig. 2 | TDF/Thermal diffusion featuresの説明図 |
| Fig. 3 | データ効率曲線: train cases vs Relative L2 |
| Fig. 4 | 収束曲線: epoch vs validation error |
| Fig. 5 | 温度場可視化: ground truth / scratch / Thermal GeoPT / error map |
| Fig. 6 | hotspot可視化: 最高温度と位置誤差 |
| Fig. 7 | アブレーション棒グラフ |
| Fig. 8 | pretraining scale: shapes/dynamics/points の影響 |

## 10.2 必須表

| Table | 内容 |
|---|---|
| Table 1 | 下流ベンチマーク仕様 |
| Table 2 | モデル・事前学習設定 |
| Table 3 | メイン結果: Relative L2 / maxT error |
| Table 4 | アブレーション |
| Table 5 | データ効率: 同精度に必要なケース数 |
| Table 6 | 計算コスト・ストレージ |

---

# 11. 投稿戦略

## 11.1 最初に狙うべき成果形態

現実的な第一成果は、次のいずれかである。

1. Workshop paper: 6-8 pages  
   Thermal GeoPTの概念、D1+D2 pilot、アブレーションをまとめる。

2. Engineering journal short/full paper  
   ヒートシンク/CHTの最高温度予測、データ削減、実装実用性を強調する。

3. arXiv + code release  
   GeoPT直後の拡張研究として素早く公開し、反応を見て本投稿へ拡張する。

## 11.2 投稿先別の必要条件

| 投稿先 | 必要条件 | 現計画との相性 |
|---|---|---|
| NeurIPS/ICLR/ICML main | 複数タスク、強baseline、理論、スケール、再現性 | 初回は難しめ |
| ML4Science / AI4Science workshop | 新規性と初期実証 | 高い |
| IJHMT | 熱工学の妥当性、CHT結果、最高温度、物理考察 | 高い |
| Applied Thermal Engineering | 実用設計・サロゲート性能 | 高い |
| Computer Methods in Applied Mechanics and Engineering | 数値解析・サロゲート・厳密性 | 中-高 |

## 11.3 論文タイトル案

- Thermal GeoPT: Diffusion-Lifted Geometric Pre-Training for Heat Transfer Surrogate Models
- Solver-Free Diffusion-Lifted Pre-Training for Neural Thermal Simulation on Complex Geometries
- Learning Heat-Aware Geometric Priors from STL-Only Data

## 11.4 論文構成案

1. Introduction
   - 物理サロゲートのデータボトルネック
   - GeoPTの貢献と限界
   - 熱拡散への拡張の必要性

2. Related Work
   - neural operators / Transolver
   - heat transfer surrogate / CHT surrogate
   - self-supervised geometry pretraining
   - stochastic representation of heat diffusion

3. Method
   - Diffusion-lifted geometric pretraining
   - Thermal diffusion feature/TDF
   - Brownian trajectory and boundary interaction tasks
   - Fine-tuning thermal prompt

4. Experiments
   - D1 solid conduction
   - D2 simplified CHT
   - OOD transfer
   - baselines and metrics

5. Results
   - data efficiency
   - convergence
   - hotspot reliability
   - ablation

6. Discussion
   - 何が効いたか
   - Neumann/Robinの近似の限界
   - 個人環境でのスケーリング

7. Conclusion

---

# 12. リスク管理

| リスク | 兆候 | 対応 |
|---|---|---|
| pretraining lossは下がるがfine-tuneに効かない | AとBが同等 | targetが下流とずれている。Boundary hit/source influenceを増やす |
| static TDFがfullと同等 | Brownianの寄与がない | tau、dt、boundary type、hit predictionを見直す |
| maxTが改善しない | L2だけ改善 | hotspot-weighted loss、source boundary samplingを増やす |
| OpenFOAMが収束しない | CHT case失敗率が高い | D1中心に戻し、D2はケース数を減らす |
| 1TBを超える | raw solver dumpが増える | raw削除、processed正本化、pretrain episodeをオンザフライ化 |
| VRAM OOM | 9k pointsで失敗 | 5k points、gradient checkpointing、chunk training |
| GeoPT-fluid転用に負ける | GがAより良い | thermal-specificの設計を再検討。流体+熱のhybrid pretrainingへ変更 |

---

# 13. 最小実行コマンド案

以下は実装後のコマンド設計例である。

```bash
# 1. Generate CAD shapes
python scripts/generate_cadquery_shapes.py \
  --families plate_fin,pin_fin,block_channel,pump_like \
  --n-total 3000 \
  --out data/meshes_raw

# 2. Preprocess meshes
python scripts/preprocess_meshes.py \
  --input data/meshes_raw \
  --out data/meshes_processed \
  --normalize unit_length \
  --repair true

# 3. Generate pretraining shards
python scripts/generate_pretrain_episodes.py \
  --meshes data/meshes_processed \
  --out data/pretrain_zarr/p3_base \
  --n-shapes 8000 \
  --n-dyn 20 \
  --n-volume 8192 \
  --n-surface 1024 \
  --tau 2 \
  --dtype fp16 \
  --tasks tdf,hit

# 4. Pretrain
python scripts/train_pretrain.py \
  --config configs/pretrain_base.yaml \
  --data data/pretrain_zarr/p3_base \
  --model transolver_base \
  --amp bf16 \
  --grad-accum 8

# 5. Generate downstream D1 data
python scripts/run_openfoam_d1.py \
  --geometries data/meshes_processed/heat_sinks \
  --n-cases 800 \
  --out data/downstream_raw/d1

python scripts/convert_openfoam_to_npz.py \
  --input data/downstream_raw/d1 \
  --out data/downstream_npz/d1 \
  --delete-raw-after-convert false

# 6. Fine-tune
python scripts/train_finetune.py \
  --config configs/finetune_d1.yaml \
  --pretrained outputs/checkpoints/pretrain_base_best.pt \
  --train-size 50 \
  --seed 0 \
  --method thermal_geopt_full

# 7. Evaluate and plot
python scripts/evaluate.py --run outputs/runs/d1_train50_seed0
python scripts/plot_data_efficiency.py --runs outputs/runs --out outputs/figures
```

---

# 14. 推奨する初回実験の具体設定

最初の4週間で実施する最小構成は以下である。

| 項目 | 設定 |
|---|---|
| 形状 | CadQuery heat sink 500 shapes |
| pretrain episodes | 5,000-10,000 |
| points | 5,120 |
| tasks | TDF trajectory + boundary hit |
| model | Transolver-Base |
| downstream | solid conduction 100 cases |
| train sizes | 10, 25, 50 |
| groups | A full, B scratch, C static TDF, E no q/h |
| seeds | 1 initially, promising points only3 |
| success | train 25 or 50でAがBより10%以上良い |

このミニ実験で効果がない場合、いきなり大規模事前学習へ進むべきではない。下流タスク・prompt表現・pretextを修正してからP3へ進む。

---

# 15. 研究の最終到達点

理想的な最終成果は次である。

1. STL-only、solver-free の Thermal Diffusion-Lifted Pretraining を提案。
2. static geometry pretraining、流体GeoPT転用、scratchと比較し、熱タスクで優位性を実証。
3. solid conduction と simplified CHT の両方で、少数ラベル時のRelative L2、maxT、hotspotを改善。
4. 1TB/単一GPU環境でも再現可能な、resource-aware pretraining recipe を提示。
5. 形状数・動的episode数・TDFチャネル・boundary hit task の寄与をアブレーションで示す。

主張は次のようにまとめると強い。

「GeoPTの本質は速度場そのものではなく、形状を下流物理で必要な dynamics-coupled space へ持ち上げる点にある。熱解析では、その持ち上げは移流軌跡ではなく、Brownian拡散、境界到達、熱源/放熱境界の影響として設計すべきである。Thermal GeoPTはこの設計により、熱解析サロゲートのラベル効率とhotspot信頼性を改善する。」

---

# 参考情報

- 添付PDF: GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training, arXiv:2602.20399v1.
- 添付マークダウン: 熱GeoPT 研究計画書.
- GeoPT official repository: Physics-Scaling/GeoPT.
- NVIDIA GeForce RTX 5070 Family specifications.
- OpenFOAM official documentation.
- NVIDIA PhysicsNeMo conjugate heat transfer and heat sink examples.
