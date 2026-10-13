# material_measurement_diagnostic_v1 — 第一批：偏振测量链与架构限制诊断
（统计与面板收尾版 v3，2026-10-10）

本文件是 **material_measurement_diagnostic_v1 的收尾版记录**，取代此前的
v1/v2 记录内容；v1 原运行 `…/20261010T114012Z_3b8a0b92/` 与补修运行
`…/20261010T135018Z_1b290029/` 全部保留。本轮收尾后的正式运行：

- 运行目录：
  `analysis/runs/material_residual_runtime_v1/measurement_diagnostic_v1/20261010T145012Z_71caa5a0/`
- 验证目录：
  `analysis/runs/material_residual_runtime_v1/measurement_diagnostic_v1/20261010T145012Z_71caa5a0_validation/`
- 一次先行失败尝试（consume 相对路径缺陷，exit 3，未消费任何证据，
  保留作证）：`…/20261010T142500Z_94851eba/`

## 0. 撤回声明（累计）

1. （v1，保留有效）"q 有效集零翻转"、"最大 0.70/p95 0.30"、"上采样
   贡献 3–5 倍"、"±1px 实例级 0.001–0.002"、"m≥1 占 74%"、"测量影响
   小、建议 B" —— 全部于补修批撤回，理由见补修记录。
2. （本轮新增撤回）**"变化以零均值像素噪声为主"** —— 撤回。数据仅支
   持较弱命题：signed 均值差接近 0 是**正负变化相互抵消**的结果；"噪
   声"暗示的随机性与分布形态未经检验，不得再引用。
3. （本轮口径更正）**"lost 653,904"** —— 少计。求和曾只覆盖非空公共
   区实例；修正后对**全部实例×干预记录**求和（含公共区为空的记录，
   它们仍可能失去全部基准有效像素），正确值 **lost 665,160**。
4. （本轮口径更正）**"abs_dq p95 像素加权 0.0820"** —— 字段名误导：
   它是**逐实例 p95 的像素数加权平均**，不是池化像素 p95。已更名为
   `abs_dq_instance_p95_pixelcount_weighted`；池化像素 p95 标记
   **NOT_COMPUTED**（不引入新的分位数估算方案）。

## 1. 收尾内容（相对补修版）

1. `aggregate_paired_stats`：lost/gained 对全部实例记录求和（含公共区
   n=0 的记录；"基准有有效像素、干预全部无效"的反例测试覆盖）；仅均
   值与分位统计使用非空公共区记录；全部公共区为空时统计值 null、有
   效数量与翻转数量保持正确（测试覆盖）。
2. p95 字段更名并同步机器报告与本文档；池化像素 p95 NOT_COMPUTED；
   测试证明加权实例 p95 与池化像素 p95 数值可不同（构造反例）。
3. 最坏像素坐标与 |Δq| 面板限定 `instance_mask & baseline_qvalid &
   arm_qvalid`；面板 lost/gained/公共着色限定当前实例掩码；空公共区
   不产生峰值坐标；入口断言记录的 peak_abs_dq 与该实例配对 abs_max
   一致；反例测试覆盖"裁剪框内更大差异位于公共区之外"与"位于掩码之
   外"两种情形。
4. 报告同步：翻转总数明确为**实例×干预记录累计，非唯一像素数量**；
   臂/类别/组汇总按 GT/预测条件分开提供（`by_arm_per_conditioning`
   等），混合口径仅作总览（`overview_mixed_conditionings`，标注不得
   与分开口径并引）；保护范围明确为**已登记来源**——诊断配置与视觉
   观察JSON 本轮已登记（登记语义：配置在启动读取后立即记录身份，观
   察 JSON 在读取后立即记录；读取与登记之间的变更不可检测，已在窗口
   声明中披露），共 291 项。HANDOVER 仅追加更正，不删历史。

## 2. 任务 A 数字（不变，由 `residual_capacity_summary.json` 复算）

2446 实例、150 错误、**107 m≥1 = 71.3%**、43 m<1；缺失偏振错误 5；
m<1 且偏振可用 42。分折 840/47/37、711/52/35、895/51/35。流水线连接
180/368/380/346，匹配实例 15 错误（11 m≥1、4 m<1、0 fallback）。
m≥1 是逐样本 accuracy 能力上界，不是 Macro-F1 上界，不换算收益。

## 3. 任务 D 修正后数字（由 `recompute_paired_summary.json` 复算；
       本轮 lost/gained 求和修正，其余与补修版一致）

总口径：2025 个实例×臂记录、1942 个非空公共区、**44,106,789 公共像素**。

| 汇总 | 像素加权 MAE | 加权实例p95 | signed 均值差 | q lost/gained（记录累计） |
|---|---|---|---|---|
| overall | 0.0233 | 0.0820 | +0.00025 | **665,160** / 618,709 |
| upsample_control | 0.0153 | 0.0833 | +0.00096 | 25,576 / 32,698 |
| shift full_recompute（4 方向） | 0.0242–0.0298 | 0.0953–0.1013 | −0.0004–+0.0007 | 137,568–182,000 / 128,793–156,743 |
| shift fixed_baseline_warp（4 方向） | 0.0213–0.0219 | 0.0638–0.0678 | −0.0015–+0.0017 | 3,201–4,929 / 3,057–4,600 |

（分方向的臂级数字按条件分开见 JSON 的 `by_arm_per_conditioning`；
混合口径 `overview_mixed_conditionings` 仅作总览。）

按类别（像素加权 MAE，条件混合总览）：metal_submarine 0.0283、
real_fish 0.0251、plastic_submarine 0.0221、plastic_fish 0.0198、
d0_prediction 0.0211。

**口径声明**：signed 均值差接近 0 源于正负变化相互抵消；本数据不支
持关于变化分布形态（如"噪声"）的进一步表述。翻转总数是实例×干预
记录累计，不是唯一像素数量。p95 列是逐实例 p95 的像素数加权平均；
池化像素 p95 NOT_COMPUTED。

最坏实例（选择指标 = 实例公共区 max |Δq|，坐标已记录且与配对
abs_max 逐项一致）：shift full_recompute 1.8369 @
`Mixed_Metal_submarine_and_Plastic_fish_0_NTU_001` 实例1 (1027,865)；
upsample_control 1.7188 @ 同帧实例1 (1080,910)。6 张面板（条件×对比
族）含实例轮廓、双臂 q、公共区 |Δq| 热图与限定实例掩码的有效集变化
叠加。空公共区不产生峰值坐标（测试覆盖）。

引擎交叉核验 60/60 逐位一致；选帧与原运行 selection_manifest 逐帧逐
理由一致（`_validation/selection_match.json`）；保护 291 项已登记来源
首次读取后零变更；基准合成场景与任务 A/E 结论与补修版一致（任务 E
仍为 claude_visual_review，human_reviewed=false）。

## 4. 验证（真实退出码）

1. collect-only：**206 collected，exit 0**。
2. 聚焦四文件：**205 passed / 1 skipped / 0 failed，exit 0**（JUnit
   `focused.xml`）。
3. 正式运行：**exit 0，overall COMPLETED**（约 5.5 分钟）。
4. `git diff --check`：**exit 0**。

## 5. 建议与语义边界（不变）

唯一近期建议仍为 **A**（先最小生产修复上采样零值插值缺陷并做版本化
影响验证，再决定第二批）；依据与局限见补修版 §5。语义边界不变：路
径差异非精度声明；±1px 为敏感性干预；对照非默认修复；m≥1 为能力上
界；混合口径不得与分开口径并引；不宣称已登记来源之外的保护。

---

## 附录：补修版（v2）记录原文

以下为 v2（2026-10-10 补修批）记录原文，保留供追溯；口径与数字以上文 v3 为准（v2 的 lost 653,904 在 v3 修正为 665,160，p95 字段已更名，"零均值噪声为主"表述已撤回）。

## 0. 撤回声明（相对 v1 报告与 HANDOVER 旧批次记录）

以下 v1 结论**撤回**，原因与修正后数字见 §2/§3：

1. **"q 有效集零翻转"** —— 撤回。v1 只统计了帧级公共区的 q 翻转且边缘
   掩码用错基准视差；修正后逐实例配对统计显示：full_recompute 各平移臂
   在实例内 q 有效翻转 13.8万–18.2万 像素，upsample_control 臂 2.6万–
   3.3万，fixed_baseline_warp 臂 0.3万–0.5万（每方向，64 帧）。
2. **"最大 |Δsigned_q| 0.70 / p95 最高 0.30"** —— 撤回。那是帧级条带区
   （含背景）数字；限定实例公共区后，最坏实例 max |Δq| = 1.84
   （shift full_recompute）与 1.72（upsample_control），逐像素坐标已记录。
3. **"上采样链贡献约为 ±1px 平移响应的 3–5 倍"** —— 撤回。该比值来自
   帧级均值差（正负抵消为主），无意义。逐像素 MAE 口径下关系相反：
   upsample_control 实例公共区 MAE ≈ 0.015，低于 full_recompute 平移臂
   （≈0.024–0.030）。
4. **"±1px 平移实例级 |Δsigned_q| ≈ 0.001–0.002"** —— 数字本身是均值差
   （被抵消主导），不得再当作敏感性结论引用；修正口径为逐像素 MAE：
   fixed_baseline_warp ≈ 0.021、full_recompute ≈ 0.024–0.030。
5. **"m≥1 占错误的 74%"** —— 算术错误，正确值 **107/150 = 71.3%**。
6. **"测量影响小，可制定第二批尺度实验"（建议 B）** —— 撤回。该结论依赖
   上述被撤回的数字；重新给出的建议见 §5。

## 1. 补修内容

1. **逐实例配对统计**（新增 `instance_paired_comparison`）：公共区 = 固定
   instance_mask & baseline_qvalid & arm_qvalid（视差同构）；分别报告全实
   例/固定内部/固定边界的 signed Δq 均值（明确标注为可抵消的均值差，
   非像素误差）、|Δq| mean/p50/p95/max（均值即公共区逐像素 MAE）、
   Δabs_q 均值及其绝对差统计（特征幅值变化，非误差）、baseline/arm/
   common 有效数与 lost/gained；视差差异限定实例公共视差有效区。GT/预
   测条件分开；按类别/采集组汇总，像素加权与实例加权两种口径及权重定
   义写入 `recompute_paired_summary.json`。空公共区显式 n=0、统计 null。
2. **移位边缘无效处理修正**：full_recompute 的不可测边缘掩码改用
   `stereo3.disparity`（实际重匹配后采样的视差）计算，fixed_baseline_warp
   用基准视差；两者与各自 warp 的双线性采样一致，填充内容不计入有效 q。
   每臂记录 `edge_mask_source` 与 `edge_excluded_pixels`。反例测试覆盖
   基准不触及/重匹配触及及相反情形。
3. **来源绑定与保护窗口**：运行时三件证据（frames_complete /
   predictions_complete / layer_b_summary）的路径、字节长度与 SHA256 从
   **已字节核验的 artifact_manifest.files** 绑定；local feature_manifest 从
   **已核验的冻结运行时 YAML** `features.local_cache.files` 绑定；绑定在
   任何消费前核验，不匹配即拒绝。其余来源无冻结身份可绑定，在
   `source_manifest.json` 中逐项披露为 unbound。每个输入在**首次读取时**
   记录身份（`RunContext.consume` 即时哈希），结束时与首次记录比较；
   `protection_check.json` 的窗口声明明确为"首次读取后变更检测"，不再
   声称晚取快照覆盖此前读取。禁止把当前重算哈希当冻结预期（预期只来
   自已验证锚点）。
4. **最坏案例与报告**：按实例公共区真实 |Δq| 选最坏实例与像素（坐标落
   盘）；面板含实例轮廓、基准 q、干预 q、|Δq| 热图与有效集变化叠加
   （red=lost / blue=gained / 绿色 tint=common），每个条件×对比族一张。
   `diagnostic_report.md` 全部数字由本运行 JSON 机器产物自动汇总。
   30 分钟共享预算保留：超预算 PARTIAL + 非零退出 + 保留已完成证据。

## 2. 修正后的任务 A 数字（由 `residual_capacity_summary.json` 直接复算）

- outer_eval：2446 实例、150 错误、**107 m≥1 = 71.3%**、43 m<1；
  缺失偏振错误 5 个（4 个 m≥1）；m<1 且偏振可用 42 个。
- 分折：fold0 840/47/37、fold1 711/52/35、fold2 895/51/35。
- 流水线连接：180/368/380/346；匹配实例 15 错误（11 m≥1、4 m<1、
  0 fallback）。语义不变：m≥1 是逐样本 accuracy 能力上界，不是
  Macro-F1 上界，不换算为已实现收益。

## 3. 修正后的任务 D 数字（由 `recompute_paired_summary.json` 直接复算）

总口径：2025 个实例×臂记录、1942 个非空公共区、44,106,789 公共像素。

| 汇总 | 像素加权 MAE | p95 | signed 均值差 | q lost/gained |
|---|---|---|---|---|
| overall | 0.0233 | 0.0820 | +0.00025 | 653,904 / 618,709 |
| upsample_control | 0.0153 | 0.0833 | +0.00096 | 25,576 / 32,698 |
| shift full_recompute（4 方向） | 0.0242–0.0298 | 0.0953–0.1013 | −0.0004–+0.0007 | 137,568–182,000 / 128,793–156,743 |
| shift fixed_baseline_warp（4 方向） | 0.0213–0.0219 | 0.0638–0.0678 | −0.0015–+0.0017 | 3,201–4,929 / 3,057–4,600 |

按类别（像素加权 MAE）：metal_submarine 0.0283、real_fish 0.0251、
plastic_submarine 0.0221、plastic_fish 0.0198、d0_prediction 0.0211。
按采集组逐项见 JSON。signed 均值差处处接近 0（|·|≤0.0017）而 MAE
0.015–0.030——变化以零均值像素噪声为主，均值差口径不得再引用。

最坏实例（选择指标 = 实例公共区 max |Δq|，坐标见 `panels/worst_cases
.json`）：shift full_recompute 1.8369 @ `Mixed_Metal_submarine_and_
Plastic_fish_0_NTU_001` 实例1 像素 (1027,865)；upsample_control 1.7188
@ 同帧实例1 像素 (1080,910)。6 张面板（条件×对比族）含轮廓、双臂 q、
|Δq| 热图与有效集变化叠加。

引擎交叉核验 60/60 逐位一致；选帧与原运行 selection_manifest 逐帧逐
理由一致（`_validation/selection_match.json`）；基准合成场景结论不变
（常量平面反例 80 个被标有效像素偏差 0.375×d0 且对照精确恢复；原位
532 触及/520 恢复）。任务 A/B（saved/synthetic）与 15 例真实鱼审阅内
容与 v1 一致（审阅仍为 claude_visual_review，human_reviewed=false）。

## 4. 任务 E（不变，附于补修运行）

15 例：5 例 confidence_related（D1/D2 conf 0.063–0.173、IoU 0.66–0.81，
044 另被 D3 找回）、10 例任何条件无正重叠候选。视觉观察与待确认事项
见 `real_fish_review.csv` 与 observations JSON；不做根因断言。

## 5. 重新给出的唯一近期建议（不预设 B）

**建议 A：先对已确认的上采样零值插值缺陷做最小生产修复与版本化影
响验证，再决定是否制定第二批尺度实验。**

依据：
- 缺陷已确认（合成解析反例 + 生产匹配器原位归因 520/532），且其下
  游影响**未知而非已知为小**——v1 的"影响小"已撤回；放任已确认的
  测量链缺陷进入第二批实验会把一个可控混杂因素带进所有新结果；
- 残差实验的期望收益有硬上界：71.3% 的 P0 错误在冻结残差规则下不可
  纠正（本批算术，来自保存 logits，不受本轮修补影响），第二批若仍以
  残差头为中心，其可改善空间有限；
- 真实鱼召回问题主要在检测层（10 例无候选），与测量链独立，不因等
  待修复而恶化。

局限：修复的下游收益未测（需版本化前后对照，属修复验证的一部分）；
若用户判断 0.015 的实例公共区 MAE（q∈[-1,1] 尺度）对 Model B 可忽
略，B（直接进入第二批）仍然成立——该判断需要的是下游敏感性数据，
本批未获授权测量。C（测量可靠性不足）不成立：测量链的敏感性已完整
量化，缺的是修复动作而非更多诊断。
