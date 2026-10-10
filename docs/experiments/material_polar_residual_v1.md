# material_polar_residual_v1 — 固定灰度基线，验证偏振增量（协议冻结）

状态：**协议与判定已冻结；P0 基线复算一致；残差训练入口尚未实现**（下一批交付，
训练由用户手动执行）。本批不训练、不修改分类算法、不重建框架。

## 1. 动机与设计

P2（G+C+S）未优于 P0（G）而 P1（G+C）与 P0 相当：本轮配置与现有数据下
未观察到稳定净收益，**不能据此证明偏振不存在可利用信息**。本协议改为
**残差修正**：灰度基线 P0 完全冻结，只训练一个
14→4 零初始化线性残差头，从 C/S 标准化槽预测对 P0 logits 的修正量：

```
delta = valid_s * 0.5 * tanh(W x)        # W: 14→4, 无 bias, 零初始化
final_logits = P0_logits + delta
loss = CE(final_logits, label) + 0.01 * mean(delta²)
```

- `x` 是该折已保存 scaler 变换后的 14 维 C/S 槽（C 6 维 + S 8 维，定义与
  material_existing_data_v1 完全相同，一个字段不改）。
- `valid_s` 是缓存的 s_valid 行（0/1）：**无效 S 样本在 RC/RCS 下都精确回退
  P0**（delta 恒为 0），不伪造测量。
- epoch 0 时 W=0 ⇒ delta=0 ⇒ 与 P0 精确一致，"不修正"恒为候选。
- 无学习型门控、无隐藏层、不解冻 P0、不做全数据 refit。

变体：**R0** = P0（对照）；**RC** = P0+C 修正（14 维槽中 S 段输入置零）；
**RCS** = P0+C+S 修正。三者共享同一 14 维输入槽、同一初始化与预算。

## 2. 来源绑定（全部经实际核验）

| 来源 | 路径 | 身份 |
|---|---|---|
| 协议 | `…/20260930T021500Z_a3f7c1/protocol/protocol.json` | 摘要 `0de5cf8a6d373eae…`（自洽校验通过） |
| 特征缓存 | `…/20260930T045500Z_c5d9e8/upgrade/corrected_features/` | cache_binding.json：cache `c2fd11bd…` / definition `e312a1ca…` / manifest `7e9dde71…`（与原始缓存逐字节一致） |
| P0 拟合 | `…/manual_20261008_085510_822/fit/` | summary SHA256 `e54984a0d29755ce…` |
| 基线核验 | `…/20261008T093000Z_f4b1a7/baseline/p0_baseline_verification.json` | `verify_p0_baseline.py` 逐折复算一致 |

数据：V5 `datasets/underwater_cls_fusion_v5_instance_mask`（指纹
`bb9e3661fb1e71e9…`）；开发池 = 原 train+val（2446 样本），test 仅排除成员。

## 3. P0 基线核验结果（`scripts/verify_p0_baseline.py` v2，exit 0）

复用既有准入链（协议自洽 → 缓存绑定逐字节 → 逐折权威索引 → 清单行序 →
数组核验），加载每折**已保存**的 scaler（不重新拟合）与 P0 线性头
（CPU），对每折 outer_eval 行做线性前向。核验范围（以实际实现为准）：

- **已核验身份**：逐样本 sample_name/group/class_id 与权威缓存清单及折
  角色逐行一致（行序即冻结契约，重复/缺失即拒绝）；checkpoint 的
  variant/fold/seed/输入维度/类别数与权重形状；fold report 与 summary 的
  fold/variant/support/selected_epoch；admission 的类别顺序、特征定义
  版本、g_dim 与预处理身份。
- **已复算数值**：outer_eval logits/预测；每折 Macro-F1、accuracy、
  各类召回与混淆矩阵；三折平均；合并外层指标（来自已校验的内存标签/
  预测，并与 summary.paired 中已保存的 P0 pooled 值交叉核对）。
- **选模记录**：history.selection 与 fold report、summary 三方一致（含
  内层 inner_val_macro_f1/inner_val_accuracy 的逐项比对），并按冻结规则
  对**完整历史记录**只读复算最佳 epoch 交叉核对；历史完整性按已准入的
  训练预算 max_epochs 核验（数量、编号集合 1..max_epochs、生产升序格式、
  checkpoint 与 summary 两处预算一致——单改一处不能使截断历史合法）。
- **折集合与 admission 身份**：summary.oof.P0.folds 在逐折循环前一次性
  核验（fold 严格来自协议，缺失/额外/重复/非法记录拒绝；variant、
  support=权威 outer_eval 数量、selected_epoch 严格整数）；admission 的
  input_dim 必须等于缓存 G+C+S 宽度且为严格正整数，imgsz/g_dim/类别
  顺序/预处理等已保存字段不得通过删除或 None 跳过检查。
- **内存汇总**：pooled 指标直接消费 verify_fold 返回的已复算、已校验
  labels/predictions（合并样本数 2446 与各折覆盖记录在案），不再重新
  读取预测 JSONL 文件。
- **scaler 全列检查**：三折全部 VERIFIED，与权威 inner_train 行（G/C 用
  全部行，S 仅有效行 1061/1206/1138）只读复算的 270 列 mean/std/usable
  **逐位一致（最大差异 0）**。
- **容差**：logit atol 1e-4（float64 累加转 float32 的复算与 PyTorch
  float32 运算**不保证逐位相同**），指标 atol 1e-6，scaler atol 1e-6；
  实测 logits 最大差 5.7e-06，指标与 scaler 均为逐位一致。容差记录在
  报告 `tolerances`，未放宽。
- **语义**：REFUSED(1)=身份/结构/绑定/非有限输入/scaler 来源不符/缺文件；
  MISMATCH(2)=合法有限的保存值与复算不一致；两者由异常类型决定，不解析
  错误文案。非有限值（NaN/±Inf）在任何减法/argmax/指标计算前即拒绝。

证据：`…/20261008T140000Z_a9b8c7/baseline_v3/p0_baseline_verification.json`
（前两轮核验 `…/20261008T093000Z_f4b1a7/baseline/`、
`…/20261008T120000Z_c8d2e6/baseline_v2/` 保留可追溯，只读）。

| 折 | outer_eval 行数 | 选模 epoch | Macro-F1 | accuracy | metal 召回 | plastic_sub 召回 | plastic_fish 召回 | real_fish 召回 |
|---|---|---|---|---|---|---|---|---|
| fold0 | 840 | 85 | 0.9449 | 0.9440 | 1.0000 | 0.8824 | 0.9704 | 0.8940 |
| fold1 | 711 | 37 | 0.9170 | 0.9269 | 0.9825 | 0.7589 | 0.9492 | 0.9810 |
| fold2 | 895 | 66 | 0.9364 | 0.9430 | 0.9839 | 0.9068 | 0.9580 | 0.9052 |

- **三折平均 Macro-F1 = 0.9328**；**合并外层预测 Macro-F1 = 0.9350**
  （accuracy 0.9387）——两个口径分列，不得混用。
- 合并召回：metal 0.9875 / plastic_submarine 0.8611 / plastic_fish 0.9606 /
  real_fish 0.9220。塑料潜艇仍是最弱类（与 fold1 的 25 例混淆相关）。
- 每折 P0 checkpoint、scaler、history、逐样本预测、fold report 的完整
  SHA256 记录在核验报告 `folds[].files`。
- inner_train/inner_val/outer_eval 样本数与协议一致（1116/490/840、
  1265/470/711、1191/360/895）。

## 4. 残差训练预算（冻结；入口在下一批实现）

CPU、full batch、200 epoch、AdamW、lr=0.01、weight_decay=0.001、seed=2026、
无增强、无类别加权、无调参。选模：inner_val Macro-F1 → accuracy → 较早
epoch，与 P0 相同规则；**outer_eval 不进入训练函数或选模逻辑**，checkpoint
冻结后只评一次。标准化：直接复用每折已保存 scaler（同数据同规则，数值与
P0 所用完全一致）；禁止重新拟合。

## 5. 预先冻结的结果判定

分别报告 RCS−R0 与 RCS−RC；三折平均与合并外层分列；各折/各类/各组的
改对改错；塑料潜艇、塑料鱼、真实鱼召回；修正方向、幅度与固定类别偏置
迹象（逐类平均 delta）。

工程推进门槛（**全部满足才推进**；只是探索门槛，不是统计显著性或独立
泛化证明）：

1. RCS 三折平均 Macro-F1 同时高于 R0、RC；
2. 至少两折相对两种对照均不下降；
3. 最差折相对任一对照 Macro-F1 下降 ≤ 1 个百分点；
4. 合并外层预测相对两种对照均净改对；
5. 各类合并召回相对任一对照下降 ≤ 1 个百分点；
6. 至少两个采集组出现相对 R0 的净改对，并报告收益集中程度。

未达门槛时如实报告；不得自动调整门槛或参数重跑。

## 6. 偏振内容诊断（打乱）协议

5 个种子 2026–2030（预先固定）。只交换**有效 S 值**；接收方 G/C 与缺失
状态不变。donor 匹配在训练前冻结：按训练组估计的 valid_ratio、gray_mean
三分位区间分箱，各 split 内、有效样本箱内构建无自身对应的置换；单元素箱
记为未匹配。逐种子报告 donor 覆盖率与同采集组 donor 比例，保留全部种子、
donor 与逐样本结果。**打乱下降不得单独解释为材质因果贡献。**

## 7. 本批已做 / 未做

| 项 | 状态 |
|---|---|
| 协议冻结（本配置） | 完成 |
| P0 基线复算（7 项合成测试 + 真实运行逐项一致） | 完成 |
| P0 核验入口加固（严格有限性/身份/汇总/scaler 全列） | 完成（v2，38 项合成测试） |
| P0 核验入口补齐（折集合/完整历史/三方内层指标/内存 pooled） | 完成（v3，70 项合成测试） |
| 残差头与训练入口 `core/material_residual.py`、`scripts/train_material_residual.py` | 完成（31 项合成测试；真实 dry-run 通过） |
| RC/RCS 真实拟合与判定 | **NOT_RUN — 等待用户手动执行**（`scripts/manual_material_residual.ps1`） |
| RCS/RC 拟合与判定 | NOT_RUN（等用户手动执行） |
| 打乱诊断 | NOT_RUN（协议已冻结） |
| 真实数据训练（本批） | NOT_RUN |

## 8. 实现说明（2026-10-08 批次）

- `core/material_residual.py`：残差头 `ResidualHead`（14→4 无 bias，零初始化，确定性 seed）；
  `delta = valid_s · 0.5 · tanh(Wx)`（|delta|≤0.5，无效 S 恒 0）；`RC` 在输入层置零 S 槽
  （对缓存 S 值不敏感，测试钉住）；P0 logits 以 detached 常量进入训练（无梯度路径）；
  训练循环只接收 train/val 张量（签名上不存在 outer_eval 参数）；epoch 0（未训练头）
  恒为首个候选，平分经共享更早 epoch 规则保留；损失 = CE + 0.01·mean(delta²)。
- 标准化：每折从 inner_train 只读拟合 C/S 14 列 scaler（C 全部行、S 仅有效行），并断言与
  该折已保存 P0 scaler 的 C/S 段完全一致（同时重跑全列来源核验）；实际变换使用已保存
  scaler 的 C/S 段（数值与新鲜拟合逐位一致）。
- 入口 `scripts/train_material_residual.py`：--dry-run/--fit 显式互斥；准入（协议摘要→缓存
  绑定→P0 admission 身份→折记录集合→每折 P0 基线完整复算——logits/指标/scaler 全列/
  完整 200 epoch 历史）在任何头构造、scaler 拟合、训练产物写入之前；P0 参数/buffers 训练
  前后快照逐一比对；产物含 admission/provenance/scaler/权重/历史/选中 epoch/逐样本
  logits+预测+delta+组+缺失状态/混淆/逐类召回/改对改错；所有选中模型冻结后运行 5 种子
  打乱诊断（donor 分箱 edges 预先从该折 inner_train 的 valid_ratio/gray_mean 三分位计算，
  同 split 同箱无自身对应，单元素箱如实记未匹配，覆盖率与同组 donor 比例逐种子保存）。
- `scripts/manual_material_residual.ps1`：默认 dry-run；显式 -Mode fit 才训练；解释器核对；
  参数数组 splatting（无反引号续行）；preflight 非零即终止；每次新输出目录；通过
  PowerShell Parser.ParseFile 语法检查（合成测试）。

## 9. 2026-10-08 补修批次：打乱诊断修复与逐样本证据

上轮实现存在五处已确认缺陷（真实缓存只读复现：fold0、seed 2026、806 个
有效接收方中会选到 36 个无效 donor，其标准化 S 为 NaN）：

1. `run_shuffle_diagnostic` 构造输入时把接收方 C 置零；
2. `run_shuffle_for_fold` 虽计算 c_std/s_std 却把原始 S 传入诊断；
3. `shuffle_assignment` 从全部行选 donor（可能选到无效 S），且各接收方
   独立有放回随机选择，donor 会被重复使用（非置换）；
4. 逐样本 logits 计算后被 pop 丢弃；
5. 手动脚本结果提示使用了不存在的 `oof.*` 字段。

修复（v2）：

- `shuffle_assignment(valid_receiver_rows, bins, split_labels, seed)` 只在
  **有效**接收方集合内、按 (split, bin) 分组构造**双射**（seeded shuffle +
  循环旋转直到无固定点；单元素箱 unmatched）；无效样本从不作 donor；映射
  只依赖冻结分箱/角色/有效状态/稳定行序/seed。真实三折五种子只读检查全部
  通过（无效 donor=0、自身对应=0、重复 donor=0、每箱双射、不跨 split/bin；
  覆盖率 fold0 1.0、fold1 0.9985（一个单元素箱）、fold2 1.0）。
- `run_shuffle_diagnostic` 新签名接收**标准化**的 `c_standardized`/
  `s_standardized`：接收方 C 逐元素不变、P0 logits 不变、s_valid 不变；
  只交换有效行间的标准化 S；恒等核验（同一 evaluator 对未打乱输入二次
  前向必须**精确**复现正常 RCS logits）内建于函数；unmatched/invalid 行
  的打乱输出必须与正常输出**精确**相等，否则拒绝。
- `run_shuffle_for_fold` 传入 c_std/s_std（与正常 RCS 评估相同的 scaler
  段与装配）；逐样本 logits 不再丢弃；每个种子保存全部评估样本恰好一条
  记录（含 invalid/unmatched），字段含 sample_name/group/class_id/
  receiver_row+index 含义/donor 双身份/has_valid_s/status/normal 与
  shuffled 的 logits、prediction、delta；成对统计（vs normal RCS、vs P0、
  vs RC/R0——单变体运行时如实标 NOT_COMPARABLE）按 sample_name 对齐；
  逐类/逐组 fixed/broke 汇总；matched 子集指标与全体指标分列。
- 事件顺序：打乱诊断在**所有**请求折/变体完成拟合并冻结 checkpoint 之后
  统一运行（合成测试以调用次序断言，修复前 fold0 的诊断先于 fold1 的
  拟合——RED 已复现）。
- 手动脚本结果提示改为实际的 `variants.*` 字段；完成报告的调用方式更正
  为 `& .\scripts\manual_material_residual.ps1`（不得 `python xxx.ps1`）。

测试（38 项，含生产调用路径级）：C 保持、标准化 S 生效、恒等前向精确一致、
无效行不作 donor、无自身/无重复/双射、单元素箱 unmatched、不跨 split/bin、
seed 复现与签名无标签参数、无效行精确回退（delta 全零）、unmatched 行
精确保持、逐样本证据完整且成对统计可复算、合成 main fit 完整、事件顺序
（RED：修复前 max(fit)=7 > min(shuffle)=2）、既有准入/epoch0/P0 冻结回归保留。

## 10. 边界声明

本协议的全部比较仍限于原 train+val 开发池的组隔离三折；不构成新的盲测、
独立泛化或偏振收益证明。P0 编码器特征来自缓存（本轮无任何图像编码器
前向）。打乱诊断是破坏对应关系的对照，不是因果分解。

## 11. 2A 批（2026-10-08）：偏振测量区域稳定性诊断——未建立可靠区域改进依据

第 1 批真实拟合结论为 `polarization_benefit = NOT_ESTABLISHED_BY_THIS_ROUND`
（RCS−R0 fixed/broke 7/6，RCS−RC 7/7）。2A 批对 S 区域做只读测量稳定性
诊断（证据：
`…/20261008T144500Z_e7f8a9/region_diagnostics/`）：

- 对应关系核验：2446/2446 开发池样本裁剪 gray 与 rectified 左帧逐位
  一致、掩码像素数与 manifest 一致、左右帧全部存在。视差/一致性逐像素
  分解 NOT_AVAILABLE（未保存；重算需 SGBM，超范围）。
- 边界环（mask−interior(erode3x3)）确实更差：有效率 0.582 vs interior
  0.761，mean abs_q 0.126 vs 0.084（三折方向一致）。
- 但对 S 统计的实际影响小：band 仅占当前 S 区域像素 4.2–4.5%、重
  |q| 像素的 8.5–10%；退化像素近零（|q|=1 0.013%、gray=0≈0、饱和
  1.5%、暗区 3.2%，两区域几乎相同）。
- ±1px 采样扰动下 S 统计最坏漂移约为样本间离散度的 1.6%
  （signed_q_mean 0.00076 vs 样本间 std 0.0465）；interior 规则不能
  降低 signed 统计的扰动敏感度（胜率 0.29/0.27），仅 abs_q_mean 略好
  （0.69）。
- 回退从未触发（fallback=0）：规则不改变任何样本缺失状态。
- 第 1 批 outer 改变样本（15/16 例）不集中于低质量区域（描述性证据）。

结论：**未建立可靠区域改进依据**——按"证据不足不硬造特征"停止，本批
不准备 reliable-region 缓存、不建新配置、不生成训练命令。候选规则
`interior_erode3x3_fallback_original_v1`（含空区域回退与五要素规范）
连同诊断/缓存机器已实现并测试（`core/polar_region_quality.py`，
45 项合成测试），供审查后决定是否实测。若采纳，残差入口的"新拟合
scaler ≡ 冻结 P0 scaler 段"断言与特征定义版本检查需要显式扩展——
已在报告 §4.5 声明，不允许静默换缓存。

## 12. 2A 补修批（2026-10-08）：三项验收缺口的纠正记录

第 11 节的 2A 诊断存在三项已确证缺口，本轮（只读）修复并留证
（证据：`analysis/runs/material_polar_residual_v1/regionqa_fix_20261008T152406Z_5aa77e04/`）：

1. **保护快照无效**：历史 `snapshot_protected.py` 以 `parents[3]` 推导
   项目根，实际落在 `analysis/runs`，26 项全部 `{"missing": true}`，
   "相等"不构成证据。历史快照现标记 **INVALID_PROTECTION_EVIDENCE**、
   `historical_before_after_integrity = UNVERIFIED`，历史文件未回写、
   未补造历史状态。新机制（真实 BEFORE/AFTER、文件集合+逐文件摘要+长
   度比较、缺失即拒绝）本轮实测 21 目标 / 3906 文件全部一致，且只证明
   本轮执行窗口。
2. **V5 完整准入未执行**：历史诊断未调用
   `admit_mask_dataset`/`verify_mask_sources`，"本链内再次复核多边形来
   源"的说法不成立。新入口在任何逐样本统计前执行项目既有完整门（期望
   指纹来自冻结配置/协议/特征定义三方，非现算），实测 2808 文件核验、
   指纹与 V4 父指纹一致；准入反例（篡改 NPZ、删标注、指纹冲突）均在统
   计开始前拒绝（统计函数调用即失败探针证明，未 mock 准入）。
3. **扰动解释越界**：四方向平移实验重命名为
   `perturbation_type = saved_q_roi_sampling_shift`——它只测已保存 q 图
   相对固定掩码的 ROI 采样敏感度，**不是**左右错位重算实验，不能排除
   左右配准/辐射不一致。历史"1.6%"是 mean/std，不是误差上界：同池
   signed_q_mean 漂移 mean=0.00076201、p95=0.00210021、**max=0.02782177**
   （约为样本间 std 0.04652325 的 60%）。mean/p50/p95/max 分列报告，
   mean/std 与 p95/std 标注为分布汇总而非误差界。

修订后结论（限定表述）："在已保存 q 图的 ±1px ROI 采样扰动下，腐蚀规
则没有一致改善 signed 统计量；当前证据不足以采纳该规则，暂不构建新缓
存或训练。本实验不能排除左右配准及辐射不一致，也不能证明其他区域表示
没有改进空间。" 第 11 节"测量区域不稳定不成立"等表述按此口径作废重述；
abs_q_mean 典型漂移略好、最坏情形更差，如实保留。聚合口径澄清：一切比
率为逐样本比率的均值；`missing_s_samples`（开发池 105）与
`rule_fallback_used=0` 分列。新增可跟踪入口
`scripts/diagnose_polar_region_quality.py`（`main(argv=None)`，拒绝已有
输出目录与输入内部目录）与 25 项入口级测试；聚焦回归 263 项通过；未训
练、未建缓存、未运行 SGBM。

## 13. 2A 测试收尾批（2026-10-09）：三项补齐的纠正记录

对第 12 节补修批的测试证据与文案收尾（证据：
`analysis/runs/material_polar_residual_v1/regionqa_test_finish_20261009T010101Z_ddf208c5/`）：

1. **拒绝型探针的测试假阳性**：上一轮三个准入反例的探针在抛出
   AssertionError 前不记录调用，而入口 main() 会把异常转换为返回码 1，
   因此"探针调用数为零 + 返回码 1"不能排除"统计已被调用、探针异常被吞
   进拒绝"的假阳性。本轮：探针改为**先记录调用再抛出**（新增校准测试
   在修复前暴露该缺陷，RED 已复现）；新增受控故障注入测试——在合法合
   成场景中让探针被入口实际执行，证明 main() 的异常转换不会抹去调用记
   录（调用记录非空，不得解释为"统计未开始"；这是探针校准，不是准入通
   过证据）；三个反例加强为同时检查：返回非零、探针调用数为零、真实
   stderr 的稳定错误片段属于**目标准入错误**（含 "AssertionError 不在
   stderr"）、无任何正常诊断产物。反例仍调用真实准入门，未 mock。
   加强过程中还暴露并修正了一个测试夹具缺陷：合成配置原写在 scratch
   根，入口的"输出不得位于输入目录内"检查（对文件锚点取父目录）会以目
   录包含为由拒绝，旧断言无法区分拒绝原因；现合成配置移入 scratch 子目
   录，stderr 片段钉住真实准入原因。入口生产代码未改。
2. **嵌套链接越界**：`build_protection_snapshot` 此前只检查顶层目标的
   真实路径。现对目录内**每一项**后代（含 rglob 不深入的 junction/符号链接节点）
   在读取/哈希之前做真实路径包含检查，越界抛出
   既有 ProtectionError 并指明目标与后代路径。验证口径：**目录链接越界
   用真实 junction（mklink /J，无需提权）验证**；文件符号链接因 Windows
   创建需要特权，本任务不申请提权，采用严格限定路径解析结果的
   monkeypatch 模拟，如实标注"模拟解析越界"；两种情形均验证拒绝发生在
   `_snapshot_file` 读取/哈希之前。普通无链接目录的快照结构与结果不变。
3. **证据层级与计数文案纠正**（历史文件未回写）：
   - 上一轮的 INVALID_PROTECTION_EVIDENCE /
     historical_before_after_integrity = UNVERIFIED 标记实际位于
     **protection_report.json**；diagnostics_summary.json 没有该字段。
   - "21 个目标 / 3906 条文件记录"中 3906 是**文件记录数**，独立文件数
     为 **3901**（缓存三文件在目标间重复出现等）。
   - V5 保护的 2808 个文件为 **2808 个 NPZ + 3 个数据集元文件**
     （manifest/summary/audit）。
   - 本批未重新执行真实诊断，无新增模型质量证据；上一轮 Codex 已独立核
     实的生产准入顺序、真实保护结果与统计数值不因本批测试证据缺陷而被
     否定。

结论保持不变：saved_q_roi_sampling_shift 扰动不能验证左右配准准确性；
腐蚀规则没有一致改善 signed 统计量；当前不采纳规则、不构建缓存、不训
练。验证：入口测试 31 项、聚焦回归 269 项通过；compileall 与
git diff --check 通过；本轮真实诊断、训练、特征构建、SGBM 均未执行。

## 14. 局部偏振统计批（2026-10-09）：方案 A 预注册与实现

在新实验 `material_polar_local_interior_band_v1`（方案 A：interior/band
分别统计）下验证局部偏振统计；协议、选择依据与门槛已预注册于
`docs/experiments/material_polar_local_interior_band_v1.md`。要点：

- **不改变区域可靠性规则**：原测量区域被划分为 interior/band（erode3x3
  冻结定义复用），不排除任何像素，`s_valid` 与全局定义构造性相同；
  被否决的 interior 回退规则仍未采纳；
- 残差输入 14 → 30 维（C 6→14 含每块 coverage/empty/gray 统计；
  S 8→16 = 每块既有 8 项统计）；G、每折 P0、折成员、预算、损失、修正
  幅度全部不变；标准化仅 inner_train，且不再断言与冻结 P0 scaler 段相
  等（预先声明的显式设计决策）；
- 真实本地缓存已构建（构建前 V5 完整准入 2808 文件、指纹一致；G 逐字
  节相同；s_valid 逐行核对一致；缺失 105 个样本与全局定义一致，其中
  1 个样本 band 空、0 个 interior 空）；真实 dry-run 通过（三折 P0 基
  线 0.9449/0.9170/0.9364 与冻结表一致）；
- 内容打乱诊断保留冻结种子与分箱双射，新增**区域守卫**：仅交换双方该
  均有效的块，空块保持显式零并逐样本记录；
- 旧全局版本（manual_20261008T221522）数字将作为同协议参照写入新
  summary（不设门槛、不混用）。
- 新入口：`scripts/train_material_local_residual.py`（dry-run/fit 显式
  互斥、输出目录拒绝覆盖）、`scripts/prepare_material_local_features.py`
  （构建前完整 V5 准入）、`scripts/manual_material_local_residual.ps1`
  （默认 dry-run，`& .\scripts\…` 调用）。真实 fit 由用户手动执行；
  验证：local+residual 67 项测试通过；compileall / git diff --check 通
  过；保护快照 24 目标 / 3919 文件前后一致。
