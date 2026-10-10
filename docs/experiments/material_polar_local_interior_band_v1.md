# material_polar_local_interior_band_v1 — 局部偏振统计验证（方案 A，预注册）

状态：**协议与判定已冻结；特征缓存与训练入口已实现；真实 fit 由用户手动
执行**。本协议在任何真实拟合之前写入，判定门槛不得事后调整。

## 0. 方案选择依据（预先写入，唯一方案，不做特征组合搜索）

本批从 2A 已有诊断中**只选用方案 A：实例内部与边界分别统计**：

1. **定义已冻结**：`interior = erode3x3_instance_mask_v1`、
   `band = instance_mask_minus_interior_v1` 是 2B/2A 批在任何残差结果出
   现之前冻结的定义（`scripts/prepare_material_probe.py` 常量 + 测试钉
   住），本批原样复用，不新增任何阈值；
2. **已有测量证据**：band 有效率 0.582 vs interior 0.761、mean|q|
   0.126 vs 0.084、重 |q|（≥0.5）像素在 band 过代表（9.4% vs 像素占比
   4.3%），三折方向一致（`regionqa_fix_20261008T152406Z_5aa77e04`）；
3. **方案 B 落选**：未饱和亮区 vs 中等亮度需要新设亮度阈值——不存在任
   何冻结定义，"预先写入"无从谈起；且诊断显示饱和（gray≥250）像素仅
   ~1.5%、退化像素近零，B 的分区间几乎退化。依据不足，不实施；
4. 未同时实施 A、B，也不做任何特征组合搜索；若审查要求改用 B，本批产
   物为独立版本目录，切换不影响任何前序产物。

**与被否决规则的区别**：2A 诊断未采纳的是可靠性规则
`interior_erode3x3_fallback_original_v1`（只用量化后的 interior 计算统
计并回退——那会改变测量区域）。本方案是**划分**而非**过滤**：原测量区
域 `instance_mask & polar_valid` 的每个像素恰好落入 interior 或 band
之一，不排除任何像素，`s_valid` 与全局定义构造性相同（测试钉住）。
区域可靠性规则、骨干、损失、训练预算均未改变。

## 1. 输入来源绑定

| 来源 | 路径 | 身份 |
|---|---|---|
| 协议 | `…/20260930T021500Z_a3f7c1/protocol/protocol.json` | `0de5cf8a…`（经冻结准入链自洽核验） |
| 全局缓存（G 来源 + P0 链输入） | `…/20260930T045500Z_c5d9e8/upgrade/corrected_features` | 绑定逐字节核验 |
| P0 拟合 | `…/manual_20261008_085510_822/fit` | 完整历史/全列 scaler 复核 |
| V5 数据 | `datasets/underwater_cls_fusion_v5_instance_mask` | `bb9e3661…`（构建前完整准入） |
| 本地特征缓存 | `…/polar_local_20261009T013057Z_ec28494d/local_features` | `existing_data_probe_v1_local_interior_band_v1` |
| 旧全局版本参照 | `…/manual_20261008T221522/fit/summary.json` | SHA256 记录于 summary.old_global_reference |

## 2. 特征定义（唯一变化：残差输入 14 → 30 维）

- **S_local（16 维）**：interior 块 8 项 + band 块 8 项——每块沿用既有
  8 项统计定义（`POLAR_STAT_KEYS`：mean/std/p10/p25/p50/p75/p90/abs_mean），
  分母为该块像素（part ∧ polar_valid）；空块 = 全 NaN（显式缺失）；
- **C_local（14 维）**：原 6 项控制不变 + 每块
  `coverage`（|part∧valid|/|part|）、`empty_flag`、`gray_mean`、
  `gray_std`（灰度统计限定 part ∧ valid，[0,1] 标度）；空块的灰度记 0
  并由 empty_flag 显式区分（控制信息约定，非伪造测量）；
- **G（256 维）**：从全局缓存逐字节复制，不变；**P0**：头与每折
  scaler 完全冻结，其 logits 由全局缓存（旧 C/S + G）经冻结 scaler 只
  读前向得到，训练前后参数/buffers 快照逐一比对；
- **缺失状态**：样本级 `s_valid` 与全局定义逐行相同（划分覆盖原区域，
  构造性保证 + 逐行断言）；块级缺失 = 全 NaN 块 + C empty_flag，不得当
  作真实零测量；残差输入中缺失/空块的列精确为零（标准化规则），delta
  门控 `valid_s` 不变；
- **标准化**：每折仅用 inner_train（C 全部行、S 按单元格有效性）；不
  再断言与冻结 P0 scaler 段相等（特征已不同——这是 2A §4.5 预先声明
  的显式设计决策），P0 链不受影响。

## 3. 变体与公平比较

- **R0** = 冻结 P0（对照）；**RC_local** = P0 + C_local 修正（偏振槽输
  入置零；区域质量信息仍可用）；**RCS_local** = P0 + C_local、S_local
  修正。三者共享同一 30 维输入槽、同一初始化与预算；
- 报告口径：P0 / RC_local / RCS_local / 旧全局版本四行同页；归因分列：
  **区域/灰度信息增益 = RC_local − R0**，**额外偏振增益 =
  RCS_local − RC_local**；旧全局版本数字仅作同协议参照（样本池与折相
  同、特征不同），不设门槛、不据此择优；
- 逐样本、逐类、逐组改对/改错、三折平均与合并外层分列——沿用既有规
  则；内容打乱诊断按冻结种子 2026–2030 与既有分箱双射规则运行，**区域
  守卫**：仅交换双方该块均有效的标准化块，空块保持显式零并逐样本记录
  `interior_block_swapped`/`band_block_swapped`。

## 4. 预先冻结的工程推进门槛（对本地变体）

1. RCS_local 三折平均 Macro-F1 同时高于 R0 与 RC_local；
2. 至少两折相对两种对照均不下降；
3. 最差折相对任一对照下降 ≤ 1 个百分点；
4. 合并外层预测相对两种对照均净改对；
5. 各类合并召回相对任一对照下降 ≤ 1 个百分点；
6. 至少两个采集组出现相对 R0 的净改对，并报告收益集中程度。

归因说明（区域/灰度 vs 偏振）分列报告。未达门槛时如实报告；不得自动
调整门槛或参数重跑。旧全局参照数字不参与门槛判定。

## 5. 边界声明

全部比较仍限于原 train+val 开发池的组隔离三折；不构成盲测、独立泛化或
偏振收益证明。真实 fit 由用户手动执行（`scripts/
manual_material_local_residual.ps1`，默认 dry-run，显式 fit 才训练，输
出目录拒绝覆盖）；Agent 不训练真实数据、不安装依赖、不改数据/正式配
置/旧权重/历史证据。

## 6. 修复批（2026-10-09）：训练阻断、统计契约与打乱协议 v2

真实入口首跑（合成验收）暴露并修复六项问题；以下为根因与最小修复记录。
打乱诊断协议由 v1 修正为 **v2**（见 §3 更正），配置 `shuffle_diagnostic`
说明同步更新——真实 fit 前生效，其他输入、预算、门槛与种子未变。

1. **未定义 `x_eval`（训练阻断）**：`fit_one_fold_variant` 只计算
   `std_eval` 未装配外层输入，外层评估处 `NameError`。修复：与训练/验
   证一致用 `_local_design` 装配（RC 置零规则不变，scaler 不重拟合），
   外层指标在 `best_state` 锁定后计算。RED：合成 `main --fit` 实跑复现
   NameError。
2. **折指标字段与汇总不一致（训练阻断）**：fold_report 保存
   `outer_eval_*` 而 run_fit 读取 `macro_f1` → KeyError。修复：统一汇总
   契约——每个变体的 per-fold 结果携带 `metrics`（由 labels/predictions
   计算），summary 三折均值/pooled 全部来自 `metrics`；fold_variant_
   report.json 保留 `outer_eval_*` 字段含义。R0 亦保存逐样本 jsonl（预
   测来自已准入链、logits 为同头同 scaler 只读复算）。禁止默认 0/忽略
   缺键/宽泛捕获。
3. **逐类改对改错混累计**：原实现把 RCS−R0 与 RCS−RC 计入同一键。修复：
   `count_fixed_broke`（纯函数）按比较分别记录 per_class/per_group，入
   口内断言逐类求和 == paired 总数（不符即 MISMATCH）。反例测试：同样
   本 R0、RC 均错、RCS 正确 → 两种比较各计 fixed=1，不得合并为 2。
4. **打乱协议 v2**：v1"行级双射+逐块守卫"不能保证各区域内容双射（A、B
   band 有效而 C band 空时，A/B 可能共用同一 donor band 而 A 的 band 弃
   用）。v2：分组改为 (split, 原三分位 bin, (interior_valid, band_valid)
   有效模式) 复合整数 bin，复用共享 `shuffle_assignment`（其默认行为未
   改）；**完整标准化 S16 来自同一 donor**（模式相同 ⇒ 两块精确交换）；
   单元素组 unmatched、不跨组补 donor；无效样本不作 donor；C/P0/缺失状
   态不变；unmatched/invalid logits 精确不变。逐种子报告总覆盖率、各区域
   有效分母/交换覆盖率、unmatched 原因与同组 donor 比例；覆盖率降低如实
   报告，未打乱样本不作为偏振不敏感证据。
5. **局部缓存数组准入补齐**：`verify_local_cache` 改为逐块"全有限 XOR
   全 NaN"状态机——拒绝 ±Inf 与部分 NaN 块；s_valid=True 行至少一块全
   有限（两块全 NaN 拒绝）；s_valid=False 行两块全 NaN；empty_flag/
   coverage/s_region_nonempty_flag 一致性检查扩展到全部行。合法情形（单
   块空、双块有效、全样本缺失）不误伤。非法用例只在合成 scratch 内构
   造，并按生产方式重签合成缓存绑定，证明拒绝来自数组语义而非摘要。
6. **诊断失败不得留下 COMPLETED**：summary 现以 `FIT_COMPLETE` 落盘，打
   乱诊断全部完成后才改写 `COMPLETED` 并附带 shuffle 结果；诊断中途失败
   → `training_status=FAILED`、`failed_phase=shuffle_diagnostic`、
   `failure` 记录异常，已保存拟合产物保留，入口返回非零。故障注入测试
   钉住该行为。

验收：合成世界覆盖双块有效/interior-only/band-only/全缺失/单元素组；
真实 `main --fit` 路径完成 6 次微型拟合与 15 份打乱结果，P0 参数与
scaler 前后不变、保存重载一致、逐样本证据独立复算通过。**真实数据 fit
仍 NOT_RUN**（等用户手动执行）；真实 dry-run 在修复后重跑通过。

## 7. P0 输入身份修复（2026-10-09）：冻结权重 ≠ 冻结前向输入

**已确认身份错误**：本入口的 `_p0_design_for_rows` 把标准化后的完整
G+C+S 直接喂给冻结 P0 头，没有执行 P0 的禁用槽置零——真正的冻结 P0
只读 G。权重未变，但前向输入变了。受影响范围：train/inner_val/
outer_eval 的 P0 底座、R0、shuffle、无效偏振回退。

**历史拟合不可验收**：
`analysis/runs/material_polar_local_interior_band_v1/manual_20261009T130447`
标记为 **INVALID_FOR_FROZEN_P0_COMPARISON**——不能用于偏振收益验收，
也不能把旧模型直接换底座修补（错误基线影响了训练与选模本身）。只读
取证（`local_p0_identity_fix_20261009T051737Z_c5d06351/
invalid_fit_forensics.json`）：R0 的 prediction 与 logits.argmax 冲突
4/3/1；fold1/fold2 的 RC/RCS 选 epoch0（delta 全零）但预测相对真 P0
漂移 3/1；105 个缺失 S 样本的 logits **无一**精确回退真 P0（最大 logit
误差至 0.664）。这些不是修复后模型质量指标。

**修复**：
1. `_p0_design_for_rows`：标准化沿用冻结 scaler，随后
   `assemble_design_matrix(..., "P0")` 将 C/S 槽精确置零（与
   verify_p0_baseline / train_material_probe 同一装配）；
2. dry-run 与 fit 共用的运行时 P0 身份校验（在任何残差头构建与训练产
   物写入之前）：独立 oracle = `<P0 fit>/models/<fold>_P0/
   outer_eval_predictions.jsonl`，读取前后核对字节 SHA 与准入
   `fold_results[fold]["files"][...]` 一致；逐项核对样本名集合/顺序/重
   复、group、class_id、支持数、四维 logits、prediction==argmax；运行
   时 logits 与 oracle 以 `LOGIT_ATOL=1e-4`（rtol=0）绝对容差比较、预测
   必须完全一致；train/inner_val/outer_eval 三处设计的禁用槽必须精确
   为零；不一致返回 MISMATCH 并报告 fold/样本/最大误差；
3. 同一条已核验 P0 链（head+scaler 对象）服务 R0、RC/RCS 的
   train/val/eval 常量、shuffle 与无效回退；R0 的 logits/prediction/
   指标全部来自同一已核验数组；
4. CLI `--fit-dir` 与配置 P0 来源按解析后路径核对，不一致即拒绝；
5. admission 与 fit summary 保存 `p0_identity`（装配规则、oracle/
   checkpoint/scaler 身份、每折样本数/最大误差/预测不一致数/禁用槽检
   查/缺失 S 行数与误差）；缓存 feature_definition 版本未改。

**修复后真实 dry-run 实测**：三折 840/711/895（共 2446）外层样本的完
整 P0 logits 与冻结 oracle **逐位一致**（max_logit_error = 0.0，容差
1e-4）；预测不一致 0；三处禁用槽检查全 0.0；105 个缺失 S 样本单列核
验（34/24/47）误差 0.0；冻结 P0 Macro-F1 精确复现
0.9449044133083752 / 0.9170174906889274 / 0.9364285704602361（身份核
对，非调参目标）。

**影响**：六次真实局部拟合必须由用户重新运行；本轮未执行任何真实
fit。选模规则、预算、门槛、分箱、种子、局部特征定义均未改变。
