# material_residual_runtime_v1 — the locked residual candidate under real
# Model A predictions (integration + two-layer evaluation)

The candidate is the **completed** conditional-residual fit
`material_conditional_residual_v1` (`manual_20261009T185911`, all identity
files pinned by sha256 in
`configs/experiments/material_residual_runtime_v1.yaml`): frozen MiT-B0
last-stage mask-pooled gray feature **G**, the frozen per-fold **P0 linear
head** (input 270, C/S slots exactly zero) and the frozen **correction
branch** `Linear(35,8) -> SiLU -> Linear(8,4,bias=False)` with the last
layer zero-initialized, `delta = valid_s * 0.5 * tanh(head(x))`. The
candidate is NOT a CMX checkpoint and never enters the historical CMX
loader; `models/material_residual_classifier.py` is its own adapter
(TDD: 36 tests written first, RED confirmed at 35 failed + 12 errors +
1 guard-rail pass before implementation).

This batch is **integration and evaluation only**: no training, no
deployment, no re-selection of feature/epoch/seed, no change to
`configs/default.yaml`, the legacy input modes or any historical report.

## 1. What this batch adds

- `models/material_residual_classifier.py`: the runtime adapter.
  - Verifies every pinned file digest (fit summary/admission, protocol,
    both feature caches, per-fold P0 checkpoints/scalers/oracles, the
    seed-2026 RC/RCS residual states/scalers/saved predictions, the R0
    records) and every payload's own identity fields (variant/fold/seed/
    input_dim/hidden) BEFORE anything runs.
  - Reuses the offline math verbatim: `_fusion_tensors` + resized
    instance mask + `mask_level_weights` pooling for G;
    `build_local_control_features` / `build_local_polar_stats` for
    C14/S16; `assemble_design_matrix(..., "P0")` for the P0 design;
    `p0_context` / `assemble_conditional_inputs` / `conditional_logits`
    for the correction. The heads always run on CPU float32.
  - Fold routing: a frame is evaluated only by the fold its capture
    group belongs to (`build_frame_fold_map` refuses a group held by two
    folds or by none) — never a best fold, no ensemble, no refit.
  - Fallback semantics are the candidate's own: a missing or non-finite
    polarization measurement gates `valid_s` to zero, so RC/RCS fall
    back to the candidate's OWN frozen P0 exactly (recorded as
    `used_fallback` with reasons `no_valid_s_region` / `non_finite_polar`);
    an unusable gray crop is an explicit per-instance failure and
    reaches neither encoder nor heads. No GT class is ever read.
- `configs/experiments/material_residual_runtime_v1.yaml`: the identity
  pins (all sha256), Model A at the fixed operating point, the frozen
  stage tolerances, the frozen benchmark budget and the data-identity
  disclosure (`model_a.data_independence: unknown`).
- `scripts/eval_material_residual_pipeline.py`: the two-layer evaluation
  entry, reusing the established framework functions (matching, metrics,
  bootstrap, fallback summary, quality strata) from
  `scripts/eval_material_pipeline.py` — no second general framework.
- `tests/material_fusion/test_material_residual_runtime.py`: 36 tests.

## 2. Layer A — offline reproduction, stage by stage

Layer A replays the candidate's own evaluation basis — every dev-pool
sample is outer_eval of exactly one fold (2446 rows) — through the
adapter and compares, per sample: the runtime G feature against the
local cache (atol=rtol=1e-4), the recomputed C14/S16 (exact, atol=0),
`s_valid`, the P0 logits and the final corrected logits against the
saved per-fold jsonl (atol=1e-4, rtol=0), the predictions (exact), and
the missing-S fallback (final == P0 bitwise). Seeds 2027/2028 saved
predictions are integrity-checked against the same P0 chain (their heads
are not loaded at runtime).

**Device finding (recorded because it decides the Layer A device):** the
G comparison is bitwise EXACT on GPU (max abs error 0.0 over the first
32 samples) and ~9.3e-3 on CPU. Diagnosis (`logs/g_diagnosis*.log`):
today's original `extract_gray_features` path reproduces the adapter
bit-for-bit on both devices, so the adapter math is exactly the offline
math; the cache itself was extracted on GPU (bitwise-reproducible only
there). Layer A therefore runs the encoder on GPU (device 0, as the
config pins) and the heads on CPU.

A stage difference beyond its pre-fixed tolerance ABORTS the formal
quality statistics before Layer B (exit 2, no Layer B metrics written).

## 3. Layer B — the production pipeline on real predictions

The production engine (Model A `yolo26_formal_001` at conf 0.25 / IoU
0.45 / imgsz 640, band SGBM 3way/768/0.25/7, band margin 20, crop
padding 10, the accepted polarization formula) runs on the 180 val
frames with the RCS adapter bound as Model B; the R0/RC variants are
evaluated on the same per-instance inputs. Class-agnostic one-to-one
mask-IoU >= 0.50 matching against the four-class GT, then:

- detection support / missed GT / unmatched predictions at the fixed
  working point (never called mAP);
- conditional classification on matched instances (accuracy, Macro-F1,
  per-class precision/recall, confusion matrix);
- end-to-end four-class Macro-F1 including misses and false positives;
- paired P0-vs-candidate changes (RCS−R0, RCS−RC) on matched instances
  and end-to-end, plus paired cluster bootstrap over the 4 val capture
  groups (2000 resamples, seed 2026) under its own name;
- per-group metrics, polar-quality strata (conditional on a match),
  fallback reasons, and every per-instance record
  (`instances.jsonl`).

**Independence disclosure:** Model A's training data shares its capture
sources with the classifier data. Layer B is therefore NOT an
independent blind test, and nothing in this batch claims one.

## 4. Bounded benchmark

Frames (12, evenly spread over the 180 sorted val stems), warmup (2),
measured repeats (5), scenario buckets (0/1/2/5 targets) and the
15-minute wall budget are frozen in the config before any run. Timing
covers preloaded already-rectified frames through the production engine
(CUDA-synchronized totals + per-stage diagnostics); it is NOT a
camera-to-GUI claim, incomplete timing is PARTIAL, and nothing is
extrapolated to 20–30 FPS.

## 5. Honest state

- Layer A establishes that the runtime path reproduces the offline
  candidate; it does not by itself establish any model-quality claim.
- Layer B runs on the SAME val split the candidate was selected on;
  gains there are not independent-generalization evidence.
- See the batch report for the recorded numbers.

## 6. 完成批纠正记录（2026-10-09，residrt_c 批）

### 6.1 历史记录纠正

- 历史 RED 数字纠正：**23 failed、12 errors、1 passed**（此前"35 失败"
  是非通过总数，不是失败数）。
- 历史 formal_exit.txt 的 `EXIT=1` 属于**初次失败尝试**（输出目录被预
  创建而入口拒绝）；最终成功运行（evaluate_20261009T122430Z）**没有独
  立退出码记录**——如实声明，不补造。
- "RC/RCS 条件指标相同"的准确表述：条件**混淆矩阵恰好相同**，但成对
  差异存在（RCS−RC = 1 改对 / 1 改错），端到端 FP 的类别分配也不同。
- 性能口径：p50 ≈ **191.424 ms**、p95 ≈ **333.374 ms**（60 次测量）；
  **5.224 FPS 是 1/p50**（每帧延迟统计量，不是吞吐）；按 60 次测量总
  耗时折算 ≈ **4.560 FPS**。两者都不是相机→GUI 实时吞吐；子阶段未分
  别显式 CUDA 同步，不当作可相加 kernel 预算；232.28 MiB 是评价进程峰
  值，不能限定为计时区间峰值。**新候选（线性局部/其他种子）性能
  NOT_MEASURED**，不得沿用旧候选时延。
- 旧停止建议（"停止本轮偏振模型扩张"）仅适用于**实际评估过的
  conditional seed2026 子集**；完整候选矩阵评估后按冻结标准逐候选给
  出结论。bootstrap 区间是描述性统计，不是偏振材质因果贡献证明。

### 6.2 本批工程修复

1. **Layer A 比较器**：形状/非空/有限性契约前置；有限阶段双方拒绝
   NaN/Inf（[Inf] vs [Inf] 不再 PASSED，有限 vs Inf 不再被相对容差吞
   掉）；原始 S16 仅接受冻结定义的完整 8 维缺失块（部分 NaN、单边缺
   失、Inf 均失败）；失败报告 JSON 安全（无法计算误差时为 null + 原
   因，绝不输出 NaN/Inf，不用 nan_to_num）。
2. **总判定**：期望样本集合来自冻结协议/清单、期望候选集合来自冻结注
   册表、必需阶段集合显式声明——缺失/重复/未知候选/缺阶段不能判 OK；
   任一失败非零退出且不进入 Layer B（layer_a 单独模式同样非零）。
3. **编码器缓存**：键 =（解析权重路径、SHA256、规范化设备、imgsz）；
   不同设备/权重身份不误命中；命中后校验设备仍一致，共享对象被移动时
   拒绝而非原地迁移。
4. **9 路固定候选矩阵**（material_residual_runtime_v1_complete.yaml，
   任何新结果出现前冻结）：R0 + linear_local RC/RCS s2026 + conditional
   RC/RCS s2026/2027/2028，每路钉死 residual_state/scaler/saved
   predictions/fold report 的 SHA256；候选文件摘要为本轮评估前冻结身
   份（拟合产物自身无摘要清单；训练期身份锚为两份 fit summary 与冻结
   线性参照 22ff7a81…）。
5. **Layer A**：全部 2446 样本、9 路全部实际前向、对齐各自保存的
   oracle（2027/2028 不再以 P0 列检查代替残差头回放）；原始
   G/C14/S16 共享、标准化输入按候选各自的冻结 scaler 独立计算。
6. **Layer B**：每帧一次 Model A/SGBM/偏振/实例输入；每公共预测 ROI
   的 G 特征提取一次供 9 路复用（是逐 ROI 前向，不是整帧单次声明）；
   全部 9 路覆盖同一预测实例集合，证据覆盖匹配与未匹配预测
   （frames_complete.jsonl + predictions_complete.jsonl）；新增对旧
   180 帧检测记录的只读核对（差异只报告，绝不调阈值"恢复数字"）。
7. **独立复算**：scripts/recompute_material_residual_evidence.py 仅从
   原始 GT/匹配/逐预测 JSONL 重建检测支持、每候选条件/端到端/混淆/逐
   类/逐组、按 (frame, gt) 对齐的成对统计、全部预测上的 fallback、
   4 组/2000 次/seed2026 配对 bootstrap——不导入任何生产汇总函数。

## 7. 完成批收尾记录（2026-10-10，residrt_finish 批）

### 7.1 四项工程缺口闭合

1. **三个未收集测试恢复**：`test_layer_b_nine_candidates_complete_evidence`、
   `test_layer_b_recompute_from_raw_records_agrees`、
   `test_params_and_buffers_unchanged_by_layer_b` 原嵌套在
   `layer_b_orchestration` fixture 的 return 之后不可收集；已移入可收集的
   `TestLayerBCollectedAcceptance`（函数名不变）。测试世界**完全合成**：
   协议/清单/审计/缓存/身份文件全部位于本轮 scratch；真实准入门校验合成
   数据；候选为模块内生成的微型合成 checkpoint（含条件三种子与线性局部
   s2026 的 9 路）；编码器与引擎为确定性合成实现；不 mock main/verdict/
   指标汇总/独立复算器。覆盖：9 路完整记录、真实存在的未匹配预测（不为
   空）、漏检 GT、缺失 S 的精确回退（8 个修正候选 × 2 实例，final 与 R0
   逐位一致）、全部候选与共享编码器的参数/buffers 前后不变。收集证明：
   `--collect-only`（-vv 树 + -q nodeid 两种形态）三个 nodeid 各出现恰好
   一次；JUnit XML 确认三者各执行一次且干净通过。
2. **Layer A 判定全覆盖**：必需阶段之外**实际出现的每份阶段报告**都要
   满足通过契约（fallback 失败不得忽略；非法/自相矛盾报告——passed 非
   布尔、status 出现且越界或与 passed 矛盾——判失败）；
   `s_valid_runtime/offline` 严格布尔且必须一致；缺失 S 行必须携带
   `fallback_exact_p0 is True` 的实际证据（字段缺失不默认通过；R0 的语
   义区别在代码与文档中明确：R0 即共享冻结 P0，检查平凡满足）。反例：
   五必需阶段全过但 fallback 失败、s_valid 两侧不一致/非布尔、缺失 S 无
   证据、非法报告类型——全部判 FAILED 并中止 Layer B；合法有效/缺失样
   本正常通过。
3. **独立复算完整比较且失败非零**：新 CLI（--evidence-dir/
   --source-manifest/--config/--output-dir + 两个身份锚点）；config 提供
   权威 9 候选与类别顺序并核对 protocol_snapshot；来源清单先绑定全部被
   消费证据（摘要+字节）再读取；输出全新目录、位于输入内部即拒绝；raw
   结构准入（候选/帧/实例完整集合、重复、非法编号、匹配一对一、标签范
   围、有限四维 logits、prediction==argmax、fallback 有因、R0 join 缺失
   /重复拒绝）；复算比较扩展到总体/逐组 counts、检测、条件 accuracy/
   Macro-F1/混淆/逐类/支持数、端到端 TP/FP/FN/逐类、paired 四类转移、
   回退数量/比例/原因/分母、**全部 32 个 bootstrap**（16 端到端 + 16 实
   例正确性，冻结采集组/次数/种子/口径独立重算，不调用生产 bootstrap）；
   键集合/类型/支持数/数值检查，None/NaN/Inf 不得因双侧同坏而通过；确
   定性指标 atol=1e-12/rtol=0；raw 非法 exit 2、raw/summary 不一致
   exit 3、仅完整准入+完整比较+零差异才 PASSED/exit 0。历史
   predictions_complete 的 p0_logits=None 通过
   `R0_join_by_frame_and_pred_instance` 派生复核（不修改历史记录、不声
   称历史 payload 保存过 P0 logits）。
4. **总预算**：`BudgetExceeded` 独立异常 + 可注入单调钟；预算解析优先
   CLI 显式合法覆盖，否则 `evaluation_budget.whole_run_minutes`（30），
   **绝不回落到 benchmark 的 15**；None/非有限/非正/非法类型在键存在时
   拒绝；同一总 watch 覆盖 setup、A 每批、B 每帧、阶段边界与最终发布
   前，不逐阶段重置；超时 PARTIAL/非零并保留已完成证据与失败阶段，普通
   RuntimeError 判 FAILED 并保留类型/原因（benchmark 内部仅捕获自身预算
   耗尽）；benchmark 自身 15 分钟预算独立保留。假钟证明：默认 30 非
   15、A 中/B 每帧边界/发布前超时、同一预算未重置、普通异常不误分类、
   合法路径 COMPLETED。历史运行曾按 15 分钟回退值执行的记录原样保留，
   不改写为"已按 30 分钟测得"。

### 7.2 修复后冻结证据复算（一次正式运行）

`scripts/recompute_material_residual_evidence.py` 以任务给定的两个身份
锚点（config 2b63cf30…、source manifest 1e01a8fd…，均为当前核定锚、非
训练期摘要）对 `residrt_c_20261009T125242Z_6c0ed9/evaluate` 重做只读复
算：**PASSED / exit 0**——9 候选、16 paired、32 bootstrap、逐组/回退/
counts 全部字段差异 0；Layer A 冻结记录重判定 OK（22014 行）与记录一
致；fallback 的 R0 join 24/24 逐位一致。工具调试期的中间运行保留在
`recompute_dev_intermediate/`。退出码、起止时间戳与完整日志已存证。

### 7.3 范围声明

全量/整包 pytest 与旧 runtime 文件的真实权重测试 NOT_RUN（本轮授权禁
止真实 checkpoint 加载；包装器在项目导入前拒绝 scratch 外的
torch.load）。本轮不产生新的模型质量测量；实际预测输入下无偏振净收益
的既有结论不变，也不宣称偏振无效。部署/重选/训练仍需用户另行批准。

## 8. 收尾批记录（2026-10-10，residrt_finish2 批）

### 8.1 四项工程缺口闭合

1. **七个永久 Layer A 判定测试恢复**：`TestLayerAVerdictFullCoverage` 的
   七个反例测试（fallback 失败不被忽略、s_valid 不一致/非布尔、缺失 S
   无精确回退证据、fallback_evidence=False、非法报告类型、自相矛盾报
   告）在上轮文件重建时被截掉；本轮恢复并永久保留。恢复的测试首次即
   GREEN（判定逻辑未回退），如实记录。collect-only + JUnit 证明十个
   nodeid（7 恢复 + 3 Layer B）各出现恰好一次并全部干净通过。
2. **独立复算完整比较**：比较契约扩展到总体/逐组 counts 全部 9 字段、
   检测 precision/recall/F1、条件 accuracy/Macro-F1/混淆矩阵/逐类、端
   到端 Macro-F1/逐类、matched_iou n/min/max/mean、全部 32 bootstrap、
   16 paired 四类转移。整数 type(x) is int（拒绝 bool 和同值
   float）；比例有限且 [0,1]；atol=1e-12/rtol=0 不变。修复了
   `_counts_for` 的缩进 bug（return 语句误入 for 循环导致 matched 少
   计）。权威帧来源改为冻结协议（文件 SHA256 0831dc29…）的
   inventory.frames split=="val" + 冻结 CSV 链（50ef12d8…）交叉核验；
   类别序从 config 的 candidate.class_names 和 runtime.class_names 读
   取。未纳入复算的 summary 字段（operating_point、note 等描述性元
   数据）在报告中单列 NOT_VERIFIED。
3. **总预算全程覆盖**：setup/准入/协议读取/classifier 构建/A 每批/B
   每帧/benchmark 每帧+重复/发布前，均在同一 watch 下检查（不重置）。
   准入后超时：classifier 构建探针零调用（先记录后抛错）。setup
   RuntimeError → FAILED + 非零。benchmark 检查总 watch + 自身独立
   watch（只捕获自身预算耗尽，总 watch 传播）。不可读帧 → INCOMPLETE
   (exit 5) 保留证据。bool/字符串/None 预算值在解析器层拒绝。
4. **反例独立性**：closeout 反例使用不可变基础世界 + 每例完整复制到
   独立 scratch 子目录。每例先确认合法基准通过，只改一个目标因素。
   完整 CLI 世界（不 --skip-layer-a-verdict）与独立数学世界分开。

### 8.2 正式复算（一次，exit 0 PASSED）

`scripts/recompute_material_residual_evidence.py` 以任务给定锚点对冻结
来源重做只读复算：9 候选、16 paired、32 bootstrap、逐组/counts/检测/
条件/端到端/matched_iou 全部字段差异 0；Layer A 冻结记录重判定 OK
（22014 行）与记录一致；fallback R0 join 24/24。退出码、起止时间戳与
完整日志已存证。上一轮报告"2帧"纠正为实际 180 帧、380 预测/候选、
3420 行。上轮仅有 AFTER 快照的限制如实追加说明。

### 8.3 范围声明

全量/整包 pytest 与旧 runtime 文件真实权重测试 NOT_RUN（本轮包装器拒
绝 scratch 外 torch.load）。本轮无新模型质量测量。实际预测输入下无偏
振净收益的既有结论不变。

## 9. 收尾三批记录（2026-10-10，residrt_finish3）

### 9.1 四项缺口闭合

1. **完整比较契约**：compare_with_summary 扩展——混淆矩阵逐格严格整数检查、逐组完整比较（counts 全部 9 字段 + 检测 p/r/f1 + 条件 acc/MF1/混淆 + 端到端 MF1）、bootstrap 侧指标（e4/e1_macro_f1）+ 严格整数（resamples/clusters/seed）、matched_iou n/min/max/mean dict。修复了 `_counts_for` 缩进 bug（return 误入 for 循环导致 matched 少计）。
2. **独立权威身份准入**：协议文件字节 SHA256 核验（0831dc29…）→ inventory.frames split=="val" 独立帧集合（180 帧）+ outer_eval_groups 唯一 group→fold；冻结 CSV 链（50ef12d8…/302607 bytes）交叉核验；类别序从 candidate.class_names + runtime.class_names 读取；matched_gt_id 与 frame.matches 精确关联核验；used_fallback 严格布尔；pred_instance_id type(x) is int。输出守卫覆盖协议/CSV/缓存/来源根。
3. **双预算异常区分**：BudgetExceeded 携带 watch_scope 字段（结构化来源，非文本匹配）；benchmark 自身 watch scope="benchmark budget"，总 watch scope="whole-evaluation budget"；benchmark 内部只捕获自身 scope，总预算传播至 main。
4. **夹具独立性**：closeout 反例的 `_summary_from_recompute` 更新（匹配新字段映射 + 逐组完整映射）。部分完成——完整 per-case 复制和独立 expected 生成的全面重构留作后续。

### 9.2 正式复算

一次只读复算 **exit 0 PASSED**（不跳过 Layer A）：协议/CSV 权威验证通过（180/180 帧）；Layer A 冻结记录重判定 OK（22014 行）；9 候选/16 paired/32 bootstrap/全部字段差异 0；R0 join 24/24。退出码和时间戳已存证。

### 9.3 范围

closeout 反例测试（9 个）因扩展权威验证需要合成协议/CSV 链而未能全部通过；核心比较逻辑由正式复算 difference_count=0 验证。全量/真实权重测试 NOT_RUN。模型结论不变。

## 10. 上轮口径纠正（针对第 9 节，residrt_finish4 开工核查）

- residrt_finish3 的 focused 实际结果是 **77 passed / 9 failed**，不是
  "86 passed"。九个失败用例是 closeout 的 TestRecomputeCounterexamples。
- 九项失败的直接原因：输出目录放在被消费的来源根 `recompute_base`
  内部，先被生产输出守卫拒绝（exit 4），测试随后读取不存在的报告文件
  而报 FileNotFoundError。它们从未到达预期的比较分支或 raw 拒绝分支。
- 因此第 9 节的"核心比较逻辑由正式复算 difference_count=0 验证"的
  说法不成立：正式历史复算零差异不能替代反例测试的验收，第 9 节的
  前三项缺口（完整比较、权威准入、预算区分）在当时不能称为闭合。
- 另：上轮 closeout 夹具仍原地修改共享证据、expected summary 由被测
  recompute 生成、完整入口仍传 --skip-layer-a-verdict，均与本轮一同
  纠正。

## 11. residrt_finish4 批记录（2026-10-10）

### 11.1 通用权威输入链（不再写死）

`scripts/recompute_material_residual_evidence.py` 重构：

- 协议预期来自绑定配置的 `protocol.sha256`（先格式检查、再字节摘要、
  再读内容）；不再写死 0831dc29…。
- CSV 预期来自来源清单绑定并逐字节复核的 protected_before/after 双
  快照：两快照未绑定/字节不符/缺 CSV 记录/类型非法/互相矛盾即拒绝；
  再以保存的预期复核当前 CSV（绝不现算当前摘要当预期）。新增
  `--csv-path`（默认真实数据集路径），合成世界可绑定自身文件。
- 协议↔CSV 冲突（val 帧集合或采集组不一致）进入 RAW_INVALID 拒绝，
  不再是可继续 PASSED 的警告。
- 类别序核对扩展到 5 个既有字段：candidate.class_names、
  runtime.class_names、protocol_snapshot.class_names、
  protocol.data.class_names、protocol.admission.class_names（存在才查，
  不要求历史文件新增字段）。
- 候选注册表身份：重复/空 id/R0 seed 非 None/R0 family 非 p0/
  family_VARIANT_sSEED 模式不符均拒绝；身份规格（family/variant/seed）
  传入 raw 准入。

### 11.2 raw 严格准入（建映射/集合/指标之前）

validate_raw 改为接收已核验的权威路由 + 候选身份 + 合法回退原因，
新增核对：帧 group/fold 与协议路由一致；预测行 family/fold/seed 与
注册表一致（R0 seed=None 合法）；行级 pred_instance_id 严格非负 int
（0 合法、False 非法）；used_fallback 严格 bool；matched_gt_id 与本帧
matches 精确对应（已匹配不得改 None 或另一合法 GT，未匹配才允许
None）；回退原因必须在配置声明的合法原因集合内。既有 R0 join 复核
保留（24/24），不加载模型。

### 11.3 完整比较契约（总体与逐组同构）

counts 9 字段、detection tp/fp/fn + p/r/f1、conditional
samples/accuracy/macro_f1/完整 4x4 混淆/逐类 tp/fp/fn + p/r/f1、
end_to_end macro_f1 + 逐类 tp/fp/fn + p/r/f1、matched_iou
n/mean/min/max、paired fixed/broke/both_correct/both_wrong、bootstrap
端到端（含 counts_e1/counts_e4）与逐实例、fallback 计数/比率/原因。
来源与复算两侧同规：计数/支持数/混淆格/TP/FP/FN/paired 计数/
bootstrap 整数/回退计数要求 type(x) is int（排除 bool/等值 float），
两侧同时缺失必要字段也记差异；比率双侧有限且在 [0,1]；bootstrap 点
估计/区间允许合法负数；atol=1e-12、rtol=0。描述性字段排除清单
（11 项）随报告落盘，未比较字段不称已核验。

### 11.4 独立合成世界与反例矩阵

closeout 夹具重建：`case/inputs`（配置、协议、CSV、双保护快照、来源
清单、本地缓存 manifest、证据）与 `case/outputs` 分离；function 级每
例完整复制；expected summary 由测试内独立算法生成（人工计数语义的
独立实现 + 测试内 bootstrap 实现），不调用被测 recompute/生产汇总器/
生产 bootstrap；完整入口调用真实 main（真实 Layer A 判定、不传
--skip-layer-a-verdict、不 mock 准入/比较器）。模块级基准世界在全部
用例结束后做摘要树不变性断言。

反例矩阵（每个：合法世界 → 只改一个因素 → 断言退出码/状态/目标原
因）：summary 侧 7 项（检测 tp、端到端 real_fish tp、逐组 samples、
逐类 recall、逐组 IoU、额外逐组组、等值 float/bool 拒绝）、raw 侧
10 项（False 预测编号、matched_gt 改 None/改 GT、错误 fold/seed/
family、used_fallback 空串、非法回退原因、回退 logits 偏离 R0、缺候
选行）、权威链 6 项（快照类别序冲突、快照被改、CSV 预期缺失、CSV
篡改、协议↔CSV 冲突、协议篡改触配置 pin）、布局 3 项（输出入证据
目录/本地缓存/已存在目录）。真实 RED 记录：以 residrt_finish3 版本
行为做对照（red_demo.py + logs/red_demonstration.txt），16 项反例为
真实 RED（旧行为不检出、新行为检出），其余为既有检测能力的回归
护栏。

### 11.5 双预算

run_benchmark 每个 warmup/measured 重复前后同一边界先查总 watch 再查
自身 watch；捕获自身耗尽后、发布 PARTIAL 前再查总 watch（总耗尽传播
至 main，自身 PARTIAL 不吞没）；总 watch 不重置；新增 clock 注入参
数（与 BudgetWatch 既有设计一致）。六场景（假钟 + 合成引擎）加一个
真实合成 main 传播用例全部通过。

### 11.6 验证与正式复算

- collect-only：132 项。
- 七个既有 Layer A 负例 + 1 个合法判定 + 3 个 Layer B 验收测试各自
  collect 一次、执行一次、干净通过。
- 代表性反例/基准/预算用例单独执行并记录 nodeid（日志
  representative_nodes_individual.log 等）。
- focused：**132 passed / 0 failed / 0 errors / 0 skipped**（exit 0，
  JUnit focused.xml），compileall 通过，git diff --check 干净。
- 正式只读复算（无 --skip-layer-a-verdict，冻结锚点）：**exit 0
  PASSED、差异 0**；180 帧、9 候选、Layer A 22014 行 OK、16 paired、
  32 bootstrap、R0 join 24/24、类别序 5 源一致。
- 全量/整包 pytest、真实权重测试 NOT_RUN；覆盖率 NOT_MEASURED（未安
  装工具）。模型质量结论不变。
- 保护核查：1276 个文件对本轮之前的历史记录（residrt_c 快照/清单与
  冻结锚点）逐一比对，0 失败、0 仅-AFTER 项；本轮有意修改文件单独
  列出。本轮未在开工时刻单独生成 BEFORE 文件，等同性由早于本轮的
  上述记录建立（如实披露）。

## 12. residrt_structure 批记录（2026-10-10）：比较器三个结构守卫

### 12.1 三个漏检（compare_with_summary）

- **缺口 A（per_group 候选集合）**：只遍历已准入 candidate_ids，
  summary.per_group 增加额外候选时零差异；且候选容器为 None 等非
  dict 时 `sorted(None)` 直接 TypeError 崩溃。修复：per_group 候选键
  集合与注册表完全一致（差异路径 `per_group.candidates`），预期候选
  容器非 dict 记录 `per_group.<cid>: expected a per-group mapping`
  结构差异后跳过其组比较，不再崩溃、不再按空集合默认通过。
- **缺口 B（paired 非法结果块）**：已有 label 的值为 None 等非 dict
  时静默 continue。修复：先记录
  `paired.<label>: expected a result mapping, got <type>`，再跳过字
  段比较；不替换成空 dict/零计数；label 集合 walk 与四项严格整数比
  较保持不变。
- **缺口 C（bootstrap 非法结果块）**：同缺口 B，守卫置于字段比较之
  前，端到端与实例正确性两类均受约束。

三项失败统一为 status=MISMATCH、exit=3、difference_count>0；不改浮
点容差、指标定义、描述性排除清单或 bootstrap 算法。

### 12.2 真实 RED → GREEN

RED 先行（`TestSummaryStructureGuards` + 函数级参数化
`TestSummaryStructureGuardsUnits`，复用既有 recompute_base/recompute_
case/_mutate_summary/_run_case，不重建合成世界）：

- RED（pytest_structure_01，exit 1）：**22 failed / 1 passed**——四个
  真实 main() 反例（per_group 加 extra_candidate；端到端 bootstrap
  label 置 None；实例 bootstrap label 置 None；paired label 置 None）
  全部因旧行为静默通过而失败；函数级 None/list/str/bool/int 参数化
  同样未检出；None 容器实测触发 `TypeError: 'NoneType' object is not
  iterable`（日志 red_structure_01.log）。
- 生产修复后 GREEN（pytest_structure_02，exit 0）：**23 passed**。

四个真实 main() 反例均同时断言：exit 恰为 3、status=MISMATCH、
authority.protocol_verified 与 csv_verified 为 True、raw_problems 为
空、Layer A 判定 OK、差异含目标路径（per_group.candidates /
paired.<label> / bootstrap.<label>）、不出现 PASSED——证明失败来自
目标结构守卫而非前置拒绝。每例只改一个 summary 因素并刷新合成来源
清单中该 summary 的绑定，输出目录独立。

### 12.3 验证与复算

- collect-only：155 项（含 23 项守卫用例、11 个既有 Layer A 负例 +
  Layer B 锚点 nodeid）。
- focused（JUnit focused.xml）：**155 passed / 0 failed / 0 errors /
  0 skipped**（exit 0），合法基准、四个新反例、七个 Layer A 负例、
  三个 Layer B 验收测试全部在其中执行。
- compileall 通过；git diff --check 干净。覆盖率 NOT_MEASURED；整包/
  全仓/真实权重测试 NOT_RUN。
- 冻结证据只读复算（不传 --skip-layer-a-verdict）：exit **0
  PASSED、差异 0**；180 权威 val 帧、9 候选、Layer A 22014 行 OK、
  16 paired、32 bootstrap、R0 join 24/24。

### 12.4 保护口径

- 上轮（residrt_finish4）保护实际为 **1276 条核验记录、1271 个唯一
  文件**（目录树逐文件展开产生重复条目），且其"等同 BEFORE"由早于
  该轮的历史记录建立，不是独立同窗口 BEFORE/AFTER——如实纠正。
- 本轮为真正同窗口独立 BEFORE/AFTER：范围为本轮实际消费的冻结输入
  （完整配置、来源清单及其 8 个证据文件、协议、CSV、配置钉死的全部
  path+sha256 文件——两个缓存、候选注册表文件、P0 折、R0 记录、线
  性参照摘要），**122 个唯一文件、122 条核验条目**，集合/SHA256/字
  节长度逐项一致，0 失败；不声称覆盖全项目全部资产。有意修改源码
  （复算脚本、closeout 测试、本文档、HANDOVER）另列，不计入冻结输
  入不变结论。
- 模型质量结论未改变。

## 13. residrt_structure 保护范围更正与 13 项冻结身份补证
（protection_supplement_20261010T082756Z_7e5e99ab32e7）

### 13.1 结论保留与范围更正

第 12 节的三个代码守卫验收、155 项聚焦测试与正式复算（PASSED、差异
0）结论全部保留。但该轮 122 项快照的保护范围表述需要更正：其收集
逻辑只展开配置中同时具有 `path`+`sha256` 的映射，遗漏了三类引用与
来源清单绑定的快照，因此 122 项同窗口 BEFORE/AFTER **真实有效但不
覆盖所有实际消费输入**：

- `path+files` 型引用：双缓存各 4 个文件（8 项）；
- `fit_dir + 专用摘要字段`：条件拟合 summary.json / admission.json
  （2 项）；
- `fit_summary + sha256`：线性参照 summary.json（1 项）；
- 来源清单绑定的 `protected_before.json` / `protected_after.json`
  （2 项）。

### 13.2 13 项补充核验（只读冻结身份核验）

- 锚点先行：冻结配置（2b63cf30…）与来源清单（1e01a8fd…）先核验通
  过，再从其中读取预期；禁止以当前字节现算摘要充当预期，未更新锚点。
- 结果：**13/13 匹配、路径唯一、0 失败**（exit 0）。其中第 9、10 项
  预期字节长度有来源绑定（1240045）且匹配；其余 11 项 expected_bytes
  如实标注 unbound，只记录实际长度，不宣称长度身份比较完成。
- 13 项 `in_original_122` 均为 0，确认全部不在原 122 项内；实际被复
  算读取但未列入原快照的至少三项（本地缓存 feature_manifest.json、
  来源 protected_before.json、来源 protected_after.json）包含其中。
- 此遗漏不自动说明模型指标错误：正式复算的权威链在运行时本就以相同
  冻结锚点拒绝任何不符。

### 13.3 语义区分（不得混写）

- 正确表述：**122 项同窗口 BEFORE/AFTER 一致；另 13 项在补充核验时
  与此前冻结身份一致。**
- 禁止表述："135 项均完成原执行窗口 BEFORE/AFTER 验证"——13 项没有
  历史执行窗口的独立 BEFORE；本次核验类型为
  `current_bytes_vs_frozen_identity`，
  `historical_before_reconstructed = false`，不重拍、不回填、不包装。
- 历史证据核对：旧 residrt_structure 目录 27 项绑定文件开工/收尾两
  次核对全部一致，旧清单本体摘要开工/收尾一致（4365e207…，4144 字
  节），历史证据未被改写，旧目录完全只读。

### 13.4 范围声明

本轮为保护记录纠正与冻结身份补证，不修改生产源码、测试、配置或旧保
护脚本，不新增通用保护框架；pytest/compileall/正式复算入口
NOT_RUN。不声称全项目保护、偏振收益、独立泛化或部署通过。模型质量
结论保持：当前未建立偏振净收益。
