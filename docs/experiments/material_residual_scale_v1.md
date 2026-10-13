# material_residual_scale_v1 — 训练侧残差幅度实验（第三批，预注册设计）

状态：**工程验收已完成（§9–§12 补正验收通过）；真实 36 次拟合已由
用户手动执行并通过独立核验（见 §13）——Gate1-4 未通过、Gate5/6 通过，
按预注册决定停止本轮模型扩展**。科学结论保持：**尚未建立偏振净收
益**。本批不宣布部署通过、检测召回改善或实时通过；真实预测 mask 的
完整系统验收属于后续批次。第三批之后的零训练收尾诊断见
`docs/experiments/material_polar_complementarity_diagnostic_v1.md`。

**最终勘误入口（2026-10-12）**：与 §13.4–§13.5 相关的四项文字订正
（守恒展示取值 PASS、pc003 日志字节数以归档清单为准、run5/run6 的准
确表述为"本次归档时未找到，不能据此判断是否曾产生或是否执行"、环境
恢复证据强度）见
[material_polar_complementarity_final_errata.md](material_polar_complementarity_final_errata.md)；
历史章节保留。

## 1. 问题（预注册）

前两批结论：原残差 ±0.5 上界限制了可纠正错误（诊断批：2446 实例中
107 个错误分数差 ≥1；实际输入 346 匹配中 11 个 ≥1）；上采样修复影响
极小且非单调，尚未建立偏振净收益；固定旧权重的结果不代表修复输入上
重训的上限。本批在同一份修复版本输入（`valid_weighted_legacy_support_v1`）、
相同小型条件头与相同训练预算下，回答：

1. 把残差幅度与冻结 P0 自身的分数尺度匹配（scaled：a_f = 0.5·s_f），
   是否比原 ±0.5 上界（fixed：a_f = 0.5）更有效？
2. 新增收益是否超过质量控制信息本身的收益（scaled RCS vs scaled RC）？

## 2. 设计（只改变残差幅度）

- **尺度规则（训练侧，预注册）**：每折只在内层训练集上，
  float64、确定性中位数：m_i = P0 logits 最大值 − 次大值；
  s_f = max(1.0, median(m_i))。不读标签，不使用 inner_val/outer_eval；
  拒绝空集合与非有限值；同一折所有种子和变体共享 s_f。
  真实尺度（本批 dry-run 实测，来源为已核验 P0 前向）：
  fold0 s_f=7.4456（n=1116）、fold1 s_f=6.7476（n=1265）、
  fold2 s_f=7.2153（n=1191）——全部高于 1.0 下限，尺度干预真实发生
  （scaled a_f≈3.37/3.61/3.72 vs fixed 0.5）。若所有折 s_f=1，入口明
  确提示退化为相同实验，不自行换规则。
- **两幅度臂**：唯一设计差异是 a_f。统一公式
  delta = s_valid·a_f·tanh(head(x))；final = P0_logits + delta（P0 不
  缩放，上下文不变）；loss = CE + 0.01·mean(delta²)。
- **变体**：R0（原样冻结 P0，真正灰度基线，不训练，输出不依赖
  C/S/偏振可用性）；RC（C14＋原 P0 上下文，S16 输入层精确置零——含
  质量/有效性信息，报告名"灰度＋质量控制修正"，不得称纯灰度）；
  RCS（C14＋S16＋上下文）。同折同种子各路径共享初始化（末层零初始化，
  epoch0 精确等于 P0）与每折一个共享 scaler（C 全行、S 有效单元格，
  inner_train only）；缺失偏振 delta 精确为零、精确回退本折原 P0。
- **预算**：CPU、full batch、≤200 epoch、AdamW lr=0.01 wd=0.001、
  种子 2026/2027/2028；无增强/类别加权/额外调参。2 臂 × RC/RCS ×
  3 折 × 3 种子 = **36 次真实拟合**，全部报告，不挑种子。
- **打乱（内容依赖诊断）**：内容种子 2026–2030；箱边界仅由本折
  inner_train 确定（修复缓存 C 的 valid_ratio/灰度均值三分位 ×
  interior/band 有效模式复合箱）；donor=同协议角色、同复合箱、偏振可
  用；不放回、无固定点、不用自身、不跨折、不用旧 test、不按 GT 类别；
  完整标准化 S16 来自同一 donor，接收方其余全部保留；所有幅度臂/种子
  复用同内容种子身份映射；正常输出绑定各自已保存的 RCS 结果。
  打乱下降只作内容依赖诊断，不单独宣称材质因果；不降或覆盖不足则明
  确"证据不足"，不算偏振证明通过。

## 3. 预注册推进门槛（全部满足才建议后续实际 mask 验证）

1. scaled RCS 三折三种子平均 Macro-F1：相对 R0 ≥ +1.0pp，相对同臂
   RC ≥ +0.5pp；
2. scaled RCS 相对同输入 fixed RCS 平均 Macro-F1 严格提高且成对累计
   净改对；
3. 至少两个种子相对 R0 与同臂 RC 均净改对；
4. 至少两折种子平均相对两种对照不下降；最差折下降 ≤1.0pp；
5. 任一类别 pooled 召回复原均值相对两种对照下降 ≤1.0pp；
6. 至少两个采集组相对 R0 净改对（报告收益集中程度）。

这些是工程推进门槛，不是独立泛化或统计显著性标准。只优于 R0 不优于
RC 不得归因于偏振内容；只优于 fixed 而不过其余门槛则报告"尺度干预有
效但收益不足"。未通过则停止本轮模型扩展，不调门槛、不挑种子、不追
加搜索。种子累计为 sample-model pairs，不当三倍独立样本；R0 每样本一
份独立观测；按 sample_name 对齐。

## 4. 修复版本特征重建（本批已完成）

- 通过**生产** StereoMatcher（`StereoMatcherConfig(upsample_version=
  valid_weighted_legacy_support_v1)`，参数与冻结运行时配置逐项相等）
  和既有后处理生成偏振输入；已校正左右图直接使用，不重复校正；不
  复制另一套 SGBM/上采样/偏振公式。
- 开发池 = 原 train+val：**21 组、945 帧、2446 实例、4 个无目标帧**
  （empty: Single_Real_fish_0_NTU_010/011/014/019）；941 个含实例帧全
  部构建。旧 test 不产出任何样本、统计、训练行或缓存行（全 V5 准入
  gate 为完整性核验会重摘要 test 资产，已在报告中如实披露）。
- core/gt_geometry.py 复现 GT 实例、band bbox 与裁剪窗口；逐样本核对
  gray 裁剪、instance_mask、类别与组身份与原 V5 一致；G256 从冻结全
  局缓存按 sample_name 复制，**数组字节不变**（实测 array_equal 为
  True）；signed_q/abs_q/polar_valid/quality 经修复路径重算，C14/S16
  与缺失状态随之重算（**不是**旧缓存改版本标记）。
- 偏振缺失状态变化：实测 **0**（修复上采样不改变 nearest 支持集，样
  本级缺失状态与旧版一致——作为实测结果如实记录，不作为承诺）。
- 空S块=完整NaN块+显式标记；拒绝部分NaN/Inf/状态矛盾（verify_local_
  cache 契约 + 本实验 marker 双重校验）。
- 新缓存 `…/cache_build/features`：cache_kind=residual_scale_v1_dev_full、
  complete=true、upsample_version 绑定、逐样本 fix_polar 溯源与区域记
  录、构建报告、来源保护清单（5293 项，全部 identical）；smoke 缓存
  标记不完整（4帧/15实例），训练入口 marker 检查拒绝消费（有测试）。

## 5. 接口（与实现一致）

- 准备：`scripts/prepare_material_residual_scale_features.py
  --config --mode preflight|smoke|build --output-dir [--max-frames 仅
  smoke] --deadline-utc`（同一真实非训练流程共享一个 deadline）。
- 训练：`scripts/train_material_residual_scale.py --config --feature-dir
  --p0-fit-dir --output-dir (--dry-run|--fit 互斥) (--budget-minutes|
  --deadline-utc 互斥)`；预算覆盖准入、36 次拟合、评估、打乱、保存与
  核验，不逐折重置；到期停新计算、发布 PARTIAL/FAILED、不判成功；
  dry-run 零训练调用（探针测试证明）。
- 核验：`scripts/verify_material_residual_scale.py --config --feature-dir
  --p0-fit-dir --experiment-fit-dir --output-dir`——独立复算尺度来源、
  P0、选模历史、预测、逐类/逐组指标（纯计数循环，不调用生产汇总器）
  及成对结果；拒绝错误尺度、错误角色、额外/缺失样本、选模历史不符、
  以及 Macro-F1 不变但逐类计数被交换的反例（均有测试）。
- 手动：`scripts/manual_material_residual_scale.ps1` 默认 -Mode dryrun；
  仅 -Mode fit 执行真实拟合并随后用同一预算正式核验；超时/失败保留已
  完成证据，不报完整成功。

## 6. 本批工程验收记录（真实命令与退出码）

唯一 run：`analysis/runs/material_residual_scale_v1/20261011T074823Z_e36bf75e/`。

1. collect-only（7 个测试文件）：**246 collected，exit 0**。
2. 聚焦 7 文件（-W error）：**246 passed，exit 0**（JUnit
   `…/focused.xml`）。
3. preflight：**exit 0 COMPLETED**（锚点 5 项核验；开发池
   945/2446/21/4 逐项符合；来源保护 24 项 identical）。
4. smoke（4 帧）：**exit 0 COMPLETED**（4帧/15实例；缓存标记
   incomplete；修复版本绑定）。
5. build（945 帧开发池）：**exit 0 COMPLETED**（941 含实例帧/2446 实
   例；G 字节不变；s_valid 变化 0；保护 5293 项 identical；完整缓存
   通过通用契约自核验）。
6. dry-run：**exit 0 COMPLETED**（P0 三折身份 max logit error 0.0、
   预测零偏差；三折尺度 7.4456/6.7476/7.2153；36 路径矩阵落盘，矩阵
   幅度 0.5/3.37/3.61/3.72；保护 21 项 identical）。
7. `git diff --check`：**exit 0**。

准入守卫实测：训练入口对 smoke 缓存/无标记旧缓存/不完整缓存拒绝
（有测试）；`frames_built` 语义 = 含实例帧数，与 945 − 4 空帧核对
（首轮 dry-run 因 941 ≠ 945 被正确拒绝一次，修正核对规则后通过——
拒绝证据保留于 `…/dryrun/`，成功运行在 `…/dryrun2/`）。

## 7. 结论与后续

- 本批工程验收通过；**真实 36 次拟合及其结果需要用户手动执行后交回
  审查**；真实拟合前不得写任何外层质量结论。
- 科学结论暂保持：工程链路已验收；**尚未建立偏振净收益**。
- 手动命令（占位符替换为实际路径；输出目录必须尚不存在）：

```powershell
& 'D:\Python\Project\my_data\scripts\manual_material_residual_scale.ps1' -Mode fit -FeatureDir '<cache_build\features 绝对路径>' -OutDir '<全新正式拟合绝对路径>' -BudgetMinutes 60
```

## 8. 限制

离线组隔离实验（外层评估基于 GT 匹配，非检测前端）；真实预测 mask
的完整系统验收属于后续批次；4 个 val 历史开发组参与协议轮转，结论不
宣称独立泛化；打乱诊断是内容依赖证据，不是材质因果证明；三折尺度来
自固定 P0 的 logits 分布，若未来 P0 变更需重算。


## 9. 复核收尾批（2026-10-11，工程补正）

第三方复核发现以下问题；本节记录修复与重新验收。此前 246 项聚焦测试
通过是历史事实，但它们**未覆盖**以下缺陷（相应测试缺失或断言不足）：

1. **PS1 原生参数传递失效**：旧脚本用哈希表 splat 传参给原生
   python.exe（Windows PowerShell 5.1 下对原生命令不可靠），且
   `$Deadline` 用自定义格式 'zzz'——它按机器本机偏移（+08:00）标注
   一个 UTC 值，入口解析后 deadlines 偏移整整 8 小时，首轮 dry-run 被
   误判超时。修复：字符串数组逐元素传参（含空格路径作为单一参数）；
   deadline 以显式字面量 'Z' 生成；新增 -PythonExe 参数仅用于测试注
   入录参替身（正式默认解释器不变）。新增真实执行测试（非仅
   ParseFile）：dryrun 参数核对（含空格路径单一参数、--dry-run、
   --deadline-utc、无 --budget-minutes）、fit 非零退出传播且跳过核
   验、fit 成功后核验以同一 deadline 执行。
2. **Gate3/Gate5 判定错误**：Gate3 旧实现把"相对 R0 有两个正种子"与
   "相对 RC 有两个正种子"分开计数，未取同一种子的交集；Gate5 把召回
   差（比例）直接与 1.0 比较，等于允许 50pp 下降。修复：Gate3 取两
   比较正种子交集 ≥2（输出 seeds_positive_in_both）；Gate5 换算百分
   点后与 1.0pp 比较。新增已知答案反例：交集仅 1 个种子必须失败；
   1pp 边界通过、1.5pp 与 20pp 失败。六条门槛原文含义不变。
3. **共享截止时间落实**：手动入口只生成一次 UTC deadline，fit 与
   verify 传同一 --deadline-utc（不再传 --budget-minutes、不重置）；
   准入、dry-run、每条拟合、每折打乱、核验各阶段及**成功发布边界**
   均检查截止；最后一条拟合/最后一段打乱或核验跨截止时发布 PARTIAL
   （含已完成拟合、gates 与打乱证据）并非零退出，绝不发布
   COMPLETED。测试覆盖：入口已过期（零训练调用）、完成发布边界过
   期（36 路径保留 + PARTIAL）、打乱中过期、核验尾部过期。
4. **独立核验补齐**：核验器新增——重载模型并重建打乱前向（reassemble
   shuffled S16 → 同一冻结头+幅度前向），逐样本核对 normal/shuffled
   logits 与预测、覆盖率、同组 donor 比例、净纠错；跨臂/跨训练种子
   donor 映射一致；汇总均值（折均值→种子均值、pooled 均值）、R0 折
   指标（含逐折条目）、逐组净纠错计数全部独立重算并比对；六条门槛从
   重算结构重新评估并比对判定与建议；模型 fold/seed/arm/variant 身
   份、有限正幅度、预算（epochs/lr/wd/aux/hidden/cpu/full-batch）与
   共享 scaler 逐路径核对。新增入口级篡改反例：跨种子均值、R0 折指
   标、逐组计数、gates、打乱预测、模型身份、scaler——逐一被拒绝。
5. **源准入保护补齐**：消费前从已核验锚点展开历史哈希绑定（运行时
   配置锚点 pins P0 checkpoint/oracle/scaler 及缓存文件；batch-1 清单
   的 consumed_inputs；全局缓存自身 binding），历史期望与当前重算哈
   希严格区分（登记行分 anchor_or_frozen 与本批 snapshot）；pair_
   manifest.csv 与实际消费记录消费前登记；缓存 binding 复核 + 链接/
   重解析点拒绝。真实入口反例：缓存 npz 在 binding 后被改动 → 拒
   绝；缓存缺 feature_manifest.json（遗漏依赖）→ 拒绝；历史绑定后
   文件漂移（合成历史清单钉住合成 P0 checkpoint，漂移后）→ 拒绝。
   不回填、不修改任何历史清单。
6. **补正验收运行记录**：唯一 run
   `analysis/runs/material_residual_scale_v1/review_closeout/20261011T090556Z_b84ec6c5/`
   ——collect-only 266 项 exit 0；聚焦 7 文件 **266 passed / 0 failed /
   0 errors / 0 skipped，exit 0**（focused.xml）；修复后的 PS1 以真实
   解释器启动 dry-run（`…/manual_dryrun/`）：**exit 0 COMPLETED**，
   36 路径矩阵、P0 三折身份 max logit error 0.0（atol 1e-4 不放宽）、
   三折尺度 7.445633/6.747567/7.215330 与原记录**逐位一致**、来源保
   护 21 项 identical（含 9 项 runtime_config_anchor 历史冻结绑定与
   batch-1 清单绑定）；`git diff --check` exit 0。旧证据（batch-2
   formal_v2 双保留身份、batch-3 完整缓存 2446 行/2341 s_valid、全部
   旧运行树）核验不变。真实 fit/外层质量继续 NOT_RUN；打乱覆盖率继
   续 NOT_MEASURED；科学结论不变：**尚未建立偏振净收益**。

## 10. 复核收尾第二批（2026-10-11，独立核验/历史准入/最终预算边界补齐）

第二次复核发现以下遗漏；本批补齐并重新验收。266 项通过是上一批的历
史事实，但它们**未覆盖**以下缺陷：

1. **绑定路径键错位（"展开成功但不命中"）**：历史绑定展开用原始字符
   串（含绝对路径）做键，而登记器按项目相对 as_posix 查找——训练入
   口自身的修复缓存 binding 也用绝对键，绑定从未命中（旧缓存拒绝反
   例是靠通用契约通过的，理由错误）。修复：bind() 统一归一化为项目
   相对 as_posix 键；项目外路径与链接/重解析点直接拒绝。反例升级：
   用**绝对路径**记录的合成历史清单钉住文件，漂移后仍被拒绝（证明键
   真正命中）；来源行区分历史绑定（runtime_config_anchor/
   batch1_source_manifest/global_cache_binding/scale_cache_binding）
   与本批 snapshot。
2. **核验入口接入历史准入**：verify 此前完全未接注册器与历史绑定；
   现在与另外两个入口一致（锚点核验 → 历史绑定展开 → 修复缓存/
   全局缓存/协议/P0 summary 与 checkpoint/scaler/history/oracle 登记
   → 通用缓存契约）。三个入口的登记覆盖以真实 dry-run/合成清单断言
   （角色逐一检查，含 p0_model:foldN_P0:history.json）。
3. **权威身份绑定**：预测行的 sample_name/class_id/group 绑定权威
   manifest；逐组计数使用**manifest 组**，行自报组不一致即拒绝（反
   例：改行 group 并同步汇总仍被拒）；不在权威名单中的行拒绝，绝不
   跳过。
4. **独立计数补齐**：折级 accuracy 与混淆矩阵（含 R0 逐折重算）；
   成对四计数与 support；打乱 macro_f1（重建前向重算）；重复/缺失
   打乱记录拒绝；各 shuffle_summary.json 与最终 summary["shuffle"] 逐
   字段对 per-seed 文件核对；六门槛除 pass 外逐字段核对详细数值/种子
   与组名单（反例：pass 不变仅改 gain 值被拒）。新反例全部经真实
   verify 入口：仅改打乱 macro_f1、仅改 shuffle_summary、仅改
   summary["shuffle"]、仅改折 accuracy、仅改 both_correct/support、
   改行组+同步汇总、gates 详细值、重复/缺失打乱记录——全部拒绝；正
   常合成拟合仍通过。
5. **最终预算边界**：prepare 的 preflight 计划发布前、train/prepare
   的保护核验完成后与成功状态发布前均检查 deadline——保护核验本身跨
   截止发布 PARTIAL（证据保留、summary 与 run_status 一致、36 路径保
   留），绝不 COMPLETED。测试覆盖：prepare preflight 跨截止（不发布
   成功计划）、prepare/train 保护边界跨截止、verify 尾部跨截止；既
   有预算反例保留。
6. **补正验收运行记录（本批）**：唯一 run
   `analysis/runs/material_residual_scale_v1/review_closeout/20261011T094336Z_5e0d6084/`
   ——collect-only 282 项 exit 0；聚焦 7 文件 **282 passed / 0 failed /
   0 errors / 0 skipped，exit 0**（focused.xml）；修复后 PS1 真实解释
   器 dry-run（`…/manual_dryrun/`）：exit 0 COMPLETED，36 路径矩阵、
   P0 三折身份 max logit error 0.0、三折尺度与原记录逐位一致、来源保
   护 28 项 identical 且**历史绑定真实命中**
   （runtime_config_anchor 13 项 = P0 checkpoint/oracle/scaler×3折 +
   encoder/缓存 pins；scale_cache_binding 3 项；batch-1 清单绑定 1
   项）；`git diff --check` exit 0；冻结完整缓存哈希与 batch-2 双保
   留身份核验不变（另有常驻测试
   test_frozen_full_cache_hash_unchanged）。真实 fit/外层质量继续
   NOT_RUN；打乱覆盖率继续 NOT_MEASURED；**尚未建立偏振净收益**。

## 11. 复核收尾第三批（2026-10-11，消费前准入/证据完整核验/最终边界补齐）

第三次复核发现三项遗漏并补齐（282 项为上一批历史事实，未覆盖以下
问题）：

1. **消费前准入顺序真正落实**：prepare 此前未实际调用历史展开函数
   （静默替换失败）；三个入口的登记次序不一。修复后共用顺序：核验
   锚点 → 展开历史期望（展开函数自身读取的 batch-1 清单与全局缓存
   binding 记录先登记再读取；全局缓存 binding 是新源、消费前
   snapshot）→ 对实际依赖登记并按期望哈希核验 → 才调用消费者
   （frozen 链 / V5 gate / 核验器）。train 与 verify 的协议、全局缓
   存、P0 summary/admission/checkpoint/scaler/history/oracle 登记全部
   移至 frozen.run_admission 之前；verify 现也发布自己的来源保护清
   单。新增三个真实 main 入口的顺序探针测试：消费者探针在冻结链/
   V5 gate 边界停止后，来源清单证明依赖已登记并按期望哈希核验；正常
   输入允许消费者执行；合成历史清单钉住文件后修改其有效内容 → 注册
   即拒绝且消费者调用数为 0（prepare/train/verify 各一，文件字节
   try/finally 恢复，共享合成世界不被破坏）。
2. **证据完整核验补齐**：summary["shuffle"] 缺失或任一 content_seed
   条目缺失 → 拒绝（不再 if 跳过）；coverage 必须存在且有限、
   same_group_donor_ratio 仅 matched=0 时允许 None、receivers/
   unmatched 重算；折级 rows/accuracy/混淆/逐类召回（含 R0 逐折）；
   pooled accuracy/accuracy_mean（折均值定义）/pooled_sample_count/
   macro_f1_seed_range；成对 total 四计数；打乱 macro_f1 重建重算。
   反例：删整个 summary["shuffle"]、删 content_seed 条目、同时删
   per-seed 文件与汇总的 coverage、仅改 pooled accuracy /
   accuracy_mean / pooled_sample_count / paired total
   both_correct / 汇总 same_group_donor_ratio / seed_range / 折
   rows —— 全部拒绝并点名对应字段；正常合成拟合仍通过。训练摘要补
   写 accuracy_mean（种子级折均值、变体级种子均值）。
3. **报告真实性**：group+同步汇总反例实际同时修改行与对应逐组汇
   总，仍被权威身份拒绝；preflight 超时反例改为前置检查通过、真实
   preflight 完成后于计划发布边界过期（不用 2000 年 deadline）；上
   一批遗留空目录 .test_artifacts/run_20261011T094401Z_43960（0 文
   件）确认归属后以 rmdir 删除；文档按实际 binding_source 统计区分
   batch-1 清单自身锚点与清单引用文件的绑定命中。
4. **补正验收运行记录（本批）**：唯一 run
   `analysis/runs/material_residual_scale_v1/review_closeout/20261011T114735Z_e1cc9fc5/`
   ——collect-only 299 项 exit 0；聚焦 7 文件 **299 passed / 0 failed /
   0 errors / 0 skipped，exit 0**（focused.xml）；修复后 PS1 真实解释
   器 dry-run（`…/manual_dryrun/`）：exit 0 COMPLETED，36 路径矩阵、
   P0 三折身份 max logit error 0.0、三折尺度与原记录逐位一致、来源保
   护 28 项 identical（runtime_config_anchor 13 项历史绑定命中、
   scale_cache_binding 3 项）；`git diff --check` exit 0；冻结完整缓
   存哈希与 batch-2 双保留身份核验不变；上一批空目录
   run_20261011T094401Z_43960（0 文件）确认后 rmdir 删除。真实
   fit/外层质量继续 NOT_RUN；打乱覆盖率继续 NOT_MEASURED；**尚未建
   立偏振净收益**。

## 12. 复核收尾第四批（2026-10-11，prepare 缓存消费顺序/统计字段核验/漂移有效性订正）

第四次复核确认三项遗漏并补齐（299 项为上一批历史事实，未覆盖以下
问题）：

1. **prepare 全局缓存消费顺序**：`register_global_cache` 此前先调用
   `baseline.verify_feature_cache` 再登记四个缓存文件——消费者先于来
   源登记运行。修复后四个文件（cache_binding.json / feature_cache.npz
   / feature_manifest.json / feature_definition.json）的登记（含对展
   开后历史期望的哈希核验）全部在消费者调用之前；角色名
   `global_cache:<文件名>`、历史绑定、返回值与既有缓存契约不变。新增
   经 prepare 真实入口的边界探针测试：探针停在 verify_feature_cache
   边界时四个角色已登记，且三个缓存文件的 binding_source =
   global_cache_binding（全局缓存自身 binding 记录的历史绑定命中；
   binding 记录本身是绑定来源、在展开期消费前 snapshot）；合成世界将
   feature_manifest.json 以**有效 JSON** 内容漂移后，登记阶段即拒
   绝、verify_feature_cache 调用数为 0；未漂移入口正常 COMPLETED。
2. **统计字段核验补齐**：独立核验器新增——每个 arm/variant/seed 的
   `per_seed[seed].accuracy_mean` 等于该种子三折重算 accuracy 的算术
   平均（含 R0）；各 variant 顶层 accuracy_mean 等于种子均值的算术平
   均（含 R0）——R0 的核验不再放在仅非 R0 执行的分支中（R0 三折跨种
   子相同，其三个种子均值精确重合，均值即该公共折均值，与训练侧
   R0 顶层写法逐位一致）。打乱两层汇总（shuffle_summary.json 的
   seeds[content_seed] 与最终 summary["shuffle"] 条目）改为与独立核
   验过的 per-content-seed 文件按 8 字段逐字段核对（receivers /
   matched / unmatched / coverage / same_group_donor_ratio / macro_f1
   / vs_normal_rcs_fixed / vs_normal_rcs_broke）；必需字段或条目缺失
   直接拒绝（不再经 get(...)->None 把缺失当成合法值），计数字段必须
   为真整数、浮点字段必须有限、same_group_donor_ratio 仅在
   matched=0 时允许显式 None；精度规则沿用既有 _clean 逐位比较，未
   放宽容差。新反例（全部经真实 verify 入口、单字段篡改）：per-seed
   accuracy_mean、R0 顶层 accuracy_mean、shuffle_summary 的
   same_group_donor_ratio、两层各自的 receivers 与 unmatched、per-
   seed 文件删除必需字段、shuffle_summary 计数改为浮点（12 == 12.0
   使值比较失效——整数计数契约是唯一拒绝点）、summary["shuffle"] 在
   matched>0 时把 ratio 置 None——全部 EXIT_MISMATCH 并点名对应字
   段；正常合成完整拟合证据仍通过。
3. **漂移有效性订正**：§11 第 1 条所述"合成历史清单钉住文件后修改
   其有效内容"，当时的实现是末字节 XOR，**不能据此声称漂移后仍为有
   效内容**（该表述超出当时的实际证据；本节按追加说明订正，不回写历
   史清单）。本批将 `_pin_and_drift` 升级为 `_valid_content_drift`：
   JSON 解析后改值重写、CSV 重新解析改数据单元格重写、checkpoint
   torch.load 改载荷后 torch.save，并在测试内断言漂移后仍可解析/加
   载且内容确实改变——漂移拒绝因此证明来自哈希不匹配，而非损坏文
   件反序列化失败。附带发现：生产 dataset_manifest.csv 写出在
   Windows 上为 \r\r\n 行尾（csv 的 \r\n 经换行翻译的 write_text），
   读取方 csv.DictReader 按规范跳过空行故不受影响；合成漂移读取按
   同样规则处理。上一批遗留空目录
   `.test_artifacts/run_20261011T114800Z_35220`（0 文件、非链接/重解
   析点、归属上一批测试窗口）复核后删除。
4. **补正验收运行记录（本批）**：唯一 run
   `analysis/runs/material_residual_scale_v1/review_closeout/20261011T123358Z_d3ea4bb5/`
   ——collect-only **312 项 exit 0**；聚焦 7 文件 **312 passed / 0
   failed / 0 errors / 0 skipped，exit 0**（focused.xml；首轮聚焦运
   行因一处测试代码缺陷失败一次，日志保留于 diagnostics/，修正后重
   跑）；PS1 真实解释器 dry-run（`…/manual_dryrun/`）：**exit 0
   COMPLETED**，36 路径矩阵、P0 三折身份 max logit error 0.0（atol
   1e-4 不放宽）、三折尺度 7.445633351802826 / 6.7475669384002686 /
   7.2153297662734985 与原记录**逐位一致**、来源保护 28 项
   identical（runtime_config_anchor 13 项历史绑定命中、
   scale_cache_binding 3 项）；`git diff --check` exit 0；冻结完整
   缓存与三个旧保留证据树 28 个文件哈希核验**不变**（首轮 dry-run
   因 shell 引号转义把输出写到了项目外的一个目录，已原样保留该次日
   志于 diagnostics/ 并以正确路径重跑，误写目录确认归属后删除）。真
   实 fit/外层质量继续 NOT_RUN；打乱覆盖率继续 NOT_MEASURED；**尚未
   建立偏振净收益**。
## 13. 真实拟合结果归档与零训练收尾诊断（2026-10-11/12）

本节追加记录真实 36 次拟合的复核结果与第三批之后的零训练收尾诊断；
不重跑训练/核验链，不建立通用验收框架。

### 13.1 真实 36 次拟合（用户手动执行，已独立核验）

- 正式拟合目录：
  `analysis/runs/material_residual_scale_v1/fit_cdd17d10ea5b49f4b99c06841ec10320`
  （run_status **COMPLETED**）；独立核验目录
  `…fit_cdd17d10ea5b49f4b99c06841ec10320_verification`
  （verification_report **problem_count=0**）。
- 门槛结果（与预注册原文一致，未调整）：**Gate1、Gate2、Gate3、Gate4
  未通过；Gate5、Gate6 通过**。按原预注册决定：停止本轮模型扩展，
  不降低门槛、不挑选种子、不追加搜索。
- 原口径（折/种子汇总，不是 pooled 指标、不是独立样本统计）
  Macro-F1 均值：R0 **0.9327834914858463**、fixed RC
  **0.931763316947873**、fixed RCS **0.9332926504493860**、scaled RC
  **0.9294368431690824**、scaled RCS **0.9316354979669029**。
  scaled RCS 相对 R0 **−0.114799pp**、相对 scaled RC **+0.219865pp**、
  相对 fixed RCS **−0.165715pp**。
- 科学结论保持：**尚未建立偏振净收益**；scaled RCS 优于 R0 与 RC 的
  推进门槛未通过。

### 13.2 零训练收尾诊断（material_polar_complementarity_v1）

第三批停止模型扩展后，按预注册的零训练诊断判断偏振与灰度错误是否存
在值得继续研究的互补性。诊断只消费已保存的 39 个 outer_eval 预测文
件、独立核验报告与冻结修复缓存：新模型拟合 0 次、新 scaler 拟合 0
次、分类前向 0 次；唯一像素计算是面板所选 39 帧按生产 recipe 的修复
版 q 局部重建（逐项与缓存核对一致）。设计与结果见
`docs/experiments/material_polar_complementarity_diagnostic_v1.md`；
要点：唯一实例 2446、R0 错误 150（有 S 的 145）、RCS 曾纠正 12/曾破
坏 17/正确性变化并集 29，全部由数据推导并与预注册期望一致；三组主要
混淆 54/52/28、其余 16；12 个曾纠正实例的原混淆 8/3/1/0。首选路线
**B**：RCS 纠正不多于破坏、三组混淆均无跨子集一致的 S16 分布分离，
主导解释在测量侧（双视点 0°/90° 差分代理、曝光/增益元数据缺失、
band 边缘配准），建议补采集/标定而非继续模型工作。科学结论不变：
**尚未建立偏振净收益**。

### 13.3 收尾补正订正（2026-10-12）

复核确认 §13.2 所述诊断的以下结论与实现需要订正（详见
material_polar_complementarity_diagnostic_v1.md §10）：

- 原诊断 15 项聚焦测试通过是历史事实，但部分场景夹具与探针未覆盖其
  声称的行为（场景选例共享实例、探针打在不存在的别名上、期望由被测
  函数生成）；补正批已修复并扩充为 37 项。
- 原首选路线 **B（测量因素主导）依据不足**，订正为 **C：证据不足**，
  停止当前数据上的模型扩张、保留灰度基线；B 只列为待验证的补采集/
  标定建议。"pooled IQR 非重叠自动选 A"、"broken>corrected 自动选 B"
  等实现时新增规则已删除，不作为路线判据。
- 原条件表只有总体分布与条件频数；补正后输出真实分层单元（真实类别
  /×折/×组/×折×联合分箱/×组×折×联合分箱，含各 S16 分量的有效 n、
  缺测 n、中位数与 Q25/Q75）与同组比较。
- 原报告"与缓存逐项核对不一致"是实现层列表判断错误；旧证据 41 项
  rebuild_check 全部为空问题列表。
- 原条件统计为 pooled median 与条件频数，不能称为按条件比较——本次
  postprocess 已补齐真实分层；未发现结果性搜索。
- 科学结论不变：**尚未建立偏振净收益**。

### 13.4 第二次收尾补正订正（2026-10-12）

§13.3 所述补正的后续订正（详见 material_polar_complementarity_
diagnostic_v1.md §11）：三类比较汇总在上一轮补正中仍被逐对覆盖（仅
余金属/塑料潜艇 412 条），本轮修复为三对完整保存（1352 = 458+482+
412，两侧有支持 246 = 82+89+75）；五层×各子集统计本轮才完整落盘
（2719 格，守恒 PASS），"已完整分层/已验证未见跨组一致"的表述撤回；
真假鱼 corrected-vs-always-wrong 实有 6 个两侧支持格（5 组）、5 格有
共同有效 S16 分量（上轮"0 格"不准确）。首选路线维持 C（当前审查结
论，非自动分类）。上一轮 runner/环境恢复日志未保留（如实订正）；本
轮已保存。科学结论不变：**尚未建立偏振净收益**。

### 13.5 最终文案与清理收尾（2026-10-12）

material_polar_complementarity_v1 的最终文案订正与 pc003 清理记录见
material_polar_complementarity_diagnostic_v1.md §12：路线理由与 Q1 引
用订正为文字订正版（统计复用已验收产物，未重算）；pc003 实际日志
（run1/2/3/4/7，run5/6 未产生）归档后清理；上一轮 runner_console 缺
失如实订正，本轮已真实保存。科学结论不变：**尚未建立偏振净收益**。

### 13.6 最终文字勘误记录（2026-10-12，只改文字）

最终勘误见
[material_polar_complementarity_final_errata.md](material_polar_complementarity_final_errata.md)。
与本文相关的订正：§13.5 中"run5/6 未产生"的表述撤回——准确表述为
"本次归档时未找到，未纳入归档；不能据此判断是否曾产生或是否执行"；
pc003 日志字节数以归档清单为准（run1 7323 / run2 5253 / run3 3236 /
run4 2198 / run7 102）。仅文字订正，统计未重算；来源文件前后 SHA256
一致。
