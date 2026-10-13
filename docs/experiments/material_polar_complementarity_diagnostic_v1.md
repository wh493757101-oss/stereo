# material_polar_complementarity_v1 — 偏振互补性零训练诊断（第三批收尾）

状态：**工程验收完成；科学结论保持"尚未建立偏振净收益"**。本诊断不是
第四批实验：新模型拟合 0 次、新 scaler 拟合 0 次、分类前向 0 次；唯一
的像素计算是对面板所选帧按生产 recipe 重建修复版 q。真实预测 mask 的
系统验收、端到端训练与部署验收仍属后续批次。

**最终勘误入口（2026-10-12）**：四处文字订正（守恒展示取值、pc003 日
志字节数、缺失日志表述、环境恢复证据强度）见
[material_polar_complementarity_final_errata.md](material_polar_complementarity_final_errata.md)；
统计未重算，历史章节保留。

## 1. 背景（第三批已核验事实的归档）

第三批（material_residual_scale_v1）真实 36 次拟合与独立核验已完成：
正式拟合 COMPLETED、独立核验 problem_count=0；Gate1-4 未通过、Gate5/6
通过；按预注册决定停止本轮模型扩展——不降门槛、不挑种子、不追加搜索。
原口径（折/种子汇总）Macro-F1 均值：R0 0.9327834914858463、fixed RC
0.931763316947873、fixed RCS 0.9332926504493860、scaled RC
0.9294368431690824、scaled RCS 0.9316354979669029；scaled RCS 相对 R0
−0.114799pp、相对 scaled RC +0.219865pp、相对 fixed RCS −0.165715pp。
本诊断只消费这些已保存结果，不重跑训练/核验链。

## 2. 问题（预注册）

在停止模型扩展之后：偏振信息与灰度分类错误之间，是否存在值得继续研究
的互补性？本诊断把问题落到三件可审查的证据上：

1. **唯一实例台账**：按 sample_name 对齐 39 个保存的 outer_eval 预测
   文件（3 个 R0 + 12 组 RC/RCS），逐实例核对身份（class/group/fold
   role/has_valid_s 与权威 manifest），保留 R0 与全部 12 条路径的
   logits/预测/正确性、RCS 相对 R0 与同臂 RC 的 fixed/broke/net 标记、
   原始 logits top1−top2 margin（字段名 raw_logit_margin，无 softmax
   口径）、C14/S16/s_valid/分块有效性。
2. **面板**：全部正确性变化实例 + 看图前固定的确定性控制例（最多
   12 个），逐实例渲染灰度 crop、修复版 signed_q/abs_q（固定色标
   [-1,1]/[0,1]）、polar_valid、GT mask、interior/band 分区、S16 数值
   与全部预测结果；invalid 区域显式遮罩，禁止逐图拉伸。
3. **条件分布**（描述性）：对三组主要混淆，按全部相关实例、R0 错误、
   曾纠正/曾破坏/始终错误子集，在冻结分箱（每折 inner_train 的 C0/C3
   三分位，build_donor_bins 语义）与组/折条件下统计 support、采集组
   数、缺测数、S16 中位数/IQR。不做探针分类器、不做特征选择、不做
   阈值搜索、不构造部署路由。

## 3. 冻结输入与准入

输入均为已核验的第三批产物；7 个文件按评审记录的 SHA256 在消费前核对
（config、fit 的 summary/admission/run_status、verification_report、
缓存的 cache_binding/feature_cache.npz），不匹配即拒绝且不更新期望值。
39 个预测文件的集合摘要（相对路径排序 + `<rel>\t<bytes>\t<sha256>\n`
行、UTF-8 无 BOM、LF）按同一算法核对为
`3219de8cd8f6efc6a2a8d1821de3dae222bf3f5a80a43dfb1b153a28e201c508`。
缓存 binding 对 manifest/definition/npz 的内部绑定、完整缓存 marker、
修复版本与协议身份全部核验。来源保护复用 SourceRegistry：先登记后读
取；历史期望来自已验证来源——拟合自身 source_manifest.json 的
at-registration 摘要（本次先 snapshot 再展开）与冻结缓存自身
binding；未被覆盖的输入记录为本次消费前 snapshot；项目相对 as_posix
路径键；拒绝项目外路径与链接/重解析点；结束时全量重比对。

## 4. 预注册计数期望（由数据推导核对，不一致即拒绝）

唯一实例 2446；R0 错误 150（其中有 S 的 145）；RCS 至少一次纠正 R0
错误的唯一实例 12、至少一次破坏 R0 正确结果的 17、并集 29（只按正确
性变化定义：RC 不参与扩大、错类变错类不进入、不等于预测/logits 变化）。
三组主要混淆按唯一 R0 错误：plastic_submarine↔plastic_fish 54
（42+12）、plastic_fish↔real_fish 52（24+28）、
metal_submarine↔plastic_submarine 28（3+25）、其余 16。12 个曾纠正
实例的原混淆：real_fish→plastic_fish 8、plastic_fish→real_fish 3、
plastic_fish→plastic_submarine 1、两种潜艇 0。正式运行实测全部一致。

## 5. 局部重建（仅选中帧）

面板实例（41 个）去重到帧（39 帧）后，对每帧执行与
run_feature_build 完全一致的生产 recipe：修复版 matcher 参数、
band_margin=20、full_image_threshold=0.9、crop_pad=10、全帧 GT 实例
生成 horizontal bands、一次 compute_bands；只为选中实例保存
crop/maps 与统计，调用 instance_stats、compute_polar_features、
quality_vector_from_result、build_local_polar_stats、
build_local_control_features、region_records、interior_and_band_masks。
选中实例重建后的 C14/S16/region 与完整修复缓存逐项核对（有限值按位
比较、NaN 位置一致、region 像素计数与 blocks 逐一相等），不一致则
拒绝并停止成功发布。空 S 块保持完整 NaN 与 empty flag。不运行 G256
编码器或任何分类前向，不创建可用于训练的新缓存。

## 6. 控制例（看图前固定）

六个混淆方向各选一对：错误控制 = 该方向 R0 错误且全部六个 RCS 仍错
误、不在 29 集合内的实例中 raw_logit_margin 距合格集中位数最近者
（并列取 sample_name）；正确控制 = 与错误控制同真实类别同采集组、
R0 正确且全部六个 RCS 正确的实例中 |C3 差|+|C0 差| 最小者（并列取
sample_name）。全局去重、候选不足记录原因、不视觉补选、总数上限
12、面板总数上限 41。正式运行选中 12 个（六方向各一对，无短缺）。

## 7. 本次正式运行结果

- 唯一 run：`analysis/runs/material_polar_complementarity_diagnostic_v1/
  20261011T160738Z_fd30770f/`（最终诊断输出在其 `diagnostic/` 子目录）。
- 聚焦测试：collect-only 15 项 exit 0；**15 passed / 0 failed /
  0 errors / 0 skipped，exit 0**（focused.xml）。
- 正式诊断：30 分钟共享预算内 **COMPLETED（exit 0，实际 27.5 秒）**；
  台账 2446 行、计数与 §4 完全一致；面板 41 张（29 变化 + 12 控制、
  六方向各一对、无短缺），39 帧局部重建全部与缓存逐项核对一致；
  条件表与分箱来源落盘；来源保护 211 项 identical——8 个诊断锚点
  （7 个评审钉定输入 + cache-build 来源清单）、global_cache_binding
  2 项、fit_source_manifest 展开 1 项、**cache_build_source_manifest
  展开 159 项**（面板帧的左右图/标签/V5 npz/V5 记录的历史绑定命中）、
  其余 41 项（39 个预测文件等，拟合从未消费自身输出，无历史覆盖）为
  本次消费前 snapshot；`git diff --check` exit 0。
- 零训练证据：入口测试以探针证明 Standardizer.fit /
  build_conditional_head / conditional_logits / torch.optim.AdamW 在
  整个入口执行路径上调用数为 0；run_status 记录零调用计数
  （new_model_fits/new_scaler_fits/classifier_forwards/optimizer_steps
  均为 0）。
- 运行尝试记录（全部保留在同一 run 目录，无结果性重试）：第一次运行
  在渲染控制面板时因 panel_id 含 `>` 在 Windows 文件名非法而失败
  （`diagnostic_attempt1_failed/`，实现缺陷，修正 panel_id 消毒后重
  跑）；第二次运行 COMPLETED 但面板帧文件以消费前 snapshot 记录
  （`diagnostic_attempt2/`）；随后把 cache-build 运行的来源清单纳入
  锚点与历史展开（清单 digest 一次性钉定，来自该次已核验运行的发布
  证据），第三次运行为最终正式结果。三次运行的冻结输入、设计、门槛
  与判据完全相同，差异仅在实现缺陷修复与来源覆盖增强。

## 8. 路线建议（一页报告的结论）

首选路线 **B（主要受测量/采集因素限制）**：RCS 纠正（12）不多于破坏
（17），且三组混淆中没有一组出现跨子集一致的 S16 分布分离（机械判据
仅 plastic_fish_vs_real_fish 的 band signed_q p10/p25 出现 IQR 不重叠，
其余两组为空）；结合第三批 scaled RCS 相对 R0 为负，主导解释在测量侧。
建议（本次不实施）：对已知材质样本在记录曝光/gain 的条件下采集配准
良好的 0°/90° 与完整偏振序列（可计算 DoLP/AoLP），补齐同步与增益
元数据，并对 band 有效性与配准做一次标定复核。保留灰度基线。

## 9. 限制

- 当前 q 是双视点 0°/90° 配准差分代理，不是完整 DoLP/AoLP；曝光、
  gain/gamma、同步历史元数据不足。
- physical_object_id 不可确认；采集组不等同于独立实物。
- 内容依赖、类差异或可视模式不证明材质因果。
- 现有外层结果已用于设计本诊断；同池后续实验属探索性开发，不构成
  新的盲测确认。
- 面板的人工观察在运行时为空（观察者字段"自动规则"之外保留未知），
  等待审查者填写；疑似标签问题只登记、不改标签、不删实例、不算
  "清洗后收益"。

## 10. 收尾补正批（2026-10-12，五项复核缺陷修复）

Codex 复核确认五项问题并修复如下。原 §7 的运行记录是历史事实，原
§8 的"首选路线 B"结论现被本节订正为 **C**；§9 限制继续有效。

1. **配置身份与消费顺序**：原 run_admission 先调用 load_scale_config
   再核验钉定输入，且 main 不限制 config 路径。修复后 main 把
   args.config 纳入固定路径检查（替代路径即使内容相同也拒绝），配置
   在登记与钉定 SHA256 核验完成后才被解析（解析器调用数 0 的真实入
   口反例：替代配置、有效内容漂移均被拒；正常入口的来源清单中配置以
   anchor:diagnostic_input 登记且绑定到钉定摘要）。不修改既有期望值，
   不提供生产绕过参数。
2. **纯离线 postprocess 模式**：新增 `--mode postprocess`
   （`--diagnostic-dir` 必填；与 full 模式的参数按模式校验，冲突参数
   拒绝）。输入固定为原诊断目录
   `20261011T160738Z_fd30770f/diagnostic`，7 个旧产物按复核记录的
   SHA256 消费前登记核验（instance_ledger / binning / counts /
   selection_manifest / panels_manifest / source_manifest /
   run_status），不匹配即拒绝、不更新期望。postprocess 从保存台账重
   新推导计数（与旧 counts.json 及复核确认的
   2446/150/145/12/17/29/32/34 交叉核对）、校验台账结构（唯一名、字
   段、C14/S16 宽度、12 条路径、缺测语义）、以保存边界按
   build_donor_bins 语义重推分箱、核对旧面板覆盖（29 变化全覆盖、控
   制例与选例清单一致、文件存在、字节数一致、rebuild_check 全空），
   然后只生成新统计/计数/报告/引用旧面板的索引。不调用 rebuild_
   frame、StereoMatcher、compute_polar_features、render_panel，不读
   像素、不渲染新图。共享 30 分钟 deadline；超时 PARTIAL 非零；输入/
   契约冲突拒绝非零。测试以真实符号探针（rebuild_frame /
   StereoMatcher / compute_polar_features / render_panel，先证明探针
   对真实符号生效）证明离线分支零调用。
3. **真正的条件分布**：原表只有总体分布与条件频数（pooled median 冒
   充条件比较）。现补齐分层单元：真实类别；类别×折；类别×组（保留
   折）；类别×折×C0/C3 联合分箱；类别×组×折×联合分箱。每单元：唯一
   实例 support、组/折、缺测与内/边带有效数、各 S16 分量的有效 n/
   缺测 n/中位数/Q25/Q75、C0/C3 分位数，以及无支持/单例/单组/不可比
   较的显式标注（缺失一侧 support=0、统计 null，不制造零测量）。同
   组比较：同（真实类别,组,折,bin）内曾纠正 vs 始终错误、曾破坏 vs
   稳定正确（R0 对且六条 RCS 全对——派生子集，不新增选例）、同组同
   折同 bin 内两个真实类别；每对给出两侧 support、有效 n、中位数差
   与方向。曾破坏子集按真实类别属于类别对筛选（其新错类不必属于同
   一方向），已注明。分箱保留折语境；类别不混；不做显著性筛选、不
   挑分量、无 PASS/FAIL。low_support 类标记只是实现层展示提示。
4. **路线推断订正**：删除"两个混淆对 pooled IQR 非重叠自动选 A"、
   "broken>corrected 自动选 B"、"至少两个采集组一致"（无逐组比较支
   撑）与"测量侧主导"归因；这些实现时新增规则不再称为预注册判据。
   首选路线订正为 **C**：现有证据不足以建立偏振净收益、空间互补性或
   测量侧主导解释；停止当前数据上的模型扩张，保留灰度基线。B 只列
   为待验证的补采集/标定建议；A 只作将来假设。四个问题按"已观察/
   尚未验证/数据不足"回答（Q2 在本诊断不可验证——S16 是固定统计
   量，空间排列问题只能由像素级表征回答；本次零重建、不读像素）。
   报告生成错误修复：rebuild problems 空列表判断、brief 文件名、分
   箱来源指向 binning.json、报告/索引链接目标存在性检查、不截断
   JSON、两个分母分开（12×2446＝29352 保存预测对；6×2446＝14676 个
   RCS-vs-R0 比较，纠正 32、破坏 34）、复用声明（旧 39 帧/41 实例
   核对一致；本次重建 0 帧、新面板 0 张）。
5. **夹具与探针修复**：_fabricate_predictions 五场景（c1/c2/b1/w1/
   rc1）逐个选例立即更新 taken、场景→实例反向映射、w1 错类在三个非
   真实类别中轮换（永不命中真实类别）、场景不足直接 fixture 失败；
   独立期望（R0 错 3、曾纠正 2、曾破坏 1、并集 3、纠正对 7、破坏对
   1、场景名单、RC-only 标记、w1 不进变化集合）由场景定义给出，不
   调用被测函数生成。零训练探针改为真实符号
   （core.material_probe.Standardizer.fit、
   core.material_conditional_residual.build_conditional_head /
   conditional_logits、torch.optim.AdamW），测试先证明每个探针对真
   实符号生效，再证明真实入口零触发；不再使用 create=True 哑探针。
   新增：直接调用真实 build_ledger 的六类身份反例（缺失/重复/未知/
   class/group/fold）、分层统计使用条件内数据的构造例、发布边界过期
   （假 watch 在最终成功发布边界过期：全部阶段完成、保护复核后
   PARTIAL 非零、证据保留——不用极小预算）、rebuild recipe 接线
   （假 matcher 证明 bands 来自全帧 GT 实例集、fix 版本传递）。
6. **补正运行记录**：唯一补正 run
   `analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T020031Z_278808d7/`
   ——collect-only 37 项 exit 0；聚焦 37 passed / 0 failed / 0 errors /
   0 skipped，exit 0（focused.xml）；正式 postprocess **单次运行
   exit 0 COMPLETED（27.5 秒内完成，实际 4.5 秒，共享 30 分钟预
   算）**：台账重新推导与旧 counts.json 及复核确认事实全部一致
   （2446/150/145/12/17/29、纠正对 32、破坏对 34）；binning 重推一
   致；旧面板覆盖核对通过（29 变化全覆盖、控制例与选例清单一致、文
   件与字节数一致、rebuild_check 全空）；分层单元 506 个、同组比较
   412 个（其中两侧均有支持的可比较单元 75 个：曾纠正 vs 始终错误
   0 个、曾破坏 vs 稳定正确 3 个、金属/塑料潜艇同格 72 个）；首选
   路线 C；本次来源保护 8 项 identical（7 个旧产物 + 配置），与旧
   211 项历史保护分开报告；`git diff --check` exit 0；旧三次证据树
   119 个文件前后哈希核验不变。过程偏差如实记录：原诊断同一 run_id
   内有三次尝试（首轮面板文件名非法失败、次轮快照来源、三轮增强来
   源展开），偏离了"正式失败后停止、不自动重跑"的约定；三次冻结输
   入/设计/判据相同，未发现结果性搜索。本补正新增一次纯离线
   postprocess，不重复 q 或模型计算（测试探针证明 rebuild_frame/
   StereoMatcher/compute_polar_features/render_panel 调用数为 0）。
   最终验收交回 Codex。

## 11. 第二次收尾补正（2026-10-12，三类比较汇总/五层统计/日志归档）

Codex 确认三项问题并全部修复：

1. **三类比较汇总被覆盖**：build_conditional_analysis 的比较列表原在
   每对循环内重建、循环后只发布最后一对（旧 conditional_tables 仅含
   金属/塑料潜艇 412 条）。修复后三对分别保存（comparisons_by_group）
   且全局列表为三对完整连接：**global 1352 = 458 + 482 + 412，守恒校
   验通过**；两侧均有实例支持 **246 = 82 + 89 + 75**。每条比较带
   confusion_group、comparison_type、comparison_id（语境唯一、顺序确
   定）与完整条件键；类间比较保留双方向表示并附镜像说明（正反方向是
   同一组数据的镜像，不计独立支持）；跨对复用的实例/比较不当独立样
   本。工程对账值（458/82、482/89、412/75）只用于固定输入核对，不是
   科学门槛。
2. **五层 × 各子集统计完整生成**：新增 layered_subsets.json——每个混
   淆对 × 7 个子集（all_instances_of_the_pair / r0_errors_in_pair /
   r0_correct_in_pair / corrected / broken / always_wrong /
   stable_correct）× 5 层（L1 真实类别；L2 类别×折；L3 类别×组保留
   折；L4 类别×折×联合bin；L5 类别×组×折×bin），每格由该子集自身唯
   一成员计算（粗层重算、不由细格合并），含 support、组/折、缺测数、
   内/边带有效数、各 S16 分量有效 n/缺测 n/中位数/Q25/Q75、C0/C3 分
   位与无支持/单例/单组标注。守恒校验 **PASS**（分区 support 之和=唯
   一成员数；n_valid+n_missing=support）。正式数据共 **2719 个格**。
   "共同有效分量"只计 16 个 S16 分量（C0/C3 保留为描述性对照）。
3. **Q1 订正**：只引用 confusion_group=plastic_fish_vs_real_fish、
   comparison_type=corrected_vs_always_wrong。由台账推导：**两侧有实
   例支持 6 格、跨 5 个采集组；有至少一个共同有效 S16 分量 5 格、跨
   4 个采集组**；Single/Plastic submarine/10 NTU（fold2/bin0）两侧有
   实例但无共同有效分量，不能称偏振数值可比较。每侧 1–3 例为小样本
   描述；共同支持不自动成为跨组稳定线索。Q3 按对报告可比较格（塑/
   鱼 8、鱼/真 11、金属/塑 3，集合有重合、去重后按格计）；Q4 说明镜
   像关系。报告删除"0 个共同支持格"与"已验证未见跨组一致"两处不准
   确表述；路线 C 为当前审查结论。
4. **日志归档与清理**：上一轮的 runner/环境恢复日志未保留（当时未存
   入正式证据，无法核验——如实订正）。本轮正式证据保存 runner.ps1 副
   本、runner_console.txt、command_exit_codes.json、
   environment_restore.json、cleanup_report.json；pc002 的
   run1–run7 日志（含失败与错误记录）先归档至
   retained_test_diagnostics/pc002/（7 个文件、逐项字节数/SHA256/归
   档 SHA256/一致性核对全部一致）后删除；本轮 scratch 同步清理。
5. **补正运行记录**：唯一 run
   `analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T025124Z_83286b31/`
   ——collect-only 46 项 exit 0；聚焦 **46 passed / 0 failed / 0
   errors / 0 skipped，exit 0**（focused.xml）；正式 postprocess 单次
   运行 **exit 0 COMPLETED（13.5 秒，共享 30 分钟预算）**：计数
   2446/150/145/12/17/29、32/34 不变，29352 与 14676 口径分开；来源
   保护 8 项 identical；原诊断三次证据树 + 上一轮补正目录 **143 个文
   件前后哈希/字节数不变**；`git diff --check` exit 0。零训练/零
   scaler/零分类前向/零重建/零新面板（真实符号探针证明）。科学结论
   保持：**尚未建立偏振净收益**。

## 12. 最终文案与清理收尾（2026-10-12，无测试/无统计/无重建）

本轮只做文案与清理；统计数字全部复用已验收的
`review_closeout/20261012T025124Z_83286b31` 产物（5 个数值文件 + 2 个
旧诊断展示输入消费前按复核 SHA256 核验，全部一致），未重算、未运行
测试/postprocess/full 诊断。

1. **路线理由订正**：suggest_route 删除"分层条件统计未见跨组一致的
   可分离模式"，改为：现有小样本条件比较不足以确认跨组稳定线索；尚
   未建立偏振净收益，也未确认空间互补性或测量侧主导解释；路线 C 为
   当前审查结论，保留逐格观察，不自动升级路线。
2. **Q1 证据引用订正**：逐格中位数差与方向引用
   same_group_comparisons.json；支持数与共同有效分量数引用
   q1_fish_pair_common_support.json（只含支持数的文件不再作为方向证
   据引用）。
3. **文字订正版导出**：临时 helper（scratch 内，不入仓库正式入口）
   读取已验收产物并调用现有展示函数（suggest_route/_question_answers），
   生成 `review_closeout/20261012T032300Z_aa394db6/report_only/`（
   report.md / route_suggestion.json / text_export_manifest.json）；
   数字证据链接指向原已验收文件（不复制），链接目标存在性全部核验；
   报告注明"本次为文案订正；统计数字复用已验收产物，未重新计算"。
   首次运行因 helper 的 scratch 路径深度与 PS5.1 对内嵌中文检查脚本
   的 GBK 乱码失败（日志保留），修正后重跑通过。
4. **pc003 归档后清理**：实际存在 run1/2/3/4/7 日志（run5/6 从未产
   生，记为 missing，不推断其未执行）；日志结尾 run1 4 failed/42
   passed、run2 3/43、run3 2/44、run4 1/45、run7 46 passed，以实际日
   志为准归档至 retained_test_diagnostics/pc003/（含 archive_manifest，
   字节数/SHA256/一致性核对通过）后删除。上一轮的 runner.ps1、
   environment_restore.json、cleanup_report.json 确实存在；上一轮
   runner_console.txt 未保存、无法引用（已如实订正；本轮已保存真实
   console）。
5. **保护与验证**：旧诊断树 + 两个 review_closeout 目录 172 个文件前
   后哈希/字节数不变；AST 语法检查、链接检查、内容检查、
   git diff --check 全部 exit 0。本轮 scratch 已清理；无受阻残留。
6. 科学结论保持：**尚未建立偏振净收益**；路线 C 不变；第四批未启动。

## 13. 最终文字勘误记录（2026-10-12，只改文字）

最终文字勘误见
[material_polar_complementarity_final_errata.md](material_polar_complementarity_final_errata.md)：
①旧文字导出报告"守恒校验 None"为展示取值错误，正确值 PASS（统计已
验收，未重算，不表述为守恒曾失败）；②pc003 日志字节数以归档清单为准
（run1 7323 / run2 5253 / run3 3236 / run4 2198 / run7 102，此前完成
报告数字有误、清单本身正确）；③run5/run6 准确表述为"本次归档时未找
到，未纳入归档；不能据此判断是否曾产生或是否执行"，撤回"从未产生"的
过度推断（历史清单原文保留，解释由本节订正）；④环境恢复证据：首次记
录含实际值及恢复匹配结果（restored_values_match_saved=true），后续记
录为说明性声明、无法独立核验同等细节，不反向推断后续未恢复。本次仅
文字订正，统计未重算；来源文件前后 SHA256 一致。
