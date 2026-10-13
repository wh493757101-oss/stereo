# material_upsample_impact_v1 — 可切换的视差上采样修复及其分类影响验证
（验收守卫补齐版 v3，2026-10-11）

状态：**完成（exit 0）**。回答：在检测 mask、模型权重、分类输入几何和
评测规则完全固定时，仅改变视差上采样，四类分类与完整系统指标怎样变
化。无训练、无重选、无工作点切换；**尚未建立偏振净收益**；指标无上
涨是如实交付的结果之一。

- 本次正式评估（本记录口径，叶目录自身身份：内部 run_id
  `20261011T054912Z_33bfd0c8`，完成于其 run_status.json 记录的
  2026-10-11T05:50:39Z）：
  `analysis/runs/material_upsample_impact_v1/20261011T054447Z_d87c01bf/formal/`
- 同批产物：`…/preflight/`、`…/smoke/`（有界 4 帧）、`…/validation/`
  （focused JUnit）；临时 scratch
  `.test_artifacts/20261011T054447Z_d87c01bf/`（收尾已清理）
- 第二批原运行（保留，供追溯，回归参照取自其叶目录）：
  `…/20261011T093000Z_d65f704e/formal_v2/`（predictions_arms SHA256
  `491c885c…eb15e2`、metrics_summary SHA256 `442c601c…52049b3`，两份
  保留身份在新运行中均实际核验）
- 上一收尾批运行（保留）：`…/20261011T130000Z_e54d0f33/`（其 formal
  清单为空、formal_v2 才是 600 项清单的当时依据；本批以 672 项清单取
  代，见 §2）

## 1. 修复契约（唯一批准的实现，本批未改）

`core/stereo_matching.py`：显式上采样版本（默认 `legacy_zero_linear_v1`
逐位不变；`valid_weighted_legacy_support_v1`：nearest 支持不变，保留位
置 `Dstar=N/W/s`（仅 W>0），有效 tap 非有限/非正拒绝，无效位置严格
0；SGBM/CLAHE/左右一致/纹理/条带/全图回退/min_valid_ratio/valid_ratio
口径不变；valid-valid 断层不修复）。单次核心复用接口
`match_cores()` + `assemble_from_cores()`，两臂从同一份核心分别装配，
证据为调用计数（本批 formal：180 帧、183 核心、legacy 调用 183、
**fix 调用 0**）。`gui/inference_engine.py` 的 `from_config` 与
`from_material_config` 均透传 `stereo.upsample_version`（缺省 legacy；
显式 fix 到达真实 matcher；未知版本拒绝经两个真实配置入口各自测试，
不只测 dataclass 构造——其中 from_material_config 的未知版本入口测试
在本批入口守卫收尾补齐，此前仅覆盖 fix 与缺省 legacy，见 §7）。
既有 YAML 字节未动。

### 兼容性记录（本批未修，不阻断 180 帧结果）

- `_match_pair` 兼容 wrapper 在返回前已应用 min_valid_ratio 策略（经
  `assemble_from_cores`），而重构前 `_match_pair` 返回未应用策略的原
  始全分辨率图、由 compute/compute_bands 在调用侧应用——直接调用该
  wrapper 的外部代码语义改变；生产 compute/compute_bands 路径不受影
  响，且 legacy 臂对 3420 行历史逐身份复现通过（本证据集上策略时点等
  价）。
- `match_cores()` 的原始重叠/越界 bands 未做裁剪归一化（与
  `compute_bands` 先裁剪合并不同）；生产调用方（引擎与评估脚本）使用
  `build_horizontal_bands`（已裁剪合并）故不受影响；直接以原始重叠/
  越界 bands 调用 `match_cores` 的调用方需自行归一化。

## 2. 固定输入、保护与身份（672 项，数量按实际记录）

四锚点核验通过并**登记为 anchor 类别**（此前版本把锚点误标为
batch2_snapshot，已更正）。`source_manifest.json` 共 **672 项**逐项记
录（路径/角色/登记与结束字节、SHA/unchanged/状态/登记类别/绑定来源）：
anchor 4、frozen_binding 217、experiment_config 1、batch2_snapshot 450
（binding_kind：anchor_or_frozen 222 + batch2_snapshot 450），全部
identical。较 600 项清单新增并补齐的实际消费集合：

- **9 候选注册表的全部 fold_files 66 个文件**（此前只枚举旧
  residual_models 注册表的 30 个路径，漏掉 66 个候选专属文件——残差
  状态、scaler、保存预测、fold 报告；路径去重，冲突绑定拒绝）；
- 本实验配置自身（experiment_config，读入建 run 后立即登记）；
- split CSV（经 batch-1 清单绑定，读前登记）；
- loader 实际消费的 fit summary 与 fit admission（runtime 配置 pin 绑
  定）、协议文件（pin 绑定）；
- 参考运行两份文件（predictions_arms 与 metrics_summary，均以实验配
  置的保留身份为绑定，消费前登记）。

冻结预期取自已核验配置/清单身份，不用当前重算哈希冒充历史。登记缺失/
结束读取失败不再抛异常吞掉清单：逐项记录 `missing_at_end`/读取失败状
态，manifest 与 FAILED 状态照常落盘。不向旧运行补造当时的保护记录；
新保护只属于新运行。

formal_v2 参照的两份保留身份在新运行中均实际核验（登记绑定 +
verify_reference_run 双重校验），6840/6840 行逐身份一致（类别/status/
fallback 精确、logits ≤1e-4）——本批守卫补齐未改变任何分类结果。

## 3. 复现门与独立复算（本批起为真逐项比对）

- legacy 臂对历史 3420 行逐身份复现（0 问题）；R0 两臂 380 对逐位一
  致；formal_v2 参照 6840 行 0 偏差（保留身份先核验）。
- **独立逐项比对（已实际执行，撤回 v2 版"未逐向量比对"的表述）**：
  生产 aggregate 的 per_class 本就包含逐类 tp/fp/fn，独立复算器（纯
  计数，不调用任何生产汇总器）与之逐项比对，`compared_fields` 共 8
  类、formal 实测 **1116 项字段比较、0 失配**：条件样本/accuracy/
  macro-F1、条件逐类 P/R/F1、**条件逐类 tp/fp/fn**、混淆矩阵、端到
  端 macro-F1 与逐类 P/R/F1、**端到端逐类 tp/fp/fn**、支持计数
  （frames/gt/predictions/matched/missed/unmatched/unclassified）、
  **fallback（生产侧直接从分类器返回结果汇总，与独立预测行计数比
  较，分母=全部 380 预测）**。变异注入测试证明：整类交换 e2e 逐类向
  量（Macro-F1 不变）、支持数、逐类 fn、混淆矩阵、fallback 计数或分
  母任何一项被改动，比较器都报错。
- 键集合完整性：预期键集合（权威帧×候选×臂）与实际行**集合相等**，
  重复键、缺失键、多余键均拒绝；正式 6840 行计数检查保留。
- 参照校验强化：logits 长度不一致明确报错（不再跳过）；valid 行
  logits 必须为 4 个有限数值（两侧都查）；两份参考文件保留身份均实
  核验。

## 4. 结果（6840 行逐身份；与 formal_v2 完全一致）

| 候选 | 臂 | 条件acc | 条件F1 | 端到端F1 | 改对/改错 | logits逐位变化对(380对中) |
|---|---|---|---|---|---|---|
| R0 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | **0** |
| conditional_RCS_s2026 | legacy | 0.9538 | 0.9205 | 0.7268 | — | 282 |
| conditional_RCS_s2026 | fix | 0.9566 | 0.9569 | 0.7301 | **+1/0** | （同对） |
| conditional_RCS_s2027 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | 377 |
| conditional_RCS_s2028 | legacy | 0.9566 | 0.9569 | 0.7301 | — | 377 |
| conditional_RCS_s2028 | fix | 0.9538 | 0.9536 | 0.7269 | **0/−1** | （同对） |
| conditional_RC_s2026 | legacy/fix | 0.9538 | 0.9205 | 0.7278 | 0/0 | **39** |
| conditional_RC_s2027 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | **0** |
| conditional_RC_s2028 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | 39 |
| linear_local_RC_s2026 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | 39 |
| linear_local_RCS_s2026 | legacy/fix | 0.9566 | 0.9569 | 0.7301 | 0/0 | 377 |

**类别不变 ≠ logits 逐位不变（v2 版表述据此更正）**：现存证据中只有
R0 与 conditional_RC_s2027 的全部 380 对 logits 逐位不变；
conditional_RC_s2026 有 39 对变化（36 对为 matched）——其类别与
fallback 在输入变化下保持不变，是"输入变化未传导到类别输出"的证据，
不是无输入变化。其余候选的 logits 变化对数见表。R0 只读灰度裁剪与
mask，其 logits 逐位不变是两臂装配共享同一核心的必然后果。

- 关键实例：RCS_s2026 `Single_Plastic_fish_20_NTU_001` pred2 改对
  （恢复其历史唯一改错）；RCS_s2028
  `Mixed_Plastic_submarine_and_Plastic_fish_20_NTU_023` pred3 改错。
  未匹配预测类别变化 1 条（RCS_s2026，
  `Single_Plastic_fish_20_NTU_021` pred3 class3→class2），逐身份证据
  （含两侧 logits）在 impact_summary.json 的 unmatched_class_changes。
- 口径区分：fix−legacy 配对 = 346 个匹配预测；within-arm 接口
  （RCS−RC、RCS−R0）= 368 GT、含 22 漏检——不混用。同臂同种子对照
  16 个标签（conditional 三种子 + linear_local s2026）随
  impact_summary 落盘；4 个采集组双臂九候选指标完整落盘
  （group_metrics.json）。
- fallback：R0 两臂均 0；其余八候选各 3/380（no_valid_s_region）；
  生产侧计数直接取自分类器返回结果，与独立行计数一致。

## 5. 测量差异量级（路径差异，非精度声明）

q 公共区 8,755,651 像素、MAE 0.016523、lost 3,467 / gained 1,266（实
例×帧记录累计口径）；视差公共区 8,795,663 像素、MAE 8.897 px、有效
性翻转 0（两臂 nearest 支持逐位相同）；池化像素 p95 NOT_COMPUTED。
（本批守卫补齐未改任何测量路径，数值与 formal_v2 一致。）

## 6. 运行与耗时（本次运行实际记录）

preflight ~4 s；smoke（4 帧）~8 s；formal：run_id
20261011T054912Z_33bfd0c8，05:49:12 起、05:50:39 完成（≈87 s 含准
入/180 帧/复算/发布），60 分钟共享预算未触顶。wall clock 口径；不作
FPS 声明。

## 7. 测试与验证（真实退出码）

1. collect-only（9 个测试文件）：**477 collected，exit 0**。
2. 聚焦 9 文件：**476 passed / 1 skipped / 0 failed，exit 0**
   （JUnit `…/20261011T054447Z_d87c01bf/validation/focused.xml`；skip
   为既有 manual benchmark 用例）。
3. preflight / smoke / formal：全部 **exit 0 COMPLETED**。
4. `git diff --check`：**exit 0**。

测试模块 33 项（删除了 3 个后定义覆盖前定义的重复测试；配置测试改用
tmp_path/本 run 唯一 scratch，不再写固定探针路径；未知版本经真实配置
入口拒绝）新增覆盖：来源登记冲突绑定拒绝、anchor 类别登记、结束快照
missing 记录；比较器变异注入（Macro-F1 不变的逐类交换、支持数、逐类
fn、混淆、fallback 计数/分母）；发布阶段越界三检查点（保护前/保护
后/finalize 前）均 PARTIAL 且证据仍落盘；重复键拒绝；logits 长度/
非有限值/双保留身份参照校验。

### 7.1 入口守卫收尾（2026-10-11 补测）

订正：上表"未知版本经真实配置入口拒绝"当时只对 `from_config` 成立，
`from_material_config` 的未知版本入口测试缺失（该入口此前仅测 fix 与
缺省 legacy，文档表述超前于测试）。本批补齐独立测试
`test_from_material_config_rejects_unknown_upsample_version`：把既有
material 运行配置复制到 tmp_path、仅把 `stereo.upsample_version` 设为
明确不存在的值，真正调用
`DualStageInferenceEngine.from_material_config`（CPU、受控
model_a_factory 桩，Model A 零加载零推理），断言 ValueError 且错误明
确来自 unknown upsample_version；原配置字节保持不变。测试模块 33→34
项。验证（唯一 scratch
`.test_artifacts/20261011T061013Z_9f14ca52/`，TEMP/TMP 指向该 scratch，
try/finally 恢复）：collect-only（本模块）**34 项 exit 0**；聚焦本模块
**34 passed / 0 skipped / 0 failed，exit 0**（JUnit
`analysis/runs/material_upsample_impact_v1/20261011T061013Z_9f14ca52/entry_guard_closeout/focused.xml`）；
`git diff --check` **exit 0**。不重跑 preflight/smoke/formal：§2–§6
的 672 项保护、1116 项比较、6840 行复现结论不受影响（纯测试新增，
零生产代码改动）。

## 8. 结论与下一批草案

结论不变：修复对现有九候选分类的影响极小且非单调（RCS_s2026 恢复历
史改错、RCS_s2028 新增错误、其余类别不变）；按候选与固定 R0 比较，
不跨候选抵消改对/改错；**尚未建立偏振净收益**。固定旧权重比较是输入
扰动影响，不是修复后重训的上限。类别不变与 logits 逐位不变两层结论
分开陈述（§4）。

下一批残差尺度实验草案保持未冻结、未实施（同前版：同版本缓存重建绑
定、训练侧冻结尺度规则、真灰度对照与质量对照单列、受控偏振打乱、预
注册推进/停止条件）；未冻结事项：donor 池定义、质量对照是否单列、修
复版本采纳决定（用户决策）。

## 9. 限制

固定旧权重（输入扰动影响，非重训上限）；仅 4 个 val 组、val 参与历
史开发、无置信区间；真实图像无视差真值（视差 MAE 8.897 px 的测距精
度影响未验证）；池化像素 p95 NOT_COMPUTED；保护 672 项为已登记来源，
非全项目资产；两条兼容性记录（§1）未修，不影响本证据集结论。
