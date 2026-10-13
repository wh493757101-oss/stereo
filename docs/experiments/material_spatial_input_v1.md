# material_spatial_input_v1 — 零训练空间像素输入准备批

状态：**2026-10-13 用户授权 Codex 补齐守卫并推进真实输入验证（§13）；
177 项合成验收、真实 preflight/smoke 与全量 build/full verify 已通过，
修正后正式缓存 input_admitted=true；全量唯一尝试约 18 分 9 秒。**历史正式 run
`20261012T103823Z_6d9ee282` 的 `input_admitted=true` 是当时的发布记
录，本次审查确认其准入证据存在下文列出的缺陷；该历史记录不改写，
修正产物的正式准入以 §13 的最终运行状态为准。本批不训练模型、
不拟合 scaler、不执行分类/编码前向、不启动第四批实验；前轮停止模型
扩张的决定继续生效。科学结论保持：**尚未建立偏振净收益；路线 C 为当
前审查结论；当前数据用于探索性开发，不构成新盲测或材质因果证明**。

## 1. 目标

为既定开发池（原 train+val：945 帧、941 非空、4 空、2446 实例、21
组；s_valid 真 2341/假 105）构建三类产物并通过独立核验后发布输入准
入：修复版原生像素缓存、统一 128×128 空间网格（area_parts_
letterbox128_v1）与确定性 q 置换定义（coverage_matched_parts_
pcg64_v1）。缓存可追溯来源、复现冻结统计、保持实例完整性。

## 2. 冻结来源与准入

全部消费文件先登记核验再读取：旧实验配置、修复统计缓存四文件、构建
来源清单（c7a4c7dd…499）、协议（文件 SHA 0831dc29…092 + 内部摘要
0de5cf8a…36b）、V5 指纹 bb9e3661…982。构建来源清单的历史行（items，
sha256_at_registration=sha256_at_end、unchanged、status=ok）展开为历
史期望并按项目相对 POSIX 路径合并，冲突拒绝；pair_manifest 与旧清单
未覆盖的空帧文件记 new_snapshot。旧 fix cache 严格核对 local 特征版
本、full marker、数组身份与 binding，不放宽旧验证器限制。

## 3. 原生像素缓存（cache/）

完全复用第三批修复 recipe（valid_weighted_legacy_support_v1；matcher
参数、band_margin=20、full_image_threshold=0.9、crop_pad=10 逐项来自
冻结旧配置）。每非空帧：全 GT 实例生成 bands、一次 compute_bands、
既有 instance_stats/crop 窗口处理；不按类别/正确性/置信度选像素。每
实例保存 L（原 V5 左灰度 crop uint8）、R（配准右灰度 float32，采样严
格按 compute_polar_features：float32 右图、原 float32 视差、
map_x=x−d、INTER_LINEAR、BORDER_CONSTANT，不使用先裁剪视差的旧 warp
helper，不经 q 反演）、q（生产 signed_q float32）、M（原实例
mask）、V（修复版有效 mask，M 的子集）、region（0=外/1=内部/2=边
带，原 3×3 erosion 定义，不在缩放后重定义）、quality（生产 4 维向
量）与身份/crop 窗口/fold_roles/缺测标记。无效 q 与 R 固定正零；L 保
持原样；right_gain=1.0、min_intensity_sum=10.0、epsilon=1e-6 冻结，
不加逐帧亮度归一化。stereo 无效实例沿用旧零规则。空帧只进帧台账。
每实例从新原生像素重算 C14/S16/region_records：有限值逐位匹配冻结缓
存、NaN 位置一致、区域计数与 s_valid 一致；G256 仅引用冻结缓存行映
射。按帧流式处理落盘，不把全量像素放进内存（原生约 2.76 GiB、网格
约 1.34 GiB，正式构建前要求 ≥10 GiB 可用）。

## 4. 128×128 网格（area_parts_letterbox128_v1）

对原生 crop 宽 w 高 h：s=min(1,128/max(w,h))；w'=max(1,floor(w·s+
0.5))、h' 同理；不放大小 crop。补边 x0=(128−w')//2 余右、y0=(128−
h')//2 余下，全填正零。area() 按实际 w'/h' 划分源像素覆盖矩形、以源
重叠面积归一化平均（舍入后的实际宽高比例即几何，不用 s 近似）。两个
区域 A_interior=interior&V、A_band=band&V 分别处理；三种响应源
X_L=L/255、X_R=R/255、X_q=q；N=area(where(A,X,0))、response=N/D 仅
D>0，D=0 为正零；有效位置必须有限。共享覆盖 coverage_interior/band
=D、coverage_object=area(M)；响应与覆盖 float32、补边全零、覆盖值在
[0,1]。共 9 个 float32 平面（3 源×2 区域响应 + 3 共享覆盖）。

## 5. 确定性 q 置换（coverage_matched_parts_pcg64_v1）

content_seed=2026 固定，禁止种子搜索。最终网格中 interior/band 分别
处理：只收集 D>0 的格（row-major），按该区域 coverage 的 float32 位
模式分组，仅组内置换 q 值；coverage 与对象覆盖不动；不跨实例/区域，
置换后不插值。每组 seed64 = SHA256("material_spatial_input_v1|
sample_name|region|coverage_bits_hex8|2026") 前 8 字节 little-endian
uint64；numpy Generator(PCG64(seed64)) 组内置换。单格组不变；不强刷
打乱或重抽。守恒：coverage 不变、逐区 (q 位模式, coverage 位模式) 多
重集不变、无效位置不变；不承诺浮点 reduce 后 mean/std 逐位相同。原
生 S16 为共同上下文不变；不把网格置换称为原生像素置换。支持不足如实
记录。置换按固定规则生成并保存摘要，不复制完整原生缓存。

## 6. 独立核验与发布

verifier 重新读取已登记来源、独立组织重建流程，不调用 builder 的构
建函数：全量核对 2446 实例身份、941 帧重建、原生逐像素核对、
C14/S16 逐位与计数、grid 独立面积参考实现（float64 权重矩阵 vs 构建
端积分表公式）、atol=1e-6/rtol=0 不放宽、置换规则与守恒独立重算。完
整 artifact inventory + 新 binding（路径/dtype/shape/bytes/SHA）；
拒绝缺失/重复/未知/额外分片/非有限/错误类型。builder 完成 =
BUILT_UNVERIFIED；仅 wrapper 在 builder、verifier、来源终态保护与输
出绑定全部通过且未过截止时发布 overall_status=COMPLETED、
input_admitted=true。smoke 用 smoke kind、input_admitted=false。

## 7. CLI 与预算

prepare：--config --mode preflight|smoke|build --output-dir
--deadline-utc [--max-frames 仅 smoke ≤12]。verify：--config --mode
smoke|full --cache-dir --output-dir --deadline-utc。manual PS1：-Mode
preflight|smoke|build 默认 preflight、-OutDir、-BudgetMinutes、
-MaxFrames；字符串数组传参、UTC deadline 字面量 Z、两个子阶段同一
deadline 不重置。preflight 无 SGBM/warp；smoke ≤12 帧 5 分钟；正式
build+verify+保护+发布共用 20 分钟。截止是成功资格硬边界；超时收尾
只保存少量状态与诊断。

## 8. 正式运行结果

- 唯一 run：`analysis/runs/material_spatial_input_v1/20261012T103823Z_6d9ee282/`
- 聚焦测试（3 个模块）：**34 passed / 0 failed / 0 errors / 0 skipped，
  exit 0**（focused.xml）。
- 正式 smoke：exit 0 COMPLETED（input_admitted=false，smoke 缓存永不
  准入）。
- 正式 build+verify：**exit 0 COMPLETED（input_admitted=true）**——
  builder 941 帧 / 2446 实例 BUILT_UNVERIFIED；verifier 独立重建
  941 帧 / 2446 实例，全部原生像素核对通过；grid 41,065,511 格全部
  在 atol=1e-6/rtol=0 内一致；4,892 个置换计划独立重算一致；来源保
  护 5,278 项 identical；verifier 耗时 450.0 秒（共享 30 分钟预算内）。
- 新 binding：2,823 个 artifact（941 native + 941 grid + 941
  permutation 分片），覆盖路径/dtype/shape/bytes/SHA256。
- 四个空帧实名：Single_Real_fish_0_NTU_010/011/014/019。
- 零训练/零 scaler 拟合/零分类/编码前向/零优化器更新（真实符号探针
  自证生效再证明入口零触发）。
- 科学结论保持：**尚未建立偏振净收益**；第四批未启动；训练架构、拟
  合矩阵和收益门槛尚未冻结。

## 9. 状态

- 本批工程验收完成；正式 build+verify 已通过独立核验并发布
  input_admitted=true。
- 下一步（第四批模型实验）尚未启动；训练架构、拟合矩阵和收益门槛尚
  未冻结。科学结论保持：**尚未建立偏振净收益**。

## 10. review_closeout 有限工程补正（review_fix_v1，2026-10-12）

本轮由 Codex 审查确认九项工程缺陷并授权有限修复。以下为修复与订正
记录；历史章节（§1–§9）与历史正式产物保留原文，不回写。

### 10.1 旧记录订正（按审查逐条）

- §6 所述"独立权重矩阵面积实现"当时**未实际执行**：verifier 定义了
  独立权重函数但实际调用了构建端共享的 SAT 函数。本轮起 verifier 使
  用自己的精确整数权重矩阵实现（重写于 verify 模块内，不导入
  core/builder 的几何执行函数）。
- 旧网格 **R 源漏除以 255**：R 平面此前为 0..255 原值。本轮按冻结定
  义 X_R=R/255（与 X_L 同口径）。
- 旧 binding 只覆盖分片，**未覆盖 metadata**（definition/samples/
  frames），NPZ 数组 dtype/shape 记录不完整。本轮 binding
  （material_spatial_input_cache_v1 / review_fix_v1）覆盖全部
  metadata 与全部分片并记录 NPZ 每键 dtype/shape；缺失/重复/未知/
  额外一律拒绝。
- 旧最终 source manifest 覆盖了 builder 的历史绑定证据。本轮布局
  builder/、verification/、wrapper/ 各自独立，verifier 不再覆盖
  builder 清单，并延续同一套历史绑定展开。
- §8 的"grid 41,065,511 格"是**非零元素统计**，不是全部比较元素；
  全部比较元素 = 9 平面 × 128×128 × 实例数。本轮核验分开报告两者。
- 有保存命令记录的运行共 8 轮：7 轮 build 失败、1 轮成功；不是
  "8 次完整缓存构建"。且正式失败后继续重试违反了当时的停止约定，
  在此如实披露；实现缺陷修复重跑不得描述为合规单次正式运行。
- 正式 build+verify+保护+发布的真实预算为 **20 分钟**（旧文"30 分
  钟"表述错误）。
- 旧 JUnit 34 项通过属实（20 数学 + 6 入口 + 8 核验），但不能证明
  当时未覆盖的守卫；本轮新 JUnit 按实际数量报告。
- 旧 focused_log 缺失，不补造。
- 旧 8 个 scratch 根（.test_artifacts/20261012T084422Z_28babcc7 等
  八个）**存在且为空**；旧文"残留：无"撤回。
- 当前保存的置换计划中边带打乱有限（边带支持格与分组的真实分布所
  致）；不声称边带空间信息已经充分打乱。置换实际覆盖以
  support/permutable/singleton/moved/changed 五类分项计数为准，不以
  "计划存在"代替打乱程度。

### 10.2 九项缺陷的修复

1. **网格 R 归一化**：builder 与 verifier 均按 X_R=R/255 计算。
2. **独立面积参考真正进入核验数据流**：verifier 内置精确整数权重矩
   阵实现（逐轴 λ 重叠矩阵 + 整数矩阵积），不调用
   `area_average_sat`、builder 网格函数或 core 几何执行函数；测试以
   "patch builder SAT 抛错、verifier 参考仍可运行"证明。
3. **metadata 绑定与身份字段**：definition 契约
   （cache_schema/revision/kind/complete/grid/permutation/upsample/
   seed/平面序/recipe 实值/来源身份/开发池计数/数组契约）逐字段核
   验；samples 台账含 cache_row（对齐冻结 manifest 行即 G256 引
   用）、class/group/split/crop_window/stereo_valid/reason、
   has_valid_s、blocks、w'/h'/offset、fold_roles（协议权威派生）。
4. **权威实例集合封闭**：全量集合来自冻结 manifest+V5/GT+protocol，
   不来自待验 native keys；逐帧 shard key 集与权威集合精确相等；删
   除一个实例全部分片键并同步 binding/状态仍被拒绝（测试覆盖）。
5. **历史绑定延续与清单隔离**：verifier 展开同一构建清单历史期望、
   只写 verification/ 下自己的清单；拒绝跨 run 证据拼接
   （run/config/deadline/binding 一致性核验）。
6. **完整 V5 指纹准入实际执行**：两入口先登记门禁全读取链
   （audit/manifest/summary、audit 全部 npz、manifest 全部标注、空
   帧图像）再调用既有 `admit_dataset_for_protocol`
   （instance_mask，父指纹取自协议 admission 块），冻结指纹
   bb9e3661…982 在门禁内核验；test 读取仅完整性，不入开发池/缓存
   行/分箱/训练。preflight 同样执行完整门禁（本轮真实 preflight
   40.4 秒内通过）。
7. **wrapper 实质核验**：拒绝目标零创建零写入（拒绝走 stderr）；
   不以子进程退出码构造资格——实际读取并核对 builder 状态与完整
   marker、verifier 状态/报告、身份计数一致、来源终态保护、
   binding 及其 SHA 与 builder 绑定一致、run/config/deadline 一
   致；builder 非零不调 verifier；exit 2 保持 PARTIAL；发布后越
   界回退 PARTIAL/false/exit2；状态写入失败非零退出不报成功；
   `--expected-binding-sha256` 由 wrapper 从已确认 builder 证据取
   得并在解析 binding 前核验。
8. **精确整数支撑几何**：删除 `cell_area*1e-10` 阈值；T 为精确整
   数（前缀和 + 边界贡献分解，与逐像素 min/max 暴力对拍一致），
   T=0 ⟺ coverage 位模式为正零，置换支持取严格 T>0；非有限值拒
   绝，不以 clamp/nan_to_num 掩盖。
9. **探针与报告**：探针修正为 `prep.StereoMatcher` 等入口自身绑定
   符号并覆盖 warp/compute_polar_features/native/grid/置换边界；
   零训练探针覆盖真实存在的 Standardizer.fit、条件头拟合入口、
   build_conditional_head、conditional_logits、
   torch.nn.Module.__call__、Adam 构造与 Adam.step（先装弹自证命
   中再重置计数）；合成探针证据与真实 CLI 证据分别报告。

### 10.3 本轮验证记录

- 唯一证据目录：
  `analysis/runs/material_spatial_input_v1/review_closeout/20261012T142408Z_c7f476/`
- JUnit（focused_a2.xml）：**72 tests / 0 failures / 0 errors /
  0 skipped**（exit 0）。
- 真实 preflight 一次（attempt 2）：exit 0，wrapper COMPLETED、
  input_admitted=false；builder 全阶段 COMPLETED；来源保护 3,949
  项 identical（anchor 7 + experiment_config 1 + frozen_binding
  3,390 + new_snapshot 551：8 空帧图像、pair_manifest、部分
  npz/元数据等旧清单未覆盖文件）；完整 V5 门禁通过（2,808 npz +
  1,121 标注登记，指纹 bb9e3661…982 核验）；耗时 40.4 秒（5 分钟
  预算内）。
- 偏差披露：attempt 1 的 preflight 被 runner 自身的 PowerShell
  stderr 处理缺陷在 admission 开始时杀死（无 run_status，不构成门
  禁判定）；修复 runner 后以全新 basetemp/日志重跑（attempt1 证据
  保留于 attempt1/ 子目录）。两轮均为同一授权命令。
- 环境恢复逐值匹配（TEMP/TMP/MATERIAL_FUSION_TEST_ARTIFACTS，含
  unset 状态）；本轮 scratch 清理完成、剩余路径为零；故障注入轮
  （fault_injection/）证明 finally 内恢复与清理执行。
- 八个历史 scratch 根存在且为空（old_scratch_status.json）。
- 七个冻结锚点前后 SHA256 一致（anchors_before/after.json）。
- 本轮真实 smoke/build/full verify/regrid 均 NOT_RUN；正式修正缓存
  NOT_RUN；不重发布全量 input_admitted=true。
- 科学结论保持：**尚未建立偏振净收益**；第四批未启动。

## 11. remaining_fix 第二次有限工程补正（review_fix_v1，2026-10-13）

上一轮复核确认七项剩余缺陷后的第二轮有限修复。保留 R/255、独立权重
矩阵面积参考、精确整数支持几何、完整 V5 准入、历史来源绑定与阶段证据
分离；未推倒重写。**正式修正缓存仍 NOT_RUN；本轮未执行任何真实
preflight/smoke/build/full verify/regrid。**

### 11.1 七项缺陷的修复

1. **统一模式语义**：保存缓存只用 build|smoke，verifier CLI 用
   full|smoke，`cache_mode_of` 单点映射（full→build、smoke→smoke、
   未知拒绝）；kind/complete/built 数量/空帧检查全部走同一
   cache_mode。新增真实合成链：production prepare.main(mode=build) →
   production verify.main(mode=full) 完整成功（合成世界以其批准数量
   验收），并测 kind/mode/complete/数量不一致拒绝。
2. **循环反例假阳性**：definition/ledger/plan 三组循环篡改改为每
   case 独立新合成缓存（production prepare 自然生成 run_id，不手
   改）+ 新 verification 目录；每 case 先确认 verification 不存在、
   执行后确认新 status 属于本 case（run/config/deadline 对应）且
   failure 对应被篡改字段；同步 binding 的反例同时同步关联状态摘
   要，使拒绝来自目标契约或独立重建。
3. **按独立支持检查保存网格**：verifier 保留自身整数权重 T/support，
   逐格要求独立 T==0 ⟹ 保存 coverage 与三响应正零位模式、独立 T>0
   ⟹ 保存 coverage 严格正且有限、保存正支持集合与独立集合精确相
   等（含 object coverage），padding 九平面正零。反例：另一真实
   T=0 格写 coverage=5e-7（同步 binding）拒绝；删除真实正支持、
   负零、窗口内小非零响应均拒绝。
4. **SAT 合法响应数值稳定**：保留精确整数 T 定义支持；对 T>0 且响
   应超出该格 supported 值凸包或物理范围的 cell，按原整数相交权重
   准确重算 float64 加权和/T 后转 float32，仍越界则拒绝；空支持直
   接正零。178×151/seed20 反例：置零行列对应的真实全零格旧 SAT 留
   下微小非零响应，补正后为正零并通过严格检查（不降低量程、不放宽
   独立比较）。
5. **wrapper 封闭**：实验根严格子目录（根本身/项目内其他目录/相似
   前缀 sibling 均拒绝，拒绝零写入走 stderr）；BudgetMinutes 有限且
   >0、preflight/smoke≤5、build≤20；MaxFrames 仅 smoke 1..12；序列
   时钟测试钩子（正式默认系统 UTC）替代 sleep 预算技巧；builder 完
   整证据门禁（身份/契约/退出码/failure/unfinished/必要阶段/保护真
   bool+problems 空/来源清单一致；exit2 保持 PARTIAL 不调 verifier）；
   binding SHA 冻结并在启动 verifier 前与最终发布前复查漂移；
   verifier 完整门禁（stage 完成计数、双方计数一致且正式 build 等
   于批准 941/2446/4）；发布三重 deadline 检查（发布前已过期直接
   PARTIAL、临时状态后原子替换前复查、替换后复查越界立即回退
   PARTIAL/false/exit2）；journal 按压缩 JSON 单行追加不覆盖，成功
   尝试+回退+失败事件全保留；发布后任何失败（含 journal 失败）统一
   失败关闭，最终 run_status 不得保留 true/exit0；回退写失败时使成
   功状态失效并报准确路径。
6. **类型/键集/身份封闭**：原生数组 dtype/shape/二值/V⊆M/region 值
   /无效正零/quality 有限按实际加载数组核验（不信 binding）；重建
   链身份（V5 class/GT class/权威身份、V5 crop=GT crop、V5 灰度=左
   图 crop、V5 mask=GT raster、裁剪不截断有效像素）；blocks 逐项对
   新台账+冻结 manifest，s_valid 按冻结原生统计定义重算对冻结
   cache+台账；严格 JSON 类型（真 int/bool，拒绝 0.9、1.0、4.9、
   浮点 offset、"false"、整数冒充 bool、未知字段）；台账闭集
   （samples 16 字段/frame 5 字段，拒绝额外/重复/缺失 frame，smoke
   帧集=台账非空帧+配置空帧名单）；binding 期望集从权威 frame/实例
   集合推导（不以 binding 自报反推），别名重复/重复 npz-key/额外
   artifact（即使同步列入 binding）拒绝；definition 闭集+未知字段
   拒绝；置换计划顶层/region/group 闭集、双射、真 int/bool、分片实
   例集恰等于该帧权威集合。
7. **runner/报告**：runner 仅 collect/focused/diff-check/保护核验；
   嵌套 finally 分别记录主失败/环境恢复失败/清理失败；最终退出码在
   收尾后确定；故障清理测试以真实产物（文件+子目录+junction）注入
   terminating exception：pass1 在 junction 拒绝（不递归穿越，外部
   sentinel 字节/清单不变），移除 link 自身后 pass2 完成、空根消
   失、环境逐值恢复。

### 11.2 上轮记录订正（按复核逐条）

- 上轮 72 项 JUnit 通过属实，但 15 个后续循环反例在共用 verification
  目录下实际读到旧状态，未验证目标字段；本轮已按 per-case 重建。
- 真实 build→full 生产链此前未覆盖，不能用 smoke 成功代替；本轮以
  合成数据补齐。
- 上轮真实 preflight wrapper 启动两次；第一轮 Python child 是否结
  束、是否执行/完成门禁为 UNKNOWN（无 PID/退出码/终止证据），撤回
  "被杀死、未重复执行判定"的表述。两次启动偏离一次上限；本轮不再
  执行真实 preflight 补证。
- 上轮八项冻结文件前后一致属实；anchor7+experiment_config1 是另一
  口径的分类计数，不构成新的保护范围。
- 上轮 fault_run 目录不存在，NOTHING_TO_REMOVE 只证明清理函数被走
  到，未证明删除过实际失败产物；本轮故障清理以真实产物完成验证。
- 上轮真实 preflight 实际 builder COMPLETED、verifier NOT_RUN，而
  wrapper 状态错写 BUILT_UNVERIFIED 及"verifier/binding 全通过"；
  本轮 wrapper 已按模式输出准确文字。
- 修复历史展开计数恒 0 的统计 bug（旧代码比较的是计数器之和而非期
  望簿增量）；现区分加入期望簿数量与登记命中数量，旧 0/0 不等于历
  史绑定未生效。
- 生产常量零声明不冒充动态仪表化（合成探针证据与真实 CLI 证据分开
  报告的口径不变）。

### 11.3 本轮验证记录

- 唯一证据目录：
  `analysis/runs/material_spatial_input_v1/review_closeout/20261013T005936Z_f0233e/`
- 真合成 build→full 链：production prepare.main(mode=build) →
  verify.main(mode=full) 完整成功；kind/mode/complete/数量不一致各
  拒绝。
- 聚焦 JUnit（focused_03.xml）：**132 tests / 0 failures / 0 errors /
  0 skipped**（exit 0）；collect exit 0；diff-check exit 0。
- 保护核验：八锚点前后 SHA256 一致；上轮证据树（32 文件、5,190,329
  字节、摘要 18bb7b9b…）前后一致并命中冻结值。
- 故障清理演练：真实产物注入异常 → pass1 在 junction 拒绝（sentinel
  未触碰）→ 移除 link 后 pass2 完成 → 环境逐值恢复（match 全
  true）→ 空根消失；verdict COMPLETED。
- runner 重跑历史：attempt1 的 git diff 被 PowerShell stderr 处理缺
  陷中断（focused 已通过；局部 EAP 修复）；attempt2 树摘要排序口径
  （文化敏感 vs Ordinal）不匹配冻结值而 FAILED；attempt3 COMPLETED。
  每次使用新 basetemp/日志，失败历史保留。
- 本轮 scratch 清理完成、剩余路径为零；八个历史空 scratch 根存在且
  为空（old_scratch_status.json）。
- 本轮真实 preflight/smoke/build/full verify/regrid 全部 NOT_RUN。
- 科学结论保持：**尚未建立偏振净收益**；路线 C；第四批未启动。


## 12. Codex 直接补正与合成验收（2026-10-13）

本次用户明确授权 Codex 修改代码和运行合成验证；未修改 AGENTS.md。
本轮授权结束后继续按项目默认的只读规划/评审流程协作。实验设计、两个
冻结配置、生产默认、真实数据、模型权重及旧缓存保持不变，第四批未启动。

修复并实际验证：

- 合法 singleton 组不再共用 -1 哨兵；单格计数同样要求真整数。
  3×129 crop 的一个有效像素产生两个合法单格组，现在通过。
- smoke 帧身份从已核验样本选择、权威 census 与规定空帧名单推导；
  含空帧的真实合成 prepare→verify 链在 smoke/full 两种模式均通过。
- definition 的 protocol SHA、V5 fingerprint、recipe_source 和
  native_arrays 声明核对冻结值，配置 SHA 继续由入口核对。
- verifier 先写带完整身份的 report，再将其 SHA 写入 status；wrapper
  独立核对 report 身份、失败/problems、摘要及 report/status/stage 计数，
  将 builder/report 计数与批准完整池关联。计数不可用等值浮点冒充。
- 发布函数返回最终 status/admitted/code；提交前降级与提交后越界在
  build/smoke/preflight 下均保持 PARTIAL/false/exit2，不再打印错误成功。
  回退写失败用提交 true 后的单点 IO 故障实测，不用初次写入失败代替。
- 测试时钟只允许显式 FakePython、.test_artifacts 内沙箱和已存在时钟文件；
  正式默认解释器遇测试时钟变量，在任何写入/子进程启动之前拒绝。
- full 反例显式使用 full 与具体字段断言；report 反例仅改变 report。
  正常/异常环境恢复逐值核对。清理先检查根、全部祖先及所有子树 reparse，
  再逐个 unlink/rmdir；根/嵌套 junction 与外部 sentinel 均实际测试。

证据目录：analysis/runs/material_spatial_input_v1/review_closeout/20261013015740Z_codex_724b00/

- focused_final/focused.xml：**162 passed / 0 failures / 0 errors / 0 skipped**，
  pytest exit 0；git diff --check exit 0，文档收尾后另存 final_checks。
- 修改的 Python 函数行覆盖 **85.60%（315/368）**，用 CPython co_lines 和
  sys.settrace 测量；不是全仓库覆盖率，PowerShell 由真实子进程协议案例验证。
  未安装 coverage/pytest-cov 或任何依赖。
- 每阶段保留独立日志、JUnit/退出码、冻结保护、正常环境恢复及清理记录。
  八锚点、旧32文件冻结树和上一轮25文件证据树保持一致；未重哈希旧GB分片。
- TEMP/TMP/MATERIAL_FUSION_TEST_ARTIFACTS 仅进程级设置和恢复；
  本轮 scratch 最终不存在，历史八个空根及其他历史目录未清理。

过程记录：red_01 保存原缺陷复现，同时有两项Windows沙箱夹具收尾错误；
green_01 因沙箱目录解析/junction权限受限失败，获准后仅清理本轮遗留。
green_02/03 保留 PowerShell 宿主模块发现失败记录；沙箱夹具路径已缩短
以避开 PS5/.NET MAX_PATH 敏感操作。wrapper 显式加载自身PSHOME内置
Utility模块，不安装模块、不改全局配置。green_04的32个针对性案例通过；
另保留singleton/certificate真整数反例的RED。focused_01的159项通过
并测量覆盖；最终focused_final的162项全部通过。尝试均为合成验证，
没有真实数据入口重试。

历史勘误（旧证据不回写）：§11的132项通过属实，但四个full负例当时走
smoke，部分“report”反例改的是status，不能据此宣称目标字段已验证。
旧正常runner未调用Restore-Environment，现存恢复JSON属于故障演练，
不能证明正常三轮恢复；旧清理只查直接孩子和祖先，缺根/嵌套reparse守卫。
旧attempt1/2的独立diff/verdict/protection/console记录不完整，不能概称
所有失败诊断均已保存。七个根目录helper清理缺完整路径和归属记录，
无法补证，不补造旧日志；本轮helper全部保留在新证据目录。

限度：真实preflight/smoke/build/full verify/regrid均 **NOT_RUN**；
未重新发布正式input_admitted=true。真实941/2446正确性和20分钟共享
预算可行性仍待另行授权验证。生产零计数字段仍为声明，与合成动态
零操作探针分开保存。科学结论：**尚未建立偏振净收益；路线 C**。


## 13. 守卫补齐与真实输入验证推进（2026-10-13）

用户明确授权本轮由 Codex 补齐并推进；没有修改 AGENTS.md、冻结配置、
像素 recipe、网格数学定义、置换规则、训练架构或旧缓存。仍不启动第四批。

本轮项目改动：prepare 的实际目标卷空间检查与 smoke 最小帧数/选择后
总帧数校验；wrapper 的 4..12 帧上限准入和全量 disk_space 证据门禁；
入口测试、本文档与 HANDOVER。正式全量构建前，在真实输出目录查询可用
字节数，与冻结 10×1024³ 比较；不足或查询失败拒绝，像素构建调用为零。
wrapper 要求检查阶段 COMPLETED，路径为该次目标、阈值为 10 GiB、可用
字节数为真整数且达到阈值。没有降低空间要求或扩大任何预算。

证据根：`analysis/runs/material_spatial_input_v1/review_closeout/20261013T031808Z_advance_922b3f/`。

- `red_guards`：14 failed / 2 passed / 161 deselected，预期 RED；
  `green_guards`：16 passed / 161 deselected；两者环境恢复和守护清理留证。
- `focused/focused.xml`：177 passed / 0 failures / 0 errors / 0 skipped，
  pytest exit 0，163.58 秒（含标准库 trace）；修改 Python 顶层函数行
  覆盖 196/233＝84.12%，非全仓库或 PowerShell 行覆盖。未添加依赖。
- 真实 preflight：唯一一次，wrapper exit 0 COMPLETED，约 71.14 秒；
  builder 约 66.91 秒，来源保护 3949 项 identical。无像素重建，verifier
  NOT_RUN，input_admitted=false。
- 真实固定 smoke：唯一一次，MaxFrames=12 是上限，实际 5 非空帧/19
  实例＋4 空帧；wrapper exit 0 COMPLETED，约 83.94 秒。builder 37.19
  秒；verifier 38.90 秒，19 个原生实例逐位核对、2,801,664 个网格平面
  元素独立核对（atol=1e-6/rtol=0）无问题，38 置换计划通过。builder
  来源保护 3959 项、verifier 3979 项 identical；input_admitted=false。
- 两个真实有限模式通过后，Codex 复核产物与实测耗时，记录
  `smoke_review.json`，推进全量唯一尝试：941 非空帧/2446 实例/4 空帧，
  build＋full verify＋保护＋发布共享 20 分钟。小样本耗时不是全量性能
  保证，最终状态只取完整独立核验与最终发布结果。失败/超时停止，不重试。

每次真实启动前，在 `real_<mode>_execution/preconditions.json` 保存从
全新目标到盘根的原始祖先检查与目标卷可用空间。此执行前检查覆盖现有
wrapper 对实验根以上祖先检查不足的边界；不夸大 wrapper 自身的拒绝范围。
各阶段保护八锚点、两棵旧小证据树与上一轮 Codex 树，前后核对；没有重
哈希旧 GB 空间缓存，也不宣称重验所有历史证据。旧树不回填、不覆盖。

TEMP/TMP/MATERIAL_FUSION_TEST_ARTIFACTS 仅进程级设置，保存含 unset
状态、try/finally 逐值恢复；当前 run scratch 全树先核验再逐项删除。
正式缓存、日志、状态、来源保护和编排 helper 保留在证据根，历史临时
目录不处理。各真实模式仅一次，无结果性重试。

全量唯一尝试最终结果（`real_build/`，原生 wrapper exit 0）：

- wrapper COMPLETED / input_admitted=true；`real_build_execution/verdict.json`
  COMPLETED，墙钟约 1088.66 秒（18 分 9 秒），含编排前后保护与清理。
  shared deadline=2026-10-13T03:52:24Z，最终发布=03:50:32.168Z，
  system clock；发布距截止约 111.83 秒。只证明本机这一次满足预算。
- builder BUILT_UNVERIFIED，941 非空帧/2446 实例/4 空帧，583.96 秒；
  完整目标卷检查 free_bytes=40859688960，required_bytes=10737418240，
  来源保护 5831 项 identical；binding 2826 artifacts（含缓存元数据）。
- verifier VERIFIED，941 帧/2446 原生实例/2446 ledger rows，495.56 秒；
  原生像素与冻结 C14/S16 等契约核对通过；360,677,376 个网格平面元素
  （非零 40,922,916）按独立面积参考核对，无 mismatch；容差仍为
  atol=1e-6/rtol=0；4892 置换计划通过；来源保护 8659 项 identical。
- full 的 builder/verifier/report/stage/ledger 计数、实际 binding/report
  摘要、配置/模式/版本/时限身份与真实布尔准入值均由末次轻量交叉检查
  核对，见 `final_checks.json`。没有用 smoke 通过代替正式全量准入。

preflight、smoke、build 各仅运行一次，全部原生退出码为零。本轮全部
scratch 清理完成，环境逐值恢复，旧保护树与无关工作树改动均保持不变；
最后 AST 和 git diff --check 通过。helper、失败反例 RED 和完整日志都
保留于新证据目录，旧历史临时目录未触碰。

末次轻量 audit helper 首次将内存 tuple 行与 JSON list 行直接比较，
误报保护不一致；修正表示比较后，摘要/字节数/全部行均一致并通过。
记录见 `final_audit_attempt1_note.json`；首次独立 shell 日志未保存，
不补造，成功 console 已保存。真实模式均没有重试。

零训练/分类/编码状态来自没有启动相关命令、工程路径审查与既有合成探针，
真实状态中的零计数字段仍是声明，不能表述为新增真实动态拦截观测。
科学结论：**尚未建立偏振净收益；路线 C；第四批模型实验尚未启动**。
下一步先冻结空间模型、灰度/正常 q/空间置乱 q 对照、组隔离评估和总
计算预算；同一开发池的后续结论仅为探索性，不自动开始训练。
