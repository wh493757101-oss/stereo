# material_existing_data_v1 — 现有数据条件下的检测诊断与偏振信息验证准备

状态：协议与全部非训练诊断已运行（2026-09-30）；P0/P1/P2 拟合**等待用户手动执行**。
本轮没有真实数据训练，没有新增独立泛化或偏振收益结论。

## 1. 目标与边界

在数据冻结的前提下（标签不修改、无法新增采集、训练由用户手动执行）回答：

- **A. 真实鱼失配主要来自置信度、输入尺度、分割轮廓，还是尚不能确定？**
  → 第二部分 A 的四条件回放给出了证据（见 §4）。
- **B. 如何在采集组隔离条件下比较灰度 / 灰度+测量质量 / 再加偏振数值？**
  → 第一部分的采集组隔离三折协议 + 第二部分 B 的 P0/P1/P2 特征槽位（见 §3、§5）。
- **C. 给出经过测试、身份与数据绑定完整的手动拟合命令。**
  → 见 §6 与本轮 `manual_commands.ps1`。

明确不声称的事：本轮不证明偏振收益、不证明独立泛化、不证明实时部署通过。
新分组验证仅表示**训练层面的组隔离**；原 val 已参与项目开发，任何折都不是新的盲测；
使用既有权重（旧 train/val 训练）的历史诊断不构成本协议下的 OOF 证据。

## 2. 数据与身份（全部 SHA256 复核通过）

| 资产 | 路径 | SHA256（前 16 位） |
|---|---|---|
| V5 主线数据 | `datasets/underwater_cls_fusion_v5_instance_mask` | 指纹 `bb9e3661fb1e71e9…`（2808 文件逐一复核） |
| Model A | `runs/train/yolo26_formal_001/model_a/weights/best.pt` | `ba049784c8b2a0d90…` |
| E1 候选（gray, s2026） | `runs/train/cmx_v2_mask_gray_s2026/material_fusion/best.pt` | `7a15d63a9458c77c…` |
| E4 候选（gray_polar, s2026） | `runs/train/cmx_v2_mask_gray_polar_s2026/material_fusion/best.pt` | `ba53c06b541c9931…` |
| 独立 Gray 回退 | `runs/train/yolo26_formal_001/gray_fusion/best.pt` | `0ca63f9e305eafdc…` |
| 官方 MiT-B0 | `weights/pretrained/mit_b0.pth` | `4af8348c2a802bf7…` |

开发池 = 原 train + val：**945 帧 / 2446 个 V5 样本 / 2446 个 GT 实例**，
类别计数 {metal_submarine: 562, plastic_submarine: 547, plastic_fish: 965, real_fish: 372}。
旧 test（180 帧 / 362 样本 / 4 组）**不并入开发池**，仅用于排除成员，未做任何评估、特征提取或数值探查。
4 个无目标帧保留并单列；GT↔NPZ 对应 0 缺口、0 异常；物理对象身份无来源，一律记 `unknown`，不从文件名猜测。

## 3. 第一部分：协议与采集组隔离三折（已运行）

入口：`scripts/prepare_existing_data_protocol.py`（复用既有 V5 准入门与指纹算法定义，未新增第二套指纹）。
产物目录：`analysis/runs/material_existing_data_v1/20260930T021500Z_a3f7c1/protocol/`
（protocol.json、folds.json、baseline_identities.json、group_audit.json、README.md）。

- 分组单元 = **完整采集组**（本数据集每组 45 帧连续序列，block_id 全为 0；序列绝不拆分）。
- 确定性算法：seed=2026 洗牌 + 全局类计数平衡贪心（逐组分配到使各折各类实例数最接近比例份额的折）；
  不使用任何模型分数。
- 外层 3 折，每折 outer_eval 的外层训练部分再按组切成 inner_train / inner_val（seed = 2026 + fold 序号，
  inner_val ≈ ceil(1/4) 的外层训练组）；inner_val 仅用于选 epoch，outer_eval 在 checkpoint 冻结后只评一次。
- fold 状态 **OK**：三折的 outer_eval / inner_train / inner_val 均覆盖全部四类。
  各折支持数（GT 实例 / 样本）：

| fold | outer_eval 组数 | outer_eval 支持（GT/样本） | inner_train 样本 | inner_val 样本 |
|---|---|---|---|---|
| fold0 | 6 | 840 GT / 840 样本 | 1116 | 490 |
| fold1 | 8 | 711 GT / 711 样本 | 1265 | 470 |
| fold2 | 7 | 895 GT / 895 样本 | 1191 | 360 |

（逐类 GT 计数见 folds.json 的 `outer_eval_class_counts` 等；每折 GT 数与样本数在本数据集恰好相等，
因为每个 GT 实例恰好对应一个 V5 样本。）

  （outer_eval/inner_train/inner_val 的逐类计数见 folds.json 的 `*_class_counts`。）

- 已知独立性限制：同一场景不同浊度的组可能拍到同一物理对象，但无任何来源记录对象身份，
  关系记为 `unconfirmed/unknown`，不按泄漏处理、也不声称无泄漏。

## 4. 第二部分 A：检测错误归因四条件回放（已运行，D0 完全复现）

入口：`scripts/diagnose_existing_detection.py`；产物：`…/20260930T021500Z_a3f7c1/detection/`。
匹配规则复用 `scripts/eval_material_pipeline` 的类无关一对一 mask IoU≥0.50（先最大匹配数再最大总 IoU），
**阈值从未降低**；跨条件 GT 身份 = (frame_id, object_index)；pred 编号不跨条件视为同一目标。

| 条件 | conf | imgsz | 预测 | 匹配 | 未匹配 GT | 未匹配预测 | real_fish 召回 |
|---|---|---|---|---|---|---|---|
| D0 | 0.25 | 640 | 380 | 346 | 22 | 34 | 3/18 (0.167) |
| D1 | 0.10 | 640 | 403 | 352 | 16 | 51 | 6/18 (0.333) |
| D2 | 0.05 | 640 | 423 | 355 | 13 | 68 | 8/18 (0.444) |
| D3 | 0.25 | 960 | 371 | 346 | 22 | 25 | 3/18 (0.167) |

D3 与 D0 的总匹配数虽然相同，但并非"没有作用"：D3 找回 7 个（3 plastic_submarine、
3 plastic_fish、1 real_fish）同时丢掉 7 个（6 plastic_fish、1 real_fish），
未匹配预测从 34 降到 25。只能说**总体匹配数与 real_fish 召回没有净提升**，
不能写"尺度无作用/与尺度无关"。

- **D0 与历史 4B 基线完全一致**（180 帧 / 368 GT / 380 预测 / 346 匹配 / 22 漏 / 34 未匹配预测，
  逐帧预测数也一致）；不一致时脚本会停止后续条件（本轮未触发）。
- 22 个 D0 未匹配 GT 的构成：15 real_fish、4 plastic_submarine、3 plastic_fish。规则化**迹象**归因
  （`待确认`，未经人工复核，不断言 NMS 根因）：
  - `confidence_related` ×9（5 real_fish + 3 plastic_submarine + 1 plastic_fish）：降低 conf 后出现可匹配预测；
  - `no_corresponding_prediction_sign` ×10（全部 real_fish）：所有条件下都无正重叠预测；
  - `input_scale_related` ×2（plastic_fish）：仅 imgsz=960 找回；
  - `duplicate_or_merge_sign` ×1（plastic_submarine）：最大重叠预测已属其他 GT。
- 对问题 A 的证据指向（限于本 val、本工作点）：**real_fish 失配与置信度强相关**（conf 0.25→0.05 召回
  0.167→0.444），提高输入尺度对 real_fish 召回没有净提升（找回 1 丢 1），另有 10 个 real_fish 在任何
  条件下都无正重叠预测（轮廓/外观层面的检出失败，需图册人工确认——图册已重绘于本轮
  `atlas_rebuild/atlas/`，预测轮廓已实际画出，仍属"迹象/待确认"，未经人工复核）。
- Model A 单阶段计时（预热 5 + 30 次，含同步；单阶段，**不是系统 FPS**）：
  D0 p50≈31.1ms / p95≈33.9ms；D3 p50≈28.4ms。
- 运行时身份：ultralytics 8.4.71、torch 2.12.1+cu130；conf/iou 按实际实现传递，不假定 NMS 参数必然生效。
- 参数/buffers 证据：每个条件首次前向后的快照为基线（首次加载/推理初始化的结构差异被显式排除并记录），
  其后全程逐一相等（不变即通过）。

## 5. 第二部分 B：偏振质量统计与冻结特征（已运行，只提取、不拟合）

入口：`scripts/prepare_material_probe.py`；产物：`…/20260930T021500Z_a3f7c1/features/`。

- 偏振质量统计（开发池，定义预先固定）：内部 = instance_mask 3×3 腐蚀；轮廓带 = mask − 内部；
  S 与 q 统计只在 `instance_mask ∩ polar_valid` 上计算。
  按类别的内部有效率：metal 0.821 / plastic_sub 0.864 / plastic_fish 0.744 / **real_fish 0.544（最低）**；
  轮廓带有效率 real_fish 0.441（同样最低）。亮度 vs 有效率相关性弱（Pearson 0.153 / Spearman 0.190）。
  曝光/增益/gamma/时间戳无来源，一律 `NOT_AVAILABLE`；同步状态 `unknown`（不变量 7）。
- 冻结灰度特征 **G**：官方 MiT-B0（SHA256 复核）最后一级特征（256 维 @7×7），实例 mask 加权池化
  （与 `mask_level_weights` 同语义）；`eval + inference_mode`，提取前后及每 200 样本一次参数/buffers
  快照对比，全程不变。mask 池化只约束池化算子本身，不声称编码器完全不受背景影响。
- 控制特征 **C**（6 维）：quality 前三项（valid_ratio / in_bounds_ratio / brightness_valid_ratio）
  + 实例内灰度均值/灰度标准差（ddof=0，不是方差） + S 区非空显式标记。**不含 quality[3]（mean_abs_q）**。
- 偏振统计 **S**（8 维，原始裁剪 NPZ 的 instance_mask ∩ polar_valid）：signed_q 均值/标准差(ddof=0)/
  10/25/50/75/90 分位（numpy.percentile 线性插值）+ abs_q 均值。无有效像素 → 显式缺失
  （has_valid_s=false，NPZ 中 NaN、JSON 中 null），**不解释为真实测量为零**。本轮 2341/2446 有有效 S。
- 缓存保存原始特征；**不利用全开发池拟合标准化/PCA**（标准化在各折 inner_train 内拟合）。
- 探查输入：P0 = G；P1 = G + C；P2 = G + C + S。泄漏规则有专门测试：
  扰动 mean_abs_q 或 signed_q/abs_q 数值不改变 C/P1；扰动 signed_q 改变 S/P2 的输入。

## 6. 第三部分：线性探针训练入口（已实现并测试；拟合由用户执行）

入口：`scripts/train_material_probe.py`。无 `--dry-run/--fit` 时不训练；两者互斥。
已验证的契约：编码器不加载、线性 4 类 softmax 头、三变体共享 [G,C,S] 槽位（不用的槽位置零，头维度/
初始化/预算一致）、标准化只在当折 inner_train 拟合（S 列仅用其中有有效 S 的样本；缺失 S 标准化后置零、
由 C 的标记区分；零方差/空列输出全零）、inner_val 只选 checkpoint、outer_eval 冻结后只评一次、
固定预算（seed 2026 / AdamW / lr 0.01 / wd 0.001 / ≤200 epoch / 全批次 / 无增强 / 无类别加权 / 无调参）、
保存模型+scaler+配置+历史+逐样本 logits、OOF/逐组/两组关键类对（plastic_fish vs real_fish、
metal vs plastic submarine）混淆、P2−P1 与 P2−P0 改对改错计数；不按 outer_eval 选模；输出目录拒绝覆盖；
BLOCKED 协议拒绝 fit；未 fit 时 training_status=NOT_RUN、classification_gain=NOT_MEASURED。

真实缓存的 dry-run 已由 Agent 执行（通过，预览见 `validation/probe_dry_run/report.json`）；
**`--fit` 命令写在 `manual_commands.ps1`，由用户执行**。

## 7. 本轮实际运行 / 仅合成测试 / 等待用户

| 项目 | 状态 |
|---|---|
| 协议生成（含 V5 全量准入） | 实际运行，fold OK |
| D0 复现核对 | 实际运行，完全一致 |
| D1/D2/D3 回放与归因 | 实际运行 |
| Model A 单阶段计时 | 实际运行（单阶段，非系统 FPS） |
| 偏振质量统计与冻结特征 | 实际运行（GPU，编码器状态验证不变） |
| 探针 dry-run | 实际运行 |
| 探针 fit（真实缓存） | **NOT_RUN — 等待用户** |
| 全量测试套件 | 实际运行：material 1063 passed / 1 failed / 4 skipped；全量 1835 passed / 1 failed / 6 skipped（唯一失败为既有 LICENSE CRLF/LF 换行差异，非本轮回归） |
| 真实数据训练 | **NOT_RUN（本轮全程未启动任何训练）** |

## 8. 2026-09-30 修复轮补充（准入/图册/入口控制流）

- **缓存绑定**：特征缓存目录新增 `cache_binding.json`（三个缓存文件的 SHA256+字节长度 +
  协议摘要 + V5 指纹 + 编码器身份 + 特征版本）。训练入口的准入链 = 绑定字节复核 →
  协议逐折权威样本索引（互斥/完整/组不交叉/每折恰一角色/outer_eval 恰一次）→ 清单逐行
  身份/角色核对（行序=冻结行序契约）→ 数组核验（键/形状/dtype/类别行/有限性/显式缺失规则）。
  任何错配在构建头、scaler.fit、写训练产物之前拒绝；`--dry-run` 与 `--fit` 共用同一准入，
  均写出 `admission.json`。
- **真实数据准入**：特征准备改用 `admit_mask_dataset` 完整门（2808 文件逐一复核 + mask 解码 +
  来源重哈希），不再以指纹身份函数冒充完整性验证。
- **缓存升级**：旧目录只读；`scripts/upgrade_probe_cache.py` 验证旧清单覆盖与摘要后按字节复制
  三个缓存文件到新目录 `upgrade/corrected_features/`，生成新绑定与 `upgrade_report.json`
  （字节一致性 original vs copied 全部 identical=True），并以修复后准入复验通过；
  真实 dry-run（新目录）通过。G/C/S 无任何重提取。
- **图册重绘**：`--rebuild-atlas` 从已保存证据重绘（不调用模型）：68 面板 = 22 未匹配 GT +
  34 未匹配预测 + 12 正常对照；GT 绿轮廓、预测红轮廓带逐预测置信度、匹配/最大重叠 IoU、
  采集组与"迹象/待确认"标注全部画出；三类各抽查一张确认预测确实绘制。
- **失败停止/设备/预算（原验收范围纠正）**：已验证 D0 不一致及诊断主体开始前超时会停止计时；
  计时设备经 `torch_device_name` 规范化（"0"→cuda:0 真同步；CPU 路径不触碰 CUDA）；
  原轮没有闭合计时阶段与提取阶段的入口超时状态/部分产物保存，不能概括为全部耗尽路径均 PARTIAL。
  上一轮历史计时是在 device="0" 未走显式同步分支下测得（生产解析含 CPU 回传，不能断言原时延偏低；
  数值未被改写为"已重新测得"）。

## 9. 下一轮唯一优先改动建议

依据检测归因证据（置信度相关为主、尺度无关）：下一轮若只做一件事，应先做**检测置信度/召回层面的
real_fish 处理**（在冻结匹配规则与 D0 工作点约束下复核 D1/D2 工作点的代价——未匹配预测从 34 升到 51/68，
需要与下游误检代价一起权衡），而不是先动分类或融合。本轮不实现该改动。

## 10. 2026-09-30 预算入口收尾（用户授权 Codex 实施）

证据：`analysis/runs/material_existing_data_v1/20260930_codex_budget_finish_b81f29/`。
只用合成 CPU 输入/假时钟验证；不重新提取真实特征、不真实计时、不训练业务模型。

- 检测计时前先检查同一个预算，耗尽时不构造计时模型；预热/测量调用前后均检查。
  计时中超时保留已完成条件与原始重复计时，顶层 summary 和 timing 状态统一为
  `PARTIAL_BUDGET_EXCEEDED`，返回 2；普通计时异常标 `FAILED`、返回 1。
  诊断子阶段可记录 `diagnosis_status=COMPLETED`，不等于整个任务完成。
- 特征提取在编码器构建前及批次边界检查；超时保存已完成 G 行和 sample_name，
  写入 `partial_gray_features.npz`（仅有完成行时）和 `partial_extraction.json`、`report.json`。
  返回 2，不发布完整 `feature_cache.npz` 或 `cache_binding.json`；训练准入拒绝该不完整目录。
  这不是续提取功能，不改变正式缓存或训练算法。
- 预算是安全边界检查，不能抢占已开始的单次调用；G 提取预算仍从提取阶段开始，
  不声称覆盖前置完整数据审计和质量统计耗时。
- 历史数字依据旧目录 `20260930T045500Z_c5d9e8/logs/`：`pytest_focused_02.log`
  为 105 passed，`pytest_material_02.log` 为 1114 passed/1 failed/4 skipped，
  `pytest_all_03.log` 为 1886 passed/1 failed/6 skipped。不回写历史日志。

在 VS Code PowerShell 终端，先激活 stereo 后用 `& .\manual_commands.ps1` 做准入检查。
这会只读输入并写新报告，不是完全无写入操作。复核通过后由用户显式运行
`& .\manual_commands.ps1 -Mode fit`。禁止 `python .\manual_commands.ps1`。
失败/中断可能保留不完整输出目录，必须使用新目录重跑，不能冒充成功产物。

本轮实测：`red_01.log` 7 failed/3 passed（真实入口反例）；`green_01.log` 10 passed；
增加完整条件保留、关闭计时与训练入口拒绝回归后，`focused_02.log` 六文件 167 passed。
`all_01.log` 全量 1899 passed/1 failed/6 skipped，exit 1：唯一失败仍为既有 LICENSE
CRLF/LF 摘要差异，未修改许可证、期望值或跳过该测试。compileall、diff 检查和 PowerShell
语法检查通过。覆盖率 NOT_MEASURED。结论为本轮修复通过，不等于全量全绿或偏振收益证明。
