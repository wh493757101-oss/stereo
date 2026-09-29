# 项目交接与验收记录

更新时间：2026-09-14

这份文档记录当前已经落盘并验证的事实、尚不能成立的结论，以及下一阶段的执行顺序。数量与哈希以 `analysis/project/project_inventory_v2.json` 为机器可读来源，运行参数以 `configs/default.yaml` 为准，使用方法以 `README.md` 为准。人工修正版重训后的最新审查见 `analysis/runs/run_20260914_manual_labels/training_review.json`。

## 一句话状态

数据、正式训练、独立测试集评测和 GUI 双阶段推理已经打通；Model A 分割与 Model B 分类可用，但偏振增益没有通过消融门槛，完整流水线只有约 6 FPS，真实距离精度和双相机长时间稳定性仍缺少外部真值验收。

## 当前系统

```text
同步左右帧
  -> 立体校正（回放已校正图时跳过）
  -> 左灰度三通道 Model A（二值实例分割）
  -> 实例 bbox 合并水平条带，每帧一次条带 SGBM（接近全图时回退单次全图）
  -> 每实例稳健中位视差、有效率和深度
  -> 逐像素可靠视差的偏振差分（无效/越界像素偏振为 0，带 valid_mask）
  -> 每实例裁剪 Model B 批量分类（默认 gray，可切换 polar）
  -> GUI 显示 mask、类别、深度、同步状态和偏振图
  -> process_frame_detailed 输出各阶段耗时和每实例偏振质量
```

必须保持的约定：

1. 标定文件中的 MATLAB 旋转矩阵进入 OpenCV 前使用 `R_cv = R.T`。
2. 左相机为 0 度偏振，右相机为 90 度偏振；偏振特征为 `|L-warp(R)|/(L+warp(R))`。
3. Model A 是单类 `target` 实例分割；四类身份由 Model B 输出。
4. 四类 id 固定为 `metal_submarine=0`、`plastic_submarine=1`、`plastic_fish=2`、`real_fish=3`。
5. Model B 的训练与推理裁剪 padding 均为 10 px。
6. `polar` 模式通道固定为 `[gray, polar, gray]`；`gray` 模式固定为 `[gray, gray, gray]`。
7. 数据按完整采集组隔离，不能把同一场景/浊度连续序列拆入多个 split。
8. 没有可测量时间戳时，同步状态必须报告为 unknown，不能写成 0 ms。

## 已完成

### 标定与数据

- 修复 MATLAB/OpenCV 旋转约定，生成 `datasets/Rectified_v2` 共 1125 对图像。
- 棋盘格校正 QA 中位垂直误差约 `0.0849 px`；场景匹配 QA 中位数约 `0.6538 px`。
- 将 1121 份旧 LabelMe 标注迁移到新校正坐标系，并补齐 4 个负样本，共 1125 份 JSON。
- 删除迁移标注中的 base64 `imageData`，降低体积并避免图像内容与路径版本冲突。
- 在生成数据中抑制 4 个完全重复多边形，四类分割数据保留 2808 个实例。
- 生成四类 segmentation、二值 Model A、配对 gray/polar Model B 数据集。
- 固定 group split：train/val/test 为 `765/180/180` 张，对应 `17/4/4` 个采集组。

### 正式训练

| 模型 | 位置 | 训练状态 |
|---|---|---|
| 四类 segmentation baseline | `runs/train/run_20260913_initial/model_baseline/weights/best.pt` | 62 epoch early stop |
| Model A 二值实例分割 | `runs/train/run_20260913_initial/model_a/weights/best.pt` | 100 epoch 完成 |
| Model B gray 分类 | `runs/train/run_20260913_initial/model_b-gray/weights/best.pt` | 45 epoch early stop |
| Model B polar 分类 | `runs/train/run_20260913_initial/model_b-polar/weights/best.pt` | 28 epoch early stop |

环境已经固定为 Python 3.11.15、PyTorch 2.12.1+cu130、torchvision 0.27.1+cu130、Ultralytics 8.4.71；RTX 4060 可用，`pip check` 通过。核心依赖和导出依赖已拆分，完整快照位于 `requirements-lock-cu130.txt`。

### 正式测试集结果

分割测试使用固定 test split：180 张图、362 个实例。

| 模型 | box mAP50 | box mAP50-95 | mask mAP50 | mask mAP50-95 |
|---|---:|---:|---:|---:|
| 四类 baseline | 0.89562 | 0.77440 | 0.89619 | 0.73232 |
| Model A（二值） | 0.90995 | 0.81432 | 0.91080 | 0.76861 |

四类 baseline 的 mask mAP50-95：

| 类别 | mAP50 | mAP50-95 |
|---|---:|---:|
| `metal_submarine` | 0.97500 | 0.86539 |
| `plastic_submarine` | 0.97369 | 0.85720 |
| `plastic_fish` | 0.99437 | 0.75459 |
| `real_fish` | 0.64169 | 0.45208 |

`real_fish` 是当前最明显的数据/泛化短板。完整报告位于 `analysis/runs/run_20260913_initial/segmentation_test.json`。

### Model B 消融结论

| 输入 | accuracy | macro-F1 |
|---|---:|---:|
| gray | 0.961326 | 0.961800 |
| polar | 0.961326 | 0.961985 |

macro-F1 差值仅 `+0.000185`，按采集组 bootstrap 的 95% CI 为 `[-0.002919, 0.011499]`，未达到预设 `+0.03` 增益且下界不大于 0。因此：

- 不能声称现有数据证明偏振提高分类性能。
- `configs/default.yaml` 正确默认到 `model_b-gray` 和 `input_mode: gray`。
- polar 数据、权重和模式保留为实验资产，不删除。

完整报告位于 `analysis/runs/run_20260913_initial/model_b_ablation_test.json`。

### 运行时与性能

- GUI 和推理引擎统一从当前配置加载正式权重。
- Model A 每帧一次，SGBM 每帧一次；所有实例复用稠密视差。
- 多实例偏振计算已从每实例一次合并为每帧一次，相关回归测试已加入。
- 同一真实校正图、2 个有效实例、RTX 4060、30 次重复下，优化前 `184.59 ms/帧`，优化后 `165.96 ms/帧`，约 `6.0 FPS`。
- 组件剖析中 SGBM 约 `83.2 ms/帧`，Model A 约 `24.7 ms/帧`；二者是下一阶段主要性能目标。

`analysis/runs/run_20260913_initial/pipeline_speed_test.json` 是当前正式性能记录。单模型随机输入速度只能用于定位开销，不能代替完整流水线指标。

### 工程验证

- 新增稳定 JSON 输出的分割评测和完整管线 benchmark。
- 修复脚本从项目外工作目录启动时的导入路径问题。
- 修复逐类 AP 对 `ap_class_index` 的索引语义，避免测试集中缺类时错配类别。
- 校正与数据准备默认路径已切换到 v2。
- 标注转 mask 工具已迁入 `scripts/labelme_json_to_mask.py`，并改为显式输入/输出参数。
- LabelMe 迁移结果保留 `imageData: null`，并使用可解析到 `Rectified_v2` 的跨目录相对 `imagePath`；1125 份 JSON 均已验证可定位对应图像。
- 2026-09-14 人工修正了 83 份 JSON 中各 1 个轮廓：491 个已有顶点移动、3 个轮廓的顶点数变化，顶点总数净增 4；类别和目标数量均未变化。
- 人工修正版已传导到四类 segmentation、二值 Model A、gray/polar Model B 数据集；split 与 pair manifest 哈希保持不变。
- 正式权重未重新训练；在修正后的 test 数据上复评，分割与 gray/polar 消融报告和修正前逐字节一致。
- 完整严格测试：`379 passed, 1 skipped`，最近一次执行时间 34.64 秒。

### 2026-09-14 推理与数据链路修正（未切换正式权重）

- 偏振差分新增结构化结果 `PolarFeatureResult`（signed_q/abs_q/valid_mask/in_bounds_mask 等）：右图越界、非正视差、左右一致性失败和暗像素一律判无效并置零，修复了越界区域伪 `polar=1` 饱和问题；旧 `compute_polar_feature` 保持原行为。
- 新增目标水平条带匹配：`build_horizontal_bands`/`merge_horizontal_bands`/`StereoMatcher.compute_bands`，仅对合并后互不重叠的条带运行 SGBM，条带为空不匹配，覆盖接近全图时回退单次全图。
- 推理引擎改为 条带匹配 -> 中位视差深度 + 逐像素偏振 -> Model B 批量分类（`predict_batch`，无则回退）；`DetailedInferenceResult` 新增阶段耗时与每实例偏振质量。Gray 模式行为与旧三元组接口不变。
- 生成 `datasets/underwater_cls_fusion_v3`（2808 个数值 npz + manifest + 审计，见 `analysis/data/fusion_v3_audit.json`），偏振通道来自逐像素可靠视差。
- 开发训练基座切换为 YOLO26nano（`yolo26n-seg.pt`/`yolo26n-cls.pt`，本地缺失、需网络下载时显式报告）；本地 `yolo26n.pt` 为检测模型不得用作 Model A/B 基座；YOLOv8 资产保留为历史对照。
- 新增 Polar Fusion 模型骨架与训练/评测入口（`models/polar_fusion.py`、`scripts/train_polar_fusion.py`、`scripts/eval_polar_fusion.py`），仅支持 `--dry-run` 验证，正式训练尚未启动。

### 2026-09-20 审查修复（Polar Fusion 训练准入加固，未启动正式训练）

- 审计绑定当前数据：`dataset_audit.json` 记录 `dataset_manifest.csv`、`dataset_summary.json` 与全部 2808 个 npz 的 SHA256；训练门（`verify_dataset_integrity`）逐项复验 manifest/summary 摘要、manifest 路径去重、引用集合与磁盘集合一致，以及每个 npz 内容摘要；旧格式报告（缺少绑定字段）一律拒绝并提示 `--audit-only` 重新审计，不静默补齐。指纹改为由审计记录的摘要派生（manifest + summary + 全部 npz），仅在验证通过后写入。
- 统一架构准入：新建、`--init-from`、同 run-id 自动恢复（joint 读 freeze/best.pt）都从实际加载/重建的 Gray 主干识别架构；非 YOLO26 必须显式 `--legacy-gray-weights`；无法识别或记录与实际不符时直接拒绝；检查先于训练输出目录创建与任何优化器更新。
- quality 语义与包含关系：`sample.valid` 无有效像素时仅要求 `valid_ratio=0`、`mean_abs_q=0`，`in_bounds_ratio`/`brightness_valid_ratio` 按各自定义校验、不强制全零（暗图 `[0,1,0,0]` 合法）；有有效像素时 `valid_ratio` 必须为正，且按存储的 float32 值满足 `valid_ratio <= in_bounds_ratio`、`valid_ratio <= brightness_valid_ratio`（有效像素必然同时在界内且足够亮，相等合法）。实例 mask 不在 NPZ 中，三个 ratio 无法从裁剪 NPZ 精确重算，审计只校验上述范围与包含关系；stereo 整体失败仍写全零 quality。
- `mean_abs_q` 实质校验：离线 valid 限定在实例 mask 内、裁剪窗完整包含 mask（生成侧显式拒绝截断配置），审计按 `rtol=0` 重算裁剪窗有效像素均值并与 NPZ（atol 1e-6）与 manifest（atol 1.5e-6，六位小数）比较，取代原 `mean<=max` 弱校验。
- 恢复训练 imgsz 统一：新建未指定为 224，恢复未指定继承 checkpoint.imgsz，显式不同尺寸提前拒绝；DataLoader、前向检查、checkpoint metadata 与 train_config.json 使用同一解析值。
- V3/V4 已按新规则显式重新审计（各 2808 样本，`audit_passed=True`）；npz/manifest/summary 与旧摘要逐一比对未变，仅 `dataset_audit.json` 与 `analysis/data/*_audit.json` 更新。

### 2026-09-20 Gray-only 训练入口与真实链路验证（未启动正式训练）

- 新增 `scripts/train_gray_fusion.py`：与 Fusion 公平对照的 YOLO26 Gray-only 入口——相同 V4 数据/split/类别顺序/直接缩放预处理/`imgsz=224`、无数据增强、全模型 CE 训练、val Macro-F1 优先选模（accuracy 次优，test 不参与）；输出 `runs/train/<run-id>/gray_fusion/{best.pt,last.pt,train_config.json,metrics.json}`（含数据指纹、类别顺序、基座来源与 SHA256、`limit_batches`/`smoke` 标记）。checkpoint 为标准 Ultralytics 格式（`{"model": ...}`，4 类头、数据集类别顺序），可被现有 `prepare_gray_backbone(..., gray_weights=...)` 直接加载；检测模型 `yolo26n.pt` 与 YOLOv8 权重在入口处被显式拒绝。
- 新增 `scripts/verify_yolo26_fusion.py`：真实 GPU 有界冒烟（2 epoch × ≤3 train/val batch/阶段），覆盖 Gray 训练 → 保存重载 → Fusion freeze → 保存重载（生产重建路径）→ 权重接续；报告 `verify_report.json` 与各阶段 run 目录保留在 `runs/train/<run-id>*` 供审查。截断指标仅证明链路可执行，不作为模型合格依据。
- 官方基座 `yolo26n-cls.pt` 已从 ultralytics 官方发布下载（`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt`，5,786,434 bytes，SHA256 `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`），`task=classify`、架构 `yolo26n-cls`。
- 冒烟结果（run `verify_yolo26_gray_fusion_20260920_111304`）：保存重载 logits 最大误差 Gray/Fusion 均为 0；freeze 阶段 236 个 gray 张量逐一完全不变；`valid_ratio=0` 与显式 `polar_invalid` 时 gate 精确为 0 且 `final_logits == gray_logits`；权重接续在首个优化器更新前 249 个张量与 checkpoint 逐一相等，`--imgsz` 继承 224、冲突尺寸提前拒绝且未创建目录。
  **更正（2026-09-21）**：该报告中 "Gray 共享参数 117 个实际更新（梯度范数 7.36）" 与 "delta 从零初始化更新到 absmax 0.118" 为误判——官方 YOLO26 分类基座加载时全部 119 个参数张量 `requires_grad=False`，Gray 入口当时未解冻主干，117 个变化项实为 BatchNorm buffers（39×running_mean + 39×running_var + 39×num_batches_tracked），实际只有替换后的分类头被训练；delta 的卷积层本身为非零初始化，0.118 反映初始化值而非更新。该报告不能证明 Gray 全模型训练成功，仅证明链路可执行，也不再作为 Gray 主干训练的验收依据；有效验证见 2026-09-21 小节。
- `--init-from` 语义保持为权重接续：重建优化器，不恢复 optimizer/RNG 状态；正式 Gray 训练尚未启动；`configs/default.yaml`、正式权重与全部数据集资产未变（V4 指纹 `8db8998d…9bd334` 复核一致）。

### 2026-09-21 Gray 全模型解冻与验证器加固（未启动正式训练）

- `scripts/train_gray_fusion.py`：构建 Gray 模型后、创建优化器前显式解冻全部参数（`requires_grad_(True)`，不重新初始化权重），并断言可训练参数数 == 总参数数、优化器精确覆盖全部模型参数；`train_config.json` 新增 `parameters_total`/`parameters_trainable`。
- `scripts/verify_yolo26_fusion.py`：验证器改为"快照对比 + 结构判定"——参数与 buffers 分开统计（`named_parameters`/`named_buffers`，BN 统计量变化不计入参数更新）；以实际训练路径中"首次优化器更新前"的快照为基线（不再与官方基座比较，避免初始化/精度/头替换造成的伪差异）；必须存在非分类头参数更新且覆盖早期特征层（`model.0.conv.weight`），并以训练路径中的梯度 hook 证明该层梯度有限非零；分类头更新或 BN buffer 变化一律不能通过。delta 更新改为训练前后参数逐一对比（卷积层非零初始化不算证据），零初始化末层变化单独报告；新增 run-id 校验（复用 `validate_run_id`，非法值在构造路径、训练调用与报告写入之前拒绝并返回非零，另有派生目录不逃逸 `runs/train` 的防御检查）。
- 新增 `tests/test_verify_yolo26_fusion.py`（19 项）：负向用例证明"仅头更新 + BN 变化"、"仅 buffer 变化"、"早期层无梯度"、"梯度非有限"、"delta 全部未更新"均被拒绝；端到端负向用例（模拟未解冻回归）使 phase A 失败；非法 run-id（`..`、`../escape`、反斜杠路径、Windows 绝对路径、空串、含空格等）在任何输出创建前拒绝；正向用例（真实主干更新、冻结检查、delta 更新）通过。
- 新冒烟（run `verify_yolo26_gray_unfreeze_20260921_002910`，RTX 4060 Laptop，batch=8，`-W error`）：Gray 参数 119/119 全部解冻；主干 114/114 个参数实际更新（含 `model.0.conv.weight` 等早期层）；训练路径观测 119 个张量梯度全部有限，早期层梯度 absmax 0.602、114 个主干参数梯度非零；BN buffers 117 个变化单独统计。freeze：gray 119 参数 + 117 buffer 完全不变；delta 8/8 参数更新（零初始化末层确认从 0 变化，训练后 delta 输出 absmax 0.012）；gate 梯度范数 2.0e-3；两种无效偏振精确回退；Fusion 保存重载误差 0。接续：更新前 131 参数 + 118 buffer 与 checkpoint 逐一相等，imgsz 继承 224，冲突尺寸提前拒绝。峰值显存：Gray 约 215 MB（解冻后梯度与优化器状态增加，此前误报约 105 MB），freeze/接续约 110 MB；阶段耗时 31.8 s / 2.5 s / 1.7 s（Gray 含数据集摘要核验；仅冒烟描述，不外推部署 FPS）。
- 旧 smoke（`verify_yolo26_gray_fusion_20260920_111047/111207/111304*`）文件保留供追溯，未删除、未改写。
- 正式 Gray 训练尚未启动；`configs/default.yaml`、正式权重与全部数据资产未变（V4 指纹 `8db8998d…9bd334` 复核一致）。

### 2026-09-21 三阶段总控与旧四阶段归档（未启动正式训练）

- 新主线 `scripts/train_pipeline.py`：固定串行 `model_a` → `gray_fusion` → `fusion_freeze`，每阶段独立 Python 子进程、同一 run id，前一阶段完成并通过产物检查后才启动下一阶段；启动前统一预检（选中阶段输出冲突、基座存在且任务/架构正确、分割数据 YAML 与引用、V4 严格数据准入与指纹、GPU 可用性、参数合法、Gray/Fusion imgsz 一致；仅选择 Fusion 时本轮 Gray 产物必须存在、兼容且非 smoke），每阶段后核对产物（best/last 可加载、真实架构/类别顺序/输入尺寸/数据来源、Macro-F1 优先选模证据、Model A 二值 mask 推理与有限前向），运行记录写入 `runs/train/<run-id>/pipeline_report.json`（命令、起止时间、退出码、产物、来源、失败原因；计划参数与实际状态、smoke 标记分离）；失败保留日志与已完成产物，不清理、不覆盖。
- 新增 `scripts/train_model_a.py`（原 stage=a 的独立入口：二值单类 `target`、YOLO26nano 分割基座、拒绝检测模型与 YOLOv8、`--fraction` 冒烟截断并记录实际数据量）与 `scripts/training_common.py`（设备解析、run-id 校验、异常类型、项目锚定与 `SEG_BASE`/`CLS_BASE` 的唯一来源，活动代码与归档代码共用，不重复实现）。
- 旧四阶段实现归档到 `scripts/legacy_four_stage/`（`train_models.py`/`train_all_models.py`/`train_seg.py` + README 含新旧路径映射与显式运行方式），修正 PROJECT_ROOT、内部导入与子进程脚本路径后可显式运行；原路径改为退役提示入口（非零退出、不训练、不静默转发）；旧回归测试调整为导入归档模块并保留覆盖，新增测试验证退役入口仅提示、活动代码不依赖退役入口。
- 新基座 `yolo26n-seg.pt` 从 ultralytics 官方发布下载（`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt`，6,719,965 bytes，SHA256 `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`，task=segment、架构 yolo26n-seg）。
- 真实总控冒烟（run `verify_three_stage_20260921_101404`，RTX 4060 Laptop，`--smoke`）：三阶段串行全部 PASSED——Model A 1 epoch × 1% 数据（实际 8 张训练图，imgsz 640/batch 2），产物检查含二值 mask 推理（detections=300、masks 有限）与分割前向输出结构/有限性；Gray 2 epoch × ≤3 batch（imgsz 224/batch 8），产物检查含 V3 类别顺序、`best_epoch=2` 与 (macro_f1, accuracy) argmax 一致、V4 指纹；Fusion freeze 2 epoch × ≤3 batch，产物检查含 `gray_weights` 指向本轮 gray `best.pt`、重载 logits (2,4) 有限。前两次尝试因检查器自身的设备/输出结构 bug 失败（阶段正确停止、产物保留），修复后以新 run-id 重跑。
- 正式三阶段训练尚未启动；`configs/default.yaml`、正式权重、数据集与既有 runs/analysis 资产未变（V4 指纹复核一致）。

### 2026-09-21 三阶段总控审查修复：准入守卫、指纹绑定、smoke 分离与报告历史（未启动正式训练）

- Model A 单类语义强制：新增共享只读校验 `train_model_a._validate_binary_dataset`（总控预检与独立入口共用，无旁路）——names 必须精确表达唯一类别 `0: target`（兼容 list/dict 表示，非 0 键被拒），提供 `nc` 时必须为 1，按实际 train/val 划分解析图片并逐一校验分割标签（类编号必须为整数 0、polygon 结构 ≥3 点、坐标有限；普通检测框标签被拒），无标签/空标签属于合法背景图但每个 split 必须至少有一个有效 polygon；不生成缓存、不重写标签；所有可预检错误在启动训练与创建输出目录前拒绝。Model A 产物验收同时检查 best 与 last 的 YOLO26 segment、单类 `target` 与二值 mask 推理。
- 指纹强绑定：总控以 `validate_dataset` 的严格校验指纹为准（空值直接拒绝）；Gray 产物验收必须核对配置 `dataset_fingerprint` 与当前已验证 V4 一致（Gray→Fusion 连续与仅 Fusion 两条路径都检查），Fusion 的 train_config 与 checkpoint 元数据指纹也必须一致；拒绝时给出字段名与双方取值；原有类别顺序/imgsz/来源路径检查保留，不以路径相同代替指纹一致。
- Fusion smoke 与正式严格分离：`train_polar_fusion` 在 train_config 与 checkpoint 元数据（`FusionCheckpointMetadata.limit_batches/smoke`，缺失读为 None）记录实际生效的 limit_batches 与 smoke 标记；正式验收要求字段存在且类型精确（smoke 必须为布尔 false、limit_batches 必须为整数 0），缺失/null/字符串/布尔型一律拒绝；配置与 checkpoint 互相矛盾时拒绝；旧 checkpoint 缺字段仍可加载推理但不再默认获得正式验收资格。Gray 与 Model A 同步采用严格字段检查。（补：checkpoint 元数据在两种模式下均独立执行严格类型/模式校验——`3.0 == 3`、`False == 0` 的数值相等不能代替类型检查；与配置的一致性检查保留，合法类型但数值不一致同样拒绝。）
- 报告历史保留：`pipeline_report.json` 升级为版本化累计结构（report_version=2，`attempts` 逐次记录命令/起止时间/退出码/产物/来源/失败或中断状态；`stage_status`/`pipeline_complete`/`remaining_stages` 汇总），同一 run-id 分阶段继续执行不丢历史；只跑部分阶段不会报告完整流水线完成；v1 旧报告无损迁移为 legacy attempt，损坏或 run-id 不一致的报告被拒绝且不覆盖；子进程启动前持久化本阶段命令与 RUNNING 状态；子进程非零退出、启动失败（LAUNCH_FAILED）、产物验收失败、可捕获中断（INTERRUPTED，退出码 130）均记录并停止后续阶段；原子写入（断电/强杀可能丢失最后一次状态更新，如实说明）。`--preflight` 与 `--preflight-only` 等价且完全只读。
- 真实有界 smoke（run `verify_three_stage_guards_20260921_112537`，RTX 4060 Laptop）：预检（`--smoke --preflight`）exit 0 且零写入；smoke 三阶段全部 PASSED（Model A 1 epoch × 1% 数据、Gray/Fusion 2 epoch × ≤3 batch，命令参数已核对）；报告 `report_version=2`、`pipeline_complete=True`；新 Fusion smoke 产物只读验证：smoke 验收通过（smoke=true/limit_batches=3），`require_formal=True` 按"smoke field must be exactly false"拒绝；同 run-id 继续执行对已存在阶段目录 exit 2 拒绝、不重跑。
- 正式三阶段训练尚未启动；数据、审计 JSON、正式权重、`configs/default.yaml` 与既有输出未修改（V4 指纹复核一致）。

### 2026-09-21 yolo26_formal_001 Fusion 退化诊断（只读，未训练）

- 新增 `scripts/diagnose_polar_fusion.py`：单 checkpoint 只读诊断（Gray-only / 真实偏振 Fusion / 两种强制无效回退精确性 / 全量 test 索引偏振打乱 5 seeds），评估前强制校验数据指纹（配置与 checkpoint）、类别顺序、imgsz、freeze/正式标记、Gray 来源，以及 Fusion 内 Gray 分支与独立 Gray 权重逐参数逐 buffer 一致；输出 `summary.json`、`predictions.csv`、`changed_predictions.csv`、`summary.md` 与变化样本拼图；已有输出目录拒绝；不训练、不做 backward、不按 test 选型。对应测试 `tests/test_diagnose_polar_fusion.py`（24 项）。
- 诊断 run：`analysis/runs/yolo26_formal_001/diagnostics_20260921_195201`。完全复现正式评测：Gray acc=0.964088（错 13）、Fusion acc=0.955801（错 16）、预测变化 5（状态 PASSED）。
- 关键证据：两种强制回退 gate 精确 0 且 final==gray（0 失败，max diff 0.0）；Fusion gray 分支与独立 Gray 权重 119 参数 + 117 buffer 完全一致；5 个变化样本 = 4 个 real_fish 由对改错（gate≈0.606–0.609、valid_ratio 0.71–0.76、brightness≈1.0、gate*delta 相对 gray logits 范数 0.39–0.59，来自同一采集组连续帧）+ 1 个 plastic_submarine 由错改对；real_fish 召回 0.9518→0.9036，其余类别不变或略升；打乱偏振 5 seeds 的 acc 0.9558–0.9586（mean 0.956354，σ=0.0011）与真实 Fusion（0.955801）几乎一致。
- 结论（限本轮 test 证据，不泛化）：偏振通道在 test 上未带来超过 gray 的净收益；Fusion 相对 Gray 的退化集中于 4 个同组 real_fish 样本上由较强 polar delta 引起的翻转。打乱实验为破坏对应关系的诊断，不构成因果证明。未训练、未替换权重、未修改受保护资产。

### 2026-09-28 CMX v2_mask 运行接入（4A）与收尾核验（未训练、未评估）

- 新增独立运行模式 `material_fusion`（`gui/inference_engine.py`）：旧 `gray`/`polar`/`fusion` 三模式与默认配置行为不变，新模式必须显式指定。适配器 `models/material_classifier.py` 严格校验 checkpoint 契约（协议版本、pooling、四类顺序、预处理/mask 版本、smoke 标记、文件 SHA256），双流变体要求显式独立 Gray 回退，E1 纯 Gray 对照拒绝挂回退；预处理复用离线同一代码路径（`_fusion_tensors` + `instance_mask_tensor`），路由复用 `eval_material_fusion.route_batch`。
- 独立实验配置 `configs/experiments/material_runtime_v2_mask.yaml`：显式固定 Model A（`ba049784…`，单类 `target`，imgsz 640）、独立回退（`0ca63f9e…`）、12 个候选 checkpoint SHA256 与工程代表 seed 2026、CMX 输入契约、标定与 SGBM/偏振参数、训练数据身份与运行数据身份分离字段。Model A 训练数据与分类数据共享采集源，独立性记为 `unknown`。
- 有界验证 `scripts/verify_material_runtime.py`（≤16 NPZ、≤8 帧、≤15 min 真实前向）：Layer A 四变体输入张量与离线逐元素相等、logits 最大误差 0.0；Layer B 8 帧 GT 回放与真实 Model A 路径，实例关联一致，参数与 buffers 未变；真实帧自然回退 0 次（仅为观测，不等于回退路径已验证）。
- 收尾核验 `scripts/closeout_material_runtime.py`（`tests/material_fusion/test_material_runtime_closeout.py`，35 项）：① 历史 4A 证据按落盘字节复核——JSON 可解析、摘要一致、12 候选+Model A+回退身份复算无差异、报告内部一致；历史 pytest 计数（33/797/1439）无留存日志，标记 UNVERIFIED。② 偏振内容一致性补证：原 4A 同一 8 帧、17 实例，`signed_q`/`abs_q` 逐元素**精确相等**（atol=1e-6、rtol=0，最大误差 0.0），无效区置零与 `valid ⊆ instance_mask` 契约成立。③ 完整 val 帧级 GT 索引：权威来源为 `datasets/underwater_seg_binary_v2/split_assignment.csv` + 四类标注 `datasets/underwater_seg_v2/labels`，180/180 帧、368 实例（90/63/197/18）、与 V5 val 样本双向关联 0 缺口，立体有效性记为 `not_evaluated`。④ 已按 LiteralPath 校验并删除本轮旧 scratch `.test_artifacts/4a_20260928T064802Z`（6 个日志先保存至证据目录并核对 SHA256）。
- 证据目录：`analysis/runs/material_runtime_v2_mask/4a_20260928T064802Z/`（4A 原证据，未修改）与 `.../4a_closeout_20260928T073812Z/`（收尾证据，含 `closeout_report.json/md`、`evidence_verification.json`、`polar_content_parity.json`、`val_frames_gt.jsonl`、`gt_index_summary.json`、`preserved_logs/`、`cleanup_report.json`、`artifact_manifest.json`）。
- 未启动 4B；未做训练、模型质量/性能评估或部署；`configs/default.yaml`、数据集、标注、审计、权重与历史报告未修改。

### 2026-09-28 4A 入口修复（独立实验配置可正式构造引擎；valid 一致性纳入判定）

- 新增正式构造入口 `DualStageInferenceEngine.from_material_config(config_path, variant=..., seed=...)`：variant/seed 必须精确匹配唯一候选（缺失/重复拒绝，不选“最新/最高分/第一个能加载的”）；候选文件 SHA256、checkpoint 元数据与配置契约（imgsz、类别顺序、预处理/mask/pooling 版本、upstream commit、V5 指纹、V4 父指纹）三方一致才加载；Model A 与 E2/E3/E4 的独立 Gray 回退均先核验摘要；E1 纯对照不绑定回退，旧默认 Gray 不作替代。配置中的设备、Model A 阈值/imgsz、裁剪 padding、标定与校正、baseline/focal、条带与 SGBM 参数全部生效；引擎未暴露为参数的固定值（精度、偏振 gain/epsilon/min_intensity_sum、条带全图阈值）与引擎实际行为比对，不一致即拒绝；相对路径一律相对项目根解析，不随工作目录变化。
- `from_config` 遇到实验配置（无 `model_b.path`）不再抛 `KeyError`，而是指向新入口；`material_fusion` + 裸 `model_b_path` 的直接构造被明确拒绝并给出指引（不再隐式产出缺少回退的双流适配器）；受控注入 `model_b` 的路径保留。`scripts/verify_material_runtime.py` 改为经该入口构造引擎、并经同一身份校验构造器构建各变体分类器，验证管线与生产管线一致。
- 修复 `scripts/closeout_material_runtime.py::compare_instance_arrays` 的判定缺口：`rebuilt_valid_equals_npz_valid` 之前只记录不参与判定（两侧全零 q、NPZ valid 全 1、rebuilt valid 全 0 仍判 PASSED）。现将 valid 缺失/形状不同/取值不同一律判 FAILED，失败原因写入记录并向上传播到帧、报告与 CLI 退出码；`q == 0` 且 valid=1 的合法测量不被误判。
- 本轮验证：聚焦 164 passed；全量 1497 passed, 1 skipped；compileall 与 `git diff --check` 通过。经工厂入口的最小真实前向（2 个 val NPZ、1 对 val 帧、seed 2026）PASSED：层 A 四变体张量精确相等、logits 最大误差 0.0，层 B 关联正确、GT 回放通过、参数与 buffers 未变，真实前向 1.9 s。另做一次内存内强制 polar_invalid 检查：4 个实例全部走独立 Gray，路由结果与已锁定独立 Gray 模型的输出完全一致，实例 ID、数量与顺序保持。
- 证据：`analysis/runs/material_runtime_v2_mask/4a_entry_fix_20260928T082519Z/`。未训练、未评估 4B、未修改数据/权重/配置/历史报告。

### 2026-09-28 4B 回顾性评估与离线回放性能测量（不训练、不用 test、无独立数据）

- 新增固定协议评估 `scripts/eval_material_pipeline.py` 与性能脚本 `scripts/benchmark_material_pipeline.py`（配套 `test_material_pipeline_eval.py` 27 项、`test_material_pipeline_speed.py` 11 项，TDD）。引擎抽出 `process_instances_detailed`，GT 受控路径复用生产阶段代码；评估/性能均经 `from_material_config` 工厂入口构造。
- 匹配契约：校正左图、类别无关一对一 mask IoU>=0.50，先最大匹配数再最大总 IoU，稳定 ID 决胜；Hungarian 实现，无新依赖。分类错误按真实类计 FN、预测类计 FP；漏检/误检/未分类全部计入端到端指标并保持计数守恒。
- 质量结果（180 帧 val，368 GT，380 预测，346 匹配）：固定工作点检测 P/R/F1=0.910/0.940/0.925；E4 s2026 匹配实例条件分类准确率 1.000、E1 0.997；端到端 Macro-F1≈0.77（检测损失主导）。同 seed E4-vs-E1 成对：s2026 改对 1/改错 0，s2027 0/0，s2028 1/0；4 组采集组 paired cluster bootstrap（2000 次、seed 2026）E4-E1 差值 CI 各 seed 均跨/触 0——偏振净收益仍未确立。层 A1 离线 V5 复核对既有成绩一致（E4 s2026 pure Macro-F1 0.9926）。本次全量评估观测到 3 例自然回退（low_valid_ratio 1、no_valid_disparity 2）。
- 性能结果（RTX 4060 Laptop、float32、预热 10/测量 100、离线回放预加载已校正帧）：1 目标 p50≈162/177 ms（6.05/5.62 FPS，E1/E4），2 目标≈205/215 ms，5 目标≈285/290 ms，最大负载≈305/317 ms（3.26/3.18 FPS）；峰值显存 135-171 MB；条带 SGBM 38-42 ms、偏振+深度 35-50 ms、Model A 23-56 ms、E4 分类 39-82 ms（最重阶段）、全回退路由 14-23 ms；在线校正单独诊断≈8.6 ms（不计入回放总时延）。0 目标自然场景不存在（conf 0.25 下 val 全帧均检出）——如实记 NOT_COVERED。队列/相机/GUI NOT_MEASURED，非相机到 GUI 端到端 FPS 结论；20-30 FPS 是目标不是结论。
- 证据：`analysis/runs/material_runtime_v2_mask/4b_20260928T091849Z/{quality,performance}`；聚焦 836 passed、全量 1534 passed 1 skipped、compileall 与 `git diff --check` 通过。独立数据泛化 INSUFFICIENT_EVIDENCE；未给部署 PASS（无预批标准）。

### 2026-09-28 4B 修复轮（更正 bootstrap、A2 几何对齐、独立数据入口关闭、性能口径纠正）

- **bootstrap 更正**：原 `bootstrap_end_to_end_macro_f1_e4_vs_e1` 实为逐 GT 0/1 正确率差值，未计算 Macro-F1、未计入未匹配预测。现拆分为 `instance_correctness_delta`（更名，`paired_gt_instance_correctness_delta`）与真正的 `end_to_end_macro_f1_e4_vs_e1`（按采集组有放回重采样 4 组、重复组按权重计入、每次重算各类 TP/FP/FN 含漏检与误检、2000 次、seed 2026；点估计取全样本 Macro-F1 差）。更正后 B-routed：E4−E1 = −0.0035 [−0.0236, +0.0855]（s2026）、−0.0038 [−0.0071, +0.0415]（s2027）、+0.0051 [−0.0218, +0.0236]（s2028）——全部跨/触 0，偏振净收益仍未确立。成对改对/改错更正为 s2026 1/0、s2027 0/1、s2028 2/0（原报告后两行有误）。
- **A2 几何对齐**：原 A2 用栅格 mask 外接框同时当条带 bbox 与裁剪窗，与 V5 有 15/368 不一致。新增共享 `core/gt_geometry.py`（条带 bbox = 多边形极值；裁剪窗 = `make_crop_window`），引擎 GT 路径经 GT 专用 `crop_windows` 覆盖传入，Model A 路径与旧行为不变。对齐中还发现第二处缺口：立体无效实例在 V5 中 quality 与偏振通道全零，而线上引擎报实测比值——GT 路径以 GT 专用 `gt_generation_contract` 复现生成契约（仅 GT，线上不启用）。全量对齐：**368/368 PASSED**（gray/instance_mask/polar_valid/signed_q/abs_q 与 NPZ 精确一致，quality atol=1e-6 rtol=0；四变体输入张量与离线逐元素相等；logits atol=rtol=1e-5）。B 层数值未因修复改变（380 预测/346 匹配/22 漏检/34 误检，各 seed Macro-F1 与原报告一致）。
- **报告完整性**：A1/A2/B 分别汇总，各自含 pure（无回退诊断）与 routed（生产路由）；逐候选支持数/accuracy/Macro-F1/逐类 P/R/F1/混淆矩阵与同 seed 成对改对改错清单；保留全精度原始 logits；按采集组与偏振质量分箱给出指标（质量分箱以"已匹配"为条件，未匹配 GT 单列且不虚构预测质量）；22 个未匹配 GT 记录（最大 mask IoU、最佳重叠预测是否已属其他 GT）、未匹配预测与未分类事件单独成文件；计数守恒成立；真实鱼 18 GT 仅 3 个匹配正确，"未匹配"不等于"完全未检出"，不以文件名作为错标证据。
- **独立数据入口关闭**：原报告的"独立数据准入门已完成"表述撤回——该辅助只检查类别顺序与来源重叠，可能被误当作准入通过。现已删除该路径；`--independent-data` 在创建输出目录、加载模型之前即以非零退出明确拒绝。真正的独立评估仍需另行设计验收。
- **性能口径纠正**：所选帧在计时前落盘（`scenario_selection.json`）；`available_frames`（85/25/3/1）与 `measured_unique_frames`（各 1）分列；设备阶段开始前与结束后均同步，整帧时延独立测量、阶段和不得充当总时延；`classification_wrapper` 如实命名（含预处理/传输/前向/结果整理），crop-only、H2D-only、纯 forward 标 NOT_MEASURED，并撤回"双流前向为首要瓶颈"的结论；E1/E4 顺序驻留与测量（先释放再加载），记录加载后显存与峰值 allocated/reserved；保存原始重复计时数组；回退测量标为分类器级并记录实际 CMX/回退路由数，单实例混合批记 NOT_APPLICABLE；预算不足/少测/失败不得汇总为 COMPLETED。更正后（100/100 次）：1 目标 E1 124 ms / E4 141 ms（8.0/7.1 FPS），2 目标 154/170 ms，5 目标 221/235 ms，最大负载 240/254 ms（4.2/3.9 FPS）；0 目标自然场景不存在 → NOT_COVERED，总状态 PARTIAL。相机/GUI/队列/丢帧仍 NOT_MEASURED，在线校正单独诊断 ≈7.7 ms 不计入回放总时延。
- 验证：聚焦 863 passed、全量 1561 passed 1 skipped、compileall 与 `git diff --check` 通过；证据 `analysis/runs/material_runtime_v2_mask/4b_fix_20260928T114135Z/`（含首次对齐失败尝试的保留日志）；原 4B 证据与历史权重、数据、默认配置未修改。

### 2026-09-28 4B 收尾轮（异常分支、严格验收状态、报告口径；无推理、无测速）

- **pure 非有限输入**：`predict_pure_batch` 原在追加后判定可评估性并弹出未追加的列表（首项 NaN 触发 IndexError），且回填索引未同步过滤。现改为追加前判定，张量/标记/原因/原始索引由 `forward_indices` 一一对应，按原始索引回填；无可前向样本时直接返回失败结果，不调用 torch.stack、CMX 或回退；routed 规则与 E1 不换 Gray 不变（用按行内容决定类别的微型 CPU 模型验证回填顺序）。
- **回退汇总**：原实现读不存在的字段（分子 0）且分母混计 pure/routed（760）。新增共享 `summarise_fallback`（评估与收尾共用），精确筛选 layer=B_model_a_predictions、condition=routed，按 variant/seed 分列，分子取实际 used_fallback，分母保留全部实例（未分类单列），原因之和必须等于回退数，重复键/缺字段/未知身份拒绝，E1 回退必须为 0。重算：E2/E3/E4 各 seed 均为 **3/380**（no_valid_disparity 2、low_valid_ratio 1），E1 全部 **0/380**。
- **严格对齐验收**：原 `_alignment_summary([])`、仅 NO_NPZ、单项 UNAVAILABLE 均可 PASSED，signed_q/abs_q 用 atol=1e-6。现改为：gray/instance_mask/polar_valid/signed_q/abs_q/输入张量要求同 shape、同 dtype 且数值精确相等；quality atol=1e-6 rtol=0；logits atol=rtol=1e-5；非有限一律不通过；预期样本集合来自已核对 manifest 的 GT 对应关系（非"实际返回条数"）；重复/额外/未知状态/缺必要检查/空 checks/行级 FAILED 均 FAILED，缺覆盖、GT 无 NPZ、必要检查未完成均 PARTIAL；报告状态与退出码同源（COMPLETED/0、FAILED/非零、PARTIAL/非零），JSON/Markdown/子报告一致。复核历史：成功尝试 PASSED（368/368 覆盖、0 失配），首次失败尝试 **FAILED**（其记录的 summary.status 曾为 COMPLETED，差异现已明示）；三项无法从历史记录复核的项目标为未验证，不冒充本轮实测。
- **零重叠更正**：原逻辑在全零 IoU 上用 argmax 选中第一个预测，导致 15 条被计为"最佳预测已匹配其他 GT"，据此得出的"20 个合并/重复信号"表述**撤回**。现：最大 IoU=0 时保留 0.0、best_prediction_index=null、matched_elsewhere=null、no_positive_overlap=true；无预测单独用 predictions_in_frame=0 与 null IoU 区分；仅正值才选最佳（稳定最小索引 tie-break）。重算：22 个未匹配 GT 中 **16 个无正重叠、5 个正重叠且该预测已匹配其他 GT**；15 个未匹配真实鱼最大 mask IoU 全为 0。不判定根因。
- **性能阶段改名**：`crop_preprocess_cpu` → `cpu_tensor_preparation`（含 renamed_from/excludes/说明），计时值、PARTIAL 总状态与 NOT_COVERED/NOT_MEASURED 项一律不变；本轮未重新测速。
- 新增只读收尾入口 `scripts/closeout_material_pipeline.py`（校验源清单摘要与必要文件、拒绝越界与已存在/相同输出目录、不加载任何模型、不执行前向/SGBM/benchmark），产物含 source_provenance、quality_summary_corrected、alignment_recheck、unmatched_gt_corrected、performance_report_corrected、raw_timings_corrected、correction_summary、artifact_manifest；一致性检查确认 48 组 A2/B 指标、三个 bootstrap 区间、逐实例预测匹配与八组计时数组均未变化（失配 NONE）。
- 验证：聚焦 896 passed、全量 1594 passed 1 skipped、compileall 与 `git diff --check` 通过；证据 `analysis/runs/material_runtime_v2_mask/4b_closeout_20260928T143326Z/`（corrections/）。环境说明：本轮要求的 `CUDA_VISIBLE_DEVICES=-1` 在该 torch 2.12.1+cu130/Windows 构建下会使 `torch.cuda.is_available()` 触发访问违例（退出码 139），故 pytest 未设该变量；本轮仍未执行任何 GPU 推理（新测试用 CPU 替身，收尾入口不加载模型）。

### 2026-09-29 4B 收尾入口 CLI 恢复（仅合成 CPU 验证；未运行真实证据收尾与全量测试）

- 恢复 `scripts/closeout_material_pipeline.py` 缺失的三个函数并统一契约；`verify_source` 改为每个证据子目录用自己的清单覆盖自己的数据文件（清单不自哈希），首次失败尝试用其自身清单校验；拒绝空清单、缺摘要、摘要错误与路径越界。
- 完整性改为对已准入配置强制：12 候选身份必须齐全，每候选四个非空指标块（layer_a2/layer_b × pure/routed）共 48 块，bootstrap 必须为三个具名 seed；"两边同样缺失"不再通过，空块不计入已比较块。
- 落盘复核改为重新读取实际写出的 `quality_summary_corrected.json`、`performance_report_corrected.json`、`raw_timings_corrected.json` 与来源比较（自洽的摘要不能代替内容保持）。
- 接通配置/快照/数据绑定（候选身份与摘要、类别顺序、material_model 字段、训练数据指纹、V5 审计），重复身份在构造映射前检出；来源记录的指纹与本轮计算的摘要分开陈述，不宣称完成新的内容重审。
- `gt_without_npz` 现进入不完整性判定（PARTIAL）。
- 新增共享合成夹具 `tests/material_fusion/pipeline_synthetic.py`，通过真实 `main()` 覆盖：合法→COMPLETED/0；alignment FAILED→FAILED/非零；覆盖不完整→PARTIAL/非零；准入与身份各类拒绝；输出目录拒绝；模型工厂以"一旦调用即失败"探针证明未构造模型。
- 验证（进程级隔离 TEMP/TMP/PYTHONPYCACHEPREFIX/PYTHONDONTWRITEBYTECODE/MATERIAL_FUSION_TEST_ARTIFACTS，CPU-only）：CLI 14 passed（exit 0）、四文件聚焦 108 passed（exit 0）、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_cli_finish_20260929_020654_8c3b5d/`。
- 未执行：真实历史证据收尾主入口、全量 pytest、真实模型前向/checkpoint 加载/引擎构造/SGBM、训练/评估/benchmark/GPU smoke；覆盖率 NOT_MEASURED（未安装覆盖率工具）。

### 2026-09-29 4B 收尾入口契约修复（仅合成 CPU 测试；未运行真实证据收尾与全量测试）

- 数据绑定落地：`audit_passed` 必须严格为布尔 True；复算 manifest/summary 的 SHA256 与审计记录严格比较；V4 parent 指纹必须与配置一致；真实 V5 审计无 `dataset_fingerprint` 字段，该缺失被如实报告，不读成 None 当作已验证，也不新增第二套指纹定义；GT 索引通过其所属证据清单绑定，"本轮计算的摘要"与"来源记录的身份"分开陈述。
- 结构校验取代"非空即通过"：48 个指标块必须具备生产字段（counts/detection/conditional_classification/end_to_end、非负整数计数、支持数一致、有限比率）；每个 bootstrap seed 必须具备真实统计结果（point_estimate/ci_low/ci_high、区间不倒置）或显式不可估计状态；"两边同样缺失"不再通过，也不计入已比较块。
- 逐实例覆盖进入最终判定：按候选×层×条件检查记录；A2 必须与已绑定 GT 索引精确对应（缺失至少 PARTIAL，额外 FAILED），B 每候选每条件必须有记录，重复与身份冲突 FAILED；缺失证据不得包装成"0 回退 0 错误"。
- 落盘失败状态一致：重新读取磁盘产物与来源比较，发现保持项变化时把报告改判为 FAILED（JSON/Markdown/清单同时更新）并返回非零，不留下仍称成功的顶层报告。
- 验证（进程级隔离 + CPU 包装）：CLI 14 passed（exit 0）、四文件聚焦 108 passed（exit 0）、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_closeout_contract_20260929_032046_6d7784/`。
- 未执行：真实历史证据 closeout main()、全量 pytest、真实模型前向/checkpoint 加载/引擎构造/SGBM、训练/评估/benchmark/GPU smoke；覆盖率 NOT_MEASURED。

### 2026-09-29 4B 收尾入口 B 实例身份覆盖修复（仅合成 CPU 测试）

- 确认漏洞：B 覆盖只比数量，等量替换 `frame_id="nonexistent_frame"` 或 `pred_instance_id=999` 仍 PASSED。
- 修复：预期身份集合改为由已绑定逐帧预测证据构建（生产编号规则为 0..predictions-1，依据 `eval_material_pipeline` 对路由结果的 enumerate）；对每候选的 B/pure 与 B/routed 按 (frame_id, pred_instance_id) 集合比对；未知帧、越界/非整数编号、同候选同条件内重复身份、额外身份判 FAILED，缺失预测实例至少 PARTIAL；零预测帧合法（证据明确 predictions=0 时无需记录），缺帧证据仍不能完整通过；矛盾记录集改为 FAILED 报告而非抛出。
- 新增 6 个真实 main() 参数化用例（pure/routed × 未知帧/越界/重复身份），修复前 6 例全 RED、修复后 GREEN；聚焦 114 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_b_identity_fix_20260929_065116_9c7378/`。
- 仍未完成（不得称为整体契约通过）：`frames.jsonl` 未强制纳入来源清单；帧证据结构/类型/重复帧校验缺失；`summary_supports` 未接入 main()；bootstrap 不可验证状态的 PARTIAL 传播、paired_changes/conclusions 同时缺失检查、落盘故障注入均未实现。

### 2026-09-29 B 实例编号异常处理收尾与验收记录纠正（仅合成 CPU 测试）

- 修复：B 记录 `pred_instance_id` 为 None/dict/list 时先在 `summarise_fallback`/覆盖检查之前校验（字段必须存在且为非负整数，bool 不算整数，不以 int() 强转字符串或浮点），非法值形成可读原因、`status=FAILED`、非零返回、Markdown 一致，不崩溃、不删记录、不伪装成零回退；A2 合法的 `pred_instance_id=None` 不受影响。
- 新增 `TestIllegalBInstanceId` 14 个真实 main() 参数化用例（pure/routed × None/True/0.0/"0"/list/dict/缺字段），修复前 13 failed（含 TypeError 逃逸）→ 修复后 GREEN；本轮聚焦 128 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_b_identity_closeout_20260929_071026_3a6b24/`。
- 验收记录纠正（不改历史日志）：上一轮 `pytest_green_02.log` 实为 19 passed/1 failed，其最终通过依据是 `pytest_focused_02.log` 的 114 passed；最初两个 duplicate_identity 用例未在同一候选同条件内构造重复，后经调试修正，故初始六项失败不等于六个独立漏洞复现（未知帧/越界四项 RED 成立）。
- 仍未完成：frames.jsonl 未强制纳入来源清单；帧证据结构/类型/重复帧校验；summary_supports 接入 main()；bootstrap 不可验证状态传播；paired_changes/conclusions 同时缺失检查；落盘故障注入与最终状态发布。整体 4B 契约仍未全部验收。

### 2026-09-29 quality/frames.jsonl 来源清单绑定（仅合成 CPU 测试）

- 修复：`quality/frames.jsonl` 纳入必要来源文件，并纳入 quality 清单必须覆盖的集合（复用既有路径/SHA256/清单校验）；未列入清单、摘要不符、文件缺失均在读取其内容之前拒绝，无静默降级；清单不自哈希、各子目录用自身清单、路径越界与失败尝试规则保持不变。
- 夹具修复：`pipeline_synthetic.perturb_alignment_row()` 重建清单时补回 `frames.jsonl`，使合法证据覆盖完整，alignment FAILED/PARTIAL 仍到达目标分支；未放宽生产准入。
- 新增 `TestFrameEvidenceBinding`（未覆盖 / 缺失 / 摘要错误）真实 main() 用例：核心"未覆盖仍被消费"修复前 RED（DID NOT RAISE）、修复后 GREEN；摘要错误用例为既有行为的回归测试。
- 本轮 CLI 37 passed、四文件聚焦 131 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_frame_binding_20260929_072524_37a7b6/`。
- 仍未完成：帧证据内部结构/类型/重复帧与完整帧覆盖校验；summary_supports 接入 main()；bootstrap 不可验证状态传播；paired_changes/conclusions 同时缺失检查；通用落盘故障注入与最终状态发布。整体 4B 契约仍未全部验收。

### 2026-09-29 frames.jsonl 来源绑定的入口级验收（仅测试与文档；生产代码未改）

- 纠正上一轮表述：其三项用例直接调用 `verify_source()`，属**辅助函数级**证据（RED 为 `DID NOT RAISE`），原报告称"三个真实 main() 用例"不准确。
- 本轮改为四种真实 `main()` 场景（各自独立输出目录、只改一个因素）：① 文件在、清单项缺失 → `ValueError`（含 frames.jsonl / not covered）；② 文件缺失、清单仍记录 → `FileNotFoundError`（含 frames.jsonl / missing）；③ 文件与清单项同时缺失 → `FileNotFoundError`（必要来源缺失分支）；④ 摘要格式合法但错误 → `ValueError`（含 digest mismatch）。生产修复已存在，故入口测试首次即 GREEN，未通过回退代码制造 RED；`main()` 抛出预期异常按"入口抛出预期异常"报告，未声称测量了独立进程退出码。
- 结果：四场景 4 passed、CLI 38 passed、四文件聚焦 132 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_frame_binding_cli_20260929_073947_a87d19/`。
- 仍未完成：帧证据内部结构/类型/重复帧与完整帧覆盖；summary_supports 接入；bootstrap 不可验证状态传播；paired_changes/conclusions 同时缺失检查；通用落盘故障注入与最终状态发布。整体 4B 契约仍未全部验收。

### 2026-09-29 frames.jsonl 结构/身份/重复帧与完整帧覆盖校验（仅合成 CPU 测试）

- 生产格式（只读核对）：`eval_material_pipeline` 每评估帧一行，`frame_id` 为非空字符串（来自权威 split 清单），`predictions = len(b_instances)` 为整数；实例为空的帧同样写入且 `predictions=0`。预期帧集合取自已绑定的 GT 索引（冻结评估范围），不来自 frames.jsonl 自身，也不来自实例记录。
- 旧逻辑（已在隔离脚本中复现留证 `prefix_logic_demo.txt`）：`str(frame_id)`/`int(predictions)` 强转（123→"123"、3→3、True→1）、重复 frame_id 被字典覆盖、缺失帧静默缺席并被当作"无预测"。
- 新增 `validate_frame_evidence()` 并接入 main()：记录须为 JSON 对象；frame_id 非空字符串且不做 str()/strip() 修补；predictions 必须 `type is int` 且 ≥0（bool/float/str/None/list/dict 全部拒绝，无 int() 强转）；重复帧（含内容相同与计数不同）拒绝；实际帧集合必须与预期集合一致。未知/额外帧、重复、非法身份或非法计数 → FAILED；纯覆盖缺失 → PARTIAL；矛盾优先于缺失。显式 `predictions=0` 的帧是合法帧且不得有 B 实例记录（有记录则 FAILED 并给出专门原因）；缺失帧绝不按零预测处理。
- 新增 `TestFrameEvidenceSchema`（25 例）经真实 main()：合法基准（现含一个 GT 实例列表为空、预测数为 0 的帧）仍 COMPLETED/0；零预测帧带 B 记录 → FAILED；删帧记录 / frames.jsonl 全空 → PARTIAL；未知帧等量替换、额外未知帧、完全相同重复、计数不同重复、10 种非法 predictions、6 种非法 frame_id、非对象记录 → 各自目标原因 + JSON/Markdown 状态一致。
- 结果：专项 25 passed、CLI 63 passed、四文件聚焦 157 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_frame_schema_20260929_075407_03ffe2/`。修复与测试同轮完成，故新增用例属入口级验收而非捕获到的 RED（旧行为以代码阅读与隔离复现留证）。
- 仍未完成：summary_supports 接入；bootstrap 不可验证状态传播；paired_changes/conclusions 同时缺失检查；通用落盘故障注入与最终状态发布。整体 4B 契约仍未全部验收。

### 2026-09-29 缺失帧证据语义修复与上轮日志纠正（仅合成 CPU 测试）

- 缺陷：覆盖检查只用"有证据的帧"构造预期身份集合，导致被删除帧的合法 B 记录被判为 unknown/out-of-range 而 FAILED——缺失的预测计数被误读为身份错误。
- 修复：main() 把独立预期帧集合（来自已绑定 GT 索引）传入覆盖检查，按三种情形区分：① 有合法证据的帧照旧范围校验（显式零预测帧出现记录、越界/负数编号、重复身份 → FAILED；应有实例缺失 → PARTIAL）；② 在冻结范围内但缺少帧证据的帧 → **无法验证**（PARTIAL，绝不按零预测、绝不报身份错误），但其上不依赖计数的矛盾（非法/负数编号、重复身份）仍 FAILED；③ 冻结范围之外的帧上的记录 → FAILED，且不被其他缺失帧降级。缺失帧的预测计数不从已观测编号最大值反推。
- 状态优先级：明确矛盾 FAILED > 证据缺失 PARTIAL > 完整通过 PASSED；帧证据缺失时 instance_coverage 本身也不得 PASSED。
- 新增 `TestMissingFrameEvidence`（12 例）经真实 main()，修复前 5 例 RED（含上述误判）、修复后全绿；结果：专项 12 passed、CLI 75 passed、四文件聚焦 169 passed、compileall 0、`git diff --check` 0；证据 `analysis/runs/material_runtime_v2_mask/4b_missing_frame_fix_20260929_081331_ae1354/`。
- 上轮日志纠正（不改历史日志）：`pytest_schema_01.log` 实为 1 failed / 24 passed / 38 deselected，不得记为 25 passed；该轮最终通过依据为 `pytest_cli_03.log` 63 passed 与 `pytest_focused_02.log` 157 passed；该轮未捕获独立 RED 的事实保留。
- 仍未完成：summary_supports 接入；bootstrap 不可验证状态传播；paired_changes/conclusions 同时缺失检查；通用落盘故障注入与最终状态发布。整体 4B 契约仍未全部验收。

### 2026-09-29 四项收尾阻断修复（用户明确授权 Codex 直接实施）

- main 已分别接入每候选 B/pure、B/routed 的 predictions 支持数；严格整数校验，完整帧证据下与预测总数核对，缺帧时保留 PARTIAL，不拿局部计数作全量总数。既有实例身份检查继续执行。
- bootstrap 的 UNVERIFIED/NOT_MEASURED/NOT_ESTIMABLE 已传播为顶层 PARTIAL/非零；明确矛盾仍优先 FAILED。
- paired_changes/conclusions 来源与更正结果分别检查必要非空结构，不能“两边同样缺失”就通过。
- 落盘先冻结预期字节，再写入并独立复核全部产物和清单，最后报告状态；失败最多重写一次 FAILED 并再次复核。持续失败时撤销本轮输出中的质量汇总、Markdown、清单并尝试写 publication_failed.json，返回非零；不会删除来源/历史证据。
- 新增 37 个真实 main 合成入口用例；首轮 RED 31 failed/4 passed，补测清单参数被修改时再发现并修复校验基准别名问题；最终五文件聚焦 215 passed。原有 9 个 support helper 回归测试保留在内。覆盖率未测量（未安装工具）。
- 证据：analysis/runs/material_runtime_v2_mask/codex_4b_blockers_20260929_c93e61/。本轮生产修改限 scripts/closeout_material_pipeline.py，新增 tests/material_fusion/test_material_pipeline_blockers.py，并更新本文件及实验文档。
- 以上四项代码契约已在合成 CPU 验收中关闭；真实历史证据 closeout、全量验证仍未运行，不能据此宣称整体 4B、偏振收益、泛化、性能或部署通过。未训练、未修改真实数据/权重/历史报告/默认配置，未提交推送。

## 已解决问题的影响

| 原问题 | 若不修复的影响 | 当前处理 |
|---|---|---|
| MATLAB 旋转矩阵直接交给 OpenCV | 校正几何、标注位置、视差和深度全部建立在错误坐标系上 | 生成 v2 并完成几何 QA |
| 连续帧跨 train/val | 指标被近重复帧抬高，无法反映泛化 | 整组隔离并落盘 split 清单 |
| Model B 训练/推理通道不明确 | 消融不可复现，模型可能接收错误物理通道 | 明确 gray/polar 两种固定通道 |
| 仅有训练/val 指标 | 无法做正式模型接受判断 | 建立独立 test split 和稳定报告 |
| 未做配对消融 | 会把形状/灰度线索误归因于偏振 | 完成配对 bootstrap，默认 gray |
| 每实例重复全图偏振 remap | 目标数增加时延迟线性恶化 | 合并为每帧一次 |
| 文档仍指向旧 NCC/旧路径 | 后续模型可能重新运行错误链路 | README、配置和本交接同步到当前实现 |

## 清理与归档

旧数据已打包并移动到：

`archives/datasets_original_20260912.zip`

归档大小 `3,996,732,140` bytes，SHA-256：

`FF502C634E80FAB68E9246FB0A5AA02915E24DF88DC4A54DDF3047D6C12EA256`

已验证归档可列出 `datasets/bd_image`、`bz_JSON`、`Mixed`、`Rectified` 和 `Single`。以下旧数据和临时运行目录已在确认归档有效后从活动目录清理：

- 错误旋转约定生成的 `datasets/Rectified`（约 2.46 GB）。
- 已迁移且含 base64 的 `datasets/bz_JSON`（约 1.03 GB）。
- `runs/segment` 与所有 `runs/train/smoke_model_*`。

旧材料先验、旧 CUDA sampler、无引用的旧分析/采集/校准脚本及缓存仍未清理。

不得删除原始 `Single/Mixed`、标定 `bd_image`、任何 v2 数据、四个正式模型或 polar 消融资产。

人工修正版另有两个独立回退归档：

- `archives/bz_JSON_v2_manual_20260914.zip`：人工修改后、元数据规范化前的完整 v2 JSON，SHA-256 `F8B2E7652C0CB1DF8C1C32DFD9FF52A41DB3FE4016C09D35879C77A5F2B42B25`。
- `archives/underwater_seg_v2_before_manual_20260914.zip`：人工修正传导前的 YOLO 标签与清单，SHA-256 `7BA81293EEBA1F325C58B497065F375FD7EA45FA536C99C20E771A780392A32F`。

## 尚未完成及执行计划

### P0：距离真值评测

解决问题：目前只能证明系统会输出深度，不能证明深度准确。

执行：在固定基线/焦距下，使用平面靶或尺寸明确的目标，在 `0.5/0.75/1.0/1.5/2.0 m` 各采集多个场景和浊度；记录激光测距或机械量具真值、同步状态、有效像素率、稳健视差和输出深度。按距离、类别、浊度报告 MAE、RMSE、相对误差中位数/P95、无效深度率。

验收：先冻结允许误差，再采测试数据；建议第一轮工程门槛为中位相对误差不高于 10%、P95 不高于 20%、无效深度率不高于 5%。若不达标，优先重新检查同步、标定和 SGBM 参数，不要用训练集调测试阈值。

### P0：真实双相机 30 分钟验收

解决问题：离线图不能暴露 SDK、触发、掉帧、显存和线程退出问题。

执行：分别测试软件触发和可用的硬件触发，持续记录左右帧号/时间戳、配对数、丢帧数、同步偏差、每阶段耗时、GPU/内存峰值和异常。覆盖启动、停止、相机断开重连和无目标场景。

验收：无崩溃和死锁；停止后线程与相机句柄释放；有可测时间戳时配对偏差不超过配置的 2 ms；掉帧和无效深度率形成可复查日志。硬件不提供时间戳时只能标为未测量。

### P1：实时化

解决问题：当前 6 FPS 不能满足 30 FPS 目标。

执行顺序：

1. 将 Model A/B 导出 TensorRT FP16，逐图比较 mask、bbox、分类和置信度偏差。
2. 对 SGBM 建立 `scale/max_disp/mode/block_size` 精度-速度矩阵，使用距离真值而不是视觉观感选点。
3. 若 CPU SGBM 仍超预算，评估 CUDA stereo 或适合水下数据的 GPU 深度网络。
4. 在单帧正确性固定后再做采集、推理、显示异步流水线，分别报告吞吐量和端到端延迟。

验收：同一正式测试集精度不超过预先约定的退化范围；真实双帧完整链路 P95 不高于 33.3 ms 才能声明 30 FPS。

### P1：补充数据

解决问题：`real_fish` 泛化显著弱于其余三类，Mixed 场景覆盖不足。

执行：优先补充不同姿态、尺度、遮挡、浊度和背景下的 real_fish/Mixed 连续采集组；新组整体进入 train 或 test，不拆帧。新增后重跑 duplicate 检查、group split、baseline 和 Model B 配对消融。

验收：先看独立组的逐类 recall 和 mAP，不以总 mAP 掩盖 `real_fish`；偏振结论必须再次通过同一 bootstrap 门槛。

## 接手检查顺序

1. 阅读 `analysis/project/project_inventory_v2.json` 和 `configs/default.yaml`。
2. 运行 `pip check`，确认当前 Python 环境没有破损依赖。
3. 运行完整 pytest；Windows 使用 `-p no:cacheprovider`。
4. 核对四个 `best.pt` 和三个正式 JSON 报告的 SHA-256。
5. 任何数据重建都从原始 `Single/Mixed` 与标定文件开始，输出到新版本目录，不覆盖 v2。
6. 任何“偏振有效”“实时”“深度准确”的表述，都必须分别由消融、完整管线 benchmark、距离真值支持。
