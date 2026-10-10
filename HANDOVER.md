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

### 2026-09-30 现有数据条件下的检测诊断与偏振信息验证准备（material_existing_data_v1；未训练）

- 新增协议层 `core/existing_data_protocol.py` 与入口 `scripts/prepare_existing_data_protocol.py`：固定基线身份（Model A/E1/E4/独立回退/官方 MiT-B0 全部 SHA256 复核）、V5 全量准入复用既有门（未新增第二套指纹）、开发池=原 train+val（945 帧/2446 样本/2446 GT，test 仅用于排除成员）、4 个无目标帧保留单列、GT↔NPZ 对应 0 缺口；整采集组为单元、seed 2026 全局类计数平衡贪心生成 3 外层折（outer_eval/inner_train/inner_val 全类覆盖，fold 状态 OK），审计组交叉/重复成员/序列不拆分。产物 `analysis/runs/material_existing_data_v1/20260930T021500Z_a3f7c1/protocol/`。解释边界已写入 protocol.json：仅为训练层组隔离，非新盲测，历史权重诊断不构成 OOF；跨组物理对象关系无来源、记 unknown。
- 新增检测诊断 `scripts/diagnose_existing_detection.py`：固定四条件 D0(0.25/640)/D1(0.10/640)/D2(0.05/640)/D3(0.25/960)，匹配复用 eval_material_pipeline（IoU≥0.50 不降阈），GT 跨条件身份=(frame_id, object_index)。**D0 与 4B 历史基线完全一致**（180/368/380/346/22/34，逐帧一致）；D1 匹配 352、D2 匹配 355、D3 匹配 346。22 个未匹配 GT 规则化迹象归因：confidence_related×9、no_corresponding_prediction×10（全 real_fish）、input_scale_related×2、duplicate_or_merge×1——real_fish 召回 conf 0.25→0.05 由 0.167 升至 0.444，imgsz 960 无增益。图册覆盖全部未匹配事件+各类对照（未人工复核）。Model A 单阶段 p50≈28-31ms（非系统 FPS）。每条件首次前向后建立参数/buffers 基线（首前向初始化差异显式排除并记录），其后全程逐一不变。产物 `…/detection/`。
- 新增 `core/material_probe.py`、`scripts/prepare_material_probe.py`、`scripts/train_material_probe.py`：G=官方 MiT-B0 最后一级（256 维）实例 mask 加权池化（eval+inference_mode，提取全程参数/buffers 快照逐一相等）；C=quality 前三项+实例内灰度均值/方差+S 区非空标记（**不含 mean_abs_q**）；S=instance_mask∩polar_valid 上 signed_q 均值/std(ddof=0)/五分位+abs_q 均值（2341/2446 有效，105 显式缺失，NaN 不当零）。缓存只存原始特征（无标准化/PCA）；标准化在拟合入口按折 inner_train 拟合（S 列仅有效样本；缺失置零；零方差列输出零）。偏振质量统计：real_fish 内部有效率 0.544 最低（轮廓带 0.441）；亮度相关弱（Pearson 0.153）；曝光/增益/gamma/时间戳 NOT_AVAILABLE、同步 unknown。泄漏规则有专门测试（扰动 mean_abs_q/signed_q 不改 C；扰动 signed_q 改 S）。产物 `…/features/`。
- 探针训练契约（真实 `--fit` 由用户手动执行，Agent 仅跑过真实缓存 dry-run 且通过）：线性 4 类头、P0/P1/P2 共享 [G,C,S] 槽位（不用槽位严格置零）、inner_val 选模（Macro-F1→accuracy→早 epoch）、outer_eval 冻结后只评一次、固定预算（seed 2026/AdamW/0.01/0.001/≤200ep/全批次/无增强无加权无调参）、不按 outer_eval 选模、输出目录拒绝覆盖、BLOCKED 协议拒绝 fit（exit 3）、NOT_RUN 时不产出训练结果。用户命令见 `manual_commands.ps1`。
- 验证：新增 103 项合成测试（35 协议+20 检测+48 探针，TDD）；聚焦三文件 103 passed、material 套件 1063 passed/1 failed/4 skipped、全量 1835 passed/1 failed/6 skipped、compileall 0、`git diff --check` 0。唯一失败为 `test_license_file_matches_upstream_digest`：**接手时既有的环境性失败**——`core.autocrlf=true` 使 models/cmx/LICENSE 工作区为 CRLF 而提交内容为 LF（提交版本 SHA256 与期望一致），本轮未触碰 models/cmx，未修改测试或许可文件。覆盖率 NOT_MEASURED（未安装 pytest-cov）。日志与退出码在 `…/logs/`。
- 受保护资产核对：协议记录的全部权重/证据 SHA256 与运行后复算一致；V5 manifest/summary 与审计记录一致；git 工作区仅含本轮新增源文件与 .gitignore 一行（新增 `analysis/runs/material_existing_data_v1/` 忽略规则）。本轮无任何真实数据训练；无偏振收益或独立泛化新结论。下一轮唯一优先建议（依据检测归因）：先处理 real_fish 检测置信度/召回层面（D1/D2 工作点代价需与误检一起权衡），不动分类/融合。

### 2026-09-30 修复轮：训练准入、缓存绑定、图册与入口控制流（material_existing_data_v1；未训练、未真实前向）

- Codex 只读核实上轮 839 个产物摘要一致、分组无交叉、D0-D3 计数可信；本轮修复的是入口保护与交付缺口（非缓存损坏证明、非实际泄漏）。
- **缓存绑定与训练准入**：`core/material_probe.py` 新增 `build_cache_binding`/`verify_cache_binding`（三个缓存文件 SHA256+长度为预期状态，禁止"现算现当预期"）、`authoritative_sample_index`（逐折互斥/完整/组不交叉/每折恰一角色/outer_eval 全局恰一次——"某折重复另一折缺失但总数相同"被拒绝）、`verify_cache_manifest`（行序=冻结行序契约，逐行身份/角色/严格布尔核对）、`verify_cache_arrays`（键/形状/dtype/类别行/G,C 有限/有效 S 有限/缺失 S 全 NaN/C 非空标记与 s_valid 一致/imgsz/预处理/槽位顺序/类别顺序/编码器身份）。`scripts/train_material_probe.py` 准入链重写，dry-run/fit 共用，产出 `admission.json`（协议/缓存绑定/类别/特征定义/fold/variant/seed/预算），错误在任何头构建、scaler.fit、训练产物写出之前拒绝。
- **特征准备真实准入**：`scripts/prepare_material_probe.py` 改用 `admit_mask_dataset` 完整门（2808 文件逐一复核 + mask 解码 + 来源重哈希 + V5/V4 指纹比较），不再以 `mask_dataset_fingerprint()` 冒充完整性验证；数据准入检查清单写入 feature_definition.json 的 `data_admission`。原轮提取预算会抛异常，但入口 PARTIAL 与已完成批次落盘当时尚未实现，不能视为已验收；后续预算收尾单独记录。
- **缓存升级**：`scripts/upgrade_probe_cache.py`——旧目录只读，验证旧 artifact_manifest 对三个缓存文件的覆盖与摘要、旧协议自洽、当前 2808 文件完整准入、缓存身份/数组后，按字节复制到新目录 `…/20260930T045500Z_c5d9e8/upgrade/corrected_features/`，生成新 `cache_binding.json` 与 `upgrade_report.json`（字节一致性 original vs copied 三文件全部 identical=True），修复后统一准入复验通过；真实 dry-run 通过（fold0 1116/490/840，fold1 1265/470/711，fold2 1191/360/895）。G/C/S 零重提取。
- **图册重绘**：`render_event_panel`/`rebuild_atlas_from_evidence` + `--rebuild-atlas` CLI——从已保存证据重绘，不调用模型。真实输出 68 面板（22 未匹配 GT + 34 未匹配预测 + 12 对照），GT 绿轮廓/预测红轮廓带逐预测置信度/匹配或最大重叠 IoU/采集组/"迹象，未人工复核"标注全部实际绘制；未匹配 GT 无相关预测时明确标注。合成绘图测试验证红色/绿色像素确实存在；人工抽查三类各一张（unmatched_gt real_fish、unmatched_pred plastic_fish 帧、control plastic_fish）确认合格。
- **失败停止/设备/预算**：真实 main() 合成测试证明 D0 不一致后不构造计时模型、不执行后续条件/计时前向（exit 2，evidence 保留，atlas 仍由已保存证据产出）；`measure_timing` 设备经 `torch_device_name` 规范化（"0"→cuda:0 每次测量前后真同步；CPU 路径 monkeypatch 证明不触碰 CUDA）。原轮 BudgetWatch 的入口 PARTIAL/2 测试仅覆盖诊断开始前耗尽，未覆盖计时中途及特征提取入口；不再概括为全部路径通过。上一轮历史计时在 device="0" 未走显式同步分支下测得（生产解析含 CPU 回传，不能断言原时延偏低；数值未改写）。
- **文档纠正**：D3 与 D0 总匹配数相同但找回 7 丢 7（real_fish 找回 1 丢 1）、未匹配预测 34→25——只说"总体匹配数与 real_fish 召回没有净提升"，不写"尺度无作用"；C 的灰度统计是标准差（ddof=0）不是方差；折支持数表修正（outer_eval GT=样本数 840/711/895）；测试结果完整记录（material 1063 passed/1 failed/4 skipped，全量 1835 passed/1 failed/6 skipped，唯一失败=既有 LICENSE CRLF/LF 换行差异，未跳过未弱化）。
- 验证（按 `20260930T045500Z_c5d9e8/logs/` 纠正）：`pytest_focused_02.log` 为 105 passed；`pytest_material_02.log` 为 1114 passed/1 failed/4 skipped；`pytest_all_03.log` 为 1886 passed/1 failed/6 skipped（LICENSE 失败同上，单列既有问题）。撤回未绑定本轮日志的 171/1887 数字，不改写历史日志。compileall 0；`git diff --check` 0。覆盖率 NOT_MEASURED（未安装工具）。`.zcodeignore` 归属不明，未改动、未删除、保持未跟踪。
- 受保护资产复核（SHA256，非 mtime）：Model A/E1/E4/回退/MiT-B0、V5 manifest/summary/audit、旧轮 839 产物清单、历史 4B 证据——运行前后一致。真实数据训练 NOT_RUN；`--fit` 命令在 manual_commands.ps1 供用户执行（使用 corrected_features 路径）。

### 2026-09-30 预算入口收尾（本次用户授权 Codex 实施；真实训练 NOT_RUN）

- 证据：`analysis/runs/material_existing_data_v1/20260930_codex_budget_finish_b81f29/`。改动仅两项生产脚本、一个新增合成测试文件、实验说明/HANDOVER/manual_commands.ps1；未改训练算法或现有缓存。
- 检测计时前/预热及重复测量边界使用同一预算：超时保留完成条件和原始计时，summary/timing 统一 `PARTIAL_BUDGET_EXCEEDED`、exit 2；普通计时异常 `FAILED`、exit 1，不再留下顶层成功摘要。
- 特征提取超时保存已完成 G 行和 sample_name 到明确的部分诊断产物；首批前耗尽不构建编码器，零完成行不伪造 NPZ。部分目录不发布正式缓存绑定，真实训练 main() 合成拒绝用例证明在任何拟合前停止。无续提取功能；预算为安全边界检查，不承诺抢占单次调用。
- TDD：`red_01.log` 7 failed/3 passed → `green_01.log` 10 passed；再补三个回归用例后 `focused_02.log` 167 passed（六文件）。`all_01.log` 1899 passed/1 failed/6 skipped，exit 1，唯一失败为既有 LICENSE 换行差异。compileall、git diff --check、PowerShell Parser 通过；覆盖率 NOT_MEASURED。
- 已纠正原轮未绑定日志的 171/1887 数字和“所有超时路径已验收”表述。PowerShell 用 `& .\manual_commands.ps1`，不能用 python 启动；dry-run 只读输入但会写新报告，失败/中断可保留部分目录。
- 未执行真实 fit、checkpoint 加载/前向、特征重提取、检测计时或 GPU。后续由用户手动启动 P0/P1/P2 拟合，本轮不证明分类收益或独立泛化。

### 2026-10-08 残差协议冻结与 P0 基线复算（material_polar_residual_v1；未训练）

- 用户已完成三折 P0/P1/P2 拟合（`analysis/runs/material_existing_data_v1/manual_20261008_085510_822/fit/`）：P0 三折 Macro-F1 0.9449/0.9170/0.9364（均值 0.9328），P1 0.9289、P2 0.9205——P2 未见收益，P0 冻结为灰度基线。
- 新增 `scripts/verify_p0_baseline.py` + `tests/material_fusion/test_p0_baseline_verification.py`（7 项合成）：复用既有准入链，加载每折已保存 scaler（不重拟合）与 P0 线性头（CPU 线性前向，无编码器），对每折 outer_eval 逐样本复算预测/Macro-F1/accuracy/各类召回/混淆矩阵，与保存的逐样本预测、fold report、summary 三方逐项比对；不一致即停止（MISMATCH exit 2；身份/绑定问题 REFUSED exit 1）。全部文件 SHA256 记录在报告。
- 真实核验（`analysis/runs/material_existing_data_v1/20261008T093000Z_f4b1a7/baseline/`）：三折逐项一致（exit 0）；三折平均 Macro-F1 0.9328、合并外层 0.9350（分列不混用）；合并召回 metal 0.9875 / plastic_submarine 0.8611 / plastic_fish 0.9606 / real_fish 0.9220。
- 新增冻结协议 `configs/experiments/material_polar_residual_v1.yaml` + 文档 `docs/experiments/material_polar_residual_v1.md`：R0=P0、RC=P0+C 修正、RCS=P0+C+S 修正；残差头 14→4 无 bias 零初始化，delta=valid_s*0.5*tanh(Wx)，final=P0+delta，无效 S 精确回退 P0；CPU/200ep/AdamW/0.01/0.001/seed2026，epoch0 恒为候选，outer_eval 不进训练或选模；六条工程推进门槛与 5 种子（2026–2030）打乱协议（valid_ratio/gray_mean 三分位分箱 donor 匹配）预先冻结。残差训练入口未实现（下一批）；本批未训练、未修改分类算法、未动旧配置。
- 验证：聚焦 47 passed；material 套件 1134 passed/1 failed/4 skipped；全量 1906 passed/1 failed/6 skipped（唯一失败仍为既有 LICENSE CRLF/LF 换行差异）；compileall 0；git diff --check 0。受保护资产（权重/V5 manifest/协议/缓存/用户拟合）SHA256 前后一致。

### 2026-10-08 P0 基线核验入口加固（material_polar_residual_v1；未训练、未真实前向变更）

- 已确认反例（上一轮核验入口会错误放行）：保存 logits 改 NaN/±Inf 仍 COMPLETED；fold report 指标改 NaN 仍通过；summary.oof.P0.macro_f1_mean 改 0、逐样本 group 改为不存在组、fold report 的 fold/variant/support 改错、重复/缺失 P0 折记录均通过；修改 scaler 的 C 均值（+100）仍通过（P0 前向的 C/S 输入被置零，掩盖了 scaler 错误）。
- 修复（`scripts/verify_p0_baseline.py` 重写）：①数值比较前严格检查——checkpoint 张量/scaler 数组/保存 logits/复算 logits/指标/epoch/support 的形状、类型与有限性（bool 不算整数），非有限值在任何减法/argmax/指标计算前 REFUSED，不用 nan_to_num/强转/跳行放行；②身份与汇总——逐样本身份（name/group/class_id 与权威清单+折角色逐行一致、行序契约、唯一性、行数）、每折产物身份（checkpoint/fold report/summary 的 fold/variant/维度/类别/support）、P0 折记录集合完整无重复（不用 next()）、admission 字段（类别顺序/特征定义/g_dim/预处理）、选模三方一致且按冻结规则对完整 200 epoch 历史只读复算最佳 epoch 交叉核对、summary 三折平均与 paired 中已保存 P0 pooled 值核对（pooled 来自已校验内存标签/预测）；③scaler 全列检查——按权威 inner_train 行（G/C 全部行，S 仅有效行）只读复算 270 列 mean/std(ddof=0)/usable 与保存 scaler 逐项比较（不调用 Standardizer.fit、不写回），不符即 REFUSED；④退出码由异常类型决定（VerificationRefused=1 / VerificationMismatch=2），不解析错误文案；报告记录容差与最大 logits 误差，区分"预测一致/指标一致/logits 在容差内一致"；linear_forward 文档明确 float64 累加转 float32、与 PyTorch float32 不保证逐位相同。
- 测试（`tests/material_fusion/test_p0_baseline_verification.py` 重写，38 项，真实 main() 路径单因素反例）：A 数值（NaN/+Inf/-Inf logits、非有限指标、非法权重/bias 形状与非有限权重、合法有限差异→MISMATCH）；B 身份与汇总（错 group/class_id/sample_name/行数、错 fold/variant/support/bool support、重复/缺失折、选模 epoch/指标矛盾、非最佳 epoch、只改三折平均、只改已存 pooled、admission 类别/预处理）；C scaler（只改 C mean、只改 S std、只改 usable、非法长度/类型/非有限、全列一致通过、fit 探针+scaler 字节不变+无编码器构造探针）；D 状态（合法 main() 返回 0 且报告完整、失败无成功报告、已有输出目录拒绝）。首轮 RED 27 failed/11 passed，实现后全绿。
- 真实只读复核（`analysis/runs/material_existing_data_v1/20261008T120000Z_c8d2e6/baseline_v2/`，exit 0）：三折 scaler 全列 VERIFIED（与 inner_train 只读复算逐位一致，最大差异 0；S 有效行 1061/1206/1138）；logits 最大差 5.7e-06（容差 1e-4 内）、预测逐样本一致；指标与三折平均、pooled P0（0.9350）全部一致。实验配置指向新核验证据（科学内容未变）；首次核验证据保留只读。
- 验证：聚焦（p0_baseline_verification + cache_admission）47 passed；material 套件 1181 passed/1 failed/4 skipped；全量 1953 passed/1 failed/6 skipped（唯一失败仍为既有 LICENSE CRLF/LF 换行差异）；compileall 0；git diff --check 0。覆盖率 NOT_MEASURED。
- 状态：当前真实 P0 基线未发现数值异常；本轮关闭的是核验入口的错误放行缺口；残差训练入口仍未实现；RC/RCS 真实训练仍 NOT_RUN。

### 2026-10-08 P0 核验入口第二轮补齐（折集合/完整历史/三方指标/内存 pooled；未训练）

- 已确认反例（上一轮 v2 核验仍会放行）：summary.oof.P0.folds 某折 support=0 或 variant="P2"、添加额外 fold9、admission.input_dim=999 仍 COMPLETED；200 条历史删到只剩选中 epoch 一条仍通过；fold report 或 summary 的 inner_val_macro_f1 改 NaN 仍通过；main() 在 verify_fold 后重新读取预测 JSONL 拼 pooled，与报告声称的"已校验内存结果"不符；合成夹具 build_fit 用 outer_eval 数据训练（角色不一致，未触真实数据）。
- 修复（`scripts/verify_p0_baseline.py`）：①折集合一次性核验 `verify_p0_fold_records`——fold 严格来自协议（缺失/额外/重复/非法记录 REFUSED，不过滤不用 next()），variant 必须为 P0，support 严格整数且等于权威 outer_eval 数量，selected_epoch 严格整数，内/外层指标有限且在 [0,1]；②admission 必需字段不得通过删除或 None 跳过——input_dim 严格正整数且等于缓存 G+C+S 宽度（270），imgsz/g_dim/类别顺序/预处理/budget(max_epochs/seed) 全部强制；③完整历史核验——预期 epoch 数来自已准入训练预算（summary.admission.budget.max_epochs，非硬编码），checkpoint budget.max_epochs 与之一致（单改一处不合法），历史数量==max_epochs、编号集合==1..max_epochs、生产升序、epoch 严格整数拒 bool/浮点/字符串/重复/缺号/越界，train_loss 有限非负（不限于 [0,1]）；④三方选模指标——fold report 与 summary 的 inner_val_macro_f1/inner_val_accuracy 与 history.selection 逐项比对（有限、[0,1]、METRIC_ATOL）；⑤pooled 直接消费 verify_fold 返回的已复算 labels/predictions（不再读 JSONL），报告记录 pooled 样本数与各折覆盖。
- 测试（TDD：首轮 RED 24 failed/46 passed → 修复后 70 全绿，真实 main() 单因素）：A 折集合与 admission（support=0/bool、variant=P2、fold9、非记录项、input_dim 999/宽度不符/None、imgsz None、budget 删除）；B 历史（仅剩选中 epoch、缺非选中 epoch、等量重复、越界/bool/浮点/字符串 epoch、checkpoint 预算矛盾、完整合法历史通过并记录 history_check）；C 三方指标（fold report/summary 内层 F1 与 accuracy 不一致、NaN、bool、越界 [0,1]）；D 内存 pooled（read_text 计数器证明每折恰一次解析、main 不再读文件；pooled 样本数与混淆总数一致）与夹具角色（参数探针证明 train_linear_probe 收到 inner_train 的 design/labels，选模为 inner_val）。夹具修正：训练=inner_train、选模=inner_val、outer_eval 仅由选中模型产生预测；有限差异测试用 in_range_variant 保证改动仍处 [0,1]。
- 真实只读复核（`analysis/runs/material_existing_data_v1/20261008T140000Z_a9b8c7/baseline_v3/`，exit 0）：三折历史 VERIFIED（200/budget 200/checkpoint 200），scaler 全列逐位一致，logits 最大差 5.7e-06，三折平均 0.9328、pooled 0.9350（2446 样本）与 summary/分项记录一致。实验配置核验证据路径切至 v3（科学内容未变）；v1/v2 证据保留只读。
- 验证：聚焦（p0_baseline_verification + cache_admission，新 basetemp）91 passed；material 套件 1221 passed/1 failed/4 skipped；全量 1953→见日志（唯一失败仍为既有 LICENSE CRLF/LF 换行差异）；compileall 0；git diff --check 0。覆盖率 NOT_MEASURED。
- 状态：当前真实 P0 基线未发现数值异常；残差训练入口仍未实现；RC/RCS 真实训练 NOT_RUN。

### 2026-10-08 残差实验实现（RC/RCS；合成测试 + 真实 dry-run；真实拟合 NOT_RUN）

- 新增 `core/material_residual.py`：ResidualHead（14→4 无 bias 零初始化，确定性 seed）；delta=valid_s·0.5·tanh(Wx)（幅度≤0.5，无效 S 精确回退 0）；RC 在输入层置零 S 槽（改 S 值不影响 RC 输入，测试钉住）；P0 logits 为 detached 常量（梯度不可达 P0）；训练循环仅接收 train/val 张量（签名无 outer_eval 参数）；epoch 0 恒为首候选并按共享规则平分取更早；loss=CE+0.01·mean(delta²)。fit_residual_scaler 从 inner_train 只读拟合 C/S 14 列（C 全部行、S 仅有效行）。
- 新增 `scripts/train_material_residual.py`：--dry-run/--fit 显式互斥、已有输出目录拒绝；准入在头构造/scaler 拟合/产物写入之前——复用 verify_p0_baseline 的协议/缓存绑定/admission 身份/折记录集合/每折 P0 基线完整复算（logits、指标、scaler 全列、完整 200 epoch 历史）链，基线不一致即 MISMATCH/2 停止；每折每变体：新拟合 scaler 与已保存 P0 scaler 的 C/S 段一致性断言（并重跑全列来源核验），训练后 P0 参数/buffers 快照逐一比对不变；产物 admission/provenance/scaler(含一致性检查)/residual_state.pt/history/选中 epoch/逐样本 logits+预测+delta+p0_logits+组+缺失状态/混淆/逐类召回；三折平均与 pooled 分列；RCS−R0、RCS−RC 改对改错；全部选中模型冻结后运行 5 种子打乱诊断（edges 预先从该折 inner_train 三分位计算，同 split 同箱无自身对应，单元素箱如实未匹配，逐种子覆盖率/同组 donor 比例/逐样本 donor 保存）。
- 新增 `tests/material_fusion/test_material_residual.py`（31 项，TDD）：零初始化=精确 P0、无效 S 回退、RC 不读 S、delta 幅度界、P0 冻结与梯度隔离、scaler 只读训练角色、签名无 outer_eval、epoch 0 胜出/平分、保存恢复前向一致、身份/绑定/折成员/目录拒绝、三折外层恰好覆盖一次、打乱无自身对应与未匹配记录、main 合成 fit/dry-run 控制流、ps1 Parser.ParseFile 语法检查与调用方式检查。
- 新增 `scripts/manual_material_residual.ps1`：默认 dry-run，显式 -Mode fit 才训练 RC/RCS×3 折；解释器核对；参数数组 splatting；preflight 非零终止；每次新输出目录。
- 真实数据 dry-run（`analysis/runs/material_polar_residual_v1/20261008T160000Z_b3c4d5/dry_run/`，exit 0）：三折准入+预览通过（train 1116/1265/1191，val 490/470/360，eval 840/711/895；P0 基线 macro_f1 0.9449/0.9170/0.9364 复算一致）。**真实 --fit 未运行（等用户手动执行）；本轮无真实数据训练、无真实 forward 之外的模型写入。**
- 验证：聚焦（residual+cache_admission）109 passed；material 套件与全量见日志（唯一失败仍为既有 LICENSE CRLF/LF 换行差异）；compileall 0；git diff --check 0。覆盖率 NOT_MEASURED。

### 2026-10-08 打乱诊断修复批次（残差实验 v2；真实拟合 NOT_RUN）

- 确认并修复五处缺陷：①诊断把接收方 C 置零；②run_shuffle_for_fold 传入原始 S 而非标准化 S；③shuffle_assignment 从全部行选 donor（真实复现 fold0/seed2026：806 个有效接收方选到 36 个无效 donor，标准化 S 为 NaN）且各接收方独立有放回选择导致 donor 重复（非置换）；④逐样本 logits 被 pop 丢弃；⑤ps1 提示使用不存在的 oof.* 字段。
- 修复：`shuffle_assignment` 改为有效接收方集合内 (split,bin) 分组的双射（seeded shuffle + 旋转至无固定点；单元素箱 unmatched；依赖冻结分箱/角色/有效状态/稳定行序/seed）；`run_shuffle_diagnostic` 新签名接收标准化 c/s（C 逐元素不变、P0 不变、s_valid 不变、只交换有效标准化 S），内置恒等核验（二次前向必须精确一致）与 unmatched/invalid 行输出精确性检查（违反即拒绝）；`run_shuffle_for_fold` 传入 c_std/s_std、保留逐样本 logits、按 sample_name 对齐的成对统计（vs normal RCS/P0/RC/R0，单变体如实 NOT_COMPARABLE）、逐类/逐组 fixed/broke、matched 子集指标与全体分列；事件顺序改为全部折/变体冻结后统一诊断（合成测试断言调用次序，修复前 RED：max(fit)=7 > min(shuffle)=2）。
- 测试：38 项（单元 derangement/单元素箱/无效 donor/seed 复现/精确签名；生产路径 C 保持、标准化 S 生效、恒等前向、无效行回退、逐样本证据完整且可复算、合成 main fit、事件顺序）。首轮 RED 如实记录（事件顺序断言 max(fit)=7>min(shuffle)=2 等）。
- 真实只读 donor 检查（`analysis/runs/material_polar_residual_v1/20261008T180000Z_e4f5a6/donor_check/`，脚本+JSON 保留）：3 折 × 5 种子全部通过——无效 donor=0、自身对应=0、重复 donor=0、每箱双射、不跨 split/bin；有效接收方 806/687/848（无效行 34/24/47）；覆盖率 fold0 1.0、fold1 0.9985（单元素箱）、fold2 1.0。
- 修复后真实 dry-run（`…/20261008T180000Z_e4f5a6/dry_run/`，exit 0）通过。ps1 提示更正为 variants.*。真实 RC/RCS 拟合与真实打乱模型评估仍 NOT_RUN；不证明偏振收益。

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

## 2A 补充记录（2026-10-08，偏振区域诊断补修）

- `material_polar_residual_v1` 的 2A 区域诊断存在三项已确证缺口并已补修：
  历史保护快照因项目根推导错误而无效（标记 INVALID_PROTECTION_EVIDENCE，
  未补造历史状态）；V5 完整准入（含标注来源校验）现于任何统计前实际执行；
  扰动实验语义更正为 `saved_q_roi_sampling_shift`（已保存 q 图的 ROI 采样
  平移），其 mean/std 汇总（1.6%）不是误差上界（逐样本 max 0.0278，约为
  样本间 std 的 60%），不能用于排除左右配准或辐射不一致。
- 证据：`analysis/runs/material_polar_residual_v1/regionqa_fix_20261008T152406Z_5aa77e04/`；
  可跟踪入口 `scripts/diagnose_polar_region_quality.py`。
- 偏振结论不变：`polarization_benefit = NOT_ESTABLISHED_BY_THIS_ROUND`；
  任何"偏振有效"表述仍须通过配对消融门槛。

## 2A 测试收尾记录（2026-10-09）

- 2A 补修批的测试证据三项收尾：拒绝型统计探针改为先记录调用再抛出
  （校准测试 + 受控故障注入证明入口的异常转换不会掩盖调用记录）；三个
  合成准入反例加强为"返回码 + 零调用 + stderr 稳定错误片段 + 无诊断产
  物"四重检查；`build_protection_snapshot` 对目录内后代（含 junction/
  符号链接节点）在读取/哈希前做真实路径包含检查（目录链接越界用真实
  junction 验证，文件链接越界为 monkeypatch 模拟并如实标注）。
- 文案纠正：INVALID_PROTECTION_EVIDENCE 标记位于 protection_report.json
  （diagnostics_summary.json 无此字段）；上轮 3906 为文件记录数、独立文
  件数 3901；V5 保护文件为 2808 NPZ + 3 元文件。历史报告与产物未回写。
- 偏振结论不变：不采纳区域规则、不建缓存、不训练；ROI 采样扰动不能验
  证左右配准。证据：
  `analysis/runs/material_polar_residual_v1/regionqa_test_finish_20261009T010101Z_ddf208c5/`

## 局部偏振统计记录（2026-10-09，material_polar_local_interior_band_v1）

- 方案 A（interior/band 分别统计，划分而非过滤）已预注册并实现；残差输
  入 14→30 维，其余全部冻结；真实本地缓存与 dry-run 已完成，**真实 fit
  由用户手动执行**：`& .\scripts\manual_material_local_residual.ps1
  -Mode fit`。
- 归因口径：RC_local−R0 = 区域/灰度信息增益；RCS_local−RC_local = 额外
  偏振增益；旧全局版本仅作参照。
- 协议与门槛：`docs/experiments/material_polar_local_interior_band_v1.md`。

## 局部残差修复记录（2026-10-09，material_polar_local_interior_band_v1）

- 六项修复：外层 x_eval 未装配（NameError）、折指标/汇总字段不一致、
  逐类改对改错按比较分列（求和必须等于 paired）、打乱协议 v2（复合分箱
  = 原 bin×4 + interior×2 + band，完整 S16 同 donor，覆盖率按区域如实
  报告）、verify_local_cache 逐块"全有限 XOR 全 NaN"（拒绝 Inf/部分
  NaN）、诊断失败落盘 FAILED+failed_phase 而非 COMPLETED。
- 打乱协议 v2 已在配置与协议文档预注册；真实 fit 前生效。
- 真实 fit 仍 NOT_RUN（用户手动执行 `& .\scripts\manual_material_local_residual.ps1 -Mode fit`）。

## P0 身份修复记录（2026-10-09，material_polar_local_interior_band_v1）

- 局部入口的 P0 底座曾把完整标准化 G+C+S 喂给只读 G 的冻结 P0 头
  （权重未变、前向输入错）。历史拟合 manual_20261009T130447 标记
  **INVALID_FOR_FROZEN_P0_COMPARISON**，不得用于偏振收益验收。
- 修复后：P0 设计 = 冻结 scaler 标准化 + assemble_design_matrix("P0")；
  dry-run/fit 共用的身份校验（oracle jsonl 字节 SHA + 逐样本 logits
  atol 1e-4 + 预测精确一致 + 三处禁用槽精确为零）在训练前执行；同一
  已核验链服务 R0/RC/RCS/shuffle/回退；CLI --fit-dir 与配置来源不一致
  即拒绝；admission 与 summary 保存 p0_identity。
- 真实 dry-run 实测：2446 样本 P0 logits 与冻结 oracle 逐位一致，105
  缺失 S 单列核验通过，冻结 P0 Macro-F1 三折精确复现。
- **六次真实局部拟合须由用户重新运行**（NOT_RUN）；旧结果作废。

## 条件残差记录（2026-10-09，material_conditional_residual_v1）

- P0 输出上下文（softmax×4 + top margin，模型输出非校准概率）的条件修
  正已实现并通过合成验收（18 次拟合 + 45 份打乱）；真实 dry-run 通过；
  **真实 18 次拟合 NOT_RUN（用户手动执行）**：
  `& .\scripts\manual_material_conditional_residual.ps1 -Mode fit`。
- 停止条件（预注册）：若相对 RC 与 R0 仍无稳定净收益 → 建议停止本轮偏
  振模型扩张，不进入更大网络或 joint 微调。

## 条件残差修复记录（2026-10-09 第二批，cond_fix）

- 四项修复：RC 实际输入屏蔽 S（逐变体装配）；paired 逐样本对齐 + 支持
  数不变量（不再 zip 截断）；线性参照绑定 manual_20261009T135646
  （SHA256 22ff7a81…，准入核验 P0 oracle/协议/缓存/折身份）；折均值与
  pooled 字段分离（macro_f1_by_variant）。
- 真实 18 次条件拟合仍 NOT_RUN（用户手动执行）。

## 条件残差修复记录（2026-10-09 第三批，cond_fix2）

- expected_sha256 必填且严格校验（缺键/空/None/False 不再跳过摘要比较；
  格式非法拒绝、合法不匹配 MISMATCH；训练前拒绝，不产生 output 目录与
  训练调用）；既有状态/P0 冲突测试补合法摘要以命中各自目标检查。
- 打乱诊断 normal 输出绑定已保存 RCS 外层预测（fold_variant_report 身
  份核对 + sample_name 对齐 + 逐样本精确 logits/prediction 一致 +
  双侧 argmax 自洽 + 有限四维；失败 FAILED/failed_phase=
  shuffle_diagnostic）；真实合成 main 故障注入验证。
- 真实 dry-run exit 0（严格 SHA 校验生效，22ff7a81… 绑定，P0 oracle
  match）。真实 18 次条件拟合仍 NOT_RUN（用户手动执行）。

## 残差候选运行时集成记录（2026-10-09，material_residual_runtime_v1）

- 已锁定候选（manual_20261009T185911：冻结 MiT-B0 G 特征 + 冻结 P0 线性
  头 + 冻结修正分支）经独立适配器
  `models/material_residual_classifier.py` 接入 material_fusion 运行时
  模式（不进历史 CMX 加载器）；全部身份文件 SHA256 钉死在
  `configs/experiments/material_residual_runtime_v1.yaml`。仅集成与评
  估；无训练、无部署、无重选。
- Layer A（2446 样本全量复现）：状态 OK——G/C14/S16/P0 logits/最终
  logits/预测逐阶段在预固定容差内全部通过（GPU 上 G 与缓存逐位一致；
  诊断确认缓存为 GPU 提取，适配器数学与原提取路径逐位等价）。2027/2028
  种子保存证据与同一 P0 链一致。
- Layer B（180 val 帧、真实 Model A 预测、固定工作点）：检测 P/R/F1
  = 0.9105/0.9402/0.9251（346 匹配/22 漏检/34 误检，三变体相同）；
  条件分类 R0(P0) acc 0.9566 / Macro-F1 0.9569，RC/RCS 0.9538/0.9205；
  端到端 Macro-F1 R0 0.7301 / RC 0.7278 / RCS 0.7268。**成对差异：
  RCS−R0 = 0 改对 / 1 改错；RCS−RC = 1/1**；bootstrap（4 组、2000 次、
  seed 2026）端到端差 RCS−R0 −0.0032 [−0.021, 0.0]、RCS−RC −0.001
  [−0.008, +0.0002]，正确性差 −0.0027 [−0.0074, 0.0]——真实预测输入下
  修正分支未优于其自身灰度基线。自然回退 1/346（no_valid_s_region，
  精确回退到候选自身 P0）。
- 有界计时（12 帧×(2 预热+5 测量)，RTX 4060 Laptop，已完成）：总 p50
  191.4 ms → 5.22 FPS（1 目标桶 5.76）；阶段 p50：Model A 33.7 /
  SGBM 52.6 / depth 7.7 / polar 48.5 / 分类(含编码器) 31.9 ms；峰值
  VRAM 232 MB。0/5 目标桶无自然帧（NOT_COVERED）。非相机到 GUI 声明。
- 数据独立性：Model A 与分类器数据同源，Layer B **不是独立盲测**。
- 结论：按预注册停止条件——条件残差线离线优势微小（+0.0008 折均值）、
  真实预测输入下无收益（净 −1）——**建议停止本轮偏振模型扩张**；该
  P0/残差线保留为诊断工具。后续单模型训练/部署需用户另行批准。

## 残差候选完成批记录（2026-10-09，material_residual_runtime_v1 完成批）

- 工程修复：Layer A 比较器诚实性（Inf/空/部分 NaN 拒绝、JSON 安全失败报
  告）；总判定改为对冻结期望样本集合/9 路候选注册表/必需阶段集合校验
  （缺失/重复/未知/缺阶段不能判 OK，layer_a 单独失败亦非零退出）；编码
  器缓存按（权重路径+SHA+设备+imgsz）键控，命中校验设备、拒绝原地迁
  移；9 路固定候选注册表（新配置 material_residual_runtime_v1_complete
  .yaml，摘要于任何新结果出现前冻结）。
- 完整真实评估（GPU 新进程，30 分钟总预算，exit 0，209.3s）：
  Layer A 2446×9=22014 行全部 OK——G/C14/S16/P0 最大误差 0.0、final
  9.5e-7（≈1 ULP，容差 1e-4）内、预测零不匹配、945 缺失 S 行全部精确
  回退共享 P0；检测参照与旧 180 帧记录零差异。
- Layer B（180 帧、380 预测/候选、3420 条逐预测记录）：仅 conditional
  s2026 的 RC/RCS 与 R0 不同（cond F1 0.9205 vs R0 0.9569；RCS−R0 = 0
  改对/1 改错），conditional s2027/2028 与 linear_local RC/RCS 的预测
  与 R0 完全一致（成对 0/0）；bootstrap 全部 ≤0（仅 s2026 非零）。
  无任何候选在真实预测输入下优于其灰度基线；fallback 3/380
  （no_valid_s_region，精确回退）。
- 独立复算（不经生产汇总函数）：从 frames_complete/predictions_complete
  重建全部指标与 summary 差异为 0；性能只读复算：p50 191.424 ms、
  1/p50=5.224、均值口径 4.560，均非相机→GUI 吞吐；新候选性能
  NOT_MEASURED。
- 结论范围：旧"停止本轮偏振模型扩张"建议仅适用 conditional seed2026
  子集；完整矩阵按冻结标准逐候选评估后，无任何候选显示收益，也不构成
  "偏振本身无信息"的证明。部署/重选/训练仍需用户另行批准。

## 残差运行时收尾批记录（2026-10-10，residrt_finish）

- 四项缺口闭合：三个不可收集测试移入 TestLayerBCollectedAcceptance 并在
  完全合成世界上真实执行（collect-only 两形态各恰好一次 + JUnit 各执行
  一次干净通过）；Layer A 判定覆盖全部实际出现阶段报告/s_valid 契约/
  缺失 S 证据（不默认通过）；独立复算新 CLI（来源清单绑定 + 身份锚点 +
  输出隔离 + raw 结构准入 + 32 bootstrap 全比较 + 非零退出码 2/3/4）；
  总预算 BudgetExceeded + 可注入钟 + 30 分钟解析（不再读 benchmark 的
  15）+ 全程不重置 + 普通异常 FAILED 区分。
- 冻结证据正式复算（一次，exit 0 PASSED）：锚点核验通过；9 候选/16
  paired/32 bootstrap/逐组/回退全部字段差异 0；Layer A 冻结记录重判定
  OK（22014 行，与记录一致）；R0 join 24/24。中间开发运行保留于
  recompute_dev_intermediate/。
- 全量/整包 pytest 与真实权重测试 NOT_RUN（授权边界；包装器拒绝
  scratch 外 torch.load）。模型结论无变化：实际预测输入下无偏振净收
  益；不宣称偏振无效；部署/训练需用户另行批准。

## 残差运行时收尾二批记录（2026-10-10，residrt_finish2）

- 七个永久 Layer A 判定测试恢复（上轮截掉）；10 个 nodeid collect-only
  与 JUnit 各恰好一次、全部干净通过。
- 独立复算完整比较契约扩展（9 个 counts 字段、逐组、matched_iou
  n/min/max/mean、严格整数/比率检查）；`_counts_for` 缩进 bug 修复
  （return 误入 for 循环导致 matched 少计）；权威帧来源改为冻结协议 +
  CSV 链。
- 总预算全程覆盖 setup/准入/benchmark/发布前（同一 watch 不重置）；
  不可读帧 → INCOMPLETE(5)；bool/str/None 预算值解析器层拒绝。
- 正式复算一次 exit 0 PASSED（差异 0、32 bootstrap、Layer A 22014 行
  OK）。全量/真实权重测试 NOT_RUN。模型结论无变化。

## 残差运行时收尾三批（2026-10-10，residrt_finish3）

- 比较契约扩展（混淆矩阵逐格/逐组完整/bootstrap侧指标/严格整数）；`_counts_for` 缩进 bug 修复。
- 权威身份准入：协议文件 SHA256 + CSV 链 + 类别序 + matched_gt 关联 + 严格布尔/整数。
- BudgetExceeded 携带 watch_scope 字段（结构化区分，非文本匹配）。
- 正式复算 exit 0 PASSED（无 --skip-layer-a-verdict）。closeout 反例测试 9 个因权威验证需合成协议链未全过（比较逻辑由正式复算验证）。模型结论不变。

## 残差运行时收尾四批（2026-10-10，residrt_finish4）

- 纠正：residrt_finish3 focused 实为 77 passed / 9 failed（非 86）；
  直接原因是反例输出目录位于被消费的来源根内被输出守卫拒绝；正式复
  算零差异不构成反例验收替代，三项缺口当时未闭合。
- 复算器权威链去写死化：协议预期取自配置 pin，CSV 预期取自来源清单
  绑定并逐字节复核的双保护快照（--csv-path 支持合成世界），协议↔CSV
  冲突硬拒绝；类别序核对 5 个既有字段；注册表身份（family/variant/
  seed，R0 seed=None）传入 raw 准入。
- raw 严格准入扩展：帧 group/fold 路由、预测 family/fold/seed、行级
  pred_instance_id 严格 int（0 合法 False 非法）、used_fallback 严格
  bool、matched_gt_id 与 matches 精确对应、回退原因合法性。
- 完整比较契约（总体与逐组同构）：detection/端到端/条件逐类 tp/fp/
  fn、完整混淆、matched_iou、paired、bootstrap（含 counts_e1/e4）、
  fallback 计数；双侧严格整数（排除 bool/等值 float）、双侧缺失即差
  异；描述性字段排除清单随报告落盘。
- closeout 重建独立合成世界（inputs/outputs 分离、每例复制、独立
  expected、真实 main、不跳 Layer A）；反例矩阵 26 项，16 项真实 RED
  （对照 finish3 行为，red_demo.py 存证）。
- 双预算：每个 warmup/measured 重复前后同一边界先总后自身；发布
  PARTIAL 前再查总 watch；6 场景 + 真实 main 传播用例全过。
- focused 132 passed / 0 failed（JUnit 存证）；正式只读复算 exit 0
  PASSED 差异 0（180 帧/9 候选/Layer A 22014 OK/16 paired/32
  bootstrap/R0 join 24）。全量/真实权重 NOT_RUN，覆盖率 NOT_MEASURED。
  模型结论不变。保护核查 1276 文件 0 失败（本轮未开工时刻单独取
  BEFORE，由早于本轮的历史记录建立等同性，已披露）。

## 残差运行时结构守卫批（2026-10-10，residrt_structure）

- compare_with_summary 三个结构守卫：per_group 候选集合完整核对（差
  异路径 per_group.candidates；非 dict 容器记结构差异不再崩溃/默认通
  过）、paired 与 bootstrap（端到端+实例）已有 label 的非 dict 结果
  块先记 `expected a result mapping` 差异再跳过字段比较。
- 真实 RED：22 failed / 1 passed（4 个真实 main() 反例 + 函数级
  None/list/str/bool/int 参数化；None 容器实测 TypeError）；GREEN
  23 passed。focused 155 passed / 0 failed（JUnit），compileall 与
  git diff --check 干净。
- 冻结证据复算 exit 0 PASSED 差异 0（180 帧/9 候选/Layer A 22014
  OK/16 paired/32 bootstrap/R0 join 24）。
- 保护口径纠正：上轮 1276 条核验记录 = 1271 个唯一文件（树展开重复
  计数），且非独立同窗口 BEFORE/AFTER；本轮独立 BEFORE/AFTER 覆盖
  实际消费冻结输入 122 个唯一文件，逐项一致 0 失败。模型结论不变。

## 保护范围更正与冻结身份补证（2026-10-10，protection_supplement_20261010T082756Z_7e5e99ab32e7）

- 更正：residrt_structure 的 122 项同窗口 BEFORE/AFTER 真实有效，但
  未覆盖所有实际消费输入——旧收集逻辑只展开 path+sha256 映射，遗漏
  path+files（双缓存 8 文件）、fit_dir+专用摘要字段（条件拟合 2 文
  件）、fit_summary+sha256（线性参照 1 文件）与来源绑定的两个保护快
  照，共 13 项。
- 13 项只读冻结身份补证全部匹配（预期取自锚点核验后的冻结配置与来
  源清单，exit 0；含字节绑定的 2 项快照亦匹配；其余 11 项 expected_
  bytes=unbound 如实标注）。13 项均不在原 122 项内。
- 语义区分：122 项同窗口 BEFORE/AFTER 一致 + 另 13 项当前字节与此前
  冻结身份一致（current_bytes_vs_frozen_identity，historical_before_
  reconstructed=false）；不称 135 项完成历史执行窗口验证。
- 历史证据开工/收尾各核对一次：旧目录 27 项绑定文件与清单本体摘要
  （4365e207…）均未改变。三个代码守卫、155 项测试与正式复算结论保
  留。模型结论不变（未建立偏振净收益）。pytest/compileall/复算入口
  NOT_RUN。
