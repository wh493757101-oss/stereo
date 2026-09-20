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
