# 水下双目偏振目标检测与测距系统

本项目使用左 0 度、右 90 度偏振相机，完成实例分割、双目测距和四类目标分类。当前数据准备、正式训练、独立测试集评测和 GUI 推理链路均已打通。

## 当前结论

- 运行时采用 `Model A（二值实例分割） -> SGBM -> 深度/偏振图 -> Model B（四类分类）`。
- Model B 的受控消融没有证明偏振输入优于灰度输入，因此默认使用 `gray` 模式和 `model_b-gray` 权重。
- Model A 在独立测试集上的 mask mAP50-95 为 `0.76861`；四类 baseline 为 `0.73232`。
- 默认完整流水线在 RTX 4060、真实校正图、2 个实例条件下为 `165.96 ms/帧`，约 `6.0 FPS`，尚不能宣称 30 FPS 实时。
- 当前没有距离真值数据，不能仅凭双目输出宣称测距精度。

机器可读的完整状态、数量和 SHA-256 见 `analysis/project/project_inventory_v2.json`，工程交接与剩余风险见 `HANDOVER.md`。

## 推理链路

1. 对输入左右帧做同步检查和立体校正；已校正回放可跳过重复校正。
2. Model A 在左灰度三通道副本上生成目标实例 mask。
3. 由实例 bbox 合并生成水平条带，每帧只对条带区域运行一次 OpenCV StereoSGBM（条带接近全图时回退单次全图匹配），所有实例复用同一稠密视差结果。
4. 深度使用实例内鲁棒中位视差；偏振差分使用逐像素可靠视差（视差无效或右图越界的像素偏振为 0，不再用固定视差填满目标）。
5. Model B 对实例裁剪批量分类（`predict_batch`，不支持时回退逐目标）。`gray` 输入为 `[gray, gray, gray]`，实验模式 `polar` 为 `[gray, polar, gray]`。
6. `process_frame_detailed` 输出各阶段耗时（诊断用，非吞吐承诺）和每实例偏振质量记录。

类别 id 固定如下，不要在同一数据集中改成另一套材料标签：

| id | 类别 |
|---:|---|
| 0 | `metal_submarine` |
| 1 | `plastic_submarine` |
| 2 | `plastic_fish` |
| 3 | `real_fish` |

## 数据与模型

| 资产 | 当前状态 |
|---|---|
| `datasets/Single`, `datasets/Mixed` | 原始左右图，保留 |
| `datasets/bd_image` | 标定图及 `stereo_calib.npz`，保留 |
| `datasets/Rectified_v2` | 正确 MATLAB 旋转约定生成，1125 对 |
| `datasets/bz_JSON_v2` | 1125 份迁移后标注，去除内嵌 base64 |
| `datasets/underwater_seg_v2` | 四类分割，2808 个多边形 |
| `datasets/underwater_seg_binary_v2` | Model A 二值分割数据 |
| `datasets/underwater_cls_gray_v2` | Model B 灰度分类裁剪，2808 张 |
| `datasets/underwater_cls_polar_v2` | Model B 偏振分类裁剪，2808 张 |
| `datasets/underwater_cls_fusion_v3` | Polar Fusion 数值 npz 数据集，2808 个样本（gray/signed_q/abs_q/valid/quality/class_id + manifest） |
| `datasets/underwater_cls_fusion_v4_band` | 同上，视差改为与在线推理一致的条带匹配（band_margin=20，训练主线默认数据） |
| `runs/train/run_20260913_initial/model_baseline` | 正式四类分割 baseline |
| `runs/train/run_20260913_initial/model_a` | 正式二值实例分割模型 |
| `runs/train/run_20260913_initial/model_b-gray` | 已验收的默认分类模型 |
| `runs/train/run_20260913_initial/model_b-polar` | 消融实验分类模型，不作为默认 |

分割数据按完整采集组隔离为 `train/val/test = 765/180/180`，对应 `17/4/4` 个采集组。同一连续采集组不会跨 split。

2026-09-14 的人工修正版调整了 83 份 JSON 中各 1 个轮廓，没有修改类别或目标数量。四类分割、二值分割及 gray/polar 分类数据均已重建，split 保持不变。候选批次 `run_20260914_manual_labels` 已完成训练和审查，但尚未切换正式配置；详情见 `analysis/data/manual_annotation_changes_20260914.json` 和 `analysis/runs/run_20260914_manual_labels/training_review.json`。

## 环境

推荐使用已经验证的 Python 3.11 / CUDA 13 环境：

```powershell
pip install -r requirements.txt
pip check
```

精确环境快照位于 `requirements-lock-cu130.txt`。只有导出 ONNX/TensorRT 时才安装：

```powershell
pip install -r requirements-export.txt
```

## 启动 GUI

```powershell
python -m gui.main_window
```

默认配置为 `configs/default.yaml`。相机输入未提供可测量时间戳时，界面会明确显示同步状态未知，不会把它伪装成零偏差。

## 正式评测

分割模型独立测试集评测：

```powershell
python scripts/eval_segmentation.py --device 0 --output analysis/runs/run_20260913_initial/segmentation_test.json
```

Model B 灰度/偏振配对消融：

```powershell
python scripts/eval_ablation.py --device 0 --output analysis/runs/run_20260913_initial/model_b_ablation_test.json
```

完整流水线速度评测：

```powershell
python scripts/eval_speed.py --mode pipeline --config configs/default.yaml --left "datasets/Rectified_v2/Mixed/Plastic submarine and Real fish/0 NTU/left/000.png" --right "datasets/Rectified_v2/Mixed/Plastic submarine and Real fish/0 NTU/right/000.png" --already-rectified --device 0 --warmup 5 --n 30 --output analysis/runs/run_20260913_initial/pipeline_speed_test.json
```

## 重建生成数据

现有 v2 数据已完成且经过验证，无需日常重复生成。确需重建时按以下顺序执行；带 `--clean` 的命令会替换对应生成目录。

```powershell
python scripts/rectify_stereo_dataset.py --calib datasets/bd_image/stereo_calib.npz --r-convention matlab
python scripts/prepare_yolo_dataset.py --clean
python scripts/prepare_binary_seg_dataset.py --clean
python scripts/prepare_cls_paired_datasets.py --clean
python scripts/prepare_cls_fusion_dataset.py --clean
```

`scripts/migrate_labelme_annotations.py` 专用于把旧错误校正坐标系中的标注迁移到 v2。其旧输入位于 `archives/datasets_original_20260912.zip`，不要直接对当前 v2 标注再次迁移。

迁移后的 JSON 保留 LabelMe 所需的 `imageData: null`，但不嵌入 base64 图像；`imagePath` 是从 JSON 到对应 `Rectified_v2` 左图的相对路径，可直接用 LabelMe 打开。

## 训练

当前主线是独立的三阶段总控 `scripts/train_pipeline.py`：固定串行顺序，每阶段独立 Python 子进程、同一 run id，前一阶段完成并通过产物检查后才启动下一阶段：

1. `model_a`：YOLO26 二值实例分割 → `runs/train/<run-id>/model_a/`
2. `gray_fusion`：与 Fusion 同数据、同预处理的四分类 Gray-only → `runs/train/<run-id>/gray_fusion/`
3. `fusion_freeze`：使用本轮 Gray 最佳权重冻结 Gray、只训练偏振增量和 gate → `runs/train/<run-id>/polar_fusion/freeze/`

```powershell
# 预检（只读：不创建训练目录或报告、不下载、不训练；--preflight 与 --preflight-only 等价）
python scripts/train_pipeline.py --run-id <run-id> --device 0 --preflight

# 正式三阶段训练
python scripts/train_pipeline.py --run-id <run-id> --device 0

# 有界全链路冒烟（Model A 1 epoch × 数据比例；Gray/Fusion 2 epoch × ≤3 batch）
python scripts/train_pipeline.py --run-id <smoke-run-id> --device 0 --smoke

# 继续尚未完成的阶段（不是完整断点续训；已存在输出的阶段会被拒绝，不会重跑）
python scripts/train_pipeline.py --run-id <run-id> --device 0 --stages gray_fusion fusion_freeze
```

各阶段参数可独立覆盖：`--model-a-epochs/--model-a-batch/--model-a-imgsz`（默认 100/8/640）、`--gray-epochs/--gray-batch/--gray-imgsz`（默认 30/32/224）、`--fusion-epochs/--fusion-batch/--fusion-imgsz`（默认 30/32/224）；Gray 与 Fusion 的 imgsz 必须一致。总控在任何子进程启动前检查：选中阶段输出冲突、基座存在且任务/架构正确（Model A 必须 `yolo26n-seg.pt`、Gray 必须 `yolo26n-cls.pt`）、分割数据为单类 `0: target`（四类 YAML 被拒；逐标签校验整数类 0、polygon 结构、有限坐标；无标签文件或空标签属于合法背景图，但每个 split 必须至少有一个有效 polygon）、V4 严格数据准入与非空指纹、GPU 可用性、参数合法性；仅选择 Fusion 时必须验证本轮 Gray 产物存在、兼容、非 smoke 且其数据指纹与当前 V4 一致。每阶段结束后核对产物（best/last 可加载、真实架构、单类/类别顺序、输入尺寸、数据来源、Macro-F1 优先选模证据、配置与 checkpoint 的 `dataset_fingerprint`/`limit_batches`/`smoke` 一致）而不是只看退出码。运行记录写入 `runs/train/<run-id>/pipeline_report.json`：版本化累计结构（`attempts` 逐次保留命令、起止时间、退出码、产物、来源与失败/中断状态；`stage_status`/`pipeline_complete`/`remaining_stages` 汇总），采用原子写入，损坏或 run-id 不一致的旧报告会被拒绝而不是覆盖；只跑部分阶段时不会报告完整流水线完成。Fusion 始终使用 `runs/train/<run-id>/gray_fusion/best.pt`，绝不回退历史 Gray 权重或 smoke Gray 权重；smoke 与正式产物通过记录在配置和 checkpoint 中的 `limit_batches`/`smoke` 严格区分，缺失或类型非法的字段不会按正式全量训练处理。

旧四阶段训练（`baseline`/`a`/`b-gray`/`b-polar`）已归档到 `scripts/legacy_four_stage/`（含 README、新旧路径映射与显式运行方式）；原 `scripts/train_models.py`、`scripts/train_all_models.py`、`scripts/train_seg.py` 现在是退役提示入口：只打印提示并以非零退出，不会训练，也不会静默转发到新三阶段。

run id 只能包含 ASCII 字母、数字、`.`、`_`、`-`，不能包含路径分隔符。训练固定 seed `2026`、确定性模式、AMP 和 early stopping。Windows 默认 `workers=0`，避免子进程重新加载 CUDA DLL 失败。开发默认基座为 YOLO26nano：分割 `yolo26n-seg.pt`、分类 `yolo26n-cls.pt`（见 `scripts/training_common.py` 的 `SEG_BASE`/`CLS_BASE`）；本地 `yolo26n.pt` 是检测模型，所有训练入口都会拒绝。已有 YOLOv8 权重、报告和配置仅作历史对照，正式运行时配置（`configs/default.yaml` 中 `model_b.path` 的 Gray 权重）保持不变。

**状态区分**：训练完成 ≠ 技术验证通过 ≠ 模型质量合格。总控的阶段产物检查只证明链路执行正确、产物技术合格；模型质量与部署由后续评审决定，冒烟截断指标不能作为合格依据。

### Gray-only（Fusion 对照）与真实链路验证

Polar Fusion 的公平对照是独立入口 `scripts/train_gray_fusion.py`：与 Fusion 使用相同数据（默认 `datasets/underwater_cls_fusion_v4_band`）、相同 split、相同类别顺序、相同直接缩放预处理（`FusionClsDataset` 的 gray 通道，无数据增强）和相同 `imgsz`（默认 224），训练整个 YOLO26 分类模型的 gray 通道；选模与 Fusion 一致（完整 val，Macro-F1 优先、accuracy 次优；test 不参与）。`--base` 必须是本地存在的 YOLO26 分类基座（如 `yolo26n-cls.pt`），检测模型 `yolo26n.pt` 与 YOLOv8 权重会被拒绝。注意：Ultralytics 分类基座加载时全部参数处于冻结状态（`requires_grad=False`），本入口在创建优化器前显式解冻全部参数（不重新初始化权重），是真正的全模型训练而非线性探测；`train_config.json` 记录 `parameters_total`/`parameters_trainable` 以便核验。

```powershell
python scripts/train_gray_fusion.py --device 0 --run-id <run-id>
```

输出在 `runs/train/<run-id>/gray_fusion/`（`best.pt`、`last.pt`、`train_config.json`、`metrics.json`），目录已存在时拒绝。checkpoint 是标准 Ultralytics 分类格式（4 类头、数据集类别顺序），可直接作为 Fusion 的 Gray 分支：

```powershell
python scripts/train_polar_fusion.py --device 0 --run-id <fusion-run-id> --phase freeze --gray-weights runs/train/<run-id>/gray_fusion/best.pt
```

smoke 与正式训练的区别：`--limit-batches` 为正数时截断每个 epoch 的 train/val batch 数并在 `train_config.json` 记录 `smoke: true`；截断运行的指标只能证明链路可执行，不能作为模型合格依据。正式训练使用 `--limit-batches 0`。

### Fusion 退化诊断（只读）

`scripts/diagnose_polar_fusion.py` 对单个已训练 Fusion checkpoint 做只读诊断（不训练、不做 backward、不按 test 分数选 checkpoint），解释 Fusion 相对 Gray 的退化：

```powershell
python scripts/diagnose_polar_fusion.py `
  --checkpoint runs/train/<run-id>/polar_fusion/freeze/best.pt `
  --gray-weights runs/train/<run-id>/gray_fusion/best.pt `
  --data datasets/underwater_cls_fusion_v4_band --split test `
  --device 0 --batch 32 --shuffle-seeds 2026 2027 2028 2029 2030 `
  --output-dir analysis/runs/<run-id>/diagnostics_<timestamp>
```

评估前强制校验：数据指纹与 Gray/Fusion 配置及 checkpoint 一致、类别顺序/imgsz/freeze/正式标记、Fusion 的 Gray 来源为本轮 Gray，且 Fusion 内 Gray 分支与独立 Gray 权重逐参数逐 buffer 一致（不一致拒绝）。条件包括：Gray-only、真实偏振 Fusion、两种强制无效回退（`polar_invalid` 与 `valid_ratio=0`；gate 必须精确为 0 且 final 与 gray logits 精确相等，否则诊断状态 FAILED）、以及全量 test 索引上的偏振对应关系打乱（每 seed 一个无固定点双射，polar 三通道与 quality 始终来自同一 donor，映射与 batch 划分无关；所有 seed 全部报告，不选最优）。输出 `summary.json`、`predictions.csv`、`changed_predictions.csv`、`summary.md` 与变化样本诊断拼图；输出目录已存在时拒绝。注意：打乱实验是破坏偏振-灰度对应关系的诊断证据，不构成因果证明；gate 是学习到的标量门控，不是校准置信度。

真实链路验证入口 `scripts/verify_yolo26_fusion.py` 在 GPU 上做有界冒烟（2 epoch × ≤3 batch/阶段）：Gray 训练 → 保存重载 → Fusion freeze（gray 冻结、delta/gate 学习、无效偏振严格回退）→ 权重接续；输出与 `verify_report.json` 保留在 `runs/train/<run-id>*` 下供审查。

Fusion 的 `--init-from` 语义是权重接续：加载 checkpoint 权重后重建优化器，不恢复 optimizer/RNG 状态；省略 `--imgsz` 时继承 checkpoint 的训练尺寸，显式冲突尺寸会提前拒绝且不创建目录。Gray/Fusion checkpoint 依赖其记录的基座文件（如 `yolo26n-cls.pt`）与 Gray 权重文件仍在原路径，重载时不做自动下载或替换。

## 测试

```powershell
python -m pytest -q -W error -p no:cacheprovider
```

测试覆盖标注迁移、校正约定、数据隔离、SGBM、偏振通道、模型加载、消融统计、完整推理引擎和 GUI 状态语义。

## 项目结构

```text
configs/     运行、训练和评测默认配置
core/        校正、SGBM、偏振计算
gui/         PySide6 采集与推理界面
models/      Ultralytics 模型适配与导出
scripts/     数据准备、训练、评测工具
analysis/    按 calibration、data、project 和训练 run 分类的报告
datasets/    原始数据、标定数据和 v2 生成数据
runs/        正式训练与评测产物
archives/    经哈希验证的原始/旧数据归档
tests/       自动化测试
```

## 尚未完成

- 使用已知距离目标完成 `0.5/0.75/1.0/1.5/2.0 m` 深度误差评测。
- 用真实双相机连续运行 30 分钟，验证掉帧、同步、显存和异常恢复。
- 为实时部署导出 TensorRT，并优化或替换当前 CPU SGBM；达到目标前保持“非实时”表述。
- 补充 Mixed 场景及 `real_fish` 数据，后者是当前四类 baseline 的主要短板。
