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
3. 每帧只运行一次 OpenCV StereoSGBM，所有实例复用同一稠密视差结果。
4. 在实例 mask 内计算稳健视差和深度；所有有效实例合并后每帧只生成一次偏振特征图。
5. Model B 对实例裁剪分类。`gray` 输入为 `[gray, gray, gray]`，实验模式 `polar` 为 `[gray, polar, gray]`。

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
```

`scripts/migrate_labelme_annotations.py` 专用于把旧错误校正坐标系中的标注迁移到 v2。其旧输入位于 `archives/datasets_original_20260912.zip`，不要直接对当前 v2 标注再次迁移。

迁移后的 JSON 保留 LabelMe 所需的 `imageData: null`，但不嵌入 base64 图像；`imagePath` 是从 JSON 到对应 `Rectified_v2` 左图的相对路径，可直接用 LabelMe 打开。

## 训练

正式权重已经存在，按 run id 分组在 `runs/train/run_20260913_initial/` 下。重新训练时必须提供新的 run id，输出会写入 `runs/train/<run-id>/model_<stage>/`。推荐用总控脚本顺序训练全部四个模型：

```powershell
python scripts/train_all_models.py --device 0 --run-id <run-id>
```

总控脚本为每个阶段启动独立 Python 进程，按 `baseline`、`a`、`b-gray`、`b-polar` 顺序执行，任何阶段失败都会停止后续阶段。若中断后只需继续未完成阶段，可显式选择：

```powershell
python scripts/train_all_models.py --device 0 --run-id <run-id> --stages b-gray b-polar
```

也可以单独训练某个阶段：

```powershell
python scripts/train_models.py --stage baseline --device 0 --run-id <run-id>
python scripts/train_models.py --stage a --device 0 --run-id <run-id>
python scripts/train_models.py --stage b-gray --device 0 --run-id <run-id>
python scripts/train_models.py --stage b-polar --device 0 --run-id <run-id>
```

run id 只能包含 ASCII 字母、数字、`.`、`_`、`-`，不能包含路径分隔符。总控脚本会在训练前检查所有选中模型的目标目录；若目录已存在，会拒绝启动，避免生成含义不清的递增目录。

训练固定 seed `2026`、确定性模式、AMP 和 early stopping。Windows 默认 `workers=0`，避免子进程重新加载 CUDA DLL 失败。

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
