# 数据归档

`datasets_original_20260912.zip` 是 v2 迁移前的只读数据快照，包含：

- 原始双目采集：`datasets/Single/`、`datasets/Mixed/`
- 标定数据：`datasets/bd_image/`
- 旧标定约定生成的校正图：`datasets/Rectified/`
- 含嵌入图像的原始 Labelme 标注：`datasets/bz_JSON/`

当前训练和推理不读取此压缩包。它只用于数据溯源和在必要时复核标注迁移；当前主链使用 `datasets/*_v2`。

归档信息：

- 文件大小：`3,996,732,140` bytes
- SHA-256：`FF502C634E80FAB68E9246FB0A5AA02915E24DF88DC4A54DDF3047D6C12EA256`
- 中央目录已通过 `tar -tf` 读取，顶层内容数量与迁移前目录一致。

人工标注修正回退点：

- `bz_JSON_v2_manual_20260914.zip`：83 份 JSON 人工修正完成后、移除 LabelMe 内嵌 `imageData` 前的完整标注快照；SHA-256 `F8B2E7652C0CB1DF8C1C32DFD9FF52A41DB3FE4016C09D35879C77A5F2B42B25`。
- `underwater_seg_v2_before_manual_20260914.zip`：人工修正传导前的四类 YOLO 标签与数据清单；SHA-256 `7BA81293EEBA1F325C58B497065F375FD7EA45FA536C99C20E771A780392A32F`。
- `runs_smoke_and_legacy_20260913.zip`：六个冒烟训练目录和错误嵌套的 `runs/segment`；SHA-256 `555FC7F0A47BC63EC0DFB6BC1D35D6D761025601724AACB60674E6B2D75BD097`。
