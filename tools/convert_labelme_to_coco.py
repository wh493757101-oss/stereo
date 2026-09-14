"""
LabelMe JSON -> COCO Instance Segmentation Format Converter

将 LabelMe polygon 标注转换为 COCO 实例分割格式，用于 Mask2Former 训练。

输入: LabelMe JSON 文件 (polygon 格式)
输出: COCO 格式 JSON 文件

用法:
    python tools/convert_labelme_to_coco.py \
        --input_dir datasets/PreparedSingle/labelme_json \
        --image_dir datasets/PreparedSingle/left \
        --output_dir datasets/coco_material \
        --split_ratio 0.8
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# 材质类别定义
MATERIAL_CATEGORIES = [
    {"id": 1, "name": "metal", "supercategory": "material"},
    {"id": 2, "name": "pvc", "supercategory": "material"},
    {"id": 3, "name": "stone", "supercategory": "material"},
]


def polygon_to_mask(points: list[list[float]], height: int, width: int) -> np.ndarray:
    """将 polygon 点转换为二进制 mask"""
    from skimage import draw

    points_array = np.array(points)
    rr, cc = draw.polygon(points_array[:, 1], points_array[:, 0], shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """将二进制 mask 转换为 RLE 格式 (COCO 格式)"""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """将二进制 mask 转换为 polygon 格式"""
    from skimage import measure

    contours = measure.find_contours(mask, 0.5)
    if len(contours) == 0:
        return []

    # 取最大的轮廓
    contour = max(contours, key=len)
    # 转换为 [x, y] 格式
    polygon = [[float(p[1]), float(p[0])] for p in contour]
    return polygon


def compute_bbox(points: list[list[float]]) -> list[float]:
    """计算 polygon 的 bounding box [x, y, width, height]"""
    points_array = np.array(points)
    x_min = float(points_array[:, 0].min())
    y_min = float(points_array[:, 1].min())
    x_max = float(points_array[:, 0].max())
    y_max = float(points_array[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_labelme_to_coco(
    labelme_dir: str,
    image_dir: str,
    output_dir: str,
    split_ratio: float = 0.8,
    categories: list[dict] | None = None,
) -> None:
    """
    将 LabelMe 标注转换为 COCO 格式

    Args:
        labelme_dir: LabelMe JSON 文件目录
        image_dir: 图像文件目录
        output_dir: 输出目录
        split_ratio: 训练集比例
        categories: 类别定义
    """
    if categories is None:
        categories = MATERIAL_CATEGORIES

    # 创建类别名称到 ID 的映射
    name_to_id = {cat["name"]: cat["id"] for cat in categories}

    # 获取所有 LabelMe JSON 文件
    labelme_files = sorted(Path(labelme_dir).glob("*.json"))
    print(f"Found {len(labelme_files)} LabelMe files")

    # 随机划分训练集和验证集
    np.random.seed(42)
    indices = np.random.permutation(len(labelme_files))
    train_count = int(len(labelme_files) * split_ratio)
    train_indices = set(indices[:train_count])

    # 初始化 COCO 格式
    def create_coco_structure() -> dict:
        return {
            "images": [],
            "annotations": [],
            "categories": categories,
        }

    train_coco = create_coco_structure()
    val_coco = create_coco_structure()

    annotation_id = 1

    for idx, labelme_file in enumerate(labelme_files):
        is_train = idx in train_indices
        coco = train_coco if is_train else val_coco

        # 读取 LabelMe JSON
        with open(labelme_file, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)

        # 获取对应的图像文件
        image_filename = labelme_file.stem + ".png"
        image_path = Path(image_dir) / image_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # 读取图像尺寸
        with Image.open(image_path) as img:
            width, height = img.size

        # 添加图像信息
        image_id = idx + 1
        coco["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
        })

        # 处理每个 shape (polygon)
        for shape in labelme_data.get("shapes", []):
            label = shape["label"]
            points = shape["points"]

            # 获取类别 ID
            if label not in name_to_id:
                print(f"Warning: Unknown label '{label}' in {labelme_file}")
                continue
            category_id = name_to_id[label]

            # 计算 bounding box
            bbox = compute_bbox(points)

            # 生成 mask 并计算面积
            mask = polygon_to_mask(points, height, width)
            area = float(mask.sum())

            if area == 0:
                print(f"Warning: Empty mask for {labelme_file}")
                continue

            # 添加标注信息
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "segmentation": [points],  # polygon 格式
                "iscrowd": 0,
            }
            coco["annotations"].append(annotation)
            annotation_id += 1

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)

    # 保存 COCO JSON
    train_json = output_path / "annotations" / "instances_train.json"
    val_json = output_path / "annotations" / "instances_val.json"

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, indent=2)
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_coco, f, indent=2)

    print(f"\n转换完成:")
    print(f"  训练集: {len(train_coco['images'])} 图像, {len(train_coco['annotations'])} 标注")
    print(f"  验证集: {len(val_coco['images'])} 图像, {len(val_coco['annotations'])} 标注")
    print(f"  输出目录: {output_path}")


def copy_images_to_coco(
    image_dir: str,
    output_dir: str,
    labelme_dir: str,
    split_ratio: float = 0.8,
) -> None:
    """将图像复制到 COCO 目录结构"""
    import shutil

    labelme_files = sorted(Path(labelme_dir).glob("*.json"))

    np.random.seed(42)
    indices = np.random.permutation(len(labelme_files))
    train_count = int(len(labelme_files) * split_ratio)
    train_indices = set(indices[:train_count])

    for idx, labelme_file in enumerate(labelme_files):
        is_train = idx in train_indices
        split = "train" if is_train else "val"

        image_filename = labelme_file.stem + ".png"
        src_path = Path(image_dir) / image_filename
        dst_path = Path(output_dir) / "images" / split / image_filename

        if src_path.exists():
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe to COCO format")
    parser.add_argument("--input_dir", type=str, required=True, help="LabelMe JSON directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--copy_images", action="store_true", help="Copy images to output directory")

    args = parser.parse_args()

    convert_labelme_to_coco(
        args.input_dir,
        args.image_dir,
        args.output_dir,
        args.split_ratio,
    )

    if args.copy_images:
        copy_images_to_coco(
            args.image_dir,
            args.output_dir,
            args.input_dir,
            args.split_ratio,
        )


if __name__ == "__main__":
    main()
