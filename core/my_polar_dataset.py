import logging
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils import frame_utils


class MyPolarDataset(Dataset):
    """Joint dataset for Left/Right/Mask aligned augmentation."""

    def __init__(self, aug_params=None, root='datasets/MyPolarData', split='train'):
        self.root = root
        self.split = split
        self.aug_params = dict(aug_params or {})

        # 增强参数只作用于内存样本：left/right/disp/seg_mask 必须同步 crop/flip/resize。
        self.crop_size = tuple(self.aug_params.get('crop_size', [])) if 'crop_size' in self.aug_params else None
        self.safe_crop_size = tuple(self.aug_params.get('safe_crop_size', self.crop_size)) if self.crop_size is not None else None
        self.h_flip_prob = float(self.aug_params.get('joint_h_flip_prob', 0.5 if split == 'train' else 0.0))
        self.v_flip_prob = float(self.aug_params.get('joint_v_flip_prob', 0.0))
        self.prefer_object_crop = bool(self.aug_params.get('prefer_object_crop', split == 'train'))
        self.object_crop_tries = int(self.aug_params.get('object_crop_tries', 20))
        self.min_object_pixels = int(self.aug_params.get('min_object_pixels', 64))
        self.stereo_safe_margin_x = int(self.aug_params.get('stereo_safe_margin_x', 96))
        self.stereo_safe_margin_y = int(self.aug_params.get('stereo_safe_margin_y', 16))

        self.left_paths = sorted(glob(os.path.join(root, 'left', '*.png')))
        self.right_paths = sorted(glob(os.path.join(root, 'right', '*.png')))

        if len(self.left_paths) != len(self.right_paths):
            raise ValueError(
                f"Left/Right 图像数量不一致: left={len(self.left_paths)}, right={len(self.right_paths)}"
            )

        self.image_pairs = list(zip(self.left_paths, self.right_paths))
        stems = [Path(path).stem for path in self.left_paths]

        if split == 'train':
            self.disp_paths = self._collect_optional_paths(os.path.join(root, 'disp'), stems, ('.pfm', '.png'))
            self.seg_paths = self._collect_optional_paths(os.path.join(root, 'labels'), stems, ('.png',))
        else:
            self.disp_paths = [None] * len(stems)
            self.seg_paths = [None] * len(stems)

        logging.info("Loaded MyPolarDataset with %d samples from %s", len(self.image_pairs), root)

    def __len__(self):
        return len(self.image_pairs)

    @staticmethod
    def _collect_optional_paths(folder, stems, extensions):
        available = {}
        for ext in extensions:
            for path in glob(os.path.join(folder, f'*{ext}')):
                available[Path(path).stem] = path
        return [available.get(stem) for stem in stems]

    @staticmethod
    def _ensure_three_channels(img):
        if img.ndim == 2:
            return np.repeat(img[..., None], 3, axis=2)
        if img.shape[2] == 1:
            return np.repeat(img, 3, axis=2)
        return img[..., :3]

    @staticmethod
    def _pad_to_min_size(img, min_h, min_w, is_mask=False):
        h, w = img.shape[:2]
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)
        if pad_h == 0 and pad_w == 0:
            return img

        if img.ndim == 2:
            border_value = 0
        else:
            border_value = [0] * img.shape[2]
        border_type = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REPLICATE
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, border_type, value=border_value)

    def _sample_crop_origin(self, seg_mask, crop_h, crop_w):
        h, w = seg_mask.shape[:2]
        max_y = max(0, h - crop_h)
        max_x = max(0, w - crop_w)

        foreground = np.argwhere(seg_mask > 0)
        if self.prefer_object_crop and len(foreground) > 0:
            # 优先围绕标注前景采样，减少训练时裁到纯背景 patch 的概率。
            ys, xs = foreground[:, 0], foreground[:, 1]
            bbox_y_min = int(ys.min())
            bbox_y_max = int(ys.max())
            bbox_x_min = int(xs.min())
            bbox_x_max = int(xs.max())

            target_y_min = max(0, bbox_y_min - self.stereo_safe_margin_y)
            target_y_max = min(h - 1, bbox_y_max + self.stereo_safe_margin_y)
            target_x_min = max(0, bbox_x_min - self.stereo_safe_margin_x)
            target_x_max = min(w - 1, bbox_x_max + self.stereo_safe_margin_x)

            y0_low = max(0, target_y_max - crop_h + 1)
            y0_high = min(target_y_min, max_y)
            x0_low = max(0, target_x_max - crop_w + 1)
            x0_high = min(target_x_min, max_x)

            if y0_low <= y0_high and x0_low <= x0_high:
                for _ in range(self.object_crop_tries):
                    y0 = random.randint(y0_low, y0_high)
                    x0 = random.randint(x0_low, x0_high)
                    crop_mask = seg_mask[y0:y0 + crop_h, x0:x0 + crop_w]
                    if np.count_nonzero(crop_mask) >= self.min_object_pixels:
                        return y0, x0

            target_center_y = int((target_y_min + target_y_max) / 2)
            target_center_x = int((target_x_min + target_x_max) / 2)
            return (
                int(np.clip(target_center_y - crop_h // 2, 0, max_y)),
                int(np.clip(target_center_x - crop_w // 2, 0, max_x)),
            )

        return (
            random.randint(0, max_y) if max_y > 0 else 0,
            random.randint(0, max_x) if max_x > 0 else 0,
        )

    def _joint_augment(self, img1, img2, disp, seg_mask):
        if self.safe_crop_size is not None:
            # safe_crop 先保证物体和立体匹配边界，最后再按 crop_size 统一缩放。
            safe_h, safe_w = self.safe_crop_size
            img1 = self._pad_to_min_size(img1, safe_h, safe_w, is_mask=False)
            img2 = self._pad_to_min_size(img2, safe_h, safe_w, is_mask=False)
            disp = self._pad_to_min_size(disp, safe_h, safe_w, is_mask=False)
            seg_mask = self._pad_to_min_size(seg_mask, safe_h, safe_w, is_mask=True)

            y0, x0 = self._sample_crop_origin(seg_mask, safe_h, safe_w)
            img1 = img1[y0:y0 + safe_h, x0:x0 + safe_w]
            img2 = img2[y0:y0 + safe_h, x0:x0 + safe_w]
            disp = disp[y0:y0 + safe_h, x0:x0 + safe_w]
            seg_mask = seg_mask[y0:y0 + safe_h, x0:x0 + safe_w]

        if self.split == 'train' and random.random() < self.h_flip_prob:
            # 同步水平翻转所有模态，防止左/右图和标签空间错位。
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp[:, ::-1])
            seg_mask = np.ascontiguousarray(seg_mask[:, ::-1])

        if self.split == 'train' and random.random() < self.v_flip_prob:
            img1 = np.ascontiguousarray(img1[::-1, :])
            img2 = np.ascontiguousarray(img2[::-1, :])
            disp = np.ascontiguousarray(disp[::-1, :])
            seg_mask = np.ascontiguousarray(seg_mask[::-1, :])

        if self.crop_size is not None and self.safe_crop_size is not None and tuple(self.crop_size) != tuple(self.safe_crop_size):
            # RGB 用线性插值，视差/标签用最近邻，避免产生插值类别。
            final_h, final_w = self.crop_size
            img1 = cv2.resize(img1, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
            disp = cv2.resize(disp, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
            seg_mask = cv2.resize(seg_mask, (final_w, final_h), interpolation=cv2.INTER_NEAREST)

        return img1, img2, disp, seg_mask

    def __getitem__(self, index):
        # 每次取样重新读文件并做随机增强；不会改写原始 left/right/disp/label 文件。
        left_path, right_path = self.image_pairs[index]

        img1 = np.array(frame_utils.read_gen(left_path)).astype(np.uint8)
        img2 = np.array(frame_utils.read_gen(right_path)).astype(np.uint8)
        img1 = self._ensure_three_channels(img1)
        img2 = self._ensure_three_channels(img2)

        h, w = img1.shape[:2]

        disp_path = self.disp_paths[index]
        if disp_path is None:
            disp = np.zeros((h, w), dtype=np.float32)
        else:
            disp = np.array(frame_utils.read_gen(disp_path), dtype=np.float32)
            if disp.ndim == 3:
                disp = disp[:, :, 0]

        seg_path = self.seg_paths[index]
        if seg_path is None:
            seg_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            seg_mask = np.array(frame_utils.read_gen(seg_path), dtype=np.uint8)
            if seg_mask.ndim == 3:
                seg_mask = seg_mask[:, :, 0]

        img1, img2, disp, seg_mask = self._joint_augment(img1, img2, disp, seg_mask)

        img1 = torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1).float()
        img2 = torch.from_numpy(np.ascontiguousarray(img2)).permute(2, 0, 1).float()
        disp = torch.from_numpy(np.ascontiguousarray(disp)).float()
        seg_mask = torch.from_numpy(np.ascontiguousarray(seg_mask)).long()

        return img1, img2, disp, seg_mask
