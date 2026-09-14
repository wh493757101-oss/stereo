from pathlib import Path

import cv2
import numpy as np

from core.utils.frame_utils import read_gen


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
MAP_EXTENSIONS = (".pfm", ".png", ".npy")


def collect_files_by_stem(folder, extensions):
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    files = {}
    for path in sorted(folder_path.iterdir()):
        if path.suffix.lower() in extensions:
            files[path.stem] = path
    return files


def require_matching_stems(reference_dir, **named_dirs):
    # 浠?reference_dir 涓轰富琛紝鍏跺畠鐩綍蹇呴』鍖呭惈鐩稿悓 stem锛岄槻姝?left/right/disp 瀵逛笉榻愩€?    reference_files = collect_files_by_stem(reference_dir, IMAGE_EXTENSIONS)
    if not reference_files:
        raise RuntimeError(f"No reference images found in {reference_dir}")

    matched = {}
    for name, (folder, extensions) in named_dirs.items():
        files = collect_files_by_stem(folder, extensions)
        missing = sorted(set(reference_files.keys()) - set(files.keys()))
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(f"Missing {name} files for stems: {preview}")
        matched[name] = files

    stems = sorted(reference_files.keys())
    return stems, reference_files, matched


def load_rgb_image(path):
    image = np.array(read_gen(str(path))).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image[..., :3]


def save_rgb_image(path, image_rgb):
    # 椤圭洰鍐呴儴澶у鐢?RGB锛孫penCV 鍐欐枃浠跺墠闇€瑕佽浆鎴?BGR銆?    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def load_map(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)

    array = np.array(read_gen(str(path)), dtype=np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def rgb_to_gray_float(image_rgb):
    if image_rgb.dtype != np.float32:
        image_rgb = image_rgb.astype(np.float32)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def normalize_map(array, valid_mask=None):
    # 鍙敤鏈夋晥鍍忕礌浼拌 min/max锛岄伩鍏嶆棤鏁堝尯鍩熸妸褰掍竴鍖栬寖鍥存媺鍋忋€?    array = np.asarray(array, dtype=np.float32)
    if valid_mask is None:
        mask = np.isfinite(array)
    else:
        mask = np.isfinite(array) & (valid_mask > 0)

    if not np.any(mask):
        return np.zeros_like(array, dtype=np.float32)

    values = array[mask]
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max - value_min < 1e-6:
        normalized = np.zeros_like(array, dtype=np.float32)
    else:
        normalized = (array - value_min) / (value_max - value_min)

    normalized[~np.isfinite(normalized)] = 0.0
    if valid_mask is not None:
        normalized *= (valid_mask > 0).astype(np.float32)
    return normalized.astype(np.float32)


def normalize_disparity(disp, valid_mask=None):
    return normalize_map(np.abs(disp).astype(np.float32), valid_mask=valid_mask)


def disparity_to_colormap(disp, valid_mask=None):
    disp_abs = np.abs(np.asarray(disp, dtype=np.float32))
    if valid_mask is not None and np.any(valid_mask > 0):
        visible = disp_abs[valid_mask > 0]
        value_min = float(visible.min())
        value_max = float(visible.max())
        scaled = np.zeros_like(disp_abs, dtype=np.uint8)
        if value_max - value_min >= 1e-6:
            scaled = ((disp_abs - value_min) / (value_max - value_min) * 255.0).clip(0, 255).astype(np.uint8)
        scaled[valid_mask <= 0] = 0
    else:
        scaled = cv2.normalize(disp_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def warp_right_to_left(right_rgb, disp, device):
    import torch
    import torch.nn.functional as F

    H, W = disp.shape[:2]

    right_t = torch.from_numpy(np.asarray(right_rgb, dtype=np.float32)).to(device)
    if right_t.dim() == 2:
        right_t = right_t.unsqueeze(-1)
    right_t = right_t.permute(2, 0, 1).unsqueeze(0)

    disp_t = torch.from_numpy(np.asarray(disp, dtype=np.float32)).to(device)
    disp_t = disp_t.unsqueeze(0).unsqueeze(0)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    shifted_x = xx + disp_t.squeeze(0).squeeze(0)
    x_norm = 2.0 * shifted_x / (W - 1) - 1.0
    y_norm = 2.0 * yy / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    right_warp = F.grid_sample(right_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    right_warp = right_warp.squeeze(0).permute(1, 2, 0).cpu().numpy()

    valid = (shifted_x >= 0) & (shifted_x < W) & (disp_t.squeeze(0).squeeze(0) > 0)
    geometry_valid_mask = valid.cpu().numpy().astype(np.float32)

    return right_warp, geometry_valid_mask
