"""Normalize LabelMe metadata without changing annotation geometry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_corresponding_image(
    json_path: Path,
    annotation_root: Path,
    image_root: Path,
) -> Path | None:
    relative_group = json_path.parent.relative_to(annotation_root)
    image_dir = image_root / relative_group / "left"
    matches = [
        image_dir / f"{json_path.stem}{suffix}"
        for suffix in IMAGE_SUFFIXES
        if (image_dir / f"{json_path.stem}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple corresponding images for {json_path}: {matches}")
    return matches[0] if matches else None


def normalize_annotation_tree(
    annotation_root: Path,
    image_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove embedded images and repair cross-root paths in existing JSON files."""
    annotation_root = annotation_root.resolve()
    image_root = image_root.resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"annotation root not found: {annotation_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root not found: {image_root}")

    summary = {
        "json_files": 0,
        "changed_files": 0,
        "embedded_image_data_removed": 0,
        "missing_images": 0,
    }
    for json_path in sorted(annotation_root.rglob("*.json")):
        summary["json_files"] += 1
        image_path = find_corresponding_image(json_path, annotation_root, image_root)
        if image_path is None:
            summary["missing_images"] += 1
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        relative_image = Path(
            os.path.relpath(image_path, start=json_path.parent)
        ).as_posix()
        embedded = data.get("imageData") is not None
        changed = embedded or data.get("imageData", object()) is not None
        changed = changed or data.get("imagePath") != relative_image
        if not changed:
            continue

        summary["changed_files"] += 1
        if embedded:
            summary["embedded_image_data_removed"] += 1
        if dry_run:
            continue

        data["imagePath"] = relative_image
        data["imageData"] = None
        temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(json_path)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/bz_JSON_v2"),
        help="Existing LabelMe annotation root.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("datasets/Rectified_v2"),
        help="Matching rectified image root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = normalize_annotation_tree(
        args.annotations,
        args.image_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["missing_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
