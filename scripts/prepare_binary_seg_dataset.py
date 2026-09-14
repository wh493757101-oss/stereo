"""Build the one-class (binary) Model-A segmentation dataset from a v2 source.

Reads the leakage-free four-class v2 YOLO-seg dataset, rewrites every
nonempty label to a single class id ``0`` (empty negative labels are kept),
and reuses the source images via hard links where the filesystem supports
them (falling back to ``shutil.copy2``). Split assignments and the source
manifests are preserved verbatim so the binary dataset stays
group-isolated across train/val/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    output_stem: str
    split: str
    group_name: str
    left_path: str
    right_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collapse a YOLO-seg dataset to one 'target' class.",
    )
    parser.add_argument(
        "--source",
        default="datasets/underwater_seg_v2",
        help="Source four-class YOLO-seg dataset root.",
    )
    parser.add_argument(
        "--output",
        default="datasets/underwater_seg_binary_v2",
        help="Output binary YOLO-seg dataset root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Reuse source images via hard links (copy fallback) or always copy.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser.parse_args()


def ensure_disjoint_roots(a: Path, b: Path, a_label: str, b_label: str) -> None:
    ra, rb = a.resolve(), b.resolve()
    if ra == rb or rb.is_relative_to(ra) or ra.is_relative_to(rb):
        raise ValueError(
            f"{a_label} ({ra}) and {b_label} ({rb}) overlap; "
            "the output root must be outside the source root"
        )


def prepare_output_root(output_root: Path, clean: bool) -> None:
    resolved = output_root.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {resolved}; pass --clean to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, destination: Path, link_mode: str) -> str:
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def read_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_stem", "split", "group_name", "left_path", "right_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f"{manifest_path} is missing required columns: {', '.join(missing)}"
            )
        for line_index, row in enumerate(reader, start=2):
            output_stem = str(row["output_stem"]).strip()
            split = str(row["split"]).strip()
            if not output_stem or split not in SPLIT_NAMES:
                raise ValueError(
                    f"invalid output_stem/split at {manifest_path}:{line_index}"
                )
            if output_stem in seen:
                raise ValueError(
                    f"duplicate output_stem {output_stem!r} in {manifest_path}"
                )
            seen.add(output_stem)
            rows.append(
                ManifestRow(
                    output_stem=output_stem,
                    split=split,
                    group_name=str(row["group_name"]).strip(),
                    left_path=str(row["left_path"]).strip(),
                    right_path=str(row["right_path"]).strip(),
                )
            )
    if not rows:
        raise ValueError(f"pair manifest is empty: {manifest_path}")
    return rows


def collapse_label_line(line: str) -> str:
    values = line.split()
    if not values:
        return ""
    return " ".join(["0", *values[1:]])


def collapse_label_file(source_label: Path, target_label: Path) -> int:
    """Rewrite one label with every class id set to 0; returns object count."""
    if source_label.is_file():
        lines = source_label.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    kept = [collapse_label_line(line) for line in lines]
    kept = [line for line in kept if line]
    target_label.write_text(
        "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
    )
    return len(kept)


def find_source_image(images_dir: Path, split: str, output_stem: str) -> Path:
    split_dir = images_dir / split
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = split_dir / f"{output_stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no source image for {output_stem!r} in {split_dir}"
    )


def write_data_yaml(output_root: Path) -> None:
    content = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: target\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def build_binary_dataset(
    source_root: Path,
    output_root: Path,
    link_mode: str,
    clean: bool,
) -> dict:
    ensure_disjoint_roots(source_root, output_root, "source root", "output root")
    rows = read_manifest(source_root / "pair_manifest.csv")
    prepare_output_root(output_root, clean)

    for split in SPLIT_NAMES:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {split: 0 for split in SPLIT_NAMES}
    object_counts = {split: 0 for split in SPLIT_NAMES}
    negative_counts = {split: 0 for split in SPLIT_NAMES}
    link_counts = {"hardlink": 0, "copy": 0}

    for row in rows:
        source_image = find_source_image(
            source_root / "images", row.split, row.output_stem
        )
        target_image = (
            output_root
            / "images"
            / row.split
            / f"{row.output_stem}{source_image.suffix.lower()}"
        )
        link_counts[link_or_copy(source_image, target_image, link_mode)] += 1

        object_count = collapse_label_file(
            source_root / "labels" / row.split / f"{row.output_stem}.txt",
            output_root / "labels" / row.split / f"{row.output_stem}.txt",
        )
        split_counts[row.split] += 1
        object_counts[row.split] += object_count
        if object_count == 0:
            negative_counts[row.split] += 1

    shutil.copy2(
        source_root / "pair_manifest.csv", output_root / "pair_manifest.csv"
    )
    source_split_assignment = source_root / "split_assignment.csv"
    if source_split_assignment.is_file():
        shutil.copy2(
            source_split_assignment, output_root / "split_assignment.csv"
        )
    write_data_yaml(output_root)

    present_splits = {row.split for row in rows}
    summary = {
        "class_names": ["target"],
        "source_root": source_root.resolve().as_posix(),
        "link_mode": link_mode,
        "images_reused": {
            "hardlink": link_counts["hardlink"],
            "copy": link_counts["copy"],
        },
        "splits": {
            split: {
                "images": split_counts[split],
                "objects": object_counts[split],
                "negatives": negative_counts[split],
            }
            for split in SPLIT_NAMES
            if split in present_splits
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_binary_dataset(
        source_root=Path(args.source),
        output_root=Path(args.output),
        link_mode=args.link_mode,
        clean=args.clean,
    )
    print(f"Binary YOLO-seg dataset: {Path(args.output).resolve()}")
    for split, counts in summary["splits"].items():
        print(
            f"  {split}: images={counts['images']}, "
            f"objects={counts['objects']}, negatives={counts['negatives']}"
        )


if __name__ == "__main__":
    main()
