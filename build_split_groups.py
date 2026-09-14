import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build split_groups.json for train_segmentation.py from rectify_manifest.csv."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to rectify_manifest.csv generated during dataset flattening.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write split_groups.json.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=8,
        help="How many consecutive frames inside the same source group share one split group id.",
    )
    return parser.parse_args()


def sort_key(row):
    source_stem = str(row.get("source_stem", ""))
    if source_stem.isdigit():
        return (0, int(source_stem))
    return (1, source_stem)


def main():
    args = parse_args()
    if args.block_size <= 0:
        raise RuntimeError("--block_size must be > 0")

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"No rows found in manifest: {manifest_path}")

    grouped_rows = defaultdict(list)
    for row in rows:
        output_stem = str(row.get("output_stem", "")).strip()
        group_name = str(row.get("group_name", "")).strip()
        if not output_stem or not group_name:
            raise RuntimeError(
                f"Manifest row missing output_stem/group_name: {row}"
            )
        grouped_rows[group_name].append(row)

    split_groups = {}
    summary = {}

    for group_name, group_rows in sorted(grouped_rows.items()):
        ordered = sorted(group_rows, key=sort_key)
        block_count = 0
        for index, row in enumerate(ordered):
            block_index = index // args.block_size
            split_groups[row["output_stem"]] = f"{group_name}_block{block_index:03d}"
            block_count = max(block_count, block_index + 1)
        summary[group_name] = {
            "samples": len(ordered),
            "blocks": block_count,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(split_groups, handle, ensure_ascii=False, indent=2)

    print(f"Saved split groups to: {output_path.resolve()}")
    print(f"Total samples: {len(split_groups)}")
    print(f"Block size: {args.block_size}")
    print("Group summary:")
    for group_name, item in summary.items():
        print(f"  {group_name}: samples={item['samples']} | blocks={item['blocks']}")


if __name__ == "__main__":
    main()
