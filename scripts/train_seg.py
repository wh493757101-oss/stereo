"""Backward-compatible training entry point (shim).

The production path now uses scripts/train_models.py, which trains on
standard 3-channel data only; the experimental first-convolution channel
surgery has been removed.

Legacy mapping:
    --stage a  -> train_models.py --stage a
    --stage b  -> train_models.py --stage b-polar  (old polar3 Material classification)

Legacy args --data and --weights are still accepted (--weights is ignored
because all stages now initialize from the configured base checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_models as tm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated shim; see scripts/train_models.py")
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--run-id", required=True,
                        help="Run id grouping outputs under runs/train/<run-id>/.")
    parser.add_argument("--model", default=None, help="Ignored; base weights are per-stage.")
    parser.add_argument("--data", required=True, help="data.yaml or classification root.")
    parser.add_argument("--weights", default=None, help="Ignored (kept for compatibility).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default=None)
    parser.add_argument("--lr0", type=float, default=None, help="Ignored (kept for compatibility).")
    parser.add_argument("--in-channels", type=int, default=3, choices=[3],
                        help="Only 3-channel input is supported.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.weights:
        print("Note: --weights is ignored; stages initialize from their base checkpoint.",
              file=sys.stderr)
    stage = "a" if args.stage == "a" else "b-polar"
    return tm.main([
        "--stage", stage,
        "--data", args.data,
        "--run-id", args.run_id,
        "--device", args.device,
        *(["--epochs", str(args.epochs)] if args.epochs is not None else []),
        *(["--imgsz", str(args.imgsz)] if args.imgsz is not None else []),
        *(["--batch", str(args.batch)] if args.batch is not None else []),
        *(["--name", args.name] if args.name else []),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
