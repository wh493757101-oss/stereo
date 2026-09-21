"""RETIRED entry: the four-stage launcher was archived 2026-09-21.

The current mainline is the three-stage pipeline:
    python scripts/train_pipeline.py --run-id <run-id> --device 0

The archived implementation lives in:
    scripts/legacy_four_stage/train_all_models.py

This stub only prints this notice and exits non-zero: it never trains and
never forwards to the new pipeline.
"""

from __future__ import annotations

import sys

MESSAGE = (
    "scripts/train_all_models.py is retired (four-stage training archived "
    "2026-09-21).\n"
    "Current mainline (three stages): python scripts/train_pipeline.py "
    "--run-id <run-id> --device 0\n"
    "Archived four-stage implementation (explicit use only): "
    "scripts/legacy_four_stage/train_all_models.py\n"
)


def main(argv: list[str] | None = None) -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
