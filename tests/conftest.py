import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Leftover pytest basetemp directories can be locked on Windows and crash
# collection with PermissionError; they never contain tests.
collect_ignore_glob = ["_tmp_*"]
