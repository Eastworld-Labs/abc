# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "tyro"]
# ///
"""Convert LeRobot v3.0 bimanual YAM datasets into the training data layout.

Thin CLI over abc_minimal/export_lerobot.py; runs standalone via `uv run` (the
header above pins the only dependencies — no torch environment needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro

from abc_minimal.export_lerobot import ExportLerobotConfig, main

if __name__ == "__main__":
    main(tyro.cli(ExportLerobotConfig))
