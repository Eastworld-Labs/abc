# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-ros2-support", "tyro"]
# ///
"""Convert ROS2 rosbag2 MCAP YAM episodes into the training data layout.

Thin CLI over abc_minimal/export_ros2_mcap.py; runs standalone via `uv run`
(the header above pins the only dependencies, no torch environment needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro

from abc_minimal.export_ros2_mcap import ExportRos2McapConfig, main

if __name__ == "__main__":
    main(tyro.cli(ExportRos2McapConfig))
