"""Read-only hardware preflight; never enables motors or starts camera streams."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pyrealsense2 as rs

from deploy.robot.config import get_i2rt_config


def main() -> int:
    os.environ.setdefault("ROBOT_PROFILE", "yam007_config")
    profile = get_i2rt_config()
    failures = []
    links = json.loads(
        subprocess.run(
            ["ip", "-j", "-details", "link", "show"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    links = {link["ifname"]: link for link in links}
    for side, robot in profile.robots.items():
        channels = [robot.follower.channel]
        if robot.leader.kind == "yam":
            channels.append(robot.leader.channel)
        for name in channels:
            link = links.get(name)
            if link is None:
                failures.append(f"{side}: CAN interface {name} is missing")
                continue
            info = link.get("linkinfo", {})
            data = info.get("info_data", {})
            bitrate = data.get("bittiming", {}).get("bitrate")
            if info.get("info_kind") != "can":
                failures.append(f"{name}: interface is not CAN")
            if "UP" not in link.get("flags", []):
                failures.append(f"{name}: interface is DOWN")
            if bitrate != 1_000_000:
                failures.append(f"{name}: bitrate {bitrate}, expected 1000000")
            if data.get("state") != "ERROR-ACTIVE":
                failures.append(f"{name}: CAN state {data.get('state')}")
            print(
                f"CAN {name}: flags={link.get('flags')} bitrate={bitrate} state={data.get('state')}"
            )

    cameras = {
        d.get_info(rs.camera_info.serial_number): d
        for d in rs.context().query_devices()
    }
    for name, camera in profile.cameras.items():
        if camera.serial not in cameras:
            failures.append(
                f"Camera {name}: serial {camera.serial} is missing/inaccessible"
            )
        else:
            device = cameras[camera.serial]
            print(
                f"Camera {name}: {device.get_info(rs.camera_info.name)} ({camera.serial})"
            )
    if not shutil.which("candump"):
        failures.append(
            "can-utils is missing; run scripts/setup_local_dagger_system.sh"
        )
    rule_paths = [
        *Path("/etc/udev/rules.d").glob("*realsense*.rules"),
        *Path("/usr/lib/udev/rules.d").glob("*realsense*.rules"),
    ]
    if not rule_paths:
        failures.append(
            "RealSense udev rules are missing; run scripts/setup_local_dagger_system.sh"
        )
    if failures:
        print("\nHardware preflight FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("Use scripts/run_local_dagger.sh --dry-run to test without hardware.")
        return 2
    print(
        "Hardware preflight passed. Camera streaming and teleop still need an operator check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
