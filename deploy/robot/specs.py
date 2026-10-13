"""Translate a robot profile into launchable camera and arm nodes."""

from deploy.robot.cameras.config import CameraNodeConfig
from deploy.robot.followers.yam_follower_config import YamFollowerConfig
from deploy.robot.launch import ProcessSpec


def camera_specs(profile) -> list[ProcessSpec]:
    specs = []
    for name, camera in profile.cameras.items():
        config = CameraNodeConfig(
            name=name,
            control_rate=camera.fps,
            camera_serial=camera.serial,
            height=camera.height,
            width=camera.width,
            rgb_socket=camera.socket,
        )
        module = "decxin" if camera.camera_type == "decxin" else "realsense"
        specs.append(
            ProcessSpec(
                f"camera_{name}", f"deploy.robot.cameras.{module}", {"cfg": config}
            )
        )
    return specs


def _follower_spec(name, follower, *, quiet: bool) -> ProcessSpec:
    config = YamFollowerConfig(
        name=f"follower_{name}",
        control_rate=follower.control_rate,
        channel=follower.channel,
        leader_name=f"leader_{name}",
        gripper_type=follower.gripper_type,
    )
    return ProcessSpec(
        f"follower_{name}",
        "deploy.robot.followers.yam_follower",
        {"cfg": config},
        quiet=quiet,
    )


def follower_specs(profile, *, quiet: bool) -> list[ProcessSpec]:
    return [
        _follower_spec(name, robot.follower, quiet=quiet)
        for name, robot in profile.robots.items()
    ]


def leader_spec(
    process_name: str,
    node_name: str,
    leader,
    *,
    quiet: bool,
    force_feedback: bool = True,
) -> ProcessSpec:
    """Launch the leader node for this profile's GELLO kind; it publishes
    ``<node_name>_actions``."""
    if leader.kind == "yam":
        # Gravity-compensated arm; no force feedback.
        return ProcessSpec(
            process_name,
            "deploy.robot.leaders.yam_leader",
            {
                "name": node_name,
                "control_rate": leader.control_rate,
                "channel": leader.channel,
                "buttons": tuple(leader.buttons),
            },
            quiet=quiet,
        )
    if leader.kind != "dynamixel":
        raise ValueError(f"unknown leader kind {leader.kind!r}")
    return ProcessSpec(
        process_name,
        "deploy.robot.leaders.gello_leader",
        {
            "name": node_name,
            "control_rate": leader.control_rate,
            "device_name": leader.device_name,
            "servo_ids": leader.servo_ids,
            "joint_signs": leader.joint_signs,
            "force_feedback": force_feedback,
        },
        quiet=quiet,
    )


def teleop_specs(profile, *, quiet: bool) -> list[ProcessSpec]:
    specs = []
    for name, robot in profile.robots.items():
        specs.extend(
            [
                leader_spec(f"leader_{name}", f"leader_{name}", robot.leader, quiet=quiet),
                _follower_spec(name, robot.follower, quiet=quiet),
            ]
        )
    return specs
