"""Teleoperation leader backed by a YAM arm with an i2rt teaching handle.

The leader is a second YAM arm in gravity compensation (moved by hand) whose
gripper is replaced by a teaching handle: a passive encoder (``0x50E``) on the
same CAN bus reports the trigger. This mirrors the station's own teleop
(yam_arm/examples/minimum_gello ``YAMLeaderRobot.get_info``): joints are the
arm's measured positions, already in the follower's frame since both arms are
YAMs, and the gripper command is ``1 - trigger position`` (1 = open).

Published like ``gello_leader``: ``<name>_actions`` carries 6 joints plus the
gripper command.

The handle's two buttons [top, second] can stand in for the foot pedal: with
``buttons`` set (``LeaderConfig.buttons``), each press is published on
``<name>_buttons`` in the key listener's message format, and the DAgger loop
reads it like a pedal press.
"""

import time

import numpy as np

from deploy.robot.node import Node


class HandleButtons:
    """Teaching-handle button presses -> keys, on the press edge only.

    Button polarity varies between handle units (some idle high, some idle low),
    so the idle level is learned from the first read: don't hold a button while
    the leader starts. A press only counts after the button was released, and
    presses of the same button closer than ``debounce_s`` are dropped.
    """

    def __init__(self, keys: tuple[str, ...], debounce_s: float = 0.3) -> None:
        self.keys = tuple(keys)
        self.debounce_s = debounce_s
        self._idle: list[bool] | None = None
        self._down: list[bool] = []
        self._last_press: dict[int, float] = {}

    def update(self, raw, now: float) -> list[str]:
        raw = [bool(b) for b in raw]
        if self._idle is None or len(self._idle) != len(raw):
            self._idle, self._down = raw, [False] * len(raw)
            return []
        pressed = []
        for i, (level, idle) in enumerate(zip(raw, self._idle)):
            down = level != idle
            if down and not self._down[i] and i < len(self.keys) and self.keys[i]:
                if now - self._last_press.get(i, float("-inf")) >= self.debounce_s:
                    self._last_press[i] = now
                    pressed.append(self.keys[i])
            self._down[i] = down
        return pressed


class YamLeaderNode(Node):
    def __init__(
        self, name: str, control_rate: float, channel: str, buttons: tuple[str, ...] = ()
    ) -> None:
        super().__init__(name, control_rate)
        self.channel = channel
        self.leader_topic_name = f"{name}_actions"
        self.create_publisher(self.leader_topic_name)
        self.buttons = HandleButtons(buttons) if any(buttons) else None
        if self.buttons is not None:
            self.buttons_topic_name = f"{name}_buttons"
            self.create_publisher(self.buttons_topic_name)

    def read_position(self) -> np.ndarray:
        joints = np.asarray(self.robot.get_observations()["joint_pos"], dtype=float)
        handle = self.robot.motor_chain.get_same_bus_device_states()
        if handle is None:
            raise RuntimeError(f"teaching handle on {self.channel!r} has not reported yet")
        return np.concatenate([joints[:6], [1.0 - handle[0].position]])

    def initial_bootup(self) -> None:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType

        # zero_gravity_mode: gravity compensation only, so the operator can
        # move the arm freely; the leader never receives position commands.
        self.robot = get_yam_robot(
            channel=self.channel,
            gripper_type=GripperType.YAM_TEACHING_HANDLE,
            zero_gravity_mode=True,
        )
        # The handle's state arrives with the first background CAN read.
        deadline = time.monotonic() + 3.0
        while self.robot.motor_chain.get_same_bus_device_states() is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"teaching handle on {self.channel!r} not reporting; is it connected?"
                )
            time.sleep(0.01)
        self.publish(
            self.leader_topic_name, self.read_position(), extras={"type": "interp"}
        )
        if self.buttons is not None:  # learn the idle levels before the first tick
            self.buttons.update(
                self.robot.motor_chain.get_same_bus_device_states()[0].io_inputs,
                time.monotonic(),
            )

    def tick(self) -> None:
        self.publish(
            self.leader_topic_name, self.read_position(), extras={"type": "servo"}
        )
        if self.buttons is not None:
            handle = self.robot.motor_chain.get_same_bus_device_states()
            for key in self.buttons.update(handle[0].io_inputs, time.monotonic()):
                self.publish(
                    self.buttons_topic_name,
                    np.array([1], dtype=np.uint8),
                    extras={"key": key, "pressed": True, "source": "handle"},
                )

    def on_shutdown(self) -> None:
        robot = getattr(self, "robot", None)
        if robot is not None:
            robot.close()


def run(
    name: str, control_rate: float, channel: str, buttons: tuple[str, ...] = ()
) -> None:
    YamLeaderNode(name, control_rate, channel, buttons).run()
