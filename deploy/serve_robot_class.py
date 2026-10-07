"""Serve ABC-DiT to robot_class's bimanual YAM runner (inference/yam/run.py).

That runner speaks openpi's websocket protocol: ``observation.state`` and HWC
``observation.images.<cam>`` frames in, and for RTC an extra
``_rtc_prev_chunk_left_over`` key in and ``raw_actions`` out, with the RTC
horizon/delay fixed by the server's connect-time metadata. This wraps the DiT
deploy policy so ABC's own WebsocketPolicyServer answers it in-process; only
key names and image layout are translated.
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import tyro

from abc_minimal.config import ClipConfig, DiTConfig
from abc_minimal.policy import InferenceConfig, shared_inference_fields
from deploy.policy import Policy, PolicyConfig
from deploy.websocket_server import WebsocketPolicyServer

# robot_class policy image key (inference/yam/utils.py POLICY_IMAGE_KEY) -> ABC camera.
CAMERA_MAP = {"head_cam": "top", "wrist_left": "left", "wrist_right": "right"}

# Same left-first 14-D joint order robot_class validates against.
ACTION_NAMES = (
    *(f"left_joint_{i}" for i in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(1, 7)),
    "right_gripper",
)


@dataclass
class Args:
    policy: InferenceConfig = field(default_factory=InferenceConfig)
    """Checkpoint, prompt, sampler, and runtime knobs."""
    dit_model: DiTConfig = field(default_factory=DiTConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    port: int = 8000
    rtc: bool = False
    """Advertise RTC to the runner; start the runner with --rtc as well."""
    execution_horizon: int = 16
    """RTC: actions the runner executes from a chunk before requesting the next."""
    inference_delay: int = 6
    """RTC: control steps one request may take (x33 ms at 30 Hz). The actions
    executed meanwhile become the DiT's RTC prefix."""

    def dit_config(self) -> PolicyConfig:
        return PolicyConfig(
            **shared_inference_fields(self.policy),
            clip=self.clip,
            model=self.dit_model,
        )


class RobotClassPolicy:
    """Translate openpi-format YAM observations to the DiT deploy contract."""

    def __init__(self, policy: Policy, inference_delay: int) -> None:
        self._policy = policy
        self._inference_delay = inference_delay

    def infer(self, obs: dict, **_) -> dict:
        prev_chunk = obs.pop("_rtc_prev_chunk_left_over", None)
        obs.pop("_rtc_prev_chunk_left_over_state", None)

        images = {}
        for source, camera in CAMERA_MAP.items():
            key = f"observation.images.{source}"
            if key not in obs:
                raise KeyError(
                    f"missing {key!r}; the DiT needs head_cam and both wrist cams "
                    f"(got {sorted(k for k in obs if k.startswith('observation.images.'))})"
                )
            image = np.asarray(obs[key])
            images[camera] = image.transpose(2, 0, 1) if image.shape[-1] == 3 else image

        dit_obs = {"state": obs["observation.state"], "images": images}
        if "prompt" in obs:
            dit_obs["prompt"] = obs["prompt"]

        # The leftover's first `inference_delay` rows are the actions executed
        # while this request is in flight -- exactly ABC's RTC prefix.
        result = self._policy.infer(
            dit_obs,
            action_prefix=prev_chunk,
            prefix_length=self._inference_delay if prev_chunk is not None else None,
        )
        actions = result["actions"]
        return {"actions": actions, "raw_actions": actions}


def server_metadata(args: Args) -> dict:
    metadata = {
        "robot": "yam",
        "action_dim": len(ACTION_NAMES),
        "action_names": ACTION_NAMES,
        "cameras": tuple(CAMERA_MAP),
        "rtc_enabled": args.rtc,
    }
    if args.rtc:
        metadata["execution_horizon"] = args.execution_horizon
        metadata["inference_delay"] = args.inference_delay
    return metadata


def main(args: Args) -> None:
    level = logging.INFO if os.environ.get("DEPLOY_VERBOSE") else logging.WARNING
    logging.basicConfig(level=level, force=True)
    chunk_len = args.dit_model.chunk_length
    if args.rtc:
        if args.inference_delay < 1 or args.execution_horizon < 1:
            raise ValueError("--execution-horizon and --inference-delay must be >= 1")
        if args.execution_horizon + args.inference_delay > chunk_len:
            raise ValueError(
                f"execution_horizon + inference_delay must be <= chunk_length={chunk_len}"
            )
        # Warm the prefix-conditioned sampler too when --policy.fast-inference is on.
        args.policy.rtc_prefix_length = args.inference_delay

    policy = Policy(args.dit_config())
    print(
        f"[serve_robot_class] steps={args.policy.diffusion_steps} "
        f"fast={args.policy.fast_inference} chunk_len={policy.chunk_len} "
        + (
            f"rtc e={args.execution_horizon} d={args.inference_delay}"
            if args.rtc
            else "chunked"
        )
    )
    print(f"[serve_robot_class] serving on 0.0.0.0:{args.port}")
    WebsocketPolicyServer(
        policy=RobotClassPolicy(policy, args.inference_delay),
        host="0.0.0.0",
        port=args.port,
        metadata=server_metadata(args),
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
