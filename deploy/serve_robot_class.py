"""Serve ABC-DiT to robot_class's bimanual YAM runner (inference/yam/run.py).

That runner speaks openpi's websocket protocol: ``observation.state`` and HWC
``observation.images.<cam>`` frames in, and for RTC an extra
``_rtc_prev_chunk_left_over`` key in and ``raw_actions`` out, with the RTC
horizon/delay fixed by the server's connect-time metadata. This wraps the DiT
deploy policy so ABC's own WebsocketPolicyServer answers it in-process; only
key names and image layout are translated.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    trace_path: str | None = None
    """Optional JSONL file of observed joints and predicted actions for diagnosis."""
    trace_interval: float = 2.0
    """Minimum seconds between trace records; 0 records every request."""

    def dit_config(self) -> PolicyConfig:
        return PolicyConfig(
            **shared_inference_fields(self.policy),
            clip=self.clip,
            model=self.dit_model,
        )


class RobotClassPolicy:
    """Translate openpi-format YAM observations to the DiT deploy contract."""

    def __init__(
        self, policy: Policy, inference_delay: int, *,
        trace_path: str | None = None, trace_interval: float = 2.0,
    ) -> None:
        if trace_interval < 0:
            raise ValueError("--trace-interval must be >= 0")
        self._policy = policy
        self._inference_delay = inference_delay
        self._trace_path = Path(trace_path).expanduser() if trace_path else None
        self._trace_interval = trace_interval
        self._last_trace = float("-inf")
        self._requests = 0
        if self._trace_path is not None:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    def infer(self, obs: dict, **_) -> dict:
        started = time.perf_counter() if self._trace_path is not None else 0.0
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
        self._write_trace(dit_obs, actions, prev_chunk is not None, started)
        return {"actions": actions, "raw_actions": actions}

    def _write_trace(self, obs: dict, actions, rtc: bool, started: float) -> None:
        if self._trace_path is None:
            return
        self._requests += 1
        now = time.perf_counter()
        if now - self._last_trace < self._trace_interval:
            return
        self._last_trace = now
        state = np.asarray(obs["state"])
        actions = np.asarray(actions)
        record = {
            "request": self._requests,
            "time_unix": time.time(),
            "prompt": obs.get("prompt"),
            "adapter_inference_ms": (now - started) * 1000,
            "rtc_prefix_length": self._inference_delay if rtc else 0,
            "state": state.tolist(),
            "actions": actions.tolist(),
            "finite": bool(np.isfinite(actions).all()),
        }
        if state.shape == (14,) and actions.ndim == 2 and actions.shape[1] == 14:
            joint_indices = [*range(6), *range(7, 13)]
            for index in (0, 15, 29):
                if index < len(actions):
                    record[f"max_joint_delta_rad_at_{index}"] = float(
                        np.abs(actions[index, joint_indices] - state[joint_indices]).max()
                    )
        try:
            with self._trace_path.open("a") as output:
                output.write(json.dumps(record) + "\n")
        except OSError:
            logging.getLogger(__name__).warning("Could not write inference trace", exc_info=True)


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
    metadata["server_identity"] = server_identity(args)
    return metadata


def server_identity(args: Args) -> dict:
    """What a replay must match to reproduce this server's outputs. The DiT starts from
    zero noise, so given the same request it needs no seed."""
    import torch

    return {
        "family": "abc-dit",
        "checkpoint": str(args.policy.checkpoint_path),
        "diffusion_steps": args.policy.diffusion_steps,
        "fast_inference": args.policy.fast_inference,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "rtc": args.rtc,
        "inference_delay": args.inference_delay if args.rtc else None,
    }


def main(args: Args) -> None:
    if args.trace_interval < 0:
        raise ValueError("--trace-interval must be >= 0")
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
        policy=RobotClassPolicy(
            policy, args.inference_delay,
            trace_path=args.trace_path, trace_interval=args.trace_interval,
        ),
        host="0.0.0.0",
        port=args.port,
        metadata=server_metadata(args),
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
