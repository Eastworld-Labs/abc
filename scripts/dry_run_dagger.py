"""Hardware-free integration check of the real policy, DAgger loop and recorder.

Runs base/fine-tuned checkpoints sequentially on CUDA. Synthetic camera,
follower and teaching-handle data are confined to a unique IPC directory.
Recordings are marked unusable for training and written under outputs/.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import socket
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro


@dataclass
class DryRunConfig:
    checkpoints: list[str] = field(
        default_factory=lambda: [
            "cache/abc_dit_xl_200k_model.pt",
            "checkpoints/unfold-10hr-data/30000_model.pt",
        ]
    )
    prompt: str = "unfold"
    profile: str = "yam007_config"
    diffusion_steps: int = 10
    rtc: bool = True
    prefix_length: int = 4
    inference_lead_steps: int = 14
    execute_chunk_dim: int = 16
    policy_chunks: int = 3
    intervention_seconds: float = 1.2
    output_dir: str = "outputs/dagger_dry_run"
    render_video: bool = True
    startup_timeout: float = 180.0


def _images(profile) -> dict[str, np.ndarray]:
    import cv2

    images = {}
    for index, (name, cam) in enumerate(profile.cameras.items()):
        y, x = np.indices((cam.height, cam.width))
        image = np.stack(
            ((x + index * 40) % 256, (y + 50) % 256, (x // 2 + y // 2 + 80) % 256),
            axis=-1,
        ).astype(np.uint8)
        cv2.putText(
            image,
            f"SYNTHETIC DRY RUN: {name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        images[name] = image
    return images


class SyntheticEnv:
    """A 14-D follower surrogate; never imports or opens hardware drivers."""

    def __init__(self, profile, shared, prompt: str):
        self.profile, self.shared, self.prompt = profile, shared, prompt
        self.images = _images(profile)
        self.init_q = np.concatenate([arm.init_q for arm in profile.robots.values()])
        self.commands = 0
        self.reset()

    def get_obs(self, state=None):
        if state is None:
            with self.shared.get_lock():
                state = np.asarray(self.shared[:], dtype=np.float32)
        return {
            "images": {k: v.transpose(2, 0, 1) for k, v in self.images.items()},
            "masks": {k: True for k in self.images},
            "state": np.asarray(state, dtype=np.float32).copy(),
            "prompt": self.prompt,
            "camera_timestamps": {k: time.perf_counter() for k in self.images},
        }

    def step_state(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid synthetic command: {action}")
        with self.shared.get_lock():
            self.shared[:] = action
        self.commands += 1
        return action.copy()

    def reset(self):
        with self.shared.get_lock():
            self.shared[:] = self.init_q
        return self.get_obs(), {}


def _sensor_process(profile, shared, ipc_dir, stop):
    import zmq
    from deploy.robot import communication as comms

    comms._IPC_DIR = ipc_dir
    context = zmq.Context()
    images = _images(profile)
    topics = [cam.socket for cam in profile.cameras.values()]
    topics += [f"follower_{s}_obs" for s in profile.robots]
    topics += [f"dagger_gello_{s}_actions" for s in profile.robots]
    publishers = {t: comms.create_publisher(context, t, linger=0) for t in topics}
    home = np.concatenate([a.init_q for a in profile.robots.values()])
    started = time.monotonic()
    try:
        while not stop.is_set():
            tick = time.monotonic()
            with shared.get_lock():
                q = np.asarray(shared[:], dtype=np.float32)
            for name, image in images.items():
                comms.publish(publishers[profile.cameras[name].socket], image)
            for index, side in enumerate(profile.robots):
                arm = q[index * 7 : (index + 1) * 7]
                comms.publish(
                    publishers[f"follower_{side}_obs"],
                    np.concatenate([arm, np.zeros(14), arm]).astype(np.float32),
                )
                leader = home[index * 7 : (index + 1) * 7].astype(np.float32).copy()
                leader[0] += 0.03 * np.sin(time.monotonic() - started)
                comms.publish(publishers[f"dagger_gello_{side}_actions"], leader)
            stop.wait(max(0.0, 1 / 30 - (time.monotonic() - tick)))
    finally:
        for publisher in publishers.values():
            publisher.close(linger=0)
        context.term()


def _recorder_process(profile, ipc_dir, output, checkpoint, cfg):
    from deploy.robot import communication as comms
    from deploy.robot.recorders.inference_recorder import InferenceRecorderNode

    comms._IPC_DIR = ipc_dir
    os.environ["DEPLOY_POST_VIDEO"] = "0"

    class SyntheticRecorder(InferenceRecorderNode):
        def _write_extra_attrs(self, file):
            super()._write_extra_attrs(file)
            file.attrs["synthetic_dry_run"] = True
            file.attrs["usable"] = False
            file.attrs["unusable"] = True

    SyntheticRecorder(
        "DryRunRecorder",
        200,
        str(output),
        profile,
        "synthetic_dagger",
        checkpoint_path=str(checkpoint),
        model_size="dit_xL",
        diffusion_steps=cfg.diffusion_steps,
        rtc=cfg.rtc,
        rtc_prefix_length=cfg.prefix_length,
        rtc_inference_lead_steps=cfg.inference_lead_steps,
        task_name=cfg.prompt,
        dagger=True,
    ).run()


def _server_process(checkpoint, prompt, cfg, port, ready):
    from abc_minimal.config import ClipConfig
    from abc_minimal.policy import InferenceConfig
    from deploy.serve_policy import create_policy
    from deploy.serve_policy_config import Args
    from deploy.websocket_server import WebsocketPolicyServer

    try:
        args = Args(
            policy=InferenceConfig(
                checkpoint_path=str(checkpoint),
                prompt=prompt,
                device="cuda",
                diffusion_steps=cfg.diffusion_steps,
                rtc_prefix_length=cfg.prefix_length if cfg.rtc else None,
            ),
            clip=ClipConfig(cache_dir=str(Path("cache/clip").resolve())),
            port=port,
        )
        policy = create_policy(args)
        ready.put({"chunk_len": policy.chunk_len, "action_dim": policy.action_dim})
        WebsocketPolicyServer(
            policy, host="127.0.0.1", port=port, metadata={"synthetic_dry_run": True}
        ).serve_forever()
    except Exception as exc:
        ready.put({"error": repr(exc)})
        raise


def _stop_process(process):
    if process.is_alive():
        process.terminate()
        process.join(timeout=15)
    if process.is_alive():
        process.kill()
    process.join(timeout=5)


def _run_one(cfg, checkpoint, output):
    import h5py
    import zmq
    from deploy.dagger_config import DaggerConfig
    from deploy.robot import communication as comms
    from deploy.robot.config import PROFILES
    from deploy.robot.gym.dagger_rollout import DaggerLoop, DaggerState

    profile = copy.deepcopy(PROFILES[cfg.profile])
    context = mp.get_context("spawn")
    shared = context.Array("f", 14)
    stop = context.Event()
    ready = context.Queue()
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "checkpoint": str(checkpoint),
        "prompt": cfg.prompt,
        "synthetic": True,
        "rtc": cfg.rtc,
    }
    processes = []
    loop = None
    key_pub = None
    key_context = None
    with tempfile.TemporaryDirectory(prefix="abc-dry-") as ipc_dir:
        previous_ipc = comms._IPC_DIR
        comms._IPC_DIR = ipc_dir
        env = SyntheticEnv(profile, shared, cfg.prompt)
        try:
            server = context.Process(
                target=_server_process, args=(checkpoint, cfg.prompt, cfg, port, ready)
            )
            server.start()
            processes.append(server)
            report["policy"] = ready.get(timeout=cfg.startup_timeout)
            if "error" in report["policy"]:
                raise RuntimeError(report["policy"]["error"])
            deadline = time.monotonic() + 10
            while True:
                try:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/healthz", timeout=1
                    ).close()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Policy server failed its health check")
                    time.sleep(0.1)
            print(f"[dry-run] Policy ready: {checkpoint}", flush=True)
            for target, args in [
                (_sensor_process, (profile, shared, ipc_dir, stop)),
                (_recorder_process, (profile, ipc_dir, output, checkpoint, cfg)),
            ]:
                process = context.Process(target=target, args=args)
                process.start()
                processes.append(process)
            args = DaggerConfig(
                prompt=cfg.prompt,
                remote_host="127.0.0.1",
                port=port,
                rtc=cfg.rtc,
                rtc_prefix_length=cfg.prefix_length,
                rtc_inference_lead_steps=cfg.inference_lead_steps,
                execute_chunk_dim=cfg.execute_chunk_dim,
                compress_images=True,
            ).dagger_rollout_config()
            loop = DaggerLoop(env, profile, args)
            # Only this synthetic runner bypasses the hardware first-action gate.
            loop._safety_confirmed = True
            key_context = zmq.Context()
            key_pub = comms.create_publisher(
                key_context, "KeyListener_key_presses", linger=0
            )
            time.sleep(1.0)

            def press(key):
                comms.publish(
                    key_pub,
                    np.ones(1, dtype=np.uint8),
                    {"key": key, "pressed": True, "source": "synthetic"},
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    keys = loop.pedals.poll()
                    if keys:
                        for received in keys:
                            loop._handle_key(received)
                        time.sleep(0.1)
                        return
                    time.sleep(0.01)
                raise RuntimeError(f"Dry-run key {key!r} was not received")

            press("a")
            assert loop.state == DaggerState.POLICY
            request_seconds = time.monotonic()
            for _ in range(cfg.execute_chunk_dim * cfg.policy_chunks):
                loop._policy_step()
            report["policy_seconds"] = round(time.monotonic() - request_seconds, 3)
            press("a")
            assert loop.state == DaggerState.INTERVENTION
            deadline = time.monotonic() + cfg.intervention_seconds
            while time.monotonic() < deadline:
                loop._intervention_step()
            press("c")
            press("a")
            assert loop.state == DaggerState.POLICY
            for _ in range(cfg.execute_chunk_dim):
                loop._policy_step()
            press("b")
            assert loop.state == DaggerState.FINISH_PENDING
            time.sleep(0.25)
            press("c")
            assert loop.state == DaggerState.PRE_RECORDING
            time.sleep(0.5)
            report["commands"] = env.commands
            report["inferences"] = loop.inference_id
        finally:
            if loop is not None:
                loop.close()
            if key_pub is not None:
                key_pub.close(linger=0)
            if key_context is not None:
                key_context.term()
            stop.set()
            for process in reversed(processes):
                _stop_process(process)
            comms._IPC_DIR = previous_ipc

    recordings = list(output.glob("*.h5"))
    if len(recordings) != 1:
        raise RuntimeError(f"Expected one DAgger recording, found {recordings}")
    recording = recordings[0]
    with h5py.File(recording, "r+") as file:
        assert file.attrs["synthetic_dry_run"] and not file.attrs["usable"]
        actions = np.asarray(file["inference/actions"])
        assert actions.shape[1:] == (cfg.execute_chunk_dim, 14)
        assert np.isfinite(actions).all()
        events = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in file["dagger_events/event"][:]
        ]
        for event in (
            "episode_start",
            "intervention_start",
            "intervention_end",
            "checkpoint",
        ):
            assert event in events, (event, events)
        assert len(file["dagger_checkpoints/index"]) == 1
        durations_ms = (
            np.asarray(file["inference/end_ts"])
            - np.asarray(file["inference/start_ts"])
        ) * 1000
        report["inference_ms"] = {
            "first": float(durations_ms[0]),
            "warm_median": float(np.median(durations_ms[1:])),
            "warm_max": float(np.max(durations_ms[1:])),
        }
        report.update(
            {
                "events": events,
                "recorded_actions_shape": list(actions.shape),
                "recording": str(recording),
                "passed": True,
            }
        )
    if cfg.render_video:
        from deploy.recording.render import render

        report["video"] = render(str(recording))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[dry-run] PASS", json.dumps(report), flush=True)
    return report


def main(cfg: DryRunConfig):
    if cfg.policy_chunks < 1 or cfg.intervention_seconds <= 0:
        raise ValueError("Use positive policy chunks and intervention duration")
    os.chdir(Path(__file__).resolve().parents[1])
    checkpoints = [Path(p).expanduser().resolve() for p in cfg.checkpoints]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    root = Path(cfg.output_dir).resolve()
    run_dir = root / time.strftime("%Y%m%d-%H%M%S")
    reports = []
    for index, checkpoint in enumerate(checkpoints):
        label = checkpoint.parent.name + "_" + checkpoint.stem
        reports.append(_run_one(cfg, checkpoint, run_dir / f"{index:02d}_{label}"))
    (run_dir / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"[dry-run] All {len(reports)} checkpoints passed. Results: {run_dir}")


if __name__ == "__main__":
    main(tyro.cli(DryRunConfig))
