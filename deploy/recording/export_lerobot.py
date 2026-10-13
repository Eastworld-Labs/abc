"""Export deploy recordings (teleop / inference / dagger H5) to a LeRobot v3.0 dataset.

The LeRobot twin of ``export.py``, for policies that train from LeRobot datasets
(MolmoAct2, pi0.5) rather than abc_minimal. Segment selection is shared with
``export.py`` (``iter_segments``), so both formats export exactly the same frames:
DAgger recordings default to the human INTERVENTION segments, other recordings to
whole episodes, and operator discards are honored either way. Each segment becomes
one LeRobot episode.

The output matches the station's LeRobot recordings (e.g. ``unfold_linen_20261009``):
``observation.state`` / ``action`` (14,) float32, left arm first; one video per camera
at native resolution, RGB; 30 fps. ``--intervention-flag`` adds a per-frame
``intervention`` (1,) float32 column (1 = human driving), useful when exporting whole
DAgger rollouts and weighting the takeovers at training time.

Not to be confused with ``abc_minimal/export_lerobot.py``, which goes the other way
(LeRobot -> abc_minimal). ABC's environment has no lerobot, so run with it overlaid:

    uv run --with 'lerobot[dataset]>=0.5,<0.6' python -m deploy.recording.export_lerobot \\
        --input-pattern 'data/dagger_h5/*.h5' --output-dir data/dagger_lerobot \\
        --task-name 'unfold linen'

``--task-name`` should be the prompt the target model was trained with; the default is
the H5's ``task_name``, i.e. the prompt the deployed policy ran with.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro

from deploy.recording import io as rio
from deploy.recording.export import iter_segments

STATE_NAMES = [f"{side}_{j}" for side in ("left", "right") for j in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]


@dataclass
class LerobotExportConfig:
    input_pattern: str
    """Glob for recording H5 files, e.g. 'data/dagger_h5/*.h5'."""

    output_dir: str
    """LeRobot dataset root to create. Its name becomes the dataset's repo_id."""

    mode: Literal["auto", "whole_episodes", "interventions"] = "auto"
    """auto: dagger recordings export INTERVENTION segments, others whole episodes."""

    task_name: str = ""
    """Prompt stored with every frame. Default: the H5 task_name attr."""

    camera_map: dict[str, str] = field(
        default_factory=lambda: {"top": "head_cam", "left": "wrist_left", "right": "wrist_right"}
    )
    """Recording camera -> LeRobot key suffix (observation.images.<suffix>)."""

    intervention_flag: bool = False
    """Add a per-frame 'intervention' column (1.0 while the human was driving)."""

    fps: int = 30
    vcodec: str = "h264"
    robot_type: str = "yam_bimanual_follower"

    modified_after: str | None = None
    """Only include files modified after this local time, e.g. '2026-08-11 14:30'."""

    collection_name: str | None = None
    exclude_files: list[str] = field(default_factory=list)
    state_label: str = "INTERVENTION"
    min_segment_frames: int = 30
    context_before_frames: int = 0
    context_after_frames: int = 0

    overwrite: bool = False
    """Remove output_dir first if it already exists."""

    @property
    def cameras(self) -> tuple[str, ...]:
        # iter_segments syncs and checks exactly these recording cameras.
        return tuple(self.camera_map)


def _features(first_frames: dict[str, np.ndarray], intervention_flag: bool) -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES},
    }
    for key, image in first_frames.items():
        feats[key] = {"dtype": "video", "shape": image.shape, "names": ["height", "width", "channels"]}
    if intervention_flag:
        feats["intervention"] = {"dtype": "float32", "shape": (1,), "names": None}
    return feats


def export_lerobot(cfg: LerobotExportConfig) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(cfg.output_dir).expanduser().resolve()
    if root.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite to replace")
        shutil.rmtree(root)

    summary: dict = {
        "input_pattern": cfg.input_pattern,
        "mode": cfg.mode,
        "state_label": rio.canonical_state(cfg.state_label),
        "camera_map": cfg.camera_map,
        "exported_episodes": [],
        "skipped_segments": [],
        "skipped_files": [],
    }
    intervention = rio.canonical_state("INTERVENTION")
    dataset = None
    shapes: dict[str, tuple] = {}

    for seg in iter_segments(cfg, summary):
        streams = rio.camera_streams(seg.f)

        def frames_at(tick: int) -> dict[str, np.ndarray]:
            out = {}
            for cam, suffix in cfg.camera_map.items():
                key = streams[cam]
                bgr = rio.decode_frame(seg.f, key, int(seg.synced.indices[key][tick]))
                out[f"observation.images.{suffix}"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return out

        images = frames_at(seg.start)
        if dataset is None:
            if abs(seg.synced.avg_hz - cfg.fps) > 0.1 * cfg.fps:
                print(f"WARNING: {seg.h5_path} camera timeline runs at {seg.synced.avg_hz:.1f} Hz, "
                      f"not {cfg.fps}; frames are stored one per camera tick.")
            shapes = {key: img.shape for key, img in images.items()}
            dataset = LeRobotDataset.create(
                repo_id=root.name, fps=cfg.fps, root=root, robot_type=cfg.robot_type,
                features=_features(images, cfg.intervention_flag), use_videos=True,
                vcodec=cfg.vcodec,
            )
        else:
            changed = {k: (shapes[k], img.shape) for k, img in images.items() if img.shape != shapes[k]}
            if changed:
                raise ValueError(
                    f"{seg.h5_path}: camera resolution differs from the first episode {changed}; "
                    "export recordings from different camera setups into separate datasets."
                )

        for tick in range(seg.start, seg.end):
            if tick > seg.start:
                images = frames_at(tick)
            frame = {
                "observation.state": seg.states[tick].astype(np.float32),
                "action": seg.actions[tick].astype(np.float32),
                "task": seg.task_name,
                **images,
            }
            if cfg.intervention_flag:
                frame["intervention"] = np.array([seg.labels[tick] == intervention], dtype=np.float32)
            dataset.add_frame(frame)
        dataset.save_episode()
        summary["exported_episodes"].append(seg.summary_entry())
        print(f"[export] {seg.episode_id}: {seg.n_frames} frames")

    if dataset is None:
        raise RuntimeError("No episodes were exported")
    dataset.finalize()
    (root / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    total = sum(ep["frames"] for ep in summary["exported_episodes"])
    print(f"Exported {len(summary['exported_episodes'])} episodes ({total} frames) to {root}")
    return summary


def main() -> None:
    export_lerobot(tyro.cli(LerobotExportConfig))


if __name__ == "__main__":
    main()
