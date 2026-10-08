"""Convert LeRobot v3.0 bimanual YAM datasets into the training data layout.

Reads one LeRobot dataset per task (``<root>/<repo>/{data,meta,videos}``),
splits whole episodes into train/val with a split file, and writes each episode
in the format export_mcap.py produces: ``states_actions.bin``, the combined
30 fps camera video, and ``episode_metadata.json``. CLI entrypoint:
scripts/export_lerobot.py.

Joint order, units and gripper range (radians, 0-1 gripper, left arm first)
already match ABC's YAM data, so state and action rows are copied unchanged.
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tyro

from abc_minimal.export_mcap import FPS, OUT_H, OUT_W, X264, X264_STRICT_FFMPEG_ARGS, probe
from abc_minimal.export_mcap import TICK_NS, TICKS_PER_FRAME, TIMESCALE

REPO_ROOT = Path(__file__).resolve().parents[1]
# ABC camera key -> LeRobot video feature. Order is the vstack order in the
# combined mp4 and matches DiTConfig.camera_keys.
CAMERAS = (
    ("top", "observation.images.head_cam"),
    ("left", "observation.images.wrist_left"),
    ("right", "observation.images.wrist_right"),
)
STATE_KEY, ACTION_KEY = "observation.state", "action"
JOINT_NAMES = (
    *(f"left_joint_{i}.pos" for i in range(1, 7)), "left_gripper.pos",
    *(f"right_joint_{i}.pos" for i in range(1, 7)), "right_gripper.pos",
)
SPLIT_DIRS = {"train": "train_real", "val": "val_real"}


@dataclass
class ExportLerobotConfig:
    """Convert LeRobot v3.0 YAM datasets into ABC train_real/ and val_real/."""

    out_dir: Path
    """Cache root to write into; episodes land in <out_dir>/{train_real,val_real}/."""
    root: Path = Path("~/cos-mount/data-collection/yam")
    """Directory holding one LeRobot dataset folder per repo."""
    split_file: Path = REPO_ROOT / "scripts" / "yam_split.json"
    """Per-repo train/val episode indices plus excluded_episodes."""
    repos: tuple[str, ...] = ()
    """Repos to convert (default: every repo in the split file)."""
    splits: tuple[str, ...] = ("train", "val")
    max_episodes: int | None = None
    """Per repo and split cap, e.g. 1 for a smoke test."""
    workers: int = 8
    overwrite: bool = False
    """Re-export episodes whose states_actions.bin already exists."""


def task_slug(task: str) -> str:
    """'Fold Handkerchief' -> 'fold_handkerchief' (ABC prompts are lowercase)."""
    return re.sub(r"[^a-z0-9]+", "_", task.strip().lower()).strip("_")


def read_episode_index(dataset: Path) -> dict[int, dict]:
    """Episode rows from meta/episodes/*/*.parquet, keyed by episode_index."""
    rows = {}
    for path in sorted((dataset / "meta" / "episodes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            rows[int(row["episode_index"])] = row
    return rows


def plan_jobs(config: ExportLerobotConfig) -> list[tuple]:
    split = json.loads(config.split_file.read_text())
    excluded = {
        repo: set(entry["episodes"]) for repo, entry in split.get("excluded_episodes", {}).items()
    }
    repos = config.repos or tuple(split["repos"])
    root = config.root.expanduser()
    jobs = []
    for repo in repos:
        if repo not in split["repos"]:
            raise ValueError(f"{repo} is not in {config.split_file}")
        dataset = root / repo
        info = json.loads((dataset / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v3.0" or info.get("fps") != FPS:
            raise ValueError(f"{repo}: expected LeRobot v3.0 at {FPS} fps")
        for key in (STATE_KEY, ACTION_KEY):
            if tuple(info["features"][key]["names"]) != JOINT_NAMES:
                raise ValueError(f"{repo}: unexpected {key} joint names")
        missing = [f for _, f in CAMERAS if f not in info["features"]]
        if missing:
            raise ValueError(f"{repo}: missing cameras {missing}")
        operators = {}
        provenance = dataset / "meta" / "episode_provenance.json"
        if provenance.exists():
            for idx, entry in json.loads(provenance.read_text())["episodes"].items():
                operator = entry.get("recording_metadata", {}).get("operator_id")
                if operator:
                    operators[int(idx)] = operator
        episodes = read_episode_index(dataset)
        for split_name in config.splits:
            chosen = [e for e in split["repos"][repo][split_name] if e not in excluded.get(repo, ())]
            for ep in chosen[: config.max_episodes]:
                out = config.out_dir / SPLIT_DIRS[split_name] / f"episode_{repo}_{ep:04d}"
                if not config.overwrite and (out / "states_actions.bin").exists():
                    continue
                jobs.append((str(dataset), info, episodes[ep], operators.get(ep), str(out), repo))
    return jobs


def export_episode(job):
    dataset, info, row, operator_id, out_dir, repo = job
    dataset, out_dir = Path(dataset), Path(out_dir)
    ep, num_steps = int(row["episode_index"]), int(row["length"])
    name = f"{repo}#{ep}"

    data_path = dataset / info["data_path"].format(
        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
    )
    table = pq.read_table(
        data_path, columns=["episode_index", "frame_index", STATE_KEY, ACTION_KEY],
        filters=[("episode_index", "=", ep)],
    )
    table = table.sort_by("frame_index")
    frames = table["frame_index"].to_numpy()
    if len(frames) != num_steps or not np.array_equal(frames, np.arange(num_steps)):
        raise RuntimeError(f"{name}: parquet has {len(frames)} rows / non-contiguous frames, "
                           f"meta says {num_steps}")
    state = np.stack(table[STATE_KEY].to_numpy(zero_copy_only=False)).astype(np.float64)
    action = np.stack(table[ACTION_KEY].to_numpy(zero_copy_only=False)).astype(np.float64)
    sa = np.concatenate([state, action], axis=-1)
    if sa.shape != (num_steps, 28) or not np.isfinite(sa).all():
        raise RuntimeError(f"{name}: bad state/action array {sa.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2")
    with tempfile.TemporaryDirectory() as work:
        mp4s = []
        for cam_key, feature in CAMERAS:
            video = dataset / info["video_path"].format(
                video_key=feature,
                chunk_index=row[f"videos/{feature}/chunk_index"],
                file_index=row[f"videos/{feature}/file_index"],
            )
            start = float(row[f"videos/{feature}/from_timestamp"])
            mp4 = str(Path(work) / f"{cam_key}.mp4")
            # Input-side -ss decodes from the previous keyframe and drops frames
            # before `start` (accurate seek), so frame 0 is the episode's first.
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(video),
                 "-frames:v", str(num_steps), "-an", "-vsync", "0", "-vf", vf,
                 *X264, "-threads", "2", mp4],
                check=True,
            )
            (n,) = probe(mp4, "-count_frames", "-show_entries", "stream=nb_read_frames")
            if n != num_steps:
                raise RuntimeError(f"{name}: {cam_key} decoded {n} frames, expected {num_steps}")
            mp4s.append(mp4)

        combined = str(out_dir / "combined_camera-images-rgb.mp4")
        filt = (
            "".join(f"[{i}:v]" for i in range(len(mp4s)))
            + f"vstack=inputs={len(mp4s)}[v0];"
            + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", *sum((["-i", p] for p in mp4s), []),
             "-filter_complex", filt, "-map", "[out]", *X264_STRICT_FFMPEG_ARGS, combined],
            capture_output=True, check=True,
        )
        (n,) = probe(combined, "-count_frames", "-show_entries", "stream=nb_read_frames")
        if n != num_steps:
            raise RuntimeError(f"{name}: combined video has {n} frames, expected {num_steps}")

    # Written last: the dataloader only picks up directories with this file,
    # so an interrupted export is never trained on.
    task = row["tasks"][0]
    meta = {"task_name": task_slug(task), "cameras": [k for k, _ in CAMERAS],
            "camera_resolutions": {k: [OUT_W, OUT_H] for k, _ in CAMERAS},
            "alignment": "lerobot_frame_index", "tick_ns": TICK_NS, "num_steps": num_steps,
            "source": {"format": "lerobot_v3.0", "repo": repo, "episode_index": ep, "task": task}}
    if operator_id:
        meta["operator_id"] = operator_id
        (out_dir / "operator.json").write_text(json.dumps({"operator_id": operator_id}))
    (out_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    sa.tofile(out_dir / "states_actions.bin")
    print(f"[OK] {out_dir.parent.name}/{out_dir.name}: {num_steps} steps, "
          f"task={meta['task_name']}", flush=True)
    return out_dir.name


def _safe_export(job):
    try:
        return export_episode(job)
    except Exception as exc:  # report and keep converting the rest
        print(f"[FAIL] {job[5]}#{job[2]['episode_index']}: {exc}", flush=True)
        return None


def main(config: ExportLerobotConfig):
    jobs = plan_jobs(config)
    print(f"{len(jobs)} episodes to export -> {config.out_dir}", flush=True)
    with Pool(config.workers) as pool:
        done = [r for r in pool.imap_unordered(_safe_export, jobs) if r]
    print(f"exported {len(done)}/{len(jobs)}")
    if len(done) != len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(ExportLerobotConfig))
