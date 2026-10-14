"""Convert ROS2 rosbag2 MCAP bimanual YAM episodes into the training data layout.

Input is one folder per episode, as the recorder writes it::

    <root>/ep_<id>_<yyyymmdd>_<hhmmss>/{ep_<...>.mcap, episode_labels.json, ...}

The bags carry CDR ``sensor_msgs/JointState`` at ~250 Hz and h264
``FFMPEGPacket`` cameras at 30 Hz, unlike the protobuf release format that
export_mcap.py reads. State is the measured joints plus gripper state and
action the commanded joints plus gripper command, left arm first, which is
the layout and meaning of the LeRobot YAM datasets (export_lerobot.py).

Episodes are placed on the same fixed 30 Hz causal clock as export_mcap.py:
each tick takes the latest message received (``log_time``) at or before it.
Camera streams begin mid-GOP, so packets before the first keyframe are
undecodable and are dropped rather than re-timed. Whole episodes are split
into train/val with evo's seeded permutation (scripts/make_yam_split.py),
over episodes in chronological order. CLI entrypoint:
scripts/export_ros2_mcap.py.
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from abc_minimal.export_mcap import (
    OUT_H,
    OUT_W,
    TICK_NS,
    TICKS_PER_FRAME,
    TIMESCALE,
    X264_STRICT_FFMPEG_ARGS,
    encode_aligned,
    floor_indices,
    probe,
)

# ABC camera key -> bag topic. Order is the vstack order in the combined mp4
# and matches DiTConfig.camera_keys.
CAMERAS = (
    ("top", "/cameras/head_cam/image"),
    ("left", "/cameras/wrist_left/image"),
    ("right", "/cameras/wrist_right/image"),
)
# (topic, dims) in ABC's 14-D joint order: left arm, left gripper, right arm, right gripper.
STATE_TOPICS = (
    ("/arms/left/joint_state/measured", 6),
    ("/arms/left/gripper/state", 1),
    ("/arms/right/joint_state/measured", 6),
    ("/arms/right/gripper/state", 1),
)
ACTION_TOPICS = (
    ("/arms/left/joint_state/commanded", 6),
    ("/arms/left/gripper/command", 1),
    ("/arms/right/joint_state/commanded", 6),
    ("/arms/right/gripper/command", 1),
)
SPLIT_DIRS = {"train": "train_real", "val": "val_real"}

# What the episode contains, from the recordings. For task_id "unfold_handkerchief"
# single-cloth episodes unfold one crumpled cloth (5-130 s) and towel episodes
# unfold several crumpled towels and pile them up (>= 200 s, always white);
# nothing lies between 126 s and 228 s, but the raw task_id calls both
# "unfold_handkerchief". Task id "unfold_linen" (80-540 s, tagged hand_towel,
# foot_towel or pillow_case) is its own kind, whatever the duration, and so is
# task id "flip_parcel" (pull a parcel from a pile, flip it label up, put it on the conveyor).
KINDS = {
    "handkerchief": "unfold_handkerchief",
    "towel_stack": "unfold_and_stack_towels",
    "linen": "unfold_linen",
    "flip_parcel": "flip_parcel",
}
TOWEL_STACK_MIN_S = 200.0
LINEN_TASK_ID = "unfold_linen"
FLIP_PARCEL_TASK_ID = "flip_parcel"


@dataclass
class ExportRos2McapConfig:
    """Convert rosbag2 MCAP YAM episodes into ABC train_real/ and val_real/."""

    root: Path
    """Directory holding one ep_* folder per episode."""
    out_dir: Path
    """Cache root to write into; episodes land in <out_dir>/{train_real,val_real}/."""
    val_fraction: float = 0.05
    seed: int = 42
    kinds: tuple[str, ...] = tuple(KINDS)
    """Which episode kinds to export (see KINDS). The split is drawn over all
    episodes first, so restricting kinds never moves an episode across it."""
    task_ids: tuple[str, ...] = ()
    """Only consider episodes whose episode_labels.json task_id is listed (empty =
    all). Unlike kinds, this narrows the set the split is drawn over, e.g.
    ('unfold_linen',) to convert an added batch without touching an earlier split."""
    exclude: tuple[str, ...] = ()
    """Episode ids to skip, e.g. ep_769d_20261006_171146 (cut off mid-task)."""
    task_name: str | None = None
    """Give every episode this task_name (and so this prompt) instead of its
    kind's, e.g. 'unfold'."""
    splits: tuple[str, ...] = ("train", "val")
    max_episodes: int | None = None
    """Per split cap, e.g. 1 for a smoke test."""
    workers: int = 8
    overwrite: bool = False
    """Re-export episodes whose states_actions.bin already exists."""


def episode_order(ids) -> list[str]:
    """Chronological order (date, time, id) of ep_<id>_<yyyymmdd>_<hhmmss> names."""
    return sorted(ids, key=lambda e: (e.split("_")[2], e.split("_")[3], e))


def split_episodes(total: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """evo's split, identical to scripts/make_yam_split.py: (train, val) indices."""
    order = np.random.default_rng(seed).permutation(total)
    num_val = round(total * val_fraction)
    return sorted(order[num_val:].tolist()), sorted(order[:num_val].tolist())


def classify(duration_s: float, task_id: str | None = None) -> str:
    if task_id == LINEN_TASK_ID:
        return "linen"
    if task_id == FLIP_PARCEL_TASK_ID:
        return "flip_parcel"
    return "towel_stack" if duration_s >= TOWEL_STACK_MIN_S else "handkerchief"


def plan_jobs(config: ExportRos2McapConfig) -> tuple[list[tuple], list[dict]]:
    root = config.root.expanduser()
    ids = episode_order(p.name for p in root.glob("ep_*") if (p / f"{p.name}.mcap").exists())
    if not ids:
        raise ValueError(f"no ep_*/ep_*.mcap episodes under {root}")
    all_labels = {ep: json.loads((root / ep / "episode_labels.json").read_text()) for ep in ids}
    if config.task_ids:
        ids = [ep for ep in ids if all_labels[ep].get("task_id") in config.task_ids]
        if not ids:
            raise ValueError(f"no episodes with task_id in {config.task_ids} under {root}")
    train, val = split_episodes(len(ids), config.val_fraction, config.seed)
    chosen_idx = {"train": train, "val": val}
    manifest, jobs = [], []
    for split_name in config.splits:
        taken = 0
        for idx in chosen_idx[split_name]:
            ep = ids[idx]
            labels = all_labels[ep]
            duration_s = json.loads((root / ep / "episode.json").read_text())["bag"]["duration_ns"] / 1e9
            kind = classify(duration_s, labels.get("task_id"))
            tags = labels.get("task_tags") or {}
            entry = {"episode": ep, "index": idx, "split": split_name, "kind": kind,
                     "duration_s": round(duration_s, 1), "colour": tags.get("colour"), "type": tags.get("type"),
                     "operator_id": labels.get("operator_id"), "task_id": labels.get("task_id")}
            manifest.append(entry)
            if kind not in config.kinds or ep in config.exclude:
                continue
            if config.max_episodes is not None and taken >= config.max_episodes:
                continue
            out = config.out_dir / SPLIT_DIRS[split_name] / f"episode_{ep}"
            if not config.overwrite and (out / "states_actions.bin").exists():
                continue
            taken += 1
            jobs.append((str(root / ep / f"{ep}.mcap"), str(out), entry, config.task_name or KINDS[kind]))
    return jobs, manifest


def task_slug(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", task.strip().lower()).strip("_")


def read_bag(mcap_path: str):
    """Scalar and camera messages by topic, each sorted by receive time."""
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    scalar_topics = {t for t, _ in STATE_TOPICS + ACTION_TOPICS}
    camera_topics = {t for _, t in CAMERAS}
    scalars, cams, flags = {}, {}, {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(
            topics=sorted(scalar_topics | camera_topics)
        ):
            if channel.topic in scalar_topics:
                scalars.setdefault(channel.topic, []).append(
                    (message.log_time, np.array(decoded.position, dtype=np.float64)))
            else:
                cams.setdefault(channel.topic, []).append((message.log_time, bytes(decoded.data)))
                flags.setdefault(channel.topic, []).append(int(decoded.flags))
    for msgs in (*scalars.values(), *cams.values()):
        msgs.sort(key=lambda x: x[0])
    return scalars, cams, flags


def export_episode(job):
    mcap_path, out_dir, entry, task_name = job
    ep, out_dir = entry["episode"], Path(out_dir)
    scalars, cams, flags = read_bag(mcap_path)

    for topic, _ in STATE_TOPICS + ACTION_TOPICS:
        if topic not in scalars:
            raise RuntimeError(f"{ep}: no {topic} messages")
    # The stream starts mid-GOP: packets before the first keyframe (flags & 1,
    # SPS+PPS+IDR) cannot be decoded, so each camera starts at its first keyframe.
    for _, topic in CAMERAS:
        if topic not in cams:
            raise RuntimeError(f"{ep}: no {topic} messages")
        keyframes = [i for i, fl in enumerate(flags[topic]) if fl & 1]
        if not keyframes:
            raise RuntimeError(f"{ep}: {topic} has no keyframe")
        cams[topic] = cams[topic][keyframes[0]:]

    streams = [cams[t] for _, t in CAMERAS] + [scalars[t] for t, _ in STATE_TOPICS + ACTION_TOPICS]
    t0 = max(s[0][0] for s in streams)
    t_end = min(s[-1][0] for s in streams)
    ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
    num_steps = len(ticks)
    if num_steps < 10:
        raise RuntimeError(f"{ep}: too short ({num_steps} steps)")

    parts = []
    for topic, _ in STATE_TOPICS + ACTION_TOPICS:
        msgs = scalars[topic]
        ts = np.array([t for t, _ in msgs], dtype=np.int64)
        parts.append(np.stack([v for _, v in msgs])[floor_indices(ts, ticks)])
    sa = np.concatenate(parts, axis=-1)
    if sa.shape != (num_steps, 28) or not np.isfinite(sa).all():
        raise RuntimeError(f"{ep}: bad state/action array {sa.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        mp4s = []
        for cam_key, topic in CAMERAS:
            msgs = cams[topic]
            cam_ts = np.array([t for t, _ in msgs], dtype=np.int64)
            h264 = str(Path(work) / f"{cam_key}.h264")
            Path(h264).write_bytes(b"".join(data for _, data in msgs))
            width, height = probe(h264, "-show_entries", "stream=width,height")
            (n_frames,) = probe(h264, "-count_frames", "-show_entries", "stream=nb_read_frames")
            if n_frames != len(cam_ts):  # one packet per frame; a gap would silently shift time
                raise RuntimeError(f"{ep}: {cam_key} decoded {n_frames} frames from {len(cam_ts)} packets")
            mp4 = str(Path(work) / f"{cam_key}.mp4")
            encode_aligned(h264, width, height, floor_indices(cam_ts, ticks), mp4)
            mp4s.append(mp4)

        combined = str(out_dir / "combined_camera-images-rgb.mp4")
        filt = (
            "".join(f"[{i}:v]" for i in range(len(mp4s)))
            + f"vstack=inputs={len(mp4s)}[v0];"
            + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", *[a for p in mp4s for a in ("-i", p)],
             "-filter_complex", filt, "-map", "[out]", *X264_STRICT_FFMPEG_ARGS, combined],
            capture_output=True, check=True,
        )
        for mp4 in (*mp4s, combined):
            (n,) = probe(mp4, "-count_frames", "-show_entries", "stream=nb_read_frames")
            if n != num_steps:
                raise RuntimeError(f"{ep}: {Path(mp4).name} has {n} frames, expected {num_steps}")

    meta = {"task_name": task_slug(task_name), "cameras": [k for k, _ in CAMERAS],
            "camera_resolutions": {k: [OUT_W, OUT_H] for k, _ in CAMERAS},
            "alignment": "fixed_clock_30hz_causal", "t0_ns": int(t0), "tick_ns": TICK_NS,
            "num_steps": num_steps,
            "source": {"format": "ros2_mcap", "episode": ep, "kind": entry["kind"],
                       "task_id": entry["task_id"], "colour": entry["colour"], "type": entry["type"],
                       "duration_s": entry["duration_s"]}}
    if entry["operator_id"]:
        meta["operator_id"] = entry["operator_id"]
        (out_dir / "operator.json").write_text(json.dumps({"operator_id": entry["operator_id"]}))
    (out_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    # Written last: the dataloader only picks up directories with this file,
    # so an interrupted export is never trained on.
    sa.tofile(out_dir / "states_actions.bin")
    print(f"[OK] {out_dir.parent.name}/{out_dir.name}: {num_steps} steps, "
          f"kind={entry['kind']}, task={meta['task_name']}", flush=True)
    return out_dir.name


def _safe_export(job):
    try:
        return export_episode(job)
    except Exception as exc:  # report and keep converting the rest
        print(f"[FAIL] {job[2]['episode']}: {exc}", flush=True)
        return None


def main(config: ExportRos2McapConfig):
    jobs, manifest = plan_jobs(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "split_ros2_mcap.json").write_text(json.dumps(
        {"seed": config.seed, "val_fraction": config.val_fraction, "method": "evo_split_episodes",
         "order": "chronological", "total_episodes": len(manifest), "episodes": manifest}, indent=1) + "\n")
    print(f"{len(jobs)} episodes to export -> {config.out_dir}", flush=True)
    with Pool(config.workers) as pool:
        done = [r for r in pool.imap_unordered(_safe_export, jobs) if r]
    print(f"exported {len(done)}/{len(jobs)}")
    if len(done) != len(jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(ExportRos2McapConfig))
