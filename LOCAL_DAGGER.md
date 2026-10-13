# Local ABC DAgger on YAM

Run from `/home/ubuntu/Desktop/EW/abc`. The local launcher selects ABC's
Python 3.12 `uv` environment, keeps the deploy extras installed, uses the
downloaded CLIP assets in `cache/clip`, and defaults to `yam007_config`.

## Hardware-free integration check

```bash
cd /home/ubuntu/Desktop/EW/abc
scripts/run_local_dagger.sh --dry-run
```

This checks the base ABC-DiT model and the unfold 30k model sequentially on
CUDA. It uses the real WebSocket policy server, DAgger state machine, RTC,
relative-pose Mink IK and H5 recorder with synthetic cameras/followers/leaders
in a private IPC namespace. It exercises start, policy execution, takeover,
checkpoint marking, handback, finish and save, then renders a review MP4.
It never starts CAN, RealSense or motor processes. `--debug` on the normal
DAgger command is different: it still opens cameras and leaders.

Results are written under `outputs/dagger_dry_run/<run>/` with a JSON report,
H5 and MP4 for each checkpoint. Synthetic recordings are marked
`synthetic_dry_run=true`, `usable=false`, `unusable=true` and kept outside the
real recording directory. Do not train on these recordings.

Check a different checkpoint or standard inference:

```bash
scripts/run_local_dagger.sh --dry-run \
  --checkpoints checkpoints/abc-yam-bs16-20261007-1833/70000_model.pt \
  --prompt 'stack cup' --no-rtc
```

The dry-run option names belong to `scripts/dry_run_dagger.py`; use
`scripts/run_local_dagger.sh --dry-run --help` to see them. The default
10-step sampler uses an RTC prefix of 4, execution chunk of 16 and inference
lead of 14. On this RTX 3080 Ti laptop, measured warm 10-step requests were
roughly 350–400 ms, so the original 7-step lead (233 ms at 30 Hz) can stall.
The 14-step lead allows 467 ms; measure again with real camera frames.

## System prerequisites and hardware mapping

Run this in your own terminal, where sudo can prompt for your password:

```bash
scripts/setup_local_dagger_system.sh
```

It installs `can-utils`, loads the in-kernel `gs_usb` driver, installs the
official RealSense 2.58.4 udev rules and reloads udev rules. Reconnect the
cameras afterward. The rules are cached in `.uv/realsense/`.

Connect the labelled adapters and map them to `can_leader_l`, `can_leader_r`,
`can_follower_l`, `can_follower_r`, UP at 1 Mbps. Derive USB-port mappings on
this machine using `commands.md` section 8.3; station yam-007's USB paths
cannot be copied unchanged to a laptop. `yam007_config` has empty adapter
serial fields, so the generic `setup_can_names.sh` does not provide this
profile's mapping.

Expected camera serials:

| Position | Model | Serial |
|---|---|---|
| Top | D455 | 260722303224 |
| Left wrist | D405 | 352122272270 |
| Right wrist | D405 | 352122273271 |

```bash
source scripts/activate_env.sh
python scripts/check_dagger_hardware.py
```

This is a read-only check of CAN names, bitrate/state, camera enumeration,
CAN utilities and RealSense rule presence. It does not prove motor responses,
camera streaming, button behavior or motion calibration. Perform the operator
teleop check in `dagger_guide.md` before a real session. The normal local
launcher runs the preflight and refuses to start hardware when it fails.

## Real local launch

After the hardware preflight and operator teleop checks pass:

```bash
scripts/run_local_dagger.sh \
  --checkpoint-path=checkpoints/unfold-10hr-data/30000_model.pt \
  --prompt='unfold' --diffusion-steps=10 \
  --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=14 \
  --execute-chunk-dim=16
```

No `--remote-host` is needed: the launcher starts the policy server locally.
Real recordings go to `data/dagger_h5`; review videos go to
`data/dagger_video`. Prompts must match training: `unfold` for the unfold
checkpoints; the lowercased task name (such as `stack cup`) for the 15-task
model. See `commands.md` section 5 for the checkpoint/prompt catalogue.

## Viewers

The DAgger review renderer includes all three camera streams, joint/command
plots, policy versus intervention state, checkpoint markers, and export
keep/discard spans. Saved episodes are rendered automatically. Manually:

```bash
source scripts/activate_env.sh
uv run --frozen --extra deploy python -m deploy.recording.render \
  data/dagger_h5/RECORDING.h5
uv run --frozen --extra deploy python -m deploy.recording.postprocess --scan data/
```

Open the resulting MP4 with your video player. The dry-run results and review
videos are indexed in `outputs/dagger_dry_run/RESULTS.md`.

`viz_policy.py` and `viz_episode.py` provide browser-based Viser viewers for
simulation policies and exported dataset episodes, respectively. They are
not live viewers of the real DAgger stack and need the relevant simulation
assets/dataset. This checkout does not include a live DAgger dashboard.
