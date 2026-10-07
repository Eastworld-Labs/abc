# ABC deployment

This is the real-robot deployment stack for the released ABC policies. It
supports local or remote inference, standard action chunking, RTC, recording,
teleoperation, and replay-based testing. Both the ABC-DiT and the Gemma/SigLIP
VLA policies are servable; the server auto-detects which one a checkpoint is
(see [VLA inference](#vla-inference)).

Install the hardware/runtime dependencies on the robot workstation:

```bash
uv sync --extra deploy
```

## Real-robot inference

```bash
ROBOT_PROFILE=gtwy_config CUDA_VISIBLE_DEVICES=0 uv run deploy/deploy_policy.py --checkpoint-path=cache/bottles_75k.pt --diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 --execute-chunk-dim=16 --prompt='throw plastic bottles in bin'
```

The checkpoint must contain normalization statistics; otherwise pass
`--norm-stats-path`. The deploy adapter uses the same fixed ABC-DiT xL model,
DINOv3/ImageNet image path, z-score normalization, flow sampler, and RTC prefix
conditioning as simulation evaluation.

For a remote GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_policy.py --policy.checkpoint-path=cache/bottles_75k.pt --policy.prompt='throw plastic bottles in bin'
```

Then run the robot side with `--remote-host=<gpu-host>`. Add
`--compress-images` when network bandwidth or latency is limited.

## Serving robot_class's YAM runner

`deploy/serve_robot_class.py` serves ABC-DiT to the bimanual YAM runner in
`robot_class` (`inference/yam/run.py --remote-host`), which speaks openpi's
websocket protocol. The policy runs in this process; only key names and image
layout are translated (`head_cam`/`wrist_left`/`wrist_right` HWC frames to
ABC's `top`/`left`/`right` CHW, RTC leftover to the DiT action prefix). The
action and state spaces already match: 14-D left-first joints plus gripper at
30 Hz. DiT only; the VLA is not exposed through this adapter.

Chunked execution:

```bash
# abc
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_robot_class.py \
    --policy.checkpoint-path=cache/bottles_75k.pt --policy.diffusion-steps=10 \
    --policy.prompt='throw plastic bottles in bin'

# robot_class (dry run; add --enable-motion once the actions look sane)
python -m inference.yam.run --left-channel can_follower_l --right-channel can_follower_r \
    --remote-host 127.0.0.1 --control-mode joint --execution-horizon 16 \
    --prompt 'throw plastic bottles in bin'
```

RTC: the server fixes the execution horizon and inference delay and sends
them to the runner in its metadata, so pass `--rtc` on both sides:

```bash
# abc
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_robot_class.py \
    --policy.checkpoint-path=cache/bottles_75k.pt --policy.diffusion-steps=5 \
    --policy.prompt='throw plastic bottles in bin' \
    --rtc --execution-horizon=16 --inference-delay=6

# robot_class
python -m inference.yam.run --left-channel can_follower_l --right-channel can_follower_r \
    --remote-host 127.0.0.1 --control-mode joint --rtc \
    --prompt 'throw plastic bottles in bin'
```

`--inference-delay` must cover one request: at 30 Hz each step is 33 ms, so a
~110 ms request (5 diffusion steps, RTX 5080, no `--policy.fast-inference`)
needs at least 4; use ~8 for 10 steps (~200 ms). `execution_horizon +
inference_delay` must not exceed the 30-action chunk. Turn off head-camera
depth in robot_class's `configs/yam_cameras.yaml` (`enable_depth: false`); the
DiT does not use it. For zero-shot use, match the ABC station: a top camera and
two wrist cameras at 640x480, and start from ABC's `init_q`
(`[0, 0.5, 1.0, -1.0, 0, 0, 1.0]` per arm) rather than robot_class's home pose.

## DAgger checkpoints

Five real-robot ABC-DiT finetunes that we improved with DAgger on our own
station. `prepare.py` downloads one (~8.1 GB, sha256-verified) and prints the
deploy command it was run with:

```bash
uv run prepare.py --dagger-checkpoint folding_paper_box
```

| Checkpoint | Task | Step | Stiffness |
| --- | --- | --- | --- |
| `folding_paper_box` | fold the paper box | 75k | default |
| `insert_earbuds_1eb` | insert the earbuds into their charging case (one earbud) | 5k | default |
| `insert_earbuds_3eb` | insert the earbuds into their charging case (three earbuds) | 5k | default |
| `using_a_key` | unlock a box with a key | 5k | 2x kp |
| `pack_student_bag` | pack a student bag | 10k | 2x kp |

All five run with 5 diffusion steps, RTC (prefix 4, lead 4, 20 actions per
chunk) and a task-specific `--init-q`. `using_a_key` and `pack_student_bag`
were trained on recordings made with doubled arm stiffness, so their commands
add `--kp-scale='[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]'` (follower kp multipliers
for the six joints and the gripper, applied to both arms). The DAgger data was
collected without the standard cage, so expect them to need adapting on other
setups.

## VLA inference

ABC-VLA is served through the same websocket protocol;
`serve_policy.py` auto-detects the checkpoint type (`--policy-type` forces it):

```bash
uv run prepare.py --vla-pretrained

CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_policy.py \
    --policy.checkpoint-path=cache/vla_abc130k_200000_v2.pt \
    --policy.prompt='connect and route the hose'
```

V2 checkpoints embed `norm_stats` (pass `--policy.norm-stats-path` for the
original release files) and inference needs a ~24 GB GPU. `deploy_policy.py` and
`dagger.py` launch this server locally and forward the same `--policy-type`,
`--dit-model.*`, `--clip.*` and `--vla-model.*` flags.

Useful flags:

- `--diffusion-steps`: Euler flow steps, default 10.
- `--rtc`: overlap inference with action execution.
- `--execute-chunk-dim`: actions executed from each prediction.
- `--no-record`: skip H5 recording (recording is on by default).
- `--debug`: run cameras and inference without commanding followers.
- `--fast-inference`: use the bf16/compiled inference path.
- `--policy-type`: force `dit`/`vla` instead of auto-detecting.
- `--model-size`: model name stamped into recordings; defaults to `dit_xL` or `vla_4b` by detected policy type.
- `--verbose`: show output from all child processes.

### Episode control

Rollouts are episodic: the loop idles until you press a key, runs the policy,
and on the next press stops, sends the robot home, and idles again. Keys are
read from the terminal keyboard: `a`/`b` start or stop+home, `c`/`j` shut
everything down. With recording on (the default) one
H5 plus a review MP4 is saved per episode; with `--no-record` the same keys
drive the loop directly (`--no-episode-control` restores free-running
inference).

## Foot pedal

`deploy_policy.py` is keyboard-only; it has no foot-pedal support and no
`--foot-pedal-device` flag. The pedal remains available for data collection
(`deploy/robot/scripts/run_data_record.py`) and DAgger (`dagger.py`), which
read keys from the terminal *and*, when present, directly from a USB foot
pedal's evdev device — so the pedal works over SSH and regardless of window
focus, and is grabbed exclusively so presses don't also type into your shell.

**One-time station setup** — the pedal's device node is `root:input`, so
grant yourself access once:

```bash
sudo usermod -aG input $USER     # durable; log out and back in once
# or, effective immediately but lost when the pedal re-enumerates:
sudo setfacl -m u:$USER:rw "$(readlink -f /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd)"
```

**Device selection** (checked at launch, before anything starts): an explicit
`--foot-pedal-device <path>` wins, then the `FOOT_PEDAL_INPUT_DEVICE`
environment variable, then the PCsensor default
(`/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd`) automatically whenever
it is plugged in. A configured device that is missing or unreadable stops the
launch with fix-it steps; `--foot-pedal-device ''` forces keyboard-only.
Without any pedal, terminal keys always work.

**If the pedal does nothing**: check it actually reaches this machine —
`ls /dev/input/by-id/` should list a `...FootSwitch-event-kbd` entry. If it
doesn't, the pedal is on a port or hub that isn't connected to this host
(unplugging/replugging it will not even show up in `lsusb`).

## Passive-GELLO DAgger

`dagger.py` runs the single supported intervention mode: the standard GELLOs
with force feedback disabled, anchored end-effector-relative motion, and Mink
IK. The policy and recording
arguments are the same as `deploy_policy.py`:

```bash
ROBOT_PROFILE=gtwy_config CUDA_VISIBLE_DEVICES=0 uv run deploy/dagger.py --checkpoint-path=cache/bottles_75k.pt --diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 --execute-chunk-dim=16 --prompt='throw plastic bottles in bin'
```

GELLO serial devices, servo IDs, and joint signs come from `ROBOT_PROFILE`,
exactly as they do for teleoperation.

Pedal controls match the original DAgger flow:

- A starts an episode and toggles between policy and intervention.
- C records a checkpoint.
- B freezes and ends the episode, then A discards the suffix after the latest
  checkpoint or C keeps and saves the complete episode.

Recordings are written under `data/dagger_h5`. A discarded suffix remains in
the H5 for auditability and is identified by `discard_segments` metadata.

## Recording videos

Every saved recording — teleop collection, `--record` rollouts, and DAgger —
is rendered to an annotated review MP4 in a detached process. Videos land in
a sibling directory named for the recording type: `data/dagger_h5/<f>.h5` →
`data/dagger_video/<f>.mp4`. DAgger videos include the controller-state
badge, checkpoint markers, and a green/red row previewing exactly what the
exporter will keep or drop. Disable with `--no-post-video` (or
`DEPLOY_POST_VIDEO=0`). The background render logs to `postprocess.log`
next to the recordings — check it if a video doesn't appear.

Manual render and catch-up over recordings missing videos:

```bash
uv run python -m deploy.recording.render data/dagger_h5/<file>.h5
uv run python -m deploy.recording.postprocess --scan data/
```

## Exporting recordings for finetuning

`deploy/recording/export.py` converts recordings of all three types into the
training episode format `abc_minimal` reads (`train/` + `val/` episode dirs
holding `combined_camera-images-rgb.mp4` and `states_actions.bin`). DAgger
recordings export the human INTERVENTION segments by default; teleop and
rollout recordings export whole episodes. Operator discard decisions are
honored in both (samples after the latest checkpoint are dropped; the
checkpoint sample itself is kept).

```bash
uv run python -m deploy.recording.export \
    --input-pattern 'data/dagger_h5/*.h5' \
    --output-dir data/dagger_export \
    --norm-stats-from-checkpoint cache/bottles_75k.pt
```

Norm stats are never recomputed silently: for finetuning, always reuse the
checkpoint's stats via `--norm-stats-from-checkpoint` (or pass an existing
file with `--norm-stats-path`; `--compute-norm-stats` is for from-scratch
experiments only). To train on the export, point a `TrainConfig`
`MixtureComponent` at the absolute `train/` and `val/` paths and resume from
the deployed checkpoint. `--concat-review-mp4` additionally writes a single
concatenated review reel; `export_summary.json` records every exported and
skipped segment with reasons.

See [robot/README.md](robot/README.md) for teleoperation, data collection, H5
replay, and robot profiles.
