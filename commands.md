# ABC-DiT on YAM: command cheatsheet

Finetunes of the released ABC-DiT (`abc_dit_xl_200k_model.pt`, the public ABC-130K
parent at step 200k) on our bimanual YAM data. Two families:

- the 15-task multi-task run from LeRobot datasets (2026-10-07), sections 1-4;
- the unfold runs from the local ROS2 MCAP recordings (2026-10-08), section 3b;
  data, converter and results are in `recipe_unfold.md`.

Section 5 says how to serve a checkpoint to robot_class and **which prompt goes
with which one**. Section 6 serves ABC-VLA, section 7 runs DAgger (split: GPU box
+ station), section 8 sets up one GPU box that does everything. The plan for
the next unfold data and DAgger rounds is `plan_unfold.md`; the DAgger operator
guide is `dagger_guide.md`.

| What | Where |
|---|---|
| Training box (multi-task run) | `ubuntu@43.202.110.152`, key `~/.ssh/h1pro-training-ap-northeast-2.pem`, repo `~/sk_vla/abc` |
| Instance (unfold runs) | `ubuntu@35.86.74.171`, key `~/.ssh/h1pro-training-us-west-2.pem`, repo `~/sk_vla/abc`, 8x A100 40 GB |
| Raw LeRobot data (remote) | `/opt/dlami/nvme/yam_data_local/<repo>/` (NVMe, wiped on instance stop) |
| Converted multi-task episodes | `/opt/dlami/nvme/abc_yam/{train_real,val_real}` + parent checkpoint (old box) |
| Unfold episodes (instance) | `/opt/dlami/nvme/abc_unfold` (first batch), `abc_unfold_linen` (second batch), `abc_unfold_all2` (links to both; what run 2 trained on) |
| Train/val split (multi-task) | `scripts/yam_split.json`: 95/5 per repo, seed 42 (evo's method; same held-out episodes as evo) |
| Checkpoints (remote) | `checkpoints/<run>/<step>.pt` (24 GB, resumable), `<step>_model.pt` (8 GB, for serving) |
| Local checkpoints | `checkpoints/unfold-10hr-data/`, `checkpoints/unfold-2hr-data/`, `checkpoints/abc-yam-bs16-20261007-1833/` (each `<step>_model.pt`); ABC-VLA parent `cache/vla_abc130k_200000_v2.pt` |
| Robot station yam-007 | `ssh robo@192.168.0.199` (Wi-Fi) / `192.168.123.227` (wired). No GPU. CAN `can_{leader,follower}_{l,r}`, D455 head + 2 D405 wrists. robot_class in `~/robot_class`, teleop/recording stack in `~/station_YAM` |
| GPU box (policy server) | `192.168.123.1`, wired to the station (~1-3 ms). Serve from here, not over Wi-Fi |

## 1. Sync code to the training box (local PC)

```bash
cd ~/Desktop/EW/abc
SSH="ssh -i ~/.ssh/h1pro-training-ap-northeast-2.pem"
git ls-files -co --exclude-standard > /tmp/abc_files.txt
rsync -avP -e "$SSH" --files-from=/tmp/abc_files.txt ./ ubuntu@43.202.110.152:~/sk_vla/abc/
```

For a new instance also copy `uv.lock` (gitignored, so not in that list), as in `recipe_unfold.md` section 2.

## 2. Setup and data conversion (training box)

```bash
cd ~/sk_vla/abc && uv python pin 3.12 && uv sync
export ABC_CACHE=/opt/dlami/nvme/abc_yam && mkdir -p $ABC_CACHE
uv run prepare.py --pretrained --cache $ABC_CACHE          # parent checkpoint, ~8 GB

# smoke test (1 train + 1 val episode per task), then everything; reruns skip finished episodes
uv run --with pyarrow scripts/export_lerobot.py --root /opt/dlami/nvme/yam_data_local --out-dir $ABC_CACHE --max-episodes 1
uv run --with pyarrow scripts/export_lerobot.py --root /opt/dlami/nvme/yam_data_local --out-dir $ABC_CACHE --workers $(( $(nproc) / 2 ))
```

New LeRobot datasets: add them with `uv run scripts/make_yam_split.py --repos <all repos...>`
(keeps the curated `excluded_episodes`), then rerun the converter. ROS2 MCAP
recordings use `scripts/export_ros2_mcap.py` instead (`recipe_unfold.md` section 1).

## 3. Train (training box, inside `tmux new -s abc`)

4x A100 40 GB on GPUs 4-7. Batch 16/GPU peaks at 38.1 GiB (DDP holds 32 GB of
weights + grads + Adam per GPU); 70k steps x 64 = 4.5M samples, ~6.5 h.

```bash
cd ~/sk_vla/abc
export ABC_CACHE=/opt/dlami/nvme/abc_yam PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(cat ~/sk_vla/molmoact2/experiments/.wandb_api_key)
export WANDB_ENTITY=$(cat ~/sk_vla/molmoact2/experiments/.wandb_entity)
RUN=abc-yam-bs16-$(date +%Y%m%d-%H%M); export WANDB_NAME=$RUN

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run torchrun --standalone --nproc-per-node 4 train.py \
  --load-pretrained --mixture-preset yam --batch-size 16 \
  --train-steps 70000 --ckpt-every 5000 --optim.learning-rate 5e-5 \
  --output-dir checkpoints/$RUN --log-wandb --wandb-project SK-ablation \
  2>&1 | tee checkpoints/$RUN.log
```

- Memory probe: same command with `--train-steps 50 --output-dir checkpoints/probe` and no wandb.
- Resume: replace `--load-pretrained` with `--resume-from checkpoints/<run>/last.pt`, same `--output-dir`.

### 3a. Small-batch options (off by default)

For 4x A100 40 GB, where the per-GPU batch is far below the paper's (1,440 DiT / 192 VLA):

| Flag | Effect |
|---|---|
| `--grad-accum-steps N` | N micro-batches per optimizer step; effective batch = `batch-size x GPUs x N`. Time per step grows ~N x; samples per second do not change. |
| `--optim.lr-decay-fraction F` | Linear LR decay to 0 over the last F of `--train-steps` (the paper keeps it constant). |
| `--optim.ema-decay D` | EMA of the weights (e.g. 0.999), sharded across GPUs (DiT ~2 GiB/GPU on 4 GPUs, VLA ~4.3 GiB under `--fsdp`). Each checkpoint also writes `<step>_ema_model.pt` (model-only, deploy-ready) and `last_ema_model.pt`; `--resume-from <step>.pt` restores the EMA from it. Validation numbers are for the raw weights, not the EMA. |

- DiT at batch 16/GPU has under 1 GiB free, so EMA needs `--batch-size 12` (probe first).
- `--resume-from` warns if `--grad-accum-steps` differs from the checkpoint (the data stream would not line up).
- Accumulation turns off DDP `static_graph` for the DiT (it cannot record its graph under `no_sync`); FSDP still reduce-scatters every micro-batch so no unsharded gradients are held.

```bash
# DiT: effective batch 16 x 4 x 4 = 256, decay over the last 15%, EMA
... train.py --load-pretrained --batch-size 12 --grad-accum-steps 4 \
  --optim.learning-rate 1e-5 --optim.lr-decay-fraction 0.15 --optim.ema-decay 0.999 ...
# VLA probe: raise --batch-size until peak ~36 GiB, then set --grad-accum-steps
... train.py --policy vla --load-pretrained --pretrained-ckpt-name vla_abc130k_200000_v2.pt \
  --fsdp --flow.num-diffusion-draws 4 --batch-size 8 --train-steps 50 --output-dir checkpoints/vla_probe
```

### 3b. Unfold runs (instance 35.86.74.171)

Both use batch 16/GPU, lr 1e-5 (the paper's finetune rate), the single prompt
`unfold`, and the same command shape as above with `--cache-root` pointing at
the unfold data (scripts are on the instance, started in tmux):

| Local folder | Script / tmux | GPUs | Data | Steps | Checkpoints |
|---|---|---|---|---|---|
| `unfold-2hr-data` | `~/run_unfold_all.sh` / `abc_unfold` | 0-7 | first batch: 138 episodes, 4.9 h | 30k | 10k, 20k, 30k (only `30000_model.pt` kept locally) |
| `unfold-10hr-data` | `~/run_unfold_all2.sh` / `abc_unfold2` | 0-3 | first batch + 103 linen episodes: 241 episodes, 14.4 h | 100k | every 10k (local: 30k, 70k, 100k) |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node 4 train.py \
  --cache-root /opt/dlami/nvme/abc_unfold_all2 \
  --load-pretrained --mixture-preset yam --batch-size 16 --num-workers 16 \
  --train-steps 100000 --ckpt-every 10000 --optim.learning-rate 1e-5 \
  --output-dir checkpoints/$RUN --log-wandb --wandb-project SK-ablation
```

Watch a run: `tmux attach -t abc_unfold2` (detach with Ctrl-b d), or `tail -f
~/sk_vla/abc/checkpoints/<run>.log`. About 2.75 it/s per 4 GPUs.

## 4. Fetch a checkpoint

```bash
# on the instance/box: 24 GB training state -> 8 GB model-only
cd ~/sk_vla/abc && uv run scripts/strip_checkpoint.py checkpoints/<run>/{30000,70000,100000}.pt

# local PC: one stream per file (about 10 MB/s in total), then verify before deleting anything remotely
K=~/.ssh/h1pro-training-us-west-2.pem; H=ubuntu@35.86.74.171     # old box: ap-northeast-2 key, 43.202.110.152
RUN=<run>; D=~/Desktop/EW/abc/checkpoints/<local name>; mkdir -p $D
for s in 30000 70000 100000; do rsync -a --partial -e "ssh -i $K" $H:sk_vla/abc/checkpoints/$RUN/${s}_model.pt $D/ & done; wait
ssh -i $K $H "cd ~/sk_vla/abc/checkpoints/$RUN && sha256sum *_model.pt"; (cd $D && sha256sum *_model.pt)   # must match
```

## 5. Serve to robot_class, and which prompt to use

**The prompt must equal the training prompt exactly.** A different or empty
prompt is out of distribution. The adapter passes the runner's prompt on every
request, so set it to the same string in **both** places: the server's
`--policy.prompt` and the runner's `--prompt`.

| Checkpoint | Trained on | Prompt |
|---|---|---|
| `checkpoints/unfold-10hr-data/{30000,70000,100000}_model.pt` | all unfold data: 241 episodes, 14.4 h (handkerchief, towel stacking, hand and foot towel, pillow case) | `unfold` |
| `checkpoints/unfold-2hr-data/30000_model.pt` | first batch: 138 episodes, 4.9 h (2.0 h handkerchief + 2.9 h towel stacking) | `unfold` |
| `checkpoints/abc-yam-bs16-20261007-1833/{35000,70000}_model.pt` | 15-task multi-task run | the task's lowercased name, list below |

The folder names are the nominal batches (the first ~2 h of handkerchief data,
then +10 h added); the hours actually trained on are in the table. For the
unfold models one prompt covers every object, so the behaviour comes from what
the cameras see (a coloured cloth, a white towel pile, a pillow case).

```bash
# abc (~8 GB VRAM, ~200 ms/request at 10 diffusion steps on an RTX 5080)
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_robot_class.py \
  --policy.checkpoint-path=checkpoints/unfold-10hr-data/70000_model.pt \
  --policy.diffusion-steps=10 --policy.prompt='unfold' --port 8000

# robot_class (dry run; add --enable-motion once actions look sane)
python -m inference.yam.run --left-channel can_follower_l --right-channel can_follower_r \
  --remote-host 127.0.0.1 --control-mode joint --execution-horizon 16 --prompt 'unfold'
```

RTC as run on yam-007 (2026-10-09; server on the GPU box, client on the station):

```bash
# GPU box 192.168.123.1
cd ~/Desktop/EW/abc && source scripts/activate_env.sh
uv run --extra deploy deploy/serve_robot_class.py \
  --policy.checkpoint-path=checkpoints/unfold-10hr-data/100000_model.pt \
  --policy.device=cuda --policy.diffusion-steps=10 --policy.prompt='unfold' \
  --clip.cache-dir=cache/clip --port=8000 \
  --rtc --execution-horizon=23 --inference-delay=7 \
  --trace-path=outputs/unfold_rtc_trace.jsonl --trace-interval=0.5

# station yam-007
cd ~/robot_class
scripts/run_yam_client.sh --left-channel can_follower_l --right-channel can_follower_r \
  --remote-host 192.168.123.1 --remote-port 8000 --prompt 'unfold' \
  --control-mode joint --joint-obs --control-hz 30 --execution-horizon 23 --rtc \
  --camera-config configs/yam_cameras.yaml --include-wrist-cams \
  --enable-motion --interactive --interactive-start-paused --no-home-ramp
```

- `--trace-path` writes one JSONL record per `--trace-interval` seconds: observed joints, predicted chunk, latency, and the largest joint jump at chunk steps 0/15/29.
- Execution horizon + inference delay must be at most 30. 23 + 7 replans every ~0.8 s; for grasping, try `--execution-horizon=8` (replans every ~0.27 s, more correction from the wrist cameras).
- Result: 0 % success on unfold with every checkpoint. The models are overfit: on held-out episodes 30k, 70k and 100k are equally good (open-loop error 0.090-0.094 rad vs 0.114 for holding still) and only slightly better than standing still; on training episodes the error is 0.028. Use 30k (lowest val_loss). Details and next steps: `plan_unfold.md`.
- robot_class's joint limiter (`robot/joint_trajectory.py`: 1 rad/s, 3 rad/s², commands anchored at the measured position) is slower and grips more weakly than teleop; raise `--max-joint-velocity-rad-s` / `--max-joint-acceleration-rad-s2` if motion looks sluggish. ABC's own stack (sections 7-8) does not use it.
- Start from the home pose with grippers open (the recordings start there, not at ABC's `init_q`), keep the 640x480 head + wrist cameras (head = top, wrist_left = left, wrist_right = right), and set `enable_depth: false` in robot_class's `configs/yam_cameras.yaml`.

Multi-task checkpoint prompts (lowercased training task names): `build cup
pyramid`, `collect pen`, `fold handkerchief`, `handover`, `peg in hole`,
`pick up bottle`, `sort block`, `sort column toy`, `sort wall plug`,
`stack cup`, `t pack bolt m12`, `t pack nut m12`, `t pack washer m16`,
`t packing`.

## 6. ABC-VLA (Gemma/SigLIP, 4B)

Measured on the RTX 5080 (16 GB), base model, 2026-10-09: loads in 3 s, **8.3 GiB peak**,
**~50 ms/request** at 10 diffusion steps (47 ms at 5, 46 ms for an RTC request); about 4x
faster than ABC-DiT. So it serves on a 16 GB card; the "~24 GB" in `deploy/README.md` is
conservative. Finetuning needs 80 GB GPUs.

```bash
uv run prepare.py --vla-pretrained --cache cache        # 8.8 GB -> cache/vla_abc130k_200000_v2.pt
uv run --extra deploy deploy/serve_policy.py \
  --policy.checkpoint-path=<vla checkpoint> --policy.prompt='<training prompt>' --port 8000
```

`serve_policy.py` detects DiT vs VLA from the checkpoint. It speaks ABC's protocol
(`deploy_policy.py`, `dagger.py`), not robot_class's: `serve_robot_class.py` is DiT only.

## 7. DAgger: GPU box serves, station runs the robot

Full operator guide: **`dagger_guide.md`** (controls, takeovers, export, training a round).
`dagger.py` must run on the machine with the CAN adapters and cameras (its processes talk over
local IPC); the policy server runs on the GPU box. Stop robot_class on the station first.

```bash
# GPU box 192.168.123.1: ABC's own server, not serve_robot_class.py
cd ~/Desktop/EW/abc && source scripts/activate_env.sh
uv run --extra deploy deploy/serve_policy.py \
  --policy.checkpoint-path=checkpoints/unfold-10hr-data/30000_model.pt \
  --policy.prompt='unfold' --policy.diffusion-steps=10 --port 8000

# station yam-007, inside tmux (a dropped SSH session must not kill a run)
cd ~/Desktop/EW/abc
ROBOT_PROFILE=yam007_config uv run --extra deploy deploy/dagger.py \
  --remote-host=192.168.123.1 --port=8000 --prompt='unfold' \
  --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 --execute-chunk-dim=16 \
  --compress-images
```

- Controls: right handle top = **a** (start / take over / hand back), right handle second =
  **c** (checkpoint, or save after ending), left handle top = **b** (end). The keyboard in
  the `dagger.py` terminal works too; a PCsensor pedal is used if plugged in.
- Don't hold a handle button while it starts (each handle learns its buttons' idle level).
- Monitoring: `[DAGGER] ...` lines in the terminal; after each saved episode a review video
  `data/dagger_video/<file>.mp4` (who was driving, checkpoints, what the export keeps).
- First time on yam-007: install ABC there (`uv sync --extra deploy`) and run the teleop check
  `ROBOT_PROFILE=yam007_config uv run deploy/robot/scripts/run_teleop.py` (both followers must
  track both leaders and triggers). Tested offline only so far.

## 8. Standalone: one GPU box runs everything

Possible because ABC's stack drives the arms itself through i2rt (followers, leaders,
cameras, recording); robot_class is not needed. Without `--remote-host`, `dagger.py` (and
`deploy_policy.py`) start the policy server as a local process. ~8 GB of GPU memory for
ABC-DiT or ABC-VLA inference.

Trade-off: the split setup (section 7) costs only 1-3 ms on the wired link and leaves
yam-007's data-collection stack (`~/station_YAM`) untouched, which the new demonstrations
need. Go standalone for fewer machines, or if `inference still pending` stalls show the
network is the bottleneck.

**8.1 Hardware.** Move the 4 CAN-USB adapters (gs_usb) and the 3 RealSense cameras to the GPU
box. Cameras on USB 3, spread over controllers rather than one hub (a camera that fails to
start or drops frames usually means bandwidth). Label each CAN cable before unplugging.

**8.2 System packages** (Ubuntu):

```bash
sudo apt install can-utils
lsmod | grep gs_usb || sudo modprobe gs_usb       # in-kernel driver for the CAN adapters
# RealSense udev rules, so cameras open without root (yam-007 uses librealsense 2.58.4):
#   install librealsense2-udev-rules and librealsense2-utils from Intel's apt repo, or copy
#   /etc/udev/rules.d/60-librealsense2-udev-rules.rules from yam-007
rs-enumerate-devices -s       # expect head D455 260722303224, wrists D405 352122272270 / 352122273271
```

**8.3 Fixed CAN names.** The profile expects `can_leader_l/r` and `can_follower_l/r`, UP at
1 Mbps. yam-007 does this with a udev rule (rename by USB port) plus a systemd unit (bring up):

```bash
scp robo@192.168.0.199:/etc/udev/rules.d/80-yam-can.rules /tmp/
scp robo@192.168.0.199:/etc/systemd/system/can-bringup@.service /tmp/
sudo cp /tmp/can-bringup@.service /etc/systemd/system/ && sudo systemctl daemon-reload
```

The rule matches **USB port paths** (`KERNELS=="3-2.2.4"`), which are different on a new
machine, so re-derive them before installing the rule. Plug the labelled adapters in **one at
a time** and after each:

```bash
ip -br link | grep can                                   # the new canX
ip -d link show canX | grep -o 'parentdev [^ ]*'         # e.g. parentdev 3-2.2.4:1.0 -> KERNELS=="3-2.2.4"
```

Edit the four `KERNELS==` values in `/tmp/80-yam-can.rules` (keep the names and
`SYSTEMD_WANTS`), then:

```bash
sudo cp /tmp/80-yam-can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ip -br link | grep can                                    # all four names, UP
ip -d link show can_follower_l | grep -o 'state [A-Z-]*'  # ERROR-ACTIVE, not DOWN
```

Without labels: followers answer on motor id 7 (gripper) and leaders do not
(`uv run python -m i2rt.motor_config_tool.ping_motors --channel canX`); left/right by moving a
leader by hand, or yam-007's wizard `~/station_YAM/yam_arm/i2rt/calibration_can_id_wizard`.
Moving any adapter to another USB port later silently breaks the names.

**8.4 Install ABC** (as for the GPU box): sync the repo (section 1 pattern, or
`rsync` from this PC), `uv sync --extra deploy`, and the checkpoint in `checkpoints/`.

**8.5 Profile.** `yam007_config` works unchanged if the same cameras and arms moved (serials
and CAN names are the same). For a different rig, copy it in `deploy/robot/config.py` with
that rig's serials.

**8.6 Check, then run.**

```bash
ROBOT_PROFILE=yam007_config uv run deploy/robot/scripts/run_teleop.py     # followers track leaders

# DAgger, all local: no --remote-host, the checkpoint instead
ROBOT_PROFILE=yam007_config CUDA_VISIBLE_DEVICES=0 uv run --extra deploy deploy/dagger.py \
  --checkpoint-path=checkpoints/unfold-10hr-data/30000_model.pt --prompt='unfold' \
  --diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 \
  --execute-chunk-dim=16

# plain policy rollouts (no takeovers), all local
ROBOT_PROFILE=yam007_config CUDA_VISIBLE_DEVICES=0 uv run --extra deploy deploy/deploy_policy.py \
  --checkpoint-path=checkpoints/unfold-10hr-data/30000_model.pt --prompt='unfold' \
  --diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 \
  --execute-chunk-dim=16
```

Watch `htop` during the first run: control loops, 3 camera nodes, inference and recording
now share one machine. A laptop GPU box (like 192.168.123.1, RTX 3080 Ti Laptop) also needs
to stay on AC power with its power profile on performance.

## 9. Local DAgger dry run and setup

See [LOCAL_DAGGER.md](LOCAL_DAGGER.md) for the hardware-free policy + DAgger
integration check, local launch wrapper, system prerequisites, and review viewers.

```bash
cd ~/Desktop/EW/abc
scripts/run_local_dagger.sh --dry-run
```

The default dry run checks the base model and unfold 30k checkpoint with RTC.
Results and review MP4s are saved under `outputs/dagger_dry_run/`.
