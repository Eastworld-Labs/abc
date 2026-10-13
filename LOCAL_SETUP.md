# Local environment and YAM deployment

Prepared on 2026-10-07 for this checkout.

```bash
cd /home/ubuntu/Desktop/EW/abc
source scripts/activate_env.sh
# Keep the hardware extras installed when syncing or running through uv.
uv sync --extra deploy
uv run --extra deploy prepare.py --sim-checkpoint-list
```

The activation script selects `.venv`, the repo-local uv cache and managed
Python 3.12, and the local FFmpeg binaries and shared libraries. The latter
are extracted Ubuntu packages under `.uv/ffmpeg/root`; no sudo password or
system package changes were needed. Source the script in every new shell,
including before `uv run`, so TorchCodec can find FFmpeg's shared libraries.
The `scikit-build-core<0.10` build constraint in `pyproject.toml` fixes
`ruckig==0.15.3`'s obsolete `cmake.targets` setting. Managed Python supplies
the development headers missing from the system Python installation.

The host GPU is an RTX 3080 Ti Laptop GPU with nominal 16 GB VRAM. Sandbox processes
cannot access it; a normal host terminal can. The ABC-DiT base checkpoint and
CLIP text assets are downloaded and their weight checksums verified. Each
model-only DiT checkpoint is approximately 8.1 GB. Datasets have not been downloaded.

Verified Python 3.12.15, PyTorch 2.11.0+cu128 GPU matrix multiplication,
TorchCodec 0.11.0+cpu decoding of an H.264 video, a MuJoCo physics step,
and imports of the YAM drivers and serving adapter.

## ABC base T-shirt adapter with RTC

The selected prompt is `folding tshirt pile and stacking`. Start the server
from this checkout:

```bash
source scripts/activate_env.sh
uv run --extra deploy deploy/serve_robot_class.py \
  --policy.checkpoint-path=cache/abc_dit_xl_200k_model.pt \
  --policy.device=cuda --policy.diffusion-steps=5 \
  --policy.prompt='folding tshirt pile and stacking' \
  --clip.cache-dir=cache/clip --port=8000 \
  --rtc --execution-horizon=16 --inference-delay=8
```

The background server started from this chat listens at `ws://127.0.0.1:8000`
(all host interfaces). Its PID is in `outputs/yam_adapter/server.pid` and
logs are in `outputs/yam_adapter/server.log`. To inspect health:

```bash
curl http://127.0.0.1:8000/healthz
tail -f outputs/yam_adapter/server.log
```

In the separate `robot_class` environment, use `--rtc` on the YAM runner;
the server advertises execution horizon 16 and inference delay 8. Set the
runner prompt to the same T-shirt task. The measured warm synthetic RTC
requests with 5 diffusion steps took approximately 185–200 ms on this GPU;
the 8-step delay allows 267 ms at 30 Hz. Measure latency again on your actual
network. Synthetic inference verifies the adapter contract, not folding success.

Stop this background instance before starting another server on port 8000:

```bash
kill "$(cat outputs/yam_adapter/server.pid)"
```

## Published model catalogue

The publisher's live manifest is
<https://abc-data.timehorizons.org/finetuned_sim/manifest.json>.
The cached snapshot is `cache/finetuned_sim_manifest.json`; it contains 12
DiT and 12 VLA sim finetunes, generated on 2026-09-06.

```bash
# Refresh and list published sim models and evaluation results.
uv run --extra deploy prepare.py --sim-checkpoint-list --sim-force
# Download a recommended DiT checkpoint, its metadata, and sim assets.
uv run --extra deploy prepare.py --sim-checkpoint pour
# Download a specific step instead.
uv run --extra deploy prepare.py --sim-checkpoint pour@25000
```

These scores measure simulation performance, not success on your station.
The model-only files default to
`cache/finetuned_sim/<catalogue-key>/<step>.pt`.

| DiT catalogue key | Recommended step | Published sim successes | Manifest prompt |
| --- | ---: | --- | --- |
| `ball_tray_balancing` | 25000 | 1/20 + 0/20 | `grab the tray handles and keep the ball balanced on the tray` |
| `conveyor_pick` | 25000 | 2/20 + 2/20 | `conveyor pick` |
| `count_into_opaque_box` | 25000 | 1/20 + 0/20 | `count into opaque box` |
| `grab_clutter` | 25000 | 5/20 + 3/20 | `grab clutter` |
| `lego_blocks_sorting` | 25000 | 0/20 + 1/20 | `lego blocks sorting` |
| `nuts_bolts_sorting` | 25000 | 0/20 + 0/20 | `nuts bolts sorting` |
| `pour` | 25000 | 13/20 + 16/20 | `sim pouring beads` |
| `put_relative` | 25000 | 5/20 + 10/20 | `put relative` |
| `sim_inhand_transfer_the_item_to_other_side` | 25000 | 18/20 + 18/20 | `sim inhand transfer the item to other side` |
| `sim_load_the_plates_into_the_dish_rack` | 20000 | 15/20 + 14/20 | `sim load the plates into the dish rack` |
| `sim_put_the_plastic_bottles_in_the_bin` | 20000 | 19/20 + 18/20 | `sim put the plastic bottles in the bin` |
| `sim_sweep_away_paper_scraps_from_the_table` | 25000 | 2/20 + 1/20 | `sim sweep away paper scraps from the table` |

`put_relative` and `grab_clutter` use per-episode directive prompts; their
sim environments replace the generic manifest prompt at reset.

For physical YAM trials, start with the bottles policy or a real-robot
DAgger finetune matching your task. The bottles release is trained from
scratch for its task, rather than being a finetune of the multitask parent.

```bash
# Bottles task policy, preview data, and simulator assets.
uv run --extra deploy prepare.py --checkpoint
# Multitask parent for finetuning or separate zero-shot experiments.
uv run --extra deploy prepare.py --pretrained
# Real-robot DAgger finetune; prints the exact original deployment flags.
uv run --extra deploy prepare.py --dagger-checkpoint folding_paper_box
```

| Real-robot DAgger key | Task | Step | Arm kp scale |
| --- | --- | ---: | --- |
| `folding_paper_box` | fold the paper box | 75000 | default |
| `insert_earbuds_1eb` | insert the wireless earbuds into their charging case (single earbud) | 5000 | default |
| `insert_earbuds_3eb` | insert the wireless earbuds into their charging case (three earbuds) | 5000 | default |
| `pack_student_bag` | pack a student bag | 10000 | 2x |
| `using_a_key` | unlock a box with a key | 5000 | 2x |

DAgger models land in `cache/dagger/<key>.pt`. All five use task-specific
initial joints and exact prompts in their metadata, plus 5 diffusion steps,
RTC prefix/lead 4, and 20 executed actions. Their recordings were made
without the standard station cage; transfer depends on camera and scene
placement. Read the downloaded metadata or the printed deploy command
before configuring your station.

## YAM deployment path

The native station stack is documented in [`deploy/README.md`](deploy/README.md).
Set up your own `ROBOT_PROFILE` in `deploy/robot/config.py` with the correct
camera, CAN, and GELLO identifiers before using it. The supplied profiles
refer to the authors' stations.

If you already use the separate `robot_class` YAM runner, this checkout
includes an adapter at `deploy/serve_robot_class.py`:

```bash
# ABC checkout on the inference host, after downloading the bottles model.
source scripts/activate_env.sh
CUDA_VISIBLE_DEVICES=0 uv run --extra deploy deploy/serve_robot_class.py \
  --policy.checkpoint-path=cache/bottles_75k.pt \
  --policy.diffusion-steps=10 \
  --policy.prompt='throw plastic bottles in bin'

# In your separately configured robot_class environment: dry run.
python -m inference.yam.run \
  --left-channel can_follower_l --right-channel can_follower_r \
  --remote-host 127.0.0.1 --control-mode joint --execution-horizon 16 \
  --prompt 'throw plastic bottles in bin'
```

Replace the channels and inference host with your station's values. The
runner keeps motion disabled until you explicitly add `--enable-motion`.
The separate `robot_class` repository is not installed by this setup.

Match the ABC observation/action contract: a top camera and two wrist
cameras at 640x480 RGB, 14-D left-first joints and grippers at 30 Hz. For
the bottles policy the documented initial pose is
`[0, 0.5, 1.0, -1.0, 0, 0, 1.0]` per arm. Match camera placement, arm
mounting, objects, and the training prompt when assessing zero-shot transfer.
Changing a prompt does not make a task-specific policy support arbitrary tasks.

DiT inference downloads its CLIP text assets on first use (the default
cache is `~/.cache/clip`; `--clip.cache-dir=cache/clip` keeps them local).
The released checkpoint includes the DINO vision weights and normalization
statistics, so you do not need a separate DINO download for inference.

Sources: [upstream setup](https://github.com/amazon-far/abc/blob/main/README.md),
[deployment and DAgger models](https://github.com/amazon-far/abc/blob/main/deploy/README.md),
[simulation model catalogue](https://github.com/amazon-far/abc/blob/main/abc_sim/README.md).
