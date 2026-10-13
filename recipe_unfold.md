# ABC-DiT finetune recipe: unfold (bimanual YAM)

Finetune of the released ABC-DiT (`abc_dit_xl_200k_model.pt`, the public
ABC-130K / 3.5K h checkpoint) on the local `yam_table_top-sunway_laundry`
recordings (ROS2 rosbag2 MCAPs). Launch pattern as `commands.md`; the cache
root, steps, learning rate and prompt differ.

| What | Value |
|---|---|
| Source | `~/Desktop/EW/dataset/yam_table_top-sunway_laundry/shoonkit/yam_table_top-sunway_laundry/ep_*/ep_*.mcap` (139 episodes, 28 GB, 4.9 h, 528k frames) |
| Format | ROS2 rosbag2 MCAP: CDR `JointState` at ~250 Hz, h264 cameras at 30 Hz (`head_cam`, `wrist_left`, `wrist_right`). **Not** readable by `scripts/export_mcap.py` (protobuf release format only; it stops with `DecoderNotFoundError ... cdr`) |
| Converter | `scripts/export_ros2_mcap.py`: state = measured joints + gripper state, action = commanded joints + gripper command, left arm first, fixed 30 Hz causal clock; cameras stacked top/left/right at 224x224 |
| Split | 95/5, seed 42 (evo's method, as `scripts/make_yam_split.py`), over episodes in chronological order: 132 train / 7 val of 139. Manifest: `<out>/split_ros2_mcap.json` |
| Excluded | `ep_769d_20261006_171146` (16 s, ends with the cloth still in the air); it was in train, so 131 train / 7 val are converted |
| Prompt | **one prompt, `unfold`, for every episode** (see section 1) |
| Operators | kim_yawn, joe, yik_heng, sheng_chong; no operator prompting is used |

## 0. What is in the data (assessment, 2026-10-08)

All 139 episodes carry `task_id: unfold_handkerchief`, QC verdict `warn` and
`outcome: stopped` (no success flag), but they are two different behaviours:

| Kind | Episodes (train/val) | Duration | Content | Frames (train) |
|---|---|---|---|---|
| `handkerchief` | 113 (107/6, after exclusion) | 16-126 s, median ~60 s | one crumpled cloth (orange / yellow / blue / green, plus one white) is lifted and laid flat | 205k (41 %) |
| `towel_stack` | 25 (24/1) | 228-532 s | white; several crumpled towels are unfolded one after another and piled on one spot | 298k (59 %) |

No episode lies between 126 s and 228 s, so duration separates them
(`TOWEL_STACK_MIN_S = 200`). The 25 towel episodes are `hand towel` in the
LeRobot labels; the raw task id is wrong for them. Each converted episode keeps
its kind in `episode_metadata.json` -> `source.kind`.

Visual check (10 random episodes, `random.Random(42).sample` over the sorted
ids, head-camera frames at every 10 %, plus 3 follow-ups):

- 6 of the 10 are single-cloth unfolds. 5 finish flat; `ep_769d` (16 s) stops with the cloth lifted, so it is excluded.
- 4 of the 10 (`ep_1229`, `ep_3dee`, `ep_1652`, `ep_3edb`, 276-518 s) unfold several towels and stack them; larger frames of `ep_1652` show a multi-layer pile.
- Short is not bad: 24-30 s episodes (`ep_2fdb`, `ep_cc8a`) are complete unfolds. The one white episode under 200 s (`ep_5725`, 72 s) is a single-cloth unfold, so colour alone does not mean stacking.
- Two table surfaces appear (wood and black); both are in the data.

The classification rests on this duration gap plus the visual checks above,
not on a frame-by-frame review of all 139.

## 1. Convert (local PC, done 2026-10-08)

```bash
cd ~/Desktop/EW/abc
uv run scripts/export_ros2_mcap.py \
  --root ~/Desktop/EW/dataset/yam_table_top-sunway_laundry/shoonkit/yam_table_top-sunway_laundry \
  --out-dir ~/Desktop/EW/dataset/abc_unfold \
  --exclude ep_769d_20261006_171146 --workers 12      # add --task-name unfold to bake in one prompt
```

- ~1 GB RAM per worker, CPU-bound on x264; the full run took ~25 min on 12 workers. Output is 1.7 GB.
- `--kinds handkerchief` exports only that kind (the split is drawn first, so no episode changes side).
- Validation: on `ep_e799` (also LeRobot unfold episode 0) state/action match the LeRobot export to 0.0002 rad with the same 1095 steps (constant 2-tick offset between the two time grids), and the stacked video matches in content.
- Camera packets start mid-GOP: the first 7 per camera precede the first keyframe, are undecodable, and are dropped (not re-timed). An episode whose decoded frame count differs from its packet count fails instead of being silently re-spaced.

**Single prompt.** The converter writes kind-specific prompts by default
(`unfold_handkerchief` -> "unfold handkerchief", `unfold_and_stack_towels` ->
"unfold and stack towels"). The training cache was changed to `unfold` for all
138 episodes by rewriting `task_name` in each `episode_metadata.json` (no
re-conversion), on the instance and then locally (pulled from the instance, so
the two are identical). The originals are in
`task_name_before_single_prompt.json` next to the episodes. To go back to two
prompts:

```bash
python3 - <<'EOF'
import json
D = "/opt/dlami/nvme/abc_unfold"     # or ~/Desktop/EW/dataset/abc_unfold, with the paths in the json rewritten
for p, name in json.load(open(f"{D}/task_name_before_single_prompt.json")).items():
    m = json.load(open(p)); m["task_name"] = name; json.dump(m, open(p, "w"), indent=2)
EOF
```

The trade-off: with one prompt the model tells a single coloured cloth from a
pile of white towels by sight alone, which is untested here; see section 5.

## 2. Instance setup (done 2026-10-08 on ubuntu@35.86.74.171, 8x A100 40GB)

Local side, key `~/.ssh/h1pro-training-us-west-2.pem`:

```bash
K=~/.ssh/h1pro-training-us-west-2.pem; H=ubuntu@35.86.74.171
cd ~/Desktop/EW/abc && git ls-files -co --exclude-standard > /tmp/abc_files.txt
rsync -a -e "ssh -i $K" --rsync-path="mkdir -p ~/sk_vla/abc && rsync" --files-from=/tmp/abc_files.txt ./ $H:~/sk_vla/abc/
rsync -a -e "ssh -i $K" uv.lock $H:~/sk_vla/abc/uv.lock          # gitignored, so not in the file list
rsync -a --partial -e "ssh -i $K" --rsync-path="mkdir -p /opt/dlami/nvme/abc_unfold && rsync" \
  ~/Desktop/EW/dataset/abc_unfold/ $H:/opt/dlami/nvme/abc_unfold/   # 1.7 GB, ~11 min at ~2.5 MB/s
```

Instance side:

```bash
sudo apt-get install -y ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/sk_vla/abc && uv python pin 3.12 && uv sync
uv run prepare.py --pretrained --cache /opt/dlami/nvme/abc_unfold   # 8 GB, sha256-verified, ~2 min
```

- wandb: the key and entity files are copied to `~/sk_vla/molmoact2/experiments/.wandb_api_key` and `.wandb_entity` (mode 600, from the local `molmoact2/experiments/`), the same paths `commands.md` reads. Project `SK-ablation`.
- NVMe contents (`/opt/dlami/nvme`) are lost when the instance stops; rsync the checkpoints you want off it first.

## 3. Train

### Run launched 2026-10-08: all kinds, single prompt, paper learning rate

Script `~/run_unfold_all.sh`, started in tmux `abc_unfold`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(cat ~/sk_vla/molmoact2/experiments/.wandb_api_key)
export WANDB_ENTITY=$(cat ~/sk_vla/molmoact2/experiments/.wandb_entity)
cd ~/sk_vla/abc
RUN=abc-unfold-all-1prompt-lr1e-5-bs16x8-$(date +%Y%m%d-%H%M); export WANDB_NAME=$RUN
uv run torchrun --standalone --nproc-per-node 8 train.py \
  --cache-root /opt/dlami/nvme/abc_unfold \
  --load-pretrained --mixture-preset yam --batch-size 16 --num-workers 12 \
  --train-steps 30000 --ckpt-every 10000 --optim.learning-rate 1e-5 \
  --output-dir checkpoints/$RUN --log-wandb --wandb-project SK-ablation \
  2>&1 | tee checkpoints/$RUN.log
```

| Setting | Value | Note |
|---|---|---|
| Data | 131 train episodes, 499k usable frames (41 % handkerchief, 59 % towel stacking) | val: 7 episodes |
| Batch | 16 per GPU x 8 = 128 | 30k steps = 3.84M samples = ~7.7 epochs |
| Learning rate | 1e-5 for vision and action head, 1000-step warmup, weight decay 0.01 | the paper's finetune setting (App. E.2) |
| Checkpoints | 10k, 20k, 30k | 24 GB training state each, in `checkpoints/<run>/` |
| Speed | 2.74 it/s, 38.1 GiB peak (40.2 of 40.96 GiB shown by nvidia-smi) | ~3.0 h of steps, ~3.2 h with saves and validation |
| Validation | every 2500 steps (default) | |

- Monitor: `tmux attach -t abc_unfold`, the `.log` next to the checkpoints, and wandb run `abc-unfold-all-1prompt-lr1e-5-bs16x8-20261008-0838`.
- The startup log says `using norm_stats embedded in the checkpoint` and `prompts=[((0, 'unfold'),)]`.
- Memory headroom is under 1 GiB; if it fails at the first validation or a checkpoint save, rerun with `--batch-size 12`.
- Why 1e-5: the paper finetunes at 1e-5 for both vision and action expert (pretraining used 1e-4 with a 0.1 vision scale, so 1e-5 is also the pretraining vision rate). The earlier 70k multi-task run used 5e-5 for both, 5x the paper's value. The paper's batch is 1,440 and 200k steps; this run is 128 and 30k, so only the learning rate follows it.

### Alternative, not run: handkerchief only

The towel-stack episodes are 17 % of the episodes but 59 % of the frames, and
frames are sampled uniformly, so the run above spends most samples on the
longer stacking behaviour. A cache holding only the handkerchief episodes
(107 train / 6 val, 205k frames) makes `--mixture-preset yam` a single-kind run:

```bash
python3 - <<'EOF'
import json, os, glob
SRC, DST = "/opt/dlami/nvme/abc_unfold", "/opt/dlami/nvme/abc_unfold_hk"
for split in ("train_real", "val_real"):
    os.makedirs(f"{DST}/{split}", exist_ok=True)
    for d in sorted(glob.glob(f"{SRC}/{split}/episode_*")):
        if json.load(open(f"{d}/episode_metadata.json"))["source"]["kind"] == "handkerchief":
            os.symlink(d, f"{DST}/{split}/{os.path.basename(d)}")
for f in ("abc_dit_xl_200k_model.pt", "abc_dit_xl_200k_model.json"):
    os.symlink(f"{SRC}/{f}", f"{DST}/{f}")
EOF
```

Train it as above with `--cache-root /opt/dlami/nvme/abc_unfold_hk`,
`--train-steps 15000 --ckpt-every 2500` (~4.7 epochs at batch 64 on 4 GPUs,
~1.5 h at 2.8 it/s). Only one change per run if you try `--flow.mask-state-ratio 0.3`.

## 4. Fetch and serve

Strip and rsync as `commands.md` step 4 (24 GB training state -> 8 GB model),
for 10k / 20k / 30k. Serve with the training prompt verbatim:

```bash
CUDA_VISIBLE_DEVICES=0 uv run deploy/serve_robot_class.py \
  --policy.checkpoint-path=checkpoints/<run>/20000_model.pt \
  --policy.diffusion-steps=10 --policy.prompt='unfold' --port 8000
```

Start each rollout from robot_class's home pose, grippers open; the episodes
start there too.

## 5. Evaluate

- Compare 10k / 20k / 30k by watching rollouts, as the paper does (it picks the best of checkpoints 50k steps apart by visual inspection).
- Both scenes, since one prompt covers both: fixed trials per cloth colour (orange, yellow, blue, green) from a crumpled start, and the white-towel pile. Note whether a single-cloth run ever tries to stack, or a towel run stops after one towel.
- Chunked first (`--execution-horizon 16`), then RTC. The paper's t-shirt result favours a short inference prefix: prefix 1 scored 4.6/5 with 8/10 completions, prefix 4 scored 3.9/5 with 5/10 (App. H.2, Table 5); it says the policy "overfits to the prefix" and ignores what it sees. `serve_robot_class.py` currently ties the prefix to `--inference-delay` (default 6), so decouple them before trying RTC.
- Held-out signal: the 7 val episodes (val loss every 2500 steps).

## 6. After the baseline: more data or DAgger

Paper reference (abc.bot/abc.pdf, §5.2-5.3, App. E and H): finetuning from the
public checkpoint averaged 0.43 mean progress over four dexterous tasks (0.25
from scratch, 0.61 from a 7K-hour internal checkpoint); box folding with 10 h
of data reached 24% and ~85% after DAgger rounds of 1-1.5 h each, mixed 80 %
previous data / 10 % interventions / 10 % rest of the rollouts, never
interventions alone. This dataset's handkerchief part is ~2 h, so expect a
rough baseline.

| What you see in rollouts | Next step |
|---|---|
| Never gets going, wrong grasp, fails by colour or table surface | more demos (the station and converter are ready) |
| Right behaviour but stalls or cannot recover mid-task | DAgger |
| Stalls only with RTC | shorter prefix |

### DAgger on station yam-007

Set up 2026-10-09 with ABC's `deploy/dagger.py` and robot profile
`yam007_config` (YAM teaching-handle leaders, handle buttons as a/b/c). Setup,
commands, controls, takeovers, export and training a round: **`dagger_guide.md`**.

## 7. Result of the launched run (finished 2026-10-08)

30k steps done in ~3.1 h (2.7-2.8 it/s). Validation on the 7 held-out episodes:

| Step | val_recon_error | val_loss |
|---|---|---|
| 2.5k | 0.1035 | 0.0829 |
| 10k | 0.0896 | 0.0972 |
| 20k | 0.0863 | 0.1239 |
| 30k | 0.0845 | 0.1477 |

`val_loss` (flow-matching loss) rose steadily while `val_recon_error` improved and
flattened after ~22.5k, so the two disagree; choose between 10k / 20k / 30k by
rollouts, not by either number. The model-only checkpoints are on the local PC in
`checkpoints/unfold-2hr-data/` (renamed from `abc-unfold-all-1prompt-lr1e-5-bs16x8-20261008-0838`;
the 10k, 20k and 30k files were sha256-verified against the instance when downloaded, but
only `30000_model.pt` is in the folder now) with the training log as `unfold-2hr-data.log`.
"2hr" is the nominal first batch; the run actually trained on 138 episodes, 4.9 h. The
full 24 GB training states on the instance were deleted, so this run cannot be
resumed; to continue from it, start a new run with `--load-pretrained
--pretrained-ckpt-name <path to a *_model.pt>` (weights only, fresh optimizer).

## 8. Added batch: unfold_linen (converted 2026-10-08)

`forge download shoonkit/yam_table_top-sunway_laundry` now holds 242 episodes: the
original 139 plus 103 new ones (9.5 h, 1.03M frames) with `task_id: unfold_linen`.

| Type tag | Episodes | Hours | Duration (min / median / max) |
|---|---|---|---|
| hand_towel | 45 | 4.2 | 108 / 328 / 518 s |
| foot_towel | 32 | 3.0 | 140 / 345 / 536 s |
| pillow_case | 26 | 2.4 | 80 / 353 / 479 s |

- Contact sheets of 6 of them (all three types) show white linens on white or wooden tables unfolded from crumpled starts; the towel episodes also pile or fold the result, as in the earlier towel-stack set. The pillow-case ends were hard to read. Not reviewed beyond those 6.
- The duration rule from section 0 does **not** apply to this batch (16 episodes fall in the old 126-228 s gap), so the converter now takes `kind = linen` from the task id; the item type is stored in each episode's `source.type`.
- They are a different task family from the 139 handkerchief/towel-stack episodes, so they went into their own folder and the existing split is untouched (the handkerchief-only plan still reproduces the old split exactly).

```bash
uv run scripts/export_ros2_mcap.py \
  --root ~/Desktop/EW/dataset/yam_table_top-sunway_laundry/shoonkit/yam_table_top-sunway_laundry \
  --out-dir ~/Desktop/EW/dataset/abc_unfold_linen \
  --task-ids unfold_linen --task-name unfold --workers 10      # ~50 min, 3.4 GB
```

- Split: 95/5, seed 42, over the 103 new episodes in chronological order: 98 train / 5 val (2 pillow_case, 2 foot_towel, 1 hand_towel; 4 of the 5 are operator joe). Manifest: `abc_unfold_linen/split_ros2_mcap.json`.
- Prompt: `unfold` for all, as in the launched run. Change `task_name` per episode if you want separate prompts (type and kind are in the metadata).
- Where: local `~/Desktop/EW/dataset/abc_unfold_linen`, instance `/opt/dlami/nvme/abc_unfold_linen` (checksum-verified, 98 train / 5 val, 959k train samples, loads through the dataloader).
- Training on old and new together is a cache that links both folders' episodes into one `train_real` / `val_real` (the two val sets stay disjoint from their own train sets):

```bash
python3 - <<'PY'
import os, glob
DST = "/opt/dlami/nvme/abc_unfold_all2"
for split in ("train_real", "val_real"):
    os.makedirs(f"{DST}/{split}", exist_ok=True)
    for src in ("/opt/dlami/nvme/abc_unfold", "/opt/dlami/nvme/abc_unfold_linen"):
        for d in sorted(glob.glob(f"{src}/{split}/episode_*")):
            os.symlink(d, f"{DST}/{split}/{os.path.basename(d)}")
for f in ("abc_dit_xl_200k_model.pt", "abc_dit_xl_200k_model.json"):
    os.symlink(f"/opt/dlami/nvme/abc_unfold/{f}", f"{DST}/{f}")
PY
```

## 9. Run 2 (launched 2026-10-08): all unfold data, single prompt, GPUs 0-3, 100k steps

Script `~/run_unfold_all2.sh` on the instance, tmux `abc_unfold2`, run `abc-unfold-all2-1prompt-lr1e-5-bs16x4-20261008-1456`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node 4 train.py \
  --cache-root /opt/dlami/nvme/abc_unfold_all2 \
  --load-pretrained --mixture-preset yam --batch-size 16 --num-workers 16 \
  --train-steps 100000 --ckpt-every 10000 --optim.learning-rate 1e-5 \
  --output-dir checkpoints/$RUN --log-wandb --wandb-project SK-ablation
```

- Data: `/opt/dlami/nvme/abc_unfold_all2` = symlinks to `abc_unfold` (handkerchief + towel-stack, 131 train / 7 val) and `abc_unfold_linen` (98 train / 5 val): 229 train / 12 val episodes, 1.46M train samples, every episode prompted `unfold`. Both source folders were checksum-compared with the local copies before launch (identical).
- Base: `abc_dit_xl_200k_model.pt`, the public ABC-DiT XL parent at step 200,000 (see the sidecar json: 3,553 h real + 98 h sim, sha256 79754ece...).
- Batch 16 x 4 = 64; 100k steps = 6.4M samples = ~4.4 epochs; ~2.8 it/s, so ~10 h plus checkpoint saves (24 GB each, every 10k). Validation every 2500 steps on the 12 val episodes (not comparable with run 1's 7).
- GPUs 4-7 are free. Folder map on the instance: `abc_unfold` (first batch, real files), `abc_unfold_hk` (symlinks to its handkerchief-only subset, unused), `abc_unfold_linen` (second batch, real files), `abc_unfold_all2` (symlinks to the first and linen folders).

### Run 2 result (finished 2026-10-09)

100k steps in ~10.4 h (2.75 it/s on GPUs 0-3). Validation on the 12 held-out episodes:

| Step | val_recon_error | val_loss |
|---|---|---|
| 30k | 0.1111 | 0.0908 |
| 70k | 0.1075 | 0.1263 |
| 100k | 0.1035 | 0.1446 |

Same pattern as run 1: `val_recon_error` creeps down while `val_loss` rises, so choose by rollouts.
Local model-only checkpoints, sha256-verified against the instance: `checkpoints/unfold-10hr-data/{30000,70000,100000}_model.pt`
(training log `checkpoints/unfold-10hr-data.log`). "10hr" is the nominal added batch; the run trained on all
241 episodes, 14.4 h. The instance still holds the full 24 GB states for 10k-100k in
`~/sk_vla/abc/checkpoints/abc-unfold-all2-1prompt-lr1e-5-bs16x4-20261008-1456/` (resumable), plus the stripped
30k/70k/100k `_model.pt` files. Serve with prompt `unfold`; see `commands.md` section 5.
