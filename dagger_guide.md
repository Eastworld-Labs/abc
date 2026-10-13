# DAgger on yam-007 — operator guide

How to run a DAgger session on station yam-007 with ABC's `deploy/dagger.py`: the policy
drives the arms, you take over with the leader arms when it is about to fail, fix the
mistake, and hand back. The takeovers are recorded and become training data for the next
finetune. The model does not learn during the session.

The same loop runs **ABC-DiT, MolmoAct2 or π0.5**. Only the policy server changes (§4).
Each model needs its own sessions: a takeover fixes the mistakes of whichever model was
driving, so collect a model's DAgger data with that model in the loop.

Background (paper numbers, when DAgger is the next step): `recipe_unfold.md` §6. Plan, round sizes and evaluation:
`plan_unfold.md` §5–6.

## 1. What you need

| Item | Notes |
|---|---|
| Station yam-007 | 2 follower YAMs (`can_follower_l/r`), 2 leader YAMs with teaching handles (`can_leader_l/r`), D455 head + 2 D405 wrist cameras |
| GPU box `192.168.123.1` | Serves the policy over wired Ethernet |
| A checkpoint | ABC-DiT (`checkpoints/unfold-10hr-data/30000_model.pt`), a MolmoAct2 HF checkpoint, or an openpi π0.5 checkpoint |
| Controls | **The teaching-handle buttons** (§5). No pedal needed. The keyboard in the launch terminal works too, and a PCsensor foot pedal is picked up if one is ever plugged in. |
| Hardware e-stop | Within reach of the operator, every session |

**No calibration needed.** The leaders are the same YAM arm as the followers, so they share
joint frames: no sign table, no zero offsets. The trigger on each teaching handle is the
gripper.

## 2. One-time setup

On the **station**:

```bash
cd ~/Desktop/EW/abc          # ABC repo, synced from the dev PC
uv sync --extra deploy
```

On the **GPU box**: the checkpoint, plus the repo that serves it: the ABC repo for
ABC-DiT, or yam-abc-reproduce for MolmoAct2 / π0.5 (see its `docs/deploy.md` for that
backend's environment).

## 3. Before every session (5 min)

1. **Stop robot_class** and anything else that uses the CAN buses or cameras. ABC opens them
   itself.
2. Check the CAN names: `ip -br link` must list `can_follower_l`, `can_follower_r`,
   `can_leader_l` and `can_leader_r`, all UP. USB re-plugs can rename them.
3. **Teleop sanity check** (the YAM leader path has only been tested offline so far; do this
   until it has a few real sessions behind it):
   ```bash
   ROBOT_PROFILE=yam007_config uv run deploy/robot/scripts/run_teleop.py
   ```
   Hold the leaders near the followers' pose before starting: the followers ramp to the
   leaders over 2 s. Check that both arms track and that both triggers open and close the
   grippers. If the followers feel soft or stiff compared with the usual teleop, adjust with
   `--kp-scale`. Ctrl-C when done.
   The first time, also check the buttons: in §4's session, each press should print a
   `[DAGGER]` line. Tested offline only so far.
4. Lay out the cloth, using the agreed start piles (photograph them; `plan_unfold.md` §6).

## 4. Start

Start the server for the model under test on the GPU box, then the station command for it.
Use the **prompt the model was trained with**: it is stored in every recording, and the
exports reuse it.

**ABC-DiT** (ABC repo on the GPU box):

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra deploy deploy/serve_policy.py \
  --policy.checkpoint-path=checkpoints/unfold-10hr-data/30000_model.pt \
  --policy.prompt='unfold' --policy.diffusion-steps=10 --port 8000
```
```bash
# station
ROBOT_PROFILE=yam007_config uv run --extra deploy deploy/dagger.py \
  --remote-host=192.168.123.1 --port=8000 --prompt='unfold' \
  --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7 --execute-chunk-dim=16 \
  --compress-images
```

**MolmoAct2** (yam-abc-reproduce on the GPU box; full setup in its `docs/deploy.md`):

```bash
cd third_party/policy/molmoact2
../../../.venv/bin/python ../../../yam_abc_reproduce/deploy/servers/molmoact_server.py \
  --checkpoint <HF checkpoint dir> --port 8202
```
```bash
# station: no --compress-images (it letterboxes to 224x224, unlike MolmoAct2's training
# frames); lead 12 steps covers ~300 ms inference + 100 ms margin at 30 Hz
ROBOT_PROFILE=yam007_config uv run --extra deploy deploy/dagger.py \
  --remote-host=192.168.123.1 --port=8202 --prompt='unfold linen' \
  --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=12 --execute-chunk-dim=16
```

**π0.5** (yam-abc-reproduce's `openpi_server.py`, with the `--config`/`--image-key-map`
your openpi train config needs): station command as for MolmoAct2, with its port and
prompt. `--compress-images` is fine here, because openpi pads to 224x224 itself. There is
no chunk continuation for π0.5 on this path, so expect a small seam at each chunk swap.

**Lead steps.** The lead must cover the server's inference time. If the launcher logs
`inference still pending` stalls, raise `--rtc-inference-lead-steps`.

- At startup the followers connect and calibrate their grippers (several seconds of
  gripper motion; keep cloth out of the fingers), then go to the start pose.
- Wait for `[DAGGER] Ready. Press A to start an episode.`

## 5. Controls

The teaching-handle buttons send the keys, so you never let go of the handles:

| Key | Button | Keyboard |
|---|---|---|
| **a** | **right handle, top button** | `a` |
| **c** | **right handle, second button** | `c` |
| **b** | **left handle, top button** | `b` |

**Don't hold any button while the station starts**: each handle learns its buttons'
resting level at startup (it varies between units), so a held button would be read as
released. The mapping is `_YAM_HANDLE_BUTTONS` in `deploy/robot/config.py`.

| State | Key | What happens |
|---|---|---|
| Ready (idle) | **a** | Start an episode. The first one of the session shows a safety table (current joints vs the first action): press **a** or ENTER to run, Ctrl-C to abort. |
| Policy running | **a** | **Take over**: the policy pauses and the leaders drive the followers. |
| You are driving | **a** | **Hand back**: the robot holds still while a new chunk is computed, then the policy resumes. |
| Policy or you | **c** | **Checkpoint**: "everything up to here is good". |
| Policy or you | **b** | **End**: the robot freezes. |
| Frozen (after b) | **c** | **Save** the whole episode. |
| Frozen (after b) | **a** | **Discard** everything after the last checkpoint (or the whole episode if there is none). |

```
Ready --a--> Policy <--a--> You driving
               |                |
               +------b---------+--> Frozen --c--> save    --> arms go home --> Ready
                                            --a--> discard --> arms go home --> Ready
```

After save or discard, **the arms drive back to the start pose immediately**. Make sure the
grippers are not holding the cloth against the table before you press the key.

## 6. Taking over

**How the mapping works.** At the moment you press **a**, your leader's pose becomes the
reference. From then on, the follower's hand moves by the *change* in your leader's hand
pose, applied from where the follower is. The follower does not jump to the leader's pose,
and the leader does not need to match the follower. The leader arms are not driven by the
robot: they stay limp (gravity compensated) the whole time.

**Do this:**

1. While the policy runs, keep a hand on each teaching handle, with the leaders in a pose
   roughly like the followers'. They do not need to match, but it leaves you room to move
   in every direction before the leader runs into its own joint limits.
2. Press **a** a moment *before* the failure, not after it. Example: fingers about to close
   on bare table, or a second grasp going for the wrong corner.
3. Fix only the mistake, in the same style as the demonstrations, then press **a** to hand
   back. Short, clean takeovers are worth more than long ones.
4. Don't take over for small wobbles the policy would correct by itself.

**Grippers are relative too.** Pull the trigger to close. At takeover each follower
gripper keeps its current opening, and moving the trigger changes it from there.

- **Take over with the trigger roughly matching the follower's gripper**: squeezed if the
  follower is holding cloth. Then letting the trigger out by about 10% snaps that gripper
  fully open, so you can release the cloth.
- Trap: take over with the trigger fully released while the follower is closed, and that
  gripper **cannot open** during this takeover. The trigger has no travel left. Hand back
  (**a**) and take over again with the trigger squeezed.

**If you see `GELLO data is missing or stale; intervention refused`:** the leader stream
dropped (it must be fresher than 0.2 s). The policy keeps control. Check the leader CAN
buses and the leader process in the launcher output. If it drops *during* a takeover, the
follower holds its last target until the stream comes back.

## 7. Ending an episode

| Situation | Keys |
|---|---|
| Task done well (by the policy, you, or both) | **b**, then **c** |
| Good until some point, then your correction was clumsy | Press **c** at the last good moment while running; at the end **b**, then **a**. Only the part after the checkpoint is dropped. |
| Whole episode unusable | **b**, then **a** with no checkpoint |

Discarded parts stay in the file (marked `discard_segments`) for auditing; the exporter
skips them.

## 8. After the session

Outputs on the station:

| Path | Contents |
|---|---|
| `data/dagger_h5/*.h5` | One file per episode: cameras, follower state, commanded actions, policy inferences, and the event log (takeover start/end, checkpoints, save/discard) |
| `data/dagger_video/*.mp4` | Review video per episode, with a controller-state badge and a green/red bar showing what the exporter keeps. Rendered in the background; check `postprocess.log` if one is missing. |

Watch a few videos before exporting. Export only the sessions of the model you are about to
train: filter with `--modified-after` / `--exclude-files`, or keep each model's sessions in
its own folder.

**For ABC-DiT** (abc_minimal format), using the deployed checkpoint's normalization stats:

```bash
uv run python -m deploy.recording.export \
  --input-pattern 'data/dagger_h5/*.h5' --output-dir data/dagger_export \
  --norm-stats-from-checkpoint checkpoints/unfold-10hr-data/30000_model.pt
```

**For MolmoAct2 / π0.5** (LeRobot v3.0, same layout as the station's own recordings:
`observation.state`, `action`, `observation.images.{head_cam,wrist_left,wrist_right}`):

```bash
uv run --with 'lerobot[dataset]>=0.5,<0.6' python -m deploy.recording.export_lerobot \
  --input-pattern 'data/dagger_h5/*.h5' --output-dir data/dagger_lerobot_molmoact_r1
```

Both exporters pick the same frames: takeover segments by default, with discards honoured.
For the "rest of the rollouts" share of the training mix, export again with
`--mode whole_episodes`. Add `--intervention-flag` to that LeRobot export for a per-frame
`intervention` column (1 = you were driving). `export_summary.json` lists every exported
and skipped segment with its reason.

**Training a round** (the ABC paper's recipe): finetune from the deployed checkpoint on
about **80% previous data, 10% takeovers, 10% rest of the rollouts**. Never train on
takeovers alone.

- ABC-DiT: `--load-pretrained --pretrained-ckpt-name <ckpt>`.
- MolmoAct2: add the LeRobot folders as repos in `experiments/launch_scripts/data_mixtures.py`.
- π0.5: add them to the openpi train config, the same way as the station recordings. Use 1–1.5 h of robot
time per round, then run the same 20-trial evaluation (`plan_unfold.md` §6) before the next
round.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| A button does nothing | Was it held at startup? Restart without touching the buttons. Still nothing: the handle isn't reporting (next row). The keyboard keys always work as a fallback. |
| A button fires on release instead of on press | The button was held at startup, so its levels were learned backwards. Restart. |
| `teaching handle on 'can_leader_x' not reporting` | Handle cable or CAN bus: check `ip -br link`, reseat the handle |
| Stuck at "Waiting for the policy server to load" | Server not up yet, or the wrong `--remote-host`/`--port` |
| Robot stalls between chunks with `--rtc` | Inference is slower than the lead: raise `--rtc-inference-lead-steps`, or shorten `--rtc-prefix-length` (`recipe_unfold.md` §6) |
| Arms move at startup | Expected: the followers go to `init_q` (the recordings' median start pose, all joints near zero). Start with them near it. |
| Gripper won't open during a takeover | You took over with the trigger released while that gripper was closed (§6): hand back, squeeze the trigger, take over again |

## 10. Known limits

- The YAM teaching-handle leader (`deploy/robot/leaders/yam_leader.py`) and the
  `yam007_config` profile were tested offline only (2026-10-09). Treat the first sessions
  as hardware bring-up: e-stop in hand, slow takeovers.
- The handle buttons, the MolmoAct2 / π0.5 server path and the LeRobot export were tested
  offline only (2026-10-12): button logic and pub/sub delivery, ABC's client against the
  MolmoAct2 server on a GPU, and an export read back with lerobot 0.5.
- π0.5 has no chunk continuation on this path; the ABC loop just skips the rows that ran
  during inference.
