# Unfold on yam-007: plan (2026-10-09)

Goal: a policy that unfolds a crumpled cloth on station yam-007 reliably, and
keeps working when the background, lighting or cloth placement changes.

How we get there: new demonstrations of one precision-tolerant strategy
recorded on yam-007, one finetune, then two rounds of DAgger, with the same
robot test after every stage. ABC-VLA is trained once on the same data as a
comparison.

## 1. Where we are

| Finding | Evidence |
|---|---|
| Current checkpoints (2hr, 10hr at 30k/70k/100k) do not complete the task on yam-007: success 0 %. They grasp, often miss, and keep re-trying the pick-up. | Robot rollouts, 2026-10-09 |
| The models are overfit: they fit training episodes closely but transfer poorly. | Open-loop error, hardest half of samples: training episodes 0.028 rad vs 0.100 for "hold still"; held-out episodes 0.090 vs 0.114. Held-out error is flat from 30k to 100k while val_loss rises from ~20k. |
| Grasp precision is the bottleneck. | The teleop strategy pinches flat cloth off the table (gripper closed 51-61 % of the time, ~10 closures/min); the model's error near the fingertips is ~1-2 cm. |
| The training data comes from a different station. | yam-007's cameras (head 260722303224, wrists 352122272270 / 352122273271) are not the recording station's. |
| The data mixes five behaviours under one prompt, `unfold`. | handkerchief, towel stack, hand towel, foot towel, pillow case |
| robot_class's limiter cuts speed and grip compared with teleop. | Commands capped at measured + 1 rad/s·dt and 3 rad/s²; teleop exceeds 1 rad/s in 14 % of frames. Not the main cause; DAgger runs on ABC's stack, which does not use it. |

## 2. Phases and gates

| # | Phase | Work | Training | Gate (decides whether to continue) |
|---|---|---|---|---|
| 0 | Prep | Section 3 | — | Teleop check passes on `yam007_config`; protocol written and practiced |
| 1a | Demos, first ~3 h | New strategy on yam-007 (section 4) | ~10k steps from the ABC parent | The robot attempts the new strategy and lands the first grasp in some trials. If not, change the protocol before recording more. |
| 1b | Demos up to ~10 h | Same protocol | ~30k steps (~3 h on 4×A100), learning-rate decay; same data also trained on ABC-VLA | Pick the step by held-out-background error; then the robot test (section 6). This sets the baseline. |
| 2 | DAgger round 1 | 1-1.5 h of robot time (section 5) | 5-10k steps from the 1b checkpoint, mix 80 % previous / 10 % takeovers / 10 % rest of rollouts | Robot test: success rate above the 1b baseline |
| 3 | DAgger round 2 | Same | 5-10k steps from the round-1 checkpoint | Robot test: success above round 1. If it stalls, review failure types before a round 3. |

Model choice: if ABC-VLA beats ABC-DiT in the 1b robot test, run the DAgger
rounds on ABC-VLA instead (it serves on one 16 GB GPU at ~50 ms/request,
measured; finetuning needs 80 GB GPUs).

## 3. Prep (phase 0)

Hardware and station:
- [ ] Reseat the left leader's CAN adapter in its original USB port (it is not enumerating; `can-bringup@can_leader_l` fails). Confirm with `ip -br link | grep can`: four interfaces.
- [ ] Foam or rubber mat on the table, the same one for demos and rollouts.
- [ ] Prepare 3-5 backgrounds and a second lighting setup.
- [ ] Foot pedal for DAgger, or the teaching-handle button mapping (section 7).

Software:
- [ ] Install ABC on yam-007: sync the repo, `uv sync --extra deploy`.
- [ ] Teleop check: `ROBOT_PROFILE=yam007_config uv run deploy/robot/scripts/run_teleop.py`. Both followers track both leaders and both grippers. If tracking feels weaker than the station's own teleop, adjust `--kp-scale`.
- [ ] Converter: one prompt per behaviour, and a validation split held out by background (section 7).
- [ ] Held-out-error check as a reusable script (section 7).

Protocol:
- [ ] Write the operator protocol (section 4) and have every operator practise it for ~10 episodes before recording counts.

## 4. Demonstration protocol (phases 1a, 1b)

One strategy, chosen to tolerate centimetre errors:

1. **Start state:** cloth crumpled into a pile inside a marked area on the mat; arms at home, grippers open.
2. **First grasp:** one hand grabs the top of the pile (a large, soft target, not an edge).
3. **Lift** high enough that the cloth hangs free.
4. **Second grasp in the air:** the other hand takes the lowest hanging corner.
5. **Stretch** between the two corners and **lay** the cloth down flat.
6. **One smoothing sweep** if needed. **Done** = cloth flat, no folds.

Rules:
- Same approach every time: top-down, fixed gripper rotation, slow final descent.
- If a grasp misses, re-grasp the same way and keep recording. Recoveries are useful data.
- Discard episodes that deviate from the protocol, and note failures in the episode labels.
- Vary on purpose, per session: background (3-5), lighting, cloth colour (same cloth type), pile position and crumpling.
- Record on yam-007 with its own cameras. Keep one background out of training for validation.
- Prompt for the new data: one fixed string (e.g. `unfold cloth`), different from the old `unfold`.

Old data: keep the 14.4 h in training at ~20-30 % of the mix, each behaviour
with its own prompt (`unfold handkerchief`, `stack towels`, `unfold linen`).

## 5. DAgger procedure (phases 2, 3)

Runs ABC's `deploy/dagger.py` on yam-007 with the policy served from the GPU
box 192.168.123.1 over Ethernet (commands and controls: `dagger_guide.md`).
robot_class must be stopped. The model does not learn during the session; the
recordings are used for the next finetune.

| Key | When |
|---|---|
| **a** (idle) | Start an episode; the first one in a session asks for confirmation |
| **a** (running) | Take over / hand back |
| **c** (running) | Checkpoint: everything up to here is good |
| **b** | End; then **c** saves, **a** discards after the last checkpoint |

Takeover rules:
- Take over just *before* a failure (fingers about to close on bare table), not at every wobble.
- Fix only the mistake, in the protocol's style, then hand back quickly.
- Discard (b, then a) any episode where your own correction was clumsy.
- About 1-1.5 h of robot time per round. Export with `deploy/recording/export.py`, which keeps the takeover segments.

## 6. Evaluation (after 1b, round 1, round 2)

The same protocol every time, so the stages can be compared:

- **20 trials:** 10 on a training background, 10 on the held-out background.
- Same cloths and the same 5 start piles (photograph them), rotated through the trials.
- **Record per trial:** success (flat, no folds) and, on failure, the stage: first grasp / lift / second grasp / lay-flat. Also the time to success.
- **Offline:** held-out error vs "hold still" on the held-out-background episodes, which picks the checkpoint step before robot time is spent.

## 7. Code tasks

| Task | Status |
|---|---|
| DAgger wiring for yam-007: YAM teaching-handle leader node, `yam007_config` profile, launcher switch | Done 2026-10-09, tested offline only |
| Teaching-handle buttons → a/b/c, so one operator can run DAgger alone | To do |
| Converter: per-behaviour prompts, validation split by background | To do |
| Held-out-error-vs-hold script in `scripts/` | To do |
| Training: learning-rate decay, mixture weights for old/new/DAgger data | To do |
| ABC-VLA finetune config for the unfold data | To do (needs 80 GB GPUs) |

## 8. Risks

| Risk | Mitigation |
|---|---|
| The new strategy is not learnable either | Gate at 1a (3 h) |
| Follower gains in ABC's stack differ from the station's teleop | Teleop check in phase 0; `--kp-scale` |
| Overfitting again | Background-held-out validation, learning-rate decay, short DAgger finetunes, pick steps by held-out error |
| USB port changes break the CAN names | Check `ip -br link` before each session |
| Wi-Fi latency if served from another machine | Serve from 192.168.123.1 (wired, ~1-3 ms) |

## 9. Rough timeline

| When | What |
|---|---|
| Week of 2026-10-12 | Prep (1 day), 1a demos and check, start 1b demos |
| Week of 2026-10-19 | Finish 1b demos (2-4 days of collection, depends on operators), 1b training and eval |
| Week of 2026-10-26 | DAgger rounds 1 and 2, about 1 day each including training and eval |
