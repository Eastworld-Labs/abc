"""Write a model-only copy of a training checkpoint for deployment.

Training checkpoints (<step>.pt, ~24 GB for ABC-DiT XL) carry optimizer and
scheduler state for resuming. Deploy and eval only need the weights and norm
stats, so this writes <step>_model.pt (~8 GB) in the layout of the released
model-only checkpoints. Keep the full file on the training box for --resume-from.

    uv run scripts/strip_checkpoint.py checkpoints/yam_run1/5000.pt
"""

import sys
from pathlib import Path

import torch

KEEP = ("model", "norm_stats", "train_config", "model_config")

for arg in sys.argv[1:]:
    src = Path(arg)
    dst = src.with_name(f"{src.stem}_model.pt")
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["step"] = ckpt.get("global_step", ckpt.get("step"))
    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(out, tmp)
    tmp.replace(dst)
    print(f"{src} -> {dst} ({dst.stat().st_size / 1e9:.1f} GB)")
