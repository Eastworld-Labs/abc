"""Exponential moving average of the trainable weights, sharded across ranks.

A full fp32 EMA copy does not fit next to DDP's replicated weights, grads and
Adam state on a 40 GB GPU (8 GB for ABC-DiT, 17 GB for ABC-VLA), so each rank
keeps only a slice: under DDP each trainable parameter is owned by one rank
(greedy balance by size); under FSDP each rank averages its own local shard.
The full average only exists when it is gathered for a checkpoint.

Saved next to each training checkpoint as ``<step>_ema_model.pt`` in the
model-only layout of scripts/strip_checkpoint.py, so deploy/eval load it like
any ``_model.pt``; ``--resume-from <step>.pt`` restores the EMA from it.
"""

from pathlib import Path

import torch
import torch.distributed as dist


def _is_dtensor(t) -> bool:
    return hasattr(t, "to_local") and hasattr(t, "device_mesh")


def _local(t):
    return t.to_local() if _is_dtensor(t) else t


def ema_path(checkpoint_path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}_ema_model.pt")


class ShardedEMA:
    def __init__(self, module, decay, *, rank, world, sharded_params):
        """``module`` is the unwrapped model (no DDP / compile wrapper), so
        parameter names match its state_dict keys. ``sharded_params`` is True
        under FSDP, where parameters are DTensors and every rank owns its shard."""
        self.decay = float(decay)
        self.rank, self.world = rank, world
        self.sharded_params = sharded_params
        self.named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
        if sharded_params or world == 1:
            self.owner = {n: rank for n, _ in self.named}
        else:
            load = [0] * world
            self.owner = {}
            for n, p in sorted(self.named, key=lambda np_: -np_[1].numel()):
                r = min(range(world), key=load.__getitem__)
                self.owner[n] = r
                load[r] += p.numel()
        self.owned = [(n, p) for n, p in self.named if self.owner[n] == rank]
        self.shadow = {n: _local(p).detach().float().clone() for n, p in self.owned}

    @torch.no_grad()
    def update(self):
        if not self.owned:
            return
        shadows = [self.shadow[n] for n, _ in self.owned]
        params = [_local(p).detach().float() for _, p in self.owned]
        torch._foreach_lerp_(shadows, params, 1.0 - self.decay)

    @torch.no_grad()
    def gather(self, keep: bool):
        """Full EMA tensors on CPU (when ``keep``), else None. A collective:
        every rank calls it with the same parameter order."""
        from torch.distributed.tensor import DTensor

        out = {} if keep else None
        for n, p in self.named:
            if self.sharded_params:
                full = DTensor.from_local(
                    self.shadow[n], p.device_mesh, p.placements,
                    shape=p.shape, stride=p.stride(),
                ).full_tensor()
            elif self.world == 1:
                full = self.shadow[n]
            else:
                src = self.owner[n]
                full = (self.shadow[n] if src == self.rank
                        else torch.empty(p.shape, dtype=torch.float32, device=p.device))
                dist.broadcast(full, src=src)
            if keep:
                out[n] = full.to("cpu", copy=True)
        return out

    @torch.no_grad()
    def load(self, state):
        """Restore from a full state dict (``_ema_model.pt``'s ``model``).
        A collective under FSDP (distribute_tensor), so every rank calls it."""
        from torch.distributed.tensor import distribute_tensor

        for n, p in self.named:
            if n not in state:
                raise KeyError(f"EMA state has no parameter {n}")
            full = state[n].to(p.device if not _is_dtensor(p) else _local(p).device,
                               dtype=torch.float32)
            if self.sharded_params:
                local = distribute_tensor(full, p.device_mesh, p.placements).to_local()
            else:
                local = full
            if n in self.shadow:
                self.shadow[n].copy_(local)
