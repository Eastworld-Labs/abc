# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tyro"]
# ///
"""Write the per-repo train/val episode split read by scripts/export_lerobot.py.

Whole episodes are held out per repo with evo's method (openpi
data_loader._split_episodes): a seeded permutation of the episode indices whose
first round(n * val_fraction) entries become val. With evo's seed (42) and
val_fraction (0.05), repos that share episode order with evo's copies hold out
the same episodes. excluded_episodes and excluded are carried over from
--previous so curation notes are not lost.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

SCRIPTS = Path(__file__).resolve().parent


@dataclass
class Args:
    repos: tuple[str, ...] = ()
    """Repos to split (default: the repos in --previous)."""
    root: Path = Path("~/cos-mount/data-collection/yam")
    """Directory holding one LeRobot dataset per repo (only meta/info.json is read)."""
    val_fraction: float = 0.05
    seed: int = 42
    previous: Path = SCRIPTS / "yam_split.json"
    """Existing split whose excluded/excluded_episodes entries are kept."""
    out: Path = SCRIPTS / "yam_split.json"


def split_episodes(total: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    order = np.random.default_rng(seed).permutation(total)
    num_val = round(total * val_fraction)
    return sorted(order[num_val:].tolist()), sorted(order[:num_val].tolist())


def main(args: Args):
    previous = json.loads(args.previous.read_text()) if args.previous.exists() else {}
    repos = args.repos or tuple(previous.get("repos", ()))
    if not repos:
        raise SystemExit("no repos: pass --repos or an existing --previous split")
    out = {"seed": args.seed, "val_fraction": args.val_fraction, "method": "evo_split_episodes",
           "repos": {}, "excluded": previous.get("excluded", {}),
           "excluded_episodes": previous.get("excluded_episodes", {})}
    root = args.root.expanduser()
    for repo in repos:
        total = json.loads((root / repo / "meta" / "info.json").read_text())["total_episodes"]
        train, val = split_episodes(total, args.val_fraction, args.seed)
        out["repos"][repo] = {"total_episodes": total, "train": train, "val": val}
        print(f"{repo:34s} {total:4d} episodes -> {len(train)} train / {len(val)} val")
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
