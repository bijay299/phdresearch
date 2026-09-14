"""
Shared utilities: config, seeding, run bookkeeping.

Reproducibility rule for this project: every number that ends up in a table
must be traceable to a config, a seed, and a git commit. `RunDir` writes all
three next to the results so you never have to reconstruct them in week eight.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, resolving a single level of `inherits:`."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("inherits", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = deep_merge(parent, cfg)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """CLI overrides like `train.epochs=5` or `head.name=arcface`."""
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override must be key=value, got '{p}'")
        key, val = p.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)   # parses ints/floats/bools
    return cfg


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "no-git"


def device_string() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


# ----------------------------------------------------------------------
# run directory
# ----------------------------------------------------------------------

@dataclass
class RunDir:
    """
    One directory per run, holding everything needed to explain a number:
    the config, the seed, the commit, the environment, and the results.
    """
    root: Path
    cfg: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, base: str | Path, name: str, cfg: Dict[str, Any]) -> "RunDir":
        """
        Refuses to start if `root` already exists and is non-empty, instead
        of silently reusing it.

        Run directory names are derived from data/head/seed, not the full
        config -- two runs that differ only in, say, `unlearn.epochs` get
        the SAME name. Before this check, a second run into an existing
        directory would overwrite config.json/env.json (so they'd describe
        the wrong run) while *appending* to results.jsonl/trajectory.jsonl,
        silently mixing rows from incompatible configs into one file. That
        is exactly how CE's `finetune` row in results.jsonl ended up
        describing a 30-epoch run under a label everything else assumes is
        3 epochs (see notes/decisions.md, 2026-09-14). A fresh or empty
        `root` is unaffected -- this only blocks reusing one that already
        has something in it.
        """
        root = Path(base) / name
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                f"refusing to start: {root} already exists and is not empty. "
                f"Run directory names don't encode the full config, so reusing "
                f"one silently mixes results from different configs into the "
                f"same results.jsonl/trajectory.jsonl (see notes/decisions.md, "
                f"2026-09-14). Remove it, move it aside, or give this run a "
                f"different name."
            )
        root.mkdir(parents=True, exist_ok=True)
        rd = cls(root=root, cfg=cfg)
        rd.write_json("config.json", cfg)
        rd.write_json("env.json", {
            "git_commit": git_commit(),
            "device": device_string(),
            "python": sys.version.split()[0],
            "argv": sys.argv,
        })
        return rd

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def write_json(self, name: str, obj: Any) -> None:
        self.path(name).write_text(json.dumps(obj, indent=2, default=_json_safe))

    def append_jsonl(self, name: str, obj: Any) -> None:
        with open(self.path(name), "a") as f:
            f.write(json.dumps(obj, default=_json_safe) + "\n")

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path("run.log"), "a") as f:
            f.write(msg + "\n")


def _json_safe(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def banner(title: str, width: int = 62) -> str:
    return "\n" + "=" * width + f"\n  {title}\n" + "=" * width
