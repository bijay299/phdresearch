#!/usr/bin/env python
"""
NC3 class-mean robustness diagnostic.

THE QUESTION THIS ANSWERS
-------------------------
`nc3_centred_forget` compares the forget class's classifier weight against
its class-mean FEATURE. That mean is estimated from the forget class's
training images -- ~4,500 of them on CIFAR-10, but only ~40 per identity on
the face set. So whenever CIFAR and faces disagree about nc3, the first
alternative explanation to rule out is not a real difference between the
datasets but a starved estimator: at n=40 the class mean might simply be too
noisy for the effect to show.

This script rules it in or out on whichever model it is pointed at, by
holding EVERYTHING fixed except the number of images used to estimate the
forget class's mean. One model, one weight matrix, one set of features; only
the rows feeding the forget-class mean change, K at a time, across
deterministic trials.

An entry in notes/decisions.md (2026-09-14) reports scratch numbers from an
ad-hoc version of this procedure. Those are a COMPARISON TARGET ONLY: nothing
here is tuned toward them, nothing hard-codes them, and no test asserts them.

REPLAYING THE CANONICAL MODEL -- read before changing the order of anything
--------------------------------------------------------------------------
The model measured here must be THE model the headline number came from, not
a same-config lookalike. Reaching it means re-running one epoch of
`random_label_clfonly` from the baseline checkpoint with the global RNG in
the same state the original sweep had -- and the RNG is touched by more than
the obvious things. Measured on this machine (torch 2.5.1):

    CONSUMES global torch CPU RNG
      build_backbone(...)                       weight init
      build_head(...)                           xavier init
      iterating ANY DataLoader with num_workers>0   worker base_seed,
                                                    even when shuffle=False
      iterating a shuffle=True DataLoader           sampler seed
    does NOT consume it
      DataLoader construction (any settings)
      metrics.linear_probe / nc3_alignment / nc1_angular

So the epoch-0 trajectory callback -- which iterates the train_eval loader
and the test loader once each -- advances the RNG by two draws before the
training epoch starts. Skipping it changes the shuffle order of the retain
and forget loaders and therefore changes the resulting model. That callback
is replayed here for exactly that reason, not because its numbers are wanted.

This script therefore mirrors `scripts/unlearn_across_classes.run_condition`
call for call, at epochs=1. It is an INTENDED EXACT REPLAY: whether it lands
on the canonical model is decided by the recorded `replay_delta` against the
reference run's own epoch-1 `nc3_centred_forget`, not by this docstring. Call
the result exact only once that delta comes back at zero.

Exactness also requires that the target condition was the FIRST condition its
sweep executed -- a sweep runs conditions in one process, so anything before
the target moved the RNG too. The reference run's recorded argv is parsed and
the preceding conditions are recorded; a non-empty prefix is reported as such
rather than papered over. This script does not execute prefixes.

Test images: the replayed callback evaluates on the test loader exactly as
production does. The class mean is computed only from training images under
eval transforms (`train_eval`); no test feature ever enters it.

Only the SAMPLING is governed by `--sample-seed`. Replay is deterministic for
a given device and library versions, not bitwise portable across them -- both
are recorded with the result.

    python scripts/nc3_mean_robustness.py \
        --run-dir logs/postfix_seed0_clean/cifar10_arcface_seed0 \
        --forget-class 0 --sample-seed 12345
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as D                      # noqa: E402
import metrics as M                   # noqa: E402
import train as TR                    # noqa: E402
import unlearn as UL                  # noqa: E402
from backbones import build_backbone  # noqa: E402
from heads import build_head          # noqa: E402
from utils import (RunDir, apply_overrides, banner, device_string,  # noqa: E402
                   git_commit, git_dirty_files, load_config, set_seed)


# The condition replayed is fixed, not configurable: this diagnostic is about
# the class-mean ESTIMATOR, so the model it measures must be the one the
# headline number came from. A method or epoch-count flag here would change
# the model as well as the estimator and answer nothing.
UNLEARN_METHOD = "random_label"
UNLEARN_TAG = "random_label_clfonly"
UNLEARN_EPOCHS = 1
CLASSIFIER_ONLY = True
SPLIT_MODE = "all"

# The order scripts/unlearn_across_classes.py runs conditions in, per class.
SWEEP_CONDITION_ORDER = ("finetune", "random_label_clfonly")


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def validate_k(k, n_available: int) -> int:
    """K must be a positive integer no larger than the pool it samples from.

    K > n_available is an error rather than a clamp: sampling is WITHOUT
    replacement, so a clamped K would silently make every trial the full-data
    computation and report zero variance as evidence of stability.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError(f"--k must be an integer, got {k!r}")
    k = int(k)
    if k < 1:
        raise ValueError(f"--k must be at least 1, got {k}")
    if k > n_available:
        raise ValueError(
            f"--k={k} exceeds the {n_available} training image(s) of the "
            f"forget class. Sampling is without replacement, so K cannot be "
            f"larger than the pool; lower --k."
        )
    return k


def validate_trials(trials) -> int:
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise ValueError(f"--trials must be an integer, got {trials!r}")
    trials = int(trials)
    if trials < 1:
        raise ValueError(f"--trials must be at least 1, got {trials}")
    return trials


def validate_forget_class(forget_class, num_classes: int,
                          targets: Optional[np.ndarray] = None) -> int:
    if isinstance(forget_class, bool) or not isinstance(forget_class, (int, np.integer)):
        raise ValueError(f"--forget-class must be an integer, got {forget_class!r}")
    fc = int(forget_class)
    if not 0 <= fc < num_classes:
        raise ValueError(
            f"--forget-class={fc} is out of range for this dataset "
            f"({num_classes} classes, valid 0..{num_classes - 1})"
        )
    if targets is not None and not (np.asarray(targets) == fc).any():
        raise ValueError(
            f"--forget-class={fc} has no training images in this split"
        )
    return fc


# ----------------------------------------------------------------------
# baseline + reference provenance
# ----------------------------------------------------------------------

def load_baseline(run_dir: Optional[Path], config_path: Optional[str],
                  ckpt_arg: Optional[str], overrides: Sequence[str]) -> Tuple[dict, Path, dict]:
    """Returns (config, checkpoint path, the baseline's own recorded env)."""
    baseline_env: dict = {}
    if run_dir is not None:
        cfg_file = run_dir / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"{cfg_file} not found -- --run-dir expects a directory written "
                f"by RunDir (config.json, env.json, ckpt.pt)."
            )
        cfg = json.loads(cfg_file.read_text())
        env_file = run_dir / "env.json"
        if env_file.exists():
            baseline_env = json.loads(env_file.read_text())
        ckpt_path = Path(ckpt_arg) if ckpt_arg else run_dir / "ckpt.pt"
    else:
        cfg = load_config(config_path)
        if ckpt_arg:
            ckpt_path = Path(ckpt_arg)
        else:
            name = f"{cfg['data']['name']}_{cfg['head']['name']}_seed{cfg['seed']}"
            ckpt_path = Path(cfg["out_dir"]) / name / "ckpt.pt"

    cfg = apply_overrides(cfg, list(overrides))
    for block, keys in (("data", ("name", "batch_size", "num_workers")),
                        ("head", ("name",)),
                        ("unlearn", ("lr", "weight_decay"))):
        for key in keys:
            if key not in cfg.get(block, {}):
                raise ValueError(
                    f"config has no {block}.{key} -- the replayed epoch takes "
                    f"its settings from the baseline's own config blocks, so "
                    f"they must be present"
                )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. Point --ckpt at the baseline "
            f"ckpt.pt, or --run-dir at the run directory that holds it."
        )
    return cfg, ckpt_path, baseline_env


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_model(cfg: dict, ckpt_path: Path, num_classes: int):
    """Rebuild the baseline model and load the checkpoint, strictly.

    Every config/checkpoint mismatch is reported rather than absorbed: a head
    whose state_dict does not match the configured head, or a weight of the
    wrong shape, means the config on hand does not describe the checkpoint on
    hand, and the geometry would be attributed to the wrong run.
    """
    backbone = build_backbone(**dict(cfg["backbone"]))
    head_cfg = dict(cfg["head"])
    head_name = head_cfg.pop("name")
    head = build_head(head_name, feat_dim=backbone.feat_dim,
                      num_classes=num_classes, **head_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("backbone", "head"):
        if key not in ckpt:
            raise ValueError(
                f"{ckpt_path} has no '{key}' entry (keys: {sorted(ckpt)}); "
                f"this is not a checkpoint written by scripts/run_experiment.py"
            )

    expected = set(head.state_dict().keys())
    found = set(ckpt["head"].keys())
    if expected != found:
        raise ValueError(
            f"wrong classifier head: the config says head '{head_name}' "
            f"(state_dict keys {sorted(expected)}) but {ckpt_path} carries "
            f"{sorted(found)}. Note that arcface and cosface are "
            f"indistinguishable by state_dict -- both hold a single `W` -- so "
            f"the config is the only record of which trained a checkpoint. "
            f"Use the config that trained this one."
        )

    try:
        backbone.load_state_dict(ckpt["backbone"], strict=True)
        head.load_state_dict(ckpt["head"], strict=True)
    except RuntimeError as e:
        raise ValueError(
            f"config/checkpoint incompatible: {ckpt_path} does not fit the "
            f"model this config builds (backbone {dict(cfg['backbone'])}, "
            f"head '{head_name}', num_classes={num_classes}). Original error: {e}"
        ) from e

    if not hasattr(head, "weight"):
        raise ValueError(
            f"head '{head_name}' exposes no `weight` -- nc3 is computed from "
            f"head.weight, which every head in src/heads.py provides"
        )
    if tuple(head.weight.shape) != (num_classes, backbone.feat_dim):
        raise ValueError(
            f"head '{head_name}' has weight shape {tuple(head.weight.shape)}, "
            f"expected ({num_classes}, {backbone.feat_dim}) -- "
            f"config/checkpoint mismatch"
        )
    return backbone, head, head_name


def sweep_condition_sequence(argv: Sequence[str]) -> Optional[List[Tuple[int, str]]]:
    """(forget_class, condition) pairs in the order a sweep's argv ran them.

    Returns None when the argv is not a `unlearn_across_classes.py` command
    line, i.e. when the ordering cannot be established from the record. None
    means UNKNOWN, never "nothing preceded it".
    """
    argv = list(argv)
    if not any("unlearn_across_classes" in a for a in argv):
        return None

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    classes = opt("--forget-classes")
    if classes is None:
        return None
    chosen = (opt("--conditions") or ",".join(SWEEP_CONDITION_ORDER)).split(",")
    order = [c for c in SWEEP_CONDITION_ORDER if c in chosen]
    try:
        return [(int(f), c) for f in classes.split(",") for c in order]
    except ValueError:
        return None


def read_reference(ref_dir: Optional[Path], forget_class: int) -> dict:
    """The reference run's recorded epoch-1 metric and condition ordering.

    This is the only thing that can decide whether the replay landed on the
    canonical model, so its absence is recorded as an absence rather than
    quietly treated as agreement.
    """
    out = {"reference_run": None, "canonical_epoch1_nc3_centred_forget": None,
           "preceding_conditions": None, "target_was_first_in_sweep": None,
           "note": "no reference run supplied or found; the replay cannot be "
                   "checked against the canonical model"}
    if ref_dir is None or not ref_dir.exists():
        return out

    out["reference_run"] = str(ref_dir)
    traj = ref_dir / "trajectory.jsonl"
    if traj.exists():
        for line in traj.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("epoch") == 1 and row.get("forget_class") == forget_class:
                out["canonical_epoch1_nc3_centred_forget"] = row.get("nc3_centred_forget")
                break

    env_file = ref_dir / "env.json"
    if env_file.exists():
        argv = json.loads(env_file.read_text()).get("argv", [])
        out["reference_argv"] = argv
        seq = sweep_condition_sequence(argv)
        if seq is not None:
            target = (forget_class, UNLEARN_TAG)
            if target in seq:
                before = seq[:seq.index(target)]
                out["preceding_conditions"] = [f"fc{f}:{c}" for f, c in before]
                out["target_was_first_in_sweep"] = not before

    if out["canonical_epoch1_nc3_centred_forget"] is None:
        out["note"] = (f"{ref_dir} has no epoch-1 trajectory row for forget "
                       f"class {forget_class}; replay cannot be checked")
    elif out["target_was_first_in_sweep"]:
        out["note"] = ("target condition was first in its sweep, so replaying "
                       "the production path from the checkpoint is expected to "
                       "reach the canonical model exactly")
    elif out["target_was_first_in_sweep"] is False:
        out["note"] = ("target condition was NOT first in its sweep: the "
                       "conditions listed in preceding_conditions advanced the "
                       "RNG first, and this script does not execute prefixes. "
                       "Expect a nonzero replay_delta; the replay is then a "
                       "fresh deterministic run, not the canonical model")
    else:
        out["note"] = ("reference argv does not record a sweep ordering, so "
                       "whether anything preceded the target is UNKNOWN")
    return out


# ----------------------------------------------------------------------
# sampling and the per-trial metric
# ----------------------------------------------------------------------

def sample_trials(rows: np.ndarray, k: int, trials: int, seed: int) -> List[np.ndarray]:
    """`trials` index sets of size K, drawn from `rows` without replacement.

    Each trial gets its own generator seeded by (sample_seed, trial index), so
    trial t is a function of the seed alone -- adding trials never moves the
    ones already recorded. Returned sorted, for auditing.
    """
    rows = np.asarray(rows)
    out = []
    for t in range(trials):
        rng = np.random.default_rng([int(seed), int(t)])
        out.append(np.sort(rng.choice(rows, size=k, replace=False)))
    return out


def sampled_centred_nc3_forget(features: np.ndarray, labels: np.ndarray,
                               weight: np.ndarray, num_classes: int,
                               forget_class: int,
                               sample_rows: Sequence[int]) -> float:
    """Centred NC3 for the forget class with its mean taken over `sample_rows`.

    Computed by handing `metrics.nc3_alignment` every retained-class row plus
    only the sampled forget-class rows. Retained rows are untouched and the
    centring reference already excludes the forget class, so the ONLY quantity
    that differs from the full-data call is the forget-class mean -- the one
    thing this diagnostic varies. Going through the production metric rather
    than an inlined cosine is deliberate: a second copy of the formula is a
    second thing that can drift from `train.evaluate`.
    """
    keep = np.where(np.asarray(labels) != forget_class)[0]
    sample_rows = np.asarray(sample_rows, dtype=keep.dtype)
    idx = np.sort(np.concatenate([keep, sample_rows]))
    r = M.nc3_alignment(features[idx], labels[idx], weight, num_classes,
                        centre=True, exclude_from_centre=forget_class)
    return float(r["forget"])


def base_indices(ds, rows: Sequence[int]) -> np.ndarray:
    """Map positions in the extraction order onto the underlying dataset's own
    indices, unwrapping however many Subsets sit in between."""
    rows = np.asarray(rows, dtype=int)
    if isinstance(ds, Subset):
        return base_indices(ds.dataset, np.asarray(ds.indices, dtype=int)[rows])
    return rows


def sample_identifiers(ds, rows: Sequence[int]) -> Tuple[List[int], Optional[List[str]]]:
    """(dataset indices, file paths if the dataset has any) for `rows`.

    Feature rows line up with dataset order because the extraction loader is
    built with shuffle=False over the whole train_eval set. Paths come from
    FaceFolder.samples; CIFAR has none, and the index is the identifier there.
    """
    idx = base_indices(ds, rows)
    root = ds
    while isinstance(root, Subset):
        root = root.dataset
    paths = None
    if hasattr(root, "samples"):
        paths = [str(root.samples[int(i)][0]) for i in idx]
    return [int(i) for i in idx], paths


def summarise(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "n_trials": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "std_ddof0": float(v.std(ddof=0)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_below_0": int((v < 0.0).sum()),
        "n_below_neg_0_5": int((v < -0.5).sum()),
        "std_convention": "std is the sample standard deviation (ddof=1); "
                          "std_ddof0 is the population one",
    }


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def main(a: argparse.Namespace) -> None:
    # ---- provenance gate, before anything scientific happens ----------
    dirty = git_dirty_files(cwd=ROOT)
    commit = git_commit()
    if dirty is None or dirty:
        detail = ("git could not report the working tree state"
                  if dirty is None else
                  "working tree has uncommitted changes:\n    " +
                  "\n    ".join(dirty[:20]) +
                  ("\n    ..." if len(dirty) > 20 else ""))
        if not a.allow_dirty:
            raise SystemExit(
                f"refusing to run from an unverifiable tree -- {detail}\n"
                f"Every number in this project must be traceable to a commit "
                f"(see CLAUDE.md). Commit or stash first. --allow-dirty "
                f"overrides this and records the dirty state in the result, "
                f"but the result is then not reproducible from {commit} alone."
            )
        print(f"WARNING: --allow-dirty -- {detail}")

    run_dir = Path(a.run_dir) if a.run_dir else None
    cfg, ckpt_path, baseline_env = load_baseline(run_dir, a.config, a.ckpt, a.set)
    trials = validate_trials(a.trials)
    device = resolve_device(a.device or cfg.get("device", "auto"))
    out_base = a.out_dir or cfg["out_dir"]
    name = a.name or (
        f"nc3_mean_robustness_{cfg['data']['name']}_{cfg['head']['name']}"
        f"_seed{cfg['seed']}_fc{a.forget_class}_k{a.k}_t{trials}_s{a.sample_seed}"
    )

    # ==================================================================
    # From here to the end of the replayed epoch, the call sequence
    # MIRRORS scripts/unlearn_across_classes.py (main() then
    # run_condition()) exactly. Every call below that touches the global
    # RNG is there because the original did it; reordering, adding or
    # dropping one changes the model that comes out. See the module
    # docstring for the measured list of which calls consume RNG.
    # ==================================================================
    set_seed(cfg["seed"])

    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(
        cfg["data"], seed=cfg["seed"])
    nw = cfg["data"]["num_workers"]
    train_eval_loader = D.make_loader(train_eval_ds, None, 512, False, nw)
    test_loader = D.make_loader(test_ds, None, 512, False, nw)

    backbone, head, head_name = load_model(cfg, ckpt_path, num_classes)

    targets = D.get_targets(train_ds)
    forget_class = validate_forget_class(a.forget_class, num_classes, targets)
    n_forget_train = int((targets == forget_class).sum())
    k = validate_k(a.k, n_forget_train)

    split = D.make_forget_split(targets, forget_class, SPLIT_MODE,
                                cfg["unlearn"].get("forget_fraction", 0.5),
                                cfg["seed"])
    bs = cfg["data"]["batch_size"]
    forget_loader = D.make_loader(train_ds, split.forget_idx, bs, True, nw)
    retain_loader = D.make_loader(train_ds, split.retain_idx, bs, True, nw)

    ref_dir = (Path(a.reference_run) if a.reference_run else
               (run_dir.parent / f"{run_dir.name}_{UNLEARN_TAG}_fc{forget_class}"
                if run_dir else None))
    reference = read_reference(ref_dir, forget_class)

    run_cfg = dict(cfg, _nc3_mean_robustness=True, _forget_class=forget_class,
                   _k=k, _trials=trials, _sample_seed=a.sample_seed,
                   _baseline_ckpt=str(ckpt_path))
    rd = RunDir.create(out_base, name, run_cfg)   # refuses a nonempty target
    rd.log(banner(f"NC3 CLASS-MEAN ROBUSTNESS  {name}"))
    rd.log(f"device: {device}   commit: {commit}   dirty: "
           f"{'unknown' if dirty is None else bool(dirty)}")
    rd.log(f"baseline: {ckpt_path}")
    rd.log(f"head: {head_name}   feat_dim: {backbone.feat_dim}   "
           f"classes: {num_classes}")
    rd.log(f"forget_class {forget_class} | {n_forget_train} training images | "
           f"K={k} | trials={trials} | sample_seed={a.sample_seed}")
    if k == n_forget_train:
        rd.log("  NOTE: K equals the full pool, so every trial is the "
               "full-data computation and the spread is zero by construction.")
    rd.log(f"reference: {reference['reference_run']}")
    rd.log(f"  canonical epoch-1 nc3_centred_forget: "
           f"{reference['canonical_epoch1_nc3_centred_forget']}")
    rd.log(f"  preceding conditions: {reference['preceding_conditions']}")
    rd.log(f"  {reference['note']}")

    # ---- the replayed epoch -------------------------------------------
    # The callback is production's own (unlearn_across_classes.run_condition),
    # replayed at epoch 0 and 1. It is required for RNG fidelity, and its
    # epoch-1 value is what the replay is checked against.
    rd.log(banner(f"REPLAY  {UNLEARN_TAG}, 1 epoch"))
    rd.log(f"  {split.summary()}")
    trajectory = {}

    def epoch_eval(bb, hd, epoch):
        point = TR.evaluate_light(bb, hd, train_eval_loader, test_loader,
                                  num_classes, device,
                                  forget_class=forget_class, seed=cfg["seed"])
        row = {"head": head_name, "method": UNLEARN_TAG,
               "forget_class": forget_class, "seed": cfg["seed"],
               "epoch": epoch, **point}
        rd.append_jsonl("trajectory.jsonl", row)
        trajectory[epoch] = point
        rd.log(f"    epoch {epoch}: nc3_centred_forget "
               f"{point['nc3_centred_forget']:+.6f}   "
               f"output_forget {point['output_forget']:.4f}")

    u = cfg["unlearn"]
    t0 = time.time()
    bb_u, hd_u = UL.run_unlearning(
        UNLEARN_METHOD, backbone, head, device,
        forget_loader=forget_loader, retain_loader=retain_loader,
        num_classes=num_classes, epochs=UNLEARN_EPOCHS, lr=u["lr"],
        weight_decay=u["weight_decay"], classifier_only=CLASSIFIER_ONLY,
        log=rd.log, epoch_eval=epoch_eval)
    replay_time_s = time.time() - t0

    replayed = trajectory.get(UNLEARN_EPOCHS, {}).get("nc3_centred_forget")
    canonical = reference["canonical_epoch1_nc3_centred_forget"]
    delta = (float(replayed) - float(canonical)
             if replayed is not None and canonical is not None else None)
    rd.log(f"  replayed epoch-1 {replayed}   canonical {canonical}   "
           f"delta {delta}")

    # ---- features and the full-data metric -----------------------------
    # Everything below runs AFTER the replayed epoch, so the extra loader
    # iteration it costs cannot affect the model being measured.
    t0 = time.time()
    feats, labels, _ = TR.extract(bb_u, hd_u, train_eval_loader, device)
    extract_time_s = time.time() - t0
    W = hd_u.weight.detach().cpu().numpy()
    if len(labels) != len(targets):
        raise RuntimeError(
            f"extracted {len(labels)} feature rows for {len(targets)} training "
            f"images; the feature order does not match the split"
        )

    full = M.nc3_alignment(feats, labels, W, num_classes, centre=True,
                           exclude_from_centre=forget_class)
    full_forget = float(full["forget"])
    rd.log(banner("FULL-DATA METRIC"))
    rd.log(f"  nc3_centred_forget      {full_forget:+.4f}   "
           f"(over all {n_forget_train} forget-class training images)")
    rd.log(f"  nc3_centred_retain_mean {full['retain_mean']:+.4f}")

    # ---- the trials -----------------------------------------------------
    rows = np.where(labels == forget_class)[0]
    trial_rows = sample_trials(rows, k, trials, a.sample_seed)
    rd.log(banner(f"TRIALS  K={k} of {n_forget_train}, without replacement"))
    trial_records, values = [], []
    t0 = time.time()
    for t, sel in enumerate(trial_rows):
        val = sampled_centred_nc3_forget(feats, labels, W, num_classes,
                                         forget_class, sel)
        ds_idx, paths = sample_identifiers(train_eval_ds, sel)
        rec = {"trial": t, "nc3_centred_forget": val, "k": k,
               "feature_rows": [int(i) for i in sel], "dataset_indices": ds_idx}
        if paths is not None:
            rec["sample_paths"] = paths
        rd.append_jsonl("trials.jsonl", rec)
        trial_records.append(rec)
        values.append(val)
        rd.log(f"  trial {t:3d}  nc3_centred_forget {val:+.4f}")
    trials_time_s = time.time() - t0

    summary = summarise(values)
    rd.log(banner("SUMMARY"))
    rd.log(f"  full-data  {full_forget:+.4f}")
    rd.log(f"  mean {summary['mean']:+.4f}   std {summary['std']:.4f}   "
           f"min {summary['min']:+.4f}   max {summary['max']:+.4f}")
    rd.log(f"  below 0: {summary['n_below_0']}/{trials}   "
           f"below -0.5: {summary['n_below_neg_0_5']}/{trials}")

    result = {
        "diagnostic": "nc3_mean_robustness",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cli": {"argv": sys.argv, "args": vars(a)},
        "resolved_config": cfg,
        "provenance": {
            "git_commit": commit,
            "git_dirty": None if dirty is None else bool(dirty),
            "git_dirty_files": dirty,
            "baseline_run_dir": str(run_dir) if run_dir else None,
            "baseline_checkpoint": str(ckpt_path),
            "baseline_checkpoint_sha256": sha256_file(ckpt_path),
            "baseline_env": baseline_env,
            "device": device,
            "device_detail": device_string(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "replay": {
            "strategy": "intended exact replay -- one epoch of "
                        "random_label_clfonly from the baseline checkpoint, "
                        "mirroring scripts/unlearn_across_classes.run_condition "
                        "call for call (including the epoch-0 trajectory "
                        "callback, which advances the global torch RNG by two "
                        "DataLoader iterations before the training epoch). "
                        "Describe it as EXACT only if replay_delta is 0.0",
            "verified": delta == 0.0 if delta is not None else None,
            "replayed_epoch1_nc3_centred_forget": replayed,
            "replay_delta": delta,
            "condition_prefix_executed": [],
            "prefixes_are_not_executed": True,
            **reference,
            "unlearn_method": UNLEARN_METHOD,
            "unlearn_epochs": UNLEARN_EPOCHS,
            "classifier_only": CLASSIFIER_ONLY,
            "split_mode": SPLIT_MODE,
            "unlearn_lr": u["lr"],
            "unlearn_weight_decay": u["weight_decay"],
            "experiment_seed": cfg["seed"],
            "epoch0_trajectory": trajectory.get(0),
            "replay_time_s": replay_time_s,
        },
        "procedure": {
            "class_mean_source": "training images under eval transforms "
                                 "(train_eval); test images reach the replayed "
                                 "trajectory callback exactly as in production, "
                                 "but never the class mean",
            "nc3_convention": "centred, exclude_from_centre=forget_class -- the "
                              "convention train.evaluate reports as "
                              "nc3_centred_forget",
            "sampling": "without replacement; per-trial rng = "
                        "numpy.random.default_rng([sample_seed, trial_index])",
            "extract_time_s": extract_time_s,
            "trials_time_s": trials_time_s,
        },
        "setup": {
            "num_classes": num_classes,
            "train_images": int(len(labels)),
            "forget_class": forget_class,
            "forget_class_train_images": n_forget_train,
            "k": k, "trials": trials, "sample_seed": a.sample_seed,
        },
        "full_data": {
            "nc3_centred_forget": full_forget,
            "nc3_centred_retain_mean": float(full["retain_mean"]),
            "nc3_centred_mean": float(full["mean"]),
        },
        "trials_detail": trial_records,
        "summary": summary,
    }
    rd.write_json("result.json", result)
    rd.log(f"\nrun complete -> {rd.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NC3 forget-class mean robustness under K-sample estimation")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="baseline run directory written by RunDir "
                                       "(config.json + ckpt.pt)")
    src.add_argument("--config", help="path to a YAML config; the checkpoint "
                                      "defaults to <out_dir>/<data>_<head>_seed<seed>/ckpt.pt")
    ap.add_argument("--ckpt", default=None, help="explicit checkpoint path")
    ap.add_argument("--forget-class", type=int, required=True)
    ap.add_argument("--k", type=int, default=40,
                    help="images used to estimate the forget-class mean per "
                         "trial (default 40, the face-set per-identity size)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sample-seed", type=int, required=True,
                    help="seed for the DIAGNOSTIC sampling only; deliberately "
                         "separate from the config's experiment seed, and "
                         "deliberately not defaulted")
    ap.add_argument("--reference-run", default=None,
                    help="the canonical random_label_clfonly run directory to "
                         "check the replay against; defaults to the sibling of "
                         "--run-dir for this forget class")
    ap.add_argument("--out-dir", default=None,
                    help="base directory for the result (default: the "
                         "config's out_dir)")
    ap.add_argument("--name", default=None, help="result directory name")
    ap.add_argument("--device", default=None, help="auto | cuda | cpu")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run from a dirty working tree, recording it in the "
                         "result; the result is then not reproducible from "
                         "the commit alone")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL")
    main(ap.parse_args())
