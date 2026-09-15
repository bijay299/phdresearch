#!/usr/bin/env python
"""
Feature-movement diagnostic: full-model and paired frozen-backbone.

THE QUESTION THIS ANSWERS
-------------------------
Every headline in this project rests on `random_label_clfonly`, which freezes
the backbone. Features cannot move under it, so it cannot address the
project's actual question -- whether an angular-margin loss forces forgetting
to proceed THROUGH the representation when the backbone is free. This script
runs full-model `random_label` from an existing baseline checkpoint and
measures, at every epoch, how far the forget class's features actually moved.

It measures. It does not tune, extend, or retrain a baseline.

WHAT IS MEASURED AT EVERY EPOCH (0..N)
--------------------------------------
  1. raw paired normalised-feature angular displacement vs epoch 0
  2. retain-anchored orthogonal-Procrustes-aligned displacement,
     forget and HELD-OUT retain, mean/median/p95, plus forget-minus-retain
  3. aligned forget-class-mean displacement + macro retain-class-mean control
  4. approximate leave-one-class-out controls -- retain classes measured the
     same WAY as the forget class. Descriptive, NOT an exchangeable null: a
     control rotation excludes two classes, the forget rotation one. See
     src/movement.py.
  5. linear CKA, SECONDARY, explicitly able to hide one-class movement
  6. centred AND uncentred per-class NC3
  7. output_forget / output_retain, and the linear probe

Epoch 0 is the pre-unlearning model, so it is both the movement reference
(displacement is identically zero there, which is the run's own correctness
gate) and the same-convention NC3 baseline. That last point matters: CLAUDE.md
forbids reading a negative uncentred NC3 as a flip without its own baseline,
and every uncentred delta written here is against this run's epoch 0.

WHAT THIS IS NOT
----------------
Not a replay. `random_label` here is reseeded immediately before it runs, and
historical CIFAR `random_label` rows were produced mid-sequence inside
`run_experiment.py` with the RNG carrying state from preceding conditions.
Runs from this script are REPLICATIONS, not bitwise reproductions. Where a
re-run agrees with an earlier artifact to four decimals, that is REPLICATION
EVIDENCE -- consistent with an unchanged computation -- and not a proof of
numerical identity, which would need a bitwise comparison this script does not
perform. `scripts/nc3_mean_robustness.py` is the script in this repo that does
exact replay; this one deliberately does not.

MODES
-----
Default: full-model `random_label`. `--classifier-only` freezes the backbone,
giving the paired control -- same code, same baseline, same loaders, same
sample stream. A pair is only a pair if both members report the SAME
`training_trace_sha256`; the driver records it so that can be checked rather
than assumed.

EXPOSURE -- two different quantities, neither of them gradient magnitude
------------------------------------------------------------------------
`random_label` runs one step per retain batch and cycles the forget loader, so
a small forget set is replayed far more often. Two summaries are recorded and
they do NOT agree:

  raw presentations per unique forget image per epoch
      CIFAR 44096/5000 = 8.8192      faces 12120/40 = 303        ~34.4x
  average unit-weight forget-CE coefficient per unique forget image per epoch
      CIFAR 352/5000 = 0.0704        faces 303/40 = 7.575        ~107.6x

The second divides forget-bearing optimiser steps by the number of unique
forget images: each step contributes one mean-reduced forget CE term of unit
weight, so replaying a small set more often does not multiply that term.
NEITHER ratio is an observed gradient magnitude -- both are counts derived
from the loop structure, and actual gradients depend on the loss surface.
`result.json` records predicted and observed counts and fails the run if they
disagree.

No checkpoint is written. The point is the trajectory, and a 3-epoch unlearned
model is reconstructible from the baseline plus this config.

    python scripts/feature_movement.py \
        --run-dir logs/postfix_seed0_clean/cifar10_arcface_seed0 \
        --forget-class 0 --out-dir logs/feature_movement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as D                      # noqa: E402
import metrics as M                   # noqa: E402
import movement as MV                 # noqa: E402
import train as TR                    # noqa: E402
import unlearn as UL                  # noqa: E402
from backbones import build_backbone  # noqa: E402
from heads import build_head          # noqa: E402
from utils import (RunDir, apply_overrides, banner, device_string,  # noqa: E402
                   git_commit, git_dirty_files, load_config, set_seed)

# Fixed, not configurable. This diagnostic exists to measure ONE condition:
# full-model random_label, the matchable clean condition on both heads (CE
# `finetune` never reaches output_forget=0 on CIFAR, so finetune cannot be
# matched across heads -- see notes/decisions.md 2026-09-13).
UNLEARN_METHOD = "random_label"
SPLIT_MODE = "all"


SCIENTIFIC_ROLES = ("new_experiment", "instrumented_replication",
                    "paired_control")


def resolve_scientific_role(requested: Optional[str],
                            classifier_only: bool) -> str:
    """Validate the run's scientific role. It is NEVER inferred.

    An earlier version guessed the role from the dataset name -- CIFAR meant
    replication, anything else meant new experiment. That is not a property of
    the dataset. It mislabelled the faces full-model run as `new_experiment`
    when it replicates the face pilot already recorded at commit c150b4c.
    Whether a run is new, a replication, or a control is a claim about the
    experimental record, and only the person launching it knows.

    Classifier-only runs exist solely to be compared against a full-model twin,
    so their role is fixed at `paired_control`. Full-model runs must state a
    role explicitly; there is no default, because the wrong default is exactly
    how the faces run was mislabelled.
    """
    if classifier_only:
        if requested not in (None, "paired_control"):
            raise ValueError(
                f"a classifier-only run is a paired_control by construction; "
                f"got --scientific-role {requested!r}"
            )
        return "paired_control"
    if requested is None:
        raise ValueError(
            "--scientific-role is required for a full-model run. Choose "
            "new_experiment (no comparable condition exists on this dataset) "
            "or instrumented_replication (this condition is already on record "
            "and is being re-measured with instrumentation). It is not "
            "inferable from the dataset name."
        )
    if requested == "paired_control":
        raise ValueError(
            "paired_control is reserved for classifier-only runs; a "
            "full-model run is the thing a control is compared against"
        )
    if requested not in SCIENTIFIC_ROLES:
        raise ValueError(
            f"unknown --scientific-role {requested!r}; "
            f"expected one of {SCIENTIFIC_ROLES}"
        )
    return requested


def exposure_accounting(n_retain: int, n_forget: int, batch_size: int,
                        epochs: int) -> dict:
    """Exact training exposure implied by `random_label`'s loader cycling.

    One optimiser step per RETAIN batch; each step pulls one forget batch and
    restarts the forget loader when it is exhausted. A small forget set is
    therefore replayed far more often than a large one, which confounds any
    cross-dataset comparison of feature movement unless it is stated. These
    are computed from the actual split sizes, not assumed.
    """
    steps = -(-n_retain // batch_size)                 # ceil
    fb = -(-n_forget // batch_size)                    # batches per forget pass
    last = n_forget - (fb - 1) * batch_size            # short batch of a pass
    full_passes, rem = divmod(steps, fb)
    restarts = full_passes - (1 if rem == 0 else 0)
    forget_pres = full_passes * n_forget
    if rem:
        forget_pres += min(rem, fb - 1) * batch_size + (last if rem == fb else 0)
    return {
        "batch_size": int(batch_size),
        "retain_samples": int(n_retain),
        "forget_samples": int(n_forget),
        "retain_batches_per_epoch": int(steps),
        "optimiser_steps_per_epoch": int(steps),
        "natural_forget_batches_per_pass": int(fb),
        "forget_loader_restarts_per_epoch": int(restarts),
        "retain_presentations_per_epoch": int(n_retain),
        "forget_presentations_per_epoch": int(forget_pres),
        "presentations_per_unique_retain_image": n_retain / max(n_retain, 1),
        "presentations_per_unique_forget_image": forget_pres / max(n_forget, 1),
        # Raw presentations and loss WEIGHTING are different quantities.
        # Every step contributes one mean-reduced forget CE term of unit
        # weight, so that term's total coefficient per epoch is `steps`,
        # spread over n_forget unique images regardless of how many times each
        # appeared. A tiny forget set is replayed more often but each
        # appearance carries a proportionally smaller share of its batch mean.
        # NEITHER of these is an observed gradient magnitude; both are counts
        # derived from the loop structure.
        "forget_bearing_optimiser_steps": int(steps),
        "avg_unit_weight_forget_ce_coefficient_per_unique_forget_image":
            steps / max(n_forget, 1),
        "total_optimiser_steps": int(steps * epochs),
        "total_forget_presentations": int(forget_pres * epochs),
    }


def forget_batch_sizes(n_forget: int, batch_size: int,
                       n_steps: int) -> list:
    """Size of the forget batch pulled at each of `n_steps` retain steps.

    `random_label` pulls one forget batch per retain step and restarts the
    forget loader when it is exhausted, so step i takes batch (i mod fb) of a
    pass and the short final batch of a pass lands on every fb-th step. Which
    IMAGES appear depends on the shuffle; how MANY does not, so the sizes are
    predictable exactly and the per-step dose can be predicted before the run.
    """
    fb = -(-n_forget // batch_size)
    last = n_forget - (fb - 1) * batch_size
    return [last if (i % fb) == fb - 1 else batch_size for i in range(n_steps)]


def dose_accounting(n_retain: int, n_forget: int, batch_size: int, epochs: int,
                    active_steps_per_epoch: Optional[int],
                    forget_loss_weight: float) -> dict:
    """Predicted dose under the optional active-step / loss-weight controls.

    CANDIDATE versus ACTIVE is the distinction the whole control rests on. A
    candidate presentation is a forget image the loop pulled; an active
    presentation is one that was actually forwarded and carried loss. With no
    controls the two are equal, which is what the uncontrolled runs measured.

    NEITHER the presentation counts NOR the weighted coefficient mass is a
    gradient magnitude. They are counts and coefficients derived from the loop
    structure; the gradients they produce depend on the loss surface and are
    not measured anywhere in this script. The same warning applies to
    `exposure_accounting` above and is repeated because the weighted numbers
    look more like a physical dose than they are.
    """
    steps = -(-n_retain // batch_size)
    sizes = forget_batch_sizes(n_forget, batch_size, steps)
    if active_steps_per_epoch is None:
        flags = [True] * steps
        m = steps
    else:
        m = int(active_steps_per_epoch)
        flags = UL.evenly_spaced_active_steps(steps, m)
    lam = float(forget_loss_weight)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(
            f"forget_loss_weight must be finite and strictly positive, got "
            f"{forget_loss_weight!r}")
    candidate = int(sum(sizes))
    active_pres = int(sum(s for s, f in zip(sizes, flags) if f))
    return {
        "dose_controlled": bool(active_steps_per_epoch is not None
                                or lam != 1.0),
        "forget_active_steps_per_epoch": int(m),
        "forget_loss_weight": lam,
        "retain_steps_per_epoch": int(steps),
        "candidate_forget_presentations_per_epoch": candidate,
        "active_forget_presentations_per_epoch": active_pres,
        "active_forget_bearing_steps_per_epoch": int(m),
        "active_presentations_per_unique_forget_image_per_epoch":
            active_pres / max(n_forget, 1),
        "total_weighted_forget_loss_coefficient_mass_per_epoch": m * lam,
        "avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch":
            m * lam / max(n_forget, 1),
        "total_candidate_forget_presentations": candidate * epochs,
        "total_active_forget_presentations": active_pres * epochs,
        "total_active_forget_bearing_steps": int(m * epochs),
        "total_weighted_forget_loss_coefficient_mass": m * lam * epochs,
        "active_step_indices_per_epoch": [i for i, f in enumerate(flags) if f],
        "not_a_gradient_magnitude": (
            "Presentation counts and weighted loss coefficients are derived "
            "from the loop structure. They are not observed gradient "
            "magnitudes and must not be reported as such."),
    }


# ----------------------------------------------------------------------
# provenance helpers
# ----------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def gpu_identity(device: str) -> Dict[str, object]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"device": device, "gpu_name": None, "gpu_uuid": None}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return {
        "device": device,
        "gpu_index": int(i),
        "gpu_name": torch.cuda.get_device_name(i),
        "gpu_total_mem_bytes": int(props.total_memory),
        "cuda_visible_devices": __import__("os").environ.get(
            "CUDA_VISIBLE_DEVICES"),
    }


def base_indices(ds) -> np.ndarray:
    """Dataset-level index of each row, in the order a shuffle=False loader
    yields them. Subsets carry their own index map; everything else is
    sequential."""
    if isinstance(ds, Subset):
        return np.asarray(ds.indices, dtype=np.int64)
    return np.arange(len(ds), dtype=np.int64)


def load_baseline(run_dir: Path, ckpt_arg: Optional[str],
                  overrides: Sequence[str]) -> Tuple[dict, Path, dict]:
    cfg_file = run_dir / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"{cfg_file} not found -- --run-dir expects a directory written by "
            f"RunDir (config.json, env.json, ckpt.pt)."
        )
    cfg = json.loads(cfg_file.read_text())
    env_file = run_dir / "env.json"
    baseline_env = json.loads(env_file.read_text()) if env_file.exists() else {}
    ckpt_path = Path(ckpt_arg) if ckpt_arg else run_dir / "ckpt.pt"
    cfg = apply_overrides(cfg, list(overrides))
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. This script never trains a "
            f"baseline -- point --run-dir at a directory that already has one."
        )
    return cfg, ckpt_path, baseline_env


def load_model(cfg: dict, ckpt_path: Path, num_classes: int):
    """Rebuild the baseline model and load strictly: any config/checkpoint
    mismatch means the config on hand does not describe the checkpoint on
    hand, and every number would be attributed to the wrong run."""
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
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    return backbone, head, head_name


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------

def main(a: argparse.Namespace) -> None:
    # ---- provenance gate, before anything scientific happens ------------
    dirty = git_dirty_files(cwd=ROOT)
    commit = git_commit()
    if dirty is None or dirty:
        detail = ("git could not report the working tree state"
                  if dirty is None else
                  "working tree has uncommitted changes:\n    "
                  + "\n    ".join(dirty[:20])
                  + ("\n    ..." if len(dirty) > 20 else ""))
        if not a.allow_dirty:
            raise SystemExit(
                f"refusing to run from an unverifiable tree -- {detail}\n"
                f"Every number in this project must be traceable to a commit "
                f"(see CLAUDE.md). Commit or stash first. --allow-dirty "
                f"overrides this and records the dirty state in the result, "
                f"but the result is then not reproducible from {commit} alone."
            )
        print(f"WARNING: --allow-dirty -- {detail}")

    run_dir = Path(a.run_dir)
    cfg, ckpt_path, baseline_env = load_baseline(run_dir, a.ckpt, a.set)
    device = resolve_device(a.device or cfg.get("device", "auto"))
    seed = int(cfg["seed"])
    fc = int(a.forget_class)
    epochs = int(a.epochs)
    if epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {epochs}")

    classifier_only = bool(a.classifier_only)
    mode_tag = "clfonly" if classifier_only else "full"
    update_mode = "classifier_only" if classifier_only else "full_model"
    method_label = f"{UNLEARN_METHOD}_{mode_tag}"
    scientific_role = resolve_scientific_role(a.scientific_role, classifier_only)
    name = a.name or (
        f"{cfg['data']['name']}_{cfg['head']['name']}_seed{seed}"
        f"_random_label_{mode_tag}_fc{fc}"
    )
    # RunDir refuses a non-empty target, so a clobbered result is impossible
    # rather than merely unlikely.
    rd = RunDir.create(a.out_dir, name, dict(
        cfg, _feature_movement=True, _method=UNLEARN_METHOD,
        _classifier_only=classifier_only, _update_mode=update_mode,
        _scientific_role=scientific_role,
        _forget_class=fc, _epochs=epochs,
        _anchor_fraction=a.anchor_fraction, _anchor_seed=a.anchor_seed,
        _n_controls=a.n_controls, _baseline_run_dir=str(run_dir),
        _forget_active_steps_per_epoch=a.forget_active_steps_per_epoch,
        _forget_loss_weight=a.forget_loss_weight,
    ))
    rd.log(banner(name))
    rd.log(f"  baseline      {run_dir}")
    rd.log(f"  checkpoint    {ckpt_path}")
    rd.log(f"  device        {device_string()} / {device}")
    rd.log(f"  commit        {commit}")
    rd.log(f"  mode          {'classifier-only (backbone FROZEN)' if classifier_only else 'full model'}")
    rd.log(f"  update mode   {update_mode}")
    rd.log(f"  science role  {scientific_role}")

    # ---- data ----------------------------------------------------------
    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(
        cfg["data"], seed=seed, log=rd.log)
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    # shuffle=False everywhere that feeds a measurement: identical samples in
    # identical order at every epoch is what makes a PAIRED angle meaningful.
    # IndexedDataset so the run records which samples it ACTUALLY saw rather
    # than re-hashing the array it asked for. Wrapping changes only what
    # __getitem__ returns -- sampler, order and worker seeding are untouched.
    train_eval_loader = D.make_loader(D.IndexedDataset(train_eval_ds), None,
                                      512, False, nw)
    test_loader = D.make_loader(test_ds, None, 512, False, nw)

    te_expected = np.arange(len(train_eval_ds), dtype=np.int64)
    te_base = base_indices(train_eval_ds)
    te_targets = D.get_targets(train_eval_ds)
    rd.log(f"  train_eval    {len(te_expected)} samples")

    targets = D.get_targets(train_ds)
    split = D.make_forget_split(targets, fc, SPLIT_MODE,
                                cfg["unlearn"]["forget_fraction"], seed)
    rd.log(f"  {split.summary()}")
    forget_loader = D.make_loader(D.IndexedDataset(train_ds),
                                  split.forget_idx, bs, True, nw)
    retain_loader = D.make_loader(D.IndexedDataset(train_ds),
                                  split.retain_idx, bs, True, nw)

    exposure = exposure_accounting(int(split.retain_idx.size),
                                   int(split.forget_idx.size), bs, epochs)
    rd.log(f"  exposure      {exposure['optimiser_steps_per_epoch']} steps/epoch, "
           f"{exposure['natural_forget_batches_per_pass']} forget batches/pass, "
           f"{exposure['forget_loader_restarts_per_epoch']} restarts/epoch, "
           f"{exposure['presentations_per_unique_forget_image']:.2f} "
           f"presentations per unique forget image per epoch")

    dose = dose_accounting(int(split.retain_idx.size),
                           int(split.forget_idx.size), bs, epochs,
                           a.forget_active_steps_per_epoch,
                           a.forget_loss_weight)
    # Two independent derivations of the candidate stream. They must agree, or
    # one of the two accountings is wrong and every dose statement is suspect.
    if (dose["candidate_forget_presentations_per_epoch"]
            != exposure["forget_presentations_per_epoch"]):
        raise ValueError(
            f"candidate accounting disagrees: dose says "
            f"{dose['candidate_forget_presentations_per_epoch']}, exposure "
            f"says {exposure['forget_presentations_per_epoch']} forget "
            f"presentations per epoch")
    rd.log(f"  dose          {'CONTROLLED' if dose['dose_controlled'] else 'uncontrolled (all steps, unit weight)'}: "
           f"{dose['active_forget_bearing_steps_per_epoch']}/"
           f"{dose['retain_steps_per_epoch']} active steps/epoch, "
           f"weight {dose['forget_loss_weight']:.10g}")
    rd.log(f"  dose          candidate {dose['candidate_forget_presentations_per_epoch']} -> "
           f"active {dose['active_forget_presentations_per_epoch']} "
           f"presentations/epoch "
           f"({dose['active_presentations_per_unique_forget_image_per_epoch']:.4f} "
           f"per unique forget image)")
    rd.log(f"  dose          weighted coefficient mass "
           f"{dose['total_weighted_forget_loss_coefficient_mass_per_epoch']:.6g}/epoch "
           f"({dose['avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch']:.6g} "
           f"per unique forget image) -- NOT a gradient magnitude")

    # Alignment anchors are chosen ONCE, over the train_eval rows, and reused
    # at every epoch -- a per-epoch redraw would make epochs incomparable.
    anchor_idx, retain_eval_idx = MV.stratified_anchor_split(
        te_targets, fc, a.anchor_fraction, a.anchor_seed)
    control_classes = MV.select_control_classes(te_targets, fc, a.n_controls,
                                                a.anchor_seed)
    n_retain_classes = int(len(np.unique(te_targets)) - 1)
    exhaustive = len(control_classes) == n_retain_classes
    rd.log(f"  anchors {anchor_idx.size} / retain-eval {retain_eval_idx.size} "
           f"/ forget {int((te_targets == fc).sum())}")
    rd.log(f"  CONTROL CLASSES: {len(control_classes)} of {n_retain_classes} "
           f"retain classes "
           f"({'exhaustive' if exhaustive else 'DETERMINISTIC SAMPLE -- descriptive, not exhaustive'})")
    rd.log(f"  control identities: {list(map(int, control_classes))}")

    backbone0, head0, head_name = load_model(cfg, ckpt_path, num_classes)

    # ---- per-epoch measurement -----------------------------------------
    state: Dict[str, object] = {}
    timings = []

    def epoch_eval(bb, hd, epoch: int) -> None:
        t0 = time.time()
        f_tr, y_tr, _p_tr, i_tr = TR.extract_with_indices(
            bb, hd, train_eval_loader, device)
        f_te, y_te, p_te = TR.extract(bb, hd, test_loader, device)

        # Validate what the loader OBSERVABLY yielded -- duplication, omission,
        # count and ordering are separated so the error names the failure.
        MV.check_observed_indices(i_tr, te_expected, "train_eval")
        if not np.array_equal(y_tr, te_targets):
            bad = int(np.flatnonzero(y_tr != te_targets)[0])
            raise ValueError(
                f"train_eval: observed label at row {bad} is {y_tr[bad]}, "
                f"dataset says {te_targets[bad]}"
            )
        observed_hash = MV.hash_indices(i_tr, y_tr)
        if epoch == 0:
            state["f0"] = f_tr
            state["y0"] = y_tr
            state["obs_hash"] = observed_hash
        # Paired check against epoch 0's OBSERVED stream, not the expected one.
        MV.check_paired(np.asarray(state["y0"]), y_tr, np.asarray(state["i0"])
                        if "i0" in state else i_tr, i_tr)
        if observed_hash != state["obs_hash"]:
            raise ValueError(
                "observed train_eval index/label stream changed mid-run")
        state["i0"] = i_tr

        mv = MV.movement_report(np.asarray(state["f0"]), f_tr, y_tr,
                                num_classes, fc, anchor_idx, retain_eval_idx,
                                control_classes=control_classes)

        out_acc = TR.output_accuracy(y_te, p_te, fc)
        probe = M.linear_probe(f_tr, y_tr, f_te, y_te, target_class=fc,
                               seed=seed)
        W = hd.weight.detach().cpu().numpy()
        nc3_c = M.nc3_alignment(f_tr, y_tr, W, num_classes, centre=True,
                                exclude_from_centre=fc)
        nc3_u = M.nc3_alignment(f_tr, y_tr, W, num_classes, centre=False)
        u_forget = float(nc3_u["per_class"][fc])
        u_keep = np.delete(nc3_u["per_class"], fc)
        u_keep = u_keep[~np.isnan(u_keep)]

        if epoch == 0:
            state["nc3_c0"] = nc3_c.get("forget", float("nan"))
            state["nc3_u0"] = u_forget

        c0 = float(state["nc3_c0"])
        u0 = float(state["nc3_u0"])
        c_now = nc3_c.get("forget", float("nan"))

        row = {
            "head": head_name, "method": method_label,
            "forget_class": fc, "seed": seed, "epoch": epoch,
            "output_forget": out_acc.get("forget", float("nan")),
            "output_retain": out_acc.get("retain", float("nan")),
            "output_overall": out_acc["overall"],
            "probe_forget": probe.get("forget", float("nan")),
            "probe_retain": probe.get("retain", float("nan")),
            "probe_overall": probe["overall"],
            "nc1_angular": M.nc1_angular(f_tr, y_tr, num_classes),
            # BOTH conventions, per CLAUDE.md, each with this run's own epoch-0
            # baseline beside it. A negative uncentred value is NOT a flip
            # unless it changed sign against `nc3_uncentred_forget_epoch0`.
            "nc3_centred_forget": c_now,
            "nc3_centred_retain_mean": nc3_c.get("retain_mean", float("nan")),
            "nc3_centred_forget_epoch0": c0,
            "nc3_centred_forget_delta_vs_epoch0": float(c_now - c0),
            "nc3_centred_sign_reversal_vs_epoch0": bool(c0 > 0 > c_now),
            "nc3_uncentred_forget": u_forget,
            "nc3_uncentred_retain_mean": (float(u_keep.mean())
                                          if u_keep.size else float("nan")),
            "nc3_uncentred_forget_epoch0": u0,
            "nc3_uncentred_forget_delta_vs_epoch0": float(u_forget - u0),
            "nc3_uncentred_sign_reversal_vs_epoch0": bool(u0 > 0 > u_forget),
            "observed_index_sha256": observed_hash,
            **mv,
        }
        rd.append_jsonl("trajectory.jsonl", row)
        rd.append_jsonl("nc3_per_class.jsonl", {
            "epoch": epoch,
            "nc3_centred_per_class": nc3_c["per_class"],
            "nc3_uncentred_per_class": nc3_u["per_class"],
        })
        timings.append({"epoch": epoch, "eval_s": time.time() - t0})
        rd.log(
            f"  ep {epoch}  out_f {row['output_forget']:.4f} "
            f"out_r {row['output_retain']:.4f}  "
            f"aligned_fgt {mv['aligned_forget']['mean']:.3f} deg  "
            f"aligned_ret {mv['aligned_retain_eval']['mean']:.3f} deg  "
            f"ctrl {mv[MV.CONTROL_MEAN_KEY]:.3f} deg  "
            f"nc3_c {row['nc3_centred_forget']:+.4f}  "
            f"nc3_u {row['nc3_uncentred_forget']:+.4f}  "
            f"cka {mv['cka_linear_secondary']:.4f}  "
            f"({time.time() - t0:.0f}s)"
        )

    # ---- training trace -------------------------------------------------
    # Hashes the exact sample stream the method trained on: retain indices,
    # forget indices and the random targets, per step. Two runs that differ
    # only in whether the backbone is frozen MUST produce the same hash --
    # that is what makes the pair "paired". The trace is read-only (it draws
    # no random numbers and touches nothing); src/test_unlearn_trace.py proves
    # enabling it leaves parameters bitwise identical.
    trace_state = {"h": hashlib.sha256(), "steps": 0,
                   "retain_pres": 0, "forget_pres": 0}

    def trace(epoch, step, retain_idx, forget_idx, rnd_targets, n_r, n_f):
        h = trace_state["h"]
        h.update(np.int64([epoch, step, n_r, n_f]).tobytes())
        for arr in (retain_idx, forget_idx, rnd_targets):
            h.update(b"\x00" if arr is None
                     else np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        trace_state["steps"] += 1
        trace_state["retain_pres"] += int(n_r)
        trace_state["forget_pres"] += int(n_f)

    # ---- dose trace, separate from the candidate trace -------------------
    # `training_trace_sha256` above stays the COMPLETE CANDIDATE stream, so it
    # remains comparable with runs made before dose control existed and with
    # runs at a different dose. What the schedule then DID with that stream --
    # which steps were active, at what weight -- is a different fact and gets
    # its own hash. Collapsing the two would make a dose change look like a
    # sample-stream change, and the pairing gate would stop meaning anything.
    dose_state = {"h": hashlib.sha256(), "rows": 0, "active_steps": 0,
                  "active_pres": 0, "candidate_pres": 0, "weight_mass": 0.0}

    def dose_trace(epoch, step, active, weight, forget_idx, rnd_targets, n_f):
        h = dose_state["h"]
        h.update(np.int64([epoch, step, 1 if active else 0, n_f]).tobytes())
        h.update(np.float64([weight]).tobytes())
        for arr in (forget_idx, rnd_targets):
            h.update(b"\x00" if arr is None
                     else np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        dose_state["rows"] += 1
        dose_state["candidate_pres"] += int(n_f)
        if active:
            dose_state["active_steps"] += 1
            dose_state["active_pres"] += int(n_f)
            dose_state["weight_mass"] += float(weight)

    # ---- run ------------------------------------------------------------
    # Explicit reseed immediately before the method, so this run does not
    # inherit RNG state from any preceding condition. This is what makes the
    # cell reproducible on its own; it is also why this is a replication of
    # the historical random_label rows, not a replay of them.
    set_seed(seed)
    t0 = time.time()
    UL.run_unlearning(
        UNLEARN_METHOD, backbone0, head0, device,
        forget_loader=forget_loader, retain_loader=retain_loader,
        num_classes=num_classes, epochs=epochs,
        lr=cfg["unlearn"]["lr"], weight_decay=cfg["unlearn"]["weight_decay"],
        classifier_only=classifier_only, log=rd.log, epoch_eval=epoch_eval,
        trace=trace, dose_trace=dose_trace,
        forget_active_steps_per_epoch=a.forget_active_steps_per_epoch,
        forget_loss_weight=a.forget_loss_weight,
    )
    method_time_s = time.time() - t0

    rows = [json.loads(line)
            for line in rd.path("trajectory.jsonl").read_text().splitlines()]
    # Every row must describe THIS run. The `method` field was once hardcoded
    # to the full-model label, so frozen-backbone runs wrote rows claiming to
    # be full-model runs -- a metadata defect that survived into four
    # published artifacts. Assert it here rather than trust it.
    bad = [r["epoch"] for r in rows if r.get("method") != method_label]
    if bad:
        raise ValueError(
            f"trajectory rows at epochs {bad} carry a method label that "
            f"disagrees with this run ({method_label!r})"
        )
    first_zero = next((r["epoch"] for r in rows
                       if r["epoch"] > 0 and r["output_forget"] == 0.0), None)

    measured_exposure = {
        "observed_optimiser_steps_total": trace_state["steps"],
        "observed_retain_presentations_total": trace_state["retain_pres"],
        "observed_forget_presentations_total": trace_state["forget_pres"],
        "observed_forget_presentations_per_epoch":
            trace_state["forget_pres"] / epochs,
        "observed_presentations_per_unique_forget_image_per_epoch":
            trace_state["forget_pres"] / epochs / max(int(split.forget_idx.size), 1),
    }
    # The predicted accounting must match what actually happened, or the
    # accounting is wrong and every exposure statement built on it is too.
    if measured_exposure["observed_optimiser_steps_total"] != exposure["total_optimiser_steps"]:
        raise ValueError(
            f"exposure accounting mismatch: predicted "
            f"{exposure['total_optimiser_steps']} steps, observed "
            f"{measured_exposure['observed_optimiser_steps_total']}")
    if measured_exposure["observed_forget_presentations_total"] != exposure["total_forget_presentations"]:
        raise ValueError(
            f"exposure accounting mismatch: predicted "
            f"{exposure['total_forget_presentations']} forget presentations, "
            f"observed {measured_exposure['observed_forget_presentations_total']}")

    n_forget_unique = max(int(split.forget_idx.size), 1)
    dose_observed = {
        "observed_steps_total": dose_state["rows"],
        "observed_candidate_forget_presentations_total":
            dose_state["candidate_pres"],
        "observed_active_forget_bearing_steps_total":
            dose_state["active_steps"],
        "observed_active_forget_presentations_total": dose_state["active_pres"],
        "observed_active_presentations_per_unique_forget_image_per_epoch":
            dose_state["active_pres"] / epochs / n_forget_unique,
        "observed_total_weighted_forget_loss_coefficient_mass":
            dose_state["weight_mass"],
        "observed_weighted_forget_loss_coefficient_mass_per_epoch":
            dose_state["weight_mass"] / epochs,
        "observed_avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch":
            dose_state["weight_mass"] / epochs / n_forget_unique,
    }
    # Predicted-versus-observed, on every count the dose claim rests on. If the
    # schedule did not do what the accounting says, the accounting is fiction.
    for label, pred, obs in (
        ("steps", exposure["total_optimiser_steps"],
         dose_observed["observed_steps_total"]),
        ("candidate forget presentations",
         dose["total_candidate_forget_presentations"],
         dose_observed["observed_candidate_forget_presentations_total"]),
        ("active forget-bearing steps",
         dose["total_active_forget_bearing_steps"],
         dose_observed["observed_active_forget_bearing_steps_total"]),
        ("active forget presentations",
         dose["total_active_forget_presentations"],
         dose_observed["observed_active_forget_presentations_total"]),
    ):
        if pred != obs:
            raise ValueError(
                f"dose accounting mismatch on {label}: predicted {pred}, "
                f"observed {obs}")
    mass_pred = dose["total_weighted_forget_loss_coefficient_mass"]
    mass_obs = dose_observed["observed_total_weighted_forget_loss_coefficient_mass"]
    if abs(mass_pred - mass_obs) > 1e-9 * max(1.0, abs(mass_pred)):
        raise ValueError(
            f"dose accounting mismatch on weighted coefficient mass: "
            f"predicted {mass_pred!r}, observed {mass_obs!r}")
    rd.log(f"  dose observed {dose_observed['observed_active_forget_bearing_steps_total']} "
           f"active steps, "
           f"{dose_observed['observed_active_forget_presentations_total']} active "
           f"presentations, mass {mass_obs:.6g} -- matches prediction")

    rd.write_json("result.json", {
        "name": name,
        "method": method_label,
        "classifier_only": classifier_only,
        # Two ORTHOGONAL fields. `update_mode` is a mechanical fact about what
        # the optimiser touched. `scientific_role` is a claim about how this
        # run relates to the record, which no code can infer -- it is supplied
        # by the person launching the run. The previous version guessed the
        # role from the dataset NAME, which mislabelled the faces full-model
        # run as a new experiment when it in fact replicates the c150b4c face
        # pilot.
        "update_mode": update_mode,
        "scientific_role": scientific_role,
        # The COMPLETE CANDIDATE stream -- unchanged by dose control, so it
        # stays comparable with runs made before dose control existed.
        "training_trace_sha256": trace_state["h"].hexdigest(),
        # What the schedule DID with that stream: per step, the active flag and
        # the applied weight alongside the forget indices and random targets.
        "active_dose_trace_sha256": dose_state["h"].hexdigest(),
        "exposure_predicted": exposure,
        "exposure_observed": measured_exposure,
        "dose_predicted": dose,
        "dose_observed": dose_observed,
        "controls": {
            "n_control_classes": int(len(control_classes)),
            "n_retain_classes": n_retain_classes,
            "exhaustive": bool(exhaustive),
            "selection": ("all retain classes" if exhaustive else
                          "deterministic sample -- descriptive, not exhaustive"),
            "control_class_identities": [int(c) for c in control_classes],
            "interpretation": (
                "APPROXIMATE leave-one-class-out controls, NOT exchangeable "
                "with the forget measurement: a control rotation excludes two "
                "classes (forget + control) while the forget rotation excludes "
                "one, so controls are fitted on fewer anchors. The DIRECTION "
                "of that asymmetry is not established -- it must not be "
                "described as inflating control displacement or as making the "
                "comparison conservative. Descriptive only: no significance "
                "claim, and no claim that the forget class exceeded ALL "
                "retain classes."),
        },
        "forget_class": fc,
        "seed": seed,
        "epochs": epochs,
        "num_classes": num_classes,
        "head": head_name,
        "first_epoch_output_forget_zero": first_zero,
        "provenance": {
            "git_commit": commit,
            "git_dirty": None if dirty is None else bool(dirty),
            "git_dirty_files": dirty,
            "argv": sys.argv,
            "baseline_run_dir": str(run_dir),
            "baseline_env": baseline_env,
            "baseline_checkpoint": str(ckpt_path),
            "baseline_checkpoint_sha256": sha256_file(ckpt_path),
            "config": cfg,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            **gpu_identity(device),
        },
        "splits": {
            "train_eval_n": int(len(te_expected)),
            "train_eval_base_index_sha256": MV.hash_indices(te_base, te_targets),
            "train_eval_observed_index_sha256": state.get("obs_hash"),
            "forget_train_n": int(split.forget_idx.size),
            "retain_train_n": int(split.retain_idx.size),
            "forget_heldout_n": int(split.forget_heldout_idx.size),
            "anchor_n": int(anchor_idx.size),
            "retain_eval_n": int(retain_eval_idx.size),
            "anchor_fraction": a.anchor_fraction,
            "anchor_seed": a.anchor_seed,
            "anchors_per_dim": float(anchor_idx.size
                                     / backbone0.feat_dim),
            "control_classes": [int(c) for c in control_classes],
        },
        "timings": {"method_s": method_time_s, "per_epoch": timings},
    })
    rd.log(f"  method_time {method_time_s:.1f}s")
    rd.log(f"  first epoch with output_forget == 0: {first_zero}")
    rd.log(f"  training trace sha256:   {trace_state['h'].hexdigest()}")
    rd.log(f"  active dose trace sha256: {dose_state['h'].hexdigest()}")
    rd.log(f"run complete -> {rd.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="baseline run directory (config.json + ckpt.pt). "
                         "This script never trains a baseline.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--forget-class", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3,
                    help="unlearning epochs (default 3, matching base.yaml)")
    ap.add_argument("--out-dir", default="logs/feature_movement")
    ap.add_argument("--name", default=None)
    ap.add_argument("--anchor-fraction", type=float, default=0.5)
    ap.add_argument("--anchor-seed", type=int, default=0,
                    help="governs the anchor/eval split and null-control "
                         "selection only; the experiment seed comes from the "
                         "baseline config and is not touched here")
    ap.add_argument("--scientific-role", default=None,
                    choices=list(SCIENTIFIC_ROLES),
                    help="required for full-model runs: new_experiment or "
                         "instrumented_replication. Never inferred. "
                         "Classifier-only runs are always paired_control.")
    ap.add_argument("--classifier-only", action="store_true",
                    help="freeze the backbone (paired control). Absent, the "
                         "full-model path is taken exactly as before.")
    ap.add_argument("--n-controls", type=int, default=8,
                    help="leave-one-class-out null controls; each costs one "
                         "extra Procrustes fit per epoch")
    ap.add_argument("--forget-active-steps-per-epoch", type=int, default=None,
                    help="dose control: carry the forget term on only this "
                         "many of the epoch's retain steps, chosen evenly "
                         "spaced. Default (unset) = every step, which is the "
                         "uncontrolled behaviour every earlier run used. The "
                         "candidate forget batch and its random targets are "
                         "still drawn on inactive steps, so the candidate "
                         "training trace stays comparable.")
    ap.add_argument("--forget-loss-weight", type=float, default=1.0,
                    help="dose control: coefficient on the forget CE term at "
                         "an active step (default 1.0). Must be finite and "
                         "positive. This is a loss coefficient, NOT a "
                         "gradient magnitude.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL")
    main(ap.parse_args())
