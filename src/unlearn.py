"""
Unlearning methods.

These are re-implementations, not the originals. Before reporting any
comparison against published numbers, reproduce the reference implementation
at github.com/ycgao1/CMF_Unlearning and check that these agree.

Implemented
-----------
retrain          gold-standard reference: train from scratch on retain only
finetune         fine-tune on retain set only (a weak but honest baseline)
neggrad          gradient ascent on forget set          (unstable by design)
neggrad_plus     ascent on forget + descent on retain   (the usable version)
random_label     relabel forget samples uniformly at random
classifier_only  *** the diagnostic, not a real method ***

`classifier_only` freezes the backbone and updates ONLY the head. The AISTATS
paper's key experiment: if this matches full-model unlearning at the output
level, then output-level forgetting says nothing about the representation.
Run it in every condition.

The margin question
-------------------
Every method here calls `head(features, labels)` during unlearning, so the
margin is applied under ArcFace/CosFace exactly as it is during training.
That is deliberate. Whether the margin blocks the classifier-drift shortcut
is the thing being tested -- do not disable it to make the methods "behave".

Per-epoch trajectories
-----------------------
`finetune`, `neggrad`, `neggrad_plus` and `random_label` take an optional
`epoch_eval(backbone, head, epoch)` callback, called once before any
training (epoch 0, the shared pre-unlearning starting point) and once after
every epoch -- unconditionally; it is the callback itself
(`scripts/run_experiment.py`'s `make_epoch_eval`, governed by
`unlearn.trajectory_every`) that decides whether a given epoch is cheap to
skip. The caller uses this to write a trajectory file -- see
notes/decisions.md, 2026-09-13: two heads can reach output_forget=0 at very
different epoch counts, so a table built at one fixed epoch count compares
methods at different points on their own trajectories, not at a matched
outcome.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import backbones as BK
import train as TR
from heads import build_head
from utils import set_seed


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _clone(backbone: nn.Module, head: nn.Module):
    return copy.deepcopy(backbone), copy.deepcopy(head)


def _params(backbone, head, classifier_only: bool):
    if classifier_only:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        return list(head.parameters())
    return list(backbone.parameters()) + list(head.parameters())


def _optimizer(params, lr: float, weight_decay: float, kind: str = "sgd"):
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def _step(backbone, head, x, y, sign: float = 1.0):
    """Forward + CE. sign=-1 turns descent into ascent."""
    logits = head(backbone(x), y)
    return sign * F.cross_entropy(logits, y)


def evenly_spaced_active_steps(n_steps: int, n_active: int) -> List[bool]:
    """Which of `n_steps` optimiser steps carry the forget term: exactly
    `n_active` of them, spread as evenly as the integers allow.

    Step i is active iff the running count `floor((i+1)*m/S)` advances past
    `floor(i*m/S)`. That is a Bresenham line, and it has the two properties
    this control needs:

      * EXACTLY m steps are selected. The count starts at 0, ends at
        floor(S*m/S) = m, and never jumps by more than 1 because m <= S.
      * The gaps are the two integers either side of S/m and nothing else, so
        the dose is spread across the epoch rather than bunched at one end.
        At the production setting S=303, m=9 the gaps are 33 and 34.

    Spacing matters because the alternative -- taking the first m steps, or
    every k-th step with k = S//m -- concentrates the whole epoch's forget
    exposure into a fraction of the epoch, which is a different intervention
    from the one being tested.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(
            f"n_steps must be a positive integer, got {n_steps!r}"
        )
    if isinstance(n_active, bool) or not isinstance(n_active, int):
        raise ValueError(
            f"forget_active_steps_per_epoch must be an integer, got "
            f"{n_active!r}"
        )
    if not 1 <= n_active <= n_steps:
        raise ValueError(
            f"forget_active_steps_per_epoch must satisfy 1 <= m <= S, got "
            f"m={n_active} with S={n_steps} retain steps per epoch. m=0 would "
            f"be no unlearning at all and m>S cannot be scheduled."
        )
    return [((i + 1) * n_active) // n_steps > (i * n_active) // n_steps
            for i in range(n_steps)]


def _unpack(batch):
    """Accept `(x, y)` or `(x, y, i)` batches; return `(x, y, i_or_None)`.

    Loaders over `data.IndexedDataset` yield the 3-tuple so a run can record
    which samples it actually trained on. Plain loaders are unaffected: the
    2-tuple path is what every existing caller takes and it behaves exactly
    as before.
    """
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


# ----------------------------------------------------------------------
# methods
# ----------------------------------------------------------------------

def retrain(backbone, head, retain_loader: DataLoader, device: str,
           backbone_cfg: dict, head_name: str, head_kwargs: dict,
           num_classes: int, train_cfg: dict, seed: int,
           classifier_only: bool = False, log: Optional[Callable] = None, **_):
    """
    Gold-standard reference: a fresh backbone and head, randomly initialised,
    trained from scratch on the retain set only. This is the 77.35 in the
    AISTATS table -- every probe_forget number elsewhere in this file is
    meaningless without a number like it to compare against.

    `backbone` and `head` are accepted only to match the call signature of
    every other function in METHODS -- they are IGNORED. Warm-starting this
    from the model being unlearned would make it stop being the reference.

    Uses the same seed as the original run (weight init + loader shuffling),
    so the only difference between this model and the original is the
    training DATA -- retain-only here, everything there. That reseeds the
    global RNG, so if you are calling this alongside other unlearning
    methods in the same process, run it last.
    """
    if classifier_only:
        raise ValueError(
            "retrain has no classifier_only diagnostic -- there is no "
            "pretrained backbone to freeze, it starts from random init"
        )
    set_seed(seed)
    fresh_backbone = BK.build_backbone(**backbone_cfg)
    fresh_head = build_head(head_name, feat_dim=fresh_backbone.feat_dim,
                            num_classes=num_classes, **head_kwargs)
    return TR.train_model(fresh_backbone, fresh_head, retain_loader, device,
                          log=log, **train_cfg)


def finetune(backbone, head, retain_loader: DataLoader, device: str,
             epochs: int = 5, lr: float = 0.01, weight_decay: float = 5e-4,
             classifier_only: bool = False, log: Optional[Callable] = None,
             epoch_eval: Optional[Callable] = None, **_):
    """Fine-tune on the retain set only. Forgetting happens by omission."""
    backbone, head = _clone(backbone, head)
    backbone.to(device); head.to(device)
    opt = _optimizer(_params(backbone, head, classifier_only), lr, weight_decay)

    if epoch_eval:
        epoch_eval(backbone, head, 0)

    for ep in range(epochs):
        if not classifier_only:
            backbone.train()
        head.train()
        tot = 0.0
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            loss = _step(backbone, head, x, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item()
        if log:
            log(f"    finetune ep {ep+1}/{epochs}  loss {tot/max(len(retain_loader),1):.4f}")
        if epoch_eval:
            epoch_eval(backbone, head, ep + 1)
    return backbone, head


def neggrad(backbone, head, forget_loader: DataLoader, device: str,
            epochs: int = 1, lr: float = 1e-4, weight_decay: float = 0.0,
            classifier_only: bool = False, log: Optional[Callable] = None,
            epoch_eval: Optional[Callable] = None, **_):
    """
    Pure gradient ascent on the forget set.

    Unbounded by construction -- the loss can grow without limit, so this
    will destroy the model if run too long. That instability is exactly what
    the AISTATS analysis is about, so keep lr small and epochs at 1.
    """
    backbone, head = _clone(backbone, head)
    backbone.to(device); head.to(device)
    opt = _optimizer(_params(backbone, head, classifier_only), lr, weight_decay)

    if epoch_eval:
        epoch_eval(backbone, head, 0)

    for ep in range(epochs):
        if not classifier_only:
            backbone.train()
        head.train()
        tot = 0.0
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            loss = _step(backbone, head, x, y, sign=-1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item()
        if log:
            log(f"    neggrad ep {ep+1}/{epochs}  loss {tot/max(len(forget_loader),1):.4f}")
        if epoch_eval:
            epoch_eval(backbone, head, ep + 1)
    return backbone, head


def neggrad_plus(backbone, head, forget_loader: DataLoader, retain_loader: DataLoader,
                 device: str, epochs: int = 3, lr: float = 1e-3,
                 weight_decay: float = 5e-4, alpha: float = 1.0,
                 classifier_only: bool = False, log: Optional[Callable] = None,
                 epoch_eval: Optional[Callable] = None, **_):
    """
    Ascent on forget, descent on retain, interleaved.

    `alpha` weights the ascent term. The retain term is what keeps the model
    usable; without it this degenerates to plain neggrad.
    """
    backbone, head = _clone(backbone, head)
    backbone.to(device); head.to(device)
    opt = _optimizer(_params(backbone, head, classifier_only), lr, weight_decay)

    if epoch_eval:
        epoch_eval(backbone, head, 0)

    for ep in range(epochs):
        if not classifier_only:
            backbone.train()
        head.train()
        f_iter = iter(forget_loader)
        tot, n = 0.0, 0
        for xr, yr in retain_loader:
            try:
                xf, yf = next(f_iter)
            except StopIteration:
                f_iter = iter(forget_loader)
                xf, yf = next(f_iter)

            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            loss = _step(backbone, head, xr, yr) - alpha * _step(backbone, head, xf, yf)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        if log:
            log(f"    neggrad+ ep {ep+1}/{epochs}  loss {tot/max(n,1):.4f}")
        if epoch_eval:
            epoch_eval(backbone, head, ep + 1)
    return backbone, head


def random_label(backbone, head, forget_loader: DataLoader, retain_loader: DataLoader,
                 device: str, num_classes: int, epochs: int = 3, lr: float = 1e-3,
                 weight_decay: float = 5e-4, classifier_only: bool = False,
                 exclude_true: bool = True, log: Optional[Callable] = None,
                 epoch_eval: Optional[Callable] = None,
                 trace: Optional[Callable] = None,
                 forget_active_steps_per_epoch: Optional[int] = None,
                 forget_loss_weight: float = 1.0,
                 dose_trace: Optional[Callable] = None, **_):
    """
    Relabel forget samples uniformly at random, then fine-tune normally.

    In the AISTATS results this was one of the two methods that left the
    representation essentially untouched (probe 92.49 vs retrain 77.35),
    so it is an important condition to include.

    EXPOSURE -- quantify this before comparing movement across datasets
    ------------------------------------------------------------------
    One step per RETAIN batch, and each step pulls one forget batch, cycling
    the forget loader whenever it runs out. So the forget set is replayed
    `ceil(n_retain/bs) / ceil(n_forget/bs)` times per epoch, and a small
    forget set is replayed far more often than a large one. At batch 128:

        CIFAR-10   45,000 retain / 5,000 forget -> 352 steps, 40 batches per
                   pass, 8 restarts, 44,096 forget presentations per epoch,
                   8.82 per unique forget image
        faces-1000 38,775 retain /    40 forget -> 303 steps,  1 batch per
                   pass, 302 restarts, 12,120 forget presentations per epoch,
                   303.00 per unique forget image

    That is a 34.4x difference in gradient exposure per unique forget image.
    Any cross-dataset comparison of feature movement is confounded by it
    unless it is stated. `scripts/feature_movement.py` records the measured
    counts in `result.json` rather than relying on this docstring.

    DOSE CONTROL -- optional, off by default
    ---------------------------------------
    `forget_active_steps_per_epoch` (m) and `forget_loss_weight` (lambda) turn
    the exposure described above into something that can be set rather than
    inherited from the split sizes. With m given, only m of the S retain steps
    carry the forget term, chosen by `evenly_spaced_active_steps`; on the other
    S-m steps the objective is retain CE alone. On an active step the objective
    is `retain_CE + lambda * forget_CE`.

    Two things are deliberate and must not be "tidied":

      * The candidate forget batch is still pulled, and its random targets are
        still drawn, on EVERY step including inactive ones. That keeps the RNG
        stream and the candidate sample stream identical to an uncontrolled
        run, so `training_trace_sha256` stays comparable across doses.
      * Inactive forget images are NOT forwarded through the backbone. A
        zero-weight forward would contribute no gradient but would still
        update BatchNorm running statistics, so the model would keep seeing
        the forget class. That is not zero active exposure, and the point of
        the control is zero active exposure.

    Defaults (m=None, lambda=1.0) take the original all-active, unit-weight
    expression unchanged -- `src/test_dose_schedule.py` pins that bit-for-bit
    against a copy of the pre-control loop.

    `trace`, when given, is called once per step as
    `trace(epoch, step, retain_idx, forget_idx, random_targets, n_retain, n_forget)`
    with CPU numpy arrays. Indices are None unless the loaders are built over
    `data.IndexedDataset`. Tracing is READ-ONLY: it draws no random numbers,
    touches no parameter and no optimiser state, and is invoked after the
    step's random targets already exist, so enabling it cannot change the run.
    `src/test_unlearn_trace.py` proves that by comparing parameters bitwise.

    `dose_trace` is the same contract for the dose schedule, called once per
    step as
    `dose_trace(epoch, step, active, weight, forget_idx, random_targets, n_forget)`
    where `weight` is the coefficient actually applied (lambda on an active
    step, 0.0 on an inactive one). It is additive: it records what the
    schedule did without altering it.
    """
    lam = float(forget_loss_weight)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(
            f"forget_loss_weight must be finite and strictly positive, got "
            f"{forget_loss_weight!r}. Zero would be a no-op dressed as an "
            f"intervention; use forget_active_steps_per_epoch to reduce dose."
        )
    n_steps = len(retain_loader)
    active_flags = (None if forget_active_steps_per_epoch is None
                    else evenly_spaced_active_steps(
                        n_steps, forget_active_steps_per_epoch))
    if log and (active_flags is not None or lam != 1.0):
        log(f"    random_label dose: "
            f"{n_steps if active_flags is None else sum(active_flags)}"
            f"/{n_steps} active forget steps per epoch, weight {lam:.10g}")

    backbone, head = _clone(backbone, head)
    backbone.to(device); head.to(device)
    opt = _optimizer(_params(backbone, head, classifier_only), lr, weight_decay)

    if epoch_eval:
        epoch_eval(backbone, head, 0)

    for ep in range(epochs):
        if not classifier_only:
            backbone.train()
        head.train()
        f_iter = iter(forget_loader)
        tot, n = 0.0, 0
        for step, rbatch in enumerate(retain_loader):
            xr, yr, ir = _unpack(rbatch)
            try:
                xf, yf, if_ = _unpack(next(f_iter))
            except StopIteration:
                f_iter = iter(forget_loader)
                xf, yf, if_ = _unpack(next(f_iter))

            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            rnd = torch.randint(0, num_classes, yf.shape, device=device)
            if exclude_true:
                # resample collisions so the "random" label is never the truth
                clash = rnd == yf
                while clash.any():
                    rnd[clash] = torch.randint(0, num_classes, (int(clash.sum()),), device=device)
                    clash = rnd == yf

            if active_flags is not None and step >= n_steps:
                raise RuntimeError(
                    f"retain loader yielded more than len()=={n_steps} "
                    f"batches; the dose schedule was built for {n_steps} "
                    f"steps and can no longer say which are active"
                )
            active = True if active_flags is None else active_flags[step]

            if trace is not None or dose_trace is not None:
                rnd_np = rnd.cpu().numpy()
                if_np = None if if_ is None else if_.cpu().numpy()
            if trace is not None:
                trace(ep, step,
                      None if ir is None else ir.cpu().numpy(),
                      if_np, rnd_np, int(yr.numel()), int(yf.numel()))
            if dose_trace is not None:
                dose_trace(ep, step, bool(active), lam if active else 0.0,
                           if_np, rnd_np, int(yf.numel()))

            # Three explicit branches rather than one weighted expression. The
            # default branch is the original line, character for character, so
            # an uncontrolled run is bitwise unchanged; the inactive branch
            # never touches xf, so no BatchNorm statistic sees the forget class
            # on a step where its loss coefficient is zero.
            r_loss = _step(backbone, head, xr, yr)
            if not active:
                loss = r_loss
            elif lam == 1.0:
                loss = r_loss + _step(backbone, head, xf, rnd)
            else:
                loss = r_loss + lam * _step(backbone, head, xf, rnd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        if log:
            log(f"    random_label ep {ep+1}/{epochs}  loss {tot/max(n,1):.4f}")
        if epoch_eval:
            epoch_eval(backbone, head, ep + 1)
    return backbone, head


METHODS: Dict[str, Callable] = {
    "retrain": retrain,
    "finetune": finetune,
    "neggrad": neggrad,
    "neggrad_plus": neggrad_plus,
    "random_label": random_label,
}


def run_unlearning(name: str, backbone, head, device: str, **kwargs):
    """
    Dispatch. Pass `classifier_only=True` to run the diagnostic version of
    any method -- backbone frozen, head updated. Comparing the two is the
    core measurement of this project.
    """
    key = name.lower()
    if key not in METHODS:
        raise ValueError(f"unknown method '{name}'; have {sorted(METHODS)}")
    return METHODS[key](backbone=backbone, head=head, device=device, **kwargs)
