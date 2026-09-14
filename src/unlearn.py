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
from typing import Callable, Dict, Optional

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
                 epoch_eval: Optional[Callable] = None, **_):
    """
    Relabel forget samples uniformly at random, then fine-tune normally.

    In the AISTATS results this was one of the two methods that left the
    representation essentially untouched (probe 92.49 vs retrain 77.35),
    so it is an important condition to include.
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

            rnd = torch.randint(0, num_classes, yf.shape, device=device)
            if exclude_true:
                # resample collisions so the "random" label is never the truth
                clash = rnd == yf
                while clash.any():
                    rnd[clash] = torch.randint(0, num_classes, (int(clash.sum()),), device=device)
                    clash = rnd == yf

            loss = _step(backbone, head, xr, yr) + _step(backbone, head, xf, rnd)
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
