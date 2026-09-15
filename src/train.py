"""
Training and evaluation.

`train_model`  standard supervised training, head-agnostic.
`extract`      features + output accuracy on a loader (no gradients).
`evaluate`     all four measurement levels + NC geometry -> a metrics.Report.

`evaluate` is where every number in the results table comes from, so read it
before trusting anything it produces.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import metrics as M


# ----------------------------------------------------------------------

def train_model(backbone: nn.Module, head: nn.Module, train_loader: DataLoader,
                device: str, epochs: int = 30, lr: float = 0.1,
                weight_decay: float = 5e-4, optimizer: str = "sgd",
                scheduler: str = "cosine", warmup_epochs: int = 0,
                log: Optional[Callable] = None) -> Tuple[nn.Module, nn.Module]:
    backbone.to(device); head.to(device)
    params = list(backbone.parameters()) + list(head.parameters())

    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)

    # T_max must match the number of times sched.step() actually fires below
    # (epochs - warmup_epochs, since warmup epochs don't call it), not the
    # raw epoch count -- otherwise the cosine cycle never completes and the
    # final LR lands well above 0. Every warmup_epochs>0 run before this fix
    # (every ArcFace run in the project so far) trained with a truncated
    # schedule: CIFAR ArcFace (epochs=30, warmup=5) ended at lr=0.00670, not
    # 0.00000, and faces ArcFace (epochs=40, warmup=5) ended at lr=0.00381.
    # See notes/decisions.md, 2026-09-14.
    cosine_epochs = epochs - warmup_epochs if warmup_epochs else epochs
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_epochs)
             if scheduler == "cosine" else None)

    for ep in range(epochs):
        # Linear LR warmup. Normalised heads (ArcFace/CosFace) frequently
        # diverge in the first epochs without it -- if ArcFace will not train,
        # try warmup_epochs=5 before concluding anything about the loss.
        if warmup_epochs and ep < warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / warmup_epochs

        backbone.train(); head.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = head(backbone(x), y)          # labels -> margin applied
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

        if sched and (not warmup_epochs or ep >= warmup_epochs):
            sched.step()

        if log:
            log(f"  ep {ep+1:3d}/{epochs}  loss {loss_sum/max(total,1):.4f}  "
                f"acc {correct/max(total,1):.4f}  lr {opt.param_groups[0]['lr']:.5f}  "
                f"({time.time()-t0:.0f}s)")

    return backbone, head


@torch.no_grad()
def extract(backbone: nn.Module, head: nn.Module, loader: DataLoader,
            device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features, labels, predictions). No margin -- head is called
    without labels, so ArcFace returns plain scaled cosine."""
    backbone.eval(); head.eval()
    feats, labs, preds = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = backbone(x)
        p = head(f).argmax(1)
        feats.append(f.cpu().numpy())
        labs.append(y.numpy())
        preds.append(p.cpu().numpy())
    return np.concatenate(feats), np.concatenate(labs), np.concatenate(preds)


@torch.no_grad()
def extract_with_indices(backbone: nn.Module, head: nn.Module,
                         loader: DataLoader, device: str
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`extract`, plus the sample indices the loader ACTUALLY yielded.

    Requires a loader over `data.IndexedDataset`, whose batches are
    `(x, y, i)`. Returns (features, labels, predictions, indices).

    The point is the fourth return value. A paired per-sample measurement is
    only valid if both sides saw the same samples in the same order, and the
    only way to know that is to record what came back rather than what was
    asked for. Everything else here is `extract` unchanged -- the head is
    still called without labels, so no margin is applied.
    """
    backbone.eval(); head.eval()
    feats, labs, preds, idxs = [], [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True)
        f = backbone(x)
        p = head(f).argmax(1)
        feats.append(f.cpu().numpy())
        labs.append(y.numpy())
        preds.append(p.cpu().numpy())
        idxs.append(np.asarray(i))
    return (np.concatenate(feats), np.concatenate(labs),
            np.concatenate(preds), np.concatenate(idxs))


def output_accuracy(labels: np.ndarray, preds: np.ndarray,
                    target_class: Optional[int]) -> Dict[str, float]:
    out = {"overall": float((preds == labels).mean())}
    if target_class is not None:
        m = labels == target_class
        out["forget"] = float((preds[m] == target_class).mean()) if m.any() else float("nan")
        out["retain"] = float((preds[~m] == labels[~m]).mean()) if (~m).any() else float("nan")
    return out


def evaluate_light(backbone: nn.Module, head: nn.Module,
                   train_eval_loader: DataLoader, test_loader: DataLoader,
                   num_classes: int, device: str, forget_class: int,
                   seed: int = 0) -> Dict:
    """
    A cheap subset of `evaluate()`, for tracking one point on an unlearning
    TRAJECTORY (measured every epoch) rather than only the end-of-run row.

    Two heads reaching output_forget=0 at very different epoch counts (see
    notes/decisions.md, 2026-09-13) means a comparison table built at a
    fixed epoch count is comparing methods at different points on their own
    trajectories. Recording a point every epoch lets a later analysis line
    heads up at matched output_forget instead of matched epoch count.

    Only the quantities needed for that: output_forget, output_retain,
    probe_forget, nc3_centred_forget, nc1_angular. Skips ncc, verif_auc,
    nc2, raw nc1, and nc3 uncentred -- those stay end-of-run only (via
    `evaluate`), because running a full evaluation (in particular fitting a
    fresh linear probe) at every epoch of every method would make the
    unlearning phase cost several times what it costs today.

    nc3 here is CENTRED, excluding the forget class from the centring
    reference -- same convention as `evaluate`, not the uncentred one
    `--compare` prints (see notes/decisions.md, centring contamination).
    """
    f_tr, y_tr, p_tr = extract(backbone, head, train_eval_loader, device)
    f_te, y_te, p_te = extract(backbone, head, test_loader, device)

    out_acc = output_accuracy(y_te, p_te, forget_class)
    probe = M.linear_probe(f_tr, y_tr, f_te, y_te, target_class=forget_class, seed=seed)
    W = head.weight.detach().cpu().numpy()
    nc3_c = M.nc3_alignment(f_tr, y_tr, W, num_classes, centre=True,
                            exclude_from_centre=forget_class)

    return {
        "output_forget": out_acc.get("forget", float("nan")),
        "output_retain": out_acc.get("retain", float("nan")),
        "probe_forget": probe.get("forget", float("nan")),
        "nc3_centred_forget": nc3_c.get("forget", float("nan")),
        "nc1_angular": M.nc1_angular(f_tr, y_tr, num_classes),
    }


def evaluate(backbone: nn.Module, head: nn.Module,
             train_eval_loader: DataLoader, test_loader: DataLoader,
             num_classes: int, device: str,
             forget_class: Optional[int] = None,
             head_name: str = "", method_name: str = "", seed: int = 0,
             forget_heldout_loader: Optional[DataLoader] = None) -> Dict:
    """
    Produce one row of the results table.

    Order of operations matters: features come from the EVAL-transform copy
    of the training set, so no random crop or flip contaminates the geometry.

    NC3 is reported BOTH centred and uncentred. They disagree -- under an
    exact weight flip the uncentred cosine is -1.00 and the centred one is
    -0.71 -- because centring subtracts a global mean that is not part of the
    theoretical claim. Reporting one without saying which invites a reviewer
    to recompute it the other way and think you erred.

    NC1 is likewise reported BOTH raw and angular (L2-normalised features).
    Raw NC1 is sensitive to feature magnitude; angular NC1 is not. Margin
    heads normalise features before comparing them, so magnitude changes
    that raw NC1 would read as "collapse" are invisible to what those heads
    actually optimise. Report both.

    When a forget class exists, the centred version excludes it from the
    centring reference. Without that, one flipped weight drags every retained
    class's score down by several points and it looks like real spillover.
    """
    f_tr, y_tr, p_tr = extract(backbone, head, train_eval_loader, device)
    f_te, y_te, p_te = extract(backbone, head, test_loader, device)

    out_acc = output_accuracy(y_te, p_te, forget_class)
    probe = M.linear_probe(f_tr, y_tr, f_te, y_te, target_class=forget_class, seed=seed)
    ncc = M.ncc_accuracy(f_tr, y_tr, f_te, y_te, num_classes, target_class=forget_class)
    verif = M.verification_auc(f_te, y_te, target_class=forget_class, seed=seed)

    W = head.weight.detach().cpu().numpy()
    nc3_c = M.nc3_alignment(f_tr, y_tr, W, num_classes, centre=True,
                            exclude_from_centre=forget_class)
    nc3_u = M.nc3_alignment(f_tr, y_tr, W, num_classes, centre=False)

    row = {
        "head": head_name,
        "method": method_name,
        "forget_class": forget_class,
        "seed": seed,
        "output_overall": out_acc["overall"],
        "output_forget": out_acc.get("forget", float("nan")),
        "output_retain": out_acc.get("retain", float("nan")),
        "probe_overall": probe["overall"],
        "probe_forget": probe.get("forget", float("nan")),
        "probe_retain": probe.get("retain", float("nan")),
        "ncc_overall": ncc["overall"],
        "ncc_forget": ncc.get("forget", float("nan")),
        "ncc_retain": ncc.get("retain", float("nan")),
        "verif_auc": verif["auc"],
        "verif_auc_forget": verif.get("auc_forget", float("nan")),
        "nc1": M.nc1_within_class_variability(f_tr, y_tr, num_classes),
        "nc1_angular": M.nc1_angular(f_tr, y_tr, num_classes),
        "nc2": M.nc2_simplex_etf(f_tr, y_tr, num_classes),
        "nc3_centred_mean": nc3_c["mean"],
        "nc3_uncentred_mean": nc3_u["mean"],
    }

    if forget_class is not None:
        row["nc3_centred_forget"] = nc3_c.get("forget", float("nan"))
        row["nc3_centred_retain_mean"] = nc3_c.get("retain_mean", float("nan"))
        row["nc3_uncentred_forget"] = float(nc3_u["per_class"][forget_class])
        keep = np.delete(nc3_u["per_class"], forget_class)
        keep = keep[~np.isnan(keep)]
        row["nc3_uncentred_retain_mean"] = float(keep.mean()) if keep.size else float("nan")

    # Forget-generalisation: same identity, TRAINING images the unlearning
    # method never saw. Distinct from the test set -- the original model
    # trained on these, so they measure whether forgetting spread within the
    # identity, not how the model generalises to unseen data.
    if forget_heldout_loader is not None and forget_class is not None:
        f_ho, y_ho, p_ho = extract(backbone, head, forget_heldout_loader, device)
        if len(y_ho):
            row["forget_heldout_output"] = float((p_ho == forget_class).mean())
            ho_probe = M.linear_probe(f_tr, y_tr, f_ho, y_ho,
                                      target_class=forget_class, seed=seed)
            row["forget_heldout_probe"] = ho_probe.get("forget", float("nan"))

    return row
