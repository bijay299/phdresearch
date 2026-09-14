#!/usr/bin/env python
"""
Unlearning-method trajectory check across forget classes.

At forget_class=0, two conditions showed a real head difference (see
notes/decisions.md, 2026-09-13/14): CE `finetune` never drove output_forget
to 0 even at 30 epochs, while ArcFace did within a few; and
`random_label_clfonly` (backbone frozen, so any nc3 change is the classifier
alone) showed CE's classifier staying aligned with the class mean while
ArcFace's flipped negative, with retain accuracy healthy in both heads. This
reruns both conditions at forget_class in {1,2,3} to check whether that
pattern holds or whether fc0 was a fluke (see notes/decisions.md, 2026-09-14:
it holds, at every class, no exceptions).

Starts from the ALREADY-TRAINED checkpoint (`<out_dir>/<name>/ckpt.pt`,
written by `scripts/run_experiment.py`) instead of retraining the 30-epoch
original model from scratch per forget class -- that part of training
doesn't depend on forget_class at all, so redoing it every time would be
pure waste (same reasoning as `scripts/retrain_stability.py`). Still writes
every result through `RunDir`, one fresh directory per (method,
forget_class), so `RunDir`'s refuse-on-reuse check (`src/utils.py`,
2026-09-14) is satisfied by construction rather than bypassed.

Two conditions only -- the two that showed a real difference at fc0:
    finetune              full backbone, `--finetune-epochs` (default 30),
                           logged every `--finetune-trajectory-every` (default 5)
    random_label_clfonly  frozen backbone, the base config's own
                           `unlearn.epochs` (3), logged every epoch

    python scripts/unlearn_across_classes.py --config configs/cifar_ce.yaml \
        --forget-classes 1,2,3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as D                      # noqa: E402
import train as TR                    # noqa: E402
import unlearn as UL                  # noqa: E402
from backbones import build_backbone  # noqa: E402
from heads import build_head          # noqa: E402
from utils import RunDir, apply_overrides, banner, load_config, set_seed  # noqa: E402


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def load_checkpoint_model(cfg: dict, ckpt_path: Path, num_classes: int):
    bb_cfg = dict(cfg["backbone"])
    backbone = build_backbone(**bb_cfg)
    head_cfg = dict(cfg["head"]); head_name = head_cfg.pop("name")
    head = build_head(head_name, feat_dim=backbone.feat_dim,
                      num_classes=num_classes, **head_cfg)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path} -- run scripts/run_experiment.py "
            f"for this config first (it writes ckpt.pt after the original "
            f"30-epoch training)."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    return backbone, head, head_name


def run_condition(cfg: dict, head_name: str, backbone0, head0, num_classes: int,
                  train_ds, train_eval_loader, test_loader, device: str,
                  forget_class: int, method: str, classifier_only: bool,
                  epochs: int, lr: float, weight_decay: float,
                  trajectory_every: int, bs: int, nw: int, out_dir: str) -> dict:
    """
    One (method, forget_class) condition, starting from the shared
    checkpoint. `finetune`/`random_label` clone their inputs internally
    (see src/unlearn.py), so passing the same `backbone0`/`head0` into
    multiple calls never mutates the shared starting point.
    """
    targets = D.get_targets(train_ds)
    split = D.make_forget_split(targets, forget_class, "all", 0.5, cfg["seed"])
    forget_loader = D.make_loader(train_ds, split.forget_idx, bs, True, nw)
    retain_loader = D.make_loader(train_ds, split.retain_idx, bs, True, nw)

    tag = f"{method}{'_clfonly' if classifier_only else ''}"
    name = f"{cfg['data']['name']}_{head_name}_seed{cfg['seed']}_{tag}_fc{forget_class}"
    run_cfg = dict(cfg, _unlearn_across_classes=True, _method=method,
                   _classifier_only=classifier_only, _forget_class=forget_class,
                   _epochs=epochs, _trajectory_every=trajectory_every)
    rd = RunDir.create(out_dir, name, run_cfg)
    rd.log(banner(name))
    rd.log(f"  {split.summary()}")

    def epoch_eval(bb, hd, epoch):
        if epoch not in (0, epochs) and epoch % trajectory_every != 0:
            return
        point = TR.evaluate_light(bb, hd, train_eval_loader, test_loader,
                                  num_classes, device,
                                  forget_class=forget_class, seed=cfg["seed"])
        rd.append_jsonl("trajectory.jsonl", {
            "head": head_name, "method": tag, "forget_class": forget_class,
            "seed": cfg["seed"], "epoch": epoch, **point,
        })

    kwargs = dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                 classifier_only=classifier_only, log=rd.log,
                 epoch_eval=epoch_eval)
    if method == "finetune":
        kwargs["retain_loader"] = retain_loader
    elif method == "random_label":
        kwargs["forget_loader"] = forget_loader
        kwargs["retain_loader"] = retain_loader
        kwargs["num_classes"] = num_classes

    t0 = time.time()
    bb_u, hd_u = UL.run_unlearning(method, backbone0, head0, device, **kwargs)
    method_time_s = time.time() - t0

    t0 = time.time()
    row = TR.evaluate(bb_u, hd_u, train_eval_loader, test_loader, num_classes,
                      device, forget_class=forget_class, head_name=head_name,
                      method_name=tag, seed=cfg["seed"])
    eval_time_s = time.time() - t0
    row["method_time_s"] = method_time_s
    row["eval_time_s"] = eval_time_s
    rd.append_jsonl("results.jsonl", row)

    rd.log(f"  method_time {method_time_s:.1f}s  eval_time {eval_time_s:.1f}s")
    rd.log(f"  output_forget {row['output_forget']:.4f}  "
          f"output_retain {row['output_retain']:.4f}")
    rd.log(f"run complete -> {rd.root}")
    return row


def main(cfg: dict, forget_classes: list[int], ckpt_path: Path,
        finetune_epochs: int, finetune_trajectory_every: int,
        conditions: list[str] = ("finetune", "random_label_clfonly")) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])   # once, up front -- matches run_experiment.py's
                            # own convention: RNG evolves naturally across
                            # the conditions that follow, not reset per one

    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(cfg["data"])
    bs, nw = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    train_eval_loader = D.make_loader(train_eval_ds, None, 512, False, nw)
    test_loader = D.make_loader(test_ds, None, 512, False, nw)

    backbone0, head0, head_name = load_checkpoint_model(cfg, ckpt_path, num_classes)
    u = cfg["unlearn"]

    for fc in forget_classes:
        if "finetune" in conditions:
            run_condition(cfg, head_name, backbone0, head0, num_classes, train_ds,
                          train_eval_loader, test_loader, device, fc,
                          method="finetune", classifier_only=False,
                          epochs=finetune_epochs, lr=u["lr"],
                          weight_decay=u["weight_decay"],
                          trajectory_every=finetune_trajectory_every,
                          bs=bs, nw=nw, out_dir=cfg["out_dir"])

        if "random_label_clfonly" in conditions:
            run_condition(cfg, head_name, backbone0, head0, num_classes, train_ds,
                          train_eval_loader, test_loader, device, fc,
                          method="random_label", classifier_only=True,
                          epochs=u["epochs"], lr=u["lr"],
                          weight_decay=u["weight_decay"], trajectory_every=1,
                          bs=bs, nw=nw, out_dir=cfg["out_dir"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget-classes", required=True,
                    help="comma-separated class indices, e.g. 1,2,3")
    ap.add_argument("--ckpt", default=None,
                    help="path to the checkpoint to start from; defaults to "
                         "<out_dir>/<data>_<head>_seed<seed>/ckpt.pt, i.e. "
                         "whatever scripts/run_experiment.py already trained "
                         "for this config")
    ap.add_argument("--finetune-epochs", type=int, default=30,
                    help="epoch budget for the full-backbone finetune "
                         "condition (default 30 -- 3 wasn't enough to see "
                         "whether CE converges, see notes/decisions.md 2026-09-13)")
    ap.add_argument("--finetune-trajectory-every", type=int, default=5)
    ap.add_argument("--conditions", default="finetune,random_label_clfonly",
                    help="comma-separated subset of {finetune,"
                         "random_label_clfonly} to run (default: both)")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL")
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), a.set)
    classes = [int(c) for c in a.forget_classes.split(",")]
    conditions = a.conditions.split(",")
    unknown = set(conditions) - {"finetune", "random_label_clfonly"}
    if unknown:
        raise ValueError(f"unknown --conditions {sorted(unknown)}; "
                         f"expected a subset of finetune,random_label_clfonly")

    if a.ckpt is not None:
        ckpt_path = Path(a.ckpt)
    else:
        run_name = f"{cfg['data']['name']}_{cfg['head']['name']}_seed{cfg['seed']}"
        ckpt_path = Path(cfg["out_dir"]) / run_name / "ckpt.pt"

    main(cfg, classes, ckpt_path, a.finetune_epochs, a.finetune_trajectory_every,
        conditions=conditions)
