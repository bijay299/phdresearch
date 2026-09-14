#!/usr/bin/env python
"""
Main entry point.

    # smoke test -- 1 epoch, tiny subset, runs on CPU in a couple of minutes
    python scripts/run_experiment.py --config configs/cifar_ce.yaml --smoke

    # the pilot pair
    python scripts/run_experiment.py --config configs/cifar_ce.yaml
    python scripts/run_experiment.py --config configs/cifar_arcface.yaml

    # compare completed runs
    python scripts/run_experiment.py --compare logs/

    # override anything from the CLI
    python scripts/run_experiment.py --config configs/cifar_ce.yaml \
        --set train.epochs=5 unlearn.enabled=true

Each run writes a directory under logs/ containing config.json, env.json
(git commit, device, argv), run.log, results.jsonl and the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data as D                     # noqa: E402
import train as TR                   # noqa: E402
import unlearn as UL                 # noqa: E402
from backbones import build_backbone  # noqa: E402
from heads import build_head          # noqa: E402
from utils import (RunDir, apply_overrides, banner, load_config,  # noqa: E402
                   set_seed)


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def run(cfg: dict, smoke: bool = False) -> None:
    if smoke:
        cfg["train"]["epochs"] = 1
        cfg["data"]["batch_size"] = 64
        cfg["data"]["num_workers"] = 0
        cfg["unlearn"]["epochs"] = 1
        cfg.setdefault("_smoke", True)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))

    name = f"{cfg['data']['name']}_{cfg['head']['name']}_seed{cfg['seed']}"
    if smoke:
        name = "smoke_" + name
    rd = RunDir.create(cfg["out_dir"], name, cfg)

    rd.log(banner(f"RUN  {name}"))
    rd.log(f"device: {device}")

    # ---- data -------------------------------------------------------
    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(cfg["data"],
                                                                     seed=cfg["seed"],
                                                                     log=rd.log)
    targets = D.get_targets(train_ds)
    rd.log(f"classes: {num_classes}   train: {len(train_ds)}   test: {len(test_ds)}")

    train_idx = test_idx = None
    if smoke:
        train_idx = D.stratified_subsample(targets, per_class=40, seed=cfg["seed"])
        # The test side needs subsampling too: evaluate() runs a full backbone
        # forward pass over this loader once per condition, so leaving it at the
        # full test set costs more than every other part of a smoke run combined.
        test_idx = D.stratified_subsample(D.get_targets(test_ds), per_class=40,
                                          seed=cfg["seed"])
        rd.log(f"SMOKE: subsampled train to {len(train_idx)}, "
               f"test to {len(test_idx)}")

    bs, nw = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    train_loader = D.make_loader(train_ds, train_idx, bs, True, nw, drop_last=True)
    train_eval_loader = D.make_loader(train_eval_ds, train_idx, 512, False, nw)
    test_loader = D.make_loader(test_ds, test_idx, 512, False, nw)

    # ---- model ------------------------------------------------------
    bb_cfg = dict(cfg["backbone"])
    backbone = build_backbone(**bb_cfg)
    head_cfg = dict(cfg["head"]); head_name = head_cfg.pop("name")
    head = build_head(head_name, feat_dim=backbone.feat_dim,
                      num_classes=num_classes, **head_cfg)
    rd.log(f"backbone: {bb_cfg['arch']} (feat_dim={backbone.feat_dim})   head: {head_name}")

    # ---- train ------------------------------------------------------
    rd.log(banner("TRAIN"))
    backbone, head = TR.train_model(
        backbone, head, train_loader, device, log=rd.log, **cfg["train"])

    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()},
               rd.path("ckpt.pt"))

    # ---- baseline geometry, before any unlearning -------------------
    rd.log(banner("BASELINE GEOMETRY"))
    t0 = time.time()
    base = TR.evaluate(backbone, head, train_eval_loader, test_loader,
                       num_classes, device, forget_class=None,
                       head_name=head_name, method_name="original",
                       seed=cfg["seed"])
    base["eval_time_s"] = time.time() - t0
    rd.append_jsonl("results.jsonl", base)
    for k in ["output_overall", "probe_overall", "ncc_overall", "verif_auc",
              "nc1", "nc2", "nc3_centred_mean", "nc3_uncentred_mean"]:
        rd.log(f"  {k:24s} {base[k]:.4f}")
    rd.log(f"  eval_time                {base['eval_time_s']:.1f}s")

    rd.log("\n  WEEK-2 GATE: compare nc3 and nc1 across heads.")
    rd.log("  Higher nc3 under arcface => margin losses enforce alignment more")
    rd.log("  strongly, which is the premise the project rests on.")

    # ---- unlearning -------------------------------------------------
    if not cfg["unlearn"]["enabled"]:
        rd.log("\nunlearning disabled (unlearn.enabled=false)")
        rd.log(f"\nrun complete -> {rd.root}")
        return

    u = cfg["unlearn"]
    split = D.make_forget_split(targets, u["forget_class"], u["split_mode"],
                                u.get("forget_fraction", 0.5), cfg["seed"])

    if smoke and train_idx is not None:
        split = D.restrict_split(split, train_idx)
        rd.log(f"SMOKE: restricted split to subset -> {split.summary()}")

    rd.log(banner("UNLEARN"))
    rd.log(f"  {split.summary()}")

    forget_loader = D.make_loader(train_ds, split.forget_idx, bs, True, nw)
    retain_loader = D.make_loader(train_ds, split.retain_idx, bs, True, nw)
    forget_heldout_loader = (
        D.make_loader(train_eval_ds, split.forget_heldout_idx, 256, False, nw)
        if len(split.forget_heldout_idx) else None)

    variants = [False]
    if u.get("classifier_only_diagnostic", True):
        variants.append(True)

    traj_every = u.get("trajectory_every", 1)

    def make_epoch_eval(tag: str):
        """
        Trajectory point every `trajectory_every` epochs of unlearning, not
        just at the end -- see notes/decisions.md 2026-09-13: ArcFace
        finetune reaches output_forget=0 by epoch 3 while CE is still at
        0.662 after 30, so a table read at one fixed epoch count compares
        methods at different points on their own trajectories. Written to
        trajectory.jsonl so a later analysis can line heads up at matched
        output_forget instead.

        Epoch 0 (the shared pre-unlearning starting point) and the final
        configured epoch are always evaluated, regardless of the interval,
        so the curve never silently drops its endpoints.
        """
        def _epoch_eval(bb, hd, epoch):
            if epoch not in (0, u["epochs"]) and epoch % traj_every != 0:
                return
            point = TR.evaluate_light(bb, hd, train_eval_loader, test_loader,
                                      num_classes, device,
                                      forget_class=u["forget_class"],
                                      seed=cfg["seed"])
            rd.append_jsonl("trajectory.jsonl", {
                "head": head_name, "method": tag,
                "forget_class": u["forget_class"], "seed": cfg["seed"],
                "epoch": epoch, **point,
            })
        return _epoch_eval

    for method in u["methods"]:
        for clf_only in variants:
            tag = f"{method}{'_clfonly' if clf_only else ''}"
            rd.log(f"\n  --- {tag} ---")

            kwargs = dict(epochs=u["epochs"], lr=u["lr"],
                          weight_decay=u["weight_decay"],
                          classifier_only=clf_only, log=rd.log,
                          epoch_eval=make_epoch_eval(tag))
            if method in ("finetune",):
                kwargs["retain_loader"] = retain_loader
            elif method in ("neggrad",):
                kwargs["forget_loader"] = forget_loader
            else:
                kwargs["forget_loader"] = forget_loader
                kwargs["retain_loader"] = retain_loader
                if method == "random_label":
                    kwargs["num_classes"] = num_classes

            t0 = time.time()
            bb_u, hd_u = UL.run_unlearning(method, backbone, head, device, **kwargs)
            method_time_s = time.time() - t0

            t0 = time.time()
            row = TR.evaluate(bb_u, hd_u, train_eval_loader, test_loader,
                              num_classes, device,
                              forget_class=u["forget_class"],
                              head_name=head_name, method_name=tag,
                              seed=cfg["seed"],
                              forget_heldout_loader=forget_heldout_loader)
            eval_time_s = time.time() - t0
            row["method_time_s"] = method_time_s
            row["eval_time_s"] = eval_time_s
            rd.append_jsonl("results.jsonl", row)

            rd.log(f"    method_time {method_time_s:.1f}s   "
                   f"eval_time {eval_time_s:.1f}s   "
                   f"total {method_time_s + eval_time_s:.1f}s")
            rd.log(f"    output_forget {row['output_forget']:.4f}   "
                   f"probe_forget {row['probe_forget']:.4f}   "
                   f"ncc_forget {row['ncc_forget']:.4f}")
            rd.log(f"    nc3_forget(uncentred) {row['nc3_uncentred_forget']:+.4f}   "
                   f"retain {row['nc3_uncentred_retain_mean']:.4f}   "
                   f"verif_auc_forget {row['verif_auc_forget']:.4f}")

    # ---- retrain-from-scratch reference ------------------------------
    # Runs last: it reseeds the global RNG (fresh init + its own loader
    # shuffling), and every probe_forget number above needs this as the
    # comparison point, not the other way around.
    if u.get("retrain_reference", True):
        rd.log(banner("RETRAIN (reference)"))
        t0 = time.time()
        bb_r, hd_r = UL.run_unlearning(
            "retrain", backbone, head, device,
            retain_loader=retain_loader, backbone_cfg=bb_cfg,
            head_name=head_name, head_kwargs=head_cfg,
            num_classes=num_classes, train_cfg=dict(cfg["train"]),
            seed=cfg["seed"], log=rd.log)
        method_time_s = time.time() - t0

        t0 = time.time()
        row = TR.evaluate(bb_r, hd_r, train_eval_loader, test_loader,
                          num_classes, device,
                          forget_class=u["forget_class"],
                          head_name=head_name, method_name="retrain",
                          seed=cfg["seed"],
                              forget_heldout_loader=forget_heldout_loader)
        eval_time_s = time.time() - t0
        row["method_time_s"] = method_time_s
        row["eval_time_s"] = eval_time_s
        rd.append_jsonl("results.jsonl", row)
        rd.log(f"    method_time {method_time_s:.1f}s   "
               f"eval_time {eval_time_s:.1f}s   "
               f"total {method_time_s + eval_time_s:.1f}s")
        rd.log(f"    output_forget {row['output_forget']:.4f}   "
               f"probe_forget {row['probe_forget']:.4f}   "
               f"ncc_forget {row['ncc_forget']:.4f}")
    else:
        rd.log("\nretrain reference skipped (unlearn.retrain_reference=false)")

    rd.log(f"\nrun complete -> {rd.root}")


def compare(log_dir: str) -> None:
    rows = []
    for p in sorted(Path(log_dir).glob("*/results.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print(f"no results found under {log_dir}")
        return

    # probe_forget is not comparable across heads on its own -- each head has
    # its own retrain reference (see notes/decisions.md). Look one up per
    # (head, forget_class) so every row can carry its own gap.
    retrain_probe = {}
    for r in rows:
        if r.get("method") == "retrain":
            retrain_probe[(r.get("head"), r.get("forget_class"))] = r.get("probe_forget")

    GAP_COL = "probe_gap_to_retrain*"
    cols = ["head", "method", "forget_class", "output_forget", "probe_forget", GAP_COL,
            "ncc_forget", "verif_auc_forget", "nc3_uncentred_forget", "nc3_uncentred_retain_mean"]
    widths = {c: max(len(c), 12) for c in cols}

    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * len(cols)))
    for r in rows:
        cells = []
        for c in cols:
            if c == GAP_COL:
                rp = retrain_probe.get((r.get("head"), r.get("forget_class")))
                pf = r.get("probe_forget")
                v = (pf - rp) if (rp is not None and pf is not None and pf == pf) else float("nan")
            else:
                v = r.get(c, float("nan"))
            if isinstance(v, str):
                text = v
            elif isinstance(v, bool) or v is None:
                text = str(v)
            elif isinstance(v, int):
                text = str(v)
            else:
                text = f"{v:.4f}"
            cells.append(text.ljust(widths[c]))
        print("  ".join(cells))

    print("\nREMINDERS")
    print("  * probe_forget is meaningless alone -- compare against a retrain")
    print("    reference. In the AISTATS CIFAR-10 setting that reference was 77.35")
    print("    on a class it never saw, because deep features transfer.")
    print(f"  * {GAP_COL} = probe_forget - THIS ROW'S HEAD's own retrain probe_forget")
    print("    for the same forget_class. This is our arithmetic, not a metric from")
    print("    the paper -- CE and ArcFace retrain references differ (see")
    print("    notes/decisions.md), so raw probe_forget must never be compared")
    print("    directly across heads.")
    print("  * always read forget alongside retain. A model that forgets because")
    print("    it broke has forgotten nothing.")
    print("  * nc3 here is UNCENTRED. The centred version gives different numbers")
    print("    on identical data -- say which convention you used.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="path to a YAML config")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL",
                    help="override config values, e.g. train.epochs=5")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a tiny subset -- checks wiring, not results")
    ap.add_argument("--compare", metavar="LOG_DIR",
                    help="print a table of all completed runs")
    a = ap.parse_args()

    if a.compare:
        compare(a.compare)
    elif a.config:
        cfg = apply_overrides(load_config(a.config), a.set)
        run(cfg, smoke=a.smoke)
    else:
        ap.print_help()
