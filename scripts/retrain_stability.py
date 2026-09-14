#!/usr/bin/env python
"""
Retrain-reference stability check across forget classes.

The CIFAR-10 pilot found CE's retrain-reference probe_forget (0.8630) and
ArcFace's (0.5540) differ substantially, at forget_class=0 only. Since every
probe_gap_to_retrain number in the comparison table is measured against this
reference, a one-class fluke would silently distort every downstream number.
This script reruns just the retrain reference (fresh backbone+head trained
on retain-only) for additional forget classes, so the gap can be checked for
stability before being trusted.

Skips the full 30-epoch "original" training and the other unlearning
methods -- retrain doesn't depend on either -- but still writes through
RunDir, per the standing rule that every result must carry a config, seed,
git commit and device.

    python scripts/retrain_stability.py --config configs/cifar_ce.yaml \
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
from utils import RunDir, apply_overrides, banner, load_config, set_seed  # noqa: E402


def resolve_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def main(cfg: dict, forget_classes: list[int]) -> None:
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))

    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(cfg["data"])
    targets = D.get_targets(train_ds)

    bs, nw = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    train_eval_loader = D.make_loader(train_eval_ds, None, 512, False, nw)
    test_loader = D.make_loader(test_ds, None, 512, False, nw)

    for fc in forget_classes:
        name = f"{cfg['data']['name']}_{cfg['head']['name']}_seed{cfg['seed']}_retrain_fc{fc}"
        run_cfg = dict(cfg, _retrain_stability_check=True, _forget_class_checked=fc)
        rd = RunDir.create(cfg["out_dir"], name, run_cfg)
        rd.log(banner(f"RETRAIN STABILITY CHECK  {name}"))
        rd.log(f"device: {device}   forget_class: {fc}")

        split = D.make_forget_split(targets, fc, "all", 0.5, cfg["seed"])
        retain_loader = D.make_loader(train_ds, split.retain_idx, bs, True, nw)
        rd.log(f"  {split.summary()}")

        bb_cfg = dict(cfg["backbone"])
        head_cfg = dict(cfg["head"]); head_name = head_cfg.pop("name")

        t0 = time.time()
        bb_r, hd_r = UL.retrain(
            None, None, retain_loader, device,
            backbone_cfg=bb_cfg, head_name=head_name, head_kwargs=head_cfg,
            num_classes=num_classes, train_cfg=dict(cfg["train"]),
            seed=cfg["seed"], log=rd.log)
        method_time_s = time.time() - t0

        t0 = time.time()
        row = TR.evaluate(bb_r, hd_r, train_eval_loader, test_loader,
                          num_classes, device, forget_class=fc,
                          head_name=head_name, method_name="retrain",
                          seed=cfg["seed"])
        eval_time_s = time.time() - t0
        row["method_time_s"] = method_time_s
        row["eval_time_s"] = eval_time_s
        rd.append_jsonl("results.jsonl", row)

        rd.log(f"  probe_forget {row['probe_forget']:.4f}   "
              f"output_forget {row['output_forget']:.4f}   "
              f"output_retain {row['output_retain']:.4f}   "
              f"method_time {method_time_s:.1f}s")
        rd.log(f"run complete -> {rd.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget-classes", required=True,
                    help="comma-separated class indices, e.g. 1,2,3")
    ap.add_argument("--set", nargs="*", default=[], metavar="k.v=VAL")
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), a.set)
    classes = [int(c) for c in a.forget_classes.split(",")]
    main(cfg, classes)
