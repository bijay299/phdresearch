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
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                      # noqa: E402
import train as TR                    # noqa: E402
import unlearn as UL                  # noqa: E402
from backbones import build_backbone  # noqa: E402
from heads import build_head          # noqa: E402
from utils import RunDir, apply_overrides, banner, load_config, set_seed  # noqa: E402

# The dose arithmetic is already written and already tested in the movement
# driver. Importing it keeps ONE definition of predicted dose rather than a
# second one here that could drift from it.
from feature_movement import dose_accounting, sha256_file  # noqa: E402


def _sha_json(obj) -> str:
    """Stable hash of a manifest. sort_keys so dict order cannot change it."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity_name(provenance: dict | None, forget_class: int):
    """The dataset's own name for the forgotten class, or None for CIFAR.

    Recorded so a later reader can confirm that fc=29 means the same person
    at K=100 and at K=1000 rather than assuming it from the index.
    """
    names = (provenance or {}).get("identity_names") or []
    return names[forget_class] if 0 <= forget_class < len(names) else None


def git_provenance() -> dict:
    def _run(*args):
        try:
            return subprocess.run(args, cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    dirty_files = _run("git", "status", "--porcelain")
    return {
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files or "",
    }


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
                  trajectory_every: int, bs: int, nw: int, out_dir: str,
                  forget_active_steps_per_epoch: int | None = None,
                  forget_loss_weight: float = 1.0,
                  per_cell_seed: bool = False,
                  provenance: dict | None = None) -> dict:
    """
    One (method, forget_class) condition, starting from the shared
    checkpoint. `finetune`/`random_label` clone their inputs internally
    (see src/unlearn.py), so passing the same `backbone0`/`head0` into
    multiple calls never mutates the shared starting point.

    DOSE CONTROL AND PER-CELL SEEDING -- both default to OFF
    -------------------------------------------------------
    With `forget_active_steps_per_epoch=None`, `forget_loss_weight=1.0` and
    `per_cell_seed=False` this function computes exactly what it computed
    before those parameters existed: those are `src/unlearn.py`'s own
    defaults, and the one-reseed-up-front convention is what `main` always
    did. `src/test_class_sweep_dose.py` pins that.

    `per_cell_seed` reseeds immediately before the method, so a cell is
    reproducible on its own rather than depending on how many cells ran
    before it. That matters for a class-count sweep: without it the random
    labels a cell draws depend on execution order, and K=100 fc29 would not
    be comparable with K=1000 fc29. It is opt-in because turning it on
    changes the existing callers' numbers.
    """
    targets = D.get_targets(train_ds)
    split = D.make_forget_split(targets, forget_class, "all", 0.5, cfg["seed"])
    forget_loader = D.make_loader(train_ds, split.forget_idx, bs, True, nw)
    retain_loader = D.make_loader(train_ds, split.retain_idx, bs, True, nw)

    n_retain, n_forget = len(split.retain_idx), len(split.forget_idx)
    steps_per_epoch = -(-n_retain // bs)
    if forget_active_steps_per_epoch is not None:
        m = int(forget_active_steps_per_epoch)
        if not 1 <= m <= steps_per_epoch:
            raise ValueError(
                f"forget_active_steps_per_epoch={m} is invalid for this cell: "
                f"{n_retain} retain images at batch_size={bs} give "
                f"S={steps_per_epoch} retain steps per epoch, and the schedule "
                f"needs 1 <= m <= S. Reduce m or stop and report the "
                f"incompatibility -- do not silently change the dose."
            )

    tag = f"{method}{'_clfonly' if classifier_only else ''}"
    name = f"{cfg['data']['name']}_{head_name}_seed{cfg['seed']}_{tag}_fc{forget_class}"
    run_cfg = dict(cfg, _unlearn_across_classes=True, _method=method,
                   _classifier_only=classifier_only, _forget_class=forget_class,
                   _epochs=epochs, _trajectory_every=trajectory_every)
    rd = RunDir.create(out_dir, name, run_cfg)
    rd.log(banner(name))
    rd.log(f"  {split.summary()}")

    dose_controlled = (forget_active_steps_per_epoch is not None
                       or float(forget_loss_weight) != 1.0)

    def epoch_eval(bb, hd, epoch):
        if epoch not in (0, epochs) and epoch % trajectory_every != 0:
            return
        point = TR.evaluate_light(bb, hd, train_eval_loader, test_loader,
                                  num_classes, device,
                                  forget_class=forget_class, seed=cfg["seed"],
                                  include_uncentred=dose_controlled)
        rd.append_jsonl("trajectory.jsonl", {
            "head": head_name, "method": tag, "forget_class": forget_class,
            "seed": cfg["seed"], "epoch": epoch, **point,
        })

    # Candidate stream and dose schedule get SEPARATE hashes, same convention
    # as scripts/feature_movement.py: the candidate trace stays comparable
    # across doses, and what the schedule did with it is a different fact.
    # Both callbacks are read-only (src/test_unlearn_trace.py proves enabling
    # tracing leaves parameters bitwise identical).
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

    kwargs = dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                 classifier_only=classifier_only, log=rd.log,
                 epoch_eval=epoch_eval)
    if method == "finetune":
        kwargs["retain_loader"] = retain_loader
    elif method == "random_label":
        kwargs["forget_loader"] = forget_loader
        kwargs["retain_loader"] = retain_loader
        kwargs["num_classes"] = num_classes
        kwargs["forget_active_steps_per_epoch"] = forget_active_steps_per_epoch
        kwargs["forget_loss_weight"] = forget_loss_weight
        kwargs["trace"] = trace
        kwargs["dose_trace"] = dose_trace

    if per_cell_seed:
        # Reseed so this cell does not inherit RNG state from whichever cells
        # ran before it. Without this the comparison across K is confounded by
        # execution order.
        set_seed(cfg["seed"])

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

    # ---- provenance + dose record ---------------------------------------
    # Written for every cell, controlled or not, so a later audit never has to
    # trust a directory name or this script's docstring for what a run was.
    if method == "random_label":
        dose_pred = dose_accounting(n_retain, n_forget, bs, epochs,
                                    forget_active_steps_per_epoch,
                                    forget_loss_weight)
        dose_obs = {
            "observed_steps_total": trace_state["steps"],
            "observed_candidate_forget_presentations_total":
                dose_state["candidate_pres"],
            "observed_active_forget_bearing_steps_total":
                dose_state["active_steps"],
            "observed_active_forget_presentations_total":
                dose_state["active_pres"],
            "observed_total_weighted_forget_loss_coefficient_mass":
                dose_state["weight_mass"],
        }
        # Integer counts must agree exactly; the weighted mass is a float sum
        # and is compared with the same relative tolerance src/unlearn.py uses.
        for key_p, key_o in [
            ("total_candidate_forget_presentations",
             "observed_candidate_forget_presentations_total"),
            ("total_active_forget_presentations",
             "observed_active_forget_presentations_total"),
            ("total_active_forget_bearing_steps",
             "observed_active_forget_bearing_steps_total"),
        ]:
            if dose_pred[key_p] != dose_obs[key_o]:
                raise ValueError(
                    f"dose accounting mismatch on {key_p}: predicted "
                    f"{dose_pred[key_p]}, observed {dose_obs[key_o]}")
        mp = dose_pred["total_weighted_forget_loss_coefficient_mass"]
        mo = dose_obs["observed_total_weighted_forget_loss_coefficient_mass"]
        if abs(mp - mo) > 1e-9 * max(1.0, abs(mp)):
            raise ValueError(
                f"dose accounting mismatch on weighted coefficient mass: "
                f"predicted {mp!r}, observed {mo!r}")
        rd.log(f"  dose observed {dose_obs['observed_active_forget_bearing_steps_total']}"
               f" active steps, "
               f"{dose_obs['observed_active_forget_presentations_total']} active"
               f" presentations, mass {mo:.6g} -- matches prediction")
        rd.log(f"  training trace sha256:   {trace_state['h'].hexdigest()}")
        rd.log(f"  active dose trace sha256: {dose_state['h'].hexdigest()}")
    else:
        dose_pred = dose_obs = None

    rd.write_json("result.json", {
        "name": name,
        "method": method,
        "tag": tag,
        "classifier_only": classifier_only,
        "update_mode": "classifier_only" if classifier_only else "full_model",
        "head": head_name,
        "num_classes": num_classes,
        "forget_class": forget_class,
        "forget_identity_name": _identity_name(provenance, forget_class),
        "seed": cfg["seed"],
        "epochs": epochs,
        "per_cell_seed": bool(per_cell_seed),
        "n_retain": int(n_retain),
        "n_forget": int(n_forget),
        "retain_steps_per_epoch": int(steps_per_epoch),
        "training_trace_sha256": trace_state["h"].hexdigest(),
        "active_dose_trace_sha256": dose_state["h"].hexdigest(),
        "dose_predicted": dose_pred,
        "dose_observed": dose_obs,
        "provenance": provenance,
        "timings": {"method_s": method_time_s, "eval_s": eval_time_s},
    })
    rd.log(f"run complete -> {rd.root}")
    return row


def build_provenance(cfg: dict, ckpt_path: Path, device: str,
                     train_ds, train_eval_ds, test_ds, num_classes: int) -> dict:
    """Everything needed to prove later what data this run actually used.

    The identity manifest is the ORDERED list of identity directory names, so
    class index -> person is recoverable, and its hash detects any reordering
    or content drift in the source directory. The split manifest hashes cover
    which images landed in train and in test.
    """
    base = getattr(train_ds, "dataset", train_ds)
    identity_names = list(getattr(base, "identity_names", []) or [])
    samples = getattr(base, "samples", None)
    file_list = [str(p) for p, _ in samples] if samples else []

    def _split_indices(ds):
        idx = getattr(ds, "indices", None)
        return [int(i) for i in idx] if idx is not None else []

    prov = {
        "source_root": str(cfg["data"].get("root")),
        "dataset_name": cfg["data"].get("name"),
        "num_classes": int(num_classes),
        "max_identities": cfg["data"].get("max_identities"),
        "min_images": cfg["data"].get("min_images"),
        "max_images_per_identity": cfg["data"].get("max_images_per_identity"),
        "test_fraction": cfg["data"].get("test_fraction"),
        "batch_size": cfg["data"].get("batch_size"),
        "identity_names": identity_names,
        "identity_manifest_sha256": _sha_json(identity_names),
        "image_manifest_sha256": _sha_json(file_list),
        "n_images_total": len(file_list),
        "train_split_sha256": _sha_json(_split_indices(train_eval_ds)),
        "test_split_sha256": _sha_json(_split_indices(test_ds)),
        "n_train": len(_split_indices(train_eval_ds)),
        "n_test": len(_split_indices(test_ds)),
        "baseline_checkpoint": str(ckpt_path),
        "baseline_checkpoint_sha256": sha256_file(ckpt_path),
        "device": device,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if device.startswith("cuda") and torch.cuda.is_available()
                     else None),
        "config": cfg,
        **git_provenance(),
    }
    return prov


def main(cfg: dict, forget_classes: list[int], ckpt_path: Path,
        finetune_epochs: int, finetune_trajectory_every: int,
        conditions: list[str] = ("finetune", "random_label_clfonly"),
        forget_active_steps_per_epoch: int | None = None,
        forget_loss_weight: float = 1.0,
        per_cell_seed: bool = False,
        out_dir: str | None = None) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])   # once, up front -- matches run_experiment.py's
                            # own convention: RNG evolves naturally across
                            # the conditions that follow, not reset per one

    train_ds, train_eval_ds, test_ds, num_classes = D.build_datasets(cfg["data"],
                                                                     seed=cfg["seed"])
    bs, nw = cfg["data"]["batch_size"], cfg["data"]["num_workers"]
    train_eval_loader = D.make_loader(train_eval_ds, None, 512, False, nw)
    test_loader = D.make_loader(test_ds, None, 512, False, nw)

    backbone0, head0, head_name = load_checkpoint_model(cfg, ckpt_path, num_classes)
    u = cfg["unlearn"]
    out = out_dir or cfg["out_dir"]
    prov = build_provenance(cfg, ckpt_path, device, train_ds, train_eval_ds,
                            test_ds, num_classes)

    for fc in forget_classes:
        if "finetune" in conditions:
            run_condition(cfg, head_name, backbone0, head0, num_classes, train_ds,
                          train_eval_loader, test_loader, device, fc,
                          method="finetune", classifier_only=False,
                          epochs=finetune_epochs, lr=u["lr"],
                          weight_decay=u["weight_decay"],
                          trajectory_every=finetune_trajectory_every,
                          bs=bs, nw=nw, out_dir=out, provenance=prov)

        if "random_label_clfonly" in conditions:
            run_condition(cfg, head_name, backbone0, head0, num_classes, train_ds,
                          train_eval_loader, test_loader, device, fc,
                          method="random_label", classifier_only=True,
                          epochs=u["epochs"], lr=u["lr"],
                          weight_decay=u["weight_decay"], trajectory_every=1,
                          bs=bs, nw=nw, out_dir=out,
                          forget_active_steps_per_epoch=forget_active_steps_per_epoch,
                          forget_loss_weight=forget_loss_weight,
                          per_cell_seed=per_cell_seed, provenance=prov)


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
    ap.add_argument("--forget-active-steps-per-epoch", type=int, default=None,
                    help="dose control: carry the forget term on only this "
                         "many of the epoch's retain steps, evenly spaced. "
                         "Default (unset) = every step, the uncontrolled "
                         "behaviour every earlier run used. random_label only.")
    ap.add_argument("--forget-loss-weight", type=float, default=1.0,
                    help="dose control: coefficient on the forget CE term at "
                         "an active step (default 1.0). A loss coefficient, "
                         "NOT a gradient magnitude.")
    ap.add_argument("--per-cell-seed", action="store_true",
                    help="reseed immediately before each cell's method, so a "
                         "cell does not depend on how many cells ran before "
                         "it. Off by default to preserve existing behaviour; "
                         "required for a controlled sweep across class counts.")
    ap.add_argument("--out-dir", default=None,
                    help="override the config's out_dir for the unlearning "
                         "cells, so a controlled sweep can write somewhere "
                         "other than the baseline's own tree")
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
        conditions=conditions,
        forget_active_steps_per_epoch=a.forget_active_steps_per_epoch,
        forget_loss_weight=a.forget_loss_weight,
        per_cell_seed=a.per_cell_seed,
        out_dir=a.out_dir)
