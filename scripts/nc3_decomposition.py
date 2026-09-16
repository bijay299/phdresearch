#!/usr/bin/env python
"""
Centred-NC3 reference-frame decomposition over the class-count sweep cells.

WHY THIS EXISTS
---------------
The controlled nested class-count sweep (notes/decisions.md, 2026-09-16) found
that the centred difference-in-changes `dArcFace - dCE` orders K=100 < 250 <
1000 while the uncentred convention shows no such ordering. Centred NC3 is a
cosine between two CENTRED vectors, and the centre it subtracts is the mean of
the RETAINED classes' weights -- a reference frame that is itself free to move
during unlearning. So a change in the centred number is not by itself a
statement about the forget-class weight, and the two conventions disagreeing is
exactly what a reference-frame effect would look like.

This script evaluates the production expression at four (weight, centre)
corners per cell-epoch and reports the exact two-factor residual. The
arithmetic lives in `src/decomposition.py`; this file only walks the artifact
tree, enforces the validation gates, and tabulates.

WHAT IT READS
-------------
`nc3_state_ep<N>.npz` files written by
`scripts/unlearn_across_classes.py --dump-nc3-state`, alongside the
`trajectory.jsonl` produced by the same run. Canonical directories are never
written to and never need to be: the canonical values are read from
`--canonical-root` READ-ONLY, purely to gate the replay against them.

GATES -- a failure here stops the analysis, it does not warn
-----------------------------------------------------------
Per cell: epochs 0-3 present; recomputed production centred and uncentred NC3
exactly equal to the replay's own trajectory AND to the canonical trajectory;
output_forget/output_retain exactly equal to canonical; trace and dose hashes
exactly equal to canonical; identity/split/checkpoint hashes equal; class means
bit-identical across epochs; every decomposition quantity finite; epoch-0
corners collapse to one baseline; the residual identity closes.

    python scripts/nc3_decomposition.py \
        --replay-root logs_classcount_decomposition \
        --canonical-root logs_classcount \
        --out logs_classcount_decomposition/decomposition.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import decomposition as DC              # noqa: E402

EPOCHS = (0, 1, 2, 3)

# Values the replay must reproduce bit-for-bit from the canonical trajectory.
# probe_forget and nc1_angular are included deliberately: they come from the
# same extracted features, so agreeing on them is extra evidence that the
# replayed model is the canonical model and not merely a close one.
TRAJECTORY_KEYS = ("output_forget", "output_retain", "probe_forget",
                   "nc3_centred_forget", "nc3_uncentred_forget", "nc1_angular")

RESULT_KEYS = ("training_trace_sha256", "active_dose_trace_sha256",
               "forget_class", "head", "num_classes", "n_forget", "n_retain",
               "epochs", "per_cell_seed", "update_mode",
               "forget_identity_name")

PROVENANCE_KEYS = ("identity_manifest_sha256", "image_manifest_sha256",
                   "train_split_sha256", "test_split_sha256",
                   "baseline_checkpoint_sha256", "num_classes")


class GateFailure(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trajectory_by_epoch(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    out: dict[int, dict] = {}
    for r in rows:
        ep = int(r["epoch"])
        if ep in out:
            raise GateFailure(f"{path}: epoch {ep} appears more than once")
        out[ep] = r
    return out


def gate(cond: bool, msg: str) -> None:
    if not cond:
        raise GateFailure(msg)


def cell_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("K*/unlearn/*")
                  if p.is_dir() and (p / "result.json").exists())


def analyse_cell(replay_dir: Path, canonical_dir: Path) -> dict:
    """One cell: gate the replay against canonical, then decompose."""
    name = replay_dir.name
    res = json.loads((replay_dir / "result.json").read_text())
    can_res = json.loads((canonical_dir / "result.json").read_text())

    # ---- provenance and identity ---------------------------------------
    for k in RESULT_KEYS:
        gate(res.get(k) == can_res.get(k),
             f"{name}: result.json['{k}'] differs from canonical: "
             f"{res.get(k)!r} vs {can_res.get(k)!r}")
    for k in PROVENANCE_KEYS:
        gate(res["provenance"].get(k) == can_res["provenance"].get(k),
             f"{name}: provenance['{k}'] differs from canonical")
    gate(res["dose_predicted"]["forget_active_steps_per_epoch"] == 9,
         f"{name}: m is not 9")
    gate(float(res["dose_predicted"]["forget_loss_weight"]) == 1.0,
         f"{name}: lambda is not 1")
    gate(res["update_mode"] == "classifier_only",
         f"{name}: not a classifier-only cell")

    # ---- trajectory agreement ------------------------------------------
    traj = trajectory_by_epoch(replay_dir / "trajectory.jsonl")
    can_traj = trajectory_by_epoch(canonical_dir / "trajectory.jsonl")
    for ep in EPOCHS:
        gate(ep in traj, f"{name}: replay trajectory missing epoch {ep}")
        gate(ep in can_traj, f"{name}: canonical trajectory missing epoch {ep}")
        for k in TRAJECTORY_KEYS:
            a, b = traj[ep][k], can_traj[ep][k]
            gate(a == b, f"{name} ep{ep}: {k} differs from canonical: "
                         f"{a!r} vs {b!r}")

    # ---- saved geometric state -----------------------------------------
    fc = int(res["forget_class"])
    nc = int(res["num_classes"])
    state = {}
    for ep in EPOCHS:
        f = replay_dir / f"nc3_state_ep{ep}.npz"
        gate(f.exists(), f"{name}: no saved state for epoch {ep}")
        z = np.load(f)
        gate(int(z["forget_class"]) == fc and int(z["num_classes"]) == nc,
             f"{name} ep{ep}: saved state disagrees on class identity")
        state[ep] = {"weight": z["weight"], "class_means": z["class_means"],
                     "centre_ref": z["centre_ref"], "present": z["present"]}

    mu0 = state[0]["class_means"]
    fp0 = DC.feature_state_fingerprint(mu0)
    for ep in EPOCHS[1:]:
        gate(DC.feature_state_fingerprint(state[ep]["class_means"]) == fp0,
             f"{name} ep{ep}: class means moved under a frozen backbone")
        gate(np.array_equal(state[ep]["centre_ref"], state[0]["centre_ref"]),
             f"{name} ep{ep}: centring reference mask changed")
    gate(bool(state[0]["present"][fc]),
         f"{name}: the forget class has no class mean")

    W0 = state[0]["weight"]

    # ---- decomposition --------------------------------------------------
    per_epoch = {}
    for ep in EPOCHS:
        d = DC.decompose_cell(mu0, W0, state[ep]["weight"], fc)

        # production values must match the metric's own output, exactly
        gate(d["nc3_centred_forget"] == traj[ep]["nc3_centred_forget"],
             f"{name} ep{ep}: recomputed centred NC3 {d['nc3_centred_forget']!r} "
             f"!= trajectory {traj[ep]['nc3_centred_forget']!r}")
        gate(d["nc3_uncentred_forget"] == traj[ep]["nc3_uncentred_forget"],
             f"{name} ep{ep}: recomputed uncentred NC3 "
             f"{d['nc3_uncentred_forget']!r} != trajectory "
             f"{traj[ep]['nc3_uncentred_forget']!r}")

        flat = [v for v in d.values() if isinstance(v, float)]
        flat += [v for sub in d.values() if isinstance(sub, dict)
                 for v in sub.values() if isinstance(v, float)]
        gate(all(math.isfinite(v) for v in flat),
             f"{name} ep{ep}: a decomposition quantity is not finite")

        if ep == 0:
            for k in ("A_wt_c0", "A_w0_ct", "A_w0_c0"):
                gate(d[k] == d["A_wt_ct"],
                     f"{name}: epoch-0 corner {k} does not collapse to the "
                     f"baseline")
            gate(d["delta_centred_production"] == 0.0,
                 f"{name}: epoch-0 production change is not zero")
            gate(d["interaction_residual"] == 0.0,
                 f"{name}: epoch-0 residual is not zero")

        lhs = d["delta_centred_production"]
        rhs = (d["delta_centred_weight_only"] + d["delta_centred_centre_only"]
               + d["interaction_residual"])
        gate(abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs)),
             f"{name} ep{ep}: residual identity does not close "
             f"({lhs!r} vs {rhs!r})")

        d["output_forget"] = traj[ep]["output_forget"]
        d["output_retain"] = traj[ep]["output_retain"]
        per_epoch[ep] = d

    return {
        "name": name,
        "K": nc,
        "head": res["head"],
        "forget_class": fc,
        "forget_identity_name": res.get("forget_identity_name"),
        "seed": res["seed"],
        "replay_dir": str(replay_dir),
        "canonical_dir": str(canonical_dir),
        "training_trace_sha256": res["training_trace_sha256"],
        "active_dose_trace_sha256": res["active_dose_trace_sha256"],
        "feature_mean_fingerprint": fp0,
        "n_centre_classes": per_epoch[0]["n_centre_classes"],
        "epochs": {str(ep): per_epoch[ep] for ep in EPOCHS},
    }


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def summarise(cells: list[dict], epoch: int) -> list[dict]:
    """Aggregate across the four preregistered forget identities, per (K, head).

    Means are DESCRIPTIVE SUMMARIES of four fixed identities at one seed. They
    are not independent replicates and no test is implied.
    """
    rows = []
    keys = sorted({(c["K"], c["head"]) for c in cells})
    for K, head in keys:
        sel = [c for c in cells if c["K"] == K and c["head"] == head]
        e = [c["epochs"][str(epoch)] for c in sel]
        rows.append({
            "K": K, "head": head, "epoch": epoch, "n_cells": len(sel),
            "forget_weight_angle_deg": _mean([x["forget_weight"]["angle_deg"] for x in e]),
            "forget_weight_norm_ratio": _mean([x["forget_weight"]["norm_ratio"] for x in e]),
            "retain_centre_angle_deg": _mean([x["retain_centre"]["angle_deg"] for x in e]),
            "retain_centre_norm_ratio": _mean([x["retain_centre"]["norm_ratio"] for x in e]),
            "centred_weight_angle_deg": _mean([x["centred_forget_weight"]["angle_deg"] for x in e]),
            "centred_weight_norm_ratio": _mean([x["centred_forget_weight"]["norm_ratio"] for x in e]),
            "delta_centred_production": _mean([x["delta_centred_production"] for x in e]),
            "delta_centred_weight_only": _mean([x["delta_centred_weight_only"] for x in e]),
            "delta_centred_centre_only": _mean([x["delta_centred_centre_only"] for x in e]),
            "interaction_residual": _mean([x["interaction_residual"] for x in e]),
            "delta_uncentred_production": _mean([x["delta_uncentred_production"] for x in e]),
        })
    return rows


def render_markdown(out: dict) -> str:
    """Per-cell evidence first, aggregates second.

    The per-cell tables are the evidence; the aggregate tables below them are
    descriptive means over four fixed identities at one seed, and are labelled
    as such wherever they appear.
    """
    L: list[str] = []
    cells = sorted(out["cells"], key=lambda c: (c["K"], c["head"],
                                                c["forget_class"]))

    for epoch in (1, 3):
        L.append(f"\n### Per-cell decomposition, epoch {epoch}\n")
        L.append("| K | head | fc | w angle | w norm | c angle | c norm | "
                 "(w-c) angle | dProd | dWeightOnly | dCentreOnly | resid | "
                 "dUncentred |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cells:
            e = c["epochs"][str(epoch)]
            L.append(
                f"| {c['K']} | {c['head']} | {c['forget_class']} | "
                f"{e['forget_weight']['angle_deg']:.3f}° | "
                f"{e['forget_weight']['norm_ratio']:.4f} | "
                f"{e['retain_centre']['angle_deg']:.3f}° | "
                f"{e['retain_centre']['norm_ratio']:.4f} | "
                f"{e['centred_forget_weight']['angle_deg']:.3f}° | "
                f"{e['delta_centred_production']:+.4f} | "
                f"{e['delta_centred_weight_only']:+.4f} | "
                f"{e['delta_centred_centre_only']:+.4f} | "
                f"{e['interaction_residual']:+.4f} | "
                f"{e['delta_uncentred_production']:+.4f} |")

    for epoch, key in ((1, "summary_epoch1"), (3, "summary_epoch3")):
        L.append(f"\n### Mean over the four forget identities, epoch {epoch}"
                 f" — descriptive summary, not replicates\n")
        L.append("| K | head | n | w angle | c angle | (w-c) angle | dProd | "
                 "dWeightOnly | dCentreOnly | resid | dUncentred |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in out[key]:
            L.append(
                f"| {r['K']} | {r['head']} | {r['n_cells']} | "
                f"{r['forget_weight_angle_deg']:.3f}° | "
                f"{r['retain_centre_angle_deg']:.3f}° | "
                f"{r['centred_weight_angle_deg']:.3f}° | "
                f"{r['delta_centred_production']:+.4f} | "
                f"{r['delta_centred_weight_only']:+.4f} | "
                f"{r['delta_centred_centre_only']:+.4f} | "
                f"{r['interaction_residual']:+.4f} | "
                f"{r['delta_uncentred_production']:+.4f} |")

    L.append("\n### Difference-in-changes, ArcFace minus CE — our arithmetic\n")
    L.append("| convention | ep | " + " | ".join(
        f"K={K}" for K in sorted({c['K'] for c in cells})) + " |")
    L.append("|---|---|" + "---|" * len({c["K"] for c in cells}))
    Ks = sorted({c["K"] for c in cells})
    for label, field in (("centred, production", "delta_centred_production"),
                         ("centred, fixed epoch-0 centre",
                          "delta_centred_weight_only"),
                         ("centred, fixed epoch-0 weight",
                          "delta_centred_centre_only"),
                         ("interaction residual", "interaction_residual"),
                         ("uncentred", "delta_uncentred_production")):
        for epoch, key in ((1, "summary_epoch1"), (3, "summary_epoch3")):
            vals = []
            for K in Ks:
                by = {r["head"]: r for r in out[key] if r["K"] == K}
                vals.append(f"{by['arcface'][field] - by['ce'][field]:+.4f}")
            L.append(f"| {label} | {epoch} | " + " | ".join(vals) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-root", required=True)
    ap.add_argument("--canonical-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default=None,
                    help="also render the tables to this path")
    a = ap.parse_args()

    replay_root, canon_root = Path(a.replay_root), Path(a.canonical_root)
    dirs = cell_dirs(replay_root)
    if not dirs:
        raise GateFailure(f"no replayed cells found under {replay_root}")

    cells = []
    for d in dirs:
        rel = d.relative_to(replay_root)
        canonical = canon_root / rel
        gate(canonical.exists(),
             f"{d}: no canonical counterpart at {canonical}")
        cells.append(analyse_cell(d, canonical))
        print(f"  gated OK  {rel}")

    out = {
        "n_cells": len(cells),
        "cells": cells,
        "summary_epoch1": summarise(cells, 1),
        "summary_epoch3": summarise(cells, 3),
        "definitions": {
            "effective_weight": "head.weight row of the forget class, RAW "
                                "(fc.weight for LinearHead, W for ArcFaceHead); "
                                "normalisation happens inside the cosine, as in "
                                "metrics.nc3_alignment",
            "retain_centre": "mean over present classes excluding the forget "
                             "class of head.weight rows -- nc3_alignment's own "
                             "exclude_from_centre reference",
            "A": "A(w, c) = cos(f0 - g0, w - c), f0 the forget-class feature "
                 "mean and g0 the nanmean of retain-class feature means; both "
                 "constant here because the backbone is frozen",
            "interaction_residual": "A(wt,ct) - A(wt,c0) - A(w0,ct) + A(w0,c0); "
                                    "exact arithmetic leftover of a nonlinear "
                                    "metric at four corners, NOT a causal or "
                                    "statistical interaction",
            "counterfactuals": "descriptive interventions on the metric, not "
                               "additive causal contributions",
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n{len(cells)} cells gated and decomposed -> {a.out}")
    if a.markdown:
        md = render_markdown(out)
        Path(a.markdown).write_text(md)
        print(md)


if __name__ == "__main__":
    main()
