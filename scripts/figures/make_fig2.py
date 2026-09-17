#!/usr/bin/env python
"""
Figure 2 -- seed-0 classifier rotation, norm ratio, and the reference-frame
decomposition of the centred difference-in-changes.

Claims addressed: C6a (rotation and norm) and C6b (the K ordering is a
reference-frame effect). Seed 0 only -- no seed-1 decomposition exists.

Source: `evidence/decomposition_fig2/fig2_plotting_inputs.csv`, extracted from
the git-ignored `logs_classcount_decomposition/decomposition.json` by
`scripts/figures/extract_fig2_inputs.py`, which records the source hash. The
figure reads the tracked table so a reviewer can check it without the artifact
tree; `--from-artifact` reads the artifact directly and asserts the two agree
exactly.

Nothing here recomputes the decomposition. All four corner quantities were
produced by `scripts/nc3_decomposition.py` under its own validation gates.

LAYOUT NOTE. K = 500 is marked as excluded in every panel that has a K axis,
including the bar panel and the feature-centre-ratio inset, so its absence reads
the same way everywhere and no value is invented for it. The gate record and the
geometric caveats live in the caption, not in the image.

    python scripts/figures/make_fig2.py --from-artifact
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figlib import (  # noqa: E402
    IDENTITY_MARKER, IDENTITY_NAME, OBJECTIVE_COLOUR, OBJECTIVE_LABEL, REPO,
    WIDTH, ZERO_LINE, apply_style, descriptive_mean_marker, identity_handle,
    save, write_provenance,
)

FIGURE_ID = "fig2_seed0_decomposition"
TABLE = REPO / "evidence" / "decomposition_fig2" / "fig2_plotting_inputs.csv"
ARTIFACT = REPO / "logs_classcount_decomposition" / "decomposition.json"
K500_BASELINES = [
    REPO / "logs_classcount" / "K500" / f"faces_{h}_seed0" / "results.jsonl"
    for h in ("ce", "arcface")
]
KS = [100, 250, 1000]
K_TICKS = [100, 250, 500, 1000]
HEADS = ("ce", "arcface")
K500_GATE_PP = 2.00
EXCLUDED_FILL = "#ECECEC"

# |g_0| / |f_0| for CE, in K order -- the feature-side centre relative to the
# forget class mean. Recorded in the 2026-09-16 decomposition entry.
CE_FEATURE_CENTRE_RATIO = {100: 0.80, 250: 0.73, 1000: 0.69}

CAPTION = (
    "Seed-0 reference-frame decomposition of centred NC3 over the 24 controlled "
    "nested class-count sweep cells. No seed-1 decomposition exists, so nothing "
    "here is a two-seed result. (A) Forget-class weight rotation from epoch 0. "
    "Per-K means place CE at 16.74 to 17.81 degrees at epoch 1 against ArcFace's "
    "3.28 to 4.74 degrees; the per-cell ranges are 16.24 to 18.56 and 2.52 to "
    "5.05 degrees. CE's rotation is non-monotone in K and nearly flat. (B) "
    "Forget-weight norm ratio: CE contracts to 0.8129 to 0.8778 across epochs 1 "
    "and 3, ArcFace holds 0.9963 to 1.0016. These ratios are descriptive. A "
    "cosine is invariant to scaling its whole argument, so the uncentred cosine "
    "is unaffected by the weight norm; the centred cosine takes w - c, and "
    "changing w against a fixed c can change that direction. No counterfactual "
    "on the norm was run, so no attribution from these ratios to the metric is "
    "claimed. (C) Each bar is a baseline-subtracted counterfactual "
    "difference-in-changes, ArcFace minus CE, of the production centred NC3 "
    "A(w, c) = cos(f0 - g0, w - c): production is A(w_t, c_t) - A(w_0, c_0); "
    "weight-only holds the retain-weight centre at epoch 0; centre-only holds "
    "the weight at epoch 0; and the residual is production minus weight-only "
    "minus centre-only, exactly, per cell. Holding the centre at epoch 0 "
    "reproduces the K ordering almost exactly, while moving only the centre "
    "contributes +0.000200 to +0.000413 at epoch 1, with the residual of the "
    "same order. At epoch 3 the centre-only term reaches +0.001007, and across "
    "all 24 cells and all epochs the largest single-cell centre-only magnitude "
    "is 0.006460 and the largest residual 0.006797, against a largest production "
    "magnitude of 0.200685. So the ordering is the sensitivity of centred NC3 to "
    "a reference frame that varies with K, not increasing classifier rotation. "
    "The four corners are an arithmetic decomposition of the metric, not an "
    "intervention on training; the residual is a leftover rather than an "
    "interaction effect, since A is nonlinear in both arguments. None of these "
    "quantities measures unlearning quality. Points are individual identities, a "
    "within-seed factor rather than replicates, so means are descriptive and no "
    "dispersion is drawn. The sweep does not isolate K: changing K also changes "
    "the head's output width, the training population (3,926 to 38,815 train "
    "images), retain steps per epoch (31, 77, 303), candidate forget exposure "
    "(1,240, 3,080, 12,120 per epoch), baseline-training BatchNorm exposure (the "
    "unlearning phase runs the backbone in eval mode and accrues none), and task "
    "difficulty; only the active forget presentations, 360 per epoch, are held "
    "fixed. K = 500 carries no unlearning cells and is marked as excluded in "
    "every panel with a K axis, including the inset. Its baselines were CE "
    "68.9676 % and ArcFace 71.3232 %, an absolute gap of 2.3556 pp against a "
    "prespecified 2.00 pp head-fairness gate. The gate is symmetric and failed "
    "because ArcFace exceeded CE, not because ArcFace underperformed; neither "
    "head was retuned to rescue it and both baselines are retained as artifacts. "
    "This is a deliberate exclusion under a prespecified rule rather than "
    "missing data: no value is imputed and no line is drawn across the gap in "
    "any panel."
)


def load_table(path: Path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({
                **{k: r[k] for k in ("cell_name", "head", "identity")},
                "K": int(r["K"]), "epoch": int(r["epoch"]),
                "forget_class": int(r["forget_class"]),
                **{k: float(r[k]) for k in r
                   if k not in ("cell_name", "head", "identity", "K", "epoch",
                                "forget_class", "seed", "n_centre_classes",
                                "n_present_classes")},
            })
    return rows


def load_artifact():
    doc = json.loads(ARTIFACT.read_text())
    rows = []
    for c in doc["cells"]:
        for ep in ("1", "2", "3"):
            e = c["epochs"][ep]
            rows.append({
                "cell_name": c["name"], "head": c["head"], "K": c["K"],
                "identity": c["forget_identity_name"],
                "forget_class": c["forget_class"], "epoch": int(ep),
                "forget_weight_angle_deg": e["forget_weight"]["angle_deg"],
                "forget_weight_norm_ratio": e["forget_weight"]["norm_ratio"],
                **{k: e[k] for k in ("delta_centred_production",
                                     "delta_centred_weight_only",
                                     "delta_centred_centre_only",
                                     "interaction_residual")},
            })
    return rows


def k500_gate():
    out = {}
    for path, head in zip(K500_BASELINES, ("ce", "arcface")):
        acc = None
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if "output_overall" in rec:
                acc = rec["output_overall"]
        out[head] = acc
    out["gap_pp"] = abs(out["ce"] - out["arcface"]) * 100.0
    return out


def sel(rows, K, head, epoch):
    return sorted((r for r in rows if r["K"] == K and r["head"] == head
                   and r["epoch"] == epoch), key=lambda r: r["forget_class"])


def scatter_by_K(ax, rows, field, epoch, alpha, size):
    """Per-identity points, jittered in x only, plus a bare descriptive mean."""
    for head in HEADS:
        for K in KS:
            cells = sel(rows, K, head, epoch)
            for i, c in enumerate(cells):
                ax.plot([K * (1.0 + 0.022 * (i - 1.5))], [c[field]],
                        marker=IDENTITY_MARKER[c["forget_class"]],
                        color=OBJECTIVE_COLOUR[head], markersize=size,
                        markerfacecolor="none", markeredgewidth=1.0,
                        linestyle="none", alpha=alpha, zorder=3)
            descriptive_mean_marker(ax, [K], [st.mean(c[field] for c in cells)],
                                    OBJECTIVE_COLOUR[head], alpha=alpha)
        # Only the 100-250 segment is drawn. A segment spanning 250 to 1000
        # would read as an interpolated observation at the excluded K = 500.
        ax.plot([100, 250],
                [st.mean(c[field] for c in sel(rows, k, head, epoch))
                 for k in (100, 250)],
                color=OBJECTIVE_COLOUR[head], linewidth=1.0,
                alpha=alpha * 0.8, zorder=2)


def mark_excluded_log(ax):
    """The K = 500 slot on a log K axis: shaded, empty, labelled."""
    ax.axvspan(430, 585, color=EXCLUDED_FILL, zorder=0)
    ax.text(500, 0.52, "K = 500\nexcluded", transform=ax.get_xaxis_transform(),
            ha="center", va="center", fontsize=7.5, color="0.40",
            rotation=90, linespacing=1.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-artifact", action="store_true",
                    help="also read the git-ignored artifact and assert "
                         "bit-equality with the tracked table")
    args = ap.parse_args()

    apply_style()
    rows = load_table(TABLE)
    assert len(rows) == 72, f"expected 72 cell-epochs, got {len(rows)}"

    sources = [TABLE] + K500_BASELINES
    notes = []
    if args.from_artifact:
        art = load_artifact()
        # Cell names repeat across K, so K is part of the key.
        by = {(r["cell_name"], r["K"], r["epoch"]): r for r in art}
        assert len(by) == len(art) == 72, "cell key is not unique"
        for r in rows:
            a = by[(r["cell_name"], r["K"], r["epoch"])]
            for k, v in a.items():
                if isinstance(v, float):
                    assert r[k] == v, f"table/artifact mismatch {k} {r['cell_name']}"
        sources.append(ARTIFACT)
        notes.append("Tracked table verified bit-equal to the artifact tree "
                     "for every field consumed.")

    gate = k500_gate()
    fig = plt.figure(figsize=(WIDTH, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    consumed = {"cells": 24, "epochs": [1, 3], "K": KS}

    # ---- Panel A: forget-weight rotation ---------------------------------
    scatter_by_K(axA, rows, "forget_weight_angle_deg", 3, 0.30, 4.4)
    scatter_by_K(axA, rows, "forget_weight_angle_deg", 1, 1.00, 4.8)
    axA.set_ylim(0, 25)
    axA.set_ylabel("rotation from epoch 0 (degrees)")
    axA.set_title("A · forget-class weight rotation")

    # ---- Panel B: forget-weight norm ratio -------------------------------
    axB.axhline(1.0, **ZERO_LINE)
    scatter_by_K(axB, rows, "forget_weight_norm_ratio", 3, 0.30, 4.4)
    scatter_by_K(axB, rows, "forget_weight_norm_ratio", 1, 1.00, 4.8)
    axB.set_ylim(0.75, 1.05)
    axB.set_ylabel(r"norm ratio  $\|w_t\|\,/\,\|w_0\|$")
    axB.set_title("B · forget-class weight norm")

    for ax in (axA, axB):
        ax.set_xscale("log")
        ax.set_xlim(80, 1300)
        ax.set_xticks(K_TICKS)
        ax.set_xticklabels([str(k) for k in K_TICKS])
        ax.minorticks_off()
        ax.set_xlabel("K (identities)")
        mark_excluded_log(ax)

    # ---- Panel C: decomposition of the centred DiC at epoch 1 ------------
    quantities = [
        ("delta_centred_production", "production", "#44546A"),
        ("delta_centred_weight_only", "weight-only", "#0072B2"),
        ("delta_centred_centre_only", "centre-only", "#56B4E9"),
        ("interaction_residual", "residual", "#B0B0B0"),
    ]
    # Panel C uses the same four K slots as A and B, with 500 left empty, so the
    # exclusion reads identically across every panel that has a K axis.
    slot = {K: i for i, K in enumerate(K_TICKS)}
    width = 0.20
    dic = {}
    for qi, (field, label, colour) in enumerate(quantities):
        heights = []
        for K in KS:
            a = st.mean(c[field] for c in sel(rows, K, "arcface", 1))
            ce = st.mean(c[field] for c in sel(rows, K, "ce", 1))
            heights.append(a - ce)
        dic[field] = dict(zip(KS, heights))
        xs = [slot[K] + (qi - 1.5) * width for K in KS]
        axC.bar(xs, heights, width=width * 0.92, color=colour, label=label,
                edgecolor="white", linewidth=0.4, zorder=3)
    consumed["panel_C_dic_epoch1"] = {
        f: {str(k): v for k, v in d.items()} for f, d in dic.items()
    }

    axC.axvspan(slot[500] - 0.5, slot[500] + 0.5, color=EXCLUDED_FILL, zorder=0)
    axC.text(slot[500], 0.5, "K = 500 excluded",
             transform=axC.get_xaxis_transform(), ha="center", va="center",
             fontsize=7.5, color="0.40", rotation=90)
    axC.axhline(0.0, **ZERO_LINE)
    axC.set_xticks(range(len(K_TICKS)))
    axC.set_xticklabels([str(k) for k in K_TICKS])
    axC.set_xlim(-0.55, len(K_TICKS) - 0.45)
    axC.set_xlabel("K (identities)")
    axC.set_ylim(-0.014, 0.248)
    axC.set_ylabel("ArcFace − CE difference-in-changes,\n"
                   "centred NC3 (Δcosine, baseline-subtracted)")
    axC.set_title("C · reference-frame decomposition at epoch 1")
    axC.legend(loc="upper left", frameon=False, fontsize=8.0,
               handlelength=1.2, labelspacing=0.26, borderpad=0.1,
               title="counterfactual corner", alignment="left")

    # The last two quantities are ~3 orders below production, so their bars are
    # correctly near-invisible. Their values are given with the scale explicit.
    small = "\n".join(
        f"{k:>5}  {dic['delta_centred_centre_only'][k] * 1e3:+6.3f}"
        f"  {dic['interaction_residual'][k] * 1e3:+6.3f}"
        for k in KS
    )
    axC.text(
        0.245, 0.985,
        "centre-only and residual, epoch 1,\n"
        "in units of $10^{-3}$ (bars are to scale)\n"
        "    K  centre-only  residual\n" + small,
        transform=axC.transAxes, ha="left", va="top", fontsize=7.5,
        family="DejaVu Sans Mono", color="0.20", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7",
                  edgecolor="0.80", linewidth=0.6),
    )

    # Inset: CE feature-side centre magnitude, own axis and title, points only.
    ins = axC.inset_axes([0.665, 0.675, 0.245, 0.275])
    ins.axvspan(slot[500] - 0.5, slot[500] + 0.5, color=EXCLUDED_FILL, zorder=0)
    ins.plot([slot[K] for K in KS], [CE_FEATURE_CENTRE_RATIO[K] for K in KS],
             marker="o", markersize=4.0, color=OBJECTIVE_COLOUR["ce"],
             linestyle="none", zorder=3)
    ins.text(slot[500], 0.5, "excluded", transform=ins.get_xaxis_transform(),
             ha="center", va="center", fontsize=6.6, color="0.45", rotation=90)
    ins.set_xticks(range(len(K_TICKS)))
    ins.set_xticklabels([str(k) for k in K_TICKS], fontsize=6.8)
    ins.set_xlim(-0.55, len(K_TICKS) - 0.45)
    ins.set_ylim(0.6, 0.9)
    ins.set_yticks([0.6, 0.75, 0.9])
    ins.tick_params(labelsize=6.8, length=2, pad=1)
    ins.set_title(r"CE  $|g_0|\,/\,|f_0|$", fontsize=7.6, pad=2)
    ins.grid(alpha=0.5)

    handles = [
        plt.Line2D([], [], color=OBJECTIVE_COLOUR[h], linewidth=2.0,
                   label=OBJECTIVE_LABEL[h]) for h in HEADS
    ] + [
        plt.Line2D([], [], color="0.35", marker="_", markersize=11,
                   markeredgewidth=1.8, linestyle="none",
                   label="descriptive mean"),
        plt.Line2D([], [], color="0.35", linewidth=2.0, alpha=0.30,
                   label="epoch 3 (lighter); epoch 1 solid"),
    ] + [identity_handle(fc, markerfacecolor="none") for fc in (0, 29, 60, 95)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, columnspacing=1.6, handlelength=2.2,
               title="seed 0 only · panels A and B: objective (colour), "
                     "forget identity (marker)")

    outputs = save(fig, FIGURE_ID)
    consumed["k500_gate"] = gate
    consumed["panel_A_field"] = "forget_weight_angle_deg"
    consumed["panel_B_field"] = "forget_weight_norm_ratio"
    write_provenance(
        FIGURE_ID, sources=sources, consumed=consumed,
        command="python scripts/figures/make_fig2.py"
                + (" --from-artifact" if args.from_artifact else ""),
        outputs=outputs, caption=CAPTION,
        notes=notes + [
            "Seed 0 only. Stratum A, 24 cells, K in {100, 250, 1000}.",
            "K=500 is marked as excluded in panels A, B, C and the inset; no "
            "value is imputed and no line spans it.",
            "The feature-centre-ratio inset shows points only, with no "
            "connecting line: three K points do not establish a trend.",
            "Panel C bars are baseline-subtracted counterfactual differences-"
            "in-changes of the metric, not causal contributions and not a "
            "measure of unlearning quality.",
            "Centre-only and residual values are printed with the 1e-3 scale "
            "stated; the bars themselves remain to scale.",
            "No error bars: identities within a seed share one backbone per "
            "head, one split and one baseline checkpoint.",
            "The gate record and the geometric caveats live in the caption.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
