#!/usr/bin/env python
"""
Figure 3 -- fixed-epoch versus own-attainment centred DiC at K=100, two seeds.

Claims addressed: C5 (two-seed fixed-epoch recurrence) and C7 (the two contrast
rules can disagree in sign), including the identity-00142 (fc29) case.

Source: `evidence/k100_seed1/tables/{fixed_epoch_dic,matched_outcome_dic}.csv`,
both derived from the 16 verbatim trajectories in the same directory. Recorded
values only; no training, inference or replay.

The two panels do not measure the same thing. Panel A reads both objectives at
a common epoch under the same active forget dose. Panel B reads each objective
at its own first epoch with 0/10 correct forget predictions, so it confounds
objective with exposure: in 4 of the 7 attained pairs the two objectives are
read at different epochs.

LAYOUT NOTE. The cross-panel arc and the paragraph-length annotation that used
to carry the identity-00142 point are replaced by a short label beside that
point in each panel, and the surrounding prose now lives in the caption.

    python scripts/figures/make_fig3.py
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figlib import (  # noqa: E402
    IDENTITY_COLOUR, IDENTITY_MARKER, IDENTITY_NAME, REPO, WIDTH, ZERO_LINE,
    apply_style, save, write_provenance,
)

FIGURE_ID = "fig3_contrast_rules"
TABLES = REPO / "evidence" / "k100_seed1" / "tables"
FIXED = TABLES / "fixed_epoch_dic.csv"
MATCHED = TABLES / "matched_outcome_dic.csv"
CLASSES = [0, 29, 60, 95]
SEEDS = [0, 1]
YLIM = (-0.042, 0.192)
UNEQUAL = "#B8860B"

CAPTION = (
    "Two contrast rules applied to the same K = 100 cells, at two pipeline "
    "seeds, in the centred NC3 convention. The quantity is the "
    "difference-in-changes, ArcFace minus CE, of the change in centred NC3 from "
    "each cell's own epoch 0; it is project arithmetic rather than a metric from "
    "the source paper. (A) Fixed-epoch rule: both objectives are read at the "
    "same epoch, having received the same active forget dose of 9 steps per "
    "epoch at lambda = 1. The four-identity mean is positive at every epoch and "
    "at both seeds (+0.0574, +0.0648, +0.0607 at seed 0; +0.0595, +0.0708, "
    "+0.0686 at seed 1), while individual identities vary widely. Those means "
    "are descriptive over a complete and identical identity set, not estimates, "
    "so no band, error bar or interval is drawn. (B) Own-attainment rule: each "
    "objective is read at its own first epoch with 0 of 10 correct forget "
    "predictions. Because the two objectives generally reach that point at "
    "different epochs -- in 4 of the 7 attained pairs, marked in gold -- this "
    "panel confounds objective with exposure and is not a like-for-like head "
    "comparison. Identity 00142 at seed 1 is negative under this rule "
    "(-0.007274) while its fixed-epoch contrast is positive at all three epochs "
    "(+0.0271, +0.0246, +0.0173); it is labelled in both panels. That sign "
    "disagreement between the two rules is the point of the figure. Identity "
    "00524 at seed 0 has no attained pair, because ArcFace never reached 0 of 10 "
    "within the budget (1 of 10 at epoch 3); it is shown in the gutter strip "
    "below the frame rather than at any y position, since a glyph placed at a "
    "value would read as a measured contrast and one placed at zero would read "
    "as a contrast of exactly zero. No epoch was substituted for it and the "
    "budget was not extended. Panel B carries no means: seed 0 has 3 attained "
    "identities and seed 1 has 4, so a cross-seed mean would compare different "
    "identity sets, and restricting to the common set 00001, 00142 and 00284 "
    "conditions on attainment in both seeds. Points are individual identities, a "
    "within-seed factor rather than replicates. A negative difference-in-changes "
    "is not an NC3 sign reversal: there are no sign reversals in any of the 16 "
    "K = 100 cells at either seed. Neither quantity measures unlearning quality "
    "or representation erasure, and two seeds do not establish robustness."
)


def main() -> int:
    apply_style()
    fixed = [r for r in csv.DictReader(open(FIXED)) if r["convention"] == "centred"]
    matched = list(csv.DictReader(open(MATCHED)))
    assert len(fixed) == 24, len(fixed)
    assert len(matched) == 8, len(matched)

    fig = plt.figure(figsize=(WIDTH, 4.75), constrained_layout=True)
    # A dedicated gutter strip under panel B holds the absent observation, so it
    # sits outside the numerical data region rather than at some y value.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.115],
                          width_ratios=[1.04, 1.0], hspace=0.04)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    gutter = fig.add_subplot(gs[1, 1], sharex=axB)
    axB.tick_params(labelleft=False)
    consumed = {"panel_A": {}, "panel_B": {}, "no_pair": [], "highlight": {}}

    # ---- Panel A: fixed-epoch DiC ----------------------------------------
    axA.axhline(0.0, **ZERO_LINE)
    for seed in SEEDS:
        for fc in CLASSES:
            pts = sorted(
                (r for r in fixed
                 if int(r["seed"]) == seed and int(r["forget_class"]) == fc),
                key=lambda r: int(r["epoch"]),
            )
            ys = [float(r["dic"]) for r in pts]
            consumed["panel_A"][f"seed{seed}/{IDENTITY_NAME[fc]}"] = ys
            axA.plot(
                [int(r["epoch"]) for r in pts], ys,
                color=IDENTITY_COLOUR[fc], linewidth=1.1, alpha=0.9,
                marker=IDENTITY_MARKER[fc], markersize=5.0,
                markerfacecolor=IDENTITY_COLOUR[fc] if seed == 0 else "white",
                markeredgecolor=IDENTITY_COLOUR[fc], markeredgewidth=1.1,
                linestyle="-" if seed == 0 else "--", zorder=3,
            )
        means = [
            st.mean(float(r["dic"]) for r in fixed
                    if int(r["seed"]) == seed and int(r["epoch"]) == e)
            for e in (1, 2, 3)
        ]
        consumed["panel_A"][f"seed{seed}/descriptive_mean"] = means
        axA.plot([1, 2, 3], means, color="0.20", linewidth=2.4,
                 linestyle="-" if seed == 0 else "--", alpha=0.85, zorder=5,
                 solid_capstyle="round")

    # Concise label on the cell whose two rules disagree in sign.
    a29 = [float(r["dic"]) for r in sorted(
        (r for r in fixed if r["seed"] == "1" and r["forget_class"] == "29"),
        key=lambda r: int(r["epoch"]))]
    consumed["highlight"]["fixed_epoch"] = a29
    # Short enough to sit inside the frame, and parallel to panel B's label.
    # va="top" keeps the leader line clear of the label's second line.
    axA.annotate("00142, seed 1\n(positive here)", xy=(3, a29[2]),
                 xytext=(3.09, a29[2] + 0.007), fontsize=7.5,
                 color=IDENTITY_COLOUR[29], ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=IDENTITY_COLOUR[29],
                                 linewidth=0.9, shrinkB=4))

    axA.set_xticks([1, 2, 3])
    axA.set_xlim(0.80, 3.98)
    axA.set_xlabel("unlearning epoch")
    axA.set_ylabel("difference-in-changes, centred NC3\n"
                   "(ArcFace − CE, Δcosine)", fontsize=8.2)
    axA.set_title("A · fixed-epoch rule\nSame epoch; matched active forget dose.",
                  fontsize=8.5)

    # ---- Panel B: own-attainment DiC -------------------------------------
    axB.axhline(0.0, **ZERO_LINE)
    xpos = {fc: i for i, fc in enumerate(CLASSES)}
    for r in matched:
        seed, fc = int(r["seed"]), int(r["forget_class"])
        x = xpos[fc] + (-0.15 if seed == 0 else 0.15)
        if r["attained_pair"] != "True":
            consumed["no_pair"].append(f"seed{seed}/{IDENTITY_NAME[fc]}")
            continue
        y = float(r["matched_dic_centred"])
        consumed["panel_B"][f"seed{seed}/{IDENTITY_NAME[fc]}"] = {
            "dic": y, "ce_epoch": int(r["ce_first_zero_epoch"]),
            "arcface_epoch": int(r["arcface_first_zero_epoch"]),
            "exposure": r["exposure"],
        }
        neg = y < 0
        axB.plot([x], [y], marker=IDENTITY_MARKER[fc],
                 markersize=8.5 if neg else 6.4, color=IDENTITY_COLOUR[fc],
                 markerfacecolor=IDENTITY_COLOUR[fc] if seed == 0 else "white",
                 markeredgewidth=2.2 if neg else 1.3, linestyle="none", zorder=4)
        unequal = r["exposure"].startswith("unequal")
        axB.annotate(
            f"{r['ce_first_zero_epoch']} / {r['arcface_first_zero_epoch']}",
            xy=(x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5,
            color=UNEQUAL if unequal else "0.35",
            weight="bold" if unequal else "normal")
        if neg:
            consumed["highlight"]["matched"] = y
            axB.annotate("00142, seed 1\n(negative here)", xy=(x, y),
                         xytext=(x + 0.20, y - 0.012), fontsize=7.5,
                         color=IDENTITY_COLOUR[29], ha="left", va="top",
                         arrowprops=dict(arrowstyle="->",
                                         color=IDENTITY_COLOUR[29],
                                         linewidth=0.9, shrinkB=6))

    axB.set_xlim(-0.55, len(CLASSES) - 0.45)
    axB.tick_params(labelbottom=False)
    axB.set_ylim(*YLIM)
    axB.set_title("B · own-attainment rule\n"
                  "Each objective at its first observed zero.", fontsize=8.5)
    axB.text(0.5, 0.982, "annotation: CE epoch / ArcFace epoch\n"
             "gold = read at different epochs", transform=axB.transAxes,
             ha="center", va="top", fontsize=7.5, color="0.30",
             linespacing=1.4)

    # ---- gutter: absent observation, outside the data region -------------
    gutter.set_ylim(0, 1)
    gutter.set_yticks([])
    gutter.set_facecolor("#F4F4F4")
    for side in ("left", "right", "top"):
        gutter.spines[side].set_visible(False)
    gutter.grid(False)
    gutter.set_xticks(range(len(CLASSES)))
    gutter.set_xticklabels([IDENTITY_NAME[fc] for fc in CLASSES])
    gutter.set_xlabel("forget identity")
    gutter.set_ylabel("outside\ndata region", fontsize=7.2, color="0.45",
                      rotation=0, ha="right", va="center", labelpad=8)
    gutter.plot([xpos[95] - 0.15], [0.55], marker="x", markersize=7.5,
                markeredgewidth=1.8, color="0.25", linestyle="none", zorder=6)
    gutter.text(xpos[95] - 0.30, 0.55, "no attained pair (seed 0)",
                fontsize=7.5, color="0.25", ha="right", va="center")

    handles = [
        plt.Line2D([], [], color=IDENTITY_COLOUR[fc], marker=IDENTITY_MARKER[fc],
                   linestyle="none", markersize=6.0,
                   label=f"{IDENTITY_NAME[fc]} (class {fc})")
        for fc in CLASSES
    ] + [
        plt.Line2D([], [], color="0.35", marker="o", markerfacecolor="0.35",
                   linestyle="-", markersize=6.0, label="seed 0 (filled)"),
        plt.Line2D([], [], color="0.35", marker="o", markerfacecolor="white",
                   linestyle="--", markersize=6.0, label="seed 1 (hollow)"),
        # Named here rather than annotated in-panel: the in-panel labels
        # overran panel A's frame at manuscript width.
        plt.Line2D([], [], color="0.20", linewidth=2.4, linestyle="-",
                   label="descriptive mean, seed 0"),
        plt.Line2D([], [], color="0.20", linewidth=2.4, linestyle="--",
                   label="descriptive mean, seed 1"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.0,
               title="forget identity (colour) · seed (fill)")

    outputs = save(fig, FIGURE_ID)
    consumed["n_attained_pairs"] = len(consumed["panel_B"])
    consumed["n_unequal_exposure"] = sum(
        v["exposure"].startswith("unequal") for v in consumed["panel_B"].values())
    write_provenance(
        FIGURE_ID, sources=[FIXED, MATCHED], consumed=consumed,
        command="python scripts/figures/make_fig3.py", outputs=outputs,
        caption=CAPTION,
        notes=[
            "Centred convention only; both panels share one y-axis.",
            "Panel A means are descriptive over a complete and identical "
            "four-identity set, drawn with no band, error bar or interval.",
            "Panel B has no means, and the absent seed-0 / 00524 observation is "
            "drawn in a gutter strip outside the numerical data region.",
            "4 of the 7 attained pairs read the two objectives at different "
            "epochs, so Panel B confounds objective with exposure.",
            "Panel A's rule matches the active forget dose and the epoch; it is "
            "not described as equal optimization.",
            "The identity-00142 case is marked with a short label in each panel "
            "rather than a cross-panel arc; the explanation is in the caption.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
