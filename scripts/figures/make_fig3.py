#!/usr/bin/env python
"""
Figure 3 -- fixed-epoch versus own-attainment centred DiC at K=100, two seeds.

Claims addressed: C5 (two-seed fixed-epoch recurrence) and C7 (the two contrast
rules can disagree in sign), including the identity-00142 (fc29) case.

Source: `evidence/k100_seed1/tables/{fixed_epoch_dic,matched_outcome_dic}.csv`,
both derived from the 16 verbatim trajectories in the same directory. Recorded
values only; no training, inference or replay.

THE TWO PANELS DO NOT MEASURE THE SAME THING. Panel A reads both objectives at
a common epoch. Panel B reads each objective at its own first epoch with 0/10
correct forget predictions, so it confounds objective with exposure -- in 4 of
the 7 attained pairs the two objectives are read at different epochs.

    python scripts/figures/make_fig3.py
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from figlib import (  # noqa: E402
    IDENTITY_COLOUR, IDENTITY_MARKER, IDENTITY_NAME, REPO, ZERO_LINE,
    apply_style, footnote, save, write_provenance,
)

FIGURE_ID = "fig3_contrast_rules"
TABLES = REPO / "evidence" / "k100_seed1" / "tables"
FIXED = TABLES / "fixed_epoch_dic.csv"
MATCHED = TABLES / "matched_outcome_dic.csv"
CLASSES = [0, 29, 60, 95]
SEEDS = [0, 1]
YLIM = (-0.040, 0.160)


def main() -> int:
    apply_style()
    fixed = [r for r in csv.DictReader(open(FIXED)) if r["convention"] == "centred"]
    matched = list(csv.DictReader(open(MATCHED)))
    assert len(fixed) == 24, len(fixed)
    assert len(matched) == 8, len(matched)

    fig = plt.figure(figsize=(9.8, 5.0), constrained_layout=True)
    # Panel B gets a dedicated gutter strip beneath its frame. The absent
    # observation is drawn there, so it sits outside the numerical data region
    # instead of at some y-value that would read as a measured contrast.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13],
                          width_ratios=[1.06, 1.0], hspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    gutter = fig.add_subplot(gs[1, 1], sharex=axes[1])
    axes[1].tick_params(labelleft=False)
    consumed = {"panel_A": {}, "panel_B": {}, "no_pair": [], "highlight": {}}

    # ---- Panel A: fixed-epoch DiC ----------------------------------------
    ax = axes[0]
    ax.axhline(0.0, **ZERO_LINE)
    for seed in SEEDS:
        for fc in CLASSES:
            pts = sorted(
                (r for r in fixed
                 if int(r["seed"]) == seed and int(r["forget_class"]) == fc),
                key=lambda r: int(r["epoch"]),
            )
            eps = [int(r["epoch"]) for r in pts]
            ys = [float(r["dic"]) for r in pts]
            consumed["panel_A"][f"seed{seed}/{IDENTITY_NAME[fc]}"] = ys
            ax.plot(
                eps, ys, color=IDENTITY_COLOUR[fc], linewidth=1.0, alpha=0.85,
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
        ax.plot([1, 2, 3], means, color="0.20", linewidth=2.4,
                linestyle="-" if seed == 0 else "--", alpha=0.85, zorder=5,
                solid_capstyle="round")
        ax.annotate(
            f"descriptive mean, seed {seed}",
            xy=(3, means[-1]), xytext=(3.10, means[-1] + (0.011 if seed else -0.011)),
            fontsize=6.3, color="0.20", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color="0.55", linewidth=0.6,
                            shrinkA=1, shrinkB=1),
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.72, 4.15)
    ax.set_xlabel("unlearning epoch (both objectives read at the SAME epoch)")
    ax.set_ylabel("difference-in-changes, centred NC3\n"
                  r"$(\Delta_{\mathrm{ArcFace}} - \Delta_{\mathrm{CE}})$,"
                  r" $\Delta$cosine")
    ax.set_title("A · fixed-epoch contrast\nlike-for-like: equal optimization")

    # ---- Panel B: own-attainment DiC -------------------------------------
    ax = axes[1]
    ax.axhline(0.0, **ZERO_LINE)
    xpos = {fc: i for i, fc in enumerate(CLASSES)}
    highlight_xy = None
    for r in matched:
        seed, fc = int(r["seed"]), int(r["forget_class"])
        x = xpos[fc] + (-0.13 if seed == 0 else 0.13)
        if r["attained_pair"] != "True":
            consumed["no_pair"].append(f"seed{seed}/{IDENTITY_NAME[fc]}")
            continue
        y = float(r["matched_dic_centred"])
        consumed["panel_B"][f"seed{seed}/{IDENTITY_NAME[fc]}"] = {
            "dic": y, "ce_epoch": int(r["ce_first_zero_epoch"]),
            "arcface_epoch": int(r["arcface_first_zero_epoch"]),
            "exposure": r["exposure"],
        }
        is_neg = y < 0
        ax.plot(
            [x], [y], marker=IDENTITY_MARKER[fc], markersize=8.0 if is_neg else 6.4,
            color=IDENTITY_COLOUR[fc],
            markerfacecolor=IDENTITY_COLOUR[fc] if seed == 0 else "white",
            markeredgewidth=2.2 if is_neg else 1.3, linestyle="none", zorder=4,
        )
        unequal = r["exposure"].startswith("unequal")
        ax.annotate(
            f"{r['ce_first_zero_epoch']} / {r['arcface_first_zero_epoch']}",
            xy=(x, y), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=6.0,
            color="#B8860B" if unequal else "0.35",
            weight="bold" if unequal else "normal",
        )
        if is_neg:
            highlight_xy = (x, y)
            consumed["highlight"]["matched"] = y

    ax.set_xlim(-0.55, len(CLASSES) - 0.45)
    ax.tick_params(labelbottom=False)
    ax.set_title("B · own-attainment contrast\nconfounds objective with"
                 " exposure — NOT like-for-like")
    ax.set_ylim(*YLIM)

    # ---- the gutter: absent observations, outside the data region ---------
    gutter.set_ylim(0, 1)
    gutter.set_yticks([])
    gutter.set_facecolor("#F4F4F4")
    for spine in ("left", "right", "top"):
        gutter.spines[spine].set_visible(False)
    gutter.grid(False)
    gutter.set_xticks(range(len(CLASSES)))
    gutter.set_xticklabels([f"{IDENTITY_NAME[fc]}\n(class {fc})" for fc in CLASSES])
    gutter.set_xlabel(
        "forget identity   ·   annotation = CE epoch / ArcFace epoch\n"
        "gold = the two objectives read at DIFFERENT epochs", fontsize=7.2,
    )
    gutter.set_ylabel("outside the\ndata region", fontsize=5.8, color="0.48",
                      rotation=0, ha="right", va="center", labelpad=10)
    gutter.plot([xpos[95] - 0.13], [0.55], marker="x", markersize=7.0,
                markeredgewidth=1.7, color="0.25", linestyle="none", zorder=6)
    gutter.text(
        xpos[95] - 0.26, 0.55,
        "no attained pair — seed 0, identity 00524:\n"
        "ArcFace never reached 0/10 (1/10 at epoch 3)",
        fontsize=6.0, color="0.25", ha="right", va="center",
    )

    # ---- the sign disagreement, drawn across the two panels ---------------
    a_pts = sorted(
        (r for r in fixed if int(r["seed"]) == 1 and int(r["forget_class"]) == 29),
        key=lambda r: int(r["epoch"]),
    )
    a_vals = [float(r["dic"]) for r in a_pts]
    consumed["highlight"]["fixed_epoch"] = a_vals
    # A thin guide, bowed below both endpoints and drawn behind everything, so
    # it reads as an annotation linking one cell across two rules -- never as a
    # data series.
    con = ConnectionPatch(
        xyA=(3, a_vals[-1]), coordsA=axes[0].transData,
        xyB=highlight_xy, coordsB=axes[1].transData,
        connectionstyle="arc3,rad=0.42", color=IDENTITY_COLOUR[29],
        linewidth=0.8, linestyle=(0, (2, 2.5)), alpha=0.7, zorder=5,
    )
    fig.add_artist(con)
    for ax_, xy in ((axes[0], (3, a_vals[-1])), (axes[1], highlight_xy)):
        ax_.plot([xy[0]], [xy[1]], marker="o", markersize=13,
                 markerfacecolor="none", markeredgecolor=IDENTITY_COLOUR[29],
                 markeredgewidth=1.0, alpha=0.8, linestyle="none", zorder=2)
    axes[0].annotate(
        "identity 00142, seed 1 — the SAME cell under both rules:\n"
        f"fixed-epoch {a_vals[0]:+.4f} / {a_vals[1]:+.4f} / {a_vals[2]:+.4f},\n"
        f"own-attainment {consumed['highlight']['matched']:+.6f}.\n"
        "The two contrast rules disagree in SIGN.",
        xy=(3, a_vals[-1]), xytext=(0.86, 0.152), fontsize=6.4,
        color=IDENTITY_COLOUR[29], ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=IDENTITY_COLOUR[29],
                        linewidth=0.9, shrinkB=9,
                        connectionstyle="arc3,rad=-0.25"),
    )

    handles = [
        plt.Line2D([], [], color=IDENTITY_COLOUR[fc], marker=IDENTITY_MARKER[fc],
                   linestyle="none", markersize=6.0,
                   label=f"class {fc} = identity {IDENTITY_NAME[fc]}")
        for fc in CLASSES
    ] + [
        plt.Line2D([], [], color="0.35", marker="o", markerfacecolor="0.35",
                   linestyle="-", markersize=6.0, label="seed 0 (filled)"),
        plt.Line2D([], [], color="0.35", marker="o", markerfacecolor="white",
                   linestyle="--", markersize=6.0, label="seed 1 (hollow)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005),
               ncol=6, frameon=False, columnspacing=1.5, handlelength=2.2,
               title="forget identity (a within-seed factor, not a replicate)"
                     "   ·   seed (the only replication unit; there are two)")
    footnote(fig, [
        "Panel B carries NO means: seed 0 has 3 attained identities and seed 1 "
        "has 4, so any cross-seed mean would compare different identity sets, "
        "and restricting to the common set {00001, 00142, 00284} conditions on "
        "attainment in both seeds.",
        "Panel A's means are descriptive over a complete and identical "
        "four-identity set — not estimates — so no band, error bar or interval "
        "is drawn.",
        "The absent seed-0 / 00524 observation sits in the gutter strip: it is "
        "not imputed, not substituted by epoch 3, and not placed at y = 0, "
        "where it would read as a contrast of exactly zero.",
        "There are ZERO centred NC3 sign reversals in all 16 K=100 cells at "
        "both seeds. A negative difference-in-changes is not a sign reversal, "
        "and neither quantity measures unlearning quality or representation "
        "erasure. Two seeds do not establish robustness.",
    ], y=-0.125)

    outputs = save(fig, FIGURE_ID)
    consumed["n_attained_pairs"] = len(consumed["panel_B"])
    consumed["n_unequal_exposure"] = sum(
        v["exposure"].startswith("unequal") for v in consumed["panel_B"].values()
    )
    write_provenance(
        FIGURE_ID, sources=[FIXED, MATCHED], consumed=consumed,
        command="python scripts/figures/make_fig3.py", outputs=outputs,
        notes=[
            "Centred convention only; both panels share one y-axis.",
            "Panel A means are descriptive over a complete and identical "
            "four-identity set, drawn with no band, error bar or interval.",
            "Panel B has no means, and the absent seed-0 / 00524 observation is "
            "drawn outside the numerical data region.",
            "4 of the 7 attained pairs read the two objectives at different "
            "epochs, so Panel B confounds objective with exposure.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
