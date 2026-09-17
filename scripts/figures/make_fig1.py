#!/usr/bin/env python
"""
Figure 1 -- K=100 output forgetting and NC3 trajectories, both seeds.

Claims addressed: C1 (ArcFace does not eliminate classifier-only output
forgetting) and C3 (reversal is not necessary for output forgetting), on the
controlled core (strata A + B).

Source: the 16 verbatim per-cell trajectories under
`evidence/k100_seed1/trajectories/`, read through the tabulated
`trajectories_both_seeds.csv` in the same directory. Recorded values only; no
training, inference or replay.

    python scripts/figures/make_fig1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figlib import (  # noqa: E402
    ANCHOR_RULE, IDENTITY_LINESTYLE, IDENTITY_MARKER, IDENTITY_NAME,
    OBJECTIVE_COLOUR, OBJECTIVE_LABEL, REPO, ZERO_LINE, apply_style,
    identity_handle, save, write_provenance,
)

FIGURE_ID = "fig1_k100_trajectories"
TABLE = REPO / "evidence" / "k100_seed1" / "tables" / "trajectories_both_seeds.csv"
TRAJ_DIR = REPO / "evidence" / "k100_seed1" / "trajectories"
CLASSES = [0, 29, 60, 95]
# Uncentred panel limits are symmetric about zero on purpose: the claim the
# panel carries is that every recorded change is negative, and a one-sided
# window would make that unreadable as a sign statement.
UNCENTRED_LIM = 0.42


def load():
    cells = {}
    with open(TABLE) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["seed"]), r["head"], int(r["forget_class"]))
            cells.setdefault(key, []).append(r)
    for rows in cells.values():
        rows.sort(key=lambda r: int(r["epoch"]))
    return cells


def first_zero(rows):
    """First post-baseline epoch with 0/10 correct forget predictions."""
    for r in rows:
        if int(r["epoch"]) > 0 and int(r["forget_correct"]) == 0:
            return int(r["epoch"])
    return None


def main() -> int:
    apply_style()
    cells = load()
    assert len(cells) == 16, f"expected 16 cells, got {len(cells)}"

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), constrained_layout=True)
    consumed = {"cells": [], "first_zero_epoch": {}, "unmarked": []}

    # Attainment markers stack exactly on (epoch, 0) whenever several cells
    # reach 0/10 at the same epoch. Concentric sizing keeps every one of them
    # individually visible without moving any marker off its true epoch.
    ring = {
        (h, fc): 8.8 - 0.95 * i - (0.5 if h == "arcface" else 0.0)
        for i, fc in enumerate(CLASSES)
        for h in ("ce", "arcface")
    }
    draw_order = sorted(ring, key=lambda k: -ring[k])

    for row, seed in enumerate((0, 1)):
        for col in range(3):
            ax = axes[row][col]
            ax.axvspan(0.88, 1.12, **ANCHOR_RULE)
        for head in ("ce", "arcface"):
            for fc in CLASSES:
                rows = cells[(seed, head, fc)]
                consumed["cells"].append(f"seed{seed}/{head}_fc{fc}")
                ep = [int(r["epoch"]) for r in rows]
                colour = OBJECTIVE_COLOUR[head]
                style = dict(
                    color=colour, linestyle=IDENTITY_LINESTYLE[fc],
                    marker=IDENTITY_MARKER[fc], markersize=3.4,
                    markerfacecolor="white", markeredgewidth=0.9, alpha=0.95,
                )

                # --- column 1: forget-class correct out of 10 -----------------
                ax = axes[row][0]
                correct = [int(r["forget_correct"]) for r in rows]
                assert all(int(r["forget_total"]) == 10 for r in rows)
                ax.plot(ep, correct, **style)
                fz = first_zero(rows)
                consumed["first_zero_epoch"][f"seed{seed}/{head}_fc{fc}"] = fz
                if fz is None:
                    consumed["unmarked"].append(f"seed{seed}/{head}_fc{fc}")

                # --- column 2: centred NC3, forget class ----------------------
                axes[row][1].plot(
                    ep, [float(r["nc3_centred"]) for r in rows], **style
                )

                # --- column 3: uncentred NC3, change from own epoch 0 ---------
                axes[row][2].plot(
                    ep, [float(r["d_nc3_uncentred"]) for r in rows], **style
                )

        # Attainment markers last, largest ring first, so none is buried.
        for head, fc in draw_order:
            fz = first_zero(cells[(seed, head, fc)])
            if fz is None:
                continue
            axes[row][0].plot(
                [fz], [0], marker=IDENTITY_MARKER[fc],
                color=OBJECTIVE_COLOUR[head], markersize=ring[(head, fc)],
                markerfacecolor="none", markeredgewidth=1.5,
                linestyle="none", zorder=6,
            )

        # --- per-row axis dressing ------------------------------------------
        a0, a1, a2 = axes[row]
        a0.set_ylim(-0.75, 10.6)
        a0.set_yticks(range(0, 11, 2))
        a0.set_ylabel("correct out of 10")
        sec = a0.secondary_yaxis(
            "right", functions=(lambda v: v / 10.0, lambda v: v * 10.0)
        )
        sec.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        sec.set_ylabel("fraction", fontsize=6.8, labelpad=1)
        sec.tick_params(labelsize=6.5, pad=1)

        a1.set_ylim(-1.0, 1.0)
        a1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        a1.axhline(0.0, **ZERO_LINE)
        a1.set_ylabel("NC3 centred (cosine)")

        a2.set_ylim(-UNCENTRED_LIM, UNCENTRED_LIM)
        a2.axhline(0.0, **ZERO_LINE)
        a2.set_ylabel(r"$\Delta$ NC3 uncentred (cosine)")

        for ax in axes[row]:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlim(-0.18, 3.25)
            ax.set_xlabel("unlearning epoch")
        axes[row][0].text(
            0.012, 0.965, f"seed {seed}", transform=axes[row][0].transAxes,
            va="top", ha="left", fontsize=9.0, weight="bold", color="0.15",
        )

    axes[0][0].set_title("output forgetting\nforget-class test accuracy")
    axes[0][1].set_title("NC3, centred convention\n"
                         "forget class excluded from centring reference")
    axes[0][2].set_title("NC3, uncentred convention\n"
                         "change from that cell's own epoch 0")
    axes[0][2].text(1.0, 0.985, "decomposition anchor  ",
                    transform=axes[0][2].get_xaxis_transform(),
                    ha="right", va="top", fontsize=6.4, color="0.42",
                    rotation=90)

    # The one cell that never attains 0/10 is annotated in place, not imputed.
    ax = axes[0][0]
    ax.annotate(
        "ArcFace identity 00524:\nnot attained within budget\n(1/10 at epoch 3;"
        " no marker)",
        xy=(3.0, 1.0), xytext=(1.30, 4.9), fontsize=6.6,
        color=OBJECTIVE_COLOUR["arcface"], ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=OBJECTIVE_COLOUR["arcface"],
                        linewidth=0.8, shrinkB=3),
    )

    handles = [
        plt.Line2D([], [], color=OBJECTIVE_COLOUR[h], linewidth=1.8,
                   label=OBJECTIVE_LABEL[h])
        for h in ("ce", "arcface")
    ] + [
        plt.Line2D([], [], color="0.35", marker="o", linestyle="none",
                   markersize=6.4, markerfacecolor="none", markeredgewidth=1.5,
                   label="first epoch at 0/10"),
    ] + [identity_handle(fc) for fc in CLASSES]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.135),
        ncol=4, frameon=False, columnspacing=2.0, handlelength=2.8,
        title="objective (colour)   ·   forget identity (line style and marker;"
              " a within-seed factor, not a replicate)   ·   attainment rings"
              " are sized per cell so coincident attainments stay separable",
    )

    outputs = save(fig, FIGURE_ID)
    consumed["panels"] = {
        "col1": "forget_correct / forget_total (denominator 10, verified)",
        "col2": "nc3_centred",
        "col3": "d_nc3_uncentred (change from that cell's own epoch 0)",
    }
    consumed["n_cells"] = len(cells)
    consumed["rows_read"] = 64
    write_provenance(
        FIGURE_ID,
        sources=[TABLE] + sorted(TRAJ_DIR.rglob("*.jsonl")),
        consumed=consumed,
        command="python scripts/figures/make_fig1.py",
        outputs=outputs,
        notes=[
            "Strata A (seed 0) and B (seed 1) only; K=100 throughout.",
            "Uncentred panel plots within-head change only. The two objectives'"
            " uncentred baselines sit on opposite sides of zero (CE +0.496 to"
            " +0.541, ArcFace -0.899 to -0.889 at epoch 0), so a level"
            " comparison on a shared axis is not drawn.",
            "No imputation: the seed-0 ArcFace identity-00524 cell never reaches"
            " 0/10 and receives no attainment marker.",
            "No uncertainty bars: identities are a within-seed factor.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
