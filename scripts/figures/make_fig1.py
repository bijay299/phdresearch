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

LAYOUT NOTE. Attainment used to be marked by nested rings stacked on (epoch, 0),
which collided whenever several cells reached 0/10 at the same epoch -- which is
most of them. It is now a compact identity x objective strip under each seed's
row, showing the first recorded zero epoch per cell and marking non-attainment
explicitly. The trajectories themselves are unchanged.

    python scripts/figures/make_fig1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from figlib import (  # noqa: E402
    ANCHOR_RULE, IDENTITY_LINESTYLE, IDENTITY_MARKER, IDENTITY_NAME,
    OBJECTIVE_COLOUR, OBJECTIVE_LABEL, REPO, WIDTH, ZERO_LINE, apply_style,
    identity_handle, save, write_provenance,
)

FIGURE_ID = "fig1_k100_trajectories"
TABLE = REPO / "evidence" / "k100_seed1" / "tables" / "trajectories_both_seeds.csv"
TRAJ_DIR = REPO / "evidence" / "k100_seed1" / "trajectories"
CLASSES = [0, 29, 60, 95]
HEADS = ("ce", "arcface")
# Symmetric on purpose: the panel's content is that every recorded change is
# negative, and a one-sided window would make that unreadable as a sign claim.
UNCENTRED_LIM = 0.42

CAPTION = (
    "Classifier-only random-label unlearning at K = 100: all four forget "
    "identities, both objectives, at two pipeline seeds (m = 9 active forget "
    "steps per epoch, lambda = 1, three epochs, backbone frozen in eval mode "
    "throughout, so no backbone parameter or BatchNorm buffer is updated during "
    "unlearning). Left: forget-class test accuracy over the 10 held-out images "
    "of that identity. Centre: centred NC3 for the forget class, with the "
    "forget class excluded from the centring reference; no cell crosses zero at "
    "either seed, and there are no sign reversals across any of the 16 cells. "
    "Right: uncentred NC3 as a change from each cell's own epoch 0, because the "
    "two objectives' uncentred baselines sit on opposite sides of zero (CE "
    "+0.4963 to +0.5409, ArcFace -0.8990 to -0.8892 at epoch 0); plotting the "
    "change rather than the level is a presentation choice following from that, "
    "not a bar on descriptive comparison of the baselines themselves. The strip "
    "beneath each seed gives the first epoch at which each cell reached 0 of 10; "
    "15 of the 16 cells reach it within the three-epoch budget, and ArcFace "
    "identity 00524 at seed 0 does not (1 of 10 at epoch 3), which is marked "
    "'none' and is not imputed, substituted or interpolated. The dashed rule at "
    "epoch 1 marks the decomposition anchor used in Figure 2; epochs 2 and 3 are "
    "sensitivity. Lines are individual identities, which within a seed share one "
    "backbone per head, one train/test split and one baseline checkpoint. They "
    "are a within-seed factor rather than replicates, so no dispersion is shown "
    "and none would be interpretable. Zero output accuracy is not representation "
    "erasure: the backbone is frozen here, so these cells do not test whether "
    "the identity remains recoverable from the features, and NC3 does not "
    "measure unlearning quality."
)


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


def draw_strip(ax, cells, seed, consumed):
    """A compact identity x objective table of first-zero epochs."""
    ax.set_xlim(-0.9, len(CLASSES))
    ax.set_ylim(-0.5, 2.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-0.85, 1.92, "First epoch at 0 of 10",
            fontsize=8.0, va="center", ha="left", color="0.25")
    for col, fc in enumerate(CLASSES):
        ax.text(col + 0.5, 1.28, IDENTITY_NAME[fc], fontsize=8.0,
                ha="center", va="center", color="0.15")
    for row, head in enumerate(HEADS):
        y = 0.62 - row * 0.62
        ax.text(-0.06, y, OBJECTIVE_LABEL[head], fontsize=8.0, ha="right",
                va="center", color=OBJECTIVE_COLOUR[head])
        for col, fc in enumerate(CLASSES):
            fz = first_zero(cells[(seed, head, fc)])
            consumed["first_zero_epoch"][f"seed{seed}/{head}_fc{fc}"] = fz
            attained = fz is not None
            ax.add_patch(Rectangle(
                (col + 0.24, y - 0.26), 0.52, 0.52,
                facecolor=OBJECTIVE_COLOUR[head] if attained else "white",
                alpha=0.16 if attained else 1.0,
                edgecolor=OBJECTIVE_COLOUR[head] if attained else "0.55",
                linewidth=0.7, linestyle="-" if attained else (0, (2, 2)),
                zorder=2,
            ))
            ax.text(col + 0.5, y, str(fz) if attained else "none",
                    fontsize=8.0 if attained else 7.5, ha="center", va="center",
                    color=OBJECTIVE_COLOUR[head] if attained else "0.35",
                    weight="bold" if attained else "normal",
                    style="normal" if attained else "italic", zorder=3)
            if not attained:
                consumed["not_attained"].append(f"seed{seed}/{head}_fc{fc}")


def main() -> int:
    apply_style()
    cells = load()
    assert len(cells) == 16, f"expected 16 cells, got {len(cells)}"
    consumed = {"cells": [], "first_zero_epoch": {}, "not_attained": []}

    fig = plt.figure(figsize=(WIDTH, 6.3), constrained_layout=True)
    # Column 0 is a gutter holding the seed label, so the seed is named wholly
    # outside every plotting region rather than annotated inside a panel.
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 0.30, 1.0, 0.30],
                          width_ratios=[0.075, 1, 1, 1], hspace=0.03)
    axes = [[fig.add_subplot(gs[r, c]) for c in (1, 2, 3)] for r in (0, 2)]
    strips = [fig.add_subplot(gs[r, 1:]) for r in (1, 3)]
    for i, seed in enumerate((0, 1)):
        g = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
        g.axis("off")
        g.text(0.5, 0.55, f"Seed {seed}", rotation=90, ha="center", va="center",
               fontsize=9.5, weight="bold", color="0.15")

    for row, seed in enumerate((0, 1)):
        for col in range(3):
            axes[row][col].axvline(1, **ANCHOR_RULE)
        for head in HEADS:
            for fc in CLASSES:
                rows = cells[(seed, head, fc)]
                consumed["cells"].append(f"seed{seed}/{head}_fc{fc}")
                ep = [int(r["epoch"]) for r in rows]
                style = dict(
                    color=OBJECTIVE_COLOUR[head], linestyle=IDENTITY_LINESTYLE[fc],
                    marker=IDENTITY_MARKER[fc], markersize=3.6,
                    markerfacecolor="white", markeredgewidth=0.9, alpha=0.95,
                )
                assert all(int(r["forget_total"]) == 10 for r in rows)
                axes[row][0].plot(
                    ep, [int(r["forget_correct"]) for r in rows], **style)
                axes[row][1].plot(
                    ep, [float(r["nc3_centred"]) for r in rows], **style)
                axes[row][2].plot(
                    ep, [float(r["d_nc3_uncentred"]) for r in rows], **style)

        a0, a1, a2 = axes[row]
        a0.set_ylim(-0.7, 10.7)
        a0.set_yticks(range(0, 11, 2))
        a0.set_ylabel("correct out of 10")

        a1.set_ylim(-1.0, 1.0)
        a1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        a1.axhline(0.0, **ZERO_LINE)
        a1.set_ylabel("NC3 centred (cosine)\nforget class out of centring ref.",
                      fontsize=7.8)

        a2.set_ylim(-UNCENTRED_LIM, UNCENTRED_LIM)
        a2.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        a2.axhline(0.0, **ZERO_LINE)
        a2.set_ylabel("change in NC3 uncentred\nfrom that cell's own epoch 0", fontsize=7.8)

        for ax in axes[row]:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlim(-0.2, 3.2)
            ax.set_xlabel("unlearning epoch")

        draw_strip(strips[row], cells, seed, consumed)

    axes[0][0].set_title("Output forgetting")
    axes[0][1].set_title("NC3, centred")
    axes[0][2].set_title("NC3, uncentred")
    axes[0][2].text(1.06, 0.975, "epoch-1 anchor",
                    transform=axes[0][2].get_xaxis_transform(),
                    ha="left", va="top", fontsize=7.5, color="0.45")

    handles = [
        plt.Line2D([], [], color=OBJECTIVE_COLOUR[h], linewidth=2.0,
                   label=OBJECTIVE_LABEL[h]) for h in HEADS
    ] + [identity_handle(fc) for fc in CLASSES]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.4,
               title="objective (colour) · forget identity (line style and marker)")

    outputs = save(fig, FIGURE_ID)
    consumed["panels"] = {
        "col1": "forget_correct out of forget_total (denominator 10, verified)",
        "col2": "nc3_centred",
        "col3": "d_nc3_uncentred (change from that cell's own epoch 0)",
        "strip": "first post-baseline epoch with 0/10 correct, per cell",
    }
    consumed["n_cells"] = len(cells)
    consumed["rows_read"] = 64
    write_provenance(
        FIGURE_ID,
        sources=[TABLE] + sorted(TRAJ_DIR.rglob("*.jsonl")),
        consumed=consumed,
        command="python scripts/figures/make_fig1.py",
        outputs=outputs,
        caption=CAPTION,
        notes=[
            "Strata A (seed 0) and B (seed 1) only; K=100 throughout.",
            "Attainment is shown in a per-seed identity x objective strip, not "
            "as markers on the trajectory, because several cells reach 0/10 at "
            "the same epoch and stacked markers were unreadable.",
            "No imputation: the seed-0 ArcFace identity-00524 cell is marked "
            "'none' rather than given a substituted epoch.",
            "No uncertainty bars: identities are a within-seed factor.",
            "Explanatory prose and caveats live in the caption, not the image.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
