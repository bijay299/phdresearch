#!/usr/bin/env python
"""
Figure S1 (supplementary) -- stratified centred-NC3 reversal counts.

Supplementary and contextual only. Protocol comparability across these settings
has not been demonstrated, so this figure may not appear as a main result and
may not be read as a dataset effect. Claim addressed: C2, descriptively.

Source: `evidence/run_inventory/clfonly_cell_inventory.csv` (columns `stratum`,
`head`, `centred_reversal_ep1`, `centred_reversal_any_epoch`), itself built from
the per-cell trajectories named in its `source_artifact` column.

Stratum F is excluded: its five cells are repeated observations of stratum C and
E fc0 cells under different doses, so including them would double-count. There
is no pooled rate anywhere in this figure.

LAYOUT NOTE. The plot carries short stratum labels; the protocol differences
that make these strata non-comparable are in the compact table beneath the
chart, with the remaining detail in the caption.

    python scripts/figures/make_figS1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figlib import (  # noqa: E402
    OBJECTIVE_COLOUR, OBJECTIVE_LABEL, REPO, WIDTH, apply_style, save,
    write_provenance,
)

FIGURE_ID = "figS1_reversal_by_stratum"
INVENTORY = REPO / "evidence" / "run_inventory" / "clfonly_cell_inventory.csv"

# Short labels in the plot; the differences live in the table below it.
STRATA = [
    ("A controlled nested sweep", "A · nested sweep, seed 0"),
    ("B seed-1 replication", "B · nested sweep, seed 1"),
    ("C historical faces-1000", "C · faces-1000"),
    ("D historical faces-100", "D · faces-100"),
    ("E CIFAR-10 context", "E · CIFAR-10"),
]
SEG = [
    ("reversed at epoch 1", 0.95),
    ("first reversed at epoch 2 or 3", 0.55),
    ("no reversal at any recorded epoch", 0.12),
]
# stratum, lineage, K, input, test images per forget class, dose protocol
PROTOCOL = [
    ("A", "faces-nested", "100 / 250 / 1000", "112×112", "10",
     "controlled (m = 9, λ = 1, reseed)"),
    ("B", "faces-nested (as A)", "100", "112×112", "10",
     "controlled (m = 9, λ = 1, reseed)"),
    ("C", "faces-1000", "1000", "112×112", "10",
     "uncontrolled (every retain step)"),
    ("D", "faces-100 (separate lineage)", "100", "112×112", "25–31",
     "uncontrolled (every retain step)"),
    ("E", "CIFAR-10", "10", "32×32", "~1000",
     "uncontrolled (every retain step)"),
]
COLX = [0.000, 0.030, 0.285, 0.445, 0.545, 0.655]
COLHEAD = ["", "lineage", "K", "input", "test/class", "dose protocol"]

CAPTION = (
    "Centred NC3 sign reversals by stratum and objective, counted at epoch 1 and "
    "by epoch 3. Supplementary and contextual only. A reversal is a sign change "
    "of centred NC3 for the same class and the same convention, measured from "
    "that cell's own epoch-0 baseline; it is defined at every recorded epoch, so "
    "no cell is missing from a row. The fraction printed at the end of each row "
    "is the number of cells reversed at any recorded epoch over the total number "
    "of cells in that row. Reversal frequency ranges from 8 of 8 for CIFAR-10 "
    "under ArcFace to 0 of 12 for the controlled nested sweep under ArcFace. The "
    "table beneath the chart gives the protocol differences that separate these "
    "strata: they differ at the same time in domain, input resolution, class "
    "count, the number of test images per forget class, and the dose protocol. "
    "Protocol comparability has not been demonstrated, the rows are therefore "
    "not points on a single axis, and they are never pooled into an overall "
    "rate; no pooled figure is computed or shown anywhere. The ordering of rows "
    "is not a dataset effect, not a class-count effect and not an effect of the "
    "objective, and CIFAR-10 and the face lineages are not two points on one "
    "axis. Stratum D uses a different source directory and a different minimum "
    "image count from the other face lineages, and its extraction seed was not "
    "recorded, so it cannot be regenerated from the repository alone. The most "
    "controlled comparison in the project is the nested class-count sweep, "
    "strata A and B, which shows no reversals at any K; that sweep too moves a "
    "bundle of correlated quantities with K rather than K alone. Five further "
    "cells, forming stratum F, are excluded because they are repeated "
    "observations of the stratum C and E cells at forget class 0 under different "
    "doses, and including them would double-count. 64 cells are plotted. No "
    "inferential claim is supported by the present analysis: these are counts of "
    "cells, and within a stratum the identities share a backbone per objective, "
    "a split and a baseline checkpoint."
)


def draw_table(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    top, step = 0.855, 0.165
    ax.text(0, 1.02, "Protocol differences between strata — not "
            "protocol-comparable, never pooled",
            fontsize=8.0, ha="left", va="top", color="0.15")
    for x, head in zip(COLX, COLHEAD):
        if head:
            ax.text(x, top, head, fontsize=7.5, ha="left", va="top",
                    color="0.35", style="italic")
    ax.plot([0, 1], [top - 0.065, top - 0.065], color="0.80", linewidth=0.7,
            clip_on=False)
    for i, row in enumerate(PROTOCOL):
        y = top - 0.100 - i * step
        for x, cell in zip(COLX, row):
            ax.text(x, y, cell, fontsize=7.5, ha="left", va="top", color="0.20")


def main() -> int:
    apply_style()
    all_rows = list(csv.DictReader(open(INVENTORY)))
    rows = [r for r in all_rows if not r["stratum"].startswith("F")]
    excluded = len(all_rows) - len(rows)
    assert len(rows) == 64, f"expected 64 non-F cells, got {len(rows)}"

    bars, labels = [], []
    consumed = {"rows": {}, "stratum_F_excluded": excluded}
    for stratum, short in STRATA:
        for head in ("ce", "arcface"):
            sub = [r for r in rows if r["stratum"] == stratum and r["head"] == head]
            if not sub:
                continue
            n = len(sub)
            ep1 = sum(r["centred_reversal_ep1"] == "True" for r in sub)
            any_ = sum(r["centred_reversal_any_epoch"] == "True" for r in sub)
            assert any_ >= ep1, f"{stratum}/{head}: epoch-1 reversals exceed any"
            bars.append((head, ep1, any_ - ep1, n - any_, n, any_))
            labels.append((short if head == "ce" else "", head))
            consumed["rows"][f"{stratum} | {head}"] = {
                "n": n, "reversed_by_epoch1": ep1,
                "reversed_only_by_epoch3": any_ - ep1,
                "no_reversal": n - any_,
            }

    fig = plt.figure(figsize=(WIDTH, 5.5), constrained_layout=True)
    # Subfigures rather than one gridspec: constrained layout aligns axes boxes
    # within a column, so a table sharing a column with the chart would inherit
    # the chart's narrowed box and lose the width its columns need.
    top_sf, bot_sf = fig.subfigures(2, 1, height_ratios=[1.0, 0.46])
    ax = top_sf.subplots()
    draw_table(bot_sf.subplots())

    ys = list(range(len(bars)))[::-1]
    for y, (head, ep1, later, none_, n, any_) in zip(ys, bars):
        base, left = OBJECTIVE_COLOUR[head], 0.0
        for value, (_, alpha) in zip((ep1, later, none_), SEG):
            if value:
                ax.barh(y, value, left=left, height=0.62, color=base,
                        alpha=alpha, edgecolor="white", linewidth=0.8, zorder=3)
            left += value
        ax.text(n + 0.25, y, f"{any_}/{n}", va="center", ha="left",
                fontsize=8.0, weight="bold", color=base)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{short}\n{OBJECTIVE_LABEL[h]}" if short else OBJECTIVE_LABEL[h]
         for short, h in labels], fontsize=8.0, linespacing=1.35)
    for tick, (short, h) in zip(ax.get_yticklabels(), labels):
        tick.set_color("0.15" if short else OBJECTIVE_COLOUR[h])
    ax.set_xlim(0, 13.2)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("number of classifier-only unlearning cells\n"
                  "k / n  =  reversed at any recorded epoch / total cells",
                  fontsize=8.2)
    ax.set_ylim(-0.7, len(bars) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title("Centred NC3 sign reversals by stratum and objective "
                 "(supplementary, contextual only)", fontsize=9.0)
    for y in range(1, len(bars) // 2):
        ax.axhline(2 * y - 0.5, color="0.86", linewidth=0.7, zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=a, label=lab)
               for lab, a in SEG]
    bot_sf.legend(handles=handles, loc="outside lower center", ncol=3,
                  frameon=False, fontsize=8.0, handlelength=1.6,
                  columnspacing=2.0)

    outputs = save(fig, FIGURE_ID)
    write_provenance(
        FIGURE_ID, sources=[INVENTORY], consumed=consumed,
        command="python scripts/figures/make_figS1.py", outputs=outputs,
        caption=CAPTION,
        notes=[
            "Supplementary and contextual only; not a main result.",
            "64 cells plotted; 5 stratum-F repeated observations excluded.",
            "No pooled rate is computed or displayed.",
            "Row labels are short; the protocol differences are in the table "
            "beneath the chart and in the caption.",
            "The printed fraction is reversals at any recorded epoch over the "
            "row total; epoch-1 reversals remain a separate bar segment from "
            "those first appearing at epoch 2 or 3.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
