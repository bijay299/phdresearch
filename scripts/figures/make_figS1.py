#!/usr/bin/env python
"""
Figure S1 (supplementary) -- stratified centred-NC3 reversal counts.

SUPPLEMENTARY AND CONTEXTUAL ONLY. Protocol comparability across these settings
has not been demonstrated, so this figure may not appear as a main result and
may not be read as a dataset effect. Claim addressed: C2, descriptively.

Source: `evidence/run_inventory/clfonly_cell_inventory.csv` (columns `stratum`,
`head`, `centred_reversal_ep1`, `centred_reversal_any_epoch`), itself built from
the per-cell trajectories named in its `source_artifact` column.

Stratum F is EXCLUDED: its five cells are repeated observations of stratum C and
E fc0 cells under different doses, so including them would double-count. There
is no pooled headline rate anywhere in this figure.

    python scripts/figures/make_figS1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402

from figlib import (  # noqa: E402
    OBJECTIVE_COLOUR, OBJECTIVE_LABEL, REPO, apply_style, footnote, save,
    write_provenance,
)

FIGURE_ID = "figS1_reversal_by_stratum"
INVENTORY = REPO / "evidence" / "run_inventory" / "clfonly_cell_inventory.csv"

# Core-first ordering. Each label carries its own incompatibilities inline --
# not in a footnote -- because the whole point of the figure is that these rows
# are not on one axis.
STRATA = [
    ("A controlled nested sweep",
     "A · nested class-count sweep, seed 0\n"
     "faces-nested · K = 100/250/1000 · 112×112 · 10 test img/identity\n"
     "CONTROLLED dose: m = 9 active forget steps/epoch, λ = 1, per-cell reseed"),
    ("B seed-1 replication",
     "B · same sweep at K = 100, seed 1\n"
     "faces-nested · K = 100 · 112×112 · 10 test img/identity\n"
     "CONTROLLED dose: m = 9, λ = 1, per-cell reseed"),
    ("C historical faces-1000",
     "C · historical faces-1000, seed 0\n"
     "same source directory, different forget ids · K = 1000 · 112×112 · 10 "
     "test img/identity\nUNCONTROLLED dose: forget term on every retain step, "
     "λ = 1"),
    ("D historical faces-100",
     "D · historical faces-100, seed 0\n"
     "SEPARATE lineage: different source directory, min_images = 100, "
     "extraction seed UNRECORDED\nK = 100 · 112×112 · 25–31 test img/identity · "
     "UNCONTROLLED dose"),
    ("E CIFAR-10 context",
     "E · CIFAR-10 context, seeds 0 and 1\n"
     "different domain · K = 10 · 32×32 · ~1000 test img/class\n"
     "UNCONTROLLED dose: forget term on every retain step, λ = 1"),
]
SEG = [
    ("reversed by epoch 1", 0.95),
    ("reversed only by epoch 3", 0.55),
    ("no reversal at any recorded epoch", 0.12),
]


def main() -> int:
    apply_style()
    rows = [r for r in csv.DictReader(open(INVENTORY))
            if not r["stratum"].startswith("F")]
    excluded = sum(1 for r in csv.DictReader(open(INVENTORY))
                   if r["stratum"].startswith("F"))
    assert len(rows) == 64, f"expected 64 non-F cells, got {len(rows)}"

    bars, labels, consumed = [], [], {"rows": {}, "stratum_F_excluded": excluded}
    for stratum, label in STRATA:
        for head in ("ce", "arcface"):
            sub = [r for r in rows if r["stratum"] == stratum and r["head"] == head]
            if not sub:
                continue
            n = len(sub)
            ep1 = sum(r["centred_reversal_ep1"] == "True" for r in sub)
            any_ = sum(r["centred_reversal_any_epoch"] == "True" for r in sub)
            assert any_ >= ep1, f"{stratum}/{head}: epoch-1 reversals exceed any"
            bars.append((head, ep1, any_ - ep1, n - any_, n, any_))
            labels.append((label if head == "ce" else "", head))
            consumed["rows"][f"{stratum} | {head}"] = {
                "n": n, "reversed_by_epoch1": ep1,
                "reversed_only_by_epoch3": any_ - ep1,
                "no_reversal": n - any_,
            }

    fig, ax = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
    ys = list(range(len(bars)))[::-1]
    for y, (head, ep1, later, none_, n, any_) in zip(ys, bars):
        base = OBJECTIVE_COLOUR[head]
        left = 0.0
        for value, (_, alpha) in zip((ep1, later, none_), SEG):
            if value:
                ax.barh(y, value, left=left, height=0.62, color=base,
                        alpha=alpha, edgecolor="white", linewidth=0.8, zorder=3)
            left += value
        ax.text(n + 0.28, y, f"{any_}/{n}", va="center", ha="left",
                fontsize=7.6, weight="bold", color=base)

    ax.set_yticks(ys)
    ax.set_yticklabels(
        [f"{OBJECTIVE_LABEL[h]}" + (f"\n{lab}" if lab else "")
         for lab, h in labels],
        fontsize=6.3, linespacing=1.45,
    )
    for tick, (lab, h) in zip(ax.get_yticklabels(), labels):
        tick.set_color(OBJECTIVE_COLOUR[h] if not lab else "0.15")
    ax.set_xlim(0, 13.4)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("number of classifier-only unlearning cells")
    ax.set_ylim(-0.7, len(bars) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Centred NC3 sign reversals by stratum and objective — "
        "SUPPLEMENTARY, CONTEXTUAL ONLY\n"
        "These settings are not protocol-comparable; the rows are not points on "
        "one axis and are never pooled into an overall rate.",
        fontsize=8.0, linespacing=1.5,
    )
    for y in range(1, len(bars) // 2):
        ax.axhline(2 * y - 0.5, color="0.86", linewidth=0.7, zorder=1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.35", alpha=a, label=lab)
        for lab, a in SEG
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, -0.028),
               ncol=3, frameon=False, fontsize=7.0, handlelength=1.6,
               columnspacing=1.8,
               title="reversal is defined at every recorded epoch, so no cell "
                     "is missing from a row")

    footnote(fig, [
        "Reversal = a sign change of centred NC3 for the same class, same "
        "convention, from that cell's own epoch-0 baseline. 'By epoch 1' counts "
        "cells whose sign had already changed at the first unlearning epoch; "
        "'only by epoch 3' counts cells that first change sign at epoch 2 or 3.",
        f"Stratum F is excluded: its {excluded} cells are repeated observations "
        "of stratum C and E fc0 cells under different doses, and including them "
        "would double-count. No pooled rate across strata is computed or shown — "
        "the strata differ simultaneously in domain, input resolution, class "
        "count, dose protocol and test-set resolution.",
        "The row ordering is not a dataset effect, not a class-count effect and "
        "not an effect of the objective; CIFAR-10 and the face lineages are not "
        "two points on one axis. The most controlled comparison in the project "
        "is the nested class-count sweep (strata A and B), which shows no "
        "reversals at any K — though it too moves a bundle of correlated "
        "quantities with K, not K alone.",
        "No inferential claim is supported by the present analysis: these are "
        "counts of cells, and within a stratum the identities share a backbone "
        "per head, a split and a baseline checkpoint.",
    ], y=-0.125)

    outputs = save(fig, FIGURE_ID)
    write_provenance(
        FIGURE_ID, sources=[INVENTORY], consumed=consumed,
        command="python scripts/figures/make_figS1.py", outputs=outputs,
        notes=[
            "Supplementary and contextual only; not a main result.",
            "64 cells plotted; 5 stratum-F repeated observations excluded.",
            "No pooled headline rate is computed or displayed.",
            "Each row label carries its own protocol incompatibilities inline.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
