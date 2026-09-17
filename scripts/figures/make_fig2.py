#!/usr/bin/env python
"""
Figure 2 -- seed-0 classifier rotation, norm ratio, and the reference-frame
decomposition of the centred difference-in-changes.

Claims addressed: C6a (rotation and norm) and C6b (the K ordering is a
reference-frame effect). SEED 0 ONLY -- no seed-1 decomposition exists.

Source: `evidence/decomposition_fig2/fig2_plotting_inputs.csv`, extracted from
the git-ignored `logs_classcount_decomposition/decomposition.json` by
`scripts/figures/extract_fig2_inputs.py`, which records the source hash. The
figure reads the tracked table so a reviewer can check it without the artifact
tree; the `--from-artifact` flag reads the artifact directly and asserts the two
agree exactly.

Nothing here recomputes the decomposition. All four corner quantities were
produced by `scripts/nc3_decomposition.py` under its own validation gates.

    python scripts/figures/make_fig2.py
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
    ZERO_LINE, apply_style, descriptive_mean_marker, footnote,
    identity_handle, save, write_provenance,
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

# K=500 passed neither head-fairness gate, so it carries no unlearning cells.
# Read back from the baselines at render time rather than transcribed.
K500_GATE_PP = 2.00

# |g_0| / |f_0| for CE, in K order -- the feature-side centre relative to the
# forget class mean. Recorded in the 2026-09-16 decomposition entry.
CE_FEATURE_CENTRE_RATIO = {100: 0.80, 250: 0.73, 1000: 0.69}


def load_table(path: Path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    **{k: r[k] for k in ("cell_name", "head", "identity")},
                    "K": int(r["K"]),
                    "epoch": int(r["epoch"]),
                    "forget_class": int(r["forget_class"]),
                    **{
                        k: float(r[k])
                        for k in r
                        if k not in ("cell_name", "head", "identity", "K",
                                     "epoch", "forget_class", "seed",
                                     "n_centre_classes", "n_present_classes")
                    },
                }
            )
    return rows


def load_artifact():
    doc = json.loads(ARTIFACT.read_text())
    rows = []
    for c in doc["cells"]:
        for ep in ("1", "2", "3"):
            e = c["epochs"][ep]
            rows.append(
                {
                    "cell_name": c["name"], "head": c["head"], "K": c["K"],
                    "identity": c["forget_identity_name"],
                    "forget_class": c["forget_class"], "epoch": int(ep),
                    "forget_weight_angle_deg": e["forget_weight"]["angle_deg"],
                    "forget_weight_norm_ratio": e["forget_weight"]["norm_ratio"],
                    **{
                        k: e[k]
                        for k in ("delta_centred_production",
                                  "delta_centred_weight_only",
                                  "delta_centred_centre_only",
                                  "interaction_residual")
                    },
                }
            )
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
    return sorted(
        (r for r in rows if r["K"] == K and r["head"] == head
         and r["epoch"] == epoch),
        key=lambda r: r["forget_class"],
    )


def scatter_by_K(ax, rows, field, epoch, alpha, size):
    """Per-identity points, jittered in x only, plus a bare descriptive mean."""
    for head in HEADS:
        for K in KS:
            cells = sel(rows, K, head, epoch)
            vals = [c[field] for c in cells]
            # Jitter is multiplicative because the x-axis is logarithmic.
            for i, c in enumerate(cells):
                ax.plot(
                    [K * (1.0 + 0.020 * (i - 1.5))], [c[field]],
                    marker=IDENTITY_MARKER[c["forget_class"]],
                    color=OBJECTIVE_COLOUR[head], markersize=size,
                    markerfacecolor="none", markeredgewidth=1.0,
                    linestyle="none", alpha=alpha, zorder=3,
                )
            descriptive_mean_marker(ax, [K], [st.mean(vals)],
                                    OBJECTIVE_COLOUR[head], alpha=alpha)
        # Connect the means, but NEVER across the empty K=500 position: a
        # segment spanning 250 to 1000 would read as an observation at 500.
        for lo, hi in ((100, 250),):
            ax.plot(
                [lo, hi],
                [st.mean(c[field] for c in sel(rows, k, head, epoch))
                 for k in (lo, hi)],
                color=OBJECTIVE_COLOUR[head], linewidth=1.0, alpha=alpha * 0.8,
                zorder=2,
            )


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
        # Cell names repeat across K -- the same forget identity is unlearned at
        # every class count -- so K is part of the key.
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
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.24]})
    consumed = {"cells": 24, "epochs": [1, 3], "K": KS}

    # ---- Panel A: forget-weight rotation ---------------------------------
    ax = axes[0]
    scatter_by_K(ax, rows, "forget_weight_angle_deg", 3, 0.32, 4.2)
    scatter_by_K(ax, rows, "forget_weight_angle_deg", 1, 1.00, 4.6)
    ax.set_ylim(0, 25)
    ax.set_ylabel("forget-class weight rotation from epoch 0 (degrees)")
    ax.set_title("A · classifier rotation")

    # ---- Panel B: forget-weight norm ratio -------------------------------
    ax = axes[1]
    ax.axhline(1.0, **ZERO_LINE)
    scatter_by_K(ax, rows, "forget_weight_norm_ratio", 3, 0.32, 4.2)
    scatter_by_K(ax, rows, "forget_weight_norm_ratio", 1, 1.00, 4.6)
    ax.set_ylim(0.75, 1.05)
    ax.set_ylabel(r"forget-class weight norm ratio  $\|w_t\|/\|w_0\|$")
    ax.set_title("B · classifier norm")

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlim(78, 1380)
        ax.set_xticks(K_TICKS)
        ax.set_xticklabels([str(k) for k in K_TICKS])
        ax.minorticks_off()
        ax.set_xlabel("K (number of identities)")
        ax.axvspan(410, 610, color="0.93", zorder=0)
        ax.text(500, 0.965, "K = 500\nno cells",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.2, color="0.42", linespacing=1.3)

    # ---- Panel C: decomposition of the centred DiC at epoch 1 ------------
    ax = axes[2]
    quantities = [
        ("delta_centred_production", r"$\Delta_{\mathrm{production}}$", "#44546A"),
        ("delta_centred_weight_only", r"$\Delta_{\mathrm{weight-only}}$", "#0072B2"),
        ("delta_centred_centre_only", r"$\Delta_{\mathrm{centre-only}}$", "#56B4E9"),
        ("interaction_residual", "residual", "#BBBBBB"),
    ]
    width = 0.20
    dic = {}
    for qi, (field, label, colour) in enumerate(quantities):
        heights = []
        for K in KS:
            a = st.mean(c[field] for c in sel(rows, K, "arcface", 1))
            ce = st.mean(c[field] for c in sel(rows, K, "ce", 1))
            heights.append(a - ce)
        dic[field] = dict(zip(KS, heights))
        xs = [i + (qi - 1.5) * width for i in range(len(KS))]
        ax.bar(xs, heights, width=width * 0.92, color=colour, label=label,
               edgecolor="white", linewidth=0.4, zorder=3)
    consumed["panel_C_dic_epoch1"] = {
        f: {str(k): v for k, v in d.items()} for f, d in dic.items()
    }

    # The two small quantities are ~3 orders below production, so their bars are
    # correctly near-invisible at this scale. They are labelled with their
    # VALUES -- never as "0" -- in a block rather than as rotated bar labels,
    # which at this aspect ratio collide with the neighbouring group.
    small = "\n".join(
        f"{k:>4}  {dic['delta_centred_centre_only'][k]*1e3:+6.3f}"
        f"  {dic['interaction_residual'][k]*1e3:+6.3f}"
        for k in KS
    )
    ax.text(
        0.028, 0.735,
        "small but NON-ZERO, epoch 1 ($\\times 10^{-3}$)\n"
        "   K  centre-  residual\n      only\n" + small,
        transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
        family="DejaVu Sans Mono", color="0.20", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7",
                  edgecolor="0.80", linewidth=0.6),
    )
    ax.axhline(0.0, **ZERO_LINE)
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("K (number of identities)")
    ax.set_ylim(-0.014, 0.225)
    ax.set_ylabel(r"ArcFace $-$ CE difference-in-changes, centred NC3"
                  "\n" r"($\Delta$cosine, baseline-subtracted)")
    ax.set_title("C · reference-frame decomposition at epoch 1")
    ax.legend(loc="upper left", frameon=False, fontsize=6.6,
              handlelength=1.3, labelspacing=0.32)

    # Inset: CE feature-side centre magnitude, on its own axis and title.
    ins = ax.inset_axes([0.625, 0.675, 0.345, 0.285])
    ins.plot(range(len(KS)), [CE_FEATURE_CENTRE_RATIO[k] for k in KS],
             marker="o", markersize=3.2, color=OBJECTIVE_COLOUR["ce"],
             linewidth=1.0)
    ins.set_xticks(range(len(KS)))
    ins.set_xticklabels([str(k) for k in KS], fontsize=5.6)
    ins.set_ylim(0.6, 0.9)
    ins.set_yticks([0.6, 0.75, 0.9])
    ins.tick_params(labelsize=5.6, length=2, pad=1)
    ins.set_title(r"CE  $|g_0|/|f_0|$  (separate axis)", fontsize=5.9, pad=2)
    ins.grid(alpha=0.5)

    # ---- footnotes, stacked so nothing overlaps the legend ----------------
    handles = [
        plt.Line2D([], [], color=OBJECTIVE_COLOUR[h], linewidth=1.8,
                   label=OBJECTIVE_LABEL[h]) for h in HEADS
    ] + [
        plt.Line2D([], [], color="0.35", marker="_", markersize=11,
                   markeredgewidth=1.8, linestyle="none",
                   label="descriptive mean (no dispersion glyph: identities "
                         "are a within-seed factor)"),
        plt.Line2D([], [], color="0.35", linewidth=1.8, alpha=0.32,
                   label="epoch 3 (lighter); epoch 1 at full weight"),
    ] + [identity_handle(fc, markerfacecolor="none") for fc in (0, 29, 60, 95)]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.008), ncol=4, frameon=False,
               columnspacing=1.8, handlelength=2.4,
               title="seed 0 only — no seed-1 decomposition exists")

    footnote(fig, [
        "Panel C. The residual is production minus weight-only minus "
        "centre-only, exactly, per cell. The four corners are an arithmetic "
        "decomposition of the metric A(w, c) = cos(f0 - g0, w - c), not an "
        "intervention on training, and not a measure of unlearning quality.",
        "Panel B. Descriptive. A cosine is invariant to scaling its whole "
        "argument, so the uncentred cosine is unaffected by the weight norm; "
        "the centred cosine takes w - c, and changing w against a fixed c can "
        "change that direction. No attribution from these ratios is claimed.",
        f"K = 500 carries no unlearning cells. Baselines: CE "
        f"{gate['ce']*100:.4f} %, ArcFace {gate['arcface']*100:.4f} %, absolute "
        f"gap {gate['gap_pp']:.4f} pp, exceeding the prespecified "
        f"{K500_GATE_PP:.2f} pp head-fairness gate. The gate is symmetric and "
        "failed because ArcFace EXCEEDED CE, not because ArcFace "
        "underperformed. Its cells were never run and neither head was retuned "
        "to rescue it; both baselines are retained as artifacts.",
        "That is a deliberate exclusion under a prespecified rule, not missing "
        "data — no value is imputed and no line is drawn across the gap in any "
        "panel.",
    ], y=-0.145)

    outputs = save(fig, FIGURE_ID)
    consumed["k500_gate"] = gate
    consumed["panel_A_field"] = "forget_weight_angle_deg"
    consumed["panel_B_field"] = "forget_weight_norm_ratio"
    write_provenance(
        FIGURE_ID, sources=sources, consumed=consumed,
        command="python scripts/figures/make_fig2.py"
                + (" --from-artifact" if args.from_artifact else ""),
        outputs=outputs,
        notes=notes + [
            "Seed 0 only. Stratum A, 24 cells, K in {100, 250, 1000}.",
            "K=500 tick is drawn with an empty data region; no connecting line "
            "spans it in any panel.",
            "Panel C bars are baseline-subtracted counterfactual differences-"
            "in-changes of the metric, not causal contributions and not a "
            "measure of unlearning quality.",
            "Small bars are labelled with their values; none is labelled '0'.",
            "No error bars anywhere: identities within a seed share one "
            "backbone per head, one split and one baseline checkpoint.",
        ],
    )
    print(f"wrote {[str(p.relative_to(REPO)) for p in outputs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
