#!/usr/bin/env python
"""
Extract the Figure 2 plotting inputs from the git-ignored decomposition tree
into a small tracked table.

WHY THIS EXISTS
---------------
Figure 2 is the only figure whose source is not in the repository:
`logs_classcount_decomposition/decomposition.json` is an artifact tree and is
git-ignored like every other. A reviewer with the repo but not the machine
cannot check the plotted numbers against anything. This script writes the
subset of recorded values the figure actually consumes -- 24 cells x 3 epochs,
a few scalars each -- into `evidence/decomposition_fig2/`, with the source
hash, so the figure is checkable from the repository alone.

It copies recorded values. It does not recompute the decomposition, does not
read any `.npz` state, and does not train, infer or replay. The arithmetic in
`decomposition.json` was produced by `scripts/nc3_decomposition.py` under its
own validation gates; nothing here revisits that.

    python scripts/figures/extract_fig2_inputs.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import REPO, git_revision, sha256  # noqa: E402

SOURCE = REPO / "logs_classcount_decomposition" / "decomposition.json"
OUTDIR = REPO / "evidence" / "decomposition_fig2"

# Panels A and B read these; panel C reads the four decomposition quantities.
CELL_FIELDS = [
    "delta_centred_production",
    "delta_centred_weight_only",
    "delta_centred_centre_only",
    "interaction_residual",
    "delta_uncentred_production",
    "nc3_centred_forget",
    "nc3_centred_forget_epoch0",
    "nc3_uncentred_forget",
    "nc3_uncentred_forget_epoch0",
]
VECTOR_FIELDS = {
    "forget_weight": ("angle_deg", "norm_ratio"),
    "retain_centre": ("angle_deg", "norm_ratio"),
    "centred_forget_weight": ("angle_deg", "norm_ratio"),
}


def main() -> int:
    if not SOURCE.exists():
        print(f"MISSING SOURCE: {SOURCE}", file=sys.stderr)
        return 2
    doc = json.loads(SOURCE.read_text())
    cells = doc["cells"]
    if len(cells) != doc["n_cells"] or len(cells) != 24:
        print(f"UNEXPECTED CELL COUNT: {len(cells)} vs n_cells={doc['n_cells']}",
              file=sys.stderr)
        return 3

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        for ep in ("1", "2", "3"):
            e = cell["epochs"][ep]
            row = {
                "cell_name": cell["name"],
                "K": cell["K"],
                "head": cell["head"],
                "seed": cell["seed"],
                "forget_class": cell["forget_class"],
                "identity": cell["forget_identity_name"],
                "epoch": int(ep),
                "n_centre_classes": e["n_centre_classes"],
                "n_present_classes": e["n_present_classes"],
            }
            for f in CELL_FIELDS:
                row[f] = repr(e[f])
            for vec, subs in VECTOR_FIELDS.items():
                for s in subs:
                    row[f"{vec}_{s}"] = repr(e[vec][s])
            # Closure of the four-corner identity, recomputed here as a check on
            # the extraction, not as a re-derivation of the decomposition.
            resid = (
                e["delta_centred_production"]
                - e["delta_centred_weight_only"]
                - e["delta_centred_centre_only"]
            )
            if abs(resid - e["interaction_residual"]) > 1e-12:
                print(f"RESIDUAL IDENTITY FAILS for {cell['name']} ep{ep}",
                      file=sys.stderr)
                return 4
            rows.append(row)

    table = OUTDIR / "fig2_plotting_inputs.csv"
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    prov = {
        "purpose": "Figure 2 plotting inputs, copied verbatim from the "
                   "git-ignored decomposition artifact tree.",
        "extraction_revision": git_revision(),
        "source": {
            "path": str(SOURCE.relative_to(REPO)),
            "sha256": sha256(SOURCE),
            "bytes": SOURCE.stat().st_size,
            "tracked_in_git": False,
            "produced_by": "scripts/nc3_decomposition.py",
        },
        "extracted_rows": len(rows),
        "cells": len(cells),
        "epochs_per_cell": [1, 2, 3],
        "fields_copied": CELL_FIELDS
        + [f"{v}_{s}" for v, subs in VECTOR_FIELDS.items() for s in subs],
        "float_encoding": "Python repr() of the float read from the source; "
                          "round-trips exactly under float().",
        "checks_run": [
            "cell count == n_cells == 24",
            "delta_production - delta_weight_only - delta_centre_only == "
            "interaction_residual to 1e-12, per cell-epoch",
        ],
        "not_done": [
            "No recomputation of the decomposition from .npz state.",
            "No training, inference or replay.",
        ],
        "outputs": {
            "fig2_plotting_inputs.csv": sha256(table),
        },
    }
    (OUTDIR / "fig2_inputs_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n"
    )
    print(f"wrote {table.relative_to(REPO)} ({len(rows)} rows)")
    print(f"source sha256 {prov['source']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
