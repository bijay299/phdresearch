#!/usr/bin/env python
"""
Shared plotting conventions and provenance machinery for the paper figures.

WHY THIS EXISTS
---------------
Every figure in `notes/figure_specs.md` is drawn from already-recorded
artifacts. Nothing here trains, infers, replays, or computes a new experimental
metric -- the scripts read recorded values, rearrange them, and render. Keeping
the style table, the colour assignment and the provenance sidecar in one module
is what makes "CE is this colour everywhere in the paper" a property of the
code rather than a thing to remember.

PROVENANCE CONTRACT
-------------------
Each figure script calls `write_provenance()` with the exact source files it
read, and this module hashes them at render time. Figures derived from
git-ignored artifact trees (Figure 2 reads `logs_classcount_decomposition/`)
therefore still carry a checkable SHA-256 of a source that is not itself in the
repository -- that is the whole point of the sidecar.

TARGET SIZE
-----------
Figures are laid out for a full manuscript column width of about 7 inches and
are rendered at that width, so the type sizes below are the sizes a reader
actually gets -- nothing is scaled down on placement. The floor is ~8 pt, with
7.5 pt reserved for dense in-panel annotation.

Long explanatory prose, caveats and research-management warnings belong in the
CAPTION, not in the image. `notes/figure_specs.md` holds the final caption for
each figure, and `write_provenance()` copies it into the sidecar so the caption
travels with the figure. What stays inside the frame: axis definitions, units,
legends, missing-outcome labels, and seed/epoch information.

WHAT THIS MODULE DELIBERATELY DOES NOT PROVIDE
----------------------------------------------
No error-bar, confidence-band or shaded-interval helper. The identities within
a seed share one backbone per head, one split and one baseline checkpoint; they
are a within-seed factor, not replicates, so a dispersion glyph over them would
assert sampling structure the design does not have. Means are drawn with
`descriptive_mean_marker()`, which is bare by construction.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "figures"

# Full manuscript width. Every figure is rendered at this width so that the
# point sizes in `apply_style` are the sizes the reader sees.
WIDTH = 7.0

# ---------------------------------------------------------------------------
# Colour and marker conventions
# ---------------------------------------------------------------------------
# Okabe-Ito: distinguishable under deuteranopia, protanopia and tritanopia, and
# separable in greyscale by luminance.
OBJECTIVE_COLOUR = {"ce": "#0072B2", "arcface": "#D55E00"}
OBJECTIVE_LABEL = {"ce": "CE", "arcface": "ArcFace"}

# Identity encoding is shared across every panel and every figure so the same
# person can be tracked between seeds and between figures.
IDENTITY_NAME = {0: "00001", 29: "00142", 60: "00284", 95: "00524"}
IDENTITY_MARKER = {0: "o", 29: "s", 60: "^", 95: "D"}
IDENTITY_LINESTYLE = {0: "-", 29: "--", 60: "-.", 95: ":"}
IDENTITY_COLOUR = {0: "#0072B2", 29: "#009E73", 60: "#CC79A7", 95: "#E69F00"}

# Seed is the only replication unit in the project, so it gets the fill channel.
SEED_FILLED = {0: True, 1: False}

ZERO_LINE = dict(color="0.35", linewidth=0.8, zorder=1)
# A thin rule, not a broad band: the band used previously sat on top of the
# observations at epoch 1, which is exactly where the data matter most.
ANCHOR_RULE = dict(color="0.55", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)


def apply_style() -> None:
    """Typography and rcParams tuned for single-column publication size."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "legend.title_fontsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "axes.grid": True,
            "grid.color": "0.90",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            # NOT "tight": a tight bbox resizes the canvas to whatever text
            # spills outside the axes, so the rendered width stops being the
            # width the type sizes were chosen for. Constrained layout keeps
            # everything inside the declared figure size instead, and figure
            # legends use loc="outside ..." so they get reserved space.
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,  # embed as TrueType, not Type 3
            "ps.fonttype": 42,
        }
    )


def identity_handle(fc: int, **kw):
    """A legend proxy for one identity, with no objective colour attached."""
    return plt.Line2D(
        [],
        [],
        color="0.35",
        marker=IDENTITY_MARKER[fc],
        linestyle=IDENTITY_LINESTYLE[fc],
        label=f"class {fc} = identity {IDENTITY_NAME[fc]}",
        **kw,
    )


def descriptive_mean_marker(ax, x, y, colour, label=None, **kw):
    """Draw a mean with NO dispersion glyph of any kind.

    Deliberately not a function that can take a `yerr`: the four identities in
    a seed are not replicates, so an error bar over them would be an inferential
    claim the design does not support.
    """
    return ax.plot(
        x, y, color=colour, marker="_", markersize=11, markeredgewidth=1.8,
        linestyle="none", label=label, zorder=5, **kw,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> dict:
    """The revision the figure was rendered at, and what "clean" means here.

    Two different commits matter and must not be conflated:

    * `rendered_at_commit` -- the revision of the plotting scripts and input
      tables that produced this image.
    * the **delivery commit** -- the commit that actually contains the rendered
      files. It is necessarily a *later* commit than `rendered_at_commit`,
      because the outputs cannot be committed until after they exist. It is not
      recorded here; it is the commit in which this sidecar first appears, and
      `verify_figures.py` checks that HEAD is `rendered_at_commit` or a
      descendant of it.

    The clean-status flag is deliberately partial and is named to say so: it
    excludes generated outputs under `figures/`, because rendering writes there
    and counting them would make the flag unconditionally dirty. It reports
    whether the *inputs to rendering* -- scripts, tables, evidence -- were
    committed.
    """
    def run(*args):
        return subprocess.run(
            args, cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    pending = [
        path
        for path in (
            line[2:].strip().strip('"')
            for line in run("git", "status", "--porcelain").splitlines()
        )
        if path and not path.startswith("figures/")
    ]
    return {
        "rendered_at_commit": run("git", "rev-parse", "HEAD"),
        "inputs_clean_excluding_generated_outputs_under_figures": not pending,
        "uncommitted_non_output_paths_at_render_time": sorted(pending)[:20],
        "delivery_commit": (
            "not recorded here: the commit containing these rendered files is a "
            "descendant of rendered_at_commit, since the outputs cannot be "
            "committed until after they exist. verify_figures.py checks that "
            "HEAD is rendered_at_commit or a descendant."
        ),
    }


def write_provenance(
    figure_id: str,
    sources: list[Path],
    consumed: dict,
    command: str,
    outputs: list[Path],
    caption: str,
    notes: list[str] | None = None,
) -> Path:
    """Write the sidecar: what was read, what was consumed, and the caption.

    The caption is stored here as well as in `notes/figure_specs.md` so that the
    prose a reader needs -- the caveats that were deliberately kept out of the
    image -- cannot drift away from the image it belongs to.
    """
    rec = {
        "figure_id": figure_id,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "render_width_inches": WIDTH,
        "execution_revision": git_revision(),
        "caption": caption,
        "plotting_command": command,
        "sources": [
            {
                "path": str(p.relative_to(REPO)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
                "tracked_in_git": _tracked(p),
            }
            for p in sources
        ],
        "consumed": consumed,
        "outputs": [
            {"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in outputs
        ],
        "notes": notes or [],
    }
    out = FIGDIR / f"{figure_id}.provenance.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    return out


def _tracked(path: Path) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return r.returncode == 0


def save(fig, figure_id: str) -> list[Path]:
    """Vector PDF for the paper, PNG preview for review.

    Asserts the canvas is still the declared manuscript width, so a layout
    change cannot silently shrink the effective type size.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    assert abs(w - WIDTH) < 1e-6, f"{figure_id}: width {w:.3f}in, expected {WIDTH}in"
    pdf = FIGDIR / f"{figure_id}.pdf"
    png = FIGDIR / f"{figure_id}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  {figure_id}: {w:.2f} x {h:.2f} in at manuscript size")
    return [pdf, png]
