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
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "figures"

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
ANCHOR_RULE = dict(color="0.72", linewidth=6.0, alpha=0.30, zorder=0)


def apply_style() -> None:
    """Typography and rcParams tuned for single-column publication size."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 7.5,
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
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
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
def footnote(fig, paragraphs, y=-0.11, width=134, fontsize=6.4):
    """Figure-level notes, hard-wrapped.

    `savefig(bbox_inches="tight")` grows the canvas to fit whatever text is
    placed outside the axes, so an unwrapped paragraph silently widens the
    figure and squashes the panels. Wrapping is not cosmetic here.

    Keep mathtext out of these: the wrapper breaks on spaces and would split a
    `$...$` span across lines.
    """
    lines = []
    for para in paragraphs:
        lines.extend(textwrap.wrap(para, width=width) or [""])
    fig.text(0.5, y, "\n".join(lines), ha="center", va="top",
             fontsize=fontsize, color="0.20", linespacing=1.55)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> dict:
    """The revision the figure was rendered at.

    `dirty` deliberately ignores anything under `figures/`: rendering writes
    there, so including it would make the flag unconditionally true and
    therefore useless. What matters for reproducibility is whether the scripts
    and the input tables were clean, which is what this reports.
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
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty_excluding_figure_outputs": bool(pending),
        "pending_paths": sorted(pending)[:20],
    }


def write_provenance(
    figure_id: str,
    sources: list[Path],
    consumed: dict,
    command: str,
    outputs: list[Path],
    notes: list[str] | None = None,
) -> Path:
    """Write the sidecar: what was read, with hashes, and what was consumed."""
    rec = {
        "figure_id": figure_id,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "execution_revision": git_revision(),
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
    """Vector PDF for the paper, PNG preview for review."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGDIR / f"{figure_id}.pdf"
    png = FIGDIR / f"{figure_id}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return [pdf, png]
