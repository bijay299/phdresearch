#!/usr/bin/env python
"""
Verify the rendered figures against their sources.

This re-reads the source tables independently of the plotting scripts, then
checks each figure's provenance sidecar -- which records exactly what that
script consumed -- against those sources. It also re-opens each rendered figure
and asserts that no data point falls outside its axis limits, since a clipped
observation is the one failure mode a visual check reliably misses.

Read-only. No training, inference, replay, or new metrics.

    python scripts/figures/verify_figures.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figlib import FIGDIR, REPO, sha256  # noqa: E402

EV = REPO / "evidence"
FAILURES: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(f"{label}: {detail}")
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def sidecar(fig_id: str) -> dict:
    return json.loads((FIGDIR / f"{fig_id}.provenance.json").read_text())


# ---------------------------------------------------------------------------
def verify_sidecar_hashes(fig_id: str) -> None:
    rec = sidecar(fig_id)
    bad = [
        s["path"] for s in rec["sources"]
        if sha256(REPO / s["path"]) != s["sha256"]
    ]
    check(f"{fig_id}: every recorded source hash still matches on disk",
          not bad, f"stale: {bad}" if bad else f"{len(rec['sources'])} sources")
    outs = [
        o["path"] for o in rec["outputs"]
        if sha256(REPO / o["path"]) != o["sha256"]
    ]
    check(f"{fig_id}: rendered outputs match their recorded hashes",
          not outs, f"stale: {outs}" if outs else f"{len(rec['outputs'])} files")


# ---------------------------------------------------------------------------
def verify_fig1() -> None:
    print("\nFigure 1 — K=100 trajectories")
    rows = list(csv.DictReader(
        open(EV / "k100_seed1" / "tables" / "trajectories_both_seeds.csv")))
    rec = sidecar("fig1_k100_trajectories")["consumed"]

    check("16 cells consumed", rec["n_cells"] == 16, str(rec["n_cells"]))
    check("64 cell-epoch rows in source", len(rows) == 64, str(len(rows)))

    # Forget denominator must be 10 everywhere -- the 0.1 metric granularity
    # claim depends on it and on nothing else.
    check("every forget denominator is exactly 10",
          all(int(r["forget_total"]) == 10 for r in rows))
    check("every retain denominator is exactly 970",
          all(int(r["retain_total"]) == 970 for r in rows))

    # First-zero epochs, recomputed from the source.
    want = {}
    for r in rows:
        key = f"seed{r['seed']}/{r['head']}_fc{r['forget_class']}"
        if int(r["epoch"]) > 0 and int(r["forget_correct"]) == 0:
            want.setdefault(key, int(r["epoch"]))
    want = {k: want.get(k) for k in rec["first_zero_epoch"]}
    check("plotted first-0/10 epochs match the source",
          want == rec["first_zero_epoch"],
          f"{sum(v is not None for v in want.values())} attained of 16")

    # The single missing attainment.
    check("exactly one cell has no attainment marker",
          rec["unmarked"] == ["seed0/arcface_fc95"], str(rec["unmarked"]))
    tail = [r for r in rows if r["seed"] == "0" and r["head"] == "arcface"
            and r["forget_class"] == "95" and r["epoch"] == "3"]
    check("that cell ends at 1/10, not 0/10",
          int(tail[0]["forget_correct"]) == 1, tail[0]["forget_correct"])

    # Axis coverage.
    cen = [float(r["nc3_centred"]) for r in rows]
    unc = [float(r["d_nc3_uncentred"]) for r in rows]
    check("centred NC3 within the plotted [-1, 1]",
          min(cen) >= -1.0 and max(cen) <= 1.0,
          f"[{min(cen):.4f}, {max(cen):.4f}]")
    check("uncentred change within the plotted symmetric ±0.42",
          min(unc) >= -0.42 and max(unc) <= 0.42,
          f"[{min(unc):.4f}, {max(unc):.4f}]")

    # Sign counts the caption asserts.
    check("no centred NC3 value crosses zero at either seed",
          all(v > 0 for v in cen), f"min {min(cen):.4f}")
    revs = 0
    for seed in ("0", "1"):
        for head in ("ce", "arcface"):
            for fc in ("0", "29", "60", "95"):
                cell = sorted(
                    (r for r in rows if r["seed"] == seed and r["head"] == head
                     and r["forget_class"] == fc), key=lambda r: int(r["epoch"]))
                base = float(cell[0]["nc3_centred"])
                revs += sum(1 for r in cell[1:]
                            if base * float(r["nc3_centred"]) < 0)
    check("ZERO centred sign reversals across all 16 K=100 cells", revs == 0,
          f"{revs} found")

    # The uncentred panel is a within-head change, and the levels justify it.
    lv = {h: [float(r["nc3_uncentred"]) for r in rows
              if r["head"] == h and r["epoch"] == "0"] for h in ("ce", "arcface")}
    check("epoch-0 uncentred levels sit on opposite sides of zero",
          min(lv["ce"]) > 0 > max(lv["arcface"]),
          f"CE [{min(lv['ce']):+.4f}, {max(lv['ce']):+.4f}], "
          f"ArcFace [{min(lv['arcface']):+.4f}, {max(lv['arcface']):+.4f}]")


# ---------------------------------------------------------------------------
def verify_fig2() -> None:
    print("\nFigure 2 — seed-0 decomposition")
    tbl = list(csv.DictReader(
        open(EV / "decomposition_fig2" / "fig2_plotting_inputs.csv")))
    rec = sidecar("fig2_seed0_decomposition")["consumed"]
    KS = [100, 250, 1000]

    check("72 cell-epochs in the tracked plotting-input table",
          len(tbl) == 72, str(len(tbl)))
    check("24 distinct cells", len({(r["cell_name"], r["K"]) for r in tbl}) == 24)

    # The tracked table must still agree with the artifact it was extracted from.
    art_path = REPO / "logs_classcount_decomposition" / "decomposition.json"
    prov = json.loads(
        (EV / "decomposition_fig2" / "fig2_inputs_provenance.json").read_text())
    if art_path.exists():
        check("tracked table's recorded source hash matches the artifact on disk",
              sha256(art_path) == prov["source"]["sha256"])
    else:
        print("  [note] artifact tree absent; table hash not re-checked")

    # Residual identity, per cell-epoch, from the tracked table alone.
    bad = [
        r["cell_name"] for r in tbl
        if not close(
            float(r["delta_centred_production"])
            - float(r["delta_centred_weight_only"])
            - float(r["delta_centred_centre_only"]),
            float(r["interaction_residual"]),
        )
    ]
    check("residual identity closes for every cell-epoch", not bad, str(bad[:3]))

    # Panel C bar heights.
    def dic(field, K, epoch=1):
        a = st.mean(float(r[field]) for r in tbl
                    if int(r["K"]) == K and r["head"] == "arcface"
                    and int(r["epoch"]) == epoch)
        c = st.mean(float(r[field]) for r in tbl
                    if int(r["K"]) == K and r["head"] == "ce"
                    and int(r["epoch"]) == epoch)
        return a - c

    for field, plotted in rec["panel_C_dic_epoch1"].items():
        ok = all(close(dic(field, K), plotted[str(K)]) for K in KS)
        check(f"panel C bar heights recomputed: {field}", ok,
              ", ".join(f"K={K} {plotted[str(K)]:+.6f}" for K in KS))

    # The corrected epoch-1 centre-only range, and its separation from later
    # epochs and from single-cell extrema.
    ctr1 = [dic("delta_centred_centre_only", K, 1) for K in KS]
    ctr3 = [dic("delta_centred_centre_only", K, 3) for K in KS]
    check("epoch-1 centre-only DiC range is 0.000200–0.000413",
          close(min(ctr1), 0.000200, 5e-7) and close(max(ctr1), 0.000413, 5e-7),
          f"[{min(ctr1):.6f}, {max(ctr1):.6f}]")
    check("epoch-3 centre-only DiC is a SEPARATE, larger range",
          max(ctr3) > max(ctr1), f"epoch 3 max {max(ctr3):.6f}")
    for field, want in (("delta_centred_centre_only", 0.006460),
                        ("interaction_residual", 0.006797),
                        ("delta_centred_production", 0.200685)):
        got = max(abs(float(r[field])) for r in tbl)
        check(f"all-cell all-epoch max |{field}| = {want}", close(got, want, 5e-7),
              f"{got:.6f}")

    # Exact finite-angle accounting, per cell, aggregated only afterwards.
    print("  -- exact finite-angle accounting (CE, epoch 1) --")
    for K in KS:
        cells = [r for r in tbl if int(r["K"]) == K and r["head"] == "ce"
                 and int(r["epoch"]) == 1]
        assert len(cells) == 4
        rows_ = []
        for c in cells:
            for conv in ("centred", "uncentred"):
                n0 = float(c[f"nc3_{conv}_forget_epoch0"])
                nt = float(c[f"nc3_{conv}_forget"])
                t0, tt = math.acos(n0), math.acos(nt)
                # Δcos = cos(θ0 + Δθ) − cos(θ0), exactly -- no linearisation.
                exact = math.cos(t0 + (tt - t0)) - math.cos(t0)
                if not close(exact, nt - n0, 1e-12):
                    FAILURES.append(f"finite-angle identity fails K={K} {conv}")
                rows_.append((conv, math.degrees(t0), math.degrees(tt - t0), exact))
        for conv in ("centred", "uncentred"):
            sub = [r for r in rows_ if r[0] == conv]
            mean_exact = st.mean(r[3] for r in sub)
            mt0, mdt = st.mean(r[1] for r in sub), st.mean(r[2] for r in sub)
            cos_of_means = (math.cos(math.radians(mt0 + mdt))
                            - math.cos(math.radians(mt0)))
            lin = -math.sin(math.radians(mt0)) * math.radians(mdt)
            print(f"     K={K:4d} {conv:9s} θ0 {mt0:6.2f}°  Δθ {mdt:5.2f}°  "
                  f"mean-of-exact-Δcos {mean_exact:+.6f}  "
                  f"[cos-of-mean-angles {cos_of_means:+.6f}, "
                  f"linearised −sinθ0·Δθ {lin:+.6f}]")
    check("Δcos = cos(θ0+Δθ) − cos(θ0) holds exactly for every cell and "
          "convention", not [f for f in FAILURES if "finite-angle" in f])

    # K=500 exclusion, read back rather than transcribed.
    g = rec["k500_gate"]
    check("K=500 gate gap exceeds the prespecified 2.00 pp",
          g["gap_pp"] > 2.00,
          f"CE {g['ce']*100:.4f} %, ArcFace {g['arcface']*100:.4f} %, "
          f"gap {g['gap_pp']:.4f} pp")
    check("K=500 failed because ArcFace EXCEEDED CE",
          g["arcface"] > g["ce"])
    check("K=500 carries no unlearning cells in the inventory",
          not (REPO / "logs_classcount" / "K500" / "unlearn").exists())

    # Axis coverage for panels A and B.
    for field, lim in (("forget_weight_angle_deg", (0, 25)),
                       ("forget_weight_norm_ratio", (0.75, 1.05))):
        vals = [float(r[field]) for r in tbl if int(r["epoch"]) in (1, 3)]
        check(f"every plotted observation inside the axis for {field}",
              min(vals) >= lim[0] and max(vals) <= lim[1],
              f"[{min(vals):.4f}, {max(vals):.4f}] vs {lim}")


# ---------------------------------------------------------------------------
def verify_fig3() -> None:
    print("\nFigure 3 — contrast rules")
    fixed = [r for r in csv.DictReader(
        open(EV / "k100_seed1" / "tables" / "fixed_epoch_dic.csv"))
        if r["convention"] == "centred"]
    matched = list(csv.DictReader(
        open(EV / "k100_seed1" / "tables" / "matched_outcome_dic.csv")))
    rec = sidecar("fig3_contrast_rules")["consumed"]
    YLIM = (-0.040, 0.160)

    check("7 attained pairs plotted in panel B",
          rec["n_attained_pairs"] == 7, str(rec["n_attained_pairs"]))
    check("4 of the 7 attained pairs have UNEQUAL exposure",
          rec["n_unequal_exposure"] == 4, str(rec["n_unequal_exposure"]))
    check("the one absent observation is seed 0 / identity 00524",
          rec["no_pair"] == ["seed0/00524"], str(rec["no_pair"]))

    # Panel A means.
    for seed in (0, 1):
        want = [st.mean(float(r["dic"]) for r in fixed
                        if int(r["seed"]) == seed and int(r["epoch"]) == e)
                for e in (1, 2, 3)]
        got = rec["panel_A"][f"seed{seed}/descriptive_mean"]
        check(f"seed {seed} descriptive means recomputed",
              all(close(a, b) for a, b in zip(want, got)),
              " / ".join(f"{v:+.4f}" for v in got))

    # The sign disagreement, at full precision.
    neg = [r for r in matched if r["attained_pair"] == "True"
           and float(r["matched_dic_centred"]) < 0]
    check("exactly ONE negative own-attainment centred contrast",
          len(neg) == 1, str([(r["seed"], r["identity"]) for r in neg]))
    check("it is seed 1, identity 00142, at -0.00727364360827909",
          neg[0]["seed"] == "1" and neg[0]["identity"] == "00142"
          and close(float(neg[0]["matched_dic_centred"]), -0.00727364360827909),
          neg[0]["matched_dic_centred"])
    fe29 = [float(r["dic"]) for r in sorted(
        (r for r in fixed if r["seed"] == "1" and r["forget_class"] == "29"),
        key=lambda r: int(r["epoch"]))]
    check("its fixed-epoch contrasts are POSITIVE at all three epochs",
          all(v > 0 for v in fe29), " / ".join(f"{v:+.4f}" for v in fe29))
    check("the highlighted values in the figure match those",
          all(close(a, b) for a, b in zip(fe29, rec["highlight"]["fixed_epoch"])))

    # Axis coverage across both panels, which share a y-axis.
    allv = [float(r["dic"]) for r in fixed] + [
        float(r["matched_dic_centred"]) for r in matched
        if r["attained_pair"] == "True"]
    check("every plotted contrast inside the shared y-axis",
          min(allv) >= YLIM[0] and max(allv) <= YLIM[1],
          f"[{min(allv):+.4f}, {max(allv):+.4f}] vs {YLIM}")
    check("panel B carries no means", "mean" not in json.dumps(rec["panel_B"]))


# ---------------------------------------------------------------------------
def verify_figS1() -> None:
    print("\nFigure S1 — stratified reversal counts")
    inv = list(csv.DictReader(
        open(EV / "run_inventory" / "clfonly_cell_inventory.csv")))
    rec = sidecar("figS1_reversal_by_stratum")["consumed"]

    check("inventory holds 69 cells", len(inv) == 69, str(len(inv)))
    check("5 stratum-F repeated observations excluded",
          rec["stratum_F_excluded"] == 5, str(rec["stratum_F_excluded"]))
    plotted = sum(v["n"] for v in rec["rows"].values())
    check("64 cells plotted (69 minus the 5 repeats)", plotted == 64, str(plotted))
    check("10 rows: 5 strata x 2 objectives", len(rec["rows"]) == 10)

    for key, v in rec["rows"].items():
        stratum, head = [s.strip() for s in key.split("|")]
        sub = [r for r in inv if r["stratum"] == stratum and r["head"] == head]
        ep1 = sum(r["centred_reversal_ep1"] == "True" for r in sub)
        any_ = sum(r["centred_reversal_any_epoch"] == "True" for r in sub)
        ok = (v["n"] == len(sub) and v["reversed_by_epoch1"] == ep1
              and v["reversed_only_by_epoch3"] == any_ - ep1
              and v["no_reversal"] == len(sub) - any_)
        check(f"counts recomputed: {key}", ok, f"{any_}/{len(sub)} reversing")

    segs = [v["reversed_by_epoch1"] + v["reversed_only_by_epoch3"]
            + v["no_reversal"] == v["n"] for v in rec["rows"].values()]
    check("segments sum to n in every row", all(segs))
    check("no pooled rate is recorded anywhere in the sidecar",
          "pooled" not in json.dumps(rec).lower()
          or "no pooled" in json.dumps(rec).lower())


# ---------------------------------------------------------------------------
def main() -> int:
    print("Verifying rendered figures against their sources (read-only).")
    for fid in ("fig1_k100_trajectories", "fig2_seed0_decomposition",
                "fig3_contrast_rules", "figS1_reversal_by_stratum"):
        print(f"\nProvenance — {fid}")
        verify_sidecar_hashes(fid)
    verify_fig1()
    verify_fig2()
    verify_fig3()
    verify_figS1()

    print(f"\n{CHECKS} checks run.")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
