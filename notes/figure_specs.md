# Figure specifications — as implemented

**Status: implemented.** Scripts live in `scripts/figures/`, rendered output in
`figures/` (vector PDF plus PNG preview), and each figure carries a provenance
sidecar `figures/<id>.provenance.json` recording its sources with SHA-256, the
execution revision, the rows or cells consumed, and the exact plotting command.

```
python scripts/figures/extract_fig2_inputs.py     # refresh the Figure 2 input table
python scripts/figures/make_fig1.py
python scripts/figures/make_fig2.py --from-artifact
python scripts/figures/make_fig3.py
python scripts/figures/make_figS1.py
python scripts/figures/verify_figures.py          # 66 checks, read-only
```

Every figure draws on **already-recorded artifacts**. None required training,
inference, replay, or any new metric.

| id | file | claim | source |
|---|---|---|---|
| Figure 1 | `fig1_k100_trajectories` | C1, C3 | `evidence/k100_seed1/` (16 verbatim trajectories) |
| Figure 2 | `fig2_seed0_decomposition` | C6a, C6b | `evidence/decomposition_fig2/` (extracted from the ignored decomposition tree) |
| Figure 3 | `fig3_contrast_rules` | C5, C7 | `evidence/k100_seed1/tables/` |
| Figure S1 | `figS1_reversal_by_stratum` | C2, descriptively | `evidence/run_inventory/clfonly_cell_inventory.csv` |

---

## Global rules, applying to every figure

- **Never draw uncertainty bars over identities.** Four identities within a seed
  share one backbone per head, one split and one baseline checkpoint. They are a
  within-seed factor, not replicates. Individual identities are drawn as
  separate lines or points; never as a mean ± SD or a CI band. `figlib` provides
  no error-bar helper, and `descriptive_mean_marker()` is bare by construction.
- **Never label an identity as a seed.** The only replication unit is the seed,
  and there are two (CIFAR-10; faces-nested K=100).
- **State the NC3 convention in every axis label**, not only in the caption.
- **No cross-objective comparison of raw uncentred NC3 as a level claim.**
  ArcFace starts near −0.89 and CE near +0.50 before any unlearning. Figure 1
  plots uncentred NC3 as a within-head change from each cell's own epoch 0.
  That is a **presentation choice** following from the reference-frame argument,
  **not a prohibition on descriptive baseline-level comparison** — the baseline
  levels are stated in C9 and in Figure 1's provenance notes, which is
  legitimate. What is forbidden is reading an uncentred level difference as a
  difference in unlearning quality or forgetting achieved.
- **Neither NC3, nor a baseline-subtracted change in it, nor DiC measures
  unlearning quality.** They are descriptive geometric quantities. Baseline
  subtraction does not upgrade them. No caption or panel may imply otherwise.
- **Epoch 1 is the decomposition anchor.** Where a single epoch is marked, it is
  epoch 1; later epochs are sensitivity.
- **Missing outcomes are drawn as absent, never imputed.** No interpolation, no
  substitution of epoch 3, no carrying a value forward.
- **Lineages are never pooled** into one "faces" series, and no figure shows a
  pooled rate across strata.
- **No inferential claim is supported by the present analysis.** Means are
  labelled descriptive wherever they appear.

**Shared encodings** (`scripts/figures/figlib.py`). Objective by colour,
Okabe-Ito: CE `#0072B2`, ArcFace `#D55E00`. Identity by marker and line style,
identical in every panel of every figure: 0 = 00001 (circle, solid),
29 = 00142 (square, dashed), 60 = 00284 (triangle, dash-dot),
95 = 00524 (diamond, dotted). Seed by marker fill where a seed dimension exists:
filled = seed 0, hollow = seed 1. PDFs embed TrueType (`pdf.fonttype 42`).

---

## Figure 1 — Output forgetting and NC3 trajectories, per identity

**Claim addressed.** C1 (ArcFace does not eliminate classifier-only output
forgetting) and C3 (reversal is not necessary for output forgetting), on the
controlled core (strata A + B, K=100).

**Source.** `evidence/k100_seed1/tables/trajectories_both_seeds.csv`, with all
16 verbatim per-cell `.jsonl` trajectories hashed into the sidecar alongside it.

**Layout.** 2 × 3. Rows = seed (0, 1). Columns = quantity. 8 lines per panel =
2 objectives × 4 identities, encoded as above.

| column | y-axis | range |
|---|---|---|
| 1 | forget-class test accuracy, correct out of 10 (secondary axis 0.0–1.0) | −0.75 to 10.6 |
| 2 | NC3 centred, forget class (forget class excluded from centring reference) | −1 to +1, zero line drawn |
| 3 | NC3 uncentred, forget class — change from that cell's own epoch 0 | symmetric ±0.42 |

Column 3's limits are **symmetric about zero on purpose**: the panel's content
is that every recorded change is negative, and a one-sided window would make
that unreadable as a sign statement. Observed range −0.3821 to 0.0000.

**x-axis.** Integer epochs 0–3, with a light band at epoch 1 labelled
"decomposition anchor" in the top-right panel.

**Attainment marking.** The first epoch at which a cell reaches 0/10 carries an
open ring. Several cells attain at the same epoch, so rings are **sized per
cell** and drawn largest-first; every marker stays at its true epoch and none is
buried. The seed-0 ArcFace identity-00524 cell **never reaches 0/10** (1/10 at
epoch 3) and therefore receives **no ring**; it is annotated in-panel as "not
attained within budget".

**Final caption.**

> **Figure 1. Classifier-only random-label unlearning at K = 100, all four
> forget identities, both objectives, at two pipeline seeds** (m = 9 active
> forget steps per epoch, λ = 1, three epochs, backbone frozen in `eval()` mode
> throughout). *Left:* forget-class test accuracy over 10 held-out images; 15 of
> the 16 cells reach 0/10 within the budget, the exception being ArcFace
> identity 00524 at seed 0 (1/10 at epoch 3, unmarked and not imputed). Rings
> mark each cell's first 0/10 epoch and are sized per cell so that coincident
> attainments remain separable. *Centre:* centred NC3 for the forget class, with
> the forget class excluded from the centring reference; no cell crosses zero at
> either seed, and there are zero sign reversals across all 16 cells. *Right:*
> uncentred NC3 as a change from each cell's own epoch 0, since the two
> objectives' uncentred baselines sit on opposite sides of zero (CE +0.4963 to
> +0.5409; ArcFace −0.8990 to −0.8892 at epoch 0). Lines are individual
> identities — a within-seed factor sharing one backbone per head, one split and
> one baseline checkpoint — not replicates, so no dispersion is shown.

**Must not imply.** That the identities are independent samples, or that the
spread between them estimates anything. That zero output accuracy is erasure —
the backbone is frozen, so features do not move here, and no probe of
recoverability was run in these cells. That NC3 behaviour measures unlearning
quality. That the absence of a zero crossing in the centre column generalises
beyond this dose, budget, lineage and K. That the uncentred panel licenses a
cross-objective level comparison; it is a within-head change by construction.

---

## Figure 2 — Seed-0 classifier rotation, norm, and the reference-frame decomposition

**Claim addressed.** C6a (rotation and norm) and C6b (the K ordering is a
reference-frame effect). **Seed 0 only** — stated in the figure's legend title,
because no seed-1 decomposition exists.

**Source and preservation.** The underlying artifact
`logs_classcount_decomposition/decomposition.json` is git-ignored like every
artifact tree. `scripts/figures/extract_fig2_inputs.py` copies the 72 cell-epoch
rows the figure consumes into the **tracked**
`evidence/decomposition_fig2/fig2_plotting_inputs.csv`, with
`fig2_inputs_provenance.json` recording the source path, its SHA-256, the
extraction revision, the fields copied, and the checks run. Floats are stored as
`repr()` and round-trip exactly. The figure reads the tracked table, so it is
checkable from the repository alone; `make_fig2.py --from-artifact` additionally
reads the artifact and asserts bit-equality of every consumed field. Nothing is
recomputed from `.npz` state.

**Panel A — forget-weight rotation.** x = K, log scale, ticks at 100, 250, 500,
1000. y = rotation from epoch 0 in degrees, 0–25°. Epoch 1 at full weight,
epoch 3 lighter. Four identities as individual points with multiplicative
x-jitter, plus a bare descriptive mean tick, **no error bar**.

**Panel B — forget-weight norm ratio.** Same x-axis. y = ‖w_t‖/‖w_0‖, 0.75–1.05,
reference line at 1.0.

**K = 500 carries no unlearning cells.** Its tick is drawn, its data region is
shaded empty and labelled "K = 500 / no cells", and **no connecting line spans
it in any panel** — only the 100→250 segment is drawn, and the K=1000 points
stand alone. A line crossing the gap would read as an interpolated observation
where none exists. The gate record is read back from
`logs_classcount/K500/faces_{ce,arcface}_seed0/results.jsonl` at render time
rather than transcribed.

**Panel C — decomposition of the centred DiC at epoch 1.** Every bar is a
**baseline-subtracted counterfactual difference-in-changes**, not a raw cosine.
Writing `A(w, c) = cos(f_0 − g_0, w − c)` for the production centred NC3 of the
forget row:

    Δ_production  = A(w_t, c_t) − A(w_0, c_0)
    Δ_weight-only = A(w_t, c_0) − A(w_0, c_0)      (centre held at epoch 0)
    Δ_centre-only = A(w_0, c_t) − A(w_0, c_0)      (weight held at epoch 0)
    residual      = Δ_production − Δ_weight-only − Δ_centre-only
    bar height    = (that quantity for ArcFace) − (same quantity for CE)

The residual identity closes exactly, per cell, for all 72 cell-epochs
(verified). Bar heights at epoch 1, in K order 100 / 250 / 1000:

| quantity | K=100 | K=250 | K=1000 |
|---|---|---|---|
| Δ_production | +0.057439 | +0.109325 | +0.139318 |
| Δ_weight-only | +0.056071 | +0.108730 | +0.139262 |
| Δ_centre-only | +0.000413 | +0.000200 | +0.000340 |
| residual | +0.000955 | +0.000395 | −0.000285 |

The third and fourth bars are correctly near-invisible at this scale and are
**labelled with their values, never as "≈ 0"**, in an in-panel block. **Keep the
scopes separate:** the epoch-1 centre-only range is **0.000200 to 0.000413**;
epoch 3 reaches +0.001007; and the largest *single-cell* |centre-only| across
all 24 cells and all epochs is 0.006460, the largest |residual| 0.006797,
against a largest |production| of 0.200685.

`|g_0|/|f_0|` for CE (0.80 / 0.73 / 0.69) appears **only as a clearly labelled
inset with its own axis and title**, never as a secondary y-axis on the bar
panel.

**Final caption.**

> **Figure 2. Seed-0 reference-frame decomposition of centred NC3 over the 24
> controlled nested-sweep cells.** *(A)* Forget-class weight rotation from
> epoch 0: per-K means put CE at 16.74–17.81° at epoch 1 against ArcFace's
> 3.28–4.74° (per-cell ranges 16.24–18.56° and 2.52–5.05°), and CE's rotation is
> non-monotone and nearly flat in K. *(B)* Forget-weight norm ratio: CE
> contracts to 0.8129–0.8778 across epochs 1 and 3, ArcFace holds 0.9963–1.0016.
> These ratios are **descriptive**. A cosine is invariant to scaling its whole
> argument, so the uncentred cosine is unaffected by the weight norm; the centred
> cosine takes `w − c`, and changing `w` against a fixed `c` can change that
> direction — no counterfactual on the norm was run, so no attribution is
> claimed. *(C)* Holding the retain-weight centre at epoch 0 reproduces the K
> ordering almost exactly, while moving only the centre contributes a small but
> non-zero +0.000200 to +0.000413 at epoch 1, with the interaction residual of
> the same order; the residual is what makes the four bars an exact identity
> rather than an additive attribution. So the ordering is the sensitivity of
> centred NC3 to a reference frame that varies with K, not increasing classifier
> rotation. Points are individual identities, not replicates, and no dispersion
> is shown. K = 500 carries no unlearning cells: it failed the prespecified
> 2.00 pp head-fairness gate (CE 68.9676 %, ArcFace 71.3232 %, gap 2.3556 pp),
> the gate is symmetric and failed because ArcFace *exceeded* CE, and neither
> head was retuned to rescue it.

**Must not imply.** That the counterfactual bars are causal contributions — `A`
is nonlinear in both arguments, so the four corners are an arithmetic
decomposition of the **metric**, not an intervention on training, and the
residual is a leftover rather than an interaction effect. That any of these
quantities measures unlearning quality. That the ordering is established across
seeds; it is one seed. **That the sweep isolates K**: it is a controlled nested
class-count comparison, and moving K also moves the head's output width, the
training population (3,926 → 38,815 train images), retain steps per epoch
(31 / 77 / 303), candidate forget exposure (1,240 / 3,080 / 12,120 per epoch),
**baseline-training** BatchNorm exposure — the unlearning phase runs the
backbone in `eval()` mode and accrues none — and task difficulty; only the
*active* forget presentations (360/epoch) are held fixed. That three K points
with per-cell overlap establish a trend. That any value exists at K = 500.

---

## Figure 3 — Fixed-epoch versus own-attainment contrasts at K=100, two seeds

**Claim addressed.** C5 (two-seed fixed-epoch recurrence) and C7 (the two
contrast rules can disagree in sign), including the identity-00142 case.

**Source.** `evidence/k100_seed1/tables/fixed_epoch_dic.csv` and
`matched_outcome_dic.csv`, both derived from the 16 verbatim trajectories.

**Layout.** Two panels sharing one y-axis (−0.040 to +0.160; observed range
−0.0214 to +0.1382, so nothing is clipped). Centred convention only. Identity by
colour, seed by marker fill.

**Panel A — fixed-epoch DiC.** x = epoch (1, 2, 3), zero line drawn. Eight
series = 4 identities × 2 seeds, plus the four-identity descriptive mean per
seed as a heavier line, labelled and drawn **without any band**. Means:
+0.0574 / +0.0648 / +0.0607 at seed 0; +0.0595 / +0.0708 / +0.0686 at seed 1.

**Panel B — own-attainment DiC.** Categorical x-axis of the four identities.
Each point is annotated `CE epoch / ArcFace epoch`, in gold and bold where the
two objectives are read at **different** epochs — 4 of the 7 attained pairs.

**Required highlights, both implemented.** Identity **00142 at seed 1** sits at
**−0.00727364360827909**, the only negative own-attainment centred contrast; it
is drawn emphasised, haloed in both panels, and connected by a thin dotted arc
bowed clear of both data regions to its Panel A counterpart, which is positive
at all three epochs (+0.0271 / +0.0246 / +0.0173). Identity **00524 at seed 0**
has **no point**: ArcFace never reached 0/10, so no attained pair exists, and
none is computed.

**Missing-outcome handling.** The absent observation is drawn as an open cross
in a **dedicated gutter axis beneath Panel B's frame**, outside the numerical
data region, labelled "no attained pair". It is not placed at any y position
inside the data region, because a glyph at a y-value reads as a measured value
and at y = 0 would read as a contrast of exactly zero.

**Panel B carries no means.** Seed 0 has 3 attained identities and seed 1 has 4,
so any cross-seed mean would compare different identity sets, and restricting to
the common set {00001, 00142, 00284} conditions on attainment in both seeds.

**Final caption.**

> **Figure 3. Two contrast rules on the same K = 100 cells, at two pipeline
> seeds.** *(A)* Fixed-epoch DiC = Δ_ArcFace − Δ_CE at a common epoch; the
> four-identity descriptive mean is positive at every epoch and at both seeds
> (+0.0574 / +0.0648 / +0.0607 at seed 0; +0.0595 / +0.0708 / +0.0686 at
> seed 1), while individual identities vary widely. The means are descriptive
> over a complete and identical identity set, not estimates, so no band is
> drawn. *(B)* Own-attainment DiC, reading each objective at its own first epoch
> with 0/10 correct forget predictions; exposure differs between objectives in 4
> of the 7 attained pairs (gold annotations), so this panel confounds objective
> with exposure and is **not** a like-for-like head comparison. Identity 00142 at
> seed 1 is negative (−0.007274) despite positive fixed-epoch contrasts at all
> three epochs — the two rules disagree in sign for that cell, which is the point
> of the figure. Identity 00524 at seed 0 has no attained pair and appears only
> in the gutter strip below the frame. Points are individual identities, not
> replicates. A negative difference-in-changes is **not** an NC3 sign reversal:
> there are zero sign reversals in all 16 K = 100 cells. Neither quantity
> measures unlearning quality or representation erasure, and two seeds do not
> establish robustness.

**Must not imply.** That the two panels measure the same thing. That the mean in
Panel A is an estimate with sampling error. That a negative DiC is an NC3 sign
reversal. That two seeds establish robustness.

---

## Figure S1 — Cross-setting reversal summary

**Status: supplementary and contextual only.** Protocol comparability across
these settings has **not** been demonstrated, so this figure may not appear as a
main result and may not be read as a dataset effect.

**Claim addressed.** C2, descriptively.

**Source.** `evidence/run_inventory/clfonly_cell_inventory.csv`, columns
`stratum`, `head`, `centred_reversal_ep1`, `centred_reversal_any_epoch`.

**Layout.** Horizontal stacked counts, one row per (stratum × objective),
core-first: A, B, C, D, E. x = number of cells, integer ticks. Segments are
reversals at epoch 1, additional reversals appearing only by epoch 3, and the
remainder non-reversing; `k/n` is printed at the end of each row. **Each row
label carries its own incompatibilities inline** — lineage, K, input resolution,
test images per forget class, and dose protocol; for stratum D, the different
source directory, different `min_images`, and **unrecorded extraction seed**.

| stratum × objective | n | reversed by ep 1 | only by ep 3 | k/n |
|---|---|---|---|---|
| A · nested sweep, CE | 12 | 0 | 0 | 0/12 |
| A · nested sweep, ArcFace | 12 | 0 | 0 | 0/12 |
| B · K=100 seed 1, CE | 4 | 0 | 0 | 0/4 |
| B · K=100 seed 1, ArcFace | 4 | 0 | 0 | 0/4 |
| C · faces-1000, CE | 4 | 0 | 0 | 0/4 |
| C · faces-1000, ArcFace | 4 | 0 | 0 | 0/4 |
| D · faces-100, CE | 4 | 0 | 0 | 0/4 |
| D · faces-100, ArcFace | 4 | 1 | 1 | 2/4 |
| E · CIFAR-10, CE | 8 | 0 | 0 | 0/8 |
| E · CIFAR-10, ArcFace | 8 | 8 | 0 | 8/8 |

**Stratum F is excluded** (5 cells): repeated observations of stratum C and E
fc0 cells under different doses, which would double-count. 64 cells are plotted.
**No pooled headline rate is computed or shown anywhere.**

**Final caption.**

> **Figure S1. Centred NC3 sign reversals by stratum and objective, at epoch 1
> and by epoch 3. Supplementary and contextual only.** Reversal frequency ranges
> from 8/8 (CIFAR-10, ArcFace) to 0/12 (controlled nested sweep, ArcFace) across
> settings that differ *simultaneously* in domain, input resolution, class count,
> dose protocol and test-set resolution; each row label states its own
> incompatibilities. **These settings are not protocol-comparable, the rows are
> not points on one axis, and they are never pooled into an overall rate.** The
> most controlled comparison in the project is the nested class-count sweep
> (strata A and B), which shows no reversals at any K — though it too moves a
> bundle of correlated quantities with K, not K alone. Reversal is a sign change
> of centred NC3 for the same class and convention from that cell's own epoch-0
> baseline, and is defined at every recorded epoch, so no cell is missing from a
> row. Stratum F's five repeated observations are excluded to avoid
> double-counting. No inferential claim is supported by the present analysis.

**Must not imply.** That the ordering of rows is a dataset effect, a class-count
effect, or an effect of the objective. That the strata can be pooled into an
overall rate. That CIFAR-10 and the face lineages are two points on one axis.

---

## Provenance and verification

Each `figures/<id>.provenance.json` records the figure id, generation time, the
execution revision (commit and whether the tree was dirty), the plotting
command, every source file with SHA-256 / size / whether it is tracked in git,
the rows or cells consumed, the output files with their hashes, and free-text
notes on what the figure does and does not claim.

`scripts/figures/verify_figures.py` re-reads the sources independently of the
plotting scripts and runs **66 checks**: source and output hashes still match;
denominators (10 forget, 970 retain); first-0/10 epochs recomputed; the single
missing attainment; every observation inside its axis limits in all four
figures; sign counts including zero centred reversals across the 16 K=100 cells;
the residual identity for all 72 cell-epochs; Panel C bar heights recomputed
from the tracked table; the corrected epoch-1 centre-only range and its
separation from later-epoch and single-cell extrema; the exact finite-angle
identity `Δcos = cos(θ₀+Δθ) − cos(θ₀)` per cell in both conventions; the K=500
gate record and direction; the fc29 sign disagreement at full precision; the
7 attained pairs and 4 unequal exposures; and the stratified S1 counts with
stratum F excluded. All 66 pass.
