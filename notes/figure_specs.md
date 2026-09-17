# Figure specifications — as implemented

**Status: implemented and laid out for manuscript width.** Scripts live in
`scripts/figures/`, rendered output in `figures/` (vector PDF plus PNG preview),
and each figure carries a provenance sidecar `figures/<id>.provenance.json`
recording its sources with SHA-256, the rendering revision, the rows or cells
consumed, the exact plotting command, and the final caption.

```
python scripts/figures/extract_fig2_inputs.py     # refresh the Figure 2 input table
python scripts/figures/make_fig1.py
python scripts/figures/make_fig2.py --from-artifact
python scripts/figures/make_fig3.py
python scripts/figures/make_figS1.py
python scripts/figures/verify_figures.py          # 90 checks, read-only
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

## Size, type and what belongs in the image

Every figure is **rendered at 7.0 inches wide**, the full manuscript column
width, and `figlib.save()` asserts that width so a layout change cannot
silently shrink the effective type size. `savefig` does **not** use a tight
bounding box, because a tight box resizes the canvas around whatever text spills
outside the axes; constrained layout keeps everything inside the declared figure
size instead, and figure legends use `loc="outside ..."` so they are given
reserved space. The type floor is **8 pt**, with 7.5 pt used only for dense
in-panel annotation. Because the figures are rendered at final size, those are
the sizes a reader gets.

**Prose belongs in the caption, not the image.** Explanatory paragraphs,
scope caveats and research-management warnings were moved out of the figures.
What remains inside each frame: axis definitions and units, legends,
missing-outcome labels, and seed/epoch information. The canonical caption for
each figure is the `CAPTION` constant in its script; it is copied into the
sidecar at render time, reproduced verbatim below, and `verify_figures.py`
checks that the three copies agree.

**Emphatic capitalisation has been removed** from every figure. Emphasis that
carries meaning is done with position, weight or colour.

## Global rules, applying to every figure

- **Never draw uncertainty bars over identities.** Four identities within a seed
  share one backbone per head, one split and one baseline checkpoint. They are a
  within-seed factor, not replicates. Individual identities are drawn as
  separate lines or points; never as a mean ± SD or a CI band. `figlib` provides
  no error-bar helper, and `descriptive_mean_marker()` is bare by construction.
- **Individual observations and descriptive means are both preserved.** Means
  are labelled descriptive wherever they appear.
- **Never label an identity as a seed.** The only replication unit is the seed,
  and there are two (CIFAR-10; faces-nested K=100).
- **State the NC3 convention in every axis label**, not only in the caption.
- **No cross-objective comparison of raw uncentred NC3 as a level claim.**
  Figure 1 plots uncentred NC3 as a within-head change from each cell's own
  epoch 0. That is a **presentation choice** following from the reference-frame
  argument, **not a prohibition on descriptive baseline-level comparison** — the
  baseline levels are stated in C9 and in Figure 1's caption, which is
  legitimate. What is forbidden is reading an uncentred level difference as a
  difference in unlearning quality or forgetting achieved.
- **Neither NC3, nor a baseline-subtracted change in it, nor DiC measures
  unlearning quality.** They are descriptive geometric quantities. Baseline
  subtraction does not upgrade them.
- **Epoch 1 is the decomposition anchor.** Where a single epoch is marked, it is
  epoch 1, drawn as a thin dashed rule rather than a broad band, so that it does
  not sit on top of the observations at the epoch that matters most.
- **Missing outcomes are drawn as absent, never imputed.** No interpolation, no
  substitution of epoch 3, no carrying a value forward.
- **Lineages are never pooled** into one "faces" series, and no figure shows a
  pooled rate across strata.
- **No inferential claim is supported by the present analysis.**

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

**Layout.** 7.0 × 6.3 in. A left-hand gutter column carries the bold seed label,
so the seed is named **wholly outside every plotting region**. For each seed:
a row of three panels, then a compact attainment strip spanning them.

| column | y-axis | range |
|---|---|---|
| 1 | forget-class test accuracy, correct out of 10 | −0.7 to 10.7 |
| 2 | NC3 centred, forget class (forget class excluded from centring reference) | −1 to +1, zero line drawn |
| 3 | change in NC3 uncentred from that cell's own epoch 0 | symmetric ±0.42 |

The redundant secondary 0.0–1.0 fraction axis has been **removed**: the count
out of 10 already carries the quantity, and the label says so.

Column 3's limits are **symmetric about zero on purpose**: the panel's content
is that every recorded change is negative, and a one-sided window would make
that unreadable as a sign statement. Observed range −0.3821 to 0.0000.

**x-axis.** Integer epochs 0–3, with a thin dashed rule at epoch 1 labelled
"epoch-1 anchor" once, in the top-right panel.

**Attainment strip.** The previous nested attainment rings stacked on
`(epoch, 0)` and collided whenever several cells reached 0/10 at the same epoch,
which is most of them. They are replaced by a compact **identity × objective
table** under each seed's row: rows are CE and ArcFace, columns are the four
identities, and each cell holds the first recorded epoch at 0 of 10. The seed-0
ArcFace identity-00524 cell reads **"none"** in a dashed, unfilled box; nothing
is imputed for it.

**Final caption.**

> Classifier-only random-label unlearning at K = 100: all four forget
> identities, both objectives, at two pipeline seeds (m = 9 active forget steps
> per epoch, lambda = 1, three epochs, backbone frozen in eval mode throughout,
> so no backbone parameter or BatchNorm buffer is updated during unlearning).
> Left: forget-class test accuracy over the 10 held-out images of that identity.
> Centre: centred NC3 for the forget class, with the forget class excluded from
> the centring reference; no cell crosses zero at either seed, and there are no
> sign reversals across any of the 16 cells. Right: uncentred NC3 as a change
> from each cell's own epoch 0, because the two objectives' uncentred baselines
> sit on opposite sides of zero (CE +0.4963 to +0.5409, ArcFace -0.8990 to
> -0.8892 at epoch 0); plotting the change rather than the level is a
> presentation choice following from that, not a bar on descriptive comparison
> of the baselines themselves. The strip beneath each seed gives the first epoch
> at which each cell reached 0 of 10; 15 of the 16 cells reach it within the
> three-epoch budget, and ArcFace identity 00524 at seed 0 does not (1 of 10 at
> epoch 3), which is marked 'none' and is not imputed, substituted or
> interpolated. The dashed rule at epoch 1 marks the decomposition anchor used
> in Figure 2; epochs 2 and 3 are sensitivity. Lines are individual identities,
> which within a seed share one backbone per head, one train/test split and one
> baseline checkpoint. They are a within-seed factor rather than replicates, so
> no dispersion is shown and none would be interpretable. Zero output accuracy
> is not representation erasure: the backbone is frozen here, so these cells do
> not test whether the identity remains recoverable from the features, and NC3
> does not measure unlearning quality.

**Must not imply.** That the identities are independent samples, or that the
spread between them estimates anything. That zero output accuracy is erasure.
That NC3 behaviour measures unlearning quality. That the absence of a zero
crossing in the centre column generalises beyond this dose, budget, lineage and
K. That the uncentred panel licenses a cross-objective level comparison.

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

**Layout.** 7.0 × 6.5 in. Panels A and B side by side on the top row; panel C
spans the full width beneath them, which is what gives its legend, its
small-value table and its inset room to sit **above** the tallest bar instead of
occluding it.

**Panel A — forget-weight rotation.** x = K, log scale, ticks at 100, 250, 500,
1000. y = rotation from epoch 0 in degrees, 0–25°. Epoch 1 at full weight,
epoch 3 lighter. Four identities as individual points with multiplicative
x-jitter, plus a bare descriptive mean tick, **no error bar**.

**Panel B — forget-weight norm ratio.** Same x-axis. y = ‖w_t‖/‖w_0‖, 0.75–1.05,
reference line at 1.0.

**K = 500 is marked as excluded in every panel that has a K axis** — A, B, C and
the inset — with the same grey band and the same word, so its absence reads
identically everywhere. No connecting line spans it: only the 100→250 segment is
drawn, and the K=1000 points stand alone. The gate record is read back from
`logs_classcount/K500/faces_{ce,arcface}_seed0/results.jsonl` at render time
rather than transcribed, and it is reported in the caption rather than as a
paragraph inside the image.

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

The third and fourth bars are correctly near-invisible at this scale. The bars
stay to scale and their values are printed in an in-panel block **with the
10⁻³ scale stated explicitly**; they are never labelled "≈ 0". **Keep the scopes
separate:** the epoch-1 centre-only range is **0.000200 to 0.000413**; epoch 3
reaches +0.001007; and the largest *single-cell* |centre-only| across all 24
cells and all epochs is 0.006460, the largest |residual| 0.006797, against a
largest |production| of 0.200685.

`|g_0|/|f_0|` for CE (0.80 / 0.73 / 0.69) appears **only as a clearly labelled
inset with its own axis and title**, drawn as **points with no connecting
line** — three K points do not establish a trend — and carrying the same K=500
exclusion marking as the other panels.

**Final caption.**

> Seed-0 reference-frame decomposition of centred NC3 over the 24 controlled
> nested class-count sweep cells. No seed-1 decomposition exists, so nothing
> here is a two-seed result. (A) Forget-class weight rotation from epoch 0.
> Per-K means place CE at 16.74 to 17.81 degrees at epoch 1 against ArcFace's
> 3.28 to 4.74 degrees; the per-cell ranges are 16.24 to 18.56 and 2.52 to 5.05
> degrees. CE's rotation is non-monotone in K and nearly flat. (B) Forget-weight
> norm ratio: CE contracts to 0.8129 to 0.8778 across epochs 1 and 3, ArcFace
> holds 0.9963 to 1.0016. These ratios are descriptive. A cosine is invariant to
> scaling its whole argument, so the uncentred cosine is unaffected by the
> weight norm; the centred cosine takes w - c, and changing w against a fixed c
> can change that direction. No counterfactual on the norm was run, so no
> attribution from these ratios to the metric is claimed. (C) Each bar is a
> baseline-subtracted counterfactual difference-in-changes, ArcFace minus CE, of
> the production centred NC3 A(w, c) = cos(f0 - g0, w - c): production is A(w_t,
> c_t) - A(w_0, c_0); weight-only holds the retain-weight centre at epoch 0;
> centre-only holds the weight at epoch 0; and the residual is production minus
> weight-only minus centre-only, exactly, per cell. Holding the centre at epoch
> 0 reproduces the K ordering almost exactly, while moving only the centre
> contributes +0.000200 to +0.000413 at epoch 1, with the residual of the same
> order. At epoch 3 the centre-only term reaches +0.001007, and across all 24
> cells and all epochs the largest single-cell centre-only magnitude is 0.006460
> and the largest residual 0.006797, against a largest production magnitude of
> 0.200685. So the ordering is the sensitivity of centred NC3 to a reference
> frame that varies with K, not increasing classifier rotation. The four corners
> are an arithmetic decomposition of the metric, not an intervention on
> training; the residual is a leftover rather than an interaction effect, since
> A is nonlinear in both arguments. None of these quantities measures unlearning
> quality. Points are individual identities, a within-seed factor rather than
> replicates, so means are descriptive and no dispersion is drawn. The sweep
> does not isolate K: changing K also changes the head's output width, the
> training population (3,926 to 38,815 train images), retain steps per epoch
> (31, 77, 303), candidate forget exposure (1,240, 3,080, 12,120 per epoch),
> baseline-training BatchNorm exposure (the unlearning phase runs the backbone
> in eval mode and accrues none), and task difficulty; only the active forget
> presentations, 360 per epoch, are held fixed. K = 500 carries no unlearning
> cells and is marked as excluded in every panel with a K axis, including the
> inset. Its baselines were CE 68.9676 % and ArcFace 71.3232 %, an absolute gap
> of 2.3556 pp against a prespecified 2.00 pp head-fairness gate. The gate is
> symmetric and failed because ArcFace exceeded CE, not because ArcFace
> underperformed; neither head was retuned to rescue it and both baselines are
> retained as artifacts. This is a deliberate exclusion under a prespecified
> rule rather than missing data: no value is imputed and no line is drawn across
> the gap in any panel.

**Must not imply.** That the counterfactual bars are causal contributions. That
any of these quantities measures unlearning quality. That the ordering is
established across seeds; it is one seed. That the sweep isolates K. That three
K points with per-cell overlap establish a trend. That any value exists at
K = 500.

---

## Figure 3 — Fixed-epoch versus own-attainment contrasts at K=100, two seeds

**Claim addressed.** C5 (two-seed fixed-epoch recurrence) and C7 (the two
contrast rules can disagree in sign), including the identity-00142 case.

**Source.** `evidence/k100_seed1/tables/fixed_epoch_dic.csv` and
`matched_outcome_dic.csv`, both derived from the 16 verbatim trajectories.

**Layout.** 7.0 × 4.75 in. Two panels sharing one y-axis (−0.042 to +0.192;
observed range −0.0214 to +0.1382, so nothing is clipped), with a gutter strip
beneath panel B. Centred convention only. Identity by colour, seed by marker
fill.

**Panel subtitles, as specified.**

- Panel A: *"Same epoch; matched active forget dose."*
- Panel B: *"Each objective at its first observed zero."*

Panel A's rule is **not** described as equal optimization: what is matched is the
epoch and the active forget dose (m = 9 steps per epoch at λ = 1).

**Panel A — fixed-epoch DiC.** x = epoch (1, 2, 3), zero line drawn. Eight
series = 4 identities × 2 seeds, plus the four-identity descriptive mean per
seed as a heavier line, labelled and drawn **without any band**. Means:
+0.0574 / +0.0648 / +0.0607 at seed 0; +0.0595 / +0.0708 / +0.0686 at seed 1.

**Panel B — own-attainment DiC.** Categorical x-axis of the four identities.
Each point keeps its `CE epoch / ArcFace epoch` annotation, in gold where the
two objectives are read at **different** epochs — 4 of the 7 attained pairs.

**The identity-00142 case.** The previous cross-panel arc and paragraph-length
annotation are replaced by a **short label beside the point in each panel**:
"00142, seed 1 (positive at every epoch)" in A, "00142, seed 1 (negative here)"
in B. The numbers and the reasoning are in the caption.

**Missing-outcome handling.** The absent seed-0 / 00524 observation is drawn as
an open cross in a **dedicated gutter axis beneath Panel B's frame**, outside
the numerical data region, labelled "no attained pair (seed 0)". It is not
placed at any y position inside the data region, because a glyph at a y-value
reads as a measured value and at y = 0 would read as a contrast of exactly zero.

**Panel B carries no means.** Seed 0 has 3 attained identities and seed 1 has 4,
so any cross-seed mean would compare different identity sets, and restricting to
the common set {00001, 00142, 00284} conditions on attainment in both seeds.

**Final caption.**

> Two contrast rules applied to the same K = 100 cells, at two pipeline seeds,
> in the centred NC3 convention. The quantity is the difference-in-changes,
> ArcFace minus CE, of the change in centred NC3 from each cell's own epoch 0;
> it is project arithmetic rather than a metric from the source paper. (A)
> Fixed-epoch rule: both objectives are read at the same epoch, having received
> the same active forget dose of 9 steps per epoch at lambda = 1. The
> four-identity mean is positive at every epoch and at both seeds (+0.0574,
> +0.0648, +0.0607 at seed 0; +0.0595, +0.0708, +0.0686 at seed 1), while
> individual identities vary widely. Those means are descriptive over a complete
> and identical identity set, not estimates, so no band, error bar or interval
> is drawn. (B) Own-attainment rule: each objective is read at its own first
> epoch with 0 of 10 correct forget predictions. Because the two objectives
> generally reach that point at different epochs -- in 4 of the 7 attained
> pairs, marked in gold -- this panel confounds objective with exposure and is
> not a like-for-like head comparison. Identity 00142 at seed 1 is negative
> under this rule (-0.007274) while its fixed-epoch contrast is positive at all
> three epochs (+0.0271, +0.0246, +0.0173); it is labelled in both panels. That
> sign disagreement between the two rules is the point of the figure. Identity
> 00524 at seed 0 has no attained pair, because ArcFace never reached 0 of 10
> within the budget (1 of 10 at epoch 3); it is shown in the gutter strip below
> the frame rather than at any y position, since a glyph placed at a value would
> read as a measured contrast and one placed at zero would read as a contrast of
> exactly zero. No epoch was substituted for it and the budget was not extended.
> Panel B carries no means: seed 0 has 3 attained identities and seed 1 has 4,
> so a cross-seed mean would compare different identity sets, and restricting to
> the common set 00001, 00142 and 00284 conditions on attainment in both seeds.
> Points are individual identities, a within-seed factor rather than replicates.
> A negative difference-in-changes is not an NC3 sign reversal: there are no
> sign reversals in any of the 16 K = 100 cells at either seed. Neither quantity
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

**Layout.** 7.0 × 5.5 in, as two subfigures: the chart above, a compact protocol
table below. They are subfigures rather than one gridspec because constrained
layout aligns axes boxes within a column, so a table sharing a column with the
chart would inherit the chart's narrowed box and lose the width its columns
need.

**Chart.** Horizontal stacked counts, one row per (stratum × objective),
core-first: A, B, C, D, E. x = number of cells, integer ticks. Row labels are
**short** — "A · nested sweep, seed 0", "C · faces-1000" and so on, with the
objective beneath and the stratum named once per pair. Segments keep **epoch-1
reversals distinguishable from those first appearing at epoch 2 or 3**, with the
remainder non-reversing. The fraction at the end of each row is defined on the
axis: **k / n = reversed at any recorded epoch / total cells**.

**Protocol table.** The detailed differences that make these strata
non-comparable moved out of the row labels into a compact table beneath the
chart — lineage, K, input resolution, test images per forget class, dose
protocol — headed "Protocol differences between strata — not
protocol-comparable, never pooled".

| stratum × objective | n | reversed at epoch 1 | first at epoch 2–3 | k/n |
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
**No pooled rate is computed or shown anywhere.**

**Final caption.**

> Centred NC3 sign reversals by stratum and objective, counted at epoch 1 and by
> epoch 3. Supplementary and contextual only. A reversal is a sign change of
> centred NC3 for the same class and the same convention, measured from that
> cell's own epoch-0 baseline; it is defined at every recorded epoch, so no cell
> is missing from a row. The fraction printed at the end of each row is the
> number of cells reversed at any recorded epoch over the total number of cells
> in that row. Reversal frequency ranges from 8 of 8 for CIFAR-10 under ArcFace
> to 0 of 12 for the controlled nested sweep under ArcFace. The table beneath
> the chart gives the protocol differences that separate these strata: they
> differ at the same time in domain, input resolution, class count, the number
> of test images per forget class, and the dose protocol. Protocol comparability
> has not been demonstrated, the rows are therefore not points on a single axis,
> and they are never pooled into an overall rate; no pooled figure is computed
> or shown anywhere. The ordering of rows is not a dataset effect, not a
> class-count effect and not an effect of the objective, and CIFAR-10 and the
> face lineages are not two points on one axis. Stratum D uses a different
> source directory and a different minimum image count from the other face
> lineages, and its extraction seed was not recorded, so it cannot be
> regenerated from the repository alone. The most controlled comparison in the
> project is the nested class-count sweep, strata A and B, which shows no
> reversals at any K; that sweep too moves a bundle of correlated quantities
> with K rather than K alone. Five further cells, forming stratum F, are
> excluded because they are repeated observations of the stratum C and E cells
> at forget class 0 under different doses, and including them would
> double-count. 64 cells are plotted. No inferential claim is supported by the
> present analysis: these are counts of cells, and within a stratum the
> identities share a backbone per objective, a split and a baseline checkpoint.

**Must not imply.** That the ordering of rows is a dataset effect, a class-count
effect, or an effect of the objective. That the strata can be pooled into an
overall rate. That CIFAR-10 and the face lineages are two points on one axis.

---

## Provenance and verification

Each `figures/<id>.provenance.json` records the figure id, generation time, the
render width in inches, the rendering revision, the caption, the plotting
command, every source file with SHA-256 / size / whether it is tracked in git,
the rows or cells consumed, the output files with their hashes, and free-text
notes on what the figure does and does not claim.

**Two commits, not one.** The sidecar's `rendered_at_commit` is the revision of
the *plotting scripts and input tables* that produced the image. The **delivery
commit** — the commit that contains the rendered files — is necessarily a
*descendant* of it, because the outputs cannot be committed until after they
exist. `verify_figures.py` checks that HEAD is `rendered_at_commit` or a
descendant of it, rather than pretending the two are the same.

**The clean-status flag is partial, and named so.**
`inputs_clean_excluding_generated_outputs_under_figures` excludes everything
under `figures/`, because rendering writes there and counting those files would
make the flag unconditionally dirty. It reports whether the *inputs to
rendering* were committed, which is what reproducing a figure depends on.

`scripts/figures/verify_figures.py` re-reads the sources independently of the
plotting scripts and runs **90 checks**: the rendering revision is HEAD or an
ancestor; render inputs were clean under the stated exclusion; the declared and
actual PDF widths are the manuscript width; the caption is stored with the
figure and matches this document verbatim; source and output hashes still match;
denominators (10 forget, 970 retain); first-0/10 epochs recomputed against the
attainment strip; the single missing attainment; every observation inside its
axis limits in all four figures; sign counts including no centred reversals
across the 16 K=100 cells; the residual identity for all 72 cell-epochs; Panel C
bar heights recomputed from the tracked table; the epoch-1 centre-only range and
its separation from later-epoch and single-cell extrema; the exact finite-angle
identity `Δcos = cos(θ₀+Δθ) − cos(θ₀)` per cell in both conventions; the K=500
gate record and direction; the fc29 sign disagreement at full precision; the
7 attained pairs and 4 unequal exposures; and the stratified S1 counts with
stratum F excluded.
