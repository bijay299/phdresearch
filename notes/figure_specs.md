# Figure specifications

Specifications only. **No plotting code and no rendered figures** — those are
deliberately deferred so the figure set can be approved first.

Every figure below draws on **already-recorded artifacts**. None requires
training, inference, replay, or any new metric.

**Global rules, applying to every figure.**

- **Never draw uncertainty bars over identities.** Four identities within a seed
  share one backbone per head, one split and one baseline checkpoint. They are a
  within-seed factor, not replicates. Show individual identities as separate
  lines or points; never as a mean ± SD or a CI band.
- **Never label an identity as a seed.** The only replication unit is the seed,
  and there are two (CIFAR-10; faces-nested K=100).
- **State the NC3 convention in every axis label**, not only in the caption.
- **Never put raw uncentred NC3 from the two heads on a shared axis as a level
  comparison.** ArcFace starts near −0.88 and CE near +0.51 to +0.72 before any
  unlearning. Plot uncentred only as a within-head change from that cell's own
  epoch 0.
- **Epoch 1 is the decomposition anchor.** Where a single epoch must be marked,
  mark epoch 1 and show later epochs as sensitivity.
- **Missing outcomes are drawn as absent, never imputed.** No interpolation, no
  substitution of epoch 3, no carrying a value forward.
- Lineages are never pooled into one "faces" series.

---

## Figure 1 (priority 1) — Output forgetting and NC3 trajectories, per identity

**Claim addressed.** C1 (ArcFace does not eliminate classifier-only output
forgetting) and C3 (reversal is not necessary for output forgetting), on the
controlled core.

**Source artifacts.**
`evidence/k100_seed1/trajectories/seed0/{ce,arcface}_fc{0,29,60,95}.jsonl` and
`.../seed1/{…}.jsonl` (16 files, verbatim copies; SHA-256-linked in
`evidence/k100_seed1/provenance.json`). Equivalently
`evidence/k100_seed1/tables/trajectories_both_seeds.csv`. Stratum A + B only.

**Layout.** A 2 × 3 grid of panels. Rows = seed (0, 1). Columns = quantity.

| column | y-axis | units | y-range |
|---|---|---|---|
| 1 | forget-class test accuracy | correct out of 10 (integer ticks 0–10; secondary label 0.0–1.0) | 0–10 |
| 2 | NC3 centred, forget class (forget class excluded from centring reference) | cosine, dimensionless | fixed −1 to +1 across all panels |
| 3 | NC3 uncentred, forget class — **change from that cell's own epoch 0** | Δcosine | symmetric about 0, common to both rows |

**x-axis.** Unlearning epoch, integer ticks 0, 1, 2, 3. Draw a light vertical
rule at epoch 1 labelled "decomposition anchor". Epochs 2–3 are sensitivity.

**Series.** 8 lines per panel = 2 objectives × 4 identities. Objective by
colour (one colour for CE, one for ArcFace, consistent everywhere in the
paper). Identity by line style or marker, with a shared legend giving both the
numeric label and the identity folder: 0 = 00001, 29 = 00142, 60 = 00284,
95 = 00524. Identity encoding must be identical in both rows so a reader can
track the same person across seeds.

**Column 2 must include a horizontal zero line**, so the absence of any sign
crossing is visible rather than asserted.

**Missing-outcome handling.** Every cell has all four epochs recorded, so no
line is truncated. In column 1, mark the **first** epoch at which a cell reaches
0/10 with a filled marker; the seed-0 ArcFace identity-00524 line **never
reaches 0/10** (it ends at 1/10) and therefore receives **no marker** — annotate
it directly in-panel as "not attained within budget". Do not extend its axis or
extrapolate.

**Draft caption.** *Classifier-only random-label unlearning at K=100, all four
forget identities, both objectives, at two pipeline seeds (m = 9 active forget
steps per epoch, λ = 1, three epochs, backbone frozen). Left: forget-class test
accuracy over 10 held-out images; every cell but one reaches 0/10 within the
budget, the exception being ArcFace identity 00524 at seed 0 (1/10 at epoch 3,
unmarked). Centre: centred NC3 for the forget class, with the forget class
excluded from the centring reference; no cell crosses zero at either seed.
Right: uncentred NC3 as a change from each cell's own epoch 0, since the two
objectives' uncentred baselines sit on opposite sides of zero. Lines are
individual identities, not replicates.*

**Must not imply.** That the identities are independent samples, or that the
spread between them estimates anything. That zero output accuracy is erasure —
the backbone is frozen, so features do not move here. That the absence of a
zero crossing in the centre column generalises beyond this dose, budget,
lineage and K. That the uncentred panel permits a cross-objective level
comparison; it is a within-head change by construction.

---

## Figure 2 (priority 2) — Seed-0 classifier rotation, norm, and the reference-frame decomposition

**Claim addressed.** C6a (rotation and norm) and C6b (the K ordering is a
reference-frame effect). **Seed 0 only** — this must be stated in the caption
and in the panel title, because no seed-1 decomposition exists.

**Source artifacts.** `logs_classcount_decomposition/decomposition.json` and
`decomposition.md`, plus the per-cell `nc3_state_ep{0,1,2,3}.npz`. These are
git-ignored; the figure script must read them from the local artifact tree and
record their SHA-256 in the figure's provenance sidecar. Values are also
tabulated in the 2026-09-16 decomposition entry in `notes/decisions.md`.

**Layout.** Three panels in a row.

**Panel A — forget-weight rotation.** x = K on a log scale with ticks at exactly
100, 250, 1000. **K = 500 carries no unlearning cells.** Draw its tick, leave
its data region empty, and **break every connecting line at that position** —
no segment may span K=250 to K=1000, because a line crossing the gap would read
as an interpolated observation at K=500 where none exists. Points must not be
joined across it in any panel.
y = rotation from epoch 0, degrees, 0–25°. Two series (CE, ArcFace) at epoch 1,
plus the same two at epoch 3 in a lighter weight. Plot the four identities as
individual points with a short horizontal offset (jitter) per K; overlay the
mean as a small tick, **without** an error bar.

**Panel B — forget-weight norm ratio.** Same x-axis. y = ‖w_t‖ / ‖w_0‖,
dimensionless, range 0.75–1.05, with a horizontal reference line at 1.0.
Same series and per-identity point treatment. Annotate that norm is
**scale-invariant to the cosine** and is descriptive only.

**Panel C — decomposition of ΔArcFace − ΔCE, centred, epoch 1.** Grouped bars
at K = 100, 250, 1000. Four bars per group, in this order: production
`A(w_t, c_t)`; centre fixed at epoch 0 `A(w_t, c_0)`; weight fixed at epoch 0
`A(w_0, c_t)`; interaction residual. y = Δcosine, 0 to +0.15 with the residual
allowed to go slightly negative. Label the third and fourth bars in-panel as
"≈ 0". An inset or second y-axis may carry `|g_0| / |f_0|` for CE
(0.80 / 0.73 / 0.69) as a line, since that is where the ordering enters.

**Missing-outcome handling — K = 500.** The tick appears; the data region is
empty; **no line connects across it.** Annotate with the verified gate record
read from `logs_classcount/K500/faces_{ce,arcface}_seed0/results.jsonl`:

> K = 500 baselines: CE **68.9676 %**, ArcFace **71.3232 %**, absolute gap
> **2.3556 pp**, exceeding the pre-registered 2.00 pp head-fairness gate. Its
> unlearning cells were never run and neither head was retuned to rescue it.
> Both baselines are retained as artifacts.

State that the gate is **symmetric** and failed because ArcFace **exceeded** CE,
not because ArcFace underperformed. Do not omit the tick, do not impute a value,
and do not describe K=500 as "missing data" — it is a deliberate exclusion under
a pre-registered rule.

**Draft caption.** *Seed-0 reference-frame decomposition of centred NC3 over the
24 controlled nested-sweep cells. (A) Forget-class weight rotation from epoch 0:
CE rotates 16.7–17.8° at epoch 1 against ArcFace's 3.3–4.7°, and CE's rotation
is essentially flat in K. (B) Forget-weight norm ratio: CE contracts to
0.83–0.87, ArcFace holds 0.996–1.001; norm does not enter the cosine and is
shown as description only. (C) Holding the retain-weight centre at epoch 0
reproduces the K ordering almost exactly, while moving only the centre
reproduces nothing and the interaction residual stays below 0.007 against a
largest production change of 0.201 — so the ordering is sensitivity of centred
NC3 to a fixed reference frame that varies with K, not increasing classifier
rotation. Points are individual identities, not replicates. K = 500 carries no
unlearning cells.*

**Must not imply.** That the counterfactual bars are causal contributions —
`A` is nonlinear in both arguments, so the four corners are an arithmetic
decomposition of the **metric**, not an intervention on training, and the
residual is a leftover rather than an interaction effect. That the ordering is
established across seeds; it is one seed. **That the sweep isolates K**: it is a
controlled nested class-count comparison, and moving K also moves the head's
output width, the training population (3,926 → 38,815 train images), retain
steps per epoch (31 / 77 / 303), candidate forget exposure (1,240 / 3,080 /
12,120 per epoch), BatchNorm exposure and task difficulty — only the *active*
forget presentations (360/epoch) are held fixed. That three K points with
per-cell overlap establish a trend. That any value exists at K = 500.

---

## Figure 3 (priority 3) — Fixed-epoch versus own-attainment contrasts at K=100, two seeds

**Claim addressed.** C5 (two-seed fixed-epoch recurrence) and C7 (the two
contrast rules can disagree in sign), including the identity-00142 (fc29) case.

**Source artifacts.** `evidence/k100_seed1/tables/fixed_epoch_dic.csv` and
`evidence/k100_seed1/tables/matched_outcome_dic.csv`. Both derive from the 16
verbatim trajectories in the same directory.

**Layout.** Two panels side by side, sharing a y-axis so the two contrast rules
are directly comparable in magnitude.

**Panel A — fixed-epoch DiC.** x = epoch (1, 2, 3). y = DiC (Δcosine,
dimensionless), centred convention, with a horizontal zero line. Eight series:
4 identities × 2 seeds. Seed by marker shape (filled = seed 0, hollow = seed 1);
identity by colour. Overlay the four-identity mean per seed as a heavier line —
**annotated "descriptive mean, not an estimate" and drawn without any band.**

**Panel B — own-attainment DiC.** A categorical x-axis of the four identities
(00001, 00142, 00284, 00524). y = matched-outcome DiC, same scale and zero line
as Panel A. Two points per identity (seed 0, seed 1) using the same marker
convention. Above each point, annotate the exposure pair as `CE ep / Arc ep`
(e.g. `1 / 2`), because 4 of the 7 attained pairs read the two objectives at
**different** epochs.

**Required highlights.**
- In Panel B, identity **00142 at seed 1** sits at **−0.007274** — the only
  negative own-attainment centred contrast. Draw it emphasised, and connect it
  with a thin guide to its Panel A counterpart, which is positive at all three
  epochs (+0.0271, +0.0246, +0.0173). The sign disagreement between the two
  contrast rules is the point of the figure.
- In Panel B, identity **00524 at seed 0** has **no point**: ArcFace never
  reached 0/10, so no attained pair exists. Mark the slot with an open cross and
  the in-panel label "no attained pair". **Do not substitute epoch 3, do not
  interpolate, and do not compute a contrast for it.**

**Missing-outcome handling.** As above — the absent seed-0/00524 point is drawn
as explicitly absent. Any mean shown in Panel B must state its n (seed 0: n = 3;
seed 1: n = 4) directly in the panel, or be omitted entirely; a cross-seed mean
comparison over different identity sets must not be drawn without that label.

**Draft caption.** *Two contrast rules on the same K=100 cells, at two pipeline
seeds. (A) Fixed-epoch DiC = ΔArcFace − ΔCE at a common epoch; the
four-identity mean is positive at every epoch and at both seeds (+0.0574 /
+0.0648 / +0.0607 at seed 0; +0.0595 / +0.0708 / +0.0686 at seed 1), while
individual identities vary widely. (B) Own-attainment DiC, reading each
objective at its own first epoch with 0/10 correct forget predictions; exposure
differs between objectives in 4 of the 7 attained pairs, and identity 00142 at
seed 1 is negative (−0.007274) despite positive fixed-epoch contrasts at all
three epochs. Identity 00524 at seed 0 has no attained pair. Points are
individual identities, not replicates.*

**Must not imply.** That the two panels measure the same thing — Panel B
confounds objective with exposure and is not a like-for-like head comparison.
That the mean in Panel A is an estimate with sampling error; it is a descriptive
mean over four non-independent identities. That a negative DiC is an NC3 sign
reversal — there are **zero** sign reversals in all 16 K=100 cells, and the
figure should carry that sentence. That two seeds establish robustness.

---

## Figure S1 (supplementary) — Cross-setting reversal summary

**Status: supplementary and contextual only.** Protocol comparability across
these settings has **not** been demonstrated, so this figure may not appear as a
main result and may not be read as a dataset effect.

**Claim addressed.** C2, descriptively.

**Source artifacts.** `evidence/run_inventory/clfonly_cell_inventory.csv`
(columns `stratum`, `head`, `centred_reversal_ep1`,
`centred_reversal_any_epoch`), which is itself built from the per-cell
trajectories named in its `source_artifact` column.

**Layout.** A single horizontal stacked-count chart. One row per
(stratum × objective), ordered core-first: A, B, then C, D, E. x = number of
cells, integer ticks. Each row shows reversals at epoch 1 and additional
reversals appearing only by epoch 3 as a distinct segment, with the remainder as
non-reversing. Print `k/n` at the end of each row.

**Each row must be annotated with its incompatibilities**, in the row label
itself, not a footnote: dose protocol (controlled m=9/λ=1 vs uncontrolled),
input resolution (32×32 vs 112×112), test images per forget class (~1000 /
25–31 / 10), and, for stratum D, that its lineage uses a different source
directory, a different `min_images`, and an **unrecorded extraction seed**.

**Missing-outcome handling.** Not applicable — reversal is defined at every
recorded epoch for every cell. Stratum F is **excluded** from this figure: its
five cells are repeated observations of stratum C and E cells at fc0 under
different doses, and including them would double-count.

**Draft caption.** *Centred NC3 sign reversals by stratum and objective, at
epoch 1 and by epoch 3. Reversal frequency ranges from 8/8 (CIFAR-10, ArcFace)
to 0/12 (controlled nested sweep, ArcFace) across settings that differ
simultaneously in domain, input resolution, class count, dose protocol and
test-set resolution. **These settings are not protocol-comparable and the
figure is contextual only**; the single comparison in this project that varies
one factor is the nested sweep across K, which shows no reversals at any K.*

**Must not imply.** That the ordering of rows is a dataset effect, a class-count
effect, or an effect of the objective. That the strata can be pooled into an
overall rate. That CIFAR-10 and the face lineages are two points on one axis.

---

## Provenance requirement for whenever these are implemented

Each figure script should write a sidecar recording: the figure id, the exact
source files with SHA-256, the git commit, and the rows or cells consumed — the
same standard the tables under `evidence/` already meet. Figures derived from
git-ignored trees (Figure 2) must record the artifact hashes, since the source
itself is not tracked.
