# Paper claims and limitations

Synthesis of every claim the existing evidence supports, the convention each
claim depends on, and what each explicitly does not assert. Written 2026-09-17
from `notes/decisions.md` and the tracked evidence under `evidence/`. **No new
experiment was run to produce this document.**

Target: CVPR 2027 — registration Nov 10 2026, submission Nov 16 2026.

**Central framing (research-lead decision, 2026-09-17).**

> In the evaluated settings, ArcFace does not eliminate classifier-only output
> forgetting. CE and ArcFace exhibit different classifier dynamics, while NC3
> interpretation depends on reference geometry and comparison rule.

This is a **qualified negative result plus mechanism analysis**. It claims
neither representation erasure nor superior unlearning by either head. The
original three-outcome framing (features must move / unlearning fails / a third
mechanism) does not fit and should not be forced onto the result.

**Evidence hierarchy (research-lead decision).** The **core** is the controlled
nested face sweep (stratum A), the seed-0 reference-frame decomposition, and
the K=100 seed-1 replication (stratum B). Historical face and CIFAR findings
(strata C, D, E) are **explicitly qualified context**. Incompatible lineages
are **never pooled** — see the run inventory for why.

**Epoch policy (research-lead decision).** Show full recorded trajectories
where available. **Epoch 1 is the established decomposition anchor**; later
epochs are sensitivity information. Historical reversal counts are always
reported at an explicitly stated epoch. **Fixed-epoch and own-attainment
contrasts are kept separate and never pooled.**

**Reading conventions used throughout.**

- **NC3, exactly as production computes it** (`src/metrics.py: nc3_alignment`).
  Let `mu[k]` be the class-mean training feature of class *k*, `W[k]` the
  classifier weight row, and define the **reference population**

      ref = {k : class k is present in the features} \ {forget class}

  i.e. every class with at least one feature, **excluding the forget class**
  (`exclude_from_centre`). Both centres are taken over that **same** population:

      feature-side centre   g_0 = nanmean_{k in ref} mu[k]
      weight-side centre    c   = mean_{k in ref} W[k]
      centred    NC3_k = cos( mu[k] - g_0 ,  W[k] - c )
      uncentred  NC3_k = cos( mu[k]      ,  W[k]     )

  The cosine L2-normalises each argument internally, so `W` is passed raw
  (`fc.weight` for CE, `W` for ArcFace; neither row-normalised beforehand).
  With `centre=False` the centring branch is skipped entirely, so `g_0` and `c`
  play no part in the uncentred convention. Every reversal claim in this project
  is **centred**.
- **Where the weight norm does and does not enter.** A cosine is invariant to
  scaling *its whole argument*. In the **uncentred** convention the argument is
  `W[k]` itself, so `‖W[k]‖` genuinely does not enter the value. In the
  **centred** convention the argument is `W[k] − c`, and the centre `c` is
  subtracted **before** normalising: scaling `W[k]` against a fixed `c` changes
  the *direction* of `W[k] − c`, so the raw weight magnitude **can** affect
  centred geometry. An earlier draft stated flatly that "weight norm does not
  enter the value"; that is true only of the uncentred convention. The measured
  norm ratios reported in C6a are **descriptive**, and no attribution from them
  to the centred metric is claimed here — establishing one would need a
  counterfactual on the norm, which was not run.
- **What the two conventions can and cannot support.** The two conventions place
  the same weight rotation at **different baseline angles**, so they have
  different sensitivity and can disagree about
  whether anything changed. **Descriptive cross-objective comparison of
  uncentred levels is possible and is reported** — CE baselines sit at +0.51 to
  +0.72 and ArcFace baselines at −0.878 to −0.895, which is itself a stated
  observation (C9). The precise restriction is narrower: because the two
  objectives' weight geometries place them in **different reference frames
  before any unlearning**, an uncentred **level** difference between objectives
  **cannot be interpreted as a difference in unlearning quality or in forgetting
  achieved**. For that purpose use the within-head change from each cell's own
  epoch 0, and for cross-objective comparison of *change* prefer the centred
  convention, whose baseline angles are closer together. Plotting uncentred NC3
  as a within-head change rather than a level is a **presentation choice** that
  follows from this, not a prohibition: a *descriptive* statement of the two
  objectives' baseline levels is reported (C9) and is legitimate.
- **Neither NC3, nor a change in it, nor DiC measures unlearning quality.**
  These are geometric descriptions of where one classifier row sits relative to
  a class-mean feature and a reference frame. They are not a measure of how much
  was forgotten, of whether the identity remains recoverable, or of the quality
  of an unlearning procedure. Baseline subtraction does not change this: a
  baseline-subtracted geometric quantity is still a geometric quantity. Output
  forgetting, metric behaviour and representation erasure are **three different
  things**, and only the first two are observed anywhere in this project.
- **Flip / sign reversal** — a sign change for the **same model, same class,
  same convention**, from that model's own baseline. Nothing else earns the word.
- **DiC** = (NC3_ArcFace,t − NC3_ArcFace,0) − (NC3_CE,t − NC3_CE,0). **Project
  arithmetic, not a metric from Gao et al.** Must be labelled as such wherever
  it appears.
- **`probe_gap_to_retrain*`** = unlearned `probe_forget` − matched retrain
  `probe_forget`. The asterisk travels with the name; also project arithmetic.
- **Fixed-epoch vs matched-outcome contrasts are different quantities** and are
  never pooled. See C7.

---

## Part I — Claims the evidence supports

### C0. Run inventory and how every aggregate count is built

**All counts below are stratified.** `evidence/run_inventory/clfonly_cell_inventory.csv`
lists all **69 authoritative classifier-only cells** with lineage, objective,
seed, K, forget identity, update mode, dose protocol, recorded epochs,
attainment, reversal epochs, source artifact, and whether the artifact is
reused elsewhere.

| stratum | lineage | dose protocol | seeds | cells | role |
|---|---|---|---|---|---|
| **A** | faces-nested (`casia-webface-folders`, min40/max50), K=100/250/1000 | m=9, λ=1, per-cell reseed | 0 | 24 | **core** |
| **B** | faces-nested, K=100 | m=9, λ=1, per-cell reseed | 1 | 8 | **core** |
| **C** | faces-1000, forget ids {0,122,389,794} | uncontrolled (every retain step) | 0 | 8 | qualified context |
| **D** | faces-100, **separate lineage** (`…-100id`, min100, extraction seed unrecorded) | uncontrolled | 0 | 8 | qualified context |
| **E** | CIFAR-10 32×32 | uncontrolled | 0, 1 | 16 | qualified context |
| **F** | fc0 only, CIFAR + faces-1000 | four different doses | 0 | 5 | **repeated observations**, not independent cells |

**Excluded as non-authoritative, with reasons:** 8 pre-LR-fix CIFAR ArcFace
cells (superseded by stratum E); 10 `unlearn_pass1_nan_uncentred` cells (nan
defect fixed at `51c38be`); and **25 decomposition/replay artifacts**.

#### The 25 replay artifacts map to 24 canonical stratum-A cells

The extra artifact is the **gate cell**,
`logs_classcount_decomposition/gate/K1000/unlearn/faces_arcface_seed0_random_label_clfonly_fc0`.
It is a **second replay of one cell** — K=1000, ArcFace, identity 00001 — run
first, on its own, to verify that instrumented replay reproduced the canonical
run bitwise before the remaining cells were launched. So:

    24 canonical stratum-A cells
      → 24 instrumented replays in logs_classcount_decomposition/K{100,250,1000}/
      + 1 gate-cell replay duplicating (K=1000, arcface, fc0)
      = 25 replay artifacts

All three copies of that cell — canonical, main replay, gate replay — were
re-verified here as **bitwise equal** on `output_forget`, `output_retain`,
`probe_forget`, both NC3 conventions and `nc1_angular`, with identical
`training_trace_sha256` (`4ce51788…`) and `active_dose_trace_sha256`
(`a6960f3e…`).

**Confirmed: no aggregate count treats any of them as independent evidence.**
Zero inventory rows take a `logs_classcount_decomposition/` path as their
`source_artifact`; all 24 stratum-A rows source from `logs_classcount/`, and the
decomposition tree appears only in the `reused_elsewhere` column as a
cross-reference.

#### Replication terminology

The two seeds are **pipeline-seed replications on shared source data**, not
independent datasets and not initialization-only replications. At K=100 the two
seeds draw on the **same source directory, the same identity roster and the same
selected image pool** (identical `identity_manifest_sha256` and
`image_manifest_sha256`); what the seed changes is the **train/test assignment,
the initialization and the shuffling**. They exist in exactly two places:
CIFAR-10 (seeds 0 and 1, both objectives, 4 classes) and faces-nested K=100
(seeds 0 and 1, both objectives, 4 identities). Everything else is seed 0 only.
**Four identities within a seed are not four seeds** and are never counted as
replications.

#### Audit of the previously quoted "31 of 32"

**The arithmetic was right and the denominator was not.** Verified: 31 of 32
ArcFace cells in strata A–E attain zero output forgetting. But that "32" was an
arbitrary boundary — it silently **excluded the 5 stratum-F ArcFace fc0 control
cells, all of which also attain** (36/37 if included) — and it pooled **two
dose protocols** (16 cells at m=9/λ=1 against 16 uncontrolled), **four
lineages**, and **three test-set resolutions** (CIFAR ~1000 test images per
class; faces-100 25–31 per identity; faces-1000 and faces-nested exactly 10).
A single ratio across those is not interpretable. **It is replaced by the
stratified table in C1 and should not be quoted again.**

---

### C1. In the evaluated settings, ArcFace does not eliminate classifier-only output forgetting

**Claim.** Classifier-only random-label unlearning drives forget-class output
accuracy to zero under ArcFace in every stratum tested, as it does under CE.
The hypothesis that a normalised angular-margin loss closes the classifier
shortcut is **not supported in these settings**.

**Evidence — stratified attainment of `output_forget = 0` within 3 epochs:**

| stratum | ArcFace | CE | not attained |
|---|---|---|---|
| **A** controlled nested sweep (seed 0) | **11/12** | 12/12 | K=100 fc95 (identity 00524), ArcFace |
| **B** seed-1 replication, K=100 | 4/4 | 4/4 | — |
| C historical faces-1000 | 4/4 | 4/4 | — |
| D historical faces-100 | 4/4 | 4/4 | — |
| E CIFAR-10 | 8/8 | 8/8 | — |
| F fc0 controls (repeated obs.) | 5/5 | n/a | — |

**The one exception, stated explicitly:** stratum A, K=100, seed 0, ArcFace,
identity 00524 ends at 1/10 and never reaches zero inside the budget. It does
**not** recur in stratum B, where the same identity attains at epoch 1. The
budget was not extended.

**Does not assert.** Equal speed (C8). That zero output accuracy means erasure
(C4). Generality beyond `random_label` classifier-only unlearning at this dose
and budget — no CosFace, and full-model unlearning only as a narrow fc0 control
(C11a).

---

### C2. Centred NC3 reversal frequency differs markedly across experimental settings

**Claim.** Reversal frequency ranges from 8/8 to 0/24 across the settings
tested, with the objective held fixed within each comparison. **The design does
not isolate which factor produces this**, so the difference is reported
descriptively across settings, not attributed to dataset.

**Evidence — centred convention, epoch stated explicitly:**

| stratum | head | n | reversals at **epoch 1** | reversals at **any epoch 1–3** |
|---|---|---|---|---|
| **A** controlled nested sweep | ArcFace | 12 | **0/12** | **0/12** |
| **A** | CE | 12 | 0/12 | 0/12 |
| **B** seed-1 K=100 | ArcFace | 4 | **0/4** | **0/4** |
| **B** | CE | 4 | 0/4 | 0/4 |
| C faces-1000 | ArcFace | 4 | 0/4 | 0/4 |
| C | CE | 4 | 0/4 | 0/4 |
| D faces-100 | ArcFace | 4 | **1/4** | **2/4** |
| D | CE | 4 | 0/4 | 0/4 |
| E CIFAR-10 | ArcFace | 8 | **8/8** | **8/8** |
| E | CE | 8 | 0/8 | 0/8 |

**Uncentred reversals: 0 of all 69 cells.**

**Why no causal attribution.** CIFAR-10 and the face lineages differ
simultaneously in domain, input resolution (32×32 vs 112×112), class count,
scale *s*, images per identity, test-set resolution and dose protocol. Strata C
and D differ from each other in source directory, `min_images`, images per
identity **and** identity membership — only 31 of D's 100 identities appear in
C's set. Stratum A's nested sweep is the **most controlled** comparison in the
project — nested identity sets, shared class indices, one config key moving —
but it is **not** an isolated causal intervention on K either (see C6c). It
shows **no** reversals at any K.

**Does not assert.** That reversal is or is not "a property of the loss" — that
phrasing is withdrawn as unsupported. That class count causes the difference.
That the settings are otherwise comparable enough to pool; **they are not, and
a cross-setting reversal summary is supplementary context only.** CIFAR-10 is
described here as a **contrasting, non-equivalent experimental setting**, not as
an outlier requiring explanation — calling it an outlier would presuppose the
settings lie on one comparable axis, which is exactly what is not established.

---

### C3. Centred reversal is neither necessary for output forgetting nor sufficient evidence of representation erasure

**Not necessary.** In strata A and B (32 cells, the controlled core) every cell
but one reaches `output_forget = 0` with **zero** centred reversals. Strata C
and D add 8 more ArcFace cells reaching zero with 0/4 and 1/4 reversal at
epoch 1.

**Not sufficient.** In stratum E, ArcFace reverses 8/8 at epoch 1 **under a
frozen backbone**, where the representation is unchanged by construction. A
reversal therefore cannot itself be evidence that features moved.

**Does not assert.** Anything about necessity or sufficiency for "the illusion"
as Gao et al. define it — only for the two measurable quantities named.

---

### C4. Three quantities are distinct and must be reported separately

**Claim.** Output forgetting, NC3 metric behaviour, and representation erasure
are three different things, and this project measures only the first two.

**Evidence, in three tiers of decreasing strength.** (i) **Code path**: in every
main cell the backbone is held in `eval()` with gradients off and the optimizer
built over head parameters only, so no backbone parameter or BatchNorm buffer is
written — state-independent reasoning, and the only tier that applies to *every*
cell. (ii) **Toy test**: `src/test_dose_schedule.py:511` asserts buffer equality
after a classifier-only run, but on an untrained `Linear + BatchNorm1d(5)` model,
not ResNet-18. (iii) **Seed-0 replay**: in the 24 stratum-A cells the
**class-mean matrix** is bit-identical at epochs 0–3 (sha256 of its float64
bytes). **That is equality of the 512-dimensional per-class means, not equality
of every individual feature vector** — it is strong evidence the mapping did not
move, and is what the decomposition requires, but it is a lower-dimensional
summary and does not by itself prove per-sample feature equality. **No equivalent
check exists for seed 1**: those cells saved no state, so direct post-unlearning
state equality is unavailable there. Separately, the probe metric cannot carry an
erasure claim: the retrained reference never saw the forget class yet scores
0.50–1.00 `probe_forget`, because the probe is fitted post hoc on labelled
examples of that class.

**Does not assert.** That representations are unchanged *in general* under
unlearning — only that these runs cannot speak to it. Full-model unlearning was
run only as a narrow paired control (C11).

---

### C5. Two-seed fixed-epoch NC3 recurrence (metric level, strata A+B)

**Claim.** At a fixed epoch and matched forget dose, the four-identity mean
centred DiC is positive at K=100 and recurs in direction and similar magnitude
across the two pipeline seeds. **This is a metric-level result. It is stated
independently of the rotation/norm mechanism in C6, which rests on seed 0
alone.**

**Evidence — fixed-epoch centred DiC, mean over four identities, K=100:**

| epoch | seed 0 (stratum A) | seed 1 (stratum B) |
|---|---|---|
| **1** (anchor) | +0.0574 | +0.0595 |
| 2 (sensitivity) | +0.0648 | +0.0708 |
| 3 (sensitivity) | +0.0607 | +0.0686 |

Sign counts, fixed-epoch, 4 identities × 3 epochs = 12 per seed: centred
**10/12** positive at seed 0 — the two negatives are identity 00524 at epochs 2
and 3, **one cell observed twice, not two cells** — and **12/12** at seed 1;
uncentred **12/12** at both seeds.

Larger K gives larger centred DiC at seed 0 (+0.1093 at K=250, +0.1393 at
K=1000, epoch 1), but see C6 before reading that as a K effect.

**Does not assert.** **Positive DiC is not superior unlearning**, and not
erasure. Four identities are not four seeds: the recurrence is across **two**
seeds at one K. The dose figures are loop-structure counts explicitly labelled
"not a gradient magnitude" in the artifacts, so this is not a statement about
gradient sizes.

---

### C6. Seed-0 mechanism: classifier rotation and norm, and the reference-frame account of the K ordering

**Scope, stated first.** Everything in this claim comes from the **seed-0**
reference-frame decomposition over stratum A's 24 cells. **No seed-1
decomposition exists**, so none of it is a two-seed result. The research lead
has closed this line: do not reopen without specific contradictory evidence.

**C6a — rotation and norm (seed 0).** Forget-weight rotation from epoch 0. The
first two columns are the spread of the **per-K means** over four identities;
the bracketed figures are the **per-cell** range, which is wider and is what
Figure 2 plots:

| epoch | CE (mean over K) | ArcFace (mean over K) | CE per-cell | ArcFace per-cell |
|---|---|---|---|---|
| 1 | 16.74–17.81° | 3.28–4.74° | 16.24–18.56° | 2.52–5.05° |
| 3 | 20.79–22.60° | 5.22–6.23° | 20.04–23.59° | 4.44–6.50° |

Forget-weight norm ratio, per-cell across both epochs: CE **0.8129–0.8778**,
ArcFace **0.9963–1.0016**. So at seed 0, CE's classifier rotates several times
further and contracts, while ArcFace's rotates little and holds its norm.

These norm ratios are **descriptive**. They are *not* asserted to be inert with
respect to the metric: as the conventions note above records, the centred cosine
takes `W[k] − c`, so a change in `‖W[k]‖` against a given `c` can move the
centred value. No counterfactual on the norm was run, so this project makes
**no attribution** from norm to `nc3_centred_forget` in either direction. The
uncentred convention is genuinely norm-invariant.

**C6b — the K ordering is a reference-frame effect, in this seed-0
decomposition.** Centred DiC orders K=100 < 250 < 1000, but:

- **Not reference drift, though the centre is not inert either.** Holding the
  retain-weight centre at epoch 0 reproduces the ordering essentially unchanged
  (+0.0561/+0.1087/+0.1393 against production +0.0574/+0.1093/+0.1393 at
  epoch 1). The **centre-only** contribution to DiC is **small but non-zero**:
  +0.000413 / +0.000200 / +0.000340 at epoch 1 — a **range of 0.000200 to
  0.000413** — and +0.000511 / +0.000626 / +0.001007 at epoch 3, i.e. two to
  three orders of magnitude below production. **Keep the three scopes apart:**
  the epoch-1 range is 0.000200–0.000413; epoch 3 reaches 0.001007; and across
  **all 24 cells and all four epochs** the largest *single-cell* |centre-only|
  is **0.006460** and the largest |residual| **0.006797**, against a largest
  |production| of **0.200685**. An earlier draft quoted "0.0002 to 0.0010" as an
  epoch-1 range; that upper end is epoch 3's.
- **Not more rotation at larger K.** CE's forget-weight rotation in **K order**
  is 17.8148° / 16.7366° / 17.4080° at epoch 1 — **non-monotone, spanning
  1.0782°** — and 22.6002° / 20.7862° / 21.4885° at epoch 3, spanning 1.8140°,
  also non-monotone. (An earlier draft quoted the epoch-3 span as if it were
  epoch 1's.)
- **Where the ordering enters — the angle domain, read in K order.**

  **Method.** For each cell, θ = arccos(NC3) in the relevant convention, and the
  change is evaluated **exactly** as

      Δcos = cos(θ₀ + Δθ) − cos(θ₀)

  with no small-angle or derivative approximation: the rotations here are 9–18°,
  where a linearisation is visibly wrong. Every quantity is computed **per cell
  first** and averaged only afterwards; the cosine of an averaged angle is never
  substituted for an averaged cosine. (`−sin θ₀` is quoted below only as *local
  sensitivity intuition* for why a larger baseline angle amplifies a given
  rotation, never as the accounting.)

  For CE at epoch 1, means over the four identities of per-cell quantities:

| K | unc θ₀ | unc Δθ | unc ΔNC3 | cen θ₀ | cen Δθ | cen ΔNC3 |
|---|---|---|---|---|---|---|
| 100 | 57.97° | 17.77° | −0.2841 | 31.93° | 9.14° | −0.0948 |
| 250 | 53.21° | 16.69° | −0.2552 | 34.15° | 9.60° | −0.1053 |
| 1000 | 54.07° | 17.35° | −0.2681 | 39.05° | 11.32° | −0.1378 |

  In the **centred** frame both the baseline angle (31.93 < 34.15 < 39.05) and
  the projected rotation (9.14 < 9.60 < 11.32) **increase monotonically with K**
  and push the same way, so centred ΔNC3 orders monotonically
  (−0.0948, −0.1053, −0.1378).

  In the **uncentred** frame the rotation Δθ is essentially flat
  (17.77 / 16.69 / 17.35, matching the weight rotation above), and the baseline
  angle θ₀ is **non-monotone in K — it falls from 100 to 250 and rises again at
  1000** (57.97 / 53.21 / 54.07). The uncentred changes inherit that shape
  (−0.2841 / −0.2552 / −0.2681). **That is why the uncentred convention shows no
  K ordering: its baseline angle is not monotone in K, not because the
  underlying rotation differs.** (An earlier draft listed these angles sorted by
  magnitude rather than by K, which made a non-monotone sequence read as a
  decreasing one.)

  **Exact transplant, per cell.** Evaluating each cell's rotation at the *other*
  convention's baseline angle — again exactly, `cos(θ₀ + Δθ) − cos(θ₀)` — shows
  how much of the difference is the frame rather than the motion:

| K | cen Δθ applied at cen θ₀ | cen Δθ applied at **unc** θ₀ | unc Δθ applied at unc θ₀ | unc Δθ applied at **cen** θ₀ |
|---|---|---|---|---|
| 100 | −0.0948 | −0.1415 | −0.2841 | −0.2019 |
| 250 | −0.1053 | −0.1420 | −0.2552 | −0.1962 |
| 1000 | −0.1378 | −0.1702 | −0.2681 | −0.2224 |

  Moving the *same* per-cell rotation into the uncentred baseline angle inflates
  its effect on the cosine by roughly 1.2–1.5× and flattens the K ordering
  (−0.1415 / −0.1420 / −0.1702 against −0.0948 / −0.1053 / −0.1378). This is
  arithmetic on the metric, not a statement about training, and it does not
  measure unlearning quality.

  For scale, the linearisation the earlier draft leaned on understates the exact
  change by about 8–11 % at these rotations (e.g. K=100 uncentred: exact
  −0.2841 against −sin θ₀·Δθ = −0.2630), which is why the exact form is used.
- **Why each objective's centre moves the metric so little, for different
  reasons.** CE's retain-weight centre is nearly the zero vector
  (`|c_0|/|w_0|` = 0.0248 / 0.0104 / 0.0012), so even a sizeable rotation of it
  (7.34° / 6.73° / 14.55° at epoch 1) moves the metric by ~1e-4. ArcFace's
  centre is nearly as long as the weight (0.96–1.01), so centring matters
  greatly to its *level*, but the centre barely rotates at all (0.0049° /
  0.0210° / 0.0210° at epoch 1), so it contributes little to the *change*.

**C6c — what the nested sweep does and does not control.** The sweep is a
**controlled nested class-count comparison, not an isolated causal intervention
on K.** Only `max_identities` and `out_dir` move in the config, identity sets
are nested, and a shared identity keeps its class index — but changing K
necessarily changes the training population and the classifier geometry too.
Read from the cells' own provenance and dose records:

| K | images | train | test | retain imgs | retain steps/epoch | candidate forget presentations/epoch | **active** forget presentations/epoch |
|---|---|---|---|---|---|---|---|
| 100 | 4,906 | 3,926 | 980 | 3,886 | 31 | 1,240 | **360** |
| 250 | 12,218 | 9,774 | 2,444 | 9,734 | 77 | 3,080 | **360** |
| 1000 | 48,519 | 38,815 | 9,704 | 38,775 | 303 | 12,120 | **360** |

**Held fixed:** active forget presentations per epoch (360), active
forget-bearing steps (9/epoch, 27 total), λ=1, 3 epochs, per-cell reseeding, the
backbone, the schedule, and the four forget identities.

**Not held fixed, and varying with K by construction:** the head's output width
(100 / 250 / 1000, i.e. the classifier geometry itself), the training population
(~10× more images from K=100 to K=1000), the number of retain optimisation steps
per epoch (31 / 77 / 303) and hence the number of retain updates between
consecutive forget steps and their spacing within an epoch, **candidate** forget
exposure (1,240 / 3,080 / 12,120 per epoch), and the difficulty of the
underlying classification problem. A K comparison therefore moves a bundle of
correlated quantities, and no cell in this project separates them.

**BatchNorm, stated precisely — the two phases differ.** During **baseline
training** the backbone is in train mode, so BatchNorm running statistics are
updated, and larger K means more batches and a different data distribution: the
trained backbones therefore differ across K, and this is part of what "the
training population changes" means. During **classifier-only unlearning** the
backbone is held in `eval()` mode, so BatchNorm uses its stored running
statistics and **does not update them** — no BatchNorm exposure accrues in the
unlearning phase at any K. Earlier drafts listed "BatchNorm exposure" among the
things varying with K without separating the phases; only the baseline-training
phase is meant.

**Scope of "frozen backbone".** Freezing is **within each cell**: a cell's
backbone is identical at its own epochs 0–3. It does **not** mean the backbone
is the same across cells. Each (K, objective) pair has its **own separately
trained backbone**, so trained weights are not held identical across K or across
objectives, and cross-K or cross-objective comparisons of *levels* rest on
different feature extractors.

**Does not assert.** That class count changes forget-class erasure. The
ordering rests on **three** K points (K=500 absent — it failed the 2.00pp
fairness gate at 2.36pp and was not tuned to rescue it), adjacent K overlap per
cell, and at epoch 3 K=100 includes a negative cell under the fixed centre
(identity 00524, −0.0226). K covaries with retain-set size, retain steps per
epoch, active-step spacing, **baseline-training** BatchNorm exposure (the
unlearning phase runs the backbone in `eval()` mode and accrues none) and task
difficulty; candidate
forget exposure varies 1240/3080/12,120 per epoch while only the *active* count
(360) is held fixed. The counterfactuals are **interventions on the metric, not
on training**; `A` is nonlinear in both arguments, so the four corners are
arithmetic leftovers, never causal contributions.

---

### C7. K=100 seed-1 replication detail: gate, contrast rules, and the fc29 sign disagreement

**Claim.** The four-identity mean fixed-epoch K=100 DiC recurs in direction and
similar magnitude across two pipeline seeds, while identity-level contributions
vary substantially.

**Evidence.** Means in C5 agree to ~0.008 (centred) and ~0.011 (uncentred) at
every epoch. Identity level, centred, epoch 3: fc29 contributes +0.1382 at
seed 0 but +0.0173 at seed 1; fc95 moves −0.0214 → +0.1276. Baseline
per-identity difficulty also shifts — identity 00524 goes from 6–7/10 to
**10/10** in both heads.

**Fairness gate passed at both seeds.** Seed 1: CE 548/980 = 55.9184%,
ArcFace 560/980 = 57.1429%, gap **1.2245pp ≤ 2.00pp**. Seed 0: 0.2041pp.
Final-epoch checkpoints, no selection; neither head retuned.

**The two contrasts can disagree in sign.** At **matched outcome** — each
objective read at its own first 0/10 epoch — **seed-1 fc29 centred DiC is
−0.00727364360827909**, negative, despite positive fixed-epoch contrasts at all
three epochs. All attained uncentred matched contrasts are positive. Matched
sign counts: centred 3/3 positive (seed 0, n=3), **3 positive / 1 negative**
(seed 1, n=4).

**Exposure is unequal in 4 of the 7 attained pairs** (seed 0 fc0, fc29; seed 1
fc0, fc29 — ArcFace +1 epoch each); 3 are equal. So a matched-outcome contrast
confounds head with exposure and is not interchangeable with fixed-epoch DiC.

**Does not assert.** Population-level robustness. Two seeds give no usable
variance estimate for a seed effect, so no inferential statement is made here.
The eight seed-1 cells are **four identities × two
objectives at one seed** — not eight independent replications; within a seed
they share one backbone per head, one split, one baseline checkpoint. Cross-seed
means of *matched* contrasts compare **different identity sets** (3 attained at
seed 0, 4 at seed 1); restricting to the common attained set {0, 29, 60} gives
+0.080129 vs +0.032539, but that set is **selected on attainment in both
seeds** and is not an unbiased subset.

**Also note:** a negative DiC is **not** an NC3 sign reversal. There remain
zero sign reversals.

---

### C8. Output forgetting is not uniformly faster under either head, and the attainment table must be quoted rather than summarised

**Claim.** CE generally reaches 0/10 at epoch 1; ArcFace is often one epoch
slower. Stated only as observed attainment, not as a rate.

**Evidence.** K=100: CE attains at epoch 1 in 4/4 identities at seed 1 and 3/4
at seed 0; ArcFace attains in 4/4 at seed 1 (epochs 2, 2, 1, 1) and 3/4 at
seed 0 (epochs 2, 3, 1, and one non-attainment). Across the sweep, CE reaches
zero at epoch 1 in 11 of 12 cells; ArcFace is slower in 7/12, faster in 1
(K=1000 fc95). No identity attains later at seed 1 than at seed 0 in either
objective.

**Does not assert.** A rate or a speed ratio. Four identities on an integer
epoch grid of {1, 2, 3}, on a **ten-image** forget-class evaluation (0.1
granularity at K=100), cannot support one.

---

### C9. Uncentred-NC3 baseline offsets are an observed head-associated difference, present before any unlearning

**Claim.** In every setting measured, models trained with the ArcFace head show
a baseline uncentred NC3 near −0.88 and models trained with the CE head near
+0.51 to +0.72. This offset is **observed at baseline, before any unlearning**,
and is therefore not produced by the unlearning procedure. It is reported as an
**objective/head-associated difference**; head and its training configuration
(scale *s*, margin *m*, warmup) covary, so no stronger attribution to the loss
function alone is claimed.

**Evidence** (`nc3_uncentred_mean` at baseline, re-read from each run's own
`results.jsonl`):

| dataset | CE | ArcFace |
|---|---|---|
| CIFAR-10 | **+0.7211** | **−0.8786** |
| faces-1000 | +0.6026 | −0.8833 |
| faces-100 historical | +0.6416 | −0.8778 |
| K=100 nested, seed 0 | +0.5205 | −0.8946 |
| K=100 nested, seed 1 | +0.5149 | −0.8945 |

So CE spans **+0.51 to +0.72** and ArcFace **−0.878 to −0.895** — opposite
sides of zero on every dataset, before any unlearning. Per-cell seed-1 ArcFace
epoch-0 values run −0.8904 to −0.8990.

**An observed structural correlate, seed 0 only.** In the seed-0 decomposition
of stratum A, ArcFace's retain-weight centre is nearly as long as the forget
weight (`|c_0|/|w_0|` ≈ 0.96–1.01) while CE's is nearly the zero vector
(0.0012–0.0248) — a shared-component offset consistent with the sign difference.
This is a **measured correlate in one seed on one lineage**, not a demonstrated
cause, and no seed-1 decomposition exists to corroborate it.

**Consequence for the write-up.** An uncentred **level** difference between
objectives must not be read as a difference in unlearning quality — the
objectives begin in different reference frames. Describing the offset itself,
as this claim does, is legitimate and necessary. This is the single most likely reviewer trap in the paper, and
the guard belongs in the methods section, not a footnote.

---

### C10. The positive CIFAR ArcFace-minus-CE probe-gap pattern does not recur in the face settings

**Claim.** On CIFAR-10 the paired ArcFace-minus-CE `probe_gap_to_retrain*`
difference is positive in 4 of 4 classes (mean +0.2405). **That pattern does
not recur in either face setting**: on faces-1000 the paired differences are
**non-positive in all four identities** (3 negative, 1 exactly zero, mean
−0.0750), and on the historical faces-100 lineage they are **mixed in sign** (2
positive, 2 negative, mean −0.0134). Reported descriptively across settings:
the design does not isolate why, since domain, resolution, class count, images
per identity, identity membership and test-set denominator all covary.

**Evidence** (macro mean over 4 classes/identities, seed 0, n=4 per group):

| group | macro mean | SD | pos/zero/neg | pooled den |
|---|---|---|---|---|
| CIFAR-10 CE | +0.1042 | 0.0391 | 4/0/0 | 4000 |
| CIFAR-10 ArcFace | **+0.3448** | 0.0421 | 4/0/0 | 4000 |
| faces-100 CE | −0.0644 | 0.0361 | 0/0/4 | 110 |
| faces-100 ArcFace | −0.0778 | 0.1361 | 1/1/2 | 110 |
| faces-1000 CE | +0.0250 | 0.0957 | 2/1/1 | 40 |
| faces-1000 ArcFace | −0.0500 | 0.1291 | 1/1/2 | 40 |

Paired ArcFace-minus-CE, computed from exact integer counts over each
identity's own denominator (a float subtraction renders the faces-1000 fc794
rational zero as a spurious negative):

| setting | values | mean | pos / zero / neg |
|---|---|---|---|
| CIFAR-10 | +0.2600, +0.3020, +0.2110, +0.1890 | **+0.2405** | 4 / 0 / 0 |
| faces-100 | +0.1852, −0.1600, +0.0323, −0.1111 | −0.0134 | **2 / 0 / 2 (mixed)** |
| faces-1000 | −0.1000, −0.1000, −0.1000, 0.0000 | −0.0750 | **0 / 1 / 3 (non-positive)** |

The faces-1000 fc794 value is a **rational zero** — both objectives give
exactly +1/10 there (CE 8/10 vs 7/10; ArcFace 6/10 vs 5/10) — and is recorded as
`0.0000`, not as a signed near-zero.

**Does not assert.** **Equivalence of the objectives in the face settings, or
the absence of a difference there** — faces-1000's paired differences are
uniformly non-positive and faces-100's are mixed, which is *not* the same as
"no difference"; what does not recur is specifically CIFAR's **positive**
pattern. **No inferential claim is supported by the present analysis**: n=4
non-independent identities per group, one seed, and denominators of 10 (faces-1000)
or 25–31 (faces-100). Representation erasure — the probe is fitted post hoc on
labelled forget-class examples, and the unlearning backbone is frozen so there
is no feature change to detect. A negative gap is a statement about the
*reference*, not about better unlearning. Resolution differs by group
(CIFAR ~1000 test images/class; faces-100 25–31; faces-1000 and K=100 exactly
**10**, so gaps there move in whole-image steps of 0.1, while faces-100's
25–31 give steps of 0.032–0.040). The two face settings are
reported separately and **never combined**.

---

### C11. Narrow mechanism controls, each scoped to what it tested

**C11a. Frozen vs trainable backbone (CIFAR only).** For ArcFace, class 0,
seed 0, on CIFAR-10, centred NC3 reversal occurred with a frozen backbone
(−0.9107) but **not** with a trainable one (+0.7653), from a +0.9692 baseline,
with both members reaching zero output forgetting at epoch 1 and matching
training-trace hashes. **faces-1000 does not support this**: neither member
reversed (frozen +0.9789, full +0.2221). Freezing is not the operative variable
there. The faces full-model member also lost 5.5pp retain accuracy at the
matched point (8.8pp by epoch 3), so its movement numbers are confounded by
degradation; no learning rate was tuned.

**C11b. Dose is a tradeoff, not free forgetting.** The dual-dose faces-1000
intervention preserved retain utility and sharply reduced feature movement, but
the full-model member did **not** reach complete output forgetting — 1 of 10
forget-class test images remained correct at epoch 3. There is therefore **no
matched-outcome full-versus-frozen NC3 comparison** for it.

**C11c. Coefficient mass alone does not determine the outcome.** The four-cell
O/D/A/B dose-component matrix separates active-step cadence from loss
coefficient; the dose changed what the loop did with the stream, never the
stream (identical candidate trace hashes throughout). The dose grid is closed.

---

### C12. Candidate methodological lessons

**Framing.** These are **candidate** methodological lessons drawn from this
project's own measurement experience. They are **not** established novel
contributions: no literature check has been done on whether any is already
known, and items 3 and 4 are project-specific diagnostics rather than general
results. Whether any belongs in the paper — and with what novelty framing — is a
decision for Dr Rawat, and would need the outstanding novelty searches first.

1. **Centring contamination.** Standard NC3 centres classifier weights by their
   global mean; flipping one class's weight shifts the centring for *every*
   class, making untouched retain classes appear misaligned. Measured: retain
   alignment fell 1.00 → **0.94** with only class 0 perturbed — a six-point
   artefact that reads as real spillover. Fix: exclude the forget class from the
   centring reference.
2. **The two conventions disagree under an exact flip** — uncentred −1.00,
   centred −0.71 on the same model. Both must be reported with the convention
   stated.
3. **Observed stability of NC3 under class-mean subsampling, in the runs
   tested.** Recomputing CIFAR's epoch-1 value from 40-image class means over 20
   trials gave mean −0.9008, SD 0.0575, with 0/20 sign changes; the tested face
   models' non-flips likewise persisted under half-pool subsampling (+0.94 to
   +0.98). **Scope:** this establishes stability **only for the specific runs
   tested** (fc0, seed 0, epoch 1, on CIFAR-10, faces-100 and faces-1000) under
   **those particular subsampling procedures and trial counts**. It does not
   rule out estimator noise as a contributor in untested cells, and it does not
   establish that estimator noise is "not the limiting factor" anywhere else.
   The other seven face identities rest on a single recorded trajectory each.
4. **A cosine-schedule `T_max` truncation** silently left every warmup run
   (i.e. every ArcFace run) on an incomplete schedule, ending at lr 0.00670
   rather than 0.00000. Fixing it moved CIFAR ArcFace seed-1 test accuracy
   91.71% → 93.09% and changed a result that had been attributed to seed
   variance. All pre-fix ArcFace points were rebuilt, not reinterpreted. This is
   an implementation-hygiene lesson, almost certainly not novel.

---

## Part II — Claims that are NOT available

State these as scope, not as apology. Each is blocked by a specific, named gap.

| not claimable | why |
|---|---|
| **Any representation-erasure claim** | Backbone frozen in every main cell (code path); seed-0 replay shows class means bit-identical across epochs — a per-class summary, **not** per-feature equality — and no such check exists for seed 1. The probe metric measures post-hoc linear decodability after fitting on labelled forget-class examples, not erasure. |
| **Class-count causality** | C6 rules out the obvious mechanism and identifies a measurement-geometry origin; K covaries with retain-set size, step counts, active-step spacing, BN exposure and task difficulty. K=500 is absent. |
| **Inferential claims from the current design** | **No inferential claim is supported by the present analysis.** No statistical test is reported anywhere, and the tests a reader would reach for are **inappropriate for this design**, not merely unrun: treating the four identities as independent replicates is invalid (within a seed they share one backbone per head, one split, one baseline checkpoint), and two seeds give no usable variance estimate for a seed effect. Descriptive means, ranges and sign counts are reported instead. This is a statement about **what these data can support**, not that no test could ever be computed on any design — a properly powered study with independent seeds as the replication unit, and identities as a within-seed factor, would admit standard inference. |
| **Within-identity generalisation to unlearner-withheld *training* images** | See the dedicated note below — the test-split evaluation that *does* exist must not be confused with the `split_mode=subset` protocol that does not. |
| **Any CosFace result** | Never run. The head exists in `src/heads.py` and in `configs/cifar_cosface.yaml`; no experiment used it. |
| **faces-100 reproducibility from the repository alone** | The extraction seed for `casia-webface-folders-100id` was never recorded, and identity selection inside the converter is `rng.choice`-driven. That subset is not reconstructable from artifacts; it is a separate lineage from the nested sweep and must not be pooled with it. |
| **Direct post-unlearning backbone state equality** | Cells save no model state; only starting baseline checkpoints exist. The frozen-backbone claim rests on the code path (`eval()` mode, gradients off, optimizer over head parameters only) plus a **toy-model** buffer test (`Linear + BatchNorm1d(5)`, untrained, CPU — not ResNet-18). Constant `nc1_angular` is *consistent with* an unchanged mapping but is **not proof**: a scalar summary is many-to-one. No rerun is authorized. |
| **A novelty claim** | Two searches remain outstanding: **ACM DL**, and a **Semantic Scholar citation-graph pass** on the AISTATS paper and on *Neural Collapse by Design* (arXiv:2605.20302). Per project rule, no novelty claim may be written before both are done. |
| **Face-recognition SOTA relevance** | Deliberately not pursued. Reviewers should be pointed at condition matching, not LFW accuracy. |
| **"ArcFace fails to train"** | A tuning problem, resolved: s=64 was the wrong scale for 1000-way; s=96 with 5 warmup epochs closed the head-to-head gap to 1.94pp. Never reportable as a finding. |

---

### Note — two different generalisation questions, only one of which is untested

These are routinely conflated and must be kept apart.

**What the existing evaluation does provide.** `stratified_image_split` reserves
a per-identity test fraction (0.2) **within every identity, including the forget
identity**, and guarantees each identity contributes at least one image to each
side. Those test images are **never trained on by anything**. At K=100 that is
**10 held-out test images of the forgotten person per cell**; on faces-1000 it
is likewise 10, and on the historical faces-100 lineage 25–31. Every
`output_forget`, `output_retain` and `probe_forget` number in this project is
measured there. So the project **does** carry a **limited assessment of
within-identity generalisation to unseen images of the forgotten identity** —
limited by the denominator (**10 images gives 0.1 granularity at K=100,
faces-1000 and the nested sweep; the historical faces-100 lineage has 25–31 test
images per identity, i.e. steps of 0.032–0.040**), by n=4 identities,
and by the frozen backbone, but real, and it is the basis of every attainment
claim in C1.

**What is untested.** `make_forget_split(mode="subset")` partitions the forget
identity's **training** images: a `forget_fraction` is handed to the unlearning
method and **the remainder becomes `forget-heldout`** — images of the same
person that **the original model did train on** but **the unlearner never
sees**. Probing those answers a distinct question: *did the forgetting
generalise within the identity, from the images the unlearner was given to
images of the same person it was not given?* That partition is explicitly **not
a test set** — the original model trained on it.

Because every run in this project uses `split_mode=all`, the entire forget
identity is handed to the unlearner and `forget-heldout` is empty. **That
zero-sized internal partition means the subset-mode question was never asked; it
does not negate or weaken the external test-split evaluation above.** The
`split_mode=subset` experiment remains unrun.

---

## Part III — Global limitations

**Seeds.** Two seeds exist only on CIFAR-10 and at K=100. Everything else —
faces-1000, faces-100, K=250, K=500, K=1000, the decomposition, all dose and
movement controls — is **seed 0 only**.

**Replication unit.** The only valid replication unit in this design is the
**seed**, and there are at most two (CIFAR-10 and faces-nested K=100). The four
identities within a seed are a **within-seed factor**, not replicates: they
share one backbone per head, one train/test split and one baseline checkpoint.
Any analysis that treats them as independent would overstate precision.

**Metric resolution.** Test images per forget identity: CIFAR-10 ~1000 per
class; faces-100 25–31; **faces-1000 and the nested sweep exactly 10**. On the
ten-image evaluations (faces-1000 and the nested sweep) `output_forget` and
`probe_forget` move in whole-image steps of **0.1**; on faces-100's 25–31 images
the step is **0.032–0.040**, still coarse but not 0.1,
and "zero" means 0/10. `nc3_*_forget` reads ~40 training images and is not
resolution-limited in the same way (C12.3).

**Epoch sensitivity is real and is handled by policy, not left open.** Under the
settled policy (epoch 1 anchor, later epochs as sensitivity, counts always
quoted at a stated epoch) the numbers still move with epoch and must be read
with the epoch attached: on faces-100 the ArcFace reversal count is **1/4 at
epoch 1 and 2/4 by epoch 3**.

**Provenance tiers are not uniform.** Most runs are clean-tree, config-tracked,
commit-recorded. But: the faces-1000 ArcFace baseline ran with `--set
head.s=96.0` against a config reading `s: 64.0` at that commit, so its
effective configuration is recoverable only from recorded `argv`; both
faces-100 baselines and sweeps name configs that **were not tracked** at the
recorded commit. In every such case the effective configuration is fully
recorded in the run's own `config.json` and matches the current config files —
"reconstructed" means the run-time config *text* is not recoverable from the
repository at that commit, **not** that the numbers are in doubt.

**One historical artifact is partly contaminated.** `logs/cifar10_ce_seed0/results.jsonl`
predates the `RunDir` reuse guard and shares a file with a known mislabelled
`finetune_ep30` row. The two rows used for the probe matrix are correctly
labelled, but the file should not be mined further without care.

**Checkpoint backup is unverified.** All baseline checkpoints exist only on the
local filesystem of the machine that produced them. No durable, mirrored or
off-machine copy has been verified. SHA-256s are recorded in
`evidence/k100_seed1/provenance.json` and in the decision entries; **a hash is
not a backup.** Reported numbers stay checkable from the tracked evidence, but
bit-exact reproduction would need those checkpoints.

**Preserved tables do not preserve rerun capability.** `evidence/k100_seed1/`
lets the K=100 numbers be verified; it does not let the runs be reproduced.

---

## Part IV — Settled decisions, and what remains open

### Settled by the research lead — not to be reopened

1. **Framing** — qualified negative result plus mechanism analysis, with the
   central framing quoted at the top of this document.
2. **Evidence hierarchy** — the controlled face evidence (strata A and B) plus
   the seed-0 decomposition is the core; historical face and CIFAR results are
   explicitly qualified context; incompatible lineages are never pooled.
3. **Epoch policy** — show full recorded trajectories; **epoch 1 is the
   decomposition anchor**, later epochs are sensitivity; historical reversal
   counts are always quoted at a stated epoch.
4. **Comparison rules** — fixed-epoch and own-attainment contrasts stay
   separate and are never pooled.
5. **Figure set** — three main figures plus one contextual supplementary, as
   specified in `notes/figure_specs.md`.

### Genuinely open

1. **The two outstanding novelty searches** — ACM DL, and a Semantic Scholar
   citation-graph pass on the AISTATS paper and on *Neural Collapse by Design*
   (arXiv:2605.20302). **No novelty claim may be written before both are done.**
   Competitive risk stands: that group works on normalised-loss geometry and
   cites the face-margin family explicitly.
2. **Whether any C12 item is framed as a contribution** — depends on (1).
3. **Whether to spend remaining time explaining or broadening.** Any additional
   seed or K value requires a new research decision tied to a specific
   unresolved claim, per the standing handoff. Nothing is authorized now.

---

## Part V — Proposed paper outline and claim-to-figure mapping

Provisional; the epoch-convention and scope decisions in Part IV can still
change it.

| § | section | carries | figures |
|---|---|---|---|
| 1 | Introduction | the AISTATS illusion result; the margin-loss hypothesis; the qualified negative outcome stated up front | — |
| 2 | Background | neural collapse, NC1–NC3; ArcFace/CosFace angular margin; the predicted `w_k^un = −(1−γ)μ_k` flip | — |
| 3 | Method and measurement | the one-variable-per-comparison design; **both NC3 conventions and why they disagree**; centring contamination and the `exclude_from_centre` fix; DiC and `probe_gap_to_retrain*` labelled as project arithmetic; the dose schedule (m, λ) and that dose counts are not gradient magnitudes | — |
| 4 | Experimental settings | the four lineages and their non-equivalence; the head-fairness gate; **why K=500 is absent** | — |
| 5 | **Result 1 — output forgetting** | C1, C8 | **Fig 1** |
| 6 | **Result 2 — NC3 behaviour under a matched dose** | C5, C7, C3 | **Fig 3**, Fig 1 (centre column) |
| 7 | **Result 3 — mechanism (seed 0)** | C6a, C6b, C6c, C9 | **Fig 2** |
| 8 | What these metrics do not measure | C4, C10, the two-generalisation note | — |
| 9 | Contrasting settings | C2, C11 — explicitly non-equivalent | **Fig S1** (supplementary) |
| 10 | Limitations | Part II and Part III in full | — |
| 11 | Conclusion | the central framing verbatim | — |

**Claim → figure map** (claims with no figure are text-only by design):

| claim | figure | notes |
|---|---|---|
| C0 inventory / stratification | — | table in §4, sourced from `evidence/run_inventory/` |
| C1 ArcFace does not eliminate output forgetting | **Fig 1** (col 1) | stratified attainment table alongside |
| C2 reversal frequency differs across settings | **Fig S1** | supplementary only; protocol comparability not demonstrated |
| C3 reversal neither necessary nor sufficient | **Fig 1** (cols 1–2) | the "not sufficient" half needs CIFAR, so cite Fig S1 context |
| C4 three quantities are distinct | — | §8 text |
| C5 two-seed fixed-epoch recurrence | **Fig 3** (panel A) | |
| C6a rotation and norm (seed 0) | **Fig 2** (panels A–B) | |
| C6b reference-frame account of K ordering | **Fig 2** (panel C) | |
| C6c what the sweep does not control | — | §4 and Fig 2 caption |
| C7 seed-1 detail, fc29 sign disagreement | **Fig 3** (panels A vs B) | the figure's reason for existing |
| C8 attainment epochs | **Fig 1** (col 1 markers) | |
| C9 uncentred baseline offsets | — | §3 methods table; guard, not a result |
| C10 probe gaps | — | §8 table |
| C11 narrow mechanism controls | — | §9 text |
| C12 candidate methodological lessons | — | §3, framing pending novelty searches |

**Figures are specified in `notes/figure_specs.md`; no plotting code exists and
none is authorized yet.**

---

## Part VI — Evidence index

Claims map to `notes/decisions.md` entries (newer entries supersede older;
supersession notes are appended in place, historical text preserved):

| claim | primary entries |
|---|---|
| C1, C2, C3 | 2026-09-14 faces non-replication; 2026-09-14 100-identity; 2026-09-15 NC3 convention clarification; 2026-09-16 sweep; 2026-09-17 seed-1 |
| C4 | 2026-09-15 probe matrix; 2026-09-16 decomposition |
| C5, C6 | 2026-09-16 sweep; 2026-09-16 reference-frame decomposition |
| C7 | 2026-09-17 K=100 seed-1 replication (+ its audit corrections) |
| C8 | 2026-09-16 sweep; 2026-09-17 seed-1 |
| C9 | 2026-09-11 CIFAR pilot; 2026-09-16 decomposition (mechanism) |
| C10 | 2026-09-15 complete face probe-gap matrix |
| C11 | 2026-09-15 paired movement controls; dual-dose; dose-component matrix |
| C12 | 2026-09-09 two measurement bugs; 2026-09-14 LR schedule truncation; 2026-09-15 NC3 subsampling diagnostics |

**Run inventory.** `evidence/run_inventory/clfonly_cell_inventory.csv` — all 69
authoritative classifier-only cells with lineage, objective, seed, K, forget
identity, update mode, dose, recorded epochs, attainment, reversal epochs,
source artifact and reuse status. **Every aggregate count in this document is
derived from it and is stratified.**

**Figure specifications.** `notes/figure_specs.md` — four specified figures
(three main, one supplementary), with claim addressed, exact source artifacts,
panels/axes/units/series, missing-outcome handling, draft caption and
must-not-imply for each. Plotting code and rendered figures are deferred.

**Tracked numeric evidence.** `evidence/k100_seed1/` — 16 verbatim per-cell
trajectories (both seeds), four corrected tables, resolved configs,
`provenance.json` with a SHA-256 link per file. Raw run trees
(`logs/`, `logs100/`, `logs_classcount/`, `logs_classcount_decomposition/`,
`runs/`) are git-ignored by convention.

**Execution revisions.** K=100 seed-1 runs at `056b522`; decomposition
instrumentation at `653d55f`, replays on clean trees; nested sweep at
`51c38be`; CIFAR post-LR-fix rebuilds at `6015e37` / `ce47876`.
