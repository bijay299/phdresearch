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

- **NC3 centred** — cosine between the forget class's classifier weight and its
  class-mean feature, weights centred by the global mean **excluding the forget
  class**. This is the **only head-comparable** convention. Every reversal
  claim in this project is centred.
- **NC3 uncentred** — the same cosine uncentred; the convention the AISTATS
  prediction is stated in. Reported **only** as a within-head change from that
  model's own baseline. ArcFace's uncentred baseline is ≈ −0.88 *before any
  unlearning*, so a negative uncentred value is never by itself a flip.
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
cells (superseded by strata E); 10 `unlearn_pass1_nan_uncentred` cells (nan
defect fixed at `51c38be`); 25 `logs_classcount_decomposition/` cells — these
are **bitwise-identical replays** of stratum A (verified equal on
`output_forget`, `output_retain`, both NC3 conventions, `nc1_angular`,
`probe_forget`), so they are repeated observations of the same runs and carry
no additional evidential weight.

**Independent seed replications exist in exactly two places:** CIFAR-10
(seeds 0 and 1, both heads, 4 classes) and faces-nested K=100 (seeds 0 and 1,
both heads, 4 identities). Everything else is seed 0 only. **Four identities
within a seed are not four seeds** and are never counted as replications.

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
C's set. The only comparison in this project that moves **one** variable is
stratum A's nested sweep across K, and it shows **no** reversals at any K.

**Does not assert.** That reversal is or is not "a property of the loss" — that
phrasing is withdrawn as unsupported. That class count causes the difference.
That the settings are otherwise comparable enough to pool; **they are not, and
a cross-setting reversal summary is supplementary context only.**

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

**Evidence.** In every main unlearning cell the backbone is frozen
(`classifier_only`), so features cannot move by construction — verified in the
decomposition replay, where the class-mean matrix is **bit-identical** at
epochs 0–3 in all 24 cells. Separately, the probe metric cannot carry an
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

**C6a — rotation and norm (seed 0).** Forget-weight rotation, mean over four
identities:

| epoch | CE | ArcFace |
|---|---|---|
| 1 | 16.7–17.8° | 3.3–4.7° |
| 3 | 20.8–22.6° | 5.2–6.2° |

Forget-weight norm ratio: CE 0.830–0.874, ArcFace 0.996–1.001. So at seed 0,
CE's classifier rotates several times further and loses norm, while ArcFace's
rotates little and preserves norm. Norm does not enter the cosine; it is a
descriptive fact about the weight, not a driver of `nc3_*_forget`.

**C6b — the K ordering is a reference-frame effect, in this seed-0
decomposition.** Centred DiC orders K=100 < 250 < 1000, but:

- **Not reference drift.** Holding the retain-weight centre at epoch 0
  reproduces the ordering essentially unchanged (+0.0561/+0.1087/+0.1393 vs
  production +0.0574/+0.1093/+0.1393 at epoch 1). Moving *only* the centre
  gives +0.0004/+0.0002/+0.0003 — flat, two to three orders of magnitude below
  production. Largest residual 0.0068 against a largest production change of
  0.2007.
- **Not more rotation.** CE's forget weight rotates 17.8°/16.7°/17.4° at
  epoch 1 — **non-monotone, spanning ~1.8°**.
- **Where it enters.** The centred frame's reference is `f_0 − g_0`, and
  `|g_0|/|f_0|` for CE is 0.80/0.73/0.69 at K=100/250/1000. Both the baseline
  angle (31.9° < 34.2° < 39.1°) and the projected rotation (9.14° < 9.60° <
  11.32°) increase with K and reinforce; since a cosine's sensitivity to
  rotation is −sin θ, the ordering follows. The **uncentred** baseline angle is
  non-monotone (58.0° > 54.1° > 53.2°) and its changes inherit that — which is
  why the uncentred convention shows **no** K ordering.
- **Why each head's centre is inert, for opposite reasons.** CE's
  retain-weight centre is nearly the zero vector (`|c_0|/|w_0|` =
  0.0248/0.0104/0.0012), so its large angular movement moves the metric by
  ~1e-4. ArcFace's centre is nearly as long as the weight (0.96–1.01), so
  centring matters greatly to its *level*, but the centre barely moves
  (0.02–0.07°) and so contributes nothing to the *change*.

**Does not assert.** That class count changes forget-class erasure. The
ordering rests on **three** K points (K=500 absent — it failed the 2.00pp
fairness gate at 2.36pp and was not tuned to rescue it), adjacent K overlap per
cell, and at epoch 3 K=100 includes a negative cell under the fixed centre
(identity 00524, −0.0226). K covaries with retain-set size, retain steps per
epoch, active-step spacing, BatchNorm exposure and task difficulty; candidate
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
epoch grid of {1, 2, 3}, at 0.1 accuracy granularity, cannot support one.

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

**Consequence for the write-up.** Raw uncentred NC3 must never be compared
across heads. This is the single most likely reviewer trap in the paper, and
the guard belongs in the methods section, not a footnote.

---

### C10. Probe gaps separate the objectives on CIFAR-10 but not in either face setting, and the two face settings differ from each other

**Claim.** `probe_gap_to_retrain*` separates the two objectives on CIFAR-10 and
does not in either face setting. Reported descriptively across settings: the
design does not isolate why, since domain, resolution, class count, images per
identity, identity membership and test-set denominator all covary.

**Evidence** (macro mean over 4 classes/identities, seed 0, n=4 per group):

| group | macro mean | SD | pos/zero/neg | pooled den |
|---|---|---|---|---|
| CIFAR-10 CE | +0.1042 | 0.0391 | 4/0/0 | 4000 |
| CIFAR-10 ArcFace | **+0.3448** | 0.0421 | 4/0/0 | 4000 |
| faces-100 CE | −0.0644 | 0.0361 | 0/0/4 | 110 |
| faces-100 ArcFace | −0.0778 | 0.1361 | 1/1/2 | 110 |
| faces-1000 CE | +0.0250 | 0.0957 | 2/1/1 | 40 |
| faces-1000 ArcFace | −0.0500 | 0.1291 | 1/1/2 | 40 |

Paired ArcFace-minus-CE: CIFAR +0.2405 (4/4 positive); faces-100 −0.0134 (2
positive, 2 negative); faces-1000 −0.0750 (0 positive, 1 exact zero, 3
negative).

**Does not assert.** Representation erasure — the probe is fitted post hoc on
labelled forget-class examples, and the unlearning backbone is frozen so there
is no feature change to detect. A negative gap is a statement about the
*reference*, not about better unlearning. Resolution differs by group
(CIFAR ~1000 test images/class; faces-100 25–31; faces-1000 and K=100 exactly
**10**, so every face gap is a whole-image step). The two face subsets are
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
3. **Class-mean estimation at n≈40 is not the limiting factor.** Recomputing
   CIFAR's epoch-1 flip from 40-image class means over 20 trials gives mean
   −0.9008, SD 0.0575, 0/20 sign flips; face non-flips likewise survive
   half-pool subsampling (+0.94 to +0.98). The face/CIFAR divergence is real,
   not a starved estimator.
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
| **Any representation-erasure claim** | Backbone frozen in every main cell; class means bit-identical across epochs. The probe metric measures post-hoc linear decodability after fitting on labelled forget-class examples, not erasure. |
| **Class-count causality** | C6 rules out the obvious mechanism and identifies a measurement-geometry origin; K covaries with retain-set size, step counts, active-step spacing, BN exposure and task difficulty. K=500 is absent. |
| **Inferential claims from the current design** | No statistical test is reported anywhere, and the tests a reader would reach for are **inappropriate for this design**, not merely unrun: treating the four identities as independent replicates is invalid (within a seed they share one backbone per head, one split, one baseline checkpoint), and two seeds give no usable variance estimate for a seed effect. Descriptive means, ranges and sign counts are reported instead. This is a statement about **what these data can support**, not that no test could ever be computed on any design — a properly powered study with independent seeds as the replication unit, and identities as a within-seed factor, would admit standard inference. |
| **Forget-set generalisation (within-identity)** | A **held-out test split does exist and is used throughout** — `stratified_image_split` reserves ~20% of every identity's images, including the forget identity's, and all `output_forget` / `output_retain` / `probe_forget` numbers are measured on it. What is untested is a **different** form of generalisation: because `split_mode=all` throughout, every *training* image of the forget identity is handed to the unlearning method, so `forget-heldout` is 0 in every run. The untested question is whether forgetting **generalises to training images of the forget identity that the unlearner never saw** — the `split_mode=subset` experiment, never run. |
| **Any CosFace result** | Never run. The head exists in `src/heads.py` and in `configs/cifar_cosface.yaml`; no experiment used it. |
| **faces-100 reproducibility from the repository alone** | The extraction seed for `casia-webface-folders-100id` was never recorded, and identity selection inside the converter is `rng.choice`-driven. That subset is not reconstructable from artifacts; it is a separate lineage from the nested sweep and must not be pooled with it. |
| **Direct post-unlearning backbone state equality** | Cells save no model state; only starting baseline checkpoints exist. The frozen-backbone claim rests on the code path (`eval()` mode, gradients off, optimizer over head parameters only) plus a **toy-model** buffer test (`Linear + BatchNorm1d(5)`, untrained, CPU — not ResNet-18). Constant `nc1_angular` is *consistent with* an unchanged mapping but is **not proof**: a scalar summary is many-to-one. No rerun is authorized. |
| **A novelty claim** | Two searches remain outstanding: **ACM DL**, and a **Semantic Scholar citation-graph pass** on the AISTATS paper and on *Neural Collapse by Design* (arXiv:2605.20302). Per project rule, no novelty claim may be written before both are done. |
| **Face-recognition SOTA relevance** | Deliberately not pursued. Reviewers should be pointed at condition matching, not LFW accuracy. |
| **"ArcFace fails to train"** | A tuning problem, resolved: s=64 was the wrong scale for 1000-way; s=96 with 5 warmup epochs closed the head-to-head gap to 1.94pp. Never reportable as a finding. |

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
face sets `output_forget` and `probe_forget` move in whole-image steps of 0.1,
and "zero" means 0/10. `nc3_*_forget` reads ~40 training images and is not
resolution-limited in the same way (C12.3).

**Epoch convention is still unresolved** — see Part IV. It materially changes
headline numbers: on faces-100 the ArcFace reversal count is 1/4 at epoch 1 and
2/4 at epoch 3.

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

## Part IV — Decisions blocking the write-up

These are for Dr Rawat, not for the execution agent, and they interlock.

1. **Which epoch convention is primary?** Epoch 1 (matched output forgetting —
   what every CIFAR claim used) or end of the unlearning budget. Standing
   recommendation is epoch 1 primary with the final epoch as sensitivity, but
   it has never been settled. **It must be fixed once and applied to CIFAR,
   faces-1000, faces-100 and the nested sweep alike.** Open since 2026-09-14.
2. **Is a qualified negative result the paper?** The plausible paper is now
   "the mechanism is loss- and geometry-sensitive, and the classifier shortcut
   is not closed by angular margin", plus the mechanism analysis in C5/C6. Not
   "margin losses close the shortcut". Confirm before more compute.
3. **The two outstanding novelty searches** (ACM DL; Semantic Scholar
   citation-graph on the AISTATS paper and on *Neural Collapse by Design*).
   No novelty claim may be written first. Note the competitive risk: the
   *Neural Collapse by Design* group works on normalised-loss geometry and
   cites the face-margin family explicitly.
4. **Scope for the remaining time** — explain or broaden. Any additional seed
   or K value requires a new research decision tied to a specific unresolved
   claim, per the standing handoff.

---

## Part V — Evidence index

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
