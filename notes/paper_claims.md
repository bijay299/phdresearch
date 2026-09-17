# Paper claims and limitations

Synthesis of every claim the existing evidence supports, the convention each
claim depends on, and what each explicitly does not assert. Written 2026-09-17
from `notes/decisions.md` and the tracked evidence under `evidence/`. **No new
experiment was run to produce this document.**

Target: CVPR 2027 — registration Nov 10 2026, submission Nov 16 2026.

**Status of the research question.** The original framing offered three mutually
exclusive outcomes: (1) unlearning under ArcFace must move features, (2)
unlearning under ArcFace fails, (3) a third, uncharacterised mechanism. The
evidence lands on **none of the three as stated**. What it supports is a
*qualified negative result plus a mechanism analysis*: the shortcut is not
closed, output forgetting succeeds under both heads, and the head difference
appears in *how far the classifier rotates* — which is a different and narrower
claim than any of the three anticipated outcomes. The write-up has to say that
plainly rather than force the result into the original trichotomy.

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

### C1. The universal hypothesis is unsupported: ArcFace does not prevent classifier-only output forgetting

**Claim.** Classifier-only random-label unlearning drives forget-class output
accuracy to zero under ArcFace as well as under CE, across every dataset and
class count tested. The hypothesis that a normalised angular-margin loss closes
the classifier shortcut is **not supported**.

**Evidence.** Attainment of `output_forget = 0` within the canonical 3-epoch
budget, ArcFace cells:

| dataset / study | ArcFace cells | attained zero | source |
|---|---|---|---|
| CIFAR-10 (4 classes × 2 seeds) | 8 | 8/8, by epoch 1 | 2026-09-14 seed-0 rebuild; 2026-09-14 seed-1 re-run |
| faces-1000 (4 identities) | 4 | 4/4 | 2026-09-14 faces entry; 2026-09-15 probe matrix |
| faces-100 historical (4 identities) | 4 | 4/4, epochs 1–3 | 2026-09-14 100-identity entry |
| nested K sweep, K=100/250/1000 | 12 | **11/12** | 2026-09-16 sweep |
| K=100 seed 1 | 4 | 4/4 | 2026-09-17 replication |

**The one exception, stated explicitly:** K=100, seed 0, ArcFace, fc95
(identity 00524) ends at 1/10 and never reaches zero inside three epochs. It
does **not** recur at seed 1, where the same identity attains at epoch 1. The
budget was not extended to resolve it.

**Does not assert.** That forgetting is *equally fast* under both heads (see
C8). That output-level zero means erasure (see C4). That this generalises to
unlearning methods other than classifier-only random-label, to CosFace, or to
full-model unlearning at a tuned learning rate.

---

### C2. Centred NC3 sign reversal is dataset-dependent, not a property of the loss

**Claim.** The CIFAR-10 finding — ArcFace's forget-class weight flipping
negative — does **not** replicate on faces. Reversal frequency varies from
8/8 to 0/24 across datasets with the loss held fixed.

**Evidence** (centred convention, `random_label_clfonly`, epoch 0 → epoch 1
unless noted):

| dataset | head | n | reversals |
|---|---|---|---|
| CIFAR-10 | ArcFace | 8 | **8/8** |
| CIFAR-10 | CE | 8 | 0/8 |
| faces-100 historical | ArcFace | 4 | **1/4** at ep1; 2/4 at ep3 |
| faces-1000 | ArcFace | 4 | **0/4** |
| nested K sweep (K=100/250/1000) | both | 24 | **0/24**, any epoch |
| K=100 seed 1 | both | 8 | **0/8**, any epoch |

So across the 16 K=100 cells at two seeds, and the 24 sweep cells, there are
**zero** sign reversals in either convention. The `w_k^un = −(1−γ)μ_k` flip the
AISTATS theory predicts does not appear under this method and budget on faces.

**Depends on.** The **centred** convention. Uncentred values must not be used
for this claim: ArcFace starts near −0.88 everywhere, so uncentred "negatives"
are not reversals.

**Does not assert.** That class count causes the difference (C6 rules out the
obvious mechanism). Domain, resolution, scale *s*, images per identity and
class count all covary between CIFAR and the face sets, and between the two
face lineages. No comparison in this project isolates any one of them.

---

### C3. Sign reversal is neither necessary for output forgetting nor sufficient evidence of representation erasure

**Claim, two halves, both supported.**

**Not necessary.** Both face subsets reach `output_forget = 0` with reversal in
0/4 (faces-1000) and 1/4 (faces-100) identities. Zero output forgetting happens
without any centred sign reversal.

**Not sufficient.** On CIFAR-10, ArcFace reverses in 8/8 cells **under a frozen
backbone**, where the representation is unchanged by construction. A reversal
therefore cannot be read as evidence that features moved.

**Source.** 2026-09-15 NC3 convention clarification, which states this as the
supported thesis under one convention throughout.

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

### C5. Under a matched forget dose, CE's forget-class weight moves substantially more than ArcFace's

**Claim.** At equal forget dose (m=9 active steps/epoch, λ=1, 3 epochs), CE's
classifier-to-class-mean cosine falls markedly while ArcFace's barely moves.
This is the project's most robust positive finding.

**Evidence — metric level.** Fixed-epoch DiC, centred, mean over four
identities:

| epoch | seed 0 | seed 1 |
|---|---|---|
| 1 | +0.0574 | +0.0595 |
| 2 | +0.0648 | +0.0708 |
| 3 | +0.0607 | +0.0686 |

Sign counts (4 identities × 3 epochs = 12 per seed): centred **10/12** positive
at seed 0 (the two negatives are fc95 at epochs 2 and 3 — **one cell observed
twice, not two cells**) and **12/12** at seed 1; uncentred **12/12** at both
seeds. Larger K gives larger centred DiC (+0.1093 at K=250, +0.1393 at K=1000,
epoch 1) — but see C6 before attributing that to K.

**Evidence — mechanism level.** Forget-weight rotation, mean over four
identities, from the decomposition:

| epoch | CE | ArcFace |
|---|---|---|
| 1 | 16.7–17.8° | 3.3–4.7° |
| 3 | 20.8–22.6° | 5.2–6.2° |

CE's forget-weight norm ratio is 0.830–0.874; ArcFace's is 0.996–1.001. So
ArcFace's classifier both rotates less and preserves its norm.

**Does not assert.** **Positive DiC is not superior unlearning.** It says CE's
cosine moves further under a matched dose; it is not a claim about erasure
quality, privacy, or which head one should prefer. It is also not a claim about
gradient magnitudes — the dose figures are loop-structure counts, explicitly
labelled "not a gradient magnitude" in the artifacts.

---

### C6. The apparent K ordering is measurement geometry, not increasing classifier rotation

**Claim.** Centred DiC orders K=100 < 250 < 1000, but this is **not** because
the forget weight rotates more at larger K, and **not** because the centring
reference drifts. It is the sensitivity of centred NC3 to a **fixed** reference
frame whose geometry varies with K.

**Evidence.**
- **Not reference drift.** Holding the retain-weight centre at epoch 0
  reproduces the ordering essentially unchanged (+0.0561/+0.1087/+0.1393 vs
  production +0.0574/+0.1093/+0.1393 at epoch 1). Moving *only* the centre
  yields +0.0004/+0.0002/+0.0003 — flat, two to three orders of magnitude
  below production. Largest residual 0.0068 against a largest production change
  of 0.2007.
- **Not more rotation.** CE's forget weight rotates 17.8°/16.7°/17.4° at epoch
  1 and 22.6°/20.8°/21.5° at epoch 3 — **non-monotone, spanning ~1.8°**.
- **Where it does live.** The centred frame's reference is `f_0 − g_0`, and
  `|g_0|/|f_0|` for CE is 0.80/0.73/0.69 at K=100/250/1000. Both the baseline
  angle (31.9° < 34.2° < 39.1°) and the projected rotation (9.14° < 9.60° <
  11.32°) increase with K and reinforce. Since a cosine's sensitivity to
  rotation is −sin θ, the ordering follows. The **uncentred** baseline angle is
  non-monotone (58.0° > 54.1° > 53.2°) and the uncentred changes inherit that
  non-monotonicity — which is why the uncentred convention shows **no** K
  ordering.
- **Why CE's centre is inert:** CE's retain-weight centre is nearly the zero
  vector (`|c_0|/|w_0|` = 0.0248/0.0104/0.0012), so its large angular movement
  moves the metric by ~1e-4. **Why ArcFace's is inert:** ArcFace's centre is
  nearly as long as the weight (0.96–1.01) so centring matters greatly to its
  *level*, but the centre barely moves (0.02–0.07°), so it contributes nothing
  to the *change*.

**Per the handoff, this is closed. Do not reopen the decomposition without
specific contradictory evidence.**

**Does not assert.** That class count changes forget-class erasure. The
ordering rests on **three** K points — K=500 is absent because it failed the
2.00pp fairness gate (ArcFace exceeded CE by 2.36pp) and was not tuned to
rescue it — adjacent K overlap per cell, and at epoch 3 K=100 includes a
negative cell under the fixed centre (fc95, −0.0226). K also covaries with
retain-set size, retain steps per epoch, spacing of the nine active steps,
BatchNorm exposure and problem difficulty. Candidate forget exposure varies
too (1240/3080/12,120 presentations per epoch); only the *active* count (360)
is held fixed.

---

### C7. Two-seed replication at K=100: the four-identity mean recurs; identity-level contributions do not

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

**Does not assert.** Population-level robustness. Two seeds support no
significance test. The eight seed-1 cells are **four identities × two
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

### C9. The uncentred-NC3 baseline offset is a property of the loss, not of unlearning

**Claim.** ArcFace's uncentred NC3 sits near −0.88 *before any unlearning*, on
every dataset; CE's sits near +0.50 to +0.72. This is a property of the
objective's learned weight geometry, present at baseline.

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
epoch-0 values run −0.8904 to −0.8990. **Mechanism identified:**
ArcFace's retain-weight centre is nearly as long as the forget weight itself
(`|c_0|/|w_0|` ≈ 0.96–1.01) — a shared-component offset — where CE's centre is
nearly the zero vector.

**Consequence for the write-up.** Raw uncentred NC3 must never be compared
across heads. This is the single most likely reviewer trap in the paper, and
the guard belongs in the methods section, not a footnote.

---

### C10. Probe gaps: the CIFAR head separation does not transfer to faces, and the two face subsets disagree

**Claim.** `probe_gap_to_retrain*` separates the heads on CIFAR-10 and does not
on either face subset.

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

### C12. Measurement integrity results worth reporting as contributions

These are methodological findings, not incidental bug fixes, and at least the
first two belong in the paper.

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
   variance. All pre-fix ArcFace points were rebuilt, not reinterpreted.

---

## Part II — Claims that are NOT available

State these as scope, not as apology. Each is blocked by a specific, named gap.

| not claimable | why |
|---|---|
| **Any representation-erasure claim** | Backbone frozen in every main cell; class means bit-identical across epochs. The probe metric measures post-hoc linear decodability after fitting on labelled forget-class examples, not erasure. |
| **Class-count causality** | C6 rules out the obvious mechanism and identifies a measurement-geometry origin; K covaries with retain-set size, step counts, active-step spacing, BN exposure and task difficulty. K=500 is absent. |
| **Any statistical significance** | No test is run anywhere in the project and none would be supportable: one seed almost everywhere, two at K=100 and on CIFAR, n=4 identities per group. No distributional assumption is made about any reported spread. |
| **Held-out forgetting generalisation** | `split_mode=all` throughout, so `forget-heldout` is 0 in every run. The subset-mode generalisation experiment was never run. |
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

**Sample size.** Four identities or classes per group throughout. n=4.

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

**Tracked numeric evidence.** `evidence/k100_seed1/` — 16 verbatim per-cell
trajectories (both seeds), four corrected tables, resolved configs,
`provenance.json` with a SHA-256 link per file. Raw run trees
(`logs/`, `logs100/`, `logs_classcount/`, `logs_classcount_decomposition/`,
`runs/`) are git-ignored by convention.

**Execution revisions.** K=100 seed-1 runs at `056b522`; decomposition
instrumentation at `653d55f`, replays on clean trees; nested sweep at
`51c38be`; CIFAR post-LR-fix rebuilds at `6015e37` / `ce47876`.
