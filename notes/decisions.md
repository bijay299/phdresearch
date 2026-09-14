# Decision log

Newer entries supersede older ones. Record the date on everything.

---

## 2026-09-09 — Direction fixed

**Question:** Is the illusion of unlearning a property of unlearning, or of
cross-entropy?

Testing whether feature–classifier misalignment (Gao et al., AISTATS 2026)
survives under margin-based losses (ArcFace, CosFace).

**Advisor:** Rawat, target CVPR 2027 (registration Nov 10, submission Nov 16).

**Why this and not the alternatives.** Four directions were killed on novelty
grounds first:

| Direction | Killed by |
|---|---|
| Retrain-free representation metric | RULER (arXiv:2605.27569) oracle-free metric M4 |
| Identity-level vs sample-level forgetting | One-Shot Unlearning (2407.12069), PPU-Bench cross-image test (2605.08800), Embedding Dispersion (2512.13317) |
| Decode-order attacks on diffusion LMs | DIJA, MaskForge, TrajHijack; defences A2D, DiffuGuard |
| General identity unlearning evaluation | MUFAC/MUCAC, NoMUS, UES |

What survived: nobody has tested unlearning under margin-based losses, and
nobody has tested this mechanism outside standard classification. The AISTATS
authors' own stated next step is diffusion and language models — not this.

**Search scope so far:** arXiv, general web, `awesome-llm-unlearning`, IEEE
Xplore. **Outstanding:** ACM DL, Semantic Scholar citation-graph pass on the
AISTATS paper and on Neural Collapse by Design (arXiv:2605.20302). Do these
before writing any novelty claim.

**Known competitive risk:** the Neural Collapse by Design group works on the
geometry of normalised losses and cites the face-recognition margin family
explicitly. Connecting that to unlearning is one step from where they are.

---

## 2026-09-09 — Two measurement bugs found before any experiment

Both caught by `src/test_metrics.py` on synthetic data with known answers.

**1. Centring contamination.** Standard NC3 centres weights by the global
weight mean. Flipping one class's weight moves that mean and shifts the
centring for every class. Measured: retain-class alignment fell 1.00 → 0.94
with only class 0 perturbed. That is a 6-point artefact that reads as real
spillover damage.

Fix: `exclude_from_centre=<forget_class>`.

**2. Centred vs uncentred disagree.** Under an exact weight flip the
uncentred cosine is −1.00 and the centred one is −0.71. The theory's claim
is about the uncentred weight. Report both and state the convention.

---

## 2026-09-10 — CASIA-WebFace extracted, retrain baseline built, three more pipeline bugs fixed

**Decided:** CASIA-WebFace is the dataset.
**Because:** the InsightFace RecordIO packaging is on Kaggle, so no Howard
access is needed. Extracted and verified end to end today.
**Supersedes:** the open "CASIA-WebFace or VGGFace2, blocked on Howard access"
item below.

### What was built

- **`retrain` baseline** in `unlearn.py` — fresh backbone and head from random
  init, trained on the retain set only, registered in `METHODS` and wired into
  `run_experiment.py`. Skippable via `unlearn.retrain_reference` because it is
  a full training run. Runs last, since it reseeds the global RNG. Verified it
  inherits the smoke-shortened epoch count (`ep 1/1`, not the configured 30).
- **`Makefile`** — `test` / `smoke` / `pilot`, so the common commands live in
  one place. `make test` fails if either suite fails.
- **`src/test_data.py`** — regression tests for the splitting and loader logic.
  `make test` now runs **20 groups: 8 in `test_metrics.py`, 12 in
  `test_data.py`.**
- **Per-stage timing** in `run_experiment.py` — each unlearning call and each
  `evaluate()` call timed separately, logged and written to `results.jsonl` as
  `method_time_s` / `eval_time_s`. This is what located bug 4 below.

### Dataset

CASIA-WebFace, InsightFace RecordIO packaging from Kaggle.

- `property`: **10,572 identities at 112×112**.
- `train.rec`: **490,623 image records** (IRHeader `flag=0`) plus **10,573
  bookkeeping records** (`flag=2` — one global header and one index-range
  marker per identity). Sums to 501,196, matching `train.idx` exactly.
- Published CASIA figures are 494,414 images / 10,575 identities. The gap is
  **likely alignment-stage detection failures — inference, not confirmed.**
- `train.lst` (494,149 lines) matches neither count. Treated as a
  pre-alignment sidecar and left unused.
- Converted with **pure `struct` + PIL, no mxnet** — the per-image label is in
  the record header, so the unmaintained dependency is avoidable entirely.
- Extracted **1000 identities / 36,364 images**, seed 0. Seeded sampling
  verified byte-identical across two runs.
- Within-identity pixel correlation: **mean 0.21, range 0.06–0.48**. That is
  real appearance variation, not near-duplicate burst frames — so the
  within-identity generalisation experiment is viable.

### Five bugs found before any real experiment

**All five share one shape: something restricted on one side of the pipeline
and not the other, producing plausible wrong numbers with no error raised.**
That is the pattern worth remembering — none of these announced themselves.

1. **NC3 centring contamination.** Global weight centring let one flipped
   class drag retain-class alignment 1.00 → 0.94. Fixed with
   `exclude_from_centre`. *(Logged 2026-09-09.)*
2. **Centred vs uncentred NC3 disagree** — −0.71 vs −1.00 on an exact flip.
   The theory concerns the uncentred weight. Both now reported, convention
   stated. *(Logged 2026-09-09.)*
3. **`--smoke` restricted the training loader but not the forget/retain
   split.** Unlearning trained on 45k retain images while the original model
   had trained on 400. Fixed via `data.restrict_split()`.
4. **`--smoke` restricted the training loaders but not `test_loader`.** Every
   `evaluate()` ran a full 10k-image forward pass — 65s of every 71s
   condition, and flat across all eight conditions, which is what gave it
   away. Fixed via `stratified_subsample` on the test targets. Smoke run
   9m24s → 1m42s.
5. **`build_datasets` constructed the faces test set from the same directory
   with the same kwargs and seed.** Verified **36,364/36,364 overlap**. Every
   faces test metric would have been train accuracy — including the "match
   test accuracy between heads" fairness gate, which is the first thing a
   reviewer attacks. Fixed with a by-image split within each identity, taken
   from a single `FaceFolder` scan.

Bug 5's fix is constrained by the experiment, not just by hygiene: the split
must be **by image, not by identity**, or the test set contains no images of
the forgotten person and forget accuracy cannot be computed at all. One scan
with shared views also removes the label-ordering hazard — two independent
`FaceFolder` constructions cannot guarantee label *k* means the same person.

**Every fix has a regression test with a negative case proving the test can
fail.** A regression test that cannot fail against the old behaviour is not
evidence of anything.

### Consequence

Faces train drops **36,364 → 29,103** after the test split (7,261 test). Any
earlier faces numbers are not comparable — though they were train accuracy in
disguise, so nothing of value is lost.

### Still open

- **`evaluate()` cost at face scale is not yet estimated** — a 1000-way linear
  probe over 29k samples. This gates whether the full experiment fits the
  compute budget. Measure before committing to a run plan.
- **Forget accuracy would be measured on 4–10 test images per identity.**
  Consider restricting eligible forget identities to those with ≥40 source
  images, so the primary outcome does not rest on single-digit samples.
- **`apply_smoke_budget` refactor**, including `max_classes`. Smoke keeps all
  1000 identities (it caps images per class, not class count), so `--smoke` is
  nearly a no-op on the faces config — ~90% of the dataset survives it.

---

## 2026-09-11 — CASIA-WebFace re-extracted with `--min-images 40`

**Decided:** re-extract at `/data/bijaypandey/archive/casia-webface-folders`
using `python scripts/convert_rec_to_folders.py --min-images 40` (the
script's default is 20).
**Because:** more headroom for the week 5-8 held-out generalisation split.
`make_forget_split(mode='subset')` needs enough images per identity that
both the forget and forget-heldout sides stay usable; 20 was thin once
`max_images_per_identity=50` and a forget fraction are applied on top.
**Supersedes:** the 2026-09-10 extraction (1000 identities / 36,364 images,
`--min-images` at the script default of 20). The underlying image set
differs between the two extractions -- do not assume identity or image
overlap; any numbers from the 36,364-image extraction are not comparable.

### Verified

- `FaceFolder` (`src/data.py`) loads it correctly: 1000 identities, 48,519
  images, matching the conversion script's own count.
- Per-identity image count: min 40, median 50, max 50, mean 48.52 -- nearly
  every identity hits the `--max-images-per-identity 50` ceiling, which is
  what the raised floor was for.
- Contact sheet at `notes/assets/casia_folders_contact_sheet.png` (3
  identities x 8 images each, one row per identity, labelled with folder
  id/label/n) -- confirmed real appearance variation (pose, lighting,
  expression) within each identity, not repeated/near-duplicate frames.
- Within-identity pixel correlation (grayscale, resized 64x64, Pearson r
  pooled over all image pairs within each of 20 random identities): **mean
  0.246, range 0.096-0.520**. Consistent with the 2026-09-10 extraction's
  0.21 mean -- same conclusion holds: real variation, not burst frames, so
  the within-identity generalisation experiment stays viable.

### Consequence

`configs/faces_ce.yaml` `data.root` updated to
`/data/bijaypandey/archive/casia-webface-folders` (`configs/faces_arcface.yaml`
inherits it).

---

## 2026-09-11 — Project moved to the shared server "hyperplane"

**Decided:** repo and data now live on hyperplane, not the laptop: repo at
`/data/bijaypandey/phdresearch`, dataset at `/data/bijaypandey/archive`
(`casia-webface-folders` extraction above, plus the raw `casia-webface`
RecordIO source).
**Because:** move to shared compute for real training runs.
**Supersedes:** any earlier assumption of exclusive, single-GPU, single-user
access to the machine running this code.

### Repo audit for old-machine references

Searched README.md, CLAUDE.md, `scripts/`, `configs/`, `notes/`, `.vscode/`
for hardcoded laptop paths (`Downloads`, `/Users/`, `~/phdresearch`, etc.).
**Found none** -- the codebase already used portable paths throughout:
`ROOT.parent / "archive" / ...` in `convert_rec_to_folders.py`, `./data` in
`base.yaml`, `${workspaceFolder}` in `.vscode/*.json`. The one path that did
need updating was `configs/faces_ce.yaml`'s `data.root`, fixed to the
absolute hyperplane path in the entry above.

### GPU sharing -- no pinning anywhere in this repo

hyperplane has 4 GPUs, shared across users. `device: auto`
(`configs/base.yaml`) resolves through `resolve_device()` in
`scripts/run_experiment.py` to plain `"cuda"`, which is PyTorch's default
device -- **always index 0** -- when nothing is pinned.
`utils.device_string()` confirms this: it reads
`torch.cuda.get_device_name(0)` unconditionally. Nothing in this repo sets
`CUDA_VISIBLE_DEVICES`.

**Consequence:** two sessions launched without pinning a device will
silently land on GPU 0 together and collide -- no error, just contention and
possibly an OOM that looks like a code bug.

**Rule going forward:** always check `nvidia-smi` for a free GPU and export
`CUDA_VISIBLE_DEVICES=<n>` before launching any real (non-smoke) run. Added
to README's Setup section so a future session doesn't have to rediscover
this.

---

## 2026-09-11 — CIFAR-10 pilot run: uncentred NC3 is not comparable across heads at baseline

**What happened:** first real GPU pilot, `configs/cifar_ce.yaml` and
`configs/cifar_arcface.yaml`, 30 epochs each, seed 0, unlearning disabled
(baseline geometry only). Both trained cleanly (loss descended monotonically
for both; ArcFace did not diverge at s=30/m=0.5/5 warmup epochs, no config
change needed). Test accuracy: CE 93.36%, ArcFace 91.79% -- comparable,
fairness gate holds.

`nc3_uncentred_mean` came back **0.7211 for CE, -0.8789 for ArcFace** -- on
the *baseline* model, before any unlearning, before any forget class exists.
Surprising enough to investigate before trusting it: the AISTATS mechanism
predicts uncentred cosine goes negative only *after* unlearning targets a
class, and `nc3_centred_mean` for the same two runs is 0.9612 vs 0.9619 --
essentially identical -- so something about the *uncentred* convention
specifically is behaving differently across heads, not the underlying
geometry.

**Diagnosis (`diag_nc3.py`, scratch, not checked in):** loaded the ArcFace
checkpoint and checked `cos(mean(W), mean(features))` directly:
**-0.9999**. The classifier weight rows and the raw backbone features share
a huge, nearly-antiparallel *shared* component -- `||mean(W)|| = 2.29`
against an average per-class `||W_k|| = 2.37`, i.e. the shared component is
almost as large as the class-specific one. Same on the feature side:
`||mean(features)|| = 2.92` against average `||mu_k|| = 3.01`.

**Why this happens:** backbone features are post-ReLU (non-negative), so
every class's raw mean feature shares a large common "DC" direction just
from that. Under CE the loss is an unnormalised dot product, so training
directly suppresses/controls that shared component. Under ArcFace both
features and weights are L2-normalised before the loss -- by design, see
`heads.py`'s docstring, this asymmetry is not a bug to fix -- so the loss
carries zero gradient signal about the shared directional component of `W`;
it drifts under weight decay + init with nothing pulling it toward
alignment with the shared feature direction, and the two ended up almost
exactly antiparallel.

This antiparallel DC term subtracts nearly the same large negative quantity
from every class's raw cosine, which is why all ten per-class ArcFace
cosines cluster tightly around -0.88 regardless of class. It does not hurt
classification: argmax only needs relative ranking, and the class-specific
angular gap survives the shared offset (mean own-class cosine -0.878 vs.
mean best-other-class cosine -0.939 -> 97.0% train accuracy under the
no-margin cosine, matching the 91.79% held-out test accuracy).

**Consequence for the project:** `nc3_uncentred` absolute values are **not
head-comparable** the way `nc3_centred` is. -0.88 at an unlearning-free
ArcFace baseline is not evidence of anything related to forgetting -- it is
present before there is a forget class to misalign. Once unlearning runs
exist, read ArcFace's uncentred forget-class number as a **delta from this
baseline**, not against the theory's raw -1 prediction, or the comparison
silently inherits a head-specific offset that has nothing to do with
unlearning. `nc3_centred` (which strips the shared component by
construction) is the safer head-to-head comparison and already shows CE and
ArcFace starting from essentially the same place (0.9612 vs 0.9619) -- a
good baseline-matched starting point for the unlearning experiments.

**Not a code bug** -- `nc3_alignment` computes exactly what its docstring
says it computes; verified per-sample cosine averaging gives the same
answer as cosine-of-the-mean (-0.879 vs -0.880), and normalised-feature
class means give the same result too (-0.880), which rules out a
magnitude-weighting artifact in the mean.

**Action:** none taken to metrics.py -- this is a fact about the two loss
functions' baseline geometry, not a defect to fix. Recorded here so week 3+
unlearning results are read against the right reference point per head.

---

## 2026-09-13 — CIFAR-10 forget_class=0 table: finetune gap is real, retrain gap is stable across classes

Three checks on the CIFAR-10 pilot's forget_class=0 results table before
trusting it for anything.

### 1. CE finetune (output_forget 0.85) vs ArcFace finetune (0.00) is not a
convergence artifact

Both use identical `unlearn.epochs=3, lr=0.001, weight_decay=5e-4` from
`base.yaml` -- hyperparameters were already matched, so the discrepancy
couldn't be a config typo. The open question was whether CE is simply slower
to reach the same place ArcFace reaches at epoch 3.

**Diagnostic** (`diag_finetune_trajectory.py`, scratch, not checked in):
loaded each head's checkpoint and ran retain-only finetune epoch-by-epoch at
the same lr/wd, evaluating `output_forget`/`output_retain`/`probe_forget`
after every epoch, out to 10x the configured budget.

| epoch | CE output_forget | ArcFace output_forget |
|---|---|---|
| 1 | 0.903 | **0.000** |
| 3 (configured budget) | 0.853 | 0.000 |
| 10 | 0.756 | -- |
| 20 | 0.729 | -- |
| 30 | **0.662** | -- |

ArcFace collapses to 0 within one epoch and stays there. CE decays slowly
and monotonically, and the rate is *flattening*, not accelerating -- at 10x
the epoch budget it has closed less than a third of the gap to zero.
`output_retain` stayed at 0.937-0.939 and `probe_forget` at 0.93-0.94
throughout the CE trajectory, so this isn't the model breaking; retain
utility is untouched while the forget class is barely being dented.

**Conclusion:** not a hyperparameter/convergence issue. Retain-only
finetuning erodes forget-class output accuracy at a genuinely different
rate under the two heads at matched settings. Whether that's the mechanism
under test (margin blocking the drift shortcut, forcing forgetting through
some other slower path) or something else is a question for the geometry
numbers, not a reason to throw out the comparison. Do not "fix" this by
giving CE more finetune epochs in `base.yaml` -- that would change the
comparison's meaning (a fixed epoch budget across heads), not correct a bug.

### 2. probe_gap_to_retrain added to `--compare`

`probe_forget` is not comparable across heads (see 2026-09-11 entry above:
retrain references differ, 0.8630 CE vs 0.5540 ArcFace). `scripts/run_experiment.py
compare()` now looks up each row's own head+forget_class retrain reference
and prints `probe_gap_to_retrain* = probe_forget - retrain_probe_forget`,
labelled with a footnote that it is our arithmetic, not an AISTATS metric.
Also added a `forget_class` column, now that more than one forget class's
`retrain` rows can coexist in the same log directory (see next section).

### 3. Retrain-reference gap (CE vs ArcFace) checked across four forget
classes -- persists

The 0.8630/0.5540 CE/ArcFace retrain-reference gap was measured at
forget_class=0 only, and every `probe_gap_to_retrain` number in the table
inherits it. Ran `scripts/retrain_stability.py` for forget_class in {1,2,3}
per head (fresh backbone+head trained on retain-only, same as the existing
retrain reference, logged via RunDir to
`logs/cifar10_<head>_seed0_retrain_fc<N>/`) to check it isn't a one-class
fluke.

| forget_class | CE retrain probe_forget | ArcFace retrain probe_forget | gap (CE - ArcFace) |
|---|---|---|---|
| 0 | 0.8630 | 0.5540 | 0.309 |
| 1 | 0.8920 | 0.5960 | 0.296 |
| 2 | 0.7490 | 0.5000 | 0.249 |
| 3 | 0.7620 | 0.5300 | 0.232 |

Both heads' probe_forget move around by class (CE: 0.749-0.892, ArcFace:
0.500-0.596) -- expected, some CIFAR-10 classes are just easier to recover
by transfer than others. The CE-ArcFace gap itself is stable, 0.23-0.31
across all four classes checked, never closing. **The retrain-reference
difference is not a forget_class=0 fluke** -- every `probe_gap_to_retrain`
computed against it for other classes should be trusted at the same level
as the fc0 numbers already in the table.

`scripts/retrain_stability.py` skips the 30-epoch "original" training and
the other unlearning methods (retrain doesn't depend on either), but still
writes through `RunDir` so these four extra runs per head carry the same
config/seed/commit/device provenance as everything else in `logs/`.

**Also done:** deleted the duplicate `original` rows from
`logs/cifar10_ce_seed0/results.jsonl` and
`logs/cifar10_arcface_seed0/results.jsonl` that predated `nc1_angular` --
`--compare` was picking up two `original` rows per head, one missing the
angular NC1 field.

---

## 2026-09-13 — Retrain reference gap (first substantive result); finetune trajectories not comparable

**RETRAIN REFERENCE GAP (first substantive result)**

Probe_forget on retrain-from-scratch models, 4 forget classes:

    CE:      0.863, 0.892, 0.749, 0.762
    ArcFace: 0.554, 0.596, 0.500, 0.530
    Gap 0.23-0.31, CE higher in every class, ranges do not overlap.

**Interpretation:** a model retrained without class k still recovers it
under a probe via transfer from remaining classes. ArcFace transfers
substantially less. No unlearning method involved, so this is a property of
the training objective, not of any forgetting procedure.

**Consequence:** probe_forget is NOT comparable across heads. Each head
requires its own retrain reference. Use probe_gap_to_retrain.

**Caveat:** 4 classes, 1 seed. Not yet enough for a confidence interval.

**FINETUNE TRAJECTORY**

ArcFace reaches output_forget=0.0 at epoch 3; CE still at 0.662 after 30.
Retain-only finetuning removes the class ~10x faster under ArcFace. The
original comparison table's finetune rows were therefore not comparable --
the methods were at different points on their trajectories.

---

## 2026-09-13 — Full per-epoch trajectories run for both heads; matched-output_forget comparison

Ran with the new per-epoch trajectory instrumentation:

    python scripts/run_experiment.py --config configs/cifar_ce.yaml --set unlearn.enabled=true
    python scripts/run_experiment.py --config configs/cifar_arcface.yaml --set unlearn.enabled=true

then, because CE `finetune` never reached output_forget=0 at the configured
3 epochs, a CE-only extension:

    python scripts/run_experiment.py --config configs/cifar_ce.yaml --set \
      unlearn.enabled=true unlearn.epochs=30 unlearn.trajectory_every=5 \
      unlearn.methods=[finetune] unlearn.retrain_reference=false

(`neggrad_plus`/`random_label` already converged within 1-2 epochs and
didn't need the extension; `retrain_reference` skipped since that reference
was already on record. ArcFace was not touched.)

**Housekeeping:** `logs/cifar10_ce_seed0/` and `logs/cifar10_arcface_seed0/`
are fixed-name directories reused across runs -- `RunDir` appends rather
than overwrites `results.jsonl`, so each of these three invocations left
duplicate `original`/method rows behind. Deduplicated after each run,
keeping the latest occurrence per (method, forget_class). One consequence
worth flagging: `results.jsonl`'s `finetune`/`finetune_clfonly` rows for CE
now reflect the 30-epoch extension, not the original 3-epoch config --
`trajectory.jsonl` has the full epoch-by-epoch record either way, and
`--compare` only ever reads the end-of-run row, so this doesn't silently
mismatch epoch counts between heads there, but it does mean CE's
`results.jsonl` row is no longer "3 epochs, same as everything else" the
way `base.yaml` implies. Worth a comment in the config if this trips
anyone up later.

### Epoch at which output_forget first hits 0.0, and the geometry there

| head | method | first ep @ 0.0 | nc3_centred_forget | nc1_angular | output_retain |
|---|---|---|---|---|---|
| CE | finetune | never (0.659 @ ep30) | -- | -- | -- |
| CE | finetune_clfonly | never (0.894 @ ep30) | -- | -- | -- |
| CE | neggrad_plus | 1 | +0.672 | 21.03 | 0.111 (chance) |
| CE | neggrad_plus_clfonly | 2 | -0.722 | 0.223 | 0.343 |
| CE | random_label | 2 | +0.395 | 0.497 | 0.921 |
| CE | random_label_clfonly | 1 | +0.409 | 0.223 | 0.936 |
| ArcFace | finetune | 1 | +0.846 | 0.051 | 0.937 |
| ArcFace | finetune_clfonly | 1 | +0.917 | 0.064 | 0.924 |
| ArcFace | neggrad_plus | 1 | +0.443 | 36.37 | 0.111 (chance) |
| ArcFace | neggrad_plus_clfonly | 1 | -0.868 | 0.064 | 0.111 (chance; recovers to 0.816 by ep3) |
| ArcFace | random_label | 1 | +0.850 | 0.142 | 0.928 |
| ArcFace | random_label_clfonly | 1 | -0.821 | 0.064 | 0.924 |

CE `finetune` extended to 30 epochs, logged every 5: 0.939 (ep0) -> 0.826
(ep5) -> 0.761 (ep10) -> 0.732 (ep15) -> 0.726 (ep20) -> 0.680 (ep25) ->
0.659 (ep30). The decay is flattening, not accelerating -- 10x the epoch
budget closed less than a third of the gap to zero. `finetune_clfonly`
bottoms out even higher, at 0.894.

### Matched comparison -- geometry at equal output-level forgetting

**Not matchable: `finetune` / `finetune_clfonly`.** CE never reaches
output_forget=0.0 within 30 epochs under either variant, so there is no
epoch to line up against ArcFace's epoch-1 collapse. Whether CE asymptotes
above zero or would eventually reach it given far more epochs is untested
-- the trajectory only shows it flattening, not approaching zero.

**Matchable but confounded: `neggrad_plus` (full model).** Both heads hit
output_forget=0.0 at epoch 1, but `output_retain` has collapsed to chance
(~0.11) for *both*. The geometry at this point (nc1_angular exploded to
21-36, meaning within-class scatter, not classifier drift, dominates)
reflects the whole model having broken, not a clean forgetting signal.
Not usable evidence for or against the mechanism.

**Matchable, clean: `random_label` (full model).** CE ep2 vs ArcFace ep1,
both retain healthy (0.921 / 0.928). Neither shows the AISTATS drift --
both stay positively aligned with the class mean (CE +0.395, ArcFace
+0.850). Consistent with random_label leaving geometry essentially
untouched under either head.

**Matchable, clean, and the sharpest result: `random_label_clfonly`.** CE
ep1 vs ArcFace ep1, both retain healthy (0.936 / 0.924), backbone frozen in
both so class-mean features cannot move -- any nc3 change is attributable
to the classifier alone. **CE stays aligned (+0.409); ArcFace flips
negative (-0.821).** At a matched, healthy, output-level outcome, ArcFace's
classifier demonstrably drifts away from the class mean while CE's does
not. This is the cleanest evidence so far for a real head-level difference
in mechanism, not just in rate.

**Matchable, mixed: `neggrad_plus_clfonly`.** CE ep2 vs ArcFace ep1, both
flip negative (-0.722 / -0.868 -- the drift mechanism shows up in both
heads here), but retain utility differs sharply at the matched point (CE
0.343 vs ArcFace 0.111, chance). ArcFace's frozen-backbone classifier
transiently collapses retain accuracy to chance at epoch 1 then recovers to
0.816 by epoch 3 -- a non-monotonic dynamic the per-epoch trajectory
surfaces that an end-of-run-only view would have missed entirely.

**Headline:** the one case that isolates classifier movement from backbone
movement and holds output-level outcome AND retain utility fixed across
heads (`random_label_clfonly`) shows ArcFace's classifier flipping away
from the class mean while CE's does not. That is the AISTATS mechanism
appearing under ArcFace and not under CE, in the cleanest condition
available so far. Everything else is either not yet matchable
(`finetune`) or confounded by model collapse (`neggrad_plus` variants).
One clean matched pair is not a paper-level result on its own -- next
step is checking whether this holds at other forget classes and, ideally,
finding a way to get CE `finetune` to actually reach output_forget=0 so
that pair becomes comparable too.

---

## 2026-09-14 — RunDir silently mixed configs into one results file; now refuses

**What happened:** run directory names (`utils.RunDir.create`, called from
`scripts/run_experiment.py` and `scripts/retrain_stability.py`) are derived
from data/head/seed only, not the full config. The 2026-09-13 CE extension
(`unlearn.epochs=30`) reused the same directory name --
`logs/cifar10_ce_seed0/` -- as the original 3-epoch run. `RunDir.create`
happily let it: `config.json`/`env.json` got overwritten to describe the
30-epoch run, while `results.jsonl` got a new `finetune`/`finetune_clfonly`
row *appended* under the same method names the 3-epoch run used. Dedup
logic (keep latest occurrence per method) then discarded the original
3-epoch rows entirely -- their `ncc`/`verif_auc`/`nc2`/raw `nc1`/uncentred
`nc3` values are not recoverable from anywhere on disk. The forget-specific
subset (`output_forget`, `output_retain`, `probe_forget`,
`nc3_centred_forget`, `nc1_angular`) survives at epochs 0-3 in
`trajectory.jsonl`, because that file is genuinely append-only across a
run's own epochs -- but the richer end-of-run row is gone.

**Fix:** `RunDir.create` now raises `FileExistsError` if the target
directory exists and is non-empty, instead of silently writing into it. A
fresh or empty directory still works exactly as before. This makes the
2026-09-13 situation impossible to repeat silently -- the CE extension run
would now refuse to start until the existing directory is moved aside, or
given a distinct name (as `retrain_stability.py` already does with its
`_retrain_fc<N>` suffix).

**Test:** `src/test_utils.py`, added to `make test`. Covers: fresh
directory succeeds, empty existing directory succeeds, non-empty directory
raises, a single unrelated stray file still counts as non-empty, and
refusal happens before any write (existing `config.json`/`results.jsonl`
content is provably untouched by the refused call).

**Data cleanup:** relabelled the two affected rows in
`logs/cifar10_ce_seed0/results.jsonl` from `finetune`/`finetune_clfonly` to
`finetune_ep30`/`finetune_clfonly_ep30`, so they can no longer be mistaken
for the base 3-epoch config `--compare` and everything else assumes. CE's
`results.jsonl` now has **no** row for the true 3-epoch `finetune`/
`finetune_clfonly` end-of-run metrics -- only the trajectory subset noted
above. Regenerating the missing full-metric row requires an actual rerun
(now safe to do without risk of silently re-mixing, since `RunDir` will
refuse if `cifar10_ce_seed0/` isn't cleared first). Not done here --
no rerun was requested, and this is local, gitignored data rather than
something the fix itself needed to touch.

---

## 2026-09-14 — finetune and random_label_clfonly patterns hold at fc0-3, no exceptions

The two conditions that showed a real head difference at forget_class=0 --
`finetune` (full backbone) and `random_label_clfonly` (frozen backbone) --
were rerun at forget_class in {1,2,3}, both heads, starting from the
existing checkpoint rather than retraining the original model per class
(`scripts/unlearn_across_classes.py`, new). Both patterns hold at every
class checked. fc0 was not unrepresentative.

**`finetune`:** CE never reaches `output_forget=0` at any class, even at 30
epochs -- final values 0.760 (fc1), 0.633 (fc2), 0.409 (fc3), retain
healthy throughout (0.93-0.96). ArcFace reaches 0 at or before epoch 5 at
every class (0.000 at fc1, fc2, fc3). `trajectory_every=5` means the exact
epoch isn't resolved for fc1-3 -- fc0's finer per-epoch logging showed
epoch 1, so it's plausible the same holds here, but this run doesn't prove
it.

**`random_label_clfonly`:** all four classes (0-3) start within
+0.96 to +1.0 of the neural-collapse ceiling -- a common, matched starting
point regardless of class or head. By epoch 1, CE stays positive at every
class (+0.41 to +0.47); ArcFace flips negative at every class (-0.79 to
-0.96). Retain accuracy stays healthy throughout (0.91-0.96), both heads,
all classes -- not a collapse artefact, same clean condition as fc0.

**Open thread, not yet investigated:** CE `finetune`'s final
`output_forget` varies considerably by class -- 0.409 to 0.760, a 0.35
spread across just 4 classes. Some classes are evidently more resistant to
retain-only finetuning than others. Doesn't change the categorical result
(CE still never reaches 0 at any class checked), but worth understanding
later -- possibly related to class separability or how many other classes
are visually/semantically close to the forgotten one.

**`scripts/unlearn_across_classes.py`** promoted from a scratch script to a
proper one, following the conventions `scripts/retrain_stability.py`
already established: loads the config with `load_config`/`apply_overrides`
(so `--set` works), writes every result through `RunDir` (one fresh
directory per method+forget_class, e.g.
`logs/cifar10_ce_seed0_finetune_fc2/`), and starts from the existing
checkpoint instead of repeating the forget-class-independent original
training. `--finetune-epochs`/`--finetune-trajectory-every` default to the
30/5 used here; `--ckpt` defaults to whatever `scripts/run_experiment.py`
already trained for the given config.

---

## 2026-09-14 — random_label_clfonly confirmed at a second seed, all 4 classes

Reran `random_label_clfonly` at forget_class 0-3, both heads, at seed=1 --
a genuinely different backbone init and data order, not just a different
forget class on the seed=0 checkpoint. Required training fresh CE and
ArcFace baselines at seed=1 first (`scripts/run_experiment.py --set seed=1
unlearn.enabled=false`, ~4 min each): the seed=0 checkpoints can't be
reused for this, since the seed affects backbone init and data order and
the question is specifically whether the pattern survives a different
trained model, not just a different forget class on the same one.

**Holds at every class, both seeds:** CE stays positive at epoch 1 in
every seed=1 case (+0.4379 to +0.4842). ArcFace flips negative in every
seed=1 case (-0.6782 to -0.9444). `output_retain` stays healthy throughout
(seed=1 range 0.9166-0.9429), both heads, all classes -- no collapse
confound.

**Combined count, both seeds:** 2 heads x 4 classes x 2 seeds = 16
(head, class, seed) points checked. **16 of 16 agree on the expected
direction** (CE positive, ArcFace negative) -- verified by reading every
trajectory.jsonl directly rather than trusting a running tally (an earlier
verbal summary in-session said "12 of 12", which was an arithmetic slip,
not a data problem -- corrected here).

**Flagged and investigated in the same session: ArcFace fc2/seed1 is
weaker than the other 7 ArcFace points.** Final nc3_centred_forget -0.7209
vs -0.8418 to -0.9753 elsewhere; it also started lower at epoch 0 (+0.8956
vs +0.9629 to +0.9993 elsewhere). Checked for a mundane explanation before
treating it as unexplained variance:

- Training/test accuracy for the seed=1 ArcFace checkpoint matches seed=0
  closely (91.71% vs 91.79% test, final-epoch loss 5.5985 vs 5.5881) --
  this is not a botched or under-converged run in the ordinary sense.
- Computed nc3_centred_forget for all 10 classes (not just 0-3) on both
  checkpoints. seed=0: tight, mean +0.962, std 0.043, min +0.848 (class 8).
  seed=1: noticeably wider, mean +0.936, std **0.090** (~2x), min **+0.688
  (class 8)** -- with classes 2, 7, and 8 all sitting below the other seven.
  Class 2 is not a unique outlier; it's one of three below-average classes
  in a checkpoint whose per-class alignment is generally less uniform than
  seed=0's. Consistent with this checkpoint's higher nc2 (ETF deviation:
  0.081 vs 0.051) and lower nc3_centred_mean (0.936 vs 0.962) already on
  record.

**Conclusion:** mundane and explained -- ordinary seed-to-seed variation in
how uniformly NC collapse lands, not a defect in this run or in the
unlearning method. Class 8 is the real outlier in the seed=1 checkpoint (a
bigger gap than class 2's), not yet checked under `random_label_clfonly` --
worth a look if seed=1 is extended further.

---

## 2026-09-14 — LR schedule truncation bug fixed; CIFAR fc0 findings re-verified, hold

**Discovery.** Investigating faces ArcFace's baseline test accuracy (61.79%
vs CE's 74.20%, a 12.4pp gap -- CIFAR's pilot only had 1.6pp). First
attempted the CIFAR fix (`s: 64 -> 30`, matching `s` tuned down for CIFAR's
10-class scale): accuracy collapsed to 25.02%, the wrong direction --
standard ArcFace guidance scales `s` *up* with class count, not down, and
faces has 1000 classes vs CIFAR's 10. Reversed course rather than chasing
`s=16` next.

Checked `train_model`'s LR schedule instead. `CosineAnnealingLR` is built
with `T_max=epochs`, but `sched.step()` only fires on post-warmup epochs --
so with `warmup_epochs>0` the schedule receives only `epochs -
warmup_epochs` steps against a cycle calibrated for the full `epochs`,
and never fully decays. Confirmed directly from real run logs: CIFAR
ArcFace (epochs=30, warmup=5) ended at lr=0.00670, not 0.00000; faces
ArcFace (epochs=40, warmup=5) ended at lr=0.00381. **Every `warmup_epochs >
0` run in the project trained with a truncated schedule** -- every ArcFace
run so far, since CE always uses `warmup_epochs=0` and was never affected.

**Fix:** `src/train.py` now sets `T_max = epochs - warmup_epochs` (the
actual step count the scheduler receives) instead of the raw `epochs`.
Added `src/test_train.py` (4 tests, in `make test`): confirms the schedule
reaches ~0 with warmup, without warmup (regression guard for CE's path),
at the real epochs=40/warmup=5 configuration, and that the warmup ramp
itself is still linear and correct.

**Effect on CIFAR ArcFace baseline accuracy:** 91.79% -> **93.41%**, now
essentially at parity with CE's 93.36% (was a 1.6pp gap, now CE and
ArcFace are within 0.05pp of each other). A real, meaningful improvement
from the fix alone.

**Effect on faces ArcFace baseline accuracy (s=64 unchanged):** 61.79% ->
64.94%. Helped, but nowhere near enough -- still 9.26pp behind CE's
74.20%. The schedule bug was real and worth fixing project-wide, but it
was not the primary explanation for the faces gap specifically (see the
augmentation entry below for what was checked next).

**Re-verification: do the committed CIFAR fc0 findings still hold under
the fixed schedule?** Both re-run from a fresh fixed-schedule ArcFace
checkpoint:

- `random_label_clfonly`, `nc3_centred_forget` @ epoch 1: -0.8214
  (committed) -> **-0.9109** (fixed schedule). @ final (ep3): -0.8674 ->
  -0.9427. Not identical -- a real ~0.09 shift -- but same sign, and the
  separation from CE's positive values (+0.33 to +0.47) *widened*, not
  narrowed. `output_retain` stayed healthy in both (~0.92-0.94).
- `finetune`, full backbone, 30 epochs: CE final `output_forget` 0.659
  (committed) -> 0.654 (re-run) -- matches, as expected, since CE's
  checkpoint was never touched by the bug. ArcFace: reached 0 by epoch 1
  (committed, fine-grained logging) -> reaches 0 by epoch 5 at the latest
  (re-run, coarser `trajectory_every=5`) -- consistent with, not
  contradicting, the original finding.

**Conclusion: the fc0 findings (both `finetune` and `random_label_clfonly`,
both heads) are re-verified against the fixed schedule and hold.** No
provisional flag needed on them going forward. The fuller sweep (fc1-3,
both seeds, all conditions) technically still reflects pre-fix ArcFace
checkpoints and hasn't been individually re-run, but given fc0 -- the
original, most-scrutinized case -- came back the same or stronger under
the fix, there's no live reason to doubt the rest of the matched-comparison
table pending a full re-run.

---

## 2026-09-14 — Faces augmentation experiment: closes ArcFace's overfitting gap, doesn't move the head-to-head gap

Faces ArcFace's no-margin train accuracy at epoch 40 was 99.35% against
64.94% test (fixed schedule, s=64) -- a 34.41pp train/test gap, much wider
than CE's own 100.00%/74.20% (25.80pp). Checked two candidate explanations
independent of the margin loss itself, per the user's request, before
touching `s` again.

**`weight_decay`: ruled out.** Verified via the actual config loader (not
just reading YAML) that all four configs -- `cifar_ce`, `cifar_arcface`,
`faces_ce`, `faces_arcface` -- resolve to `weight_decay: 0.0005`. `faces_ce.yaml`
doesn't override it; it inherits `base.yaml`'s default, identical to
CIFAR's. No CIFAR-vs-faces discrepancy exists in this setting.

**Augmentation: a real gap found and fixed.** `src/data.py`'s
`cifar_transforms` applies `RandomCrop(32, padding=4)` + flip for training;
`face_transforms` applied flip *only* -- no translation jitter at all,
so the model saw the exact same pixel-aligned crop of every face every
epoch. Added `RandomCrop(size, padding=8)` before the flip, the same idiom
as CIFAR's own augmentation scaled to the face image size -- pure
translation jitter, no scale or aspect-ratio distortion (unlike
`RandomResizedCrop`), consistent with the existing docstring's caution
against aggressive augmentation changing identity-relevant appearance.

Retrained both faces heads (s=64, m=0.5, fixed schedule) with the new
augmentation:

| | train acc | test acc | gap |
|---|---|---|---|
| CE, no augmentation | 100.00% | 74.20% | 25.80pp |
| CE, with augmentation | 99.97% | 74.65% | 25.32pp |
| ArcFace, no augmentation | 99.35% | 64.94% | 34.41pp |
| ArcFace, with augmentation | **90.98%** | 64.63% | **26.35pp** |

**Augmentation worked exactly as regularization should -- for ArcFace,
substantially.** Its train accuracy dropped a real 8.37pp (99.35% ->
90.98%), closing the train/test gap by 8.06pp. CE barely moved (it wasn't
overfitting as severely to begin with).

**But test accuracy itself did not move, for either head** (CE +0.45pp,
ArcFace -0.31pp, both noise-level), **and the CE-vs-ArcFace head-to-head
gap is unchanged**: 9.26pp (no augmentation) -> 10.02pp (with
augmentation). ArcFace now generalizes better relative to its own
(less-memorized) training fit, but not better relative to CE.

**Conclusion: `weight_decay` and augmentation are both ruled out as
explanations for the CE-vs-ArcFace head-to-head test-accuracy gap at
faces scale.** Two independent, real, working fixes, neither of which
closed the gap that actually matters for the comparison. `s`/`m` remains
the more likely lever -- an `s=96` attempt (augmentation kept, fixed
schedule) is in progress as this entry is being written; not yet reported
here.

---

## 2026-09-14 — Faces ArcFace tuning resolved: s=64 was the wrong scale for 1000-way classification

`s=64` -- the value the config already called "standard for face
recognition with many identities" -- was itself the problem. Standard
ArcFace guidance scales `s` up with class count; 1000 classes needs more
than the 64 carried over from a smaller setup. (The first tuning attempt
went the opposite direction, `s=64->30` matching CIFAR's own fix for its
10-class scale, and collapsed accuracy to 25.02% -- the wrong direction
entirely, since CIFAR and faces needed opposite corrections. See the
LR-schedule entry above for that detour.)

Raised `s: 64 -> 96` (m=0.5 unchanged, augmentation kept, fixed schedule):

| | test acc | train acc (no-margin) | gap |
|---|---|---|---|
| s=64 | 64.63% | 90.98% | 26.35pp |
| **s=96** | **72.71%** | 96.40% | 23.69pp |

**Head-to-head gap: 9.26pp -> 1.94pp** (CE 74.65% vs ArcFace 72.71%) --
tighter than the CIFAR pilot's own 1.6pp gap. Target met (roughly 2-3pp
was the bar).

**Diagnostic signature worth keeping in mind for future tuning:** test
accuracy grew *faster* than train accuracy this time (+8.08pp vs
+5.42pp). That's the opposite pattern from the augmentation experiment
(previous entry), where train accuracy fell 8.37pp and test accuracy
didn't move at all -- reduced memorization with no generalisation payoff.
Here, both rose and test rose more, which is the signature of a genuine
capacity/gradient-informativeness fix (the softmax was under-scaled for a
1000-way problem) rather than a regularization effect. The two
interventions looked superficially similar (both "improve the ArcFace
face model") but were mechanistically different, and only one of them
actually closed the gap that mattered.

`configs/faces_arcface.yaml` updated to `s: 96.0` so the checked-in config
matches what produced this result. Not yet re-run: the retrain-reference,
CIFAR-style forget-class trajectory, or any unlearning condition on faces
-- this entry only resolves baseline accuracy matching, the CLAUDE.md gate
that has to clear before any geometry comparison is meaningful.

---

## 2026-09-14 — The CIFAR result does NOT replicate on faces: ArcFace does not flip

First unlearning condition run on faces, now that baselines are matched
(CE 74.65%, ArcFace 72.71%, 1.94pp apart -- see the entry above).
`random_label_clfonly` -- the cleanest CIFAR finding, backbone frozen so
any nc3 movement is the classifier alone -- at 4 forget identities (0,
122, 389, 794), both heads, seed 0. All four identities sit at the
50-image cap (train 40 / test 10 each), so sample size is comparable
across them; 739 of the 1000 identities are at that cap, so these are
typical, not cherry-picked.

| head | identity | ep0 | ep1 | final (ep3) | output_retain (final) |
|---|---|---|---|---|---|
| CE | 0 | +0.7887 | +0.2346 | +0.2345 | 0.7232 |
| CE | 122 | +0.8212 | +0.2084 | +0.2084 | 0.7202 |
| CE | 389 | +0.7865 | +0.1857 | +0.1857 | 0.7276 |
| CE | 794 | +0.8508 | +0.3047 | +0.3045 | 0.7238 |
| ArcFace | 0 | +0.8875 | +0.9793 | +0.9484 | 0.7237 |
| ArcFace | 122 | +0.9502 | +0.9752 | +0.9373 | 0.7232 |
| ArcFace | 389 | +0.8539 | +0.4499 | +0.2922 | 0.7221 |
| ArcFace | 794 | +0.9403 | +0.6283 | +0.4680 | 0.7241 |

**The CIFAR pattern does not replicate.** On CIFAR, ArcFace flipped
negative at epoch 1 in 16/16 (head, class, seed) checks. On faces,
**ArcFace stays positive at all 4 identities -- 0/4 flip.** CE also stays
positive (+0.19 to +0.30 at epoch 1, tight), which is the same direction
CE showed on CIFAR; it is specifically ArcFace's behaviour that differs
between the two datasets. ArcFace's values span +0.29 to +0.98 across the
four identities, a much wider spread than CE's.

**`output_retain` is healthy throughout** -- 0.720 to 0.728 across both
heads and all four identities, each tracking its own head's baseline. No
collapse confound; this is not a case of "forgot everything because the
model broke."

**Measurement soundness checked before concluding anything.** The obvious
worry was that faces' forget-class mean is estimated from ~40 training
images against CIFAR's ~4,500, and that nc3 is simply too noisy at that
size to show a flip. Tested directly by recomputation on the CIFAR ArcFace
fc0 epoch-1 weight matrix (reproduced deterministically and verified
against the committed -0.9109, matched at -0.9108): estimate the
forget-class mean from 40 randomly sampled training images instead of all
~4,500, 20 trials, everything else held fixed.

    mean -0.9008   std 0.0575   min -0.9693   max -0.7288
    20/20 trials below -0.5      0/20 sign flips

Tight and consistently negative. **At n=40, CIFAR's flip is robust, so the
~40-sample class-mean estimate is not what prevents a flip on faces.** The
divergence is real, not an artefact of a starved estimator.

**Separate limitation, still standing:** `output_forget` and
`probe_forget` on faces have 0.1 granularity -- identities have 8-10 test
images each (min 8, max 10, mean 9.70, median 10), against CIFAR's ~1,000
test images per class. The constant 0.7000 seen in these runs is literally
7/10. Those two metrics are resolution-limited on faces in a way that
`nc3_centred_forget` (which reads the ~40 training images, and was just
shown to be stable at that size) is not. Worth fixing or at least stating
loudly before any claim rests on faces `probe_forget` numbers; raising
`test_fraction` or the per-identity image cap would both help.

**Open, unexplained:** ArcFace's between-identity variance is much larger
than CE's (+0.29 to +0.98 vs +0.19 to +0.30). Two identities (0, 122) sit
near +0.95, two (389, 794) are considerably weaker. Flagged rather than
averaged away, same as the ArcFace fc2/seed1 outlier on CIFAR. Not yet
diagnosed.

**Not yet done, deliberately:** no mechanism diagnosis, no seed
replication, no CosFace, no `finetune` condition on faces, no
generalisation/held-out-appearance experiment. The next question is *why*
ArcFace behaves differently here -- 1000-way vs 10-way geometry, the s=96
vs s=30 scale, or something about the face feature space -- and that
should start from a clean look at this table rather than from more runs
stacked on top of it.

---

## 2026-09-14 — 100-identity face subset: a partial flip, but it does not isolate class count

**Recorded after the fact.** These ten runs were executed before this entry
was written; everything below is read back out of `logs100/` rather than
from memory of the session that produced them. Where a fact is not in an
artifact it is marked as unrecorded, not reconstructed.

**Why it was run.** The 1000-identity result (entry above) showed the CIFAR
ArcFace nc3 flip does not replicate on faces. Class count was the first
hypothesis: CIFAR has 10 classes, the face set has 1000. A 100-identity
subset sits between them. `configs/faces100_ce.yaml` and
`configs/faces100_arcface.yaml` hold the executed setup; both are untracked
as of this entry and should be committed with it.

### Provenance

| | |
|---|---|
| Dataset | `/data/bijaypandey/archive/casia-webface-folders-100id` |
| Identities / images | 100 / 15,707 |
| Images per identity | min 101, median 152.5, mean 157.07, max 200 |
| Split | `stratified_image_split`, `test_fraction` 0.2 → train 12,568 / test 3,139; 20-40 test images per identity (mean 31.4) |
| Code commit | `61abcbf` on all ten runs (`env.json`) |
| Device | `cuda (NVIDIA A100-SXM4-40GB)` on all ten runs |
| Seed | 0 |
| LR schedule | post-fix -- both baselines end at `lr 0.00000` in `run.log` |
| Head settings | ArcFace s=96.0, m=0.5, 5 warmup epochs -- the 1000-identity values, deliberately not retuned |
| Everything else | inherited unchanged from `faces_ce.yaml` → `base.yaml`: resnet18, `small_input: false`, `feat_dim: 512`, 40 epochs, lr 0.1, wd 5e-4, SGD, cosine, batch 128, same augmentation |

**The run commands are recorded**, in `env.json` `argv`. Baselines:
`scripts/run_experiment.py --config configs/faces100_{ce,arcface}.yaml --set
unlearn.enabled=false`. Unlearning: `scripts/unlearn_across_classes.py
--config configs/faces100_{ce,arcface}.yaml --forget-classes 0,29,60,95
--conditions random_label_clfonly`. Loading each config through
`utils.load_config` and diffing against the stored `config.json` gives an
exact match on every key except `unlearn.enabled`, which the `--set`
override above accounts for -- so the two config files reproduce the
executed setup.

**Unrecorded: the extraction seed.** The conversion command that produced
this directory was never written down. The observed per-identity range
(101-200) is consistent with `convert_rec_to_folders.py --min-images 100
--max-identities 100 --max-images-per-identity 200`, and the extraction ran
2026-09-14 12:38, but `--seed` is not recorded anywhere -- not in the run
configs (which capture `FaceFolder` arguments, not the converter's), not in
`env.json`, not here until now. Identity selection inside the converter is
`rng.choice`-driven, so **this exact 100-identity set is not reproducible
from the artifacts alone.** A re-extraction at the script default (seed 0)
may or may not reproduce it; that has not been tested. Treat the directory
itself as the primary artifact until this is resolved.

**The two face sets are not nested.** Only **31 of the 100** identities also
appear in the 1000-identity set, and the shared ones carry different image
counts (e.g. `00009`: 50 images there, 160 here). So relative to the
1000-identity experiment this changes class count *and* images per identity
*and* identity membership. The config header's claim that "class count and
images-per-identity are the only things that move" is wrong on the third
count.

### Baselines -- accuracy-matched

| head | test acc | probe_overall | verif_auc | ncc_overall | nc1 | nc1_angular | nc2 | nc3_centred_mean | nc3_uncentred_mean |
|---|---|---|---|---|---|---|---|---|---|
| CE | **0.8468** | 0.8433 | 0.9598 | 0.8394 | 0.5461 | 0.5164 | 0.1398 | +0.8895 | **+0.6416** |
| ArcFace | **0.8407** | 0.8343 | 0.9570 | 0.5221 | 1.3110 | 0.0533 | 0.1381 | +0.8511 | **-0.8778** |

**0.61pp apart** -- the tightest head-to-head match in the project so far
(CIFAR 0.05pp, faces-1000 1.94pp). The fairness gate passes, and it passes
with s=96 carried over untouched, so the 1000-way scale is not mistuned at
100-way. The uncentred-NC3 baseline offset between heads (+0.64 vs -0.88)
reproduces the CIFAR and faces-1000 pattern exactly; it remains a property
of the loss, not of any unlearning.

### `random_label_clfonly`, 4 identities, both heads

Backbone frozen, so every nc3 movement below is the classifier alone.
`nc3_centred_forget`, per epoch, from `trajectory.jsonl`:

| head | label | folder | ep0 | ep1 | ep2 | ep3 (final) |
|---|---|---|---|---|---|---|
| CE | 0 | `00006` | +0.9009 | +0.5395 | +0.5318 | **+0.5222** |
| CE | 29 | `01240` | +0.9092 | +0.4750 | +0.4663 | **+0.4560** |
| CE | 60 | `03938` | +0.8603 | +0.4335 | +0.4280 | **+0.4202** |
| CE | 95 | `07818` | +0.9207 | +0.4547 | +0.4500 | **+0.4446** |
| ArcFace | 0 | `00006` | +0.9615 | +0.6113 | +0.5373 | **+0.4636** |
| ArcFace | 29 | `01240` | +0.6855 | **-0.3718** | -0.4857 | **-0.5778** |
| ArcFace | 60 | `03938` | +0.9970 | +0.2955 | +0.1820 | **+0.0667** |
| ArcFace | 95 | `07818` | +0.9981 | +0.4161 | +0.1083 | **-0.2205** |

**Utility and forget outcome are matched and healthy throughout.**
`output_forget` is exactly 0.0000 at epochs 1, 2 and 3 in all eight runs.
`output_retain` at the final epoch: ArcFace 0.8413 / 0.8420 / 0.8414 /
0.8416, CE 0.8217 / 0.8231 / 0.8250 / 0.8300 -- each within ~0.02 of its own
head's baseline, so nothing here is a collapse artefact. Uncentred
`nc3_uncentred_forget` at the final epoch: ArcFace -0.966 to -0.969, CE
-0.079 to -0.144.

### The result depends on which epoch you read it at

This is the part that must not be lost.

- **At epoch 1** -- the matched-output_forget point at which *every CIFAR
  claim in this project has been stated* -- only label 29 is negative.
  **1 of 4.** Labels 60 and 95 are still clearly positive (+0.2955,
  +0.4161).
- **At epoch 3** (end of the 3-epoch budget) -- 2 negative (29 at -0.5778,
  95 at -0.2205), 1 near zero (60 at +0.0667), 1 positive (0 at +0.4636).
  **2 of 4.**

Both readings are correct; they are different conventions. On CIFAR the
distinction never mattered because the flip was complete by epoch 1 and
stayed there. Here ArcFace drifts monotonically negative across all three
epochs at three of four identities, so the headline number is a function of
where the budget stops. **No comparison across datasets is meaningful until
one convention is fixed and applied to CIFAR, faces-1000 and faces-100
alike.**

Two further qualifications on the epoch-3 reading:

1. ArcFace label 0's "positive" value (+0.4636) is numerically
   indistinguishable from CE's four values (+0.4202 to +0.5222). At that
   identity ArcFace does not behave like CIFAR ArcFace; it behaves like CE.
2. The two identities that go negative (29, 95) happen to be two of the 31
   that also exist in the 1000-identity set, and the two that do not (0, 60)
   are new. With n=4 that is a coincidence worth noting and nothing more --
   it is not offered as a pattern.

CE, by contrast, is flat and boring: four identities, all +0.42 to +0.52 at
epoch 3, spread 0.10. ArcFace's spread is 1.04 (-0.58 to +0.46). The
between-identity variance gap flagged on faces-1000 reproduces here and is
still undiagnosed.

### What this does and does not establish

**Consistent with a class-count effect.** Ordering the three datasets by
class count gives 10 (CIFAR: 4/4 flip hard, at epoch 1), 100 (2/4 at epoch
3, 1/4 at epoch 1), 1000 (0/4). Monotone, in the predicted direction.

**But it does not isolate class count.** Against faces-1000 this experiment
moves at least three things at once: class count (1000→100), images per
identity (48.5→157), and identity membership (69 of 100 identities are new).
Against CIFAR it additionally moves domain and scale s. **This is
suggestive, not a controlled test of the hypothesis, and it should not be
written up as one.** The controlled version is a subsample of the *existing*
1000-identity directory down to 100 identities at the same ~48.5 images
each, holding membership and density fixed -- not a fresh extraction.

**No probe claim is available on any face dataset.** There is still no
retrain reference for faces-1000 or faces-100, for either head:
`probe_gap_to_retrain` is null in every row of all eight runs above, and no
`faces*_retrain*` run directory exists. The standing rule is that absolute
probe accuracy is not evidence -- only the gap against a reference is -- so
the `probe_forget` values recorded here (CE 0.7742-0.9200, ArcFace
0.7600-0.9259) are uninterpretable as they stand and must not enter a table.
Generating the faces retrain references is a precondition for any faces
probe number, on either dataset.

The 0.1-granularity problem from faces-1000 *is* fixed here: 20-40 test
images per identity instead of 8-10. That improves the resolution of
`output_forget` and `probe_forget` -- it does not make them interpretable
without the reference.

**Not done:** no seed replication (and note that a faces seed replication is
currently blocked -- `build_datasets` reads `seed` from the `data` sub-dict,
which no config sets, so `--set seed=1` would not change the faces split),
no CosFace, no `finetune` on faces, no mechanism diagnosis, no retrain
reference.

### Decision needed from Rawat before the next run

Three questions, and they interlock -- answering them separately will not
work:

1. **Is a partial negative result acceptable?** This is no longer
   hypothetical, which is what the standing open decision anticipated. The
   effect is complete on CIFAR, absent at 1000 identities, and partial and
   high-variance at 100. The plausible paper is now "the mechanism is
   moderated by class count / classification geometry", not "margin losses
   close the shortcut". Confirm that is a result worth writing before more
   compute goes into it.
2. **Which epoch convention do we report?** Epoch 1 (matched-output, what
   every CIFAR claim used) gives 1/4 negative. Epoch 3 (end of budget) gives
   2/4. The convention must be fixed once and applied retroactively to all
   three datasets; the choice materially changes the headline.
3. **Scope -- explain or broaden?** Roughly eight weeks remain to the Nov 16
   deadline. Explaining the divergence means a controlled class-count
   experiment plus the faces retrain references plus a mechanism diagnosis.
   Broadening means CosFace, more identities, more seeds. There is not time
   to do both properly.

**Decided pending that conversation:** nothing new is run. The immediate
queue if the answers point at "explain" is (a) faces retrain references,
both heads, both datasets, (b) the post-LR-fix CIFAR ArcFace seed-1 rerun,
(c) a class-count experiment that actually holds membership and density
fixed.

---

## 2026-09-14 — Face dataset seed never reached sampling or splitting; fixed

**Supersedes** the parenthetical in the entry above that says a faces seed
replication is blocked. It is no longer blocked.

**The defect.** `build_datasets` resolved its seed as `cfg.get("seed", 0)`,
where `cfg` is the **`data` sub-dict**. No config has ever set `data.seed` --
the seed lives at the top level of the config -- so the lookup fell through
to the default on every run. Both consumers of that value are faces-only:
`FaceFolder`'s per-identity image subsampling (`max_images_per_identity`,
`rng.choice`) and `stratified_image_split`. **Every face run therefore used
seed 0 for sampling and splitting, and `--set seed=1` would have changed the
weight init and the training shuffle while silently reusing the seed-0
split.** Nothing crashes and nothing logs a warning; the run just reports
the seed it was asked for.

**The fix, and the contract.** `build_datasets(cfg, seed=0, log=None)`: the
experiment seed is an explicit argument and the **sole** source of sampling
and split randomness. All three call sites pass `seed=cfg["seed"]`
(`run_experiment.py`, `unlearn_across_classes.py`, `retrain_stability.py`).
A `seed` key inside the `data` block is now **rejected with an error**
pointing at the top-level seed, not honoured as an override -- two places to
set one seed is how this defect hid in the first place. A separate data/split
seed may be worth having, but that is a design decision to take deliberately;
it is not available today.

**Existing results are unaffected.** Every faces and faces-100 run was seed 0
(`config.json`, all 18 run directories) and the default reproduces seed-0
sampling and splitting exactly. No recorded number changes; no rerun needed.
CIFAR never touched this path -- that branch uses torchvision's fixed split
and never reads `seed`.

**Going forward**, a faces seed replication gets genuinely different
subsampling *and* a different train/test partition. State that when reporting
one: seed 0 and seed 1 will differ in which images were extracted per
identity as well as in initialisation, so the two are not a paired comparison
over one fixed sample.

**Validation** -- five CPU-only groups in `src/test_data.py`, on a synthetic
3-identity directory with 12 images each capped to 8 so the subsampling is a
real random choice: the negative case (old lookup returned 0 for every
experiment seed); equal seeds give identical selected samples and identical
partitions; a different seed changes the selection or the partition; the
default still equals seed 0; a stray `data.seed` raises for faces and CIFAR
alike, before any dataset is touched; and at seeds 0/1/2/7 the split stays
disjoint, identity-complete, and label-consistent through one shared scan.
`src/test_data.py` 12 -> 17 groups; `make test` 30 -> 35.

---

## Open decisions

- [x] Dataset — **CASIA-WebFace**, resolved 2026-09-10. Kaggle RecordIO
      packaging, no Howard access needed.
- [ ] Is a negative result acceptable? (If ArcFace shows the same
      misalignment as CE, the paper becomes "the mechanism is general".)
      Get this agreed with Rawat now, not in week eight.
- [ ] Which epoch convention do we report `nc3_*_forget` at — epoch 1
      (matched output_forget, what every CIFAR claim used) or end of the
      unlearning budget? Raised 2026-09-14 by the 100-identity run, where
      the two give 1/4 and 2/4 negative. Must be fixed once and applied to
      CIFAR, faces-1000 and faces-100 alike.
- [ ] Scope — one dataset done properly, or breadth?

---

## Template

## YYYY-MM-DD — <what was decided>
**Decided:** …
**Because:** …
**Supersedes:** …
