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

> **Cross-reference (2026-09-15).** This remains the correct statement of
> what the *theory* predicts. It is not a statement about empirical
> comparability: 2026-09-11 found ArcFace's uncentred baseline already sits
> near −0.88 before any unlearning, so uncentred values are not comparable
> across heads and a negative one is not by itself a sign reversal. Both
> statements stand; see 2026-09-15 — NC3 convention clarification.

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
   stated. *(Logged 2026-09-09; see 2026-09-11 for why uncentred values are
   still not comparable across heads, and 2026-09-15 for the flip
   definition.)*
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

> **Superseded as the authoritative result (2026-09-15).** The scratch
> numbers immediately above were produced ad hoc and their sampling seed was
> never recorded, so their individual trials cannot be reproduced. They are
> kept here as historical evidence of what was checked and when. The
> authoritative result is now the committed, reproducible artifact described
> in "2026-09-15 — Reproducible 40-image NC3 class-mean robustness
> diagnostic" below, which reaches the same qualitative conclusion from a
> verified exact replay of the canonical model.
>
> **Cross-reference (2026-09-15).** The check above tested the 40-image
> estimator on the *CIFAR* model and inferred from it that a starved
> estimator is not what prevents a flip on faces. The face models have since
> been tested directly -- see "2026-09-15 — Face NC3 non-flips survive
> class-mean subsampling" below, where fc0 on this 1000-identity model holds
> at +0.94 to +0.98 across 20 half-pool draws. The inference above is now
> backed by evidence from the face models themselves rather than by
> extrapolation from CIFAR.

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
-0.079 to -0.144. **ArcFace's uncentred baseline on this model is already
`nc3_uncentred_mean` -0.8778**, so -0.966 to -0.969 is **not an uncentred
sign reversal** — it is a within-head move from an already-negative start,
and it does not contradict the centred non-flips tabulated above (CE
baseline for comparison: +0.6416, within-head, not head-to-head).

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

> **Superseded on 2026-09-15** by "Complete face probe-gap matrix: CIFAR
> separation does not transfer cleanly". The precondition stated here has
> since been met: retrain references now exist for all four identities on
> both face subsets and both heads, so `probe_gap_to_retrain*` is computable
> and the `probe_forget` values above are no longer the only face probe
> numbers on record. The eight faces-100 `probe_forget` values quoted here
> are unchanged and re-used verbatim in that entry. The reasoning above --
> that absolute probe accuracy is not evidence without a reference -- still
> stands and is why the later entry reports gaps rather than levels.

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

## 2026-09-14 — CIFAR ArcFace seed-1 re-run post-LR-fix; fc2 anomaly was the bug, not seed variance

**Supersedes:** the ArcFace half of "2026-09-14 — random_label_clfonly
confirmed at a second seed, all 4 classes", and its conclusion that seed
1's weak fc2 point was "ordinary seed-to-seed variation". Those ArcFace
seed-1 numbers were produced at commit `ec70213`, before the
`CosineAnnealingLR` `T_max` fix (`1e833ed`). **They are pre-fix history and
are not authoritative.** The CE seed-1 numbers in that entry stand
unchanged: `configs/cifar_ce.yaml` inherits `warmup_epochs: 0`, so CE never
took the truncated-schedule path (its seed-1 baseline ends at lr 0.00000 on
the pre-fix commit, confirmed in `logs/cifar10_ce_seed1/run.log`).

**Commands** (GPU 3, sequential, both at commit `ce47876`):

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_experiment.py \
  --config configs/cifar_arcface.yaml \
  --set seed=1 out_dir=logs/postfix_seed1 unlearn.enabled=false

CUDA_VISIBLE_DEVICES=3 python scripts/unlearn_across_classes.py \
  --config configs/cifar_arcface.yaml --forget-classes 0,1,2,3 \
  --ckpt logs/postfix_seed1/cifar10_arcface_seed1/ckpt.pt \
  --conditions random_label_clfonly \
  --set seed=1 out_dir=logs/postfix_seed1
```

Written to `logs/postfix_seed1/`, a fresh tree — the pre-fix seed-1
directories under `logs/` were left untouched for comparison, not
overwritten.

**Provenance.** Every new run records `git_commit: ce47876`. The baseline
ends at **lr 0.00000** (pre-fix seed-1 ended at lr 0.00670 — the truncation
bug, visible directly in the two `run.log`s).

**Baseline accuracy and head fairness.** ArcFace seed-1 test accuracy
91.71% -> **93.09%**. CE seed-1 is 93.14%, so the heads now match to
**0.05pp** at seed 1 (pre-fix gap: 1.43pp), mirroring seed 0's post-fix
parity (93.36 / 93.41). The geometry comparison below is therefore not
confounded by one head simply being a worse model. Collapse also tightened:
nc2 0.0813 -> **0.0348**, nc3_centred_mean 0.936 -> **0.970**, nc1_angular
0.0607 -> **0.0384**.

**`random_label_clfonly`, ArcFace, seed 1 (frozen backbone — any nc3 change
is the classifier alone).** `nc3_centred_forget`; `output_forget` is 0.0000
at every epoch >= 1 in all four classes:

| fc | ep1 pre-fix | ep1 post-fix | final pre-fix | final post-fix | out_retain post-fix |
|---|---|---|---|---|---|
| 0 | -0.8490 | **-0.8832** | -0.8879 | **-0.9161** | 0.9351 |
| 1 | -0.8458 | **-0.9315** | -0.8916 | **-0.9601** | 0.9291 |
| 2 | -0.6782 | **-0.8647** | -0.7209 | **-0.8983** | 0.9403 |
| 3 | -0.9444 | **-0.8890** | -0.9558 | **-0.9039** | 0.9494 |

Uncentred final `nc3_uncentred_forget` is -0.981 in all four. Reported per
CLAUDE.md, but **not** as independent proof of a flip: this ArcFace baseline
is already at `nc3_uncentred_mean` **-0.8787** before any unlearning, so
-0.981 is a within-head move from an already-negative starting point, not a
sign reversal. The sign reversal above is in the **centred** column. Both are
the same model under two conventions; the theory's prediction concerns the
uncentred weight (see 2026-09-09), but the uncentred convention is not
head-comparable (see 2026-09-11). Retain-class alignment stays high
(nc3_centred_retain_mean
+0.966 to +0.975) and `output_retain` 0.9291-0.9494 against a 0.9309
baseline, so nothing collapsed to buy the forgetting.

**The flagged fc2 anomaly is gone, and the earlier explanation was wrong.**
Pre-fix fc2 was the one weak ArcFace point (ep1 -0.6782, final -0.7209,
with a low epoch-0 start of +0.8956). Post-fix it starts at +0.9993 and
lands at -0.8983, inside the family with the other three. The previous
entry attributed this to the seed-1 checkpoint's wider per-class alignment
spread (std 0.090 vs seed 0's 0.043) read as seed noise. That spread was
itself a symptom of the truncated LR schedule — it disappears when the
schedule decays fully. Post-fix the four ArcFace ep1 values span
-0.8647..-0.9315, a **0.067** range against pre-fix's 0.266. No anomaly
remains to explain.

**Does the 16/16 directional claim survive? Yes, and on firmer ground.**
All 4 ArcFace seed-1 points still flip negative at epoch 1; CE seed-1 stays
positive at epoch 1 in all 4 (+0.4354 to +0.4842, re-read from
`trajectory.jsonl`, not from a tally). Of the 16 (head, class, seed)
points, 13 now rest on runs that are either post-fix or provably unaffected
by the bug: 8 CE (warmup_epochs=0 throughout), 4 ArcFace seed-1 (this
entry), 1 ArcFace seed-0 fc0 (re-verified in the LR-fix entry). **The
remaining 3 — ArcFace seed-0, fc1/fc2/fc3 — are still pre-fix** and should
be re-run before the count is quoted in the paper. Every post-fix ArcFace
point re-run so far has come back the same sign or stronger, so there is no
live reason to expect those three to differ.

**Not run here, deliberately:** `finetune`, retrain references, other
methods, other seeds. This was a scoped re-verification of one condition.

---

## 2026-09-14 — CIFAR ArcFace seed-0 rebuilt post-fix; the last 3 pre-fix points are gone, 16/16 holds

**Decided:** The three ArcFace seed-0 points flagged as still pre-fix in the
entry above (fc1/fc2/fc3) are now replaced by post-fix runs, and the
seed-0 retrain references are rebuilt on the same footing. The 16/16
directional claim is quotable: **all 16 canonical points are now valid
under the corrected schedule: all eight ArcFace points were run post-fix,
while all eight CE points were unaffected because CE uses no warmup.** No
pre-fix artifact was modified or reused.

**Provenance.** A clean seed-0 ArcFace baseline was trained from scratch
into a fresh tree (`logs/postfix_seed0_clean/`), its four
`random_label_clfonly` classes run from that checkpoint, and four
retrain references built independently in
`logs/retrain_postfix_arcface_seed0/`. All **9** run directories record
`git_commit: 6015e37`, all end at **lr 0.00000**, all log `run complete`.
Workstream A ran on GPU 0, workstream B on GPU 1; GPUs 2-3 belonged to
another user and were not touched. The two out_dirs are disjoint, so no
`RunDir` target was contended.

**Baseline fairness versus CE holds at seed 0.** The clean baseline
reproduces the existing post-fix seed-0 run to four decimals — test
accuracy **0.9341**, nc1 1.1107, nc2 0.0467, nc3_centred_mean 0.9617,
nc3_uncentred_mean -0.8786, identical to `logs/cifar10_arcface_seed0`.
CE seed-0 is 0.9336, so the heads match to **0.05pp**, the same parity
already established at seed 1. The geometry comparison below is not
confounded by one head being the weaker model.

**`random_label_clfonly`, ArcFace, seed 0, post-fix** (frozen backbone, so
any nc3 movement is the classifier alone). `output_forget` is 0.0000 at
every epoch >= 1 in all four classes:

| fc | ep0 nc3_c | ep1 nc3_c | final nc3_c | final nc3_u | out_retain | test_acc |
|---|---|---|---|---|---|---|
| 0 | +0.9692 | **-0.9109** | -0.9427 | -0.9809 | 0.9380 | 0.8442 |
| 1 | +0.9585 | **-0.9133** | -0.9459 | -0.9812 | 0.9326 | 0.8393 |
| 2 | +0.9528 | **-0.7893** | -0.8236 | -0.9809 | 0.9427 | 0.8484 |
| 3 | +0.9695 | **-0.9452** | -0.9539 | -0.9812 | 0.9538 | 0.8584 |

Both conventions are reported per CLAUDE.md: `nc3_c` is centred with
`exclude_from_centre=<forget_class>`, `nc3_u` uncentred, where all four land
at -0.981. The sign reversal is established in the **centred** column
(+0.95..+0.97 at epoch 0 to -0.79..-0.95 at epoch 1); `nc3_u` is **not**
independent proof of it, because this baseline's `nc3_uncentred_mean` is
already **-0.8786** before any unlearning, so -0.981 is a within-head move
from an already-negative start. The theory's prediction concerns the
uncentred weight (2026-09-09); its cross-head comparability is what
2026-09-11 rules out. Retain-class alignment stays high
(`nc3_centred_retain_mean` +0.9599 to +0.9630) and `output_retain`
0.9326-0.9538 against a 0.9341 baseline, so nothing collapsed to buy the
forgetting.

**fc1 and fc3 came back unchanged in sign and close in magnitude; fc2
moved.** Pre-fix seed-0 ep1 values were fc1 -0.7891, fc2 -0.9600, fc3
-0.9403. Post-fix they are fc1 -0.9133, fc2 -0.7893, fc3 -0.9452. fc1
strengthened, fc3 is flat, and **fc2 weakened** — it is now the softest of
the four rather than the firmest. Note this is the opposite direction from
the seed-1 story, where fixing the LR bug *rescued* fc2 from -0.6782 to
-0.8647. So fc2's position within the family is not stable across seeds,
and the earlier reading that fc2 is intrinsically the weak class does not
survive either. What is stable is the sign: fc2 is comfortably negative in
both seeds post-fix. The seed-0 post-fix ep1 spread is 0.156
(-0.7893..-0.9452), wider than seed 1's 0.067.

**Does the 16/16 directional claim survive? Yes. All 16 canonical points
are now valid under the corrected schedule: all eight ArcFace points were
run post-fix, while all eight CE points were unaffected because CE uses no
warmup.** All 4 ArcFace seed-0 points flip negative at epoch 1; all 4 CE
seed-0 points stay positive (+0.4085, +0.4655, +0.4306, +0.4331, re-read
from `trajectory.jsonl`). With the 8 seed-1 points from the entry above,
that is 16/16 — and the **3 pre-fix holdouts named there are now retired.**

**Retrain references, ArcFace seed-0, post-fix** (fresh backbone+head on
retain-only; the comparison point every probe gap is measured against):

| fc | probe_forget | out_forget | out_retain | test_acc | nc3_c fgt | nc3_u fgt |
|---|---|---|---|---|---|---|
| 0 | 0.6030 | 0.0000 | 0.9352 | 0.8417 | -0.9914 | -0.9925 |
| 1 | 0.5950 | 0.0000 | 0.9310 | 0.8379 | -0.7681 | -0.9753 |
| 2 | 0.5440 | 0.0000 | 0.9390 | 0.8451 | -0.9952 | -0.9961 |
| 3 | 0.5680 | 0.0000 | 0.9484 | 0.8536 | -0.9736 | -0.9856 |

The reference is stable across classes: probe_forget mean **0.5775**, sd
0.0269, range 0.059. That answers the worry `retrain_stability.py` was
written for — the fc0 value is not a fluke, so gaps measured against it
are trustworthy.

**Probe gap to retrain** — *our arithmetic (unlearned `probe_forget` minus
retrain `probe_forget`), not a metric from the AISTATS paper.* Computed
only from the seed-0 post-fix artifacts listed above; no pre-fix or
cross-seed number enters it:

| fc | unlearned | retrain | gap |
|---|---|---|---|
| 0 | 0.9410 | 0.6030 | **+0.3380** |
| 1 | 0.9760 | 0.5950 | **+0.3810** |
| 2 | 0.9160 | 0.5440 | **+0.3720** |
| 3 | 0.8560 | 0.5680 | **+0.2880** |

Mean **+0.345**, sd 0.042. The gap is large and consistent at all four
classes.

**Scientific verdict, updated.** Under ArcFace the classifier weight
undergoes a **centred NC3 sign reversal** at every class and both seeds
(epoch 0 +0.94..+1.00 to epoch 1 -0.79..-0.95); the final uncentred value is
-0.981, but that convention's baseline was already about **-0.879**, so it
records a within-head deepening rather than an independent sign reversal.
Meanwhile the features of the forget class remain **substantially more
linearly decodable than after retraining** — a probe gap of roughly 35
points against the retrained reference. That is a statement about how much
class information a linear probe can still recover, not a demonstration
that the representation was left unchanged; these runs do not measure
feature movement directly, and the margin loss may well have moved the
features without closing the probe gap. The result is consistent with the
output-level illusion described by Gao et al., but direct equivalence
remains unverified until their reference implementation is reproduced. The
premise that normalisation closes the shortcut is **not supported on
CIFAR-10** — outcome (2) from CLAUDE.md is not what happened, and outcome
(1) is not supported by the probe gap. Note also that the
retrain references themselves sit at nc3_c -0.77 to -0.995. The retrained
backbone and head received **no positive forget-class examples**, but the
head keeps all `num_classes` output rows (`unlearn.retrain` builds it with
`num_classes`, and `ForgetSplit` indexes the train split without
relabelling), so the forget-class row still participated in the softmax
denominator and received **negative gradients from every retain example**.
Its anti-alignment is the endpoint of negative-only optimisation, not a
temporal flip and not an untouched random weight. Consequently **negative
NC3 alignment alone does not distinguish the resulting models** — the probe
gap is what does. This matches the faces result (2026-09-14) in direction of
conclusion though not in mechanism, where ArcFace showed no centred sign
reversal at all.

**Limitations.**
- `evaluate_light` writes only `nc3_centred_forget` to `trajectory.jsonl`,
  so the **epoch-1 uncentred value does not exist** for any run. Every ep1
  number quoted here and in the entries above is centred. If the paper
  reports an epoch-1 claim under the uncentred convention, the trajectory
  evaluator has to be extended and these runs redone.
- One seed per cell (seed 0 here, seed 1 above); n=2 seeds total, no
  error bars.
- `random_label_clfonly` only. `finetune`, other methods and other
  datasets were out of scope for this run.
- `forget-heldout 0` in every split (`split_mode=all`), so these runs say
  nothing about held-out generalisation.
- The epoch-convention open decision below (epoch 1 vs end of budget) is
  still unresolved and still changes what gets quoted.

**Supersedes:** the "remaining 3 — ArcFace seed-0, fc1/fc2/fc3 — are still
pre-fix" caveat in the 2026-09-14 seed-1 entry, and that entry's
attribution of fc2's behaviour to a per-class property.

---

## 2026-09-15 — Reproducible 40-image NC3 class-mean robustness diagnostic

**Purpose.** Test whether a forget-class mean estimated from only 40
training images — the per-identity budget on the face set — can make the
CIFAR ArcFace nc3 flip disappear. If it can, the CIFAR-versus-faces
divergence reported on 2026-09-14 is an estimator artefact rather than a
real difference between the datasets. This run replaces the ad-hoc check in
that entry with a committed, reproducible artifact.

**Provenance.**

| item | value |
|---|---|
| diagnostic code | `scripts/nc3_mean_robustness.py` at commit `77e58f3`, clean tree |
| baseline run | `logs/postfix_seed0_clean/cifar10_arcface_seed0` (recorded commit `6015e37`) |
| reference run | `logs/postfix_seed0_clean/cifar10_arcface_seed0_random_label_clfonly_fc0` (recorded commit `6015e37`) |
| baseline checkpoint SHA-256 | `b5d408cb6f8374483d37f4d25d9ddcd6689fefe75c6030f52cabfaf148aa8742` |
| device | NVIDIA A100-SXM4-40GB, CUDA, python 3.10.12 |
| forget class | 0 (5,000 training images) |
| epoch | 1 of `random_label_clfonly` |
| K | 40 images per trial, sampled without replacement |
| trials | 20 |
| diagnostic sample seed | 0 (separate from the experiment seed, which stays 0) |
| output | `logs/nc3_mean_robustness/cifar10_arcface_seed0_fc0_k40_n20_s0` |

**Replay gate — passed.** The diagnostic replays one epoch of
`random_label_clfonly` from the baseline checkpoint rather than measuring a
same-config lookalike, and gates itself on landing on the canonical model:

    canonical epoch-1 nc3_centred_forget   -0.910924935275755
    replayed epoch-1 nc3_centred_forget    -0.910924935275755
    replay_delta                            0.0
    verified                                true
    condition_prefix_executed               []   (fc0 was first in its sweep)

The empty prefix matters: the reference sweep ran `--forget-classes 0,1,2,3`
in one process, so only a first-position target can be reached exactly from
the checkpoint. It was in first position, and the delta is exactly zero.

**Trial results.** Recomputed directly from the 20 rows in `trials.jsonl`,
independently of the `summary` block in `result.json`; the two agree.

| statistic | value |
|---|---|
| full-data `nc3_centred_forget` (all 5,000 images) | -0.910924935275755 |
| mean over 20 trials | -0.9011804731354751 |
| std, ddof=1 (sample) | 0.047615275838726635 |
| std, ddof=0 (population) | 0.046409630127141964 |
| minimum (most negative) | -0.9673207969625082 |
| maximum (least negative) | -0.8078338493014008 |
| trials below 0 | 20 / 20 |
| trials below -0.5 | 20 / 20 |
| sign flips (value >= 0) | 0 / 20 |
| max abs deviation from full data | 0.10309108597435424 (trial 3, -0.8078338493014008) |

Max absolute deviation is our arithmetic over the trial rows, not a quantity
the script reports. Mean, both standard deviations, min, max and both counts
agree with `result.json`'s `summary` block to floating-point precision.
Retain-class alignment at the same weight matrix is +0.9615065222041199, and
`exclude_from_centre=0` is used throughout, so the centring artefact
documented in CLAUDE.md is not in play.

**Historical comparison — similar, deliberately not identical.** The scratch
check in the 2026-09-14 faces entry reported mean -0.9008, std 0.0575, min
-0.9693, max -0.7288, 20/20 below -0.5, 0/20 flips. That run's sampling seed
was never recorded, so its individual trials cannot be reproduced and this is
**not** a trial-by-trial reproduction of it. The two agree closely on the
mean (-0.9008 vs -0.9012) and on both counts. They differ in spread and
extremes; because the scratch sampling seed was not recorded, the two sets of
draws may differ, and that is a sufficient account of the difference without
appealing to anything else. The older numbers stay in the file as historical
evidence; a note beside them marks this entry as the authoritative result.

**Conclusion, scoped to what was measured.** For this CIFAR-10 ArcFace
seed-0, fc0, epoch-1 model, reducing the class-mean estimator from 5,000
training images to 40 did not change the sign in any of 20 draws, and every
value remained below -0.5. Within this CIFAR control, reducing estimator
sample count alone was insufficient to remove the flip. Thus, the simple
hypothesis that 40 images are inherently too few to estimate a sign-stable
class mean is not supported here. This does **not** establish that face
identity means are stable at 40 images, nor rule out an interaction between
sample count and face-specific within-class variability or geometry; a direct
face-specific control remains necessary.

**Limitations.**

- One model, one class (0), one seed (0), one sample size (K=40), one epoch
  (1). Nothing here generalises to other classes, seeds, K, or epochs.
- The 20 trials resample the class mean from **one fixed model's features**.
  They are not 20 independent training runs, and the spread above is
  estimator noise only — it carries no information about run-to-run variance.
- The diagnostic measures `nc3_centred_forget` alone. It supports no
  inference about representation erasure, probe behaviour, or whether
  anything was actually forgotten.
- Replay is deterministic for this device and library versions, not bitwise
  portable across them; both are recorded in the artifact.

**Supersedes:** the scratch 40-sample numbers in the 2026-09-14 faces entry
as the authoritative result for this check. It does not supersede that
entry's finding, which is unchanged.

---

## 2026-09-15 — Face NC3 non-flips survive class-mean subsampling

The CIFAR entry above closed the estimator question on CIFAR only. These two
runs put the same question to the face models themselves, at the two targets
where an exact replay is available.

### Provenance

Both executed at commit `1a59ca3` with a clean tree, concurrently on two
separate A100-SXM4-40GB GPUs, diagnostic sample seed 0, 20 trials each.

| | faces-100 (Run A) | faces-1000 (Run B) |
|---|---|---|
| baseline | `logs100/faces_arcface_seed0` (recorded `61abcbf`) | `logs/faces_arcface_seed0` (recorded `ea54072`) |
| reference | `logs100/faces_arcface_seed0_random_label_clfonly_fc0` (`61abcbf`) | `logs/faces_arcface_seed0_random_label_clfonly_fc0` (`4516b1f`) |
| baseline ckpt SHA-256 | `92d20ba7eb23d3a0e2a557d7202064263f332f677b846767839c03a262677c11` | `48c19bfd2ff84d17eda447dfd73417894859b5c6a71936a8888ff8b75fea5d80` |
| output | `logs100/nc3_mean_robustness/faces100_arcface_seed0_fc0_k40_n20_s0` | `logs/nc3_mean_robustness/faces1000_arcface_seed0_fc0_k20_n20_s0` |

**Both replay gates passed.** faces-100: canonical and replayed epoch-1
`nc3_centred_forget` both **+0.61126007226103**, delta **0.0**. faces-1000:
both **+0.9792542591275493**, delta **0.0**. `verified` is true on both and
the executed condition prefix is empty on both, so each landed on the
canonical model rather than a same-config lookalike.

### Results

Recomputed from `trials.jsonl` in each directory; both agree with the stored
`summary` block.

| | faces-100 | faces-1000 |
|---|---|---|
| full-data `nc3_centred_forget` | +0.61126007226103 | +0.9792542591275493 |
| K / pool | **40 of 110** | **20 of 40** |
| mean | +0.6431207413927054 | +0.9683103440984387 |
| std (ddof=1) | 0.13543511115804396 | 0.012532928505315242 |
| min | +0.39485476073452863 | +0.9419258390082438 |
| max | +0.8505011109674718 | +0.9831519597294159 |
| positive / non-positive | **20 / 0** | **20 / 0** |
| above +0.5 | **18 / 20** | **20 / 20** |
| max abs deviation from full data | 0.23924103870644187 | 0.03732842011930548 |

Retain-class alignment at the same weight matrices: +0.8490 and +0.8781.
Max absolute deviation is our arithmetic over the trial rows, not a quantity
the script reports.

### Conclusion, stated narrowly

On the fixed faces-100 model, every 40-image estimate preserved the positive
epoch-1 non-flip. On the fixed faces-1000 model, every 20-of-40 estimate
preserved its strongly positive non-flip. **These results do not support sign
instability from class-mean subsampling as the explanation for the observed
non-flips in these two targets.**

Run B is a half-pool noise-slope stress test. It is **not** a 40-image
replication and **not** a CIFAR-equivalent experiment: on faces-1000 the 40
images are the entire pool available for that identity, so there is no larger
sample to reduce from.

The larger dispersion at faces-100 (std 0.135 vs 0.013) is **not** attributed
here to within-class variability. Sample fraction, the metric's operating
point, feature geometry and finite-population effects all differ between the
two runs, and this pair of runs cannot separate them.

No distributional inference is drawn from these spreads -- no
"N standard deviations from zero" claim, and no assumption that the trial
values are normally distributed. Nothing here isolates class count, proves
anything about the unknown population identity means, or establishes
representation erasure.

### Limitations

- One identity, one model, one seed, one epoch per dataset (forget class 0,
  seed 0, epoch 1).
- The 20 trials per run are repeated estimates from one fixed model's
  features, not independent training runs; the spreads are estimator noise
  only and say nothing about run-to-run variance.
- Epoch-1 convention only. The open epoch-convention decision below still
  applies, and neither run speaks to the end-of-budget reading.
- No face retrain reference exists on either dataset, so no probe gap is
  computable and no probe number on faces is interpretable.
  > **Superseded on 2026-09-15** by "Complete face probe-gap matrix: CIFAR
  > separation does not transfer cleanly". Retrain references now exist for
  > all four identities on both face subsets and both heads, so face probe
  > gaps are computable. This limitation does not otherwise affect the NC3
  > subsampling result recorded in this entry, which never used a probe.
- No historical dataset manifest exists. Dataset identity was checked by
  counts and folder names against the recorded `run.log` headers, not by
  content hashes.
- The faces-100 extraction seed remains unrecorded, so that subset is still
  not reproducible from the repository alone.
- Both results therefore require the current dataset directory contents to
  reproduce exactly; the directories are the primary artifacts.

---

## 2026-09-15 — Complete face probe-gap matrix: CIFAR separation does not transfer cleanly

**Decided:** With the last three faces-1000 retrain references recovered, the
face probe-gap matrix is complete at 16 cells and is recorded here as the
reference table. The CIFAR-10 result — consistently positive gaps, much
larger under ArcFace — does not transfer cleanly to either face subset, and
the two face subsets do not agree with each other. They are reported
separately throughout and are **not** combined into one "faces" result.

**Because:** every previous face probe statement in this file was blocked on
missing retrain references (see the supersession notes on the 2026-09-14
100-identity entry and the 2026-09-15 NC3 subsampling entry). Those
references now exist for all four identities on both subsets and both heads.

### The metric

    probe_gap_to_retrain* = unlearned probe_forget - matched retrain probe_forget

The asterisk is deliberate and travels with the name. **This is project
arithmetic, not a metric defined by Gao et al.** Say so wherever it appears,
in this file, in a table, in a slide, in the paper. "Matched" means the
retrain reference for the same dataset, head, forget class and seed as the
unlearned run it is subtracted from — never a reference borrowed across any
of those four.

Unlearned condition is `random_label_clfonly` throughout. Retrain is a fresh
backbone and head trained from scratch on the retain set only.
`output_forget` is 0.0000 in all 24 unlearned and all 24 retrain models, so
every gap below is measured at equal output-level forgetting.

### Resolution and probe protocol — read before interpreting any number

**CIFAR-10 is 32x32** (native, `cifar_transforms`, 3x3 stem via
`small_input: true`). **Only the face experiments are 112x112**
(`image_size: 112`, standard stem). An earlier draft of this analysis
attributed 112 px to all 24 cells; that was wrong.

**The retrained backbone never saw the forget class.** It is trained on the
retain set only, from random initialisation.

**The post-hoc logistic-regression probe was then fitted on the full labelled
training-feature set, including the forget class, and evaluated on held-out
test images of that class.** `train_eval_loader` is built with `None` indices
in both drivers that produced these cells
(`scripts/unlearn_across_classes.py`, `scripts/retrain_stability.py`), so the
probe's fitting set is the whole training set. This is why a retrained model
scores high here: it is not recognising a class it was trained on, it is
being handed labelled examples of that class after the fact.

**CE and ArcFace probes used the same estimator, hyperparameters, fit and
evaluation policy, and seed** — one call site (`src/train.py:192`),
`LogisticRegression(max_iter=2000, random_state=0)`, identical fit and
evaluation sets, no head-dependent branching. `extract` calls the head
without labels, so no margin is applied and both heads hand the probe raw
backbone features.

**Frozen-backbone unlearning leaves the unlearned features identical to the
original features.** `random_label_clfonly` updates the classifier only. No
gap below is evidence of an unlearning-induced feature change, because there
is no feature change.

**Negative probe gaps do not mean better unlearning.** A negative gap says
the retrained reference's representation was more linearly decodable for
that class than the original's. That is a statement about the reference.

**High retrain probe accuracy indicates relative linear decodability after
labelled probe fitting; it is not evidence of representation erasure.**
Retrain `probe_forget` spans 0.7742-1.0000 on faces-100, 0.5000-0.9000 on
faces-1000 and 0.5440-0.8920 on CIFAR-10. Generic features appear sufficient
to separate an identity once a probe is given labels for it, which limits
this metric's discriminative value in this setup — most visibly on faces-100,
where the retrain references pool to 0.8909.

### Face retrain matrix — 16 cells

Seed 0. `u` = unlearned (`random_label_clfonly`), `r` = retrain reference.
Correct counts are exact integers over each identity's own test denominator;
every `probe_forget` recorded is an exact k/denominator ratio.

#### faces-100 (100 identities, `casia-webface-folders-100id`, 112x112)

| head | fc | u probe | r probe | u correct | r correct | den | gap* | u outR | r outR | u probeR | r probeR | r NC3 centred | r NC3 uncentred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ce | 0 | 0.8148 | 0.9259 | 22 | 25 | 27 | -0.1111 | 0.8217 | 0.8384 | 0.8435 | 0.8316 | +0.0445 | -0.7060 |
| ce | 29 | 0.9200 | 0.9600 | 23 | 24 | 25 | -0.0400 | 0.8231 | 0.8417 | 0.8426 | 0.8340 | +0.0811 | -0.6465 |
| ce | 60 | 0.7742 | 0.8065 | 24 | 25 | 31 | -0.0323 | 0.8250 | 0.8279 | 0.8440 | 0.8266 | -0.0641 | -0.6910 |
| ce | 95 | 0.8148 | 0.8889 | 22 | 24 | 27 | -0.0741 | 0.8300 | 0.8361 | 0.8435 | 0.8339 | +0.0247 | -0.6261 |
| arcface | 0 | 0.9259 | 0.8519 | 25 | 23 | 27 | +0.0741 | 0.8413 | 0.8631 | 0.8335 | 0.8114 | -0.9987 | -0.9912 |
| arcface | 29 | 0.7600 | 0.9600 | 19 | 24 | 25 | -0.2000 | 0.8420 | 0.8539 | 0.8349 | 0.8067 | -0.9968 | -0.9888 |
| arcface | 60 | 0.7742 | 0.7742 | 24 | 24 | 31 | 0.0000 | 0.8414 | 0.8581 | 0.8349 | 0.7960 | -0.9988 | -0.9916 |
| arcface | 95 | 0.8148 | 1.0000 | 22 | 27 | 27 | -0.1852 | 0.8416 | 0.8554 | 0.8345 | 0.8078 | -0.9990 | -0.9901 |

#### faces-1000 (1000 identities, `casia-webface-folders`, 112x112)

| head | fc | u probe | r probe | u correct | r correct | den | gap* | u outR | r outR | u probeR | r probeR | r NC3 centred | r NC3 uncentred |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ce | 0 | 0.7000 | 0.8000 | 7 | 8 | 10 | -0.1000 | 0.7232 | 0.7455 | 0.7373 | 0.7285 | +0.0588 | -0.6605 |
| ce | 122 | 0.7000 | 0.6000 | 7 | 6 | 10 | +0.1000 | 0.7202 | 0.7458 | 0.7373 | 0.7325 | +0.0053 | -0.6799 |
| ce | 389 | 0.8000 | 0.8000 | 8 | 8 | 10 | 0.0000 | 0.7276 | 0.7477 | 0.7372 | 0.7284 | +0.0470 | -0.6319 |
| ce | 794 | 0.8000 | 0.7000 | 8 | 7 | 10 | +0.1000 | 0.7238 | 0.7442 | 0.7372 | 0.7273 | -0.1041 | -0.7395 |
| arcface | 0 | 0.7000 | 0.9000 | 7 | 9 | 10 | -0.2000 | 0.7237 | 0.7310 | 0.7025 | 0.7077 | -0.9799 | -0.9862 |
| arcface | 122 | 0.6000 | 0.6000 | 6 | 6 | 10 | 0.0000 | 0.7232 | 0.7232 | 0.7026 | 0.7053 | -0.9795 | -0.9897 |
| arcface | 389 | 0.7000 | 0.8000 | 7 | 8 | 10 | -0.1000 | 0.7221 | 0.7253 | 0.7025 | 0.7002 | -0.9734 | -0.9873 |
| arcface | 794 | 0.6000 | 0.5000 | 6 | 5 | 10 | +0.1000 | 0.7241 | 0.7135 | 0.7026 | 0.6906 | -0.9625 | -0.9898 |

Note the denominators. faces-1000 gives 10 test images per identity, so every
gap there is a whole-image step of 0.1. faces-100 gives 25-31, so its steps
are 0.032-0.040. The two subsets are not measured at the same resolution.

### CIFAR-10 contextual comparison — 8 cells, reported separately

The per-class CIFAR raw tables are already recorded: CE and the seed-0
ArcFace rebuild in "2026-09-14 — CIFAR ArcFace seed-0 rebuilt post-fix", the
seed-1 replication in "2026-09-14 — CIFAR ArcFace seed-1 re-run post-LR-fix",
and the fc0-3 sweep in "2026-09-14 — finetune and random_label_clfonly
patterns hold at fc0-3". They are not duplicated here. What this entry needs
is the two group summaries, for comparison against the four face groups:

| group | macro mean | sample SD | median | range | pos/zero/neg | pooled den | pooled u | pooled r | pooled signed gap* |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 CE | +0.1042 | 0.0391 | +0.0890 | +0.0780 .. +0.1610 | 4/0/0 | 4000 | 0.9207 (3683) | 0.8165 (3266) | +0.1042 (+417/4000) |
| CIFAR-10 ArcFace | +0.3448 | 0.0421 | +0.3550 | +0.2880 .. +0.3810 | 4/0/0 | 4000 | 0.9223 (3689) | 0.5775 (2310) | +0.3448 (+1379/4000) |

CIFAR-10 is 32x32; these eight cells are context for the face matrix, not
part of it.

### Six group summaries

Sample SD throughout (ddof=1), over the four classes in each group. Pooled
accuracies are total correct over total denominator.

| group | macro mean | sample SD | median | range | pos/zero/neg | pooled den | pooled u | pooled r | pooled signed gap* |
|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 CE | +0.1042 | 0.0391 | +0.0890 | +0.0780 .. +0.1610 | 4/0/0 | 4000 | 0.9207 (3683) | 0.8165 (3266) | +0.1042 (+417/4000) |
| CIFAR-10 ArcFace | +0.3448 | 0.0421 | +0.3550 | +0.2880 .. +0.3810 | 4/0/0 | 4000 | 0.9223 (3689) | 0.5775 (2310) | +0.3448 (+1379/4000) |
| faces-100 CE | -0.0644 | 0.0361 | -0.0570 | -0.1111 .. -0.0323 | 0/0/4 | 110 | 0.8273 (91) | 0.8909 (98) | -0.0636 (-7/110) |
| faces-100 ArcFace | -0.0778 | 0.1361 | -0.0926 | -0.2000 .. +0.0741 | 1/1/2 | 110 | 0.8182 (90) | 0.8909 (98) | -0.0727 (-8/110) |
| faces-1000 CE | +0.0250 | 0.0957 | +0.0500 | -0.1000 .. +0.1000 | 2/1/1 | 40 | 0.7500 (30) | 0.7250 (29) | +0.0250 (+1/40) |
| faces-1000 ArcFace | -0.0500 | 0.1291 | -0.0500 | -0.2000 .. +0.1000 | 1/1/2 | 40 | 0.6500 (26) | 0.7000 (28) | -0.0500 (-2/40) |

### Three paired ArcFace-minus-CE summaries

Paired within dataset, class by class. Computed from exact integer counts
over each identity's denominator, not by floating-point subtraction of the
stored accuracies — the faces-1000 fc794 difference is a rational zero and a
float subtraction of 0.8-0.7 against 0.6-0.5 renders it as a spurious
negative.

| dataset | fc values | mean | sample SD | median | range | pos/zero/neg |
|---|---|---|---|---|---|---|
| CIFAR-10 | +0.2600, +0.3020, +0.2110, +0.1890 | +0.2405 | 0.0506 | +0.2355 | +0.1890 .. +0.3020 | 4/0/0 |
| faces-100 | +0.1852, -0.1600, +0.0323, -0.1111 | -0.0134 | 0.1555 | -0.0394 | -0.1600 .. +0.1852 | 2/0/2 |
| faces-1000 | -0.1000, -0.1000, -0.1000, 0.0000 | -0.0750 | 0.0500 | -0.1000 | -0.1000 .. 0.0000 | 0/1/3 |

faces-100 order is fc0, 29, 60, 95; faces-1000 is fc0, 122, 389, 794;
CIFAR-10 is fc0-3.

**faces-1000 fc794 is exactly zero.** Both heads give a gap of exactly
`+1/10` there (CE 8/10 unlearned against 7/10 retrain; ArcFace 6/10 against
5/10), so the paired difference is the rational 0. Record it as `0.0000`.
Do not record it as a signed near-zero.

### Separate observation — faces-100 ArcFace `probe_retain` deficit

Reported on its own because it is not part of the gap result and must not be
folded into it.

| quantity (ArcFace minus CE) | fc0 | fc29 | fc60 | fc95 | mean |
|---|---|---|---|---|---|
| unlearned `probe_retain` | -0.0100 | -0.0077 | -0.0090 | -0.0090 | -0.0089 |
| retrain `probe_retain` | -0.0202 | -0.0273 | -0.0306 | -0.0260 | -0.0260 |

The deficit is systematic — negative in all four identities on both sides,
and larger on the retrain side (ArcFace retrain `probe_retain` 0.7960-0.8114
against CE 0.8266-0.8340). It moves opposite to task accuracy: ArcFace's
`output_retain` is *higher* than CE's on the same runs (retrain 0.8539-0.8631
against CE 0.8279-0.8417; mean difference +0.0216 retrain, +0.0166
unlearned).

**This is not evidence of representation collapse and is not labelled as
such.** The two heads' probes are fitted identically — same estimator, same
hyperparameters, same fit and evaluation sets, same seed — so the difference
is not a probe-procedure artefact. What differs is that independently fitted
probes receive representations with different feature geometry, ArcFace's
having been learned under a normalised angular-margin objective.
Distinguishing that from degraded retain-class information would need a
matched-accuracy control that has not been run.

### Provenance

Four tiers, separated because they are not equivalent.

**Direct clean retrain references at `589b421`.** All eight faces-100 retrain
references and five of the eight faces-1000 retrain references (CE fc0,
fc122, fc389; ArcFace fc0, fc122). Each is a self-contained
`scripts/retrain_stability.py` invocation naming a config tracked at that
commit, with no command-line override of any model or data setting, training
from scratch with no checkpoint dependency.

**The three recovered faces-1000 cells at `0a92c69`.** CE fc794, ArcFace
fc389, ArcFace fc794. Each records `0a92c69` in its own `env.json`, written
by `RunDir` at run time on a clean tree, from a self-contained invocation
against a config tracked at that commit. Their predecessors were killed by
host-RAM exhaustion during evaluation and are preserved under
`logs/failed_runs/2026-09-15_faces1000_wave2_oom/` with a SHA-256 manifest;
those archived logs independently confirm identical overlapping training
trajectories (CE fc794 epochs 1-11; ArcFace fc389 all 40 epochs, loss, acc
and LR identical). The archive corroborates these cells — it is not their
source, and they are not reconstructed from it.

**Reconstructed faces-1000 baseline and unlearning provenance.** The
faces-1000 ArcFace baseline (`logs/faces_arcface_seed0`, `ea54072`) ran with
`--set head.s=96.0` against a config that read `s: 64.0` at that commit, so
its effective configuration is recoverable only from the recorded `argv` and
resolved `config.json`, not from the config file at the recorded commit. The
unlearning sweep at `4516b1f` consumed that baseline's `ckpt.pt` and inherits
the same status.

**Reconstructed faces-100 baseline and unlearning provenance — different
cause.** Both faces-100 baselines and both unlearning sweeps record
`61abcbf` and invoke `configs/faces100_ce.yaml` and
`configs/faces100_arcface.yaml`, **but neither file was tracked in git at
`61abcbf`**; both were first committed later the same day at `6b1d54e`. The
config text used at run time is therefore not recoverable from the repository
at the recorded commit.

In every reconstructed case the effective configuration is fully recorded in
the run's own `config.json`, matches the HEAD config files exactly, and the
baseline `ckpt.pt` survives. "Reconstructed" here means the run-time config
text is not recoverable from the repository at the recorded commit. It does
not mean the numbers are in doubt.

**CIFAR provenance** spans `a963fb7`, `d67a72c`, `e5b0928` and `6015e37`.
CIFAR CE fc0 uniquely takes both its cells from the multi-row
`logs/cifar10_ce_seed0/results.jsonl`, written before the `RunDir` reuse
guard (see "2026-09-14 — RunDir silently mixed configs into one results
file"). The two rows used here (`random_label_clfonly`, `retrain`) are
correctly labelled, but they share a file with the known mislabelled
`finetune_ep30` row.

### Conclusion, scoped to what was measured

On CIFAR-10 at 32x32, `probe_gap_to_retrain*` is positive in all eight cells
and substantially larger under ArcFace (macro +0.3448) than under CE
(+0.1042), with the paired ArcFace-minus-CE difference positive in 4 of 4
classes. On faces-100 at 112x112 the mean gap is negative under both heads
(CE -0.0644, negative in 4 of 4; ArcFace -0.0778, 1 positive / 1 zero / 2
negative), and its paired ArcFace-minus-CE differences have mixed signs (2
positive, 2 negative) with no consistent head direction. On faces-1000 at
112x112 the gaps cluster around zero at one-image resolution (CE +0.0250,
that is +1 image of 40; ArcFace -0.0500, that is -2 images of 40), and its
paired differences are non-positive in these four identities (0 positive, 1
exactly zero, 3 negative) on a denominator of only 10 test images per
identity. The CIFAR-10 separation between heads therefore does not transfer
cleanly to either face subset, and the two face subsets do not agree with
each other; they are reported separately and are not combined. These gaps are
evidence about the relative linear decodability of the original versus the
retrained representation once a probe has been fitted on labelled examples of
the forget class; they are not evidence of representation erasure, and
because the unlearning backbone is frozen they are not evidence of any
unlearning-induced change in the features. No result here is attributed
causally to class count, head type or dataset, and no significance claim is
made.

### Limitations

- One seed (seed 0) per cell. No error bars, no seed replication on faces.
- Four identities per group, four classes per CIFAR group. n=4.
- Coarse face denominators: 10 test images per identity on faces-1000
  (every gap a 0.1 step), 25-31 on faces-100.
- Frozen-backbone unlearning (`random_label_clfonly`). The unlearned features
  are identical to the original features, so nothing here measures
  unlearning-induced representation movement.
- No significance test was run, and none is implied. No distributional or
  normality assumption is made about any spread reported above.
- `split_mode=all` gives `forget-heldout 0` in every run, so nothing here
  speaks to held-out generalisation.
- The epoch-convention decision below (epoch 1 versus end of unlearning
  budget) is still unresolved and still changes what may be quoted.
- The faces-100 extraction seed remains unrecorded, so that subset is not
  reproducible from the repository alone; both face results require the
  current dataset directory contents.
- Dataset root, class count, images per identity and identity membership all
  covary between faces-100 and faces-1000. No comparison here separates them,
  and none should be read as isolating any one of them.

**Supersedes:** the "no probe claim is available on any face dataset" block
in "2026-09-14 — 100-identity face subset", and the "no face retrain
reference exists on either dataset" limitation in "2026-09-15 — Face NC3
non-flips survive class-mean subsampling". Supersession notes are appended
beside both; their historical text is preserved.

---

## 2026-09-15 — NC3 convention clarification: reversal is established in the centred metric

**Decided:** A **flip** (equivalently, sign reversal) means a sign change
between epoch 0 and epoch 1 for the **same model, same class, same
convention**. Nothing else earns the word. All project claims of reversal are
in the **centred** convention.

**Counts, `random_label_clfonly`, epoch 0 → epoch 1, centred, from
`trajectory.jsonl`:**

| dataset | head | n | ep0 | ep1 | sign reversals |
|---|---|---|---|---|---|
| CIFAR-10 | ArcFace | 8 (4 cls x 2 seeds) | +0.937..+0.999 | -0.789..-0.945 | **8/8** |
| CIFAR-10 | CE | 8 | +0.951..+0.967 | +0.408..+0.484 | **0/8** |
| faces-100 | ArcFace | 4 | +0.686..+0.998 | -0.372..+0.611 | **1/4** |
| faces-1000 | ArcFace | 4 | +0.854..+0.950 | +0.450..+0.979 | **0/4** |

**No epoch-1 uncentred per-class values exist in current artifacts.**
`evaluate_light` writes only `nc3_centred_forget` per epoch, and the baseline
`original` row stores only `nc3_uncentred_mean` across all classes — never a
per-class forget value. Uncentred epoch-1 numbers must be written as `not
recorded`; they are never inferred from final-epoch values.

**ArcFace's uncentred baselines are already about -0.88** before any
unlearning (CIFAR -0.8786 / -0.8787, faces-100 -0.8778, faces-1000 -0.8833;
CE's are +0.60 to +0.72). A negative post-unlearning uncentred value is
therefore a within-head change, not a sign reversal, and raw uncentred values
are not compared across heads. See 2026-09-11 for the mechanism.

**Supported thesis, one convention throughout:** a centred NC3 reversal is
**not necessary** for zero output-level forgetting — both face subsets reach
`output_forget`=0 with reversal in 0/4 and 1/4 identities — and it is **not
sufficient evidence of representation erasure**, because on CIFAR-10 ArcFace
reverses in 8/8 points under a frozen backbone, where the representation is
unchanged by construction. No claim is made here about necessity or
sufficiency for "the illusion" itself.

**Scope of the subsampling controls.** The class-mean subsampling runs
(2026-09-15, both entries) directly establish sign stability **only for the
tested fc0, seed-0, epoch-1 models** on CIFAR-10, faces-100 and faces-1000.
They do **not** individually validate the other seven face identities; those
rest on their single recorded trajectory each.

**Still a recommendation, not a settled decision:** reporting `nc3_*_forget`
at epoch 1 as primary with the final epoch as sensitivity. That remains open
below and is Dr Rawat's call.

---

## 2026-09-15 — Paired full-versus-frozen movement controls (ArcFace, fc0, seed 0)

**Decided:** Full-model and frozen-backbone `random_label` were run as matched
pairs on CIFAR-10 and faces-1000 ArcFace. On CIFAR the centred NC3 reversal
appears only in the frozen member. faces-1000 does not support that
conclusion: neither member reversed.

### Provenance — both pairs verified paired

All four runs at commit `0bf8f05`, clean tree, forget class 0, seed 0, epochs
0-3, `split_mode=all`, no baseline retrained, no checkpoint written.

| | CIFAR-10 ArcFace | faces-1000 ArcFace |
|---|---|---|
| baseline | `logs/postfix_seed0_clean/cifar10_arcface_seed0` | `logs/faces_arcface_seed0` (effective `s=96` via recorded argv) |
| baseline ckpt SHA-256 | `b5d408cb6f83…` (both members) | `48c19bfd2ff8…` (both members) |
| **training trace SHA-256** | `94dec65196210be212a2699cf6f4e2d1c1cc0f13a28434f92d5f7c6eedc1347f` | `f9c5f9fdc3c6fcfdfe88a37b708d77168c0e4271f74802e38c2b780e2f707806` |
| observed train_eval index SHA-256 | `e3d6b519cf63…` | `1e80ea55e15b…` |
| physical GPU | 2 (`00000000:81:00.0`) | 3 (`00000000:C1:00.0`) |
| controls | 9 of 9 retain classes, exhaustive | 32 of 999, deterministic sample |
| artifacts | `logs/feature_movement_paired/cifar10_arcface_seed0_random_label_{full,clfonly}_fc0` | `…/faces_arcface_seed0_random_label_{full,clfonly}_fc0` |

**The trace hash matches within each pair**, so both members provably trained
on the same retain indices, forget indices and random targets, step for step.
That is what makes each a pair rather than two similar runs. The frozen
members show exactly 0.000 deg displacement at every epoch.

### Epoch-1 paired tables (the matched point: both members reach `output_forget`=0 at epoch 1)

**CIFAR-10 ArcFace** — epoch-0 baselines: centred **+0.9692**, uncentred **-0.8781**

| mode | out_f | out_r | probe_f | nc3_c | centred reversal | nc3_u | uncentred reversal | aligned forget | controls |
|---|---|---|---|---|---|---|---|---|---|
| full_model | 0.0000 | 0.9256 | 0.9180 | **+0.7653** | **no** | -0.9983 | no | 13.390 deg | 4.295 deg |
| classifier_only | 0.0000 | 0.9381 | 0.9410 | **-0.9107** | **yes** | -0.9806 | no | 0.000 deg | 0.000 deg |

**faces-1000 ArcFace** — epoch-0 baselines: centred **+0.8875**, uncentred **-0.8846**

| mode | out_f | out_r | probe_f | nc3_c | centred reversal | nc3_u | uncentred reversal | aligned forget | controls |
|---|---|---|---|---|---|---|---|---|---|
| full_model | 0.0000 | 0.6724 | 0.7000 | **+0.2221** | **no** | -0.9519 | no | 9.493 deg | 4.950 deg |
| classifier_only | 0.0000 | 0.7198 | 0.7000 | **+0.9789** | **no** | -0.9534 | no | 0.000 deg | 0.000 deg |

Uncentred NC3 sign-reverses in none of the eight rows: both baselines are
already near -0.88, exactly the case the CLAUDE.md guard covers. Every
reversal statement here is **centred**.

### Conclusion, scoped narrowly

**For ArcFace, class 0, seed 0 and this isolated random-label stream on
CIFAR-10, centred NC3 reversal occurred with a frozen backbone but not with a
trainable backbone.** Both members reached zero output-level forgetting at
epoch 1 with retain utility healthy (0.9381 frozen, 0.9256 full, against a
0.9332 baseline), and their trace hashes match.

**faces-1000 does not support that conclusion.** Neither member reversed
(frozen +0.9789, full +0.2221, from a +0.8875 baseline), so freezing is not
the operative variable there.

Not generalised to other classes, seeds, methods or datasets.

### Face full-model retain-utility loss

The faces full-model member lost retain accuracy at the matched point:
**0.7272 -> 0.6724, -5.5pp**, against its frozen twin's -0.7pp (0.7198). By
epoch 3 it is 0.6396, -8.8pp. The loss is real and asymmetric within the
pair, so the faces full-model movement numbers remain confounded by model
degradation and are not clean forgetting-induced movement. No learning rate
was tuned.

### Exposure — two different quantities, neither of them gradient magnitude

`random_label` takes one optimiser step per retain batch and cycles the forget
loader, so the two datasets receive very different forget dosing:

| quantity, per unique forget image per epoch | CIFAR-10 | faces-1000 | ratio |
|---|---|---|---|
| raw presentations | 44096/5000 = **8.8192** | 12120/40 = **303** | **~34.4x** |
| avg unit-weight, mean-reduced forget-CE coefficient | 352/5000 = **0.0704** | 303/40 = **7.575** | **~107.6x** |

The second divides forget-bearing optimiser steps by the number of unique
forget images: each step contributes one mean-reduced forget CE term of unit
weight, so replaying a small set more often does not multiply that term.
**Neither ratio is observed gradient magnitude** -- both are counts derived
from the loop structure; actual gradients depend on the loss surface.

**Prospective dual-dose approximation (design calculation only, not run).**
Nine full-pool face forget batches per epoch would approximately match raw
presentations (9 against CIFAR's 8.8192) but, unweighted, would leave
coefficient exposure at 9/40 = 0.225, about **3.20x** the CIFAR target of
0.0704. Matching both scalar dose summaries would require a forget-loss
weight of

    lambda = (352/5000) / (9/40) = 0.3128888889

This is a **prospective dual-dose approximation, not a CIFAR-equivalent
experiment**: optimiser cadence, data geometry and actual gradients would all
still differ. Not implemented and not run here.

### What is NOT claimed

- **No CE comparison and no head comparison.** This phase ran ArcFace only.
- **No representation-erasure claim.** Displacement demonstrates movement, not
  loss of information.
- **No significance claim** of any kind, and no distributional assumption.
- **No cross-dataset movement-magnitude claim.** The two exposure ratios above
  forbid it, and they disagree with each other.
- No claim that ArcFace is "more localised" -- no selectivity measure
  supported that direction consistently, and the earlier wording to that
  effect is withdrawn.

### Controls are descriptive and non-exchangeable

Control classes are measured the same WAY as the forget class -- dropped from
the anchors, scored under a rotation fitted without them -- but a control
rotation excludes **two** classes (forget + control) while the forget rotation
excludes **one**. Measured anchor counts: CIFAR controls 20,000 vs forget
22,500; faces controls 19,365-19,369 vs forget 19,385. The two quantities are
therefore **not exchangeable, and the direction of that asymmetry is
unknown** -- it must not be described as inflating control displacement, as
making the comparison conservative, or as a quantified amount of design bias
explained. On faces only 32 of 999 retain classes were computed, so no
statement about all retain classes is available either.

### Artifact metadata defects — recorded, not rewritten

The four `logs/feature_movement_paired/` artifacts, and the four earlier
`logs/feature_movement/` artifacts, carry two metadata defects:

1. **every trajectory row records `method: "random_label_full"`**, including
   the frozen-backbone runs, because the label was hardcoded rather than
   taken from the mode;
2. **`run_classification` was inferred from the dataset name**, which
   mislabelled the faces full-model run as `new_experiment`. It is an
   **instrumented replication** of the face pilot already recorded at commit
   `c150b4c`.

Both are fixed at commit `1f3a271` ("fix: correct movement diagnostic
semantics"), which also splits the field into `update_mode` and an explicit,
never-inferred `scientific_role`, and asserts row/result agreement.

**The artifacts were not rewritten.** Their numerical measurements -- every
NC3 value, displacement, output, probe and exposure count quoted above -- were
re-read directly from the files and are valid. Read the mode from the
directory name, not from the `method` field, for runs produced before
`1f3a271`.

### Also on record

The CIFAR full-model run in this phase agrees with the earlier
`logs/feature_movement/cifar10_arcface_seed0_random_label_full_fc0` artifact
to four decimals on `nc3_centred_forget`, `aligned_forget` and
`output_forget` at every epoch, on a different GPU. That is **replication
evidence consistent with an unchanged computation**, not a proof of numerical
identity; no bitwise comparison was performed.

### Limitations

- One forget class, one seed, one method, 3 epochs, ArcFace only.
- faces `probe_forget` is 0.1-granular (10 test images per identity).
- faces controls are a 32-of-999 sample.
- `split_mode=all`, so nothing here speaks to held-out generalisation.
- The faces full-model member is confounded by its retain-utility loss.

---

## 2026-09-15 — faces-1000 dual-dose control: a dose/forgetting/utility tradeoff, not cheaper forgetting

**Decided:** Record the dual-dose faces-1000 intervention as a demonstrated
tradeoff. It does **not** supersede the uncontrolled faces pair recorded above;
it sits beside it as a second, lower-dose point on the same baseline.

### The conclusion, stated exactly

At the fixed three-epoch budget, the dual-dose faces-1000 intervention
preserved retain utility and sharply reduced feature movement, but the
full-model member did not reach complete output forgetting: **one of ten
forget-class test images remained correct at epoch 3.** The result therefore
demonstrates a **dose-forgetting-utility tradeoff**. It does **not** establish
equally effective forgetting with less utility damage, and there is **no
matched-outcome full-versus-frozen NC3 comparison**, because no epoch exists at
which both members show zero output forgetting.

### Provenance

- Commit `0f5bcc2` ("feat: add controlled random-label dose schedule"), clean
  tree at run time (`git_dirty: false` in both artifacts).
- Baseline run dir `logs/faces_arcface_seed0`, checkpoint
  `logs/faces_arcface_seed0/ckpt.pt`, sha256
  `48c19bfd2ff84d17eda447dfd73417894859b5c6a71936a8888ff8b75fea5d80`
  (recomputed from disk during the 2026-09-15 audit).
- Artifacts:
  - `logs/feature_movement_dose/faces_arcface_seed0_random_label_dualdose_full_fc0`
    (`scientific_role: new_experiment`, `update_mode: full_model`)
  - `logs/feature_movement_dose/faces_arcface_seed0_random_label_dualdose_clfonly_fc0`
    (`scientific_role: paired_control`, `update_mode: classifier_only`)
- Seed 0, forget class 0, 3 epochs, ArcFace, 32-of-999 controls, one physical
  A100 (`CUDA_VISIBLE_DEVICES=1`), members run sequentially.

### Dose schedule and accounting — predicted == observed

`S = 303` retain steps per epoch, `m = 9` active forget steps, selected by
`floor((step+1)*m/S) > floor(step*m/S)` at indices

    [33, 67, 100, 134, 168, 201, 235, 269, 302]

with forget-loss weight `lambda = 0.3128888888888889`.

| quantity | per epoch | over 3 epochs |
|---|---|---|
| candidate forget presentations | **12,120** | 36,360 |
| active forget presentations | **360** | 1,080 |
| active forget-bearing steps | 9 | 27 |
| active presentations per unique forget image | **9** | — |
| total weighted forget-loss coefficient mass | **2.816** | 8.448 |
| avg weighted coefficient per unique forget image | **0.0704** | — |

Predicted and observed agree exactly on all four totals in both members.

**Neither the presentation counts nor the weighted coefficient is a gradient
magnitude.** Both are counts and loss coefficients derived from the loop
structure; actual gradients depend on the loss surface and were not measured.

### Trace gates — all pass

- Candidate `training_trace_sha256` equal within the pair.
- Candidate trace **exactly equals the prior uncontrolled faces pair**:
  `f9c5f9fdc3c6fcfdfe88a37b708d77168c0e4271f74802e38c2b780e2f707806`.
  The dose changed what the loop did with the stream, not the stream.
- `active_dose_trace_sha256` equal within the pair
  (`203df28a2ac48a4264c12ea6a005ed4f7bf12a417b9c2a5c211d3a7a76026ff8`), and
  distinct from the candidate hash.
- Baseline checkpoint sha and observed train_eval index sha equal within the
  pair.
- Epoch-0 rows identical across the pair.

### Epoch tables

**Dual-dose full model** — epoch-0 baselines: centred **+0.8875**, uncentred
**-0.8846**

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | fgt-ctrl | cka |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.7000 | 0.7272 | 0.7000 | +0.8875 | -0.8846 | 0.000 | 0.000 | +0.000 | 1.0000 |
| 1 | 0.5000 | 0.7215 | 0.7000 | +0.6609 | -0.9022 | 1.414 | 1.269 | +0.145 | 0.9922 |
| 2 | 0.2000 | 0.7205 | 0.7000 | +0.6243 | -0.9104 | **2.091** | 1.596 | +0.495 | 0.9879 |
| 3 | 0.1000 | 0.7266 | 0.7000 | +0.6472 | -0.9160 | 1.865 | 1.430 | +0.436 | 0.9904 |

**Dual-dose frozen backbone (paired control)**

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.7000 | 0.7272 | 0.7000 | +0.8875 | -0.8846 | 0.000 | 0.000 | 1.0000 |
| 1 | 0.3000 | 0.7247 | 0.7000 | +0.9144 | -0.9081 | 0.000 | 0.000 | 1.0000 |
| 2 | **0.0000** | 0.7248 | 0.7000 | +0.9242 | -0.9154 | 0.000 | 0.000 | 1.0000 |
| 3 | 0.0000 | 0.7263 | 0.7000 | +0.9307 | -0.9199 | 0.000 | 0.000 | 1.0000 |

What the tables say:

- **Full output forget: 0.7 -> 0.5 -> 0.2 -> 0.1.** It never reaches zero.
  At 10 test images per identity, 0.1 is one image still classified correctly.
- **Full output retain: 0.7272 -> 0.7215 -> 0.7205 -> 0.7266.** Retain utility
  is essentially held at baseline (-0.06pp at epoch 3).
- **Full aligned forget movement peaks at 2.091 deg** (epoch 2), not at the end.
- **Full centred NC3 stays positive** at every epoch (+0.6243 minimum).
- **Frozen reaches output forget zero at epoch 2** and holds it.
- **Frozen retain accuracy remains approximately baseline** (0.7247-0.7263
  against 0.7272).
- **Frozen centred NC3 remains positive** (+0.9144 to +0.9307) and **features do
  not move**: raw paired displacement is exactly 0.000 and CKA exactly 1.0000 at
  every epoch.

**No sign reversal occurs in any row of either member, in either convention.**
Both uncentred baselines sit at -0.8846, exactly the case the CLAUDE.md guard
covers; every uncentred value is reported as a within-head change from that
epoch-0 value and never as a CE-versus-ArcFace comparison.

### Dual-dose versus uncontrolled, at the same epoch-3 budget

| | out_f | out_r | aligned forget | nc3_c |
|---|---|---|---|---|
| uncontrolled full | 0.0000 | 0.6396 | 11.484 deg | +0.2473 |
| dual-dose full | 0.1000 | **0.7266** | **1.865 deg** | **+0.6472** |

The two runs share the same baseline checkpoint and the same candidate sample
stream (identical `training_trace_sha256`), so the comparison is controlled in
those respects. **But the intervention jointly changes two things** -- the
active-step cadence (303 -> 9 forget-bearing steps per epoch) and the forget
loss weight (1.0 -> 0.3128888888888889). **It cannot identify which component
caused the change.** Read it as one lower-dose point against one higher-dose
point, not as an attribution.

### Proposed follow-up ablations — design calculations only, NOT implemented, NOT run

Two ablations would separate the two components. Neither has been implemented
or run, and neither is authorised here.

1. **`m = 9, lambda = 1`.** Same active cadence and same raw active
   presentations as the dual-dose run (9 per unique forget image per epoch),
   without loss down-weighting. Weighted coefficient mass would be 9.0 per
   epoch, 0.225 per unique forget image. **This isolates the effect of lambda
   at the nine-step cadence.**

2. **`m = 303, lambda = 0.009293729372937293`.** Every step remains active
   (303 active presentations per unique forget image per epoch) while total
   weighted coefficient exposure equals the dual-dose run's **0.0704** per
   unique image (mass 2.816 per epoch). **Comparing this with the dual-dose run
   tests cadence and forward exposure at matched coefficient mass.**

These are **mechanistic ablations, not CIFAR-equivalent experiments**:
optimiser cadence, data geometry and actual gradients still differ from CIFAR,
and no cross-dataset movement-magnitude claim follows from either.

**No automatic epoch extension is proposed.** Running the dual-dose condition
longer would change cumulative dose *and* the number of retain updates at the
same time, so a longer run would not be the same intervention observed for
longer and would not produce a matched-outcome comparison either.

### Limitations

- One dataset, one identity, one seed, one head, one dose, and a fixed
  three-epoch budget.
- `output_forget` and `probe_forget` rest on **ten** forget-class test images,
  so both are 0.1-granular; `probe_forget` never moved off 0.7000 in any row of
  either member.
- Controls are a **32-of-999 deterministic sample**, descriptive and
  **non-exchangeable** with the forget measurement (a control rotation excludes
  two classes, the forget rotation one; the direction of that asymmetry is not
  established). No significance claim, and no claim about all retain classes.
- `split_mode=all`, so nothing here speaks to held-out generalisation.
- **Actual gradient magnitude is unmeasured.** Every dose quantity above is a
  count or a coefficient.
- **Incomplete full-model forgetting prevents a matched-outcome conclusion**,
  including any statement about whether freezing alters the centred-NC3 result
  under this dose.

**Supersedes:** nothing. The uncontrolled faces pair and its
retain-utility-loss finding stand as recorded.

---

## 2026-09-15 — faces-1000 dose-component matrix: coefficient mass alone does not determine the outcome

**Decided:** Record the four-cell dose-component matrix (O/D/A/B) as a
mechanism diagnostic and **stop the dose grid here**. The two ablations
proposed on 2026-09-15 (`m=9, lambda=1` and
`m=303, lambda=0.009293729372937293`) have now been run.

### Conditions

| cell | m (active steps/epoch) | lambda |
|---|---|---|
| **O** uncontrolled | 303 | 1 |
| **D** dual-dose | 9 | 0.3128888888888889 |
| **A** low-cadence / unit-weight | 9 | 1 |
| **B** all-steps / coefficient-matched | 303 | 0.009293729372937293 |

### Provenance

All eight runs share baseline `logs/faces_arcface_seed0`, checkpoint sha256
`48c19bfd2ff84d17eda447dfd73417894859b5c6a71936a8888ff8b75fea5d80`, observed
train_eval index sha256 `1e80ea55e15b5c0a46bff56de8577f767cc261d3b65b73ceb50732baf8027f6c`,
ArcFace, seed 0, forget class 0, 3 epochs, 32-of-999 deterministic controls,
`git_dirty: false`. Every candidate `training_trace_sha256` equals
`f9c5f9fdc3c6fcfdfe88a37b708d77168c0e4271f74802e38c2b780e2f707806` — the dose
changed what the loop did with the stream, never the stream.

| cell | active_dose_trace_sha256 | full member | frozen member | commit | phys GPU |
|---|---|---|---|---|---|
| O | (none — pre-dose artifact) | `feature_movement_paired/..._full_fc0` | `..._clfonly_fc0` | `0bf8f05` | 3 |
| D | `203df28a2ac48a4264c12ea6a005ed4f7bf12a417b9c2a5c211d3a7a76026ff8` | `feature_movement_dose/..._dualdose_full_fc0` | `..._dualdose_clfonly_fc0` | `0f5bcc2` | 1 |
| A | `6cb9499aa1b6dee13ce2727d1fc2dc493a0be94a3f49869d038ac02d63d09b94` | `feature_movement_ablation/..._m9_w1_full_fc0` | `..._m9_w1_clfonly_fc0` | `87c7a0e` | 0 |
| B | `ea53dbabf1ced2f4880d99b3836724537fc4d2aef1f956d0831b946e402974cf` | `feature_movement_ablation/..._m303_coeffmatched_full_fc0_gpu0rep` | `..._m303_coeffmatched_clfonly_fc0` | `87c7a0e` | 0 |

Dose hashes are equal within each pair and distinct across cells, as their
schedules and weights differ.

**Authoritative B pair is same-device (both physical GPU 0).** The first
B-full (`..._m303_coeffmatched_full_fc0`, physical GPU 2) is retained **only as
cross-device replication evidence**: it was re-run on GPU 0 as
`scientific_role: instrumented_replication`, and the two agree **bitwise** —
maximum absolute difference 0.000e+00 across `output_retain`,
`output_overall`, `probe_retain`, `nc1_angular`, centred and uncentred NC3
(forget and retain-mean), `aligned_forget`, `aligned_retain_eval`,
`control_mean_deg` and `cka_linear_secondary` at every epoch, with identical
discrete outputs, first-zero epoch, sign flags and hashes. The weighted-mass
residual `8.448000000000079` against a predicted `8.448` is a deterministic
float accumulation identical in all three B artifacts and well inside the
driver's 1e-9 relative tolerance.

### Dose accounting per epoch — predicted == observed

| cell | candidate | active | active/image | active steps | weighted mass | coeff/image |
|---|---|---|---|---|---|---|
| O | 12,120 | 12,120 | 303 | 303 | 303* | 7.575* |
| D | 12,120 | 360 | 9 | 9 | 2.816 | 0.0704 |
| A | 12,120 | 360 | 9 | 9 | **9** | **0.225** |
| B | 12,120 | 12,120 | 303 | 303 | **2.816** | **0.0704** |

*O has no dose-accounting block; its figures are **our arithmetic** from its
uncontrolled schedule (every step active, weight 1), not read from the
artifact. D, A and B agree exactly with prediction on all four totals in both
members.

**Neither the presentation counts nor the weighted coefficient is a gradient
magnitude.** Both are counts and loss coefficients derived from the loop
structure; actual gradients were not measured.

### New epoch tables

Epoch 0 is identical in all eight runs: out_f 0.7000, out_r 0.7272,
probe_f 0.7000, centred **+0.8875**, uncentred **-0.8846**, movement 0.000,
CKA 1.0000.

**A full model** (`m=9, lambda=1`)

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5000 | 0.7225 | 0.7000 | +0.6688 | -0.9027 | 1.730 | 1.282 | 0.9921 |
| 2 | 0.2000 | 0.7209 | 0.7000 | +0.6502 | -0.9112 | 2.396 | 1.611 | 0.9876 |
| 3 | 0.1000 | 0.7257 | 0.7000 | +0.6312 | -0.9174 | 2.317 | 1.475 | 0.9898 |

**A frozen backbone (paired control)**

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0000 | 0.7244 | 0.7000 | +0.9280 | -0.9173 | 0.000 | 0.000 | 1.0000 |
| 2 | 0.0000 | 0.7249 | 0.7000 | +0.9380 | -0.9241 | 0.000 | 0.000 | 1.0000 |
| 3 | 0.0000 | 0.7259 | 0.7000 | +0.9445 | -0.9282 | 0.000 | 0.000 | 1.0000 |

**B full model** (`m=303, lambda=0.009293729372937293`; GPU-0 replication)

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2000 | 0.6850 | 0.7000 | +0.2404 | -0.9289 | 6.121 | 3.498 | 0.9366 |
| 2 | 0.0000 | 0.6799 | 0.7000 | +0.2525 | -0.9366 | 6.797 | 3.775 | 0.9275 |
| 3 | 0.0000 | 0.6874 | 0.7000 | +0.2737 | -0.9404 | 6.221 | 3.508 | 0.9376 |

**B frozen backbone (paired control)**

| ep | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3000 | 0.7249 | 0.7000 | +0.9146 | -0.9083 | 0.000 | 0.000 | 1.0000 |
| 2 | 0.0000 | 0.7251 | 0.7000 | +0.9243 | -0.9155 | 0.000 | 0.000 | 1.0000 |
| 3 | 0.0000 | 0.7257 | 0.7000 | +0.9307 | -0.9200 | 0.000 | 0.000 | 1.0000 |

Frozen feature movement is **exactly 0.000** with CKA **exactly 1.0000** in
every row of both frozen runs.

### Fixed-epoch comparison — full members

| ep | cell | out_f | out_r | probe_f | nc3_c | nc3_u | aligned_fgt | ctrl | cka |
|---|---|---|---|---|---|---|---|---|---|
| 1 | O | 0.0000 | 0.6724 | 0.7000 | +0.2221 | -0.9519 | 9.493 | 4.950 | 0.8791 |
| 1 | D | 0.5000 | 0.7215 | 0.7000 | +0.6609 | -0.9022 | 1.414 | 1.269 | 0.9922 |
| 1 | A | 0.5000 | 0.7225 | 0.7000 | +0.6688 | -0.9027 | 1.730 | 1.282 | 0.9921 |
| 1 | B | 0.2000 | 0.6850 | 0.7000 | +0.2404 | -0.9289 | 6.121 | 3.498 | 0.9366 |
| 3 | O | 0.0000 | 0.6396 | 0.9000 | +0.2473 | -0.9701 | 11.484 | 6.745 | 0.7439 |
| 3 | D | 0.1000 | 0.7266 | 0.7000 | +0.6472 | -0.9160 | 1.865 | 1.430 | 0.9904 |
| 3 | A | 0.1000 | 0.7257 | 0.7000 | +0.6312 | -0.9174 | 2.317 | 1.475 | 0.9898 |
| 3 | B | 0.0000 | 0.6874 | 0.7000 | +0.2737 | -0.9404 | 6.221 | 3.508 | 0.9376 |

### Fixed-epoch comparison — frozen members

| ep | cell | out_f | out_r | nc3_c | nc3_u |
|---|---|---|---|---|---|
| 1 | O | 0.0000 | 0.7198 | +0.9789 | -0.9534 |
| 1 | D | 0.3000 | 0.7247 | +0.9144 | -0.9081 |
| 1 | A | 0.0000 | 0.7244 | +0.9280 | -0.9173 |
| 1 | B | 0.3000 | 0.7249 | +0.9146 | -0.9083 |
| 3 | O | 0.0000 | 0.7234 | +0.9474 | -0.9601 |
| 3 | D | 0.0000 | 0.7263 | +0.9307 | -0.9199 |
| 3 | A | 0.0000 | 0.7259 | +0.9445 | -0.9282 |
| 3 | B | 0.0000 | 0.7257 | +0.9307 | -0.9200 |

### First epoch at which BOTH members show zero output forgetting

| cell | full | frozen | first common |
|---|---|---|---|
| O | 1 | 1 | **1** |
| D | never | 2 | **none** |
| A | never | 1 | **none** |
| B | 2 | 2 | **2** |

### Conclusions, scoped

1. **A versus D — coefficient mass at the fixed nine-step cadence.** Raising
   weighted mass from 2.816 to 9 (lambda 0.3129 -> 1) produced the **same
   output_forget trajectory at the available 0.1 resolution** (0.5 / 0.2 / 0.1)
   and only modest differences in continuous metrics: epoch-3 centred NC3
   +0.6312 against +0.6472, movement 2.317 deg against 1.865 deg, retain 0.7257
   against 0.7266. State this as **no detectable output difference in this
   cell**, *not* as "lambda is inert" — the forget test set resolves only to
   0.1, and the continuous metrics did move.

2. **B versus O — lambda at the all-step cadence.** Reducing lambda delayed
   complete output forgetting from **epoch 1 to epoch 2**, reduced movement
   (epoch 3: 6.221 deg against 11.484 deg), reduced fixed-budget utility loss
   (retain 0.6874 against 0.6396) and left centred NC3 **broadly similar**
   (+0.2737 against +0.2473).

3. **B versus D — identical total coefficient mass, different distribution.**
   Both carry 2.816 weighted mass per epoch and 0.0704 per unique forget image.
   Spread across **303** active forwards rather than **nine**, the same mass
   produced stronger forgetting (epoch 3 out_f 0.0000 against 0.1000), greater
   movement (6.221 deg against 1.865 deg) and more utility degradation (retain
   0.6874 against 0.7266). **Therefore total weighted coefficient mass alone is
   insufficient to determine the outcome.** The contrast **bundles** temporal
   distribution, active forward exposure, BatchNorm exposure and per-step
   weighting; it is **not a pure cadence effect** and no component is isolated.

4. **B matched outcome — both members reach `output_forget = 0` at epoch 2.**
   Full centred NC3 **+0.2525**; frozen centred NC3 approximately **+0.9243**;
   full aligned forget movement **6.797 deg**; frozen movement **0 deg**; full
   retain **0.6799**; frozen retain approximately **0.7251**. The
   **4.52pp full-versus-frozen retain-accuracy gap is a utility confound.**
   The centred-NC3 gap therefore repeats at a matched output outcome, but
   **it cannot be attributed solely to freezing.**

5. **No face condition exhibits a centred or uncentred sign reversal**, in any
   row of any of the eight runs. Uncentred baselines are **already negative**
   (-0.8846 at epoch 0), exactly the case the CLAUDE.md guard covers; every
   uncentred value here is a within-head change from that baseline and never a
   CE-versus-ArcFace comparison.

6. **Stop the dose grid here.** These four cells are a **mechanism
   diagnostic**, not a parameter search and not a CIFAR-equivalent comparison.

No claim is made about gradient magnitude, representation erasure, statistical
significance, head superiority, class-count causality, or clean isolation of
cadence and BatchNorm effects.

### Limitations

- One identity, one seed, one head, one method, fixed 3-epoch budget.
- The forget test set is **ten images**, so `output_forget` and `probe_forget`
  are 0.1-granular; `probe_forget` never left 0.7000 in any A or B row.
- Controls are a **32-of-999 deterministic sample**, descriptive and
  **non-exchangeable** with the forget measurement. No significance claim.
- `split_mode=all`, so nothing here speaks to held-out generalisation.
- **O's dose values are derived arithmetic**, not read from an artifact.
- The **cross-device B run is retained only as replication evidence**; the
  authoritative B pair is the same-device GPU-0 pair.

**Supersedes:** nothing. The uncontrolled faces pair, the dual-dose tradeoff
entry and their findings all stand as recorded. This entry closes the two
follow-up ablations proposed in the dual-dose entry above.

---

## 2026-09-16 — controlled nested class-count sweep, K=100/250/500/1000

**Decided:** Record the nested class-count sweep as 24 valid unlearning cells
at K=100, 250 and 1000, plus four baselines at each of the four K. **K=500 is
excluded from the unlearning comparison by the pre-registered head-fairness
gate** and neither head was retuned to rescue it.

**Because:** the 100-identity result (2026-09-14) came from one class count.
If the CE-versus-ArcFace difference is a property of the loss it should not
depend on how many classes the head has; if it does depend on K, a single-K
claim is not safe to write. This sweep varies **only** `max_identities` and
`out_dir`: identity sets are nested, a shared identity keeps its class index,
and at seed 0 the selected images and the train/test split of a shared
identity are identical at every K.

### What ran

One variable moves per comparison. Every cell: `random_label`,
classifier-only (backbone frozen), `m=9` active forget steps per epoch,
`lambda=1`, 3 unlearning epochs, seed 0, per-cell reseeding,
`split_mode=all`, `n_forget=40`, forget classes {0, 29, 60, 95} =
identities {00001, 00142, 00284, 00524} — the same four people at every K,
confirmed from each cell's own identity manifest.

Wave 1 (K=100, K=1000) was completed by adding the four missing
K=1000 × ArcFace cells; Wave 2 added K=250 and K=500 baselines and, after the
gate, the eight K=250 cells.

### Baselines and the head-fairness gate

| K | CE test acc | ArcFace test acc | gap (pp) | gate (2.00pp) |
|---|---|---|---|---|
| 100 | 57.65 | 57.45 | 0.20 | pass |
| 250 | 67.47 | 68.58 | 1.10 | pass |
| 500 | 68.97 | 71.32 | **2.36** | **fail** |
| 1000 | 74.65 | 72.71 | 1.94 | pass |

All eight baselines: 40/40 epochs, final `lr 0.00000`, seed 0, valid
checkpoint. **K=500 failed because ArcFace exceeded CE by 2.36pp**, not
because ArcFace underperformed; the gate is symmetric and pre-registered, so
it fails either way and its unlearning cells were never run. Its baselines are
retained as artifacts.

### Validation — 24/24 cells

Recomputed from the artifacts, not from any prior summary: commit `51c38be`
with `git_dirty=False`; baseline checkpoint sha256 re-hashed from disk and
matched; identity/image/train/test manifests constant within each K;
`update_mode=classifier_only`; `per_cell_seed=True`; `m=9`, `lambda=1`;
predicted dose equals observed dose (27 active forget-bearing steps, 1080
active presentations, weighted mass 27, re-derived independently of the file);
exactly four trajectory rows at epochs 0-3; centred **and** uncentred NC3
finite in every row; `n_forget=40`. **No canonical value was read from a
`unlearn_pass1_nan_uncentred` directory** — those are preserved untouched as
non-authoritative diagnostic history of the nan defect fixed in `51c38be`.

### First epoch with `output_forget = 0`

| K | fc | CE | ArcFace | first common |
|---|---|---|---|---|
| 100 | 0 | 1 | 2 | 2 |
| 100 | 29 | 2 | 3 | 3 |
| 100 | 60 | 1 | 1 | 1 |
| 100 | 95 | 1 | never | **none** |
| 250 | 0 | 1 | 2 | 2 |
| 250 | 29 | 1 | 2 | 2 |
| 250 | 60 | 1 | 3 | 3 |
| 250 | 95 | 1 | 2 | 2 |
| 1000 | 0 | 1 | 1 | 1 |
| 1000 | 29 | 1 | 1 | 1 |
| 1000 | 60 | 1 | 2 | 2 |
| 1000 | 95 | 2 | 1 | 2 |

CE reaches zero at epoch 1 in 11 of 12 cells. ArcFace is slower in 7 of 12 and
faster in 1 (K=1000 fc95). **Eleven of twelve pairs have a matched epoch.**
K=100 fc95 has none — ArcFace never reaches 0 inside the 3-epoch budget — and
is recorded as **"no matched-outcome comparison"**; the budget was not extended
and no other epoch was substituted.

### Matched-outcome geometry — raw endpoint levels

| K | fc | ep | head | out_f | out_r | probe_f | nc3 centred | nc3 uncentred |
|---|---|---|---|---|---|---|---|---|
| 100 | 0 | 2 | ce | 0.0000 | 0.5711 | 0.7000 | +0.7478 | +0.1706 |
| 100 | 0 | 2 | arcface | 0.0000 | 0.5711 | 0.6000 | +0.9785 | -0.9228 |
| 100 | 29 | 3 | ce | 0.0000 | 0.5629 | 0.8000 | +0.7296 | +0.1739 |
| 100 | 29 | 3 | arcface | 0.0000 | 0.5711 | 0.8000 | +0.9430 | -0.9320 |
| 100 | 60 | 1 | ce | 0.0000 | 0.5742 | 0.3000 | +0.7566 | +0.2535 |
| 100 | 60 | 1 | arcface | 0.0000 | 0.5711 | 0.6000 | +0.6713 | -0.9232 |
| 250 | 0 | 2 | ce | 0.0000 | 0.6742 | 0.7000 | +0.6893 | +0.2969 |
| 250 | 0 | 2 | arcface | 0.0000 | 0.6828 | 0.8000 | +0.9680 | -0.9265 |
| 250 | 29 | 2 | ce | 0.0000 | 0.6726 | 0.9000 | +0.7421 | +0.3432 |
| 250 | 29 | 2 | arcface | 0.0000 | 0.6882 | 0.9000 | +0.9845 | -0.9285 |
| 250 | 60 | 3 | ce | 0.0000 | 0.6684 | 0.5000 | +0.6858 | +0.2699 |
| 250 | 60 | 3 | arcface | 0.0000 | 0.6873 | 0.5000 | +0.9792 | -0.9326 |
| 250 | 95 | 2 | ce | 0.0000 | 0.6746 | 0.7000 | +0.6725 | +0.2835 |
| 250 | 95 | 2 | arcface | 0.0000 | 0.6865 | 0.9000 | +0.8333 | -0.9303 |
| 1000 | 0 | 1 | ce | 0.0000 | 0.7477 | 0.7000 | +0.6605 | +0.3300 |
| 1000 | 0 | 1 | arcface | 0.0000 | 0.7244 | 0.7000 | +0.9280 | -0.9173 |
| 1000 | 29 | 1 | ce | 0.0000 | 0.7461 | 0.9000 | +0.6774 | +0.3303 |
| 1000 | 29 | 1 | arcface | 0.0000 | 0.7260 | 0.8000 | +0.8856 | -0.9183 |
| 1000 | 60 | 2 | ce | 0.0000 | 0.7452 | 0.5000 | +0.6297 | +0.2758 |
| 1000 | 60 | 2 | arcface | 0.0000 | 0.7244 | 0.7000 | +0.9927 | -0.9255 |
| 1000 | 95 | 2 | ce | 0.0000 | 0.7471 | 1.0000 | +0.5316 | +0.2591 |
| 1000 | 95 | 2 | arcface | 0.0000 | 0.7236 | 1.0000 | +0.9887 | -0.9258 |

**These raw levels must not be read as an unlearning-induced head effect.**
The two heads start at different geometry — centred NC3 near +0.70 to +0.99
for ArcFace against +0.70 to +0.87 for CE, and uncentred on **opposite sides
of zero** (CE about +0.56 to +0.63, ArcFace about -0.88). The level difference
at the matched epoch is mostly the starting difference.

### Matched-outcome within-head change and difference-in-changes

`d` is the matched-epoch value minus **that head's own epoch-0 value**.
`dArcFace - dCE` is **our arithmetic**, not a metric from the AISTATS paper.

| K | fc | ep | dCE (centred) | dArcFace (centred) | dArcFace - dCE | dCE (uncentred) | dArcFace (uncentred) |
|---|---|---|---|---|---|---|---|
| 100 | 0 | 2 | -0.1223 | -0.0006 | +0.1217 | -0.3475 | -0.0277 |
| 100 | 29 | 3 | -0.1320 | +0.0062 | +0.1382 | -0.3637 | -0.0428 |
| 100 | 60 | 1 | -0.0855 | -0.0670 | +0.0185 | -0.2755 | -0.0265 |
| 250 | 0 | 2 | -0.1382 | +0.0153 | +0.1535 | -0.2903 | -0.0410 |
| 250 | 29 | 2 | -0.1139 | -0.0011 | +0.1128 | -0.2848 | -0.0433 |
| 250 | 60 | 3 | -0.1374 | +0.0058 | +0.1432 | -0.3126 | -0.0422 |
| 250 | 95 | 2 | -0.1287 | -0.0053 | +0.1234 | -0.3141 | -0.0415 |
| 1000 | 0 | 1 | -0.1282 | +0.0405 | +0.1687 | -0.2607 | -0.0328 |
| 1000 | 29 | 1 | -0.1507 | -0.0407 | +0.1100 | -0.2803 | -0.0364 |
| 1000 | 60 | 2 | -0.1514 | +0.0055 | +0.1568 | -0.3057 | -0.0408 |
| 1000 | 95 | 2 | -0.1690 | -0.0006 | +0.1685 | -0.3050 | -0.0405 |

`dArcFace - dCE` for centred NC3 is **positive in all eleven matched cells**.
CE's classifier-to-class-mean cosine falls by 0.086 to 0.169 while ArcFace's
moves by -0.067 to +0.041.

### Fixed-epoch difference-in-changes — controls for the varying matched epoch

The matched epoch differs by cell (1, 2 or 3), and `dCE` grows with epoch, so
the matched-outcome column above mixes two things. At a **fixed** epoch, with
all four classes at every K:

| convention | ep | K=100 | K=250 | K=1000 |
|---|---|---|---|---|
| centred, mean dCE | 1 | -0.0948 | -0.1053 | -0.1378 |
| centred, mean dArcFace | 1 | -0.0374 | +0.0041 | +0.0015 |
| centred, mean (dArc - dCE) | 1 | **+0.0574** | **+0.1093** | **+0.1393** |
| centred, mean (dArc - dCE) | 3 | **+0.0607** | **+0.1449** | **+0.1799** |
| uncentred, mean (dArc - dCE) | 1 | +0.2599 | +0.2202 | +0.2338 |
| uncentred, mean (dArc - dCE) | 3 | +0.3284 | +0.2773 | +0.2910 |

**The centred difference-in-changes increases monotonically with K at both
fixed epochs. The uncentred one does not** (K=100 is the largest, K=250 the
smallest, at both epochs). The two conventions therefore disagree about
whether anything varies with K, which is itself a reason not to state a K
trend as a property of the mechanism.

Per-cell spread at epoch 1, centred: K=100 spans +0.0185 to +0.0971, K=250
+0.0909 to +0.1342, K=1000 +0.1100 to +0.1687. Adjacent K **overlap**, and
K=100's low mean rests heavily on fc60 (+0.0185) and fc95 (+0.0205).

### Retain utility as a within-head change

Reference is each head's **own** pre-unlearning retain accuracy (that cell's
epoch-0 `output_retain`). Raw CE-versus-ArcFace retain accuracy is **not** the
cost of the intervention — the heads differ in retain accuracy before any
unlearning.

| K | worst CE change (pp) | worst ArcFace change (pp) |
|---|---|---|
| 100 | -1.13 | -0.41 |
| 250 | -0.70 | -0.25 |
| 1000 | -0.15 | -0.32 |

Every matched-epoch within-head retain change is within **1.13pp** of that
head's own baseline, in both directions, at every K. No cell forgot by
breaking the model. End-of-budget `probe_retain` and `ncc_retain` are likewise
flat within each head.

### Sign changes

**No centred or uncentred sign reversal occurs in any of the 24 cells**, at
any epoch. ArcFace's uncentred NC3 is **already negative at epoch 0**
(about -0.88 to -0.90) — exactly the case the CLAUDE.md guard covers — so its
negative post-unlearning value is **not** a flip. Every uncentred figure here
is a within-head change from that cell's own baseline and is never used as a
CE-versus-ArcFace level comparison.

### Conclusions, scoped — five distinct things, not one

1. **Sign reversal: none, in either head, at either convention, in all 24
   cells.** The `w_k^un = -(1-gamma) mu_k` flip the AISTATS theory predicts
   does not appear under this method and budget at any K, for CE either.
2. **Magnitude of within-head centred NC3 movement: CE moves, ArcFace barely
   does.** CE -0.086 to -0.169 at the matched epoch; ArcFace -0.067 to +0.041.
   Difference-in-changes positive in 11/11 matched cells. In the **uncentred**
   convention both heads move in the same direction and CE still moves more
   (about -0.26 to -0.36 against -0.027 to -0.043).
3. **Raw endpoint geometry: not interpretable as a head effect.** ArcFace ends
   higher on centred NC3 and far lower on uncentred, but it started that way.
   Use (2), not the endpoint levels.
4. **Output forgetting: both heads succeed, CE sooner.** CE reaches
   `output_forget=0` by epoch 1 in 11/12 cells; ArcFace is slower in 7/12 and
   has one cell (K=100 fc95) that never reaches 0 in three epochs.
5. **Retain utility: essentially unchanged in both heads,** within 1.13pp of
   each head's own baseline everywhere.

On whether the CE-versus-ArcFace difference **changes systematically with K**:
the centred difference-in-changes orders monotonically K=100 < K=250 < K=1000
at both fixed epochs, driven mainly by CE's own movement growing with K, while
ArcFace stays near zero at K=250 and K=1000. **This is descriptive only.**
Adjacent K overlap per-cell, the uncentred convention shows **no** such
ordering, K=100's mean depends heavily on two of its four cells, and K=500 —
which would have sat between K=250 and K=1000 — is **absent**, so the ordering
rests on three points.

### Explicitly not claimed

- **No statistical significance claim.** One seed, four forget classes per K,
  three K values with cells. Nothing here is a test.
- **No representation-erasure claim.** The backbone is frozen in every cell,
  so features cannot move by construction; `probe_forget` is a 10-image
  measurement at 0.1 granularity and is not evidence about representations.
- **No class-count causality claim.** K co-varies with retain-set size, number
  of retain steps per epoch, spacing of the nine active steps within an epoch,
  BatchNorm exposure and the difficulty of the classification problem itself.
- **Fixed forget dose is not fixed total optimization.** `m=9`, `lambda=1` and
  27 active forget-bearing steps are identical at every K, but retain steps per
  epoch are 31 (K=100), 77 (K=250) and 303 (K=1000), so the number of retain
  updates between consecutive forget steps, and the spacing of those steps,
  **still vary with K**.
- **Candidate forget exposure also varies with K**: 1240, 3080 and 12,120
  candidate presentations per epoch at K=100, 250 and 1000. Only the *active*
  count is held at 360.
- **Not combined with the historical faces-100 dataset** in
  `configs/faces100_*.yaml`: different source directory, different
  `min_images`, unrecorded extraction seed. Those runs are a separate lineage.
- **Four K values and one seed are limited evidence**, and only three of the
  four carry unlearning cells.

### Provenance and preservation

All 24 cells at commit `51c38be`, clean tree, every run through `RunDir`.
K=1000 cells reuse the `logs/faces_{ce,arcface}_seed0` baselines (configs
verified identical to `configs/facesK1000_*.yaml` on data, head, train,
backbone and seed); K=100/250 use their own in-tree baselines. K=1000 ArcFace
cells ran on the same physical GPU as all other Wave-1 cells; each K in Wave 2
ran its CE and ArcFace halves on one GPU, so **no head comparison carries a
device difference**. The `unlearn_pass1_nan_uncentred` trees at K=100 and
K=1000 are preserved unmodified and were not read for any value here.

**Supersedes:** nothing. The 2026-09-14 100-identity entry stands; this
extends it across class count and adds the within-head-change framing that the
raw cross-head levels do not support. The open question of which epoch
convention to report `nc3_*_forget` at remains open — this entry reports
matched-outcome, epoch-1 and epoch-3 side by side rather than choosing.

---

## 2026-09-16 — centred-NC3 reference-frame decomposition: the K trend is not the retain centre, and it is not more weight movement either

**Decided:** The centred class-count trend is **not** an artefact of the
retain-weight centre moving. Holding that centre fixed at epoch 0 reproduces
the ordering essentially unchanged, and moving only the centre reproduces
nothing. But the trend is **also not** more forget-weight movement at larger
K: CE's forget weight rotates by the same amount at every K. The ordering is a
property of the **baseline angle** the centred frame places that rotation at,
which varies with K through the **feature-side** centre. This **refines** the
sweep's conclusion; it does not overturn it.

**Because:** the 2026-09-16 sweep left `dArcFace - dCE` ordering K=100 < 250 <
1000 in the centred convention and not in the uncentred one. Centred NC3
subtracts a reference frame that can itself move, so a change in the metric is
not by itself a statement about the forget-class weight. That had to be
resolved before spending a seed on replication.

### What was measured, and how it was obtained

No canonical cell ever saved head state: each holds `config.json`, `env.json`,
`result.json`, `results.jsonl`, `run.log`, `trajectory.jsonl` and nothing else,
and the only six `ckpt.pt` files under `logs_classcount/` are *baselines*. The
audit therefore concluded the decomposition was impossible from existing
artifacts, and all 24 cells were **deterministically replayed** with opt-in
instrumentation into `logs_classcount_decomposition/`. **Canonical directories
were read only, never written.**

The instrumentation (commit `653d55f`) is `evaluate_light(state_hook=...)`,
which hands the hook the arrays that function **already extracted**. It does
not re-extract: a fresh DataLoader iterator draws a base seed from the global
CPU RNG, which would reshuffle every later unlearning epoch. That inertness is
asserted bitwise in `src/test_decomposition_state.py` (parameters, global RNG
state and training-trace hash identical with and without the hook).

### Definitions — taken from production, not reinvented

`train.evaluate_light` passes `head.weight` to `metrics.nc3_alignment` **raw**
(`fc.weight` for CE, `W` for ArcFace, neither row-normalised); the cosine
normalises internally. `src/decomposition.py` imports `_l2` and `class_means`
from `metrics` rather than retyping them, including the `1e-12` denominator
guard, so it cannot drift from the metric it decomposes.

    w_t = W_t[fc]                          effective forget weight, raw row
    c_t = W_t[ref].mean(0)                 retain-weight centre, ref = present & != fc
    f_0 = mu[fc]                           forget-class feature mean
    g_0 = nanmean(mu[ref])                 feature-side centre
    A(w, c) = cos(f_0 - g_0, w - c)        production centred NC3, forget row
    U(w)    = cos(f_0, w)                  production uncentred NC3, forget row

`f_0` and `g_0` are constant across epochs because the backbone is frozen.
That was **verified, not assumed**: the class-mean matrix is bit-identical at
epochs 0-3 in all 24 cells (sha256 of the float64 bytes), and the `present` and
centring masks are unchanged.

The two counterfactuals are **descriptive interventions on the metric**. `A`
normalises both arguments, so it is not additive; the residual
`A(w_t,c_t) - A(w_t,c_0) - A(w_0,c_t) + A(w_0,c_0)` is the exact arithmetic
leftover of a nonlinear function at four corners, **not** a causal or
statistical interaction. `test_residual_is_nonzero_when_both_move` exists
specifically so the corners can never be read as contributions.

### Replay gates — 24/24, exact

Representative cell first (K=1000, ArcFace, fc0, GPU 1): per-epoch training
losses identical (21.0504 / 21.2566 / 21.2445), `training_trace_sha256`
`4ce51788…` and `active_dose_trace_sha256` `a6960f3e…` identical, all 24
trajectory values at epochs 0-3 identical **bit for bit**. Only then were the
remaining cells launched.

All 24 cells then passed, with no exceptions: epochs 0-3 present; `output_forget`,
`output_retain`, `probe_forget`, `nc3_centred_forget`, `nc3_uncentred_forget`
and `nc1_angular` exactly equal to canonical at every epoch; both trace hashes
exactly equal; identity/image/train/test/baseline-checkpoint hashes equal;
`m=9`, `lambda=1`, `update_mode=classifier_only`, `per_cell_seed=True`; centred
and uncentred NC3 **recomputed from the saved weight and class-mean matrices**
exactly equal to the trajectory values; epoch-0 corners collapsing to one
baseline with zero change and zero residual; the residual identity closing to
floating-point tolerance; every quantity finite.

The replayed per-head means reproduce this entry's predecessor exactly — at
epoch 1, centred `dCE` = -0.0948 / -0.1053 / -0.1378 and `dArcFace` =
-0.0374 / +0.0041 / +0.0015 at K = 100 / 250 / 1000.

### The decomposition — `dArcFace - dCE`, our arithmetic

| convention | ep | K=100 | K=250 | K=1000 |
|---|---|---|---|---|
| centred, production `A(w_t,c_t)` | 1 | +0.0574 | +0.1093 | +0.1393 |
| centred, **centre fixed at epoch 0** `A(w_t,c_0)` | 1 | +0.0561 | +0.1087 | +0.1393 |
| centred, **weight fixed at epoch 0** `A(w_0,c_t)` | 1 | +0.0004 | +0.0002 | +0.0003 |
| interaction residual | 1 | +0.0010 | +0.0004 | -0.0003 |
| uncentred, production | 1 | +0.2599 | +0.2202 | +0.2338 |
| centred, production | 3 | +0.0607 | +0.1449 | +0.1799 |
| centred, **centre fixed at epoch 0** | 3 | +0.0587 | +0.1435 | +0.1810 |
| centred, **weight fixed at epoch 0** | 3 | +0.0005 | +0.0006 | +0.0010 |
| interaction residual | 3 | +0.0014 | +0.0007 | -0.0022 |
| uncentred, production | 3 | +0.3284 | +0.2773 | +0.2910 |

Across all 24 cells and 4 epochs: largest `|dCentreOnly|` = **0.0065**, largest
`|residual|` = **0.0068**, against a largest `|dProd|` of **0.2007**.

### Movement of each moving part — mean over the four identities

| K | head | ep | forget-weight rotation | forget-weight norm ratio | retain-centre rotation | centred-weight rotation |
|---|---|---|---|---|---|---|
| 100 | ce | 1 | 17.82° | 0.874 | 7.34° | 17.81° |
| 250 | ce | 1 | 16.74° | 0.851 | 6.73° | 16.74° |
| 1000 | ce | 1 | 17.41° | 0.851 | 14.55° | 17.41° |
| 100 | arcface | 1 | 3.28° | 1.001 | 0.005° | 5.30° |
| 250 | arcface | 1 | 4.74° | 1.001 | 0.021° | 2.83° |
| 1000 | arcface | 1 | 4.56° | 0.999 | 0.021° | 5.55° |
| 100 | ce | 3 | 22.60° | 0.857 | 9.11° | 22.60° |
| 250 | ce | 3 | 20.79° | 0.830 | 8.14° | 20.79° |
| 1000 | ce | 3 | 21.49° | 0.830 | 17.77° | 21.49° |
| 100 | arcface | 3 | 5.22° | 1.001 | 0.024° | 9.82° |
| 250 | arcface | 3 | 6.23° | 1.000 | 0.035° | 4.10° |
| 1000 | arcface | 3 | 6.15° | 0.996 | 0.068° | 8.16° |

### Why the centre contributes nothing, for two opposite reasons

| K | head | mean `\|c_0\|` / `\|w_0\|` |
|---|---|---|
| 100 | ce | 0.0248 |
| 250 | ce | 0.0104 |
| 1000 | ce | **0.0012** |
| 100 | arcface | 0.9598 |
| 250 | arcface | 1.0074 |
| 1000 | arcface | 0.9692 |

**CE's retain-weight centre is nearly the zero vector** — 999 weights that
almost cancel — and gets 20x smaller from K=100 to K=1000. Its large *angular*
movement (up to 17.8°) is the rotation of a near-zero-length vector and moves
the metric by ~1e-4. This is why CE's centred-weight rotation equals its
forget-weight rotation to three decimals in every row above.

**ArcFace's centre is nearly as long as the weight itself** (ratio ~0.96-1.01)
— the shared-component offset behind its baseline uncentred NC3 of about -0.88
(CLAUDE.md's empirical guard). Centring therefore matters a great deal to
ArcFace's *level*, but the centre barely *moves* (0.02-0.07°), so it
contributes nothing to the *change*.

### Where the K ordering actually lives

CE's forget weight rotates by **the same amount at every K** — 17.8° / 16.7° /
17.4° at epoch 1, 22.6° / 20.8° / 21.5° at epoch 3, non-monotone and spanning
about 1.8°. The centred `dCE` nevertheless grows monotonically. Reading both
conventions in the angle domain (mean over the four identities, epoch 1):

| K | head | baseline θ centred | Δθ centred | baseline θ uncentred | Δθ uncentred |
|---|---|---|---|---|---|
| 100 | ce | 31.93° | +9.14° | 57.97° | +17.77° |
| 250 | ce | 34.15° | +9.60° | 53.21° | +16.69° |
| 1000 | ce | 39.05° | +11.32° | 54.07° | +17.35° |

In the **uncentred** frame essentially the whole rotation is directed away from
`f_0` (Δθ ≈ the full 16.7-17.8°), and the baseline angle orders
58.0° > 54.1° > 53.2° — **non-monotone**, K=100 largest and K=250 smallest.
Since a cosine's sensitivity to rotation is `-sin θ`, the uncentred changes
inherit exactly that non-monotone order: -0.2841 / -0.2552 / -0.2681.

In the **centred** frame only about half the rotation is directed away from the
reference, but both the baseline angle (31.9° < 34.2° < 39.1°) and the
projected rotation (9.14° < 9.60° < 11.32°) increase with K, and they push the
same way.

The centred frame's reference direction is `f_0 - g_0`, and `|g_0|/|f_0|` for
CE is 0.80 / 0.73 / 0.69 at K = 100 / 250 / 1000. **So the K ordering enters
through the FEATURE-side centre `g_0`, not through the retain-weight centre
`c`.** `g_0` is fixed in time within a cell — verified bit-identical across
epochs — so it is a fixed reference frame that happens to differ with K, not
something unlearning moved.

### Answers

1. **Does the K ordering remain with the centre held at epoch 0?** Yes, nearly
   unchanged: +0.0561 / +0.1087 / +0.1393 against a production
   +0.0574 / +0.1093 / +0.1393 at epoch 1, and
   +0.0587 / +0.1435 / +0.1810 against +0.0607 / +0.1449 / +0.1799 at epoch 3.
2. **Does changing only the centre reproduce the ordering?** No. It yields
   +0.0004 / +0.0002 / +0.0003 and +0.0005 / +0.0006 / +0.0010 — flat, and two
   to three orders of magnitude below production.
3. **Is ArcFace's stability forget-weight stability, centre behaviour, or
   interaction?** The metric decomposition indicates **forget-weight
   stability**. ArcFace's weight rotates 3.3-6.2° against CE's 16.7-22.6° and
   keeps its norm (ratio 0.996-1.001 against CE's 0.830-0.874); its centre is
   effectively frozen; the residual never exceeds 0.0068. The one caveat: with
   ArcFace's own changes as small as they are, the residual reaches ~11% of the
   production change in a single cell (K=1000 fc29, epoch 3), so "negligible in
   absolute terms" is the defensible statement, not "negligible relative to
   ArcFace".
4. **Why no K ordering in the uncentred convention?** Both conventions read the
   same, essentially K-independent, weight rotation, but at different baseline
   angles where the cosine has different sensitivity. The uncentred baseline
   angle is non-monotone in K and the uncentred changes track it; the centred
   baseline angle and projected rotation are both monotone in K and reinforce.
5. **Does this weaken, preserve or refine the sweep's conclusion?** It
   **refines** it, and tightens two of its five conclusions. Conclusion 2
   (CE moves, ArcFace barely does) is **strengthened**: it survives with the
   reference frame nailed to epoch 0, and is now attributable to the forget
   weight itself rather than to the metric's frame. The **K-trend** statement is
   **narrowed**: it is not reference-frame drift, but neither is it "the forget
   weight moves more at larger K", which the movement table rules out. It is
   sensitivity of centred NC3 to a fixed reference frame that varies with K —
   a statement about the measurement geometry, **not** evidence that class count
   changes forget-class erasure.

### Explicitly not claimed

- **No causal isolation.** The counterfactuals are interventions on the metric,
  not on the training process; `A` is nonlinear in both arguments.
- **No statistical significance.** One seed, four fixed identities per K, three
  K values. The means above are descriptive summaries, not replicates.
- **No representation-erasure claim.** The backbone is frozen in every cell and
  the class means are bit-identical across epochs — features cannot move here
  by construction. This says nothing about representations.
- **No generality beyond this sweep.** Within these fixed models and identities
  only.
- **The ordering still rests on three points and still overlaps per cell.**
  Matched per-identity `dArcFace - dCE` under the fixed epoch-0 centre spans
  +0.0172 to +0.0956 (K=100), +0.0902 to +0.1335 (K=250) and +0.1124 to +0.1687
  (K=1000) at epoch 1; at epoch 3 K=100 includes a **negative** cell (fc95,
  -0.0226). Adjacent K overlap, exactly as in the production column. K=500
  remains absent.
- **Norm changes do not enter the metric.** Cosine is scale-invariant, so the
  forget-weight norm shrinkage (CE 0.83-0.87) is reported as a descriptive
  fact about the weight, not as a driver of `nc3_*_forget`.

### Provenance

Instrumentation at `653d55f`, table renderer at `bcbcd92`, both pushed before
any replay; every replay ran on a clean tree with `git_dirty=False`. Replays
through `RunDir` into `logs_classcount_decomposition/` (gitignored) — K=1000 CE
on GPU 0, K=1000 ArcFace on GPU 1, both K=250 halves on GPU 2, both K=100
halves on GPU 3, all four A100-SXM4-40GB, disjoint run directories, parents
pre-created so no two jobs raced on `mkdir`, and no other user's process
touched. Artifacts: `logs_classcount_decomposition/decomposition.json` and
`decomposition.md`, plus `nc3_state_ep{0,1,2,3}.npz` per cell. The gate cell is
preserved separately under `logs_classcount_decomposition/gate/`.
`make test` passes 11/11 suites, including the 12 new groups in
`src/test_decomposition_state.py`.

**Supersedes:** nothing. The 2026-09-16 sweep entry stands in full; this entry
qualifies its K-trend paragraph and strengthens its conclusion 2.

---

## 2026-09-17 — K=100 seed-1 replication: direction and magnitude replicate, cell-level detail does not

**Decided:** Record the K=100 seed-1 replication as eight valid unlearning
cells plus two baselines, gate passed. Conclusions, as corrected:

1. **A positive mean fixed-epoch K=100 DiC recurs across two seeds** — centred
   +0.0574/+0.0648/+0.0607 (seed 0) against +0.0595/+0.0708/+0.0686 (seed 1) at
   epochs 1/2/3; uncentred agrees to within 0.011 at every epoch.
2. **Identity-level contributions vary substantially** — fc29 contributes
   +0.1382 at seed 0 but +0.0173 at seed 1 (centred, epoch 3), while fc95 moves
   −0.0214 → +0.1276. The means agree far more closely than the identities.
3. **Seed 1 fc29 has a negative centred DiC under its own-attainment
   (matched-outcome) comparison, −0.007274, despite positive fixed-epoch
   contrasts at all three epochs.** The two contrasts disagree in sign for this
   cell; a summary must state which it means.
4. **Four of the seven attained pairs have unequal exposure** (seed 0 fc0,
   fc29; seed 1 fc0, fc29 — ArcFace +1 epoch each); three are equal. Seed 0
   fc95 has no attained pair at all.
5. **Direct actual-run backbone state equality remains unavailable** — the
   cells saved no model state, so it cannot be established from any surviving
   artifact.
6. **The seed-0 reference-frame decomposition remains closed** — this entry
   opens nothing there and the 2026-09-16 decomposition entry stands unchanged.

Seed 0's one non-attaining cell does not recur.

Lightweight tracked evidence: `evidence/k100_seed1/`.

> **Audited and corrected 2026-09-17.** An evidence audit of this entry and of
> the report under the output root corrected four things, all recorded below:
> the matched-outcome table read both objectives at a single common epoch
> instead of each at its own; seed 0's negative DiC was described as "two
> negative cells" when it is one identity observed at two epochs; constant
> `nc1_angular` was described as proof the feature mapping did not move, which
> a scalar summary cannot establish; and an unsupported claim about image
> selection at larger K was removed. No number below was produced by re-running
> anything — all are recomputed from the existing trajectories.

**Because:** the 2026-09-16 sweep rests on one seed. If the CE-versus-ArcFace
difference is a property of the loss it should survive a change of seed.

Full report and machine-readable tables:
`runs/robustness/k100_seed1/20260917T151714Z_056b522/` (untracked, like every
artifact tree here). Commit `056b522`, clean tree, GPU 1, all ten runs on one
physical device so no head comparison carries a device difference.

### Seed contract at K=100 — a nuance worth pinning

This is an end-to-end new-seed replication, **not** initialization-only. But at
K=100 the identity roster **and** the image manifest are byte-identical across
seeds 0 and 1 (`identity_manifest_sha256 f1b43b10…`, `image_manifest_sha256
3176469c…`): identity selection is deterministic, and no identity in the
first-100 roster exceeds the `max_images_per_identity=50` cap, so the sampling
RNG is a no-op at this K. Precisely, the K=100 seed change gives:

- **same identity roster** (same 100 identities, same order, same class
  indices — {0,29,60,95} = {00001,00142,00284,00524} at both seeds);
- **same selected image pool** (the same 4906 images, identical
  `image_manifest_sha256`);
- **changed train/test assignments** (`bb00eccc…` → `90627643…`, `130db252…` →
  `aab38157…`), **changed initialization, changed shuffling**.

Whether the selected image pool likewise stays fixed across seeds at
K = 250/500/1000 is **not audited and is not asserted either way** — it would
need seed-1 manifests at those K, which do not exist. (An earlier version of
this entry asserted identities above the cap "almost certainly exist" there;
that had no manifest evidence behind it and is withdrawn.)

### Baselines and the gate

| seed | CE | ArcFace | gap (pp) | gate (2.00pp) |
|---|---|---|---|---|
| 0 | 57.6531 (565/980) | 57.4490 (563/980) | 0.2041 | pass |
| 1 | 55.9184 (548/980) | 57.1429 (560/980) | 1.2245 | pass |

Metric is `output_overall` over all 980 test images; checkpoint rule is final
epoch, no selection. Neither head was retuned and no training extended.

**Aggregate parity is not per-identity parity.** Baseline forget accuracy at
label 95 moves from 6–7/10 at seed 0 to **10/10 in both heads** at seed 1. No
post hoc eligibility gate was added on that basis.

### Fixed-epoch DiC = dArcFace − dCE

Sign convention established from raw values: recomputing seed 0 under this
formula reproduces the 2026-09-16 table exactly. "CE–ArcFace difference" in the
supplied summary is a loose label, not a sign inversion.

| convention | ep | seed 0 mean (range, sign) | seed 1 mean (range, sign) |
|---|---|---|---|
| centred | 1 | +0.0574 ([+0.019,+0.097], 4/4) | +0.0595 ([+0.027,+0.097], 4/4) |
| centred | 2 | +0.0648 ([−0.002,+0.127], 3/4) | +0.0708 ([+0.025,+0.123], 4/4) |
| centred | 3 | +0.0607 ([−0.021,+0.138], 3/4) | +0.0686 ([+0.017,+0.128], 4/4) |
| uncentred | 1 | +0.2599 (4/4) | +0.2533 (4/4) |
| uncentred | 2 | +0.3166 (4/4) | +0.3075 (4/4) |
| uncentred | 3 | +0.3284 (4/4) | +0.3174 (4/4) |

Centred means agree to ~0.008; uncentred to ~0.011.

**Sign accounting, stated precisely.** Seed 1 is positive in 4/4 identities at
every epoch and both conventions. Seed 0's only negative centred DiC is at
**fc95 (identity 00524), at epochs 2 and 3 — one unlearning cell observed at
two epochs, not two distinct cells.** Counting it as two overstates the
evidence, and an earlier version of this entry did. Separately: **a negative
DiC is not an NC3 sign reversal.** Across all 16 K=100 cells at both seeds
there are **zero** sign reversals — no same-class, same-convention sign change
from a cell's own baseline.

Per-identity values do **not** track between seeds — fc29's centred DiC at ep3
is +0.1382 (seed 0) against +0.0173 (seed 1); fc95 −0.0214 against +0.1276. The
four-identity means agree far more closely than the identities behind them.

### Matched outcome — each objective at ITS OWN first 0/10 epoch

The rule is per objective. An earlier version of this entry reported a single
"matched epoch" per identity, `max(e_CE, e_ArcFace)`, and read **both** heads
there — which gave CE exposure past its own attainment point and inflated its
Δ. Corrected: each objective is read at its own first-zero epoch, and the
differing exposures are stated.

| fc | identity | CE 1st 0/10 | ArcFace 1st 0/10 | exposure |
|---|---|---|---|---|
| 0 | 00001 | s0 1 / s1 1 | s0 2 / s1 2 | ArcFace +1 at both seeds |
| 29 | 00142 | s0 2 / s1 1 | s0 3 / s1 2 | ArcFace +1 at both seeds |
| 60 | 00284 | s0 1 / s1 1 | s0 1 / s1 1 | equal at both seeds |
| 95 | 00524 | s0 1 / s1 1 | s0 **NONE** / s1 1 | s0: **no pair**; s1 equal |

**Seed 0, fc95, ArcFace never reaches 0/10 in three epochs** (1/10 at epoch 3);
**no attained-pair contrast is computed for it** and no epoch was substituted.
That non-attainment does not recur — at seed 1 ArcFace fc95 attains at epoch 1.

**Matched-outcome DiC** — `ΔArcFace` at ArcFace's own first-zero epoch minus
`ΔCE` at CE's own first-zero epoch, from full-precision trajectories. Because
**four of the seven** attained pairs sit at different epochs, this confounds
head with exposure; the fixed-epoch table above is the head comparison. It is
reported because the two contrasts do **not** always agree in sign.

| seed | fc | CE ep | Arc ep | DiC centred | DiC uncentred |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 2 | +0.093026 | +0.258486 |
| 0 | 29 | 2 | 3 | +0.128843 | +0.302625 |
| 0 | 60 | 1 | 1 | +0.018519 | +0.248986 |
| 0 | 95 | 1 | **none** | **not computed** | **not computed** |
| 1 | 0 | 1 | 2 | +0.064305 | +0.248214 |
| 1 | 29 | 1 | 2 | **−0.007274** | +0.244957 |
| 1 | 60 | 1 | 1 | +0.040584 | +0.243585 |
| 1 | 95 | 1 | 1 | +0.096737 | +0.259220 |

**Correction: seed 1 fc29 (identity 00142) has a NEGATIVE centred
matched-outcome DiC, −0.007274**, while its fixed-epoch centred DiC is positive
at all three epochs (+0.0271, +0.0246, +0.0173). An earlier version of this
entry, written under the superseded common-epoch reading, implied all seed-1
contrasts were positive. That holds for **fixed-epoch** contrasts only. This
negative DiC is **not** an NC3 sign reversal — there are still zero of those.

Sign counts, kept separate by contrast type:

| contrast | conv | seed | n | pos | neg | zero |
|---|---|---|---|---|---|---|
| matched-outcome | centred | 0 | 3 | 3 | 0 | 0 |
| matched-outcome | centred | 1 | 4 | 3 | **1** (fc29) | 0 |
| matched-outcome | uncentred | 0 | 3 | 3 | 0 | 0 |
| matched-outcome | uncentred | 1 | 4 | 4 | 0 | 0 |
| fixed-epoch | centred | 0 | 12 | 10 | **2** (fc95 ×2 epochs) | 0 |
| fixed-epoch | centred | 1 | 12 | 12 | 0 | 0 |
| fixed-epoch | uncentred | 0 | 12 | 12 | 0 | 0 |
| fixed-epoch | uncentred | 1 | 12 | 12 | 0 | 0 |

**Cross-seed means of matched contrasts compare different identity sets** (seed
0 has 3 attained, seed 1 has 4): centred +0.080129 (s0, n=3) vs +0.048588 (s1,
n=4). Restricted to the **common attained set {0, 29, 60}**: +0.080129 vs
+0.032539. That set is **selected on attainment in both seeds** — it excludes
exactly the identity whose ArcFace cell failed at seed 0 — so it is not an
unbiased subset and is not a like-for-like seed contrast.

No cell was already at zero at baseline (seed-1 baselines range 4/10 to 10/10).
Matching is on **observed output accuracy only** — not exposure, not utility,
not representation state.

**Observed attainment, narrowly.** CE attains at epoch 1 in 4/4 identities at
seed 1 and 3/4 at seed 0; ArcFace attains in 4/4 at seed 1 (epochs 2,2,1,1) and
3/4 at seed 0 (epochs 2,3,1, one non-attainment). No identity attains later at
seed 1 than at seed 0 in either objective. With four identities on an integer
epoch grid of {1,2,3} this describes these cells; it is not a rate.

### Also holding at seed 1

- **No sign reversal in any of the 8 cells, either convention.** Centred NC3
  stays in [+0.633, +0.972]; ArcFace uncentred is already ≈ −0.89 at epoch 0,
  the case the guard covers.
- **Retain utility** within 0.82pp of each head's own epoch-0 baseline (CE
  −0.82 to +0.10pp, ArcFace −0.31 to +0.31pp). No cell forgot by breaking.

### Frozen backbone — what is actually established, in three tiers

1. **Code path (applies to these runs).** `src/unlearn.py: _params` under
   `classifier_only=True` sets `requires_grad_(False)` on every backbone
   parameter and calls `backbone.eval()`; the epoch loop guards the re-enable
   with `if not classifier_only`, so it never returns to train mode, and the
   optimizer is built over head parameters only. BatchNorm in `eval()` does not
   update `running_mean`/`running_var`/`num_batches_tracked`. This reasoning is
   state-independent, so it carries to a trained checkpoint.
2. **Test suite.** `src/test_dose_schedule.py:511` compares the frozen run's
   backbone buffers against `fresh()[0]`'s. That is meaningful because `run()`
   itself starts from the same deterministic `fresh(seed)`, so it is really
   "buffers after == buffers at the run's own start". **But** the harness model
   is a toy `Linear + BatchNorm1d(5)` on CPU, not ResNet-18, and its starting
   buffers are construction defaults (0, 1, 0), **not** a 40-epoch-trained
   checkpoint's converged statistics. A fresh model is not equivalent to a
   trained starting checkpoint and must not be cited as if it were.
3. **Direct evidence from these eight runs: not available.** The cells saved no
   model state — only `config/env/result/results/run.log/trajectory` files. The
   only `.pt` files are the two *baseline* (starting) checkpoints, which do
   carry 60 parameter tensors and 60 BN buffers each, but there is no saved
   post-run counterpart. **Exact per-cell state equality of backbone parameters
   and BN buffers before versus after unlearning cannot now be established from
   the available artifacts,** and nothing was rerun to obtain it.

`nc1_angular` is bitwise identical across all four epochs of every cell (CE
0.9740622302714209, ArcFace 0.8210257728777127). That is **consistent with** an
unchanged feature mapping but is **not proof**: a scalar summary of the feature
matrix is many-to-one, so it can hold constant under representation changes. An
earlier version of this entry called it confirmation that the mapping "never
moved"; that overstated it.

**For future runs:** persisting the post-unlearning backbone `state_dict`, or
just a hash of its parameters and buffers, would make tier 3 available at
negligible cost. That is a driver change, not a protocol change.

### Compute

Three quantities of different provenance, not to be merged:

1. **Logged method + evaluation duration, 8 cells: 436.9 s = 0.1214 GPU-h** —
   directly measured in-process (`method_s + eval_s` per `result.json`).
2. **Coarsely logged baseline duration, 2 baselines: ≈90.1 s ≈ 0.0250 GPU-h** —
   41 s + 41 s of per-epoch times printed at **1 s resolution** over 40 epochs
   each (±40 s quantization possible), plus 3.5 s / 4.5 s eval.
   Sum of (1)+(2): **≈527 s ≈ 0.146 GPU-h**, inheriting (2)'s coarseness.
3. **Approximate end-to-end single-device execution interval: ≈626 s ≈ 10.4 min
   ≈ 0.174 h** — **file-mtime-derived and approximate** (`environment.txt`
   11:17:25 to last cell `run.log` 11:27:51).

**(3) is not GPU-active time.** It is the wall-clock span of the sequential
session and includes startup, CPU dataset scans, model construction, checkpoint
I/O and inter-run gaps; the ≈99 s by which it exceeds (1)+(2) is inferred from
the difference, not instrumented. The GPU was also not exclusively held —
another user's job ran on GPU 0 throughout — so even (1) is elapsed in-process
time on a shared machine. An earlier completion report said "well under 0.1
GPU-hours"; that was wrong and is withdrawn.

### Artifact persistence

Everything under the output root stays **git-ignored** (`runs/`, plus `*.pt`):
full report, run logs, per-cell `result.json` dose records, and both baseline
checkpoints. Private agent instructions (`CLAUDE.md`), internal planning notes
and checkpoints remain ignored and were **not** restored.

**Lightweight scientific evidence is now tracked** at `evidence/k100_seed1/` —
a dedicated directory outside the ignored `runs/` tree, so no ignore rule was
weakened. It holds: a README (metric definitions, matched-outcome selection
rule, limitations, relation to the ignored raw tree); `provenance.json`
(execution revisions, identity/image/train/test manifest hashes for both seeds,
baseline checkpoint SHA-256s, source-artifact paths, and a SHA-256 link for
every preserved file); **verbatim, unaltered** copies of all 16 per-cell
`trajectory.jsonl` files for both seeds, byte-identity verified; four corrected
result tables; and the two resolved configs with the dataset's absolute path
redacted. Excluded by design: checkpoints, raw images, verbose environment
dumps, personal machine paths, credentials, internal management documents.

**Preserving tables does not preserve rerun capability.** These files let the
reported numbers be *checked*; reproducing the runs needs the dataset and
pipeline, and reproducing the exact values bit-for-bit additionally needs the
baseline checkpoints, which are not stored there.

**Checkpoint backup remains unresolved.** The seed-1 baseline checkpoints
(~45 MB each) and the seed-0 checkpoints exist only on the local filesystem of
the machine that produced them. **No durable, mirrored or off-machine copy has
been verified to exist.** Their SHA-256s are recorded in
`evidence/k100_seed1/provenance.json` so a future copy can be integrity-checked
— a hash is not a backup.

### Explicitly not claimed

- **Two seeds are not population-level robustness.** No significance test.
- **The eight cells are four identities × two objectives at one seed**, not
  eight independent seed replications: within a seed they share one backbone per
  head, one split, one baseline checkpoint.
- **Positive DiC is not superior unlearning.** It says CE's
  classifier-to-class-mean cosine moves further than ArcFace's under a matched
  forget dose. Nothing more.
- **Three different things are being measured and must stay separate**: output
  forgetting (attainment of 0/10), metric behaviour of NC3 under a frozen
  classifier-only update, and representation erasure. Only the first two are
  observed here. **Zero output accuracy is not erasure**, and nothing in these
  cells measures whether the identity remains recoverable from the features.
- **Integer counts**: forget accuracy is over 10 test images (0.1 granularity).
- Per-identity `s1−s0` contrasts mix seed effect with shifted per-identity
  baseline difficulty; same person, different split.
- **Attainment epochs are not comparable exposures** — four of the seven
  attained pairs reach 0/10 at different epochs.

**Supersedes:** nothing. The 2026-09-16 sweep entry stands; this adds a second
seed at K=100 only.

---

## 2026-09-17 — Paper figures implemented from recorded evidence (CPU only)

**Decided:** Render the three main figures and one supplementary figure from
already-recorded artifacts, with tracked plotting inputs, provenance sidecars
and an independent verification pass. Scripts in `scripts/figures/`, output in
`figures/`.

- **Figure 1** `fig1_k100_trajectories` — K=100 output forgetting and NC3
  trajectories, both seeds, both objectives, all four identities (C1, C3).
- **Figure 2** `fig2_seed0_decomposition` — seed-0 rotation, norm ratio and the
  reference-frame decomposition of centred DiC (C6a, C6b).
- **Figure 3** `fig3_contrast_rules` — fixed-epoch versus own-attainment centred
  DiC (C5, C7).
- **Figure S1** `figS1_reversal_by_stratum` — stratified reversal counts,
  supplementary and contextual only (C2).

**Because:** the claims were settled and the evidence was already recorded; no
GPU work, training, inference, replay, new seed, new K or new metric was needed
or authorized, and none was run.

**Scientific corrections applied in the same pass** (to `notes/paper_claims.md`
and `notes/figure_specs.md`):

- **The weight norm can affect centred geometry.** A cosine is invariant to
  scaling its whole argument, so the *uncentred* cosine is norm-invariant — but
  the centred cosine takes `w − c`, and changing `w` against a fixed `c` changes
  that direction. The previous blanket "weight norm does not enter the value"
  was true only of the uncentred convention. Norm ratios are now stated as
  descriptive, with no attribution claimed in either direction; no
  counterfactual on the norm was run.
- **Exact finite-angle accounting replaces the linearisation.** C6b now uses
  `Δcos = cos(θ₀ + Δθ) − cos(θ₀)` computed per cell and aggregated afterwards,
  never the cosine of averaged angles. `−sin θ₀` is retained only as local
  sensitivity intuition; at these 9–18° rotations it understates the exact
  change by 8–11 %. An exact per-cell transplant table was added.
- **Epoch-1 centre-only DiC range corrected to 0.000200–0.000413.** The earlier
  "0.0002 to 0.0010" imported epoch 3's maximum (+0.001007). Epoch-1,
  later-epoch and single-cell-extremum scopes are now stated separately.
- **Neither NC3, nor a baseline-subtracted change in it, nor DiC measures
  unlearning quality.** Stated as a reading convention and repeated in the
  captions; baseline subtraction does not upgrade a geometric quantity.
- **Statistical-impossibility wording removed**, replaced by "no inferential
  claim is supported by the present analysis".
- **BatchNorm phases separated everywhere.** Baseline training updates running
  statistics; classifier-only unlearning runs the backbone in `eval()` and
  accrues no BatchNorm exposure at any K. The C6c list and the Figure 2
  "must not imply" block now say so.
- **Uncentred-change-only plotting is a presentation choice**, not a prohibition
  on descriptive baseline-level comparison (which C9 reports).

**Preservation:** `logs_classcount_decomposition/decomposition.json` is
git-ignored, so `scripts/figures/extract_fig2_inputs.py` copies the 72 cell-epoch
rows Figure 2 consumes into the tracked
`evidence/decomposition_fig2/fig2_plotting_inputs.csv`, with the source SHA-256
and extraction provenance. `make_fig2.py --from-artifact` asserts the tracked
table is bit-equal to the artifact for every consumed field. That assertion
caught a real bug during implementation: cell names repeat across K, so an
artifact lookup keyed on `(cell_name, epoch)` silently collapsed 72 rows to 24.
K is now part of the key.

**Verification:** `scripts/figures/verify_figures.py` runs 66 read-only checks
and all pass — hashes, denominators, first-0/10 epochs, the single missing
attainment, axis coverage in all four figures, zero centred sign reversals
across the 16 K=100 cells, the residual identity for all 72 cell-epochs, Panel C
bar heights, the corrected centre-only range, the exact finite-angle identity,
the K=500 gate record and direction, the fc29 sign disagreement at full
precision, and the stratified S1 counts.

**Not done:** no checkpoint backup (still unresolved); no new experiment; no
recomputation of the decomposition from `.npz` state; no pooled rate across
strata anywhere.

**Note on older entries.** The 2026-09-16 entries describe the head-fairness
gate as "pre-registered". It is **prespecified** — fixed in advance in this
repository — but no registration record exists. `notes/paper_claims.md` and
`notes/figure_specs.md` use "prespecified"; those older entries stand as written
and are superseded on this point.

**Supersedes:** the figure specifications in `notes/figure_specs.md`, which were
specification-only and are now marked as implemented, with final captions.

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
