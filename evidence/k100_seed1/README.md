# K=100 seed-1 replication — preserved scientific evidence

Lightweight, tracked evidence for the K=100 seed-1 robustness replication of the
CE-versus-ArcFace classifier-only random-label unlearning comparison.

Execution revision **`056b522`** (seed-1 runs, clean tree). Seed-0 comparison
artifacts come from commit `51c38be` (the 2026-09-16 nested class-count sweep)
and were read only, never rewritten.

## What is here

| path | what |
|---|---|
| `provenance.json` | execution revisions, identity/split/image manifest hashes for both seeds, baseline checkpoint SHA-256s, source-artifact references, and a SHA-256 link for every preserved file |
| `trajectories/seed{0,1}/{ce,arcface}_fc{0,29,60,95}.jsonl` | **verbatim, unaltered** copies of the 16 per-cell trajectory files that every number below is derived from; byte-identical to their sources (verified) |
| `tables/trajectories_both_seeds.csv` | all 32 cell-epochs, both seeds, with integer counts alongside the raw floats |
| `tables/fixed_epoch_dic.csv` | ΔCE, ΔArcFace and DiC per identity at epochs 1–3, both conventions, both seeds |
| `tables/matched_outcome_dic.csv` | matched-outcome contrasts under the own-attainment rule, with exposure flags |
| `tables/baselines.csv` | baseline accuracies, counts, gate decision, checkpoint hashes |
| `configs/facesK100_{ce,arcface}.resolved.json` | fully resolved configurations as executed |

## Metric definitions

**Forget accuracy** — fraction of that identity's **10** test images predicted as
that identity. One image is 0.1; "zero" means 0/10.

**Retain accuracy** — micro-accuracy over the other **970** test images
(`(preds == labels)` on the non-forget mask). All retain counts in these tables
are exact integers.

**Baseline / fairness accuracy** — `output_overall`, top-1 over the **entire**
980-image test set. The head-fairness gate is |CE − ArcFace| ≤ 2.00pp.

**NC3 centred** — cosine between the forget class's classifier weight and its
class-mean feature, with weights centred by the global mean **excluding the
forget class** from the centring reference. This is the head-comparable
convention.

**NC3 uncentred** — the same cosine without centring. This is the convention the
AISTATS prediction is stated in. It is reported **only** as a within-head change
from that cell's own epoch-0 value, never as a CE-versus-ArcFace level
comparison: ArcFace's uncentred NC3 sits near −0.89 *before* any unlearning.

**DiC** (our arithmetic, not a metric from the AISTATS paper):

    DiC = (NC3_ArcFace,t − NC3_ArcFace,0) − (NC3_CE,t − NC3_CE,0)

## Two different contrasts — do not mix them

**Fixed-epoch DiC** (`tables/fixed_epoch_dic.csv`) reads both objectives at the
**same** epoch t. This is the quantity to use for the head comparison, because
both heads have had the same amount of optimization.

**Matched-outcome DiC** (`tables/matched_outcome_dic.csv`) reads **each objective
at its own first post-baseline epoch with 0/10 correct forget predictions**.
Selection is on **observed output accuracy only** — not on exposure, not on
utility, not on representation state.

Because the two objectives generally attain 0/10 at different epochs, a
matched-outcome DiC **confounds head with exposure**. Of the 7 attained pairs,
**4 have unequal exposure** (seed 0 fc0 and fc29; seed 1 fc0 and fc29 — ArcFace
+1 epoch in each) and **3 are equal** (seed 0 fc60; seed 1 fc60 and fc95).

The two contrasts disagree in sign for one cell: **seed 1, fc29 (identity
00142), centred — matched-outcome DiC is −0.007274, while its fixed-epoch DiC is
positive at all three epochs.** That is a real feature of the data, and any
summary must state which contrast it is using.

**Missing attainment:** seed 0, fc95 (identity 00524), ArcFace never reaches
0/10 within the three-epoch budget (1/10 at epoch 3). **No matched-outcome
contrast exists for it.** No epoch was substituted and the budget was not
extended. Consequently seed 0 has **3** attained identities and seed 1 has
**4** — any cross-seed mean of matched contrasts compares different identity
sets unless restricted to the common attained set **{0, 29, 60}**, which is
itself *selected on attainment in both seeds*.

## Known limitations

- **Two seeds are not population-level robustness.** No significance test is
  reported, and **no inferential claim is supported by the present analysis** —
  the identities within a seed are not replicates, and two seeds give no usable
  variance estimate for a seed effect.
- **The eight seed-1 cells are four identities × two objectives at one seed** —
  not eight independent replications. Within a seed they share one backbone per
  head, one train/test split, and one baseline checkpoint.
- **A negative DiC is not an NC3 sign reversal.** DiC is a difference of two
  within-head changes; a sign reversal is a same-class, same-convention sign
  change from a cell's own baseline. There are **zero** sign reversals across
  all 16 K=100 cells at both seeds.
- **Positive DiC is not superior unlearning.** It says CE's
  classifier-to-class-mean cosine moves further than ArcFace's under a matched
  forget dose. Nothing more.
- **Output forgetting, metric behaviour and representation erasure are three
  different things.** Only the first two are observed here. Zero output
  accuracy is not erasure, and the backbone is frozen, so these runs do not
  measure whether the identity remains recoverable from the features.
- **Direct actual-run backbone state equality is unavailable.** The eight cells
  saved no model state; only the *starting* baseline checkpoints exist. Exact
  per-cell equality of backbone parameters and BatchNorm buffers before versus
  after unlearning cannot be established from any surviving artifact. The
  frozen-backbone claim rests on the code path (`eval()` mode, gradients off,
  optimizer over head parameters only) and on a toy-model test, not on these
  runs. `nc1_angular` being bitwise constant across epochs is *consistent with*
  an unchanged feature mapping but is not proof — a scalar summary is
  many-to-one over feature configurations.
- **The K=100 seed change** holds the identity roster **and** the selected image
  pool fixed (identical `image_manifest_sha256`) and varies train/test
  assignment, initialization and shuffling. Whether the image pool is likewise
  fixed at larger K is **not audited and not asserted**.

## Relationship to the ignored raw run directory

The raw run tree lives at

    runs/robustness/k100_seed1/20260917T151714Z_056b522/

and is **git-ignored**, following this repository's convention for every
artifact tree (`logs/`, `logs100/`, `logs_classcount/`,
`logs_classcount_decomposition/`, `runs/`). It holds the full report, run logs,
per-cell `result.json` dose records, and the two baseline checkpoints.

The seed-0 comparison artifacts live under `logs_classcount/K100/`, also ignored.

`provenance.json` records the source path and SHA-256 of every file preserved
here, so each preserved trajectory can be matched back to the run directory it
came from.

**This directory does not preserve rerun capability.** It preserves the numbers
and the provenance needed to *check* them. Reproducing the runs would need the
dataset and the training pipeline; reproducing the exact reported values
bit-for-bit would additionally need the baseline checkpoints, which are **not**
stored here.

**Checkpoint backup is unresolved.** The two seed-1 baseline checkpoints
(~45 MB each) and the seed-0 checkpoints exist only on the local filesystem of
the machine that produced them. No durable, mirrored, or off-machine copy has
been verified to exist. Their SHA-256s are recorded in `provenance.json` so
that any future copy can be checked for integrity, but the hashes are not a
backup.
