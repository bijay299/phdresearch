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

## Open decisions

- [x] Dataset — **CASIA-WebFace**, resolved 2026-09-10. Kaggle RecordIO
      packaging, no Howard access needed.
- [ ] Is a negative result acceptable? (If ArcFace shows the same
      misalignment as CE, the paper becomes "the mechanism is general".)
      Get this agreed with Rawat now, not in week eight.
- [ ] Scope — one dataset done properly, or breadth?

---

## Template

## YYYY-MM-DD — <what was decided>
**Decided:** …
**Because:** …
**Supersedes:** …
