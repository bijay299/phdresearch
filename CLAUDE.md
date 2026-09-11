# CLAUDE.md

Context for Claude Code working in this repository. Read this before making changes.

---

## What this project is

A research codebase for a CVPR 2027 submission. **Registration Nov 10 2026,
submission Nov 16 2026.** Roughly ten weeks from the start of the project.

**Research question:** Is the illusion of unlearning a property of unlearning,
or a property of cross-entropy?

Gao, Unal, Rangamani & Zhu (AISTATS 2026, *An Illusion of Unlearning?*) showed
that machine unlearning often moves the **classifier** while leaving the
**representation** intact. On CIFAR-10 the method SalUn scored 0.00% output
forget accuracy and 92.57% under a linear probe, against 77.35% for a retrained
reference. Their theory assumes neural collapse with fixed class means, and
predicts the forget-class weight is driven to

    w_k^un = -(1 - gamma) * mu_k

i.e. the classifier vector flips away from the class mean while the features
never move.

Margin-based losses (ArcFace, CosFace) L2-normalise both features and weights
and penalise angular distance between a sample and its class weight. **That is
precisely the drift the mechanism requires.** So we test whether the mechanism
survives when the loss forbids the shortcut.

Three possible outcomes, all publishable:
1. Unlearning under ArcFace must move the features — forgetting is deeper.
2. Unlearning under ArcFace fails — the shortcut is closed.
3. A third mechanism appears that nobody has characterised.

---

## The experimental design — do not break this

**Exactly one variable changes: the classification head.**

Backbone, dataset, augmentation, schedule, seed, epochs, batch size — all held
identical across conditions. `configs/base.yaml` holds the shared defaults and
each experiment config inherits from it, overriding only its head. That
structure exists to make it hard to accidentally vary two things at once.

If you are asked to change a training hyperparameter, change it in
`base.yaml` so it applies to every condition, or ask first. Changing it in one
config silently confounds the comparison.

**Primary outcome:** `nc3_*_forget` — cosine between the forget class's
classifier weight and its class-mean feature.

---

## Two measurement bugs already found — do not reintroduce

Both were caught by `src/test_metrics.py` on synthetic data with known answers.
The tests exist to keep them fixed.

### 1. Centring contamination

Standard NC3 centres classifier weights by their global mean. When unlearning
flips one class's weight, that flip moves the global mean and shifts the
centring for **every** class. Retained classes then appear misaligned when
nothing happened to them.

Measured: with only class 0 perturbed, retain-class alignment fell from 1.00 to
**0.94** — a six-point artefact that reads as real spillover damage.

**Always pass `exclude_from_centre=<forget_class>` to `nc3_alignment` when a
forget class exists.** `train.evaluate` already does this.

### 2. Centred and uncentred NC3 disagree

The theory's prediction concerns the **uncentred** weight. Under an exact flip:

| convention | forget cosine |
|---|---|
| uncentred | -1.00 |
| centred (corrected) | -0.71 |

Same model, same flip. Centring subtracts a global mean that is not part of the
theoretical claim, so it dilutes the effect.

**Report both. State the convention.** `train.evaluate` returns both; do not
drop one to simplify a table.

---

## Standing rules

**Run `make test` after touching `src/metrics.py` or `src/data.py`.** It runs
`src/test_metrics.py` (NC geometry, on synthetic data with known answers) and
`src/test_data.py` (smoke-mode splitting -- forget/retain/held-out must stay
inside whatever subset --smoke built) and fails if either does.

**Never report forget accuracy without utility.** A method that forgets
perfectly because the model broke has forgotten nothing. Every table row needs
both.

**Absolute probe accuracy is not evidence of failed unlearning.** A retrained
reference scored 77.35 on a class it never saw, because deep features transfer.
Only the gap against a reference means anything.

**Label derived quantities.** "Probe gap to retrain" is our arithmetic, not a
metric from the paper. Say so wherever it appears.

**Match test accuracy between heads before comparing geometry.** If ArcFace is
simply a worse model, any NC difference is confounded. This is the first thing
a reviewer will attack.

**Do not chase face-recognition SOTA.** Reviewers will not care about LFW
accuracy. They will care whether the conditions were matched.

**Held-out images must never enter training or unlearning.** `data.ForgetSplit`
keeps them separate. If they leak, the generalisation result is meaningless.

**Every run logs config, seed, git commit and device.** `utils.RunDir` handles
it. Do not add a code path that writes results without going through it.

---

## Layout

```
configs/base.yaml       shared defaults; others inherit and override
configs/cifar_*.yaml    pilot, one per head
configs/faces_*.yaml    the real experiment

src/heads.py            CE / ArcFace / CosFace, matched interfaces
src/backbones.py        ResNet feature extractors (features only, no logits)
src/data.py             datasets + forget/retain/held-out splits
src/unlearn.py          finetune, neggrad, neggrad_plus, random_label
src/train.py            training loop + evaluate() harness
src/metrics.py          NC1-3, linear probe, NCC, verification AUC
src/test_metrics.py     regression tests for metrics.py
src/test_data.py        regression tests for data.py (smoke-mode splitting)
src/utils.py            config, seeding, run directories

scripts/run_experiment.py   entry point
notes/decisions.md          decision log, dated
```

**Architectural invariant:** backbones return features, heads turn features
into logits. Nothing else may assume a particular loss. `metrics.py` must stay
loss-agnostic — it reads `head.weight`, which every head exposes.

---

## Commands

```bash
# wiring check, CPU, ~2 min
python scripts/run_experiment.py --config configs/cifar_ce.yaml --smoke

# the pilot pair
python scripts/run_experiment.py --config configs/cifar_ce.yaml
python scripts/run_experiment.py --config configs/cifar_arcface.yaml
python scripts/run_experiment.py --compare logs/

# override anything
python scripts/run_experiment.py --config configs/cifar_ce.yaml \
    --set train.epochs=5 unlearn.enabled=true
```

---

## Known issues to expect

**ArcFace may not converge at s=64 on CIFAR-10.** That scale is tuned for
thousands of identities. `configs/cifar_arcface.yaml` already uses s=30 with 5
warmup epochs. If it still diverges, try s=16, or a margin warmup (m=0 for the
first epochs then ramp). **This is a tuning problem, not a finding** — do not
report "ArcFace fails to train" as a result.

**Nothing here has been run on a GPU yet.** The code is syntax-checked and the
metrics are tested, but the training path is unexercised. Expect small fixes on
first real execution.

---

## What is NOT being claimed

- Not a new unlearning method.
- Not a claim that identity unlearning is unexplored — it is not.
- Novelty rests on searches of arXiv, general web, `awesome-llm-unlearning`,
  and IEEE Xplore. **Outstanding:** ACM DL, and a Semantic Scholar
  citation-graph pass on the AISTATS paper and on *Neural Collapse by Design*
  (arXiv:2605.20302). Do not write a novelty claim before those are done.

**Competitive risk:** the *Neural Collapse by Design* group works on the
geometry of normalised losses and cites the face-recognition margin family
explicitly. Connecting that to unlearning is one step from where they are.

---

## Working style for this repo

- Prefer small, testable changes. This is research code that produces numbers
  going into a paper; silent breakage is expensive.
- When adding a metric, add a test with a known answer first.
- When a result looks surprising, suspect the measurement before the finding.
  Both bugs above looked like real effects.
- Record decisions in `notes/decisions.md` with the date. Newer entries
  supersede older ones.
