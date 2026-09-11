# Is the illusion of unlearning a property of unlearning, or of cross-entropy?

Testing whether **feature–classifier misalignment** survives under margin-based losses.

Target: CVPR 2027 — registration **Nov 10 2026**, submission **Nov 16 2026**.

---

## The claim being tested

Gao, Unal, Rangamani & Zhu (AISTATS 2026) showed that machine unlearning often
moves the *classifier* while leaving the *representation* intact. On CIFAR-10,
SalUn scored 0.00% output forget accuracy and 92.57% under a linear probe,
against 77.35% for a retrained reference.

Their explanation assumes **neural collapse** and **fixed class means**, and
predicts that NegGrad unlearning drives the forget-class weight to

    w_k^un  =  -(1 - gamma) * mu_k

i.e. the classifier vector flips away from the class mean while the features
stay put.

**Margin-based losses (ArcFace, CosFace) L2-normalise both features and weights
and explicitly penalise angular distance between a sample and its class weight.
That is the exact drift the mechanism requires.**

So: does the illusion survive when the loss forbids the shortcut?

Three mutually exclusive outcomes, all publishable:

1. Unlearning under ArcFace must move the features — forgetting is deeper.
2. Unlearning under ArcFace fails — the shortcut is closed and methods break.
3. A third mechanism appears that nobody has characterised.

---

## Design

One variable. Everything else held fixed.

| | Model A | Model B | Model C (optional) |
|---|---|---|---|
| Head | softmax CE | ArcFace | CosFace |
| Backbone | same | same | same |
| Data, augmentation, schedule, seed | same | same | same |

Then unlearn the same identities from each and measure.

**Primary outcome:** `nc3_forget` — cosine between the forget class's
classifier weight and its class-mean feature. If this collapses under CE but
holds under ArcFace, the mechanism is loss-dependent.

**Secondary:** probe gap vs reference, NCC forget accuracy, verification AUC.

---

## Setup

```bash
git init
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/test_metrics.py         # must print "8 test groups passed"
```

Open the folder in VS Code. `.vscode/` is preconfigured: `src/` is on the
Python path, pytest is enabled, and there are launch configs for the smoke
test and both pilot runs (F5 to pick one).

**No GPU yet?** The smoke test runs on CPU in a couple of minutes and checks
that every module is wired together:

```bash
python scripts/run_experiment.py --config configs/cifar_ce.yaml --smoke
```

**On a shared GPU machine, always pin a device.** `device: auto` in
`configs/base.yaml` resolves to plain `"cuda"`, which is PyTorch's default
device -- GPU 0 -- if nothing is pinned. Nothing in this repo sets
`CUDA_VISIBLE_DEVICES` for you, so two people running at once will silently
collide on the same card. Check for a free GPU and pin one before any real
(non-smoke) run:

```bash
nvidia-smi                              # find a free GPU
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/cifar_ce.yaml
```

## Running

```bash
# the pilot pair -- 20-30 min each on a 48 GB card
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/cifar_ce.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/cifar_arcface.yaml
python scripts/run_experiment.py --compare logs/

# turn on unlearning
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/cifar_ce.yaml \
    --set unlearn.enabled=true

# faces, once you have the data
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py --config configs/faces_arcface.yaml
```

Every run writes `logs/<name>/` containing `config.json`, `env.json` (git
commit, device, argv), `run.log`, `results.jsonl` and `ckpt.pt`. Every number
is traceable to a config, a seed and a commit.

## Layout

```
configs/base.yaml       shared defaults; other configs inherit and override
configs/cifar_*.yaml    pilot: one per head
configs/faces_*.yaml    the real experiment

src/heads.py            CE / ArcFace / CosFace, matched interfaces
src/backbones.py        ResNet feature extractors (no classifier)
src/data.py             datasets + forget/retain/held-out splits
src/unlearn.py          finetune, neggrad, neggrad+, random_label
src/train.py            training loop + the evaluate() harness
src/metrics.py          NC1-3, linear probe, NCC, verification AUC
src/test_metrics.py     regression tests -- run after every metrics change
src/utils.py            config, seeding, run directories

scripts/run_experiment.py   entry point
logs/                       one directory per run
notes/decisions.md          decision log with dates
```

Run the tests before trusting any number:

```bash
python src/test_metrics.py
```

---

## Two measurement traps already found

These were caught by the test suite on synthetic data. Both would have
silently corrupted results.

### 1. Centring contamination

Standard NC3 centres weights by the global weight mean. When unlearning flips
one class's weight, that flip moves the global mean and shifts the centring for
**every** class. Retained classes then look misaligned when nothing happened
to them.

Measured: with only class 0 perturbed, retain-class mean alignment fell from
1.00 to **0.94** — a 6-point artefact that reads as real spillover.

**Fix:** pass `exclude_from_centre=<forget_class>` so the centring reference
comes from retained classes only. Always do this when a forget class exists.

### 2. Centred vs uncentred changes the headline number

The theory's prediction is about the **uncentred** weight. Under an exact flip:

| convention | forget cosine |
|---|---|
| uncentred | **-1.00** |
| centred (corrected) | **-0.71** |

Same model, same flip, different numbers. Centring subtracts a global mean that
is not part of the theoretical claim, so it dilutes the effect.

**Report both, and state the convention.** A reviewer who recomputes this the
other way will otherwise think you made an error.

---

## Week-by-week, with gates

### Week 1 — infrastructure only

- [ ] **Check face dataset access first.** CASIA-WebFace or VGGFace2 through
      Howard. Fifteen minutes, and it can invalidate everything below.
- [ ] Clone `github.com/ycgao1/CMF_Unlearning`; reproduce one Table 1 number
      (CIFAR-10, one forget class, SalUn, probe ≈ 92.57).
- [ ] Build the identity subset: 500–1000 identities, 20–50 images each.
      Hold out images per identity for later generalisation tests.
- [ ] Pin versions, seed everything.

**GATE:** if their pipeline will not reproduce on the setting they built it
for, do not trust it on a setting they did not. Stuck past a week — email
Yichen Gao (Ohio State).

### Week 2 — the two models

- [ ] Train Model A (CE) and Model B (ArcFace) to convergence.
- [ ] Match final retain accuracy as closely as possible.
- [ ] Measure NC1/NC2/NC3 on both **before any unlearning**.

**GATE:** does neural collapse geometry differ between the two heads?
- Tighter alignment under ArcFace → theory holds, prediction is on solid ground.
- Worse alignment under ArcFace → plausible given imbalance and many classes;
  changes what to expect downstream. Still a finding.
- Neither shows collapse → the alignment metric may be uninformative here.
  Come back and rethink the measurement before spending weeks 3–4.

### Weeks 3–4 — the core experiment

- [ ] Methods: SalUn (essential — the sharpest failure case), SCRUB, NegGrad+.
- [ ] For each method × each head: unlearn one identity, measure all four levels.
- [ ] Repeat over 5–10 forget identities; report mean and standard deviation.

**GATE:** does `nc3_forget` differ between heads?
- Large under CE, small under ArcFace → **the paper.**
- Same under both → mechanism is general. Weaker, still publishable.
- Incoherent → setup problem. Debug before continuing.

### Weeks 5–8 — depth

- [ ] Add CosFace to separate "margin losses" from "ArcFace specifically".
- [ ] Generalisation across appearance: unlearn on images A, probe on held-out B.
- [ ] Similar-identity control: nearest neighbours of the forget identity in
      embedding space, to check forgetting is targeted rather than global damage.
- [ ] Package the audit suite for release.

### Weeks 9–10 — write and submit

---

## Standing rules

**Report forget and utility as a pair, always.** A method that forgets
perfectly because the model broke has forgotten nothing.

**Absolute probe accuracy is not evidence.** A retrained reference scored 77.35
on a class it never saw. Only the gap against a reference means anything.

**Label derived quantities.** Probe gap is our arithmetic, not a metric from
the paper. Say so in every table.

**Do not chase face recognition SOTA.** Reviewers will not care about LFW
accuracy. They will care whether the two conditions were matched.

**Log config, seed, and commit hash with every run.** In week eight you will
need to know exactly what produced a number.

---

## What is not being claimed

- Not a new unlearning method.
- Not a claim that identity unlearning is unexplored — it is not.
- Novelty rests on searches of arXiv, web, an index repo and IEEE Xplore.
  **Still outstanding:** ACM DL and a Semantic Scholar citation-graph pass on
  both the AISTATS paper and *Neural Collapse by Design* (arXiv:2605.20302).
  Do these before writing any novelty claim.

**Known competitive risk:** the *Neural Collapse by Design* group works on the
geometry of normalised losses and cites the face-recognition margin family
explicitly. Connecting that to unlearning is one step from where they are.
