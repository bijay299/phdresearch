"""
Regression tests for scripts/nc3_mean_robustness.py.

Run:  python src/test_nc3_mean_robustness.py

CPU only, synthetic data, no checkpoint and no GPU. These cover the parts of
the diagnostic that can silently produce a plausible-looking number:

  * the sampler -- same seed must reproduce the same images, a different seed
    must not, and K images must mean K DISTINCT images (sampling with
    replacement would shrink the effective sample size and understate the
    spread, which is the one quantity the diagnostic exists to report);
  * K validation -- a K larger than the pool must fail loudly rather than
    quietly become the full-data computation and report zero variance as
    stability;
  * the per-trial metric -- it must equal the centred-NC3 formula
    `metrics.nc3_alignment` implements, with ONLY the forget-class mean
    restricted to the sampled images and every retained class untouched.

Nothing here asserts a historical value from notes/decisions.md. Those are a
comparison target for the real run; a test fitted to them would defeat the
purpose of reimplementing the computation reproducibly.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import metrics as M                     # noqa: E402
import nc3_mean_robustness as R         # noqa: E402


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def synthetic_model(K=6, D=16, N=60, noise=0.05, seed=0, flip_class=None):
    """Tight clusters with weights set to the class means -- the same
    construction test_metrics.py uses, optionally with one weight flipped so
    the forget class sits near -1 rather than +1."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(K, D))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    feats = np.vstack([c[k] + noise * rng.normal(size=(N, D)) for k in range(K)])
    labs = np.concatenate([np.full(N, k) for k in range(K)])
    W = np.stack([feats[labs == k].mean(0) for k in range(K)])
    if flip_class is not None:
        W[flip_class] = -W[flip_class]
    return feats, labs, W, K


def explicit_centred_nc3_forget(feats, labs, W, K, fc, sample_rows):
    """The centred-NC3 formula written out independently of metrics.py.

    Class means for retained classes from ALL their rows, the forget mean
    from `sample_rows` only; centring reference from retained classes only
    (the exclude_from_centre correction); cosine of the two centred forget
    vectors.
    """
    mu = np.stack([
        feats[sample_rows].mean(axis=0) if k == fc else feats[labs == k].mean(axis=0)
        for k in range(K)
    ])
    ref = [k for k in range(K) if k != fc]
    mu_c = mu - mu[ref].mean(axis=0)
    w_c = W - W[ref].mean(axis=0)
    a = mu_c[fc] / np.linalg.norm(mu_c[fc])
    b = w_c[fc] / np.linalg.norm(w_c[fc])
    return float(a @ b)


# ----------------------------------------------------------------------
# sampling determinism
# ----------------------------------------------------------------------

def test_identical_seed_gives_identical_samples_and_metrics():
    print("sampler: the same seed reproduces the same images and the same numbers")
    feats, labs, W, K = synthetic_model(flip_class=0)
    rows = np.where(labs == 0)[0]

    a = R.sample_trials(rows, k=10, trials=8, seed=4242)
    b = R.sample_trials(rows, k=10, trials=8, seed=4242)
    check("same number of trials", len(a) == len(b) == 8)
    check("every trial's sampled set is identical",
          all(np.array_equal(x, y) for x, y in zip(a, b)))

    va = [R.sampled_centred_nc3_forget(feats, labs, W, K, 0, s) for s in a]
    vb = [R.sampled_centred_nc3_forget(feats, labs, W, K, 0, s) for s in b]
    check("every trial's metric is identical", va == vb)

    # Trial t depends on the seed alone, not on how many trials were asked
    # for: extending a run must not move the trials already recorded.
    longer = R.sample_trials(rows, k=10, trials=12, seed=4242)
    check("asking for more trials leaves the first ones unchanged",
          all(np.array_equal(x, y) for x, y in zip(a, longer[:8])))


def test_different_seed_changes_at_least_one_sampled_set():
    print("sampler: a different seed draws different images")
    feats, labs, _, _ = synthetic_model()
    rows = np.where(labs == 0)[0]

    a = R.sample_trials(rows, k=10, trials=8, seed=4242)
    b = R.sample_trials(rows, k=10, trials=8, seed=4243)
    differing = sum(not np.array_equal(x, y) for x, y in zip(a, b))
    check(f"at least one trial differs ({differing}/8 differ)", differing >= 1)


def test_sampling_is_without_replacement_and_returns_k():
    print("sampler: K distinct images, all of them from the forget class")
    feats, labs, _, _ = synthetic_model(K=6, N=60)
    rows = np.where(labs == 2)[0]
    pool = set(int(i) for i in rows)

    for k in (1, 7, 40, len(rows)):
        sets = R.sample_trials(rows, k=k, trials=5, seed=7)
        check(f"K={k}: every trial has exactly {k} entries",
              all(len(s) == k for s in sets))
        check(f"K={k}: no repeats within a trial",
              all(len(set(int(i) for i in s)) == k for s in sets))
        check(f"K={k}: every sampled row belongs to the forget class",
              all(set(int(i) for i in s) <= pool for s in sets))


# ----------------------------------------------------------------------
# validation / error handling
# ----------------------------------------------------------------------

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def test_invalid_k_and_arguments_are_rejected():
    print("validation: bad K, trials, forget class and output directory all fail loudly")
    check("K=0 rejected", _raises(lambda: R.validate_k(0, 100)))
    check("negative K rejected", _raises(lambda: R.validate_k(-5, 100)))
    check("K larger than the pool rejected", _raises(lambda: R.validate_k(101, 100)))
    check("non-integer K rejected", _raises(lambda: R.validate_k(40.0, 100)))
    check("boolean K rejected", _raises(lambda: R.validate_k(True, 100)))
    check("K=1 accepted", R.validate_k(1, 100) == 1)
    check("K equal to the pool accepted", R.validate_k(100, 100) == 100)
    check("default K=40 accepted against a CIFAR-sized pool",
          R.validate_k(40, 5000) == 40)

    check("trials=0 rejected", _raises(lambda: R.validate_trials(0)))
    check("non-integer trials rejected", _raises(lambda: R.validate_trials(2.5)))
    check("default trials=20 accepted", R.validate_trials(20) == 20)

    targets = np.concatenate([np.full(5, c) for c in range(10)])
    check("forget class >= num_classes rejected",
          _raises(lambda: R.validate_forget_class(10, 10, targets)))
    check("negative forget class rejected",
          _raises(lambda: R.validate_forget_class(-1, 10, targets)))
    check("forget class absent from the split rejected",
          _raises(lambda: R.validate_forget_class(3, 10, targets[targets != 3])))
    check("in-range forget class accepted",
          R.validate_forget_class(3, 10, targets) == 3)

    # The nonempty-output-directory refusal is RunDir.create's rule and is
    # covered by src/test_utils.py; this script calls it rather than
    # reimplementing it.


# ----------------------------------------------------------------------
# replay provenance
# ----------------------------------------------------------------------

SWEEP_ARGV = [
    "scripts/unlearn_across_classes.py", "--config", "configs/cifar_arcface.yaml",
    "--forget-classes", "0,1,2,3", "--ckpt", "logs/x/ckpt.pt",
    "--conditions", "random_label_clfonly", "--set", "seed=0",
]


def test_sweep_ordering_decides_whether_a_replay_can_be_exact():
    """A sweep runs its conditions in ONE process, so anything executed before
    the target advanced the RNG too. Getting this ordering wrong is how a
    replay would be called exact when it cannot be."""
    print("replay: the recorded argv decides what preceded the target condition")

    seq = R.sweep_condition_sequence(SWEEP_ARGV)
    check("only the requested condition is in the sequence",
          seq == [(0, "random_label_clfonly"), (1, "random_label_clfonly"),
                  (2, "random_label_clfonly"), (3, "random_label_clfonly")])

    both = R.sweep_condition_sequence(
        [x if x != "random_label_clfonly" else "random_label_clfonly,finetune"
         for x in SWEEP_ARGV])
    check("finetune runs before random_label_clfonly within a class, "
          "whatever order --conditions lists them in",
          both[:2] == [(0, "finetune"), (0, "random_label_clfonly")])

    default = R.sweep_condition_sequence(
        [x for i, x in enumerate(SWEEP_ARGV)
         if x != "--conditions" and SWEEP_ARGV[i - 1] != "--conditions"])
    check("omitting --conditions means both conditions ran",
          default[:2] == [(0, "finetune"), (0, "random_label_clfonly")])

    check("a non-sweep argv yields UNKNOWN, not 'nothing preceded it'",
          R.sweep_condition_sequence(["scripts/run_experiment.py"]) is None)


def test_reference_reading_reports_absence_as_absence():
    print("replay: a missing or unusable reference is recorded, never assumed to agree")

    with tempfile.TemporaryDirectory() as base:
        missing = R.read_reference(Path(base) / "nope", 0)
        check("absent reference gives no canonical value",
              missing["canonical_epoch1_nc3_centred_forget"] is None)
        check("absent reference is not claimed to be first in a sweep",
              missing["target_was_first_in_sweep"] is None)

        ref = Path(base) / "ref"
        ref.mkdir()
        (ref / "env.json").write_text(json.dumps({"argv": SWEEP_ARGV}))
        (ref / "trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"forget_class": 0, "epoch": 0, "nc3_centred_forget": 0.97},
            {"forget_class": 0, "epoch": 1, "nc3_centred_forget": -0.5},
            {"forget_class": 0, "epoch": 2, "nc3_centred_forget": -0.9},
        ]))

        got = R.read_reference(ref, 0)
        check("reads the epoch-1 row, not epoch 0 or 2",
              got["canonical_epoch1_nc3_centred_forget"] == -0.5)
        check("fc0 was first in this sweep", got["target_was_first_in_sweep"] is True)
        check("no preceding conditions recorded", got["preceding_conditions"] == [])

        later = R.read_reference(ref, 2)
        check("fc2 was NOT first in the same sweep",
              later["target_was_first_in_sweep"] is False)
        check("its prefix is named in order",
              later["preceding_conditions"] ==
              ["fc0:random_label_clfonly", "fc1:random_label_clfonly"])
        check("a class with no epoch-1 row gives no canonical value",
              later["canonical_epoch1_nc3_centred_forget"] is None)


# ----------------------------------------------------------------------
# the metric itself
# ----------------------------------------------------------------------

def test_sampled_metric_matches_the_centred_nc3_formula():
    print("metric: the sampled value equals the centred-NC3 formula, with only "
          "the forget mean restricted")
    feats, labs, W, K = synthetic_model(K=6, D=16, N=60, noise=0.25, flip_class=0)
    fc = 0
    rows = np.where(labs == fc)[0]

    # Using every forget-class image must reproduce the production full-data
    # call exactly -- the sampled path is the same computation at K = N.
    full = M.nc3_alignment(feats, labs, W, K, centre=True, exclude_from_centre=fc)
    at_full_k = R.sampled_centred_nc3_forget(feats, labs, W, K, fc, rows)
    check(f"K=N reproduces the full-data metric ({at_full_k:+.6f} vs "
          f"{full['forget']:+.6f})", abs(at_full_k - full["forget"]) < 1e-9)

    # And at K < N it must equal the formula written out by hand, which
    # leaves every retained class's mean and the centring reference alone.
    worst = 0.0
    for sel in R.sample_trials(rows, k=12, trials=10, seed=99):
        got = R.sampled_centred_nc3_forget(feats, labs, W, K, fc, sel)
        want = explicit_centred_nc3_forget(feats, labs, W, K, fc, sel)
        worst = max(worst, abs(got - want))
    check(f"K<N matches the independent formula (max |diff| {worst:.2e})",
          worst < 1e-9)

    # A subsample must move the number without changing its meaning: the
    # sampled values scatter, and they scatter around the full-data value
    # rather than around something else.
    vals = [R.sampled_centred_nc3_forget(feats, labs, W, K, fc, s)
            for s in R.sample_trials(rows, k=12, trials=10, seed=99)]
    check("subsampling actually changes the estimate",
          len(set(vals)) > 1)
    check(f"sampled values stay near the full-data value "
          f"({min(vals):+.4f}..{max(vals):+.4f} vs {full['forget']:+.4f})",
          abs(float(np.mean(vals)) - full["forget"]) < 0.1)


def test_retained_classes_are_untouched_by_the_subsample():
    print("metric: subsampling the forget class leaves retained classes alone")
    feats, labs, W, K = synthetic_model(K=6, D=16, N=60, flip_class=0)
    fc = 0
    rows = np.where(labs == fc)[0]
    sel = R.sample_trials(rows, k=10, trials=1, seed=5)[0]

    keep = np.where(labs != fc)[0]
    idx = np.sort(np.concatenate([keep, sel]))
    sub = M.nc3_alignment(feats[idx], labs[idx], W, K, centre=True,
                          exclude_from_centre=fc)
    full = M.nc3_alignment(feats, labs, W, K, centre=True, exclude_from_centre=fc)
    check(f"retain_mean unchanged ({sub['retain_mean']:.6f} vs "
          f"{full['retain_mean']:.6f})",
          abs(sub["retain_mean"] - full["retain_mean"]) < 1e-9)
    check("per-class retain cosines unchanged",
          np.allclose(sub["per_class"][1:], full["per_class"][1:], atol=1e-9))


def test_summary_counts_and_spread():
    print("summary: counts below 0 and below -0.5 are reported as counted")
    s = R.summarise([-0.9, -0.6, -0.4, 0.2, 0.3])
    check("n_trials", s["n_trials"] == 5)
    check("min", abs(s["min"] + 0.9) < 1e-12)
    check("max", abs(s["max"] - 0.3) < 1e-12)
    check("mean", abs(s["mean"] - np.mean([-0.9, -0.6, -0.4, 0.2, 0.3])) < 1e-12)
    check("n_below_0 counts 3", s["n_below_0"] == 3)
    check("n_below_neg_0_5 counts 2", s["n_below_neg_0_5"] == 2)
    check("both std conventions reported",
          np.isfinite(s["std"]) and np.isfinite(s["std_ddof0"]))

    one = R.summarise([0.5])
    check("a single trial has no sample std but still reports min/max",
          np.isnan(one["std"]) and one["min"] == one["max"] == 0.5)


def test_sample_identifiers_survive_subsetting():
    print("audit trail: sampled rows map back to the underlying dataset's own indices")

    class _Tiny:
        def __init__(self, n):
            self.samples = [(f"/img/{i:04d}.jpg", i % 3) for i in range(n)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return torch.zeros(1), self.samples[i][1]

    base = _Tiny(50)
    train_idx = np.arange(10, 40)          # what stratified_image_split hands back
    view = Subset(base, list(train_idx))

    rows = [0, 5, 29]
    idx, paths = R.sample_identifiers(view, rows)
    check("indices are remapped through the Subset", idx == [10, 15, 39])
    check("paths point at the right images",
          paths == ["/img/0010.jpg", "/img/0015.jpg", "/img/0039.jpg"])

    plain_idx, plain_paths = R.sample_identifiers(base, rows)
    check("an unwrapped dataset needs no remapping", plain_idx == rows)
    check("paths still resolve", plain_paths[0] == "/img/0000.jpg")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print()
    print(f"{len(tests)} test groups passed.")
