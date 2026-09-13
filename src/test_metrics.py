"""
Regression tests for metrics.py.

Run:  python src/test_metrics.py

These use synthetic data where the correct answer is known in advance.
If any of these break, the numbers in your results table are not trustworthy.
Run them after every change to metrics.py.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M


def collapsed_model(K=6, D=32, N=200, noise=0.05, seed=0):
    """A near-neural-collapsed model: tight clusters, weights == class means."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(K, D))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    feats = np.vstack([c[k] + noise * rng.normal(size=(N, D)) for k in range(K)])
    labs = np.concatenate([np.full(N, k) for k in range(K)])
    W = np.stack([feats[labs == k].mean(0) for k in range(K)])
    return feats, labs, W, K


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def test_nc3_aligned():
    print("nc3: a collapsed model is perfectly aligned")
    f, l, W, K = collapsed_model()
    r = M.nc3_alignment(f, l, W, K)
    check("mean cosine ~ 1", r["mean"] > 0.99)


def test_nc3_flip_uncentred():
    print("nc3: flipping one weight gives cosine -1 (uncentred)")
    f, l, W, K = collapsed_model()
    Wf = W.copy(); Wf[0] = -Wf[0]
    r = M.nc3_alignment(f, l, Wf, K, centre=False)
    check("forget cosine ~ -1", r["per_class"][0] < -0.999)
    check("retain unaffected", np.nanmean(r["per_class"][1:]) > 0.99)


def test_nc3_centring_contamination():
    """The bug this suite exists for.

    Global centring lets one flipped class drag every other class's score
    down. Excluding the forget class from the centring reference fixes it.
    """
    print("nc3: global centring contaminates retain classes")
    f, l, W, K = collapsed_model()
    Wf = W.copy(); Wf[0] = -Wf[0]

    bad = M.nc3_alignment(f, l, Wf, K)                        # global
    good = M.nc3_alignment(f, l, Wf, K, exclude_from_centre=0)

    bad_retain = float(np.nanmean(bad["per_class"][1:]))
    check(f"global centring depresses retain ({bad_retain:.3f} < 0.99)", bad_retain < 0.99)
    check(f"corrected keeps retain clean ({good['retain_mean']:.3f})", good["retain_mean"] > 0.99)
    check("corrected still detects the flip", good["forget"] < -0.5)


def test_nc1_nc2_sane():
    print("nc1 / nc2: tighter clusters give lower values")
    f1, l1, _, K = collapsed_model(noise=0.02)
    f2, l2, _, _ = collapsed_model(noise=0.30)
    check("nc1 lower when collapsed",
          M.nc1_within_class_variability(f1, l1, K) < M.nc1_within_class_variability(f2, l2, K))
    check("nc2 finite", np.isfinite(M.nc2_simplex_etf(f1, l1, K)))


def test_nc1_angular_scale_invariant():
    """The bug angular NC1 exists to avoid: a per-sample magnitude change
    with no change in direction should not look like a change in collapse."""
    print("nc1_angular: invariant to per-sample feature magnitude, raw nc1 is not")
    f, l, _, K = collapsed_model()
    rng = np.random.default_rng(2)
    scale = rng.uniform(0.1, 10.0, size=(f.shape[0], 1))
    f_scaled = f * scale

    raw_orig = M.nc1_within_class_variability(f, l, K)
    raw_scaled = M.nc1_within_class_variability(f_scaled, l, K)
    ang_orig = M.nc1_angular(f, l, K)
    ang_scaled = M.nc1_angular(f_scaled, l, K)

    check(f"raw nc1 changes under per-sample scaling ({raw_orig:.4f} -> {raw_scaled:.4f})",
          abs(raw_orig - raw_scaled) > 1e-6)
    check(f"angular nc1 unchanged under per-sample scaling ({ang_orig:.6f} -> {ang_scaled:.6f})",
          abs(ang_orig - ang_scaled) < 1e-9)


def test_probe_and_ncc():
    print("probe / ncc: recover a separable class")
    f, l, _, K = collapsed_model()
    rng = np.random.default_rng(1)
    tr = rng.random(len(l)) < 0.7
    p = M.linear_probe(f[tr], l[tr], f[~tr], l[~tr], target_class=0)
    n = M.ncc_accuracy(f[tr], l[tr], f[~tr], l[~tr], K, target_class=0)
    check("probe recovers forget class", p["forget"] > 0.95)
    check("ncc recovers forget class", n["forget"] > 0.95)


def test_probe_gap_sign():
    print("probe_gap: sign convention")
    check("above reference is positive", M.probe_gap(92.57, 77.35) > 0)
    check("below reference is negative", M.probe_gap(67.09, 77.35) < 0)
    check("matches published SalUn gap", abs(M.probe_gap(92.57, 77.35) - 15.22) < 0.01)


def test_verification():
    print("verification: separable identities give high AUC")
    f, l, _, _ = collapsed_model()
    v = M.verification_auc(f, l, target_class=0, num_pairs=4000)
    check("auc > 0.95", v["auc"] > 0.95)
    check("forget-pair auc finite", np.isfinite(v["auc_forget"]))


def test_missing_class_is_nan():
    """A class with no samples must be NaN, never a zero vector --
    a zero row would look like a legitimate direction."""
    print("class_means: absent classes are NaN, not zero")
    f, l, _, K = collapsed_model()
    keep = l != 3
    mu = M.class_means(f[keep], l[keep], K)
    check("absent class is NaN", np.isnan(mu[3]).all())
    check("present classes finite", np.isfinite(mu[0]).all())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print()
    print(f"{len(tests)} test groups passed.")
