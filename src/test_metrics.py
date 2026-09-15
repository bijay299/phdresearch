"""
Regression tests for metrics.py.

Run:  python src/test_metrics.py

These use synthetic data where the correct answer is known in advance.
If any of these break, the numbers in your results table are not trustworthy.
Run them after every change to metrics.py.
"""

import sys, os, warnings
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


def _legacy_ncc(train_feats, train_labels, test_feats, test_labels,
                num_classes, target_class=None):
    """The pre-chunking `ncc_accuracy`, verbatim, kept here as the reference.

    `ncc_accuracy` was rewritten to chunk over test samples because the
    original built the whole (n_test, n_classes, feat_dim) float64 tensor and
    exhausted host memory at the faces-1000 shape. The rewrite is supposed to
    be bit-identical, not merely close, so the old code has to stay available
    to diff against. Do not "simplify" this into a call to the new one.
    """
    mu = M.class_means(train_feats, train_labels, num_classes)
    present = np.where(~np.isnan(mu).any(axis=1))[0]
    Mu = mu[present]

    d = ((test_feats[:, None, :] - Mu[None, :, :]) ** 2).sum(-1)
    pred = present[d.argmin(axis=1)]

    out = {"overall": float((pred == test_labels).mean())}
    if target_class is not None:
        m = test_labels == target_class
        out["forget"] = float((pred[m] == target_class).mean()) if m.any() else float("nan")
        out["retain"] = float((pred[~m] == test_labels[~m]).mean()) if (~m).any() else float("nan")
    return out


def _same(a, b):
    """Dicts equal key-for-key, treating NaN as equal to NaN."""
    if set(a) != set(b):
        return False
    return all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)


def test_ncc_matches_legacy_both_dtypes():
    """Exact agreement with the pre-chunking reference, float64 and float32.

    Chunking splits rows, never the class axis, and the reduction length
    (feat_dim) is untouched, so every distance should be bit-identical.
    """
    print("ncc: chunked implementation is exact against the legacy reference")
    rng = np.random.default_rng(7)
    for dtype in (np.float64, np.float32):
        f = rng.normal(size=(311, 41)).astype(dtype)
        l = rng.integers(0, 9, size=311)
        tr = rng.random(311) < 0.7
        args = (f[tr], l[tr], f[~tr], l[~tr], 9)
        got = M.ncc_accuracy(*args, target_class=3)
        ref = _legacy_ncc(*args, target_class=3)
        check(f"{np.dtype(dtype).name}: overall/forget/retain identical "
              f"({got['overall']:.6f}/{got['forget']:.6f}/{got['retain']:.6f})",
              _same(got, ref))
        check(f"{np.dtype(dtype).name}: no target_class identical",
              _same(M.ncc_accuracy(*args), _legacy_ncc(*args)))


def test_ncc_chunk_boundaries():
    """Chunk size must not change any answer: size 1, a size that divides the
    test set exactly, and sizes leaving a final partial chunk."""
    print("ncc: answers are invariant to chunk size, including partial chunks")
    rng = np.random.default_rng(11)
    f = rng.normal(size=(240, 17))
    l = rng.integers(0, 6, size=240)
    tr = rng.random(240) < 0.6
    args = (f[tr], l[tr], f[~tr], l[~tr], 6)
    n_test = int((~tr).sum())
    ref = _legacy_ncc(*args, target_class=2)

    real_budget = M.NCC_TEMP_BUDGET_BYTES
    try:
        per_sample = 6 * 17 * 8
        # 1 row; a divisor of n_test if one exists; several partial-chunk cases
        divisors = [c for c in range(2, n_test) if n_test % c == 0]
        sizes = [1] + divisors[:1] + [3, 7, n_test - 1, n_test, n_test + 5]
        saw_exact, saw_partial, saw_multi = False, False, False
        for rows in dict.fromkeys(sizes):              # de-duplicated, ordered
            M.NCC_TEMP_BUDGET_BYTES = rows * per_sample
            plan = M.ncc_memory_plan(n_test, 6, 17, 8, M.NCC_TEMP_BUDGET_BYTES)
            got = M.ncc_accuracy(*args, target_class=2)
            tail = n_test % rows
            saw_exact |= (rows <= n_test and tail == 0)
            saw_partial |= (rows < n_test and tail != 0)
            saw_multi |= plan["n_chunks"] > 2
            check(f"chunk_rows={plan['chunk_rows']:3d} n_chunks={plan['n_chunks']:3d} "
                  f"final_chunk={tail or min(rows, n_test):3d} -> identical",
                  plan["chunk_rows"] == rows and _same(got, ref))
        check(f"n_test={n_test}: covered an exactly-dividing chunk size", saw_exact)
        check(f"n_test={n_test}: covered a partial final chunk", saw_partial)
        check(f"n_test={n_test}: covered >2 chunk boundaries", saw_multi)
    finally:
        M.NCC_TEMP_BUDGET_BYTES = real_budget


def test_ncc_tie_goes_to_first_index():
    """Two class means equidistant from a sample: numpy's argmin takes the
    lower class index. Chunking over classes would have broken this, which is
    why chunks keep every class together."""
    print("ncc: exact ties resolve to the lowest class index")
    # classes 0 and 2 share a mean; class 1 sits far away.
    train_feats = np.array([[1.0, 0.0], [-9.0, 0.0], [1.0, 0.0]])
    train_labels = np.array([0, 1, 2])
    test_feats = np.array([[0.0, 5.0], [0.0, -5.0]])   # equidistant from 0 and 2
    test_labels = np.array([0, 2])
    args = (train_feats, train_labels, test_feats, test_labels, 3)

    got = M.ncc_accuracy(*args, target_class=2)
    ref = _legacy_ncc(*args, target_class=2)
    check("tie broken identically to legacy", _same(got, ref))
    check(f"tie goes to class 0, so class 2 scores 0 (forget={got['forget']:.1f})",
          got["forget"] == 0.0 and got["retain"] == 1.0)

    real_budget = M.NCC_TEMP_BUDGET_BYTES
    try:
        M.NCC_TEMP_BUDGET_BYTES = 3 * 2 * 8          # one row per chunk
        check("tie unchanged at chunk_rows=1", _same(M.ncc_accuracy(*args, target_class=2), ref))
    finally:
        M.NCC_TEMP_BUDGET_BYTES = real_budget


def test_ncc_missing_and_empty_behaviour():
    """Absent classes, an absent target class and an empty test set must
    behave exactly as before the rewrite -- including the failure mode when
    no class has any samples at all."""
    print("ncc: missing-class / empty behaviour is unchanged")
    f, l, _, K = collapsed_model(K=5, N=40)

    keep = l != 3                                     # class 3 never present
    args = (f[keep], l[keep], f, l, K)
    check("absent class 3 still predicted-over identically",
          _same(M.ncc_accuracy(*args, target_class=3), _legacy_ncc(*args, target_class=3)))

    no_target = (f[keep], l[keep], f[l != 3], l[l != 3], K)
    got = M.ncc_accuracy(*no_target, target_class=3)  # target absent from test
    check("target absent from test set -> forget is NaN",
          np.isnan(got["forget"]) and _same(got, _legacy_ncc(*no_target, target_class=3)))

    empty = (f, l, f[:0], l[:0], K)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # mean of empty slice
        got_e = M.ncc_accuracy(*empty, target_class=0)
        ref_e = _legacy_ncc(*empty, target_class=0)
    check("empty test set matches legacy (NaN overall)", _same(got_e, ref_e))

    # No class present at all: legacy reached argmin over an empty axis and
    # raised. Preserve that rather than silently returning.
    for n_rows in (3, 0):
        blank = (f[:0], l[:0], f[:n_rows], l[:n_rows], K)
        raised = []
        for fn in (M.ncc_accuracy, _legacy_ncc):
            try:
                fn(*blank)
            except ValueError:
                raised.append(True)
            except Exception:
                raised.append(False)
            else:
                raised.append(False)
        check(f"no present class, n_test={n_rows}: both raise ValueError", raised == [True, True])


def test_ncc_memory_plan_bounds_faces1000():
    """The regression that motivated the rewrite.

    Nothing here allocates the faces-1000 tensor -- the point is that the plan
    refuses to ask for it.
    """
    print("ncc: faces-1000 shape stays inside the temporary budget")
    n_test, n_cls, d, itemsize = 9704, 1000, 512, 8
    plan = M.ncc_memory_plan(n_test, n_cls, d, itemsize)
    budget = M.NCC_TEMP_BUDGET_BYTES
    gib = 1024 ** 3

    check(f"legacy would have needed {plan['legacy_temp_bytes']/gib:.1f} GiB",
          plan["legacy_temp_bytes"] > 70 * gib)
    check(f"peak temporary {plan['peak_temp_bytes']/1024**2:.1f} MiB <= budget "
          f"{budget/1024**2:.0f} MiB", plan["peak_temp_bytes"] <= budget)
    check(f"chunk_rows={plan['chunk_rows']} is far below n_test={n_test} "
          f"(never the full 3-D tensor)", 1 <= plan["chunk_rows"] < n_test)
    check(f"n_chunks={plan['n_chunks']} covers every row",
          plan["n_chunks"] * plan["chunk_rows"] >= n_test
          and (plan["n_chunks"] - 1) * plan["chunk_rows"] < n_test)
    check(f"reduction factor {plan['legacy_temp_bytes']/plan['peak_temp_bytes']:.0f}x",
          plan["legacy_temp_bytes"] / plan["peak_temp_bytes"] > 100)

    # Budget scales with the problem, not with a dataset-specific constant.
    wide = M.ncc_memory_plan(n_test, n_cls, 4 * d, itemsize)
    check("doubling feature width shrinks the chunk proportionally",
          wide["chunk_rows"] == plan["chunk_rows"] // 4
          and wide["peak_temp_bytes"] <= budget)
    huge = M.ncc_memory_plan(10, 10 ** 6, 10 ** 4, itemsize)
    check("a single row over budget still yields chunk_rows=1, never 0",
          huge["chunk_rows"] == 1)


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
