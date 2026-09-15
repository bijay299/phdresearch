"""
Regression tests for movement.py.

Run:  python src/test_movement.py

Synthetic data with known answers, in the spirit of src/test_metrics.py. The
point of this file is that a feature-movement number is only worth anything if
it reports ZERO for changes that are not movement -- a global rotation, a
global rescale -- and nonzero only for the thing it claims to detect.

If any of these break, do not trust a movement number in a results table.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movement as MV


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def raises(name, fn, needle=""):
    try:
        fn()
    except Exception as e:                       # noqa: BLE001 -- that's the test
        ok = needle.lower() in str(e).lower()
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            raise AssertionError(f"{name}: wrong error message: {e}")
        return
    print(f"  FAIL  {name}")
    raise AssertionError(f"{name}: expected an exception, none raised")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

K, D, N = 6, 16, 40
FORGET = 0


def toy(seed=0):
    """Well-separated clusters; feature dim small enough that the anchor count
    comfortably exceeds D, so the anchor-sufficiency guard is not what is
    under test here."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(K, D))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    feats = np.vstack([centres[k] + 0.08 * rng.normal(size=(N, D))
                       for k in range(K)])
    labels = np.concatenate([np.full(N, k) for k in range(K)])
    return feats, labels


def random_orthogonal(d, seed=0, det=None):
    """Random orthogonal matrix. `det=+1` forces a proper rotation, `det=-1`
    an improper one (a reflection). QR alone gives whichever sign it happens
    to give, so the sign is set explicitly rather than assumed."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    q = q * np.sign(np.diag(r))                  # fix QR sign ambiguity
    if det is not None and np.sign(np.linalg.det(q)) != np.sign(det):
        q[:, 0] *= -1.0
    return q


def split(labels, seed=0):
    return MV.stratified_anchor_split(labels, FORGET, 0.5, seed)


def report(f0, f1, labels, seed=0, controls=True):
    a, e = split(labels, seed)
    cc = (MV.select_control_classes(labels, FORGET, K - 1, seed)
          if controls else None)
    return MV.movement_report(f0, f1, labels, K, FORGET, a, e,
                              control_classes=cc)


def big_toy(seed=0, n=1200):
    """Production-like anchor-to-dimension ratio (~37:1 here, ~40:1 on the
    real datasets), so the retain-anchored design's own bias is in the regime
    the GPU cells actually run at."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(K, D))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    feats = np.vstack([centres[k] + 0.08 * rng.normal(size=(n, D))
                       for k in range(K)])
    labels = np.concatenate([np.full(n, k) for k in range(K)])
    return feats, labels


# ----------------------------------------------------------------------
# 1. identical features
# ----------------------------------------------------------------------

def test_identical():
    print("identical features: every displacement is zero, CKA is 1")
    f, labels = toy()
    r = report(f, f.copy(), labels)
    check("raw forget mean ~ 0", r["raw_forget"]["mean"] < 1e-9)
    check("raw retain mean ~ 0", r["raw_retain_eval"]["mean"] < 1e-9)
    check("aligned forget mean ~ 0", r["aligned_forget"]["mean"] < 1e-8)
    check("aligned retain mean ~ 0", r["aligned_retain_eval"]["mean"] < 1e-8)
    check("aligned forget p95 ~ 0", r["aligned_forget"]["p95"] < 1e-8)
    check("forget-minus-retain ~ 0", abs(r["aligned_forget_minus_retain_mean"]) < 1e-8)
    check("class-mean forget ~ 0", r["aligned_class_mean_forget_deg"] < 1e-8)
    check("class-mean retain macro ~ 0",
          r["aligned_class_mean_retain_macro_deg"] < 1e-8)
    check("CKA ~ 1", abs(r["cka_linear_secondary"] - 1.0) < 1e-10)
    check("rotation is orthogonal", r["rotation_orthogonality_err"] < 1e-10)


# ----------------------------------------------------------------------
# 2. global rotation -- the reason alignment exists
# ----------------------------------------------------------------------

def test_global_rotation():
    print("global rotation: raw displacement is large, aligned is zero")
    f, labels = toy()
    q = random_orthogonal(D, seed=3, det=+1)
    r = report(f, f @ q, labels)
    check("raw forget mean is large (>20 deg)", r["raw_forget"]["mean"] > 20.0)
    check("raw retain mean is large (>20 deg)",
          r["raw_retain_eval"]["mean"] > 20.0)
    check("aligned forget mean ~ 0", r["aligned_forget"]["mean"] < 1e-6)
    check("aligned retain mean ~ 0", r["aligned_retain_eval"]["mean"] < 1e-6)
    check("aligned forget p95 ~ 0", r["aligned_forget"]["p95"] < 1e-6)
    check("forget-minus-retain ~ 0",
          abs(r["aligned_forget_minus_retain_mean"]) < 1e-6)
    check("class-mean forget ~ 0", r["aligned_class_mean_forget_deg"] < 1e-6)
    check("CKA ~ 1 (rotation invariant)",
          abs(r["cka_linear_secondary"] - 1.0) < 1e-10)


def test_reflection_is_not_movement():
    print("reflection: an improper orthogonal map is also not movement")
    f, labels = toy()
    q = random_orthogonal(D, seed=5, det=-1)
    check("fixture really is a reflection", np.linalg.det(q) < -0.99)
    r = report(f, f @ q, labels)
    check("aligned forget mean ~ 0", r["aligned_forget"]["mean"] < 1e-6)
    check("aligned retain mean ~ 0", r["aligned_retain_eval"]["mean"] < 1e-6)
    check("fitted rotation has det ~ -1", r["rotation_det"] < -0.99)
    check("proper rotations would have been forced to +1 -- they are not",
          abs(abs(r["rotation_det"]) - 1.0) < 1e-9)


# ----------------------------------------------------------------------
# 3. global scale
# ----------------------------------------------------------------------

def test_global_scale():
    print("global scale: normalisation removes it entirely")
    f, labels = toy()
    r = report(f, 7.5 * f, labels)
    check("raw forget mean ~ 0", r["raw_forget"]["mean"] < 1e-9)
    check("aligned forget mean ~ 0", r["aligned_forget"]["mean"] < 1e-8)
    check("class-mean forget ~ 0", r["aligned_class_mean_forget_deg"] < 1e-8)
    check("CKA ~ 1", abs(r["cka_linear_secondary"] - 1.0) < 1e-10)


def test_per_sample_scale():
    print("per-sample rescaling is also not angular movement")
    f, labels = toy()
    rng = np.random.default_rng(11)
    scaled = f * rng.uniform(0.2, 5.0, size=(f.shape[0], 1))
    r = report(f, scaled, labels)
    check("raw forget mean ~ 0", r["raw_forget"]["mean"] < 1e-9)
    check("aligned forget mean ~ 0", r["aligned_forget"]["mean"] < 1e-8)


# ----------------------------------------------------------------------
# 4. forget-only movement
# ----------------------------------------------------------------------

def test_forget_only_movement():
    print("forget-only movement: forget moves, held-out retain does not")
    f, labels = toy()
    rng = np.random.default_rng(7)
    moved = f.copy()
    m = labels == FORGET
    moved[m] = moved[m] + 0.9 * rng.normal(size=(m.sum(), D))
    r = report(f, moved, labels)
    check("aligned forget mean > 10 deg", r["aligned_forget"]["mean"] > 10.0)
    check("aligned retain mean < 1 deg", r["aligned_retain_eval"]["mean"] < 1.0)
    check("forget-minus-retain clearly positive",
          r["aligned_forget_minus_retain_mean"] > 9.0)
    check("class-mean forget > class-mean retain macro",
          r["aligned_class_mean_forget_deg"]
          > r["aligned_class_mean_retain_macro_deg"] + 5.0)
    check("class-mean forget-minus-retain positive",
          r["aligned_class_mean_forget_minus_retain_deg"] > 5.0)
    check("forget exceeds every sampled control", r["aligned_forget_above_all_sampled_controls"])
    check("forget-minus-control clearly positive",
          r["aligned_forget_minus_control_mean"] > 9.0)


def test_controls_are_fitted_on_fewer_anchors():
    """The asymmetry that makes these descriptive, not exchangeable: a control
    rotation excludes forget AND the control class; the forget rotation
    excludes only forget. Recorded per control so it is auditable."""
    print("controls: fitted on strictly fewer anchors than the forget class")
    f, labels = big_toy()
    a, _e = split(labels, 0)
    cc = MV.select_control_classes(labels, FORGET, K - 1, 0)
    nulls = MV.control_class_angles(MV.unit_rows(f), MV.unit_rows(f), labels, a, cc)
    check("every control has fewer anchors than the forget rotation",
          all(v["n_anchor"] < a.size for v in nulls.values()))
    check("n_anchor is recorded for every control",
          all("n_anchor" in v for v in nulls.values()))
    check("the deficit equals that class's anchor count",
          all(nulls[int(c)]["n_anchor"] == a.size - int((labels[a] == c).sum())
              for c in cc))


def test_forget_only_movement_hides_in_cka():
    print("forget-only movement: global CKA barely notices (why it is secondary)")
    f, labels = toy()
    rng = np.random.default_rng(7)
    moved = f.copy()
    m = labels == FORGET
    moved[m] = moved[m] + 0.9 * rng.normal(size=(m.sum(), D))
    r = report(f, moved, labels)
    check("CKA still high (>0.9) despite a moved class",
          r["cka_linear_secondary"] > 0.9)
    check("but aligned forget displacement is large",
          r["aligned_forget"]["mean"] > 10.0)


# ----------------------------------------------------------------------
# 5. equal forget-and-retain movement
# ----------------------------------------------------------------------

def test_equal_movement():
    print("equal forget/retain movement: forget sits inside the control spread")
    f, labels = big_toy()
    rng = np.random.default_rng(13)
    moved = f + 0.25 * rng.normal(size=f.shape)  # independent, not a rotation
    r = report(f, moved, labels)
    check("aligned forget mean > 10 deg", r["aligned_forget"]["mean"] > 10.0)
    check("aligned retain mean > 10 deg",
          r["aligned_retain_eval"]["mean"] > 10.0)
    check("independent noise is NOT absorbed by alignment",
          r["aligned_retain_eval"]["mean"] > 10.0)
    check("controls were computed", r["control_n"] == K - 1)
    check("forget does NOT exceed the control maximum",
          not r["aligned_forget_above_all_sampled_controls"])
    check("forget-minus-control is small (<2 deg)",
          abs(r["aligned_forget_minus_control_mean"]) < 2.0)


def test_equal_movement_bias_is_bounded_at_production_ratio():
    print("equal movement: the retain-anchored bias is small but not zero")
    f, labels = big_toy()
    rng = np.random.default_rng(13)
    moved = f + 0.25 * rng.normal(size=f.shape)
    r = report(f, moved, labels)
    bias = r["aligned_forget_minus_retain_mean"]
    check("pooled forget-minus-retain is biased positive, not zero", bias > 0.0)
    check("but under 2 deg at a production anchor ratio", bias < 2.0)
    check("the controls absorb most of that bias",
          abs(r["aligned_forget_minus_control_mean"]) < abs(bias))


def test_control_class_properties():
    print("controls: deterministic, forget-free, anchor-free by class")
    f, labels = big_toy()
    cc = MV.select_control_classes(labels, FORGET, 3, 0)
    check("never selects the forget class", FORGET not in cc)
    check("selects the requested count", cc.size == 3)
    check("sorted", np.array_equal(cc, np.sort(cc)))
    check("deterministic",
          np.array_equal(cc, MV.select_control_classes(labels, FORGET, 3, 0)))
    check("caps at the number of retain classes",
          MV.select_control_classes(labels, FORGET, 99, 0).size == K - 1)
    check("n_controls=0 gives nothing",
          MV.select_control_classes(labels, FORGET, 0, 0).size == 0)

    a, _e = split(labels, 0)
    u = MV.unit_rows(f)
    nulls = MV.control_class_angles(u, u.copy(), labels, a, cc)
    check("identical features -> every control ~0",
          all(v["mean"] < 1e-8 for v in nulls.values()))
    for c in cc:
        n_all = int((labels[a] != c).sum())
        check(f"class {c} excluded from its own anchors",
              nulls[int(c)]["n_anchor"] == n_all)
    check("a control class is never its own anchor",
          all(nulls[int(c)]["n_anchor"] < a.size for c in cc))


# ----------------------------------------------------------------------
# 6. shuffled indices / mismatches
# ----------------------------------------------------------------------

def test_pairing_guards():
    print("pairing guards: index, order and label mismatches are rejected")
    idx = np.arange(20)
    lab = np.arange(20) % K
    MV.check_paired(lab, lab.copy(), idx, idx.copy())      # must not raise
    print("  PASS  identical pairing accepted")

    shuffled = np.random.default_rng(0).permutation(idx)
    raises("shuffled index order rejected",
           lambda: MV.check_paired(lab, lab, idx, shuffled), "index mismatch")
    raises("index length mismatch rejected",
           lambda: MV.check_paired(lab, lab, idx, idx[:10]), "index shape")
    bad_lab = lab.copy(); bad_lab[3] += 1
    raises("label mismatch rejected",
           lambda: MV.check_paired(lab, bad_lab, idx, idx), "label mismatch")
    raises("label length mismatch rejected",
           lambda: MV.check_paired(lab, lab[:10], idx, idx), "label shape")


def test_shuffled_features_move():
    print("shuffled rows: pairing the wrong images reads as large movement")
    f, labels = toy()
    rng = np.random.default_rng(2)
    perm = rng.permutation(f.shape[0])
    r = report(f, f[perm], labels)
    check("aligned forget mean is large", r["aligned_forget"]["mean"] > 20.0)
    check("aligned retain mean is large",
          r["aligned_retain_eval"]["mean"] > 20.0)


def test_index_hash():
    print("index hash: order-sensitive and label-sensitive")
    idx = np.arange(50)
    lab = np.arange(50) % K
    h = MV.hash_indices(idx, lab)
    check("stable across calls", h == MV.hash_indices(idx, lab))
    check("changes when order changes",
          h != MV.hash_indices(idx[::-1], lab))
    other = lab.copy(); other[0] += 1
    check("changes when labels change", h != MV.hash_indices(idx, other))


# ----------------------------------------------------------------------
# 7. zero-norm features
# ----------------------------------------------------------------------

def test_zero_norm():
    print("zero-norm features: rejected, never silently scored as 0 degrees")
    f, labels = toy()
    bad = f.copy()
    bad[5] = 0.0
    raises("zero-norm row rejected", lambda: MV.unit_rows(bad), "zero norm")
    raises("zero-norm row rejected inside movement_report",
           lambda: report(f, bad, labels), "zero norm")
    raises("zero-norm baseline rejected too",
           lambda: report(bad, f, labels), "zero norm")
    check("non-zero rows still fine", MV.unit_rows(f).shape == f.shape)
    check("unit_rows really normalises",
          np.allclose(np.linalg.norm(MV.unit_rows(f), axis=1), 1.0))


# ----------------------------------------------------------------------
# 8. insufficient alignment anchors
# ----------------------------------------------------------------------

def test_insufficient_anchors():
    print("insufficient anchors: an underdetermined rotation is refused")
    rng = np.random.default_rng(0)
    src = rng.normal(size=(D - 1, D))
    dst = rng.normal(size=(D - 1, D))
    raises("fewer anchors than dimensions rejected",
           lambda: MV.orthogonal_procrustes_rotation(MV.unit_rows(src),
                                                     MV.unit_rows(dst)),
           "insufficient alignment anchors")
    exact = MV.orthogonal_procrustes_rotation(
        MV.unit_rows(rng.normal(size=(D, D))),
        MV.unit_rows(rng.normal(size=(D, D))))
    check("exactly D anchors accepted", exact.shape == (D, D))


def test_anchor_split_disjoint():
    print("anchor split: disjoint, forget-free, deterministic, stratified")
    _f, labels = toy()
    a, e = split(labels, 0)
    check("anchors and eval are disjoint", np.intersect1d(a, e).size == 0)
    check("no forget rows in anchors", not (labels[a] == FORGET).any())
    check("no forget rows in eval", not (labels[e] == FORGET).any())
    check("covers every retain row", a.size + e.size == (labels != FORGET).sum())
    check("every retain class is represented in anchors",
          len(np.unique(labels[a])) == K - 1)
    check("every retain class is represented in eval",
          len(np.unique(labels[e])) == K - 1)
    a2, e2 = split(labels, 0)
    check("same seed reproduces the split",
          np.array_equal(a, a2) and np.array_equal(e, e2))
    a3, _ = split(labels, 1)
    check("a different seed changes it", not np.array_equal(a, a3))
    check("anchors are sorted", np.array_equal(a, np.sort(a)))

    raises("anchor_fraction 0 rejected",
           lambda: MV.stratified_anchor_split(labels, FORGET, 0.0, 0),
           "strictly between")
    raises("anchor_fraction 1 rejected",
           lambda: MV.stratified_anchor_split(labels, FORGET, 1.0, 0),
           "strictly between")


def test_anchor_leak_is_refused():
    print("anchor leak: overlapping anchor/eval sets are refused")
    f, labels = toy()
    a, e = split(labels, 0)
    leaky = np.concatenate([e, a[:5]])
    raises("overlap rejected",
           lambda: MV.movement_report(f, f, labels, K, FORGET, a, leaky),
           "overlap")
    with_forget = np.concatenate([a, np.flatnonzero(labels == FORGET)[:3]])
    raises("forget rows in anchors rejected",
           lambda: MV.movement_report(f, f, labels, K, FORGET, with_forget, e),
           "forget-class rows")


# ----------------------------------------------------------------------
# 9. deterministic repetition
# ----------------------------------------------------------------------

def test_deterministic_repetition():
    print("deterministic repetition: identical inputs, bit-identical outputs")
    f, labels = toy()
    rng = np.random.default_rng(21)
    moved = f + 0.4 * rng.normal(size=f.shape)
    r1 = report(f, moved, labels, seed=4)
    r2 = report(f, moved, labels, seed=4)
    for k, v in r1.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                check(f"{k}.{kk} identical", vv == r2[k][kk])
        else:
            check(f"{k} identical", v == r2[k])


# ----------------------------------------------------------------------
# angle correctness, chunking, CKA
# ----------------------------------------------------------------------

def test_known_angles():
    print("angles: exact against hand-computed values, in degrees")
    a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    ang = MV.paired_angles_deg(MV.unit_rows(a), MV.unit_rows(b))
    check("0 deg", abs(ang[0] - 0.0) < 1e-12)
    check("90 deg", abs(ang[1] - 90.0) < 1e-12)
    check("180 deg", abs(ang[2] - 180.0) < 1e-10)

    theta = 1e-7
    c = np.array([[np.cos(theta), np.sin(theta), 0.0]])
    tiny = MV.paired_angles_deg(MV.unit_rows(a[:1]), MV.unit_rows(c))[0]
    check("accurate at 1e-7 rad (arccos would lose this)",
          abs(tiny - np.degrees(theta)) < 1e-12)


def test_chunking_invariance():
    print("chunking: results do not depend on the chunk width")
    f, labels = toy()
    rng = np.random.default_rng(31)
    moved = f + 0.3 * rng.normal(size=f.shape)
    u0, u1 = MV.unit_rows(f), MV.unit_rows(moved)
    full = MV.paired_angles_deg(u0, u1, chunk=10 ** 6)
    small = MV.paired_angles_deg(u0, u1, chunk=7)
    check("paired angles identical", np.array_equal(full, small))
    a, _e = split(labels, 0)
    r_full = MV.orthogonal_procrustes_rotation(u1[a], u0[a])
    g_small = MV._cross_gram(u1[a], u0[a], chunk=7)
    g_full = MV._cross_gram(u1[a], u0[a], chunk=10 ** 6)
    check("cross-gram close across chunk widths",
          np.allclose(g_small, g_full, atol=1e-10))
    check("rotation stays orthogonal",
          np.allclose(r_full.T @ r_full, np.eye(D), atol=1e-10))
    check("CKA close across chunk widths",
          abs(MV.linear_cka(u0, u1, chunk=7)
              - MV.linear_cka(u0, u1, chunk=10 ** 6)) < 1e-12)


def test_rotate_means_equals_rotate_features():
    print("class means: rotating the means == rotating every feature")
    f, labels = toy()
    rng = np.random.default_rng(51)
    moved = f + 0.3 * rng.normal(size=f.shape)
    u0, u1 = MV.unit_rows(f), MV.unit_rows(moved)
    a, _e = split(labels, 0)
    rot = MV.orthogonal_procrustes_rotation(u1[a], u0[a])

    cheap = MV.class_mean_angles_deg(u0, u1, labels, K, rot)
    # The expensive form the production path deliberately avoids: rotate the
    # whole (n, d) matrix first, then take class means of that.
    expensive = MV.class_mean_angles_deg(u0, MV.unit_rows(u1 @ rot), labels, K)
    check("identical to 1e-9 deg", np.allclose(cheap, expensive, atol=1e-9))

    counts = np.bincount(labels, minlength=K)
    mu = MV.class_mean_unit(u0, labels, K)
    check("class means are unit norm",
          np.allclose(np.linalg.norm(mu, axis=1), 1.0))
    check("every class present", not np.isnan(mu).any())
    check("absent classes are NaN, not zero",
          np.isnan(MV.class_mean_unit(u0, labels, K + 2)[K:]).all())
    check("counts sum to n", counts.sum() == labels.size)


def test_cka_bounds():
    print("CKA: 1 for identical, lower for unrelated")
    f, _labels = toy()
    rng = np.random.default_rng(41)
    noise = rng.normal(size=f.shape)
    u = MV.unit_rows(f)
    check("identical -> 1", abs(MV.linear_cka(u, u) - 1.0) < 1e-12)
    check("unrelated -> well below 1",
          MV.linear_cka(u, MV.unit_rows(noise)) < 0.5)
    check("symmetric", abs(MV.linear_cka(u, MV.unit_rows(noise))
                           - MV.linear_cka(MV.unit_rows(noise), u)) < 1e-12)


def test_summarize():
    print("summary: mean/median/p95 on a known vector")
    v = np.arange(101, dtype=float)              # 0..100
    s = MV.summarize_angles(v)
    check("mean 50", abs(s["mean"] - 50.0) < 1e-12)
    check("median 50", abs(s["median"] - 50.0) < 1e-12)
    check("p95 95", abs(s["p95"] - 95.0) < 1e-12)
    check("max 100", abs(s["max"] - 100.0) < 1e-12)
    check("n 101", s["n"] == 101)
    raises("non-finite rejected",
           lambda: MV.summarize_angles(np.array([1.0, np.nan])), "non-finite")


def test_shape_guards():
    print("shape guards")
    f, labels = toy()
    raises("paired shape mismatch",
           lambda: MV.paired_angles_deg(MV.unit_rows(f), MV.unit_rows(f[:10])),
           "same shape")
    raises("1-D input rejected",
           lambda: MV.unit_rows(np.ones(5)), "must be 2-D")
    raises("empty input rejected",
           lambda: MV.unit_rows(np.zeros((0, 3))), "empty")
    raises("CKA row mismatch",
           lambda: MV.linear_cka(f, f[:10]), "same samples")
    a, e = split(labels, 0)
    raises("absent forget class rejected",
           lambda: MV.movement_report(f, f, labels, K, K + 3, a, e),
           "no samples")


# ----------------------------------------------------------------------

def main():
    tests = [
        test_identical,
        test_global_rotation,
        test_reflection_is_not_movement,
        test_global_scale,
        test_per_sample_scale,
        test_forget_only_movement,
        test_controls_are_fitted_on_fewer_anchors,
        test_forget_only_movement_hides_in_cka,
        test_equal_movement,
        test_equal_movement_bias_is_bounded_at_production_ratio,
        test_control_class_properties,
        test_pairing_guards,
        test_shuffled_features_move,
        test_index_hash,
        test_zero_norm,
        test_insufficient_anchors,
        test_anchor_split_disjoint,
        test_anchor_leak_is_refused,
        test_deterministic_repetition,
        test_known_angles,
        test_chunking_invariance,
        test_rotate_means_equals_rotate_features,
        test_cka_bounds,
        test_summarize,
        test_shape_guards,
    ]
    for t in tests:
        t()
    print(f"\nmovement: {len(tests)} groups passed")


if __name__ == "__main__":
    main()
