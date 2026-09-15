"""
Feature-movement diagnostics.

THE QUESTION THIS ANSWERS
-------------------------
Every headline in this project so far rests on `random_label_clfonly`, which
freezes the backbone. Features cannot move under it, so it cannot say whether
unlearning moves the representation when the backbone is free. These metrics
measure that directly: how far each sample's feature travelled between the
pre-unlearning model and the model after N epochs of full-model unlearning.

WHY A ROTATION HAS TO BE DIVIDED OUT
------------------------------------
A paired per-sample angle is NOT invariant to a global orthogonal map of the
representation. Unlearning can rotate the entire feature space while changing
nothing about what the features encode -- every distance, every angle between
pairs, every probe accuracy is preserved -- and a naive paired angle scores
that as large movement. So the raw paired angle is reported (it is the honest
"what happened to the coordinates" number) but the PRIMARY quantity is the
angle after an orthogonal Procrustes rotation fitted on RETAIN ANCHORS ONLY.

Anchors and evaluation samples are disjoint by construction. Fitting the
rotation on the same retain samples it is then evaluated on would drive the
retain control toward zero by fitting it, and the forget-minus-retain
difference would be an artefact of that leak. `stratified_anchor_split`
exists to make the split explicit rather than incidental.

THE RESIDUAL BIAS THAT DISJOINTNESS DOES NOT FIX -- read before comparing
forget against retain
-------------------------------------------------------------------------
Disjoint anchors are necessary but not sufficient. The forget class is absent
from the anchor set entirely, while every retain class -- including the rows
held out for evaluation -- has same-class siblings among the anchors. The
fitted rotation therefore absorbs some of the retain classes' movement and
none of the forget class's, so `forget - retain` is biased POSITIVE even when
forget and retain moved by exactly the same amount.

Measured here on synthetic data with equal, independent movement everywhere,
as a function of the anchor-count-to-feature-dimension ratio:

    anchors/d      6      31     156     281     625
    forget-retain  +13.4  +2.2   +0.1    +0.5    +0.03   (degrees)

Production sits near 40:1 (CIFAR ~22,500 anchors / 512 dims; faces-1000
~19,000 / 512), so the residual is small but not zero, and it points in the
direction that would flatter the hypothesis.

`null_control_angles` is the answer to it: each control class is measured
exactly as the forget class is -- excluded from the anchor set, then scored
under the rotation fitted without it. That yields a null distribution for
"displacement of a class that was NOT unlearned but was treated identically".
A forget-class number is only interpretable against that distribution, never
against the pooled retain-eval number alone.

Reflections are allowed. The full orthogonal group is used, not SO(d): a
reflected representation carries exactly the same information, so restricting
to rotations would score a reflection as movement.

EVERYTHING IS NORMALISED FIRST
------------------------------
All angles are computed on L2-normalised features. Feature magnitude is not
movement -- margin heads normalise before the loss ever sees a feature, so a
norm change is invisible to what ArcFace optimises (same argument as
`metrics.nc1_angular`). A zero-norm feature has no direction and is rejected
rather than silently assigned one.

Angles come back in DEGREES, via `2*atan2(|a-b|, |a+b|)`, not `arccos(a.b)`.
arccos loses precision exactly where this diagnostic needs it most -- near
zero, where the epoch-0 self-comparison gate lives.

CKA IS SECONDARY AND IS LABELLED AS SUCH
----------------------------------------
`linear_cka` aggregates over every sample of every class. Movement confined to
one forget class among 10 -- still more among 1000 -- can leave it essentially
unchanged. It is reported as a global-similarity context number and must never
be used on its own to conclude that the representation did not move.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# Row-chunk width for the accumulations below. Bounds the dominant temporary
# (chunk x feat_dim float64) rather than the full (n x feat_dim) array, so a
# 50k-sample CIFAR extraction never materialises a second float64 copy of
# itself. See notes/decisions.md 2026-09-15 for why memory bounds are not
# optional in this repo.
CHUNK_ROWS = 8192


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

def check_paired(labels_a: np.ndarray, labels_b: np.ndarray,
                 index_a: np.ndarray, index_b: np.ndarray) -> None:
    """Reject any index, label or ordering mismatch between two extractions.

    Paired displacement is meaningless unless row i is the SAME image under
    both models, in the same position. Nothing downstream re-checks this, so
    it is checked loudly here.
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError(
            f"label shape mismatch: {labels_a.shape} vs {labels_b.shape} -- "
            f"the two extractions did not see the same number of samples"
        )
    if index_a.shape != index_b.shape:
        raise ValueError(
            f"index shape mismatch: {index_a.shape} vs {index_b.shape}"
        )
    if not np.array_equal(index_a, index_b):
        bad = int(np.flatnonzero(index_a != index_b)[0])
        raise ValueError(
            f"dataset index mismatch at row {bad}: {index_a[bad]} vs "
            f"{index_b[bad]} -- the extraction order changed between epochs, "
            f"so rows are not the same image and no paired angle is valid"
        )
    if not np.array_equal(labels_a, labels_b):
        bad = int(np.flatnonzero(labels_a != labels_b)[0])
        raise ValueError(
            f"label mismatch at row {bad}: {labels_a[bad]} vs {labels_b[bad]}"
        )


def hash_indices(index: np.ndarray, labels: np.ndarray) -> str:
    """Stable digest of (index, label) in extraction order.

    Recorded with every run so a later reader can prove two extractions
    compared the same samples in the same order without keeping the arrays.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(index, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def unit_rows(x: np.ndarray, name: str = "features",
              eps: float = 1e-12) -> np.ndarray:
    """L2-normalise rows to float64, rejecting rows with no direction.

    `metrics._l2` adds eps to the denominator, which maps a zero row to a zero
    row and lets it flow onward as a silent 0-degree "no movement". Here that
    is an error: a feature with no direction cannot have an angle.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    norms = np.linalg.norm(x, axis=1)
    bad = np.flatnonzero(norms <= eps)
    if bad.size:
        raise ValueError(
            f"{name}: {bad.size} row(s) have ~zero norm (first at index "
            f"{int(bad[0])}); a zero-norm feature has no direction, so no "
            f"angular displacement is defined for it"
        )
    return x / norms[:, None]


# ----------------------------------------------------------------------
# angles
# ----------------------------------------------------------------------

def _angles_deg_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between rows of two already-unit-normalised arrays, in degrees.

    Uses 2*atan2(|a-b|, |a+b|) rather than arccos(a.b). For unit vectors
    |a-b| = 2 sin(t/2) and |a+b| = 2 cos(t/2), so this is exact, and unlike
    arccos it stays accurate as t -> 0 -- which is precisely the regime the
    epoch-0 gate tests.
    """
    diff = np.linalg.norm(a - b, axis=1)
    summ = np.linalg.norm(a + b, axis=1)
    return np.degrees(2.0 * np.arctan2(diff, summ))


def paired_angles_deg(a_unit: np.ndarray, b_unit: np.ndarray,
                      rotation: Optional[np.ndarray] = None,
                      chunk: int = CHUNK_ROWS) -> np.ndarray:
    """Per-row angle between `a_unit` and `b_unit @ rotation`, in degrees.

    `rotation` is applied to the SECOND argument -- the convention throughout
    this module is that the later (unlearned) features are rotated into the
    earlier (baseline) frame, never the reverse.
    """
    if a_unit.shape != b_unit.shape:
        raise ValueError(
            f"paired arrays must have the same shape, got {a_unit.shape} "
            f"and {b_unit.shape}"
        )
    n = a_unit.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        blk_b = b_unit[s:s + chunk]
        if rotation is not None:
            blk_b = blk_b @ rotation
        out[s:s + chunk] = _angles_deg_unit(a_unit[s:s + chunk], blk_b)
    return out


def summarize_angles(angles: np.ndarray) -> Dict[str, float]:
    """mean / median / p95 / max / n over a vector of angles."""
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan"), "n": 0}
    if not np.isfinite(angles).all():
        raise ValueError("angle vector contains non-finite values")
    return {
        "mean": float(angles.mean()),
        "median": float(np.median(angles)),
        "p95": float(np.percentile(angles, 95)),
        "max": float(angles.max()),
        "n": int(angles.size),
    }


# ----------------------------------------------------------------------
# retain-anchored orthogonal Procrustes
# ----------------------------------------------------------------------

def stratified_anchor_split(labels: np.ndarray, exclude_class: Optional[int],
                            anchor_fraction: float = 0.5,
                            seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic per-class split of the RETAIN rows into anchor / eval.

    `exclude_class` (the forget class) appears in neither side: the rotation
    must not be fitted on the class whose movement is being measured, and the
    retain control must be a held-out set, not the anchors themselves.

    Both returned arrays are sorted, so downstream indexing order does not
    depend on the RNG draw order.
    """
    labels = np.asarray(labels)
    if not 0.0 < anchor_fraction < 1.0:
        raise ValueError(
            f"anchor_fraction must be strictly between 0 and 1, got "
            f"{anchor_fraction}"
        )
    rng = np.random.default_rng(seed)
    anchors, evals = [], []
    for k in np.unique(labels):
        if exclude_class is not None and k == exclude_class:
            continue
        idx = np.flatnonzero(labels == k)
        perm = rng.permutation(idx)
        n_anchor = int(round(anchor_fraction * perm.size))
        # Keep both sides non-empty whenever the class has >= 2 samples.
        if perm.size >= 2:
            n_anchor = max(1, min(perm.size - 1, n_anchor))
        anchors.extend(perm[:n_anchor].tolist())
        evals.extend(perm[n_anchor:].tolist())
    return (np.array(sorted(anchors), dtype=np.int64),
            np.array(sorted(evals), dtype=np.int64))


def _cross_gram(a: np.ndarray, b: np.ndarray, chunk: int = CHUNK_ROWS) -> np.ndarray:
    """a.T @ b accumulated in float64 over row chunks."""
    out = np.zeros((a.shape[1], b.shape[1]), dtype=np.float64)
    for s in range(0, a.shape[0], chunk):
        out += a[s:s + chunk].T @ b[s:s + chunk]
    return out


def orthogonal_procrustes_rotation(src_unit: np.ndarray,
                                   dst_unit: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||src_unit @ R - dst_unit||_F.

    Derivation, because getting the SVD the wrong way round silently returns
    R^T and every angle it produces is wrong but plausible:

        ||S R - D||^2 = tr(S'S) + tr(D'D) - 2 tr(R' S' D)
        maximise tr(R' M) with M = S' D = U Sigma V'
        tr(R' U Sigma V') = tr(V' R' U Sigma) = tr(Z Sigma), Z orthogonal
        <= sum(Sigma), with equality at Z = I, i.e. R = U V'

    `src` is the LATER (unlearned) representation and `dst` the baseline, so
    the returned R maps unlearned features into the baseline frame.

    Reflections are permitted: det(R) is not forced to +1. A reflected
    representation is information-identical to the original, so forcing a
    proper rotation would report a reflection as movement.
    """
    if src_unit.shape != dst_unit.shape:
        raise ValueError(
            f"anchor arrays must have the same shape, got {src_unit.shape} "
            f"and {dst_unit.shape}"
        )
    n, d = src_unit.shape
    if n < d:
        raise ValueError(
            f"insufficient alignment anchors: {n} anchor rows for {d} feature "
            f"dimensions. An orthogonal map in R^{d} is underdetermined by "
            f"fewer than {d} points, so the fitted rotation would absorb real "
            f"movement. Widen the anchor set or reduce the feature dimension."
        )
    m = _cross_gram(src_unit, dst_unit)
    u, _s, vt = np.linalg.svd(m)
    return u @ vt


# ----------------------------------------------------------------------
# class-mean movement
# ----------------------------------------------------------------------

def class_mean_unit(features_unit: np.ndarray, labels: np.ndarray,
                    num_classes: int) -> np.ndarray:
    """Mean of the UNIT features of each class, itself renormalised.

    Rows for absent classes are NaN, matching `metrics.class_means` -- a zero
    row would look like a legitimate direction.

    Grouped by one stable argsort rather than `np.add.at` or a per-class
    boolean mask. `np.add.at` is unbuffered and crawls; a mask per class is
    O(num_classes * n) and would scan 38,800 rows a thousand times over on
    faces-1000. This touches each row once.
    """
    labels = np.asarray(labels)
    d = features_unit.shape[1]
    counts = np.bincount(labels, minlength=num_classes)
    order = np.argsort(labels, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(counts)])

    out = np.full((num_classes, d), np.nan, dtype=np.float64)
    for k in range(num_classes):
        if counts[k] == 0:
            continue
        rows = order[bounds[k]:bounds[k + 1]]
        mu = features_unit[rows].mean(axis=0)
        norm = np.linalg.norm(mu)
        if norm <= 1e-12:
            raise ValueError(
                f"class {k}: the mean of its unit features has ~zero norm; "
                f"its samples point in cancelling directions and no mean "
                f"direction is defined"
            )
        out[k] = mu / norm
    return out


def class_mean_angles_deg(base_unit: np.ndarray, later_unit: np.ndarray,
                          labels: np.ndarray, num_classes: int,
                          rotation: Optional[np.ndarray] = None
                          ) -> np.ndarray:
    """Per-class angle between the baseline and later class-mean directions.

    The rotation is applied to the (num_classes, d) class means, NOT to the
    (n, d) feature matrix. These are identical -- an orthogonal map commutes
    with the mean and preserves the norm, so
    `unit(mean(x_i R)) == unit(mean(x_i)) R` -- but rotating the means avoids
    materialising a second float64 copy of the whole extraction (205 MB at
    CIFAR's 50k x 512). `test_rotate_means_equals_rotate_features` pins the
    equivalence.
    """
    mu0 = class_mean_unit(base_unit, labels, num_classes)
    mu1 = class_mean_unit(later_unit, labels, num_classes)
    if rotation is not None:
        mu1 = mu1 @ rotation
    present = ~(np.isnan(mu0).any(1) | np.isnan(mu1).any(1))
    out = np.full(num_classes, np.nan, dtype=np.float64)
    out[present] = _angles_deg_unit(mu0[present], mu1[present])
    return out


# ----------------------------------------------------------------------
# null control: a retain class treated exactly like the forget class
# ----------------------------------------------------------------------

def select_control_classes(labels: np.ndarray, forget_class: int,
                           n_controls: int, seed: int = 0) -> np.ndarray:
    """Deterministically pick retain classes to use as null controls."""
    retain = np.array([k for k in np.unique(labels) if k != forget_class])
    if retain.size == 0 or n_controls <= 0:
        return np.array([], dtype=retain.dtype)
    n = int(min(n_controls, retain.size))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(retain, size=n, replace=False))


def null_control_angles(base_unit: np.ndarray, later_unit: np.ndarray,
                        labels: np.ndarray, anchor_idx: np.ndarray,
                        control_classes: Sequence[int]) -> Dict[int, Dict]:
    """Displacement of each control class, measured exactly as the forget
    class is: the class is dropped from the anchor set, a fresh rotation is
    fitted on what remains, and every row of the class is then scored.

    This is the null distribution the forget-class number must be read
    against. Without it, the retain-anchored design's own positive bias (see
    the module docstring) is indistinguishable from real forget-class
    movement.
    """
    labels = np.asarray(labels)
    out: Dict[int, Dict] = {}
    anchor_labels = labels[anchor_idx]
    for c in control_classes:
        c = int(c)
        keep = anchor_idx[anchor_labels != c]
        rows = np.flatnonzero(labels == c)
        if rows.size == 0:
            continue
        rot = orthogonal_procrustes_rotation(later_unit[keep], base_unit[keep])
        ang = paired_angles_deg(base_unit[rows], later_unit[rows], rot)
        summary = summarize_angles(ang)
        summary["n_anchor"] = int(keep.size)
        out[c] = summary
    return out


# ----------------------------------------------------------------------
# secondary: linear CKA
# ----------------------------------------------------------------------

def linear_cka(a: np.ndarray, b: np.ndarray, chunk: int = CHUNK_ROWS) -> float:
    """Linear CKA between two representations of the same samples.

    Feature-space form, ||B'A||_F^2 / (||A'A||_F ||B'B||_F), on column-centred
    inputs. The sample-space form would need an (n, n) Gram -- 2.5e9 entries at
    CIFAR's 50k training samples -- so it is not used. The two are algebraically
    identical; only the memory differs.

    SECONDARY METRIC. It is a global average over every class. Movement
    confined to a single forget class can leave it at ~1.0 while the forget
    class has moved substantially, so it must not be read as evidence that the
    representation is unchanged. Report it beside the per-class angles, never
    instead of them.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"CKA needs the same samples in both arrays, got {a.shape[0]} "
            f"and {b.shape[0]} rows"
        )
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    cross = _cross_gram(bc, ac, chunk)
    aa = _cross_gram(ac, ac, chunk)
    bb = _cross_gram(bc, bc, chunk)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom <= 0:
        return float("nan")
    return float((cross ** 2).sum() / denom)


# ----------------------------------------------------------------------
# bundle
# ----------------------------------------------------------------------

def movement_report(base_feats: np.ndarray, later_feats: np.ndarray,
                    labels: np.ndarray, num_classes: int, forget_class: int,
                    anchor_idx: np.ndarray, eval_idx: np.ndarray,
                    control_classes: Optional[Sequence[int]] = None) -> Dict:
    """Every movement quantity for one (baseline, later) pair.

    `anchor_idx` and `eval_idx` must be disjoint retain rows from
    `stratified_anchor_split`; that disjointness is asserted here rather than
    assumed, because an anchor leak would quietly zero the retain control and
    inflate forget-minus-retain.
    """
    base_unit = unit_rows(base_feats, "baseline features")
    later_unit = unit_rows(later_feats, "later features")
    if base_unit.shape != later_unit.shape:
        raise ValueError(
            f"feature shape mismatch: {base_unit.shape} vs {later_unit.shape}"
        )
    labels = np.asarray(labels)
    forget_idx = np.flatnonzero(labels == forget_class)
    if forget_idx.size == 0:
        raise ValueError(f"forget class {forget_class} has no samples")
    if np.intersect1d(anchor_idx, eval_idx).size:
        raise ValueError(
            "alignment anchors overlap the retain evaluation set -- the "
            "rotation would be fitted on the rows it is scored against"
        )
    for name, idx in (("anchor", anchor_idx), ("retain-eval", eval_idx)):
        if np.intersect1d(idx, forget_idx).size:
            raise ValueError(f"{name} set contains forget-class rows")

    rotation = orthogonal_procrustes_rotation(later_unit[anchor_idx],
                                              base_unit[anchor_idx])
    # Orthogonality is cheap to verify and expensive to get wrong.
    d = rotation.shape[0]
    orth_err = float(np.abs(rotation.T @ rotation - np.eye(d)).max())

    raw = paired_angles_deg(base_unit, later_unit)
    aligned = paired_angles_deg(base_unit, later_unit, rotation)
    cm_aligned = class_mean_angles_deg(base_unit, later_unit, labels,
                                       num_classes, rotation)
    retain_mask = np.ones(num_classes, dtype=bool)
    retain_mask[forget_class] = False
    cm_retain = cm_aligned[retain_mask]
    cm_retain = cm_retain[~np.isnan(cm_retain)]

    raw_f = summarize_angles(raw[forget_idx])
    raw_r = summarize_angles(raw[eval_idx])
    ali_f = summarize_angles(aligned[forget_idx])
    ali_r = summarize_angles(aligned[eval_idx])

    nulls = (null_control_angles(base_unit, later_unit, labels, anchor_idx,
                                 control_classes)
             if control_classes is not None else {})
    null_means = np.array([v["mean"] for v in nulls.values()], dtype=np.float64)

    return {
        "null_control_per_class": {str(k): v for k, v in nulls.items()},
        "null_control_mean_deg": (float(null_means.mean())
                                  if null_means.size else float("nan")),
        "null_control_max_deg": (float(null_means.max())
                                 if null_means.size else float("nan")),
        "null_control_min_deg": (float(null_means.min())
                                 if null_means.size else float("nan")),
        "null_control_n": int(null_means.size),
        # The headline contrast: forget-class displacement measured against
        # classes treated identically but never unlearned. Positive means the
        # forget class moved MORE than the design's own bias explains.
        "aligned_forget_minus_null_mean": (
            float(ali_f["mean"] - null_means.mean())
            if null_means.size else float("nan")),
        "aligned_forget_exceeds_null_max": (
            bool(ali_f["mean"] > null_means.max())
            if null_means.size else False),
        "raw_forget": raw_f,
        "raw_retain_eval": raw_r,
        "raw_forget_minus_retain_mean": raw_f["mean"] - raw_r["mean"],
        "aligned_forget": ali_f,
        "aligned_retain_eval": ali_r,
        "aligned_forget_minus_retain_mean": ali_f["mean"] - ali_r["mean"],
        "aligned_forget_minus_retain_median": ali_f["median"] - ali_r["median"],
        "aligned_forget_minus_retain_p95": ali_f["p95"] - ali_r["p95"],
        "aligned_class_mean_forget_deg": float(cm_aligned[forget_class]),
        "aligned_class_mean_retain_macro_deg": (
            float(cm_retain.mean()) if cm_retain.size else float("nan")),
        "aligned_class_mean_retain_max_deg": (
            float(cm_retain.max()) if cm_retain.size else float("nan")),
        "aligned_class_mean_forget_minus_retain_deg": (
            float(cm_aligned[forget_class] - cm_retain.mean())
            if cm_retain.size else float("nan")),
        "cka_linear_secondary": linear_cka(base_unit, later_unit),
        "n_anchor": int(anchor_idx.size),
        "n_retain_eval": int(eval_idx.size),
        "n_forget": int(forget_idx.size),
        "rotation_orthogonality_err": orth_err,
        "rotation_det": float(np.linalg.det(rotation)),
    }
