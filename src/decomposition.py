"""
Centred-NC3 reference-frame decomposition.

WHAT THIS ANSWERS
-----------------
`nc3_centred_forget` is a cosine between two CENTRED vectors:

    A(w, c) = cos( f_0 - g_0 ,  w - c )

`w` is the forget class's classifier weight and `c` is the mean of the
RETAINED classes' weights -- the reference frame that centring subtracts.
Both sides of that subtraction can move, so a change in the metric is not by
itself a statement about the forget-class weight. This module separates the
two by evaluating the SAME production expression at four (weight, centre)
pairs and reporting the exact two-factor residual.

NOT A CAUSAL DECOMPOSITION
--------------------------
`A` normalises both arguments, so it is not linear in either. The
counterfactual pairs below are DESCRIPTIVE INTERVENTIONS ON THE METRIC:
"what would the production number have read if the reference frame had not
moved". They are not additive causal contributions and the residual is not an
interaction effect in the statistical sense -- it is the exact arithmetic
leftover of a nonlinear function evaluated at four corners.

SIGN AND REPRESENTATION CONVENTION -- taken from production, not reinvented
--------------------------------------------------------------------------
`train.evaluate_light` and `train.evaluate` both pass `head.weight` to
`metrics.nc3_alignment` RAW -- `fc.weight` for `LinearHead` (its bias is not
consulted) and `W` for `ArcFaceHead`/`CosFaceHead`, neither row-normalised
before the call. `nc3_alignment` then normalises inside the cosine. So the
"effective weight" here is the raw weight row, and the normalisation happens
where production puts it. `_l2` is IMPORTED from `metrics` rather than
retyped, including its `1e-12` denominator guard, so a change there cannot
silently desynchronise this file from the metric it decomposes.

The centring reference is `metrics.nc3_alignment`'s own: classes PRESENT in
the feature set, minus the forget class (`exclude_from_centre`). The feature
side uses `np.nanmean` over that mask and the weight side uses `.mean`, which
is what production does -- class means can be NaN for absent classes, weights
cannot.

FROZEN BACKBONE
---------------
Every cell this is written for is `classifier_only`, so `f_0` and `g_0` are
constant across epochs by construction. That is VERIFIED per cell rather than
assumed (`feature_state_fingerprint`), because "the backbone was frozen" is a
claim about the code path, and the decomposition's whole meaning depends on
the feature side not having moved.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from metrics import _l2, class_means


# ----------------------------------------------------------------------
# the production reference frame
# ----------------------------------------------------------------------

def present_mask(mu: np.ndarray) -> np.ndarray:
    """Classes with a defined class mean -- `metrics.nc3_alignment`'s `present`."""
    return ~np.isnan(mu).any(axis=1)


def centre_mask(mu: np.ndarray, forget_class: Optional[int]) -> np.ndarray:
    """`nc3_alignment`'s centring reference: present classes minus the forget
    class, with its own degenerate fallback to all-present."""
    present = present_mask(mu)
    ref = present.copy()
    if forget_class is not None and 0 <= forget_class < mu.shape[0]:
        ref[forget_class] = False
    if not ref.any():
        ref = present
    return ref


def weight_centre(weight: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`c`: the retain-weight centre centred NC3 subtracts."""
    return weight[ref].mean(axis=0)


def feature_centre(mu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`g`: the feature-side centre. Constant under a frozen backbone."""
    return np.nanmean(mu[ref], axis=0)


# ----------------------------------------------------------------------
# the two production alignments, as functions of (w, c)
# ----------------------------------------------------------------------

def centred_alignment(f: np.ndarray, g: np.ndarray,
                      w: np.ndarray, c: np.ndarray) -> float:
    """A(w, c) = cos(f - g, w - c). Identical arithmetic to
    `metrics.nc3_alignment(..., centre=True)` for the forget row."""
    return float(np.dot(_l2(f - g), _l2(w - c)))


def uncentred_alignment(f: np.ndarray, w: np.ndarray) -> float:
    """U(w) = cos(f, w). Identical to `nc3_alignment(..., centre=False)`."""
    return float(np.dot(_l2(f), _l2(w)))


# ----------------------------------------------------------------------
# movement descriptors
# ----------------------------------------------------------------------

def movement(v0: np.ndarray, vt: np.ndarray) -> Dict[str, float]:
    """Angular and magnitude movement of one vector from its epoch-0 value.

    `angle_deg` is the angle between the two, via the same `_l2` used
    everywhere else; the cosine is clipped only for the arccos, so
    `cosine` itself is reported unclipped.
    """
    cos = float(np.dot(_l2(v0), _l2(vt)))
    n0, nt = float(np.linalg.norm(v0)), float(np.linalg.norm(vt))
    return {
        "cosine_to_epoch0": cos,
        "angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
        "norm_epoch0": n0,
        "norm": nt,
        "norm_ratio": float(nt / n0) if n0 > 0 else float("nan"),
        "norm_change": float(nt - n0),
    }


def feature_state_fingerprint(mu: np.ndarray) -> str:
    """Exact-bytes fingerprint of the class-mean matrix.

    Used to assert the feature side did not move. A hash, not a tolerance:
    under a frozen backbone the extracted features are the same floats, so
    anything other than exact equality means the run was not what it claimed
    to be.
    """
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(mu, dtype=np.float64)
                          .tobytes()).hexdigest()


# ----------------------------------------------------------------------
# the decomposition
# ----------------------------------------------------------------------

def decompose_cell(mu: np.ndarray, weight_0: np.ndarray, weight_t: np.ndarray,
                   forget_class: int) -> Dict[str, float]:
    """
    Every quantity the reference-frame decomposition needs, for one epoch `t`
    of one cell, against that cell's own epoch 0.

    `mu` is the class-mean matrix -- the SAME one at both epochs, since the
    backbone is frozen; pass epoch t's and check it against epoch 0's
    separately (`feature_state_fingerprint`).

    Returns the four corners, both production values, the exact residual, and
    the movement of `w`, `c` and `w - c`.
    """
    ref = centre_mask(mu, forget_class)
    f0 = mu[forget_class]
    g0 = feature_centre(mu, ref)

    w0, wt = weight_0[forget_class], weight_t[forget_class]
    c0, ct = weight_centre(weight_0, ref), weight_centre(weight_t, ref)

    A_tt = centred_alignment(f0, g0, wt, ct)      # production, epoch t
    A_t0 = centred_alignment(f0, g0, wt, c0)      # current weight, epoch-0 centre
    A_0t = centred_alignment(f0, g0, w0, ct)      # epoch-0 weight, current centre
    A_00 = centred_alignment(f0, g0, w0, c0)      # baseline

    return {
        "n_centre_classes": int(ref.sum()),
        "n_present_classes": int(present_mask(mu).sum()),

        # production values, exactly as the metric computes them
        "nc3_centred_forget": A_tt,
        "nc3_centred_forget_epoch0": A_00,
        "nc3_uncentred_forget": uncentred_alignment(f0, wt),
        "nc3_uncentred_forget_epoch0": uncentred_alignment(f0, w0),

        # the four corners of the (weight, centre) square
        "A_wt_ct": A_tt,
        "A_wt_c0": A_t0,
        "A_w0_ct": A_0t,
        "A_w0_c0": A_00,

        # changes from baseline
        "delta_centred_production": A_tt - A_00,
        "delta_centred_weight_only": A_t0 - A_00,     # centre frozen at epoch 0
        "delta_centred_centre_only": A_0t - A_00,     # weight frozen at epoch 0
        "delta_uncentred_production": (uncentred_alignment(f0, wt)
                                       - uncentred_alignment(f0, w0)),

        # exact two-factor residual of the nonlinear metric
        "interaction_residual": A_tt - A_t0 - A_0t + A_00,

        # movement of each moving part
        "forget_weight": movement(w0, wt),
        "retain_centre": movement(c0, ct),
        "centred_forget_weight": movement(w0 - c0, wt - ct),

        "feature_mean_fingerprint": feature_state_fingerprint(mu),
    }


def class_mean_matrix(features: np.ndarray, labels: np.ndarray,
                      num_classes: int) -> np.ndarray:
    """`metrics.class_means`, re-exported so callers import one definition."""
    return class_means(features, labels, num_classes)
