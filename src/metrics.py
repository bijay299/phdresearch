"""
Measurement instruments.

Everything here operates on FROZEN features. Nothing in this file trains
the backbone. That separation is the whole point: we are asking what is
present in the representation, independently of what the classifier says.

Four levels of reading, weakest to strongest evidence of real forgetting:

    1. output_accuracy      what the full network predicts   (the standard metric)
    2. linear_probe         freeze features, fit a new linear layer
    3. ncc_accuracy         nearest class centre, trains nothing
    4. verification_auc     embeddings only, no classifier at all  (deployment)

Plus nc1/nc2/nc3, which describe the geometry rather than the accuracy.
nc3 is the PRIMARY OUTCOME of the project: it measures how far each class's
classifier weight has drifted from its class-mean feature.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _l2(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def class_means(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class mean feature. Rows for absent classes are NaN, not zero --
    a zero row would silently look like a legitimate direction."""
    d = features.shape[1]
    mu = np.full((num_classes, d), np.nan, dtype=np.float64)
    for k in range(num_classes):
        m = labels == k
        if m.any():
            mu[k] = features[m].mean(axis=0)
    return mu


# ----------------------------------------------------------------------
# neural collapse geometry
# ----------------------------------------------------------------------

def nc1_within_class_variability(features: np.ndarray, labels: np.ndarray,
                                 num_classes: int) -> float:
    """
    Within-class scatter relative to between-class scatter.
    Lower => tighter collapse of each class onto its mean.
    """
    mu = class_means(features, labels, num_classes)
    present = ~np.isnan(mu).any(axis=1)
    mu_g = np.nanmean(mu[present], axis=0)

    within, n = 0.0, 0
    for k in np.where(present)[0]:
        m = labels == k
        diff = features[m] - mu[k]
        within += (diff ** 2).sum()
        n += m.sum()
    within /= max(n, 1)

    between = ((mu[present] - mu_g) ** 2).sum() / max(present.sum(), 1)
    return float(within / (between + 1e-12))


def nc1_angular(features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Angular (L2-normalised) variant of nc1_within_class_variability.

    Raw NC1 is a ratio of squared-distance scatters, so it moves if a
    sample's feature magnitude changes even when its direction does not.
    Margin-based heads (ArcFace/CosFace) constrain direction only -- they
    normalise both features and weights before comparing them -- so a
    magnitude-sensitive NC1 can show "collapse" that is really just norm
    growth. Normalising every feature to unit length before computing the
    same ratio removes that sensitivity; what is left is variability in
    direction alone, which is what these heads can actually be judged on.

    Report alongside raw nc1, the same way nc3 is reported both centred and
    uncentred: they answer different questions and neither replaces the
    other.
    """
    return nc1_within_class_variability(_l2(features), labels, num_classes)


def nc2_simplex_etf(features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Deviation of centred, normalised class means from a simplex ETF.
    Under perfect NC2 every off-diagonal cosine equals -1/(K-1).
    Returns mean absolute deviation. Lower => closer to ETF.
    """
    mu = class_means(features, labels, num_classes)
    present = ~np.isnan(mu).any(axis=1)
    M = mu[present]
    K = M.shape[0]
    if K < 2:
        return float("nan")

    M = _l2(M - M.mean(axis=0, keepdims=True))
    G = M @ M.T
    off = G[~np.eye(K, dtype=bool)]
    return float(np.abs(off - (-1.0 / (K - 1))).mean())


def nc3_alignment(features: np.ndarray, labels: np.ndarray, weight: np.ndarray,
                  num_classes: int, centre: bool = True,
                  exclude_from_centre: Optional[int] = None) -> Dict[str, float]:
    """
    *** PRIMARY OUTCOME OF THIS PROJECT ***

    Per-class alignment between the classifier weight and the class-mean
    feature, measured as cosine similarity after centring.

    Under neural collapse this is ~1 for every class. The AISTATS paper's
    finding is that after unlearning it stays ~1 for retained classes and
    collapses (or goes negative) for the forgotten class -- that is
    "feature-classifier misalignment".

    Their theory predicts w_k -> -(1-gamma) * mu_k for the forgotten class,
    i.e. cosine -> -1. Watch for negative values, not just small ones.

    CENTRING CONTAMINATION -- read this before using the numbers
    ------------------------------------------------------------
    Standard NC3 centres weights by the global weight mean. If unlearning
    flips one class's weight, that flip moves the global mean, which shifts
    the centring for EVERY class. Retained classes then appear misaligned
    when nothing happened to them.

    Measured on synthetic data with only class 0 perturbed, global centring
    dragged the retain-class mean from 1.00 down to 0.94 -- a 6-point
    artefact that would look like real spillover in a results table.

    Pass `exclude_from_centre=<forget class>` to compute the centring
    reference from retained classes only. Do this whenever a forget class
    exists. The default (None) reproduces the standard definition.

    Returns per-class cosines plus summary stats.
    """
    mu = class_means(features, labels, num_classes)
    present = ~np.isnan(mu).any(axis=1)

    if centre:
        ref = present.copy()
        if exclude_from_centre is not None and 0 <= exclude_from_centre < num_classes:
            ref[exclude_from_centre] = False
        if not ref.any():                      # degenerate: fall back to all present
            ref = present

        mu_c = mu - np.nanmean(mu[ref], axis=0)
        w_c = weight - weight[ref].mean(axis=0, keepdims=True)
    else:
        mu_c, w_c = mu, weight

    cos = np.full(num_classes, np.nan)
    for k in np.where(present)[0]:
        a, b = _l2(mu_c[k]), _l2(w_c[k])
        cos[k] = float(np.dot(a, b))

    valid = cos[~np.isnan(cos)]
    out = {
        "per_class": cos,
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "min": float(valid.min()) if valid.size else float("nan"),
    }
    if exclude_from_centre is not None:
        retain = np.delete(cos, exclude_from_centre)
        retain = retain[~np.isnan(retain)]
        out["forget"] = float(cos[exclude_from_centre])
        out["retain_mean"] = float(retain.mean()) if retain.size else float("nan")
    return out


# ----------------------------------------------------------------------
# recoverability
# ----------------------------------------------------------------------

def linear_probe(train_feats: np.ndarray, train_labels: np.ndarray,
                 test_feats: np.ndarray, test_labels: np.ndarray,
                 target_class: Optional[int] = None,
                 max_iter: int = 2000, seed: int = 0) -> Dict[str, float]:
    """
    Freeze the features, fit a fresh linear classifier, measure how much of
    the forgotten class comes back.

    IMPORTANT -- interpreting the number:
    A model retrained from scratch WITHOUT the forgotten class still scores
    high here, because deep features transfer. In the AISTATS CIFAR-10 setting
    the retrained reference scored 77.35 on a class it never saw.

    So absolute probe accuracy is not evidence of failed unlearning.
    Always report the gap against a reference. See probe_gap().
    """
    # NOTE: `multi_class` was removed in scikit-learn 1.7+; multinomial is
    # now the default for multiclass problems. Do not re-add it.
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(train_feats, train_labels)
    pred = clf.predict(test_feats)

    out = {"overall": float((pred == test_labels).mean())}
    if target_class is not None:
        m = test_labels == target_class
        out["forget"] = float((pred[m] == target_class).mean()) if m.any() else float("nan")
        out["retain"] = float((pred[~m] == test_labels[~m]).mean()) if (~m).any() else float("nan")
    return out


def probe_gap(method_probe_forget: float, reference_probe_forget: float) -> float:
    """
    Excess recoverability over a reference model.

    Positive  => more of the class is recoverable than in the reference
    Near zero => as unrecoverable as the reference
    Negative  => less recoverable than the reference

    NOTE: this is a derived quantity, not something the AISTATS authors
    report. Label it as such in any table or slide.
    """
    return float(method_probe_forget - reference_probe_forget)


def ncc_accuracy(train_feats: np.ndarray, train_labels: np.ndarray,
                 test_feats: np.ndarray, test_labels: np.ndarray,
                 num_classes: int, target_class: Optional[int] = None) -> Dict[str, float]:
    """
    Nearest class centre. Trains nothing at all -- assigns each test sample
    to the closest class mean. Stricter than the probe: it asks whether the
    forgotten class still occupies its own region of feature space.
    """
    mu = class_means(train_feats, train_labels, num_classes)
    present = np.where(~np.isnan(mu).any(axis=1))[0]
    M = mu[present]

    d = ((test_feats[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    pred = present[d.argmin(axis=1)]

    out = {"overall": float((pred == test_labels).mean())}
    if target_class is not None:
        m = test_labels == target_class
        out["forget"] = float((pred[m] == target_class).mean()) if m.any() else float("nan")
        out["retain"] = float((pred[~m] == test_labels[~m]).mean()) if (~m).any() else float("nan")
    return out


def verification_auc(features: np.ndarray, labels: np.ndarray,
                     target_class: Optional[int] = None,
                     num_pairs: int = 20000, seed: int = 0) -> Dict[str, float]:
    """
    *** THE DEPLOYMENT METRIC -- not in the AISTATS paper ***

    Face recognition trains a classification head and then discards it.
    Verification is a cosine comparison between two embeddings.

    So if unlearning works by moving the classifier, this number should be
    UNAFFECTED -- the modified component never runs at deployment. That is
    the sharpest version of the project's argument, and it needs its own
    measurement.

    Builds balanced same/different identity pairs and reports ROC-AUC.
    When target_class is given, also reports AUC restricted to pairs
    involving the forgotten identity.
    """
    rng = np.random.default_rng(seed)
    X = _l2(features)

    by_class = {k: np.where(labels == k)[0] for k in np.unique(labels)}
    eligible = [k for k, idx in by_class.items() if len(idx) >= 2]
    if len(eligible) < 2:
        return {"auc": float("nan")}

    a_idx, b_idx, same = [], [], []
    for _ in range(num_pairs // 2):
        k = rng.choice(eligible)
        i, j = rng.choice(by_class[k], size=2, replace=False)
        a_idx.append(i); b_idx.append(j); same.append(1)

        k1, k2 = rng.choice(eligible, size=2, replace=False)
        a_idx.append(rng.choice(by_class[k1]))
        b_idx.append(rng.choice(by_class[k2]))
        same.append(0)

    a_idx = np.array(a_idx); b_idx = np.array(b_idx); same = np.array(same)
    scores = (X[a_idx] * X[b_idx]).sum(-1)

    out = {"auc": float(roc_auc_score(same, scores))}
    if target_class is not None:
        m = (labels[a_idx] == target_class) | (labels[b_idx] == target_class)
        if m.sum() > 10 and len(np.unique(same[m])) == 2:
            out["auc_forget"] = float(roc_auc_score(same[m], scores[m]))
        else:
            out["auc_forget"] = float("nan")
    return out


# ----------------------------------------------------------------------
# bundle
# ----------------------------------------------------------------------

@dataclass
class Report:
    """One row of the results table."""
    head: str
    method: str
    forget_class: int
    seed: int
    output_forget: float = float("nan")
    output_retain: float = float("nan")
    probe_forget: float = float("nan")
    probe_retain: float = float("nan")
    ncc_forget: float = float("nan")
    ncc_retain: float = float("nan")
    verif_auc: float = float("nan")
    verif_auc_forget: float = float("nan")
    nc1: float = float("nan")
    nc1_angular: float = float("nan")
    nc2: float = float("nan")
    nc3_forget: float = float("nan")
    nc3_retain_mean: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)
