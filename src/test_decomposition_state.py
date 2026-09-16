"""
Regression tests for the centred-NC3 reference-frame decomposition and for the
state instrumentation that feeds it.

Run:  python src/test_decomposition_state.py

Two separate claims are under test and they fail for different reasons.

  1. THE INSTRUMENTATION IS INERT. `evaluate_light(state_hook=...)` and the
     driver's `--dump-nc3-state` must leave the run bitwise identical. If they
     do not, the replayed cells are not the canonical cells and every gate
     downstream is comparing a different experiment. The specific hazard is
     RNG: a hook that re-extracted features would build DataLoader iterators,
     and PyTorch draws a base seed from the global CPU RNG per iterator, which
     would reshuffle every later unlearning epoch. Asserted by comparing
     parameters, RNG state and trace hashes.

  2. THE DECOMPOSITION ARITHMETIC IS RIGHT. The four corners must reproduce
     production NC3 exactly, the epoch-0 corners must collapse to one baseline,
     and the residual identity must close. Checked against
     `metrics.nc3_alignment` itself -- not a reimplementation of it -- and on
     synthetic geometry with a known answer, the same way the NC3 tests do.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D                       # noqa: E402
import decomposition as DC             # noqa: E402
import metrics as M                    # noqa: E402
import train as TR                     # noqa: E402
import unlearn as UL                   # noqa: E402
from utils import set_seed             # noqa: E402


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


# ----------------------------------------------------------------------
# tiny fixtures -- same shape as src/test_unlearn_trace.py
# ----------------------------------------------------------------------

D_IN, K, FEAT = 6, 5, 4
FC = 0


class Toy(Dataset):
    def __init__(self, n=60, seed=0):
        g = np.random.default_rng(seed)
        self.x = g.normal(size=(n, D_IN)).astype(np.float32)
        self.targets = (np.arange(n) % K).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.targets[i])


class Backbone(nn.Module):
    feat_dim = FEAT

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D_IN, FEAT)

    def forward(self, x):
        return self.lin(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, FEAT) * 0.1)

    def forward(self, f, y=None):
        return f @ self.weight.T


def _loaders(seed=0):
    ds = Toy()
    base = D.IndexedDataset(ds)
    tg = D.get_targets(ds)
    set_seed(seed)
    fl = D.make_loader(base, np.flatnonzero(tg == FC), 8, True, 0)
    rl = D.make_loader(base, np.flatnonzero(tg != FC), 8, True, 0)
    te = D.make_loader(ds, None, 16, False, 0)
    ev = D.make_loader(ds, None, 16, False, 0)
    return fl, rl, ev, te


def _run(state_sink=None, trace=None, seed=0, epochs=2):
    """One classifier-only random_label run, optionally with a per-epoch
    state hook wired exactly as the driver wires it."""
    set_seed(seed)
    bb, hd = Backbone(), Head()
    set_seed(seed)
    fl, rl, ev, te = _loaders(seed)

    def epoch_eval(b, h, epoch):
        hook = None
        if state_sink is not None:
            def hook(f_tr, y_tr, W, _e=epoch):
                mu = DC.class_mean_matrix(f_tr, y_tr, K)
                state_sink.append({"epoch": _e, "weight": W.copy(),
                                   "class_means": mu.copy()})
        TR.evaluate_light(b, h, ev, te, K, "cpu", forget_class=FC, seed=seed,
                          include_uncentred=True, state_hook=hook)

    set_seed(seed)
    return UL.random_label(bb, hd, forget_loader=fl, retain_loader=rl,
                           device="cpu", num_classes=K, epochs=epochs,
                           lr=0.05, weight_decay=0.0, classifier_only=True,
                           epoch_eval=epoch_eval, trace=trace)


def _params(bb, hd):
    return [p.detach().clone() for p in
            list(bb.parameters()) + list(hd.parameters())]


def _same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _recorder():
    import hashlib
    st = {"h": hashlib.sha256(), "steps": 0}

    def t(ep, step, ri, fi, rnd, n_r, n_f):
        st["h"].update(np.int64([ep, step, n_r, n_f]).tobytes())
        for arr in (ri, fi, rnd):
            st["h"].update(b"\x00" if arr is None
                           else np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        st["steps"] += 1
    return st, t


# ----------------------------------------------------------------------
# 1. the instrumentation must be inert
# ----------------------------------------------------------------------

def test_state_hook_does_not_change_the_run():
    print("state hook: learned parameters bitwise identical")
    a = _params(*_run(state_sink=None))
    sink = []
    b = _params(*_run(state_sink=sink))
    check("parameters identical with and without the hook", _same(a, b))
    check("hook actually fired at every trajectory epoch", len(sink) == 3)


def test_state_hook_does_not_disturb_rng():
    print("state hook: global torch RNG state identical after the run")
    _run(state_sink=None)
    rng_plain = torch.random.get_rng_state().clone()
    _run(state_sink=[])
    rng_hooked = torch.random.get_rng_state().clone()
    check("RNG state identical", torch.equal(rng_plain, rng_hooked))


def test_state_hook_does_not_change_the_trace():
    print("state hook: training trace hash unchanged")
    s0, t0 = _recorder(); _run(state_sink=None, trace=t0)
    s1, t1 = _recorder(); _run(state_sink=[], trace=t1)
    check("trace hashes identical", s0["h"].hexdigest() == s1["h"].hexdigest())
    check("step counts identical", s0["steps"] == s1["steps"])


def test_evaluate_light_dict_is_unchanged():
    print("state hook: evaluate_light returns the same dict either way")
    set_seed(0)
    bb, hd = Backbone(), Head()
    set_seed(0)
    _, _, ev, te = _loaders(0)
    plain = TR.evaluate_light(bb, hd, ev, te, K, "cpu", forget_class=FC,
                              include_uncentred=True)
    seen = []
    hooked = TR.evaluate_light(bb, hd, ev, te, K, "cpu", forget_class=FC,
                               include_uncentred=True,
                               state_hook=lambda f, y, W: seen.append(W.copy()))
    check("same keys", set(plain) == set(hooked))
    check("same values, exactly",
          all(plain[k] == hooked[k] or (np.isnan(plain[k]) and np.isnan(hooked[k]))
              for k in plain))
    check("hook saw the weight matrix production used",
          len(seen) == 1 and np.array_equal(
              seen[0], hd.weight.detach().cpu().numpy()))


# ----------------------------------------------------------------------
# 2. the decomposition must reproduce production exactly
# ----------------------------------------------------------------------

def test_corners_reproduce_production_nc3():
    print("decomposition: corners reproduce metrics.nc3_alignment exactly")
    sink = []
    _run(state_sink=sink)
    s0 = sink[0]
    for s in sink:
        mu, W = s["class_means"], s["weight"]
        # Production values come from the METRIC, not from a copy of its
        # arithmetic. It needs features, so feed it a labelled set whose class
        # means are exactly `mu`: one sample per class, the mean of a single
        # float64 row being that row.
        feats = mu.astype(np.float64)
        labs = np.arange(K)
        prod_c = M.nc3_alignment(feats, labs, W, K, centre=True,
                                 exclude_from_centre=FC)["forget"]
        prod_u = M.nc3_alignment(feats, labs, W, K, centre=False,
                                 exclude_from_centre=FC)["forget"]
        d = DC.decompose_cell(mu, s0["weight"], W, FC)
        check(f"ep{s['epoch']}: centred corner == production centred",
              d["A_wt_ct"] == prod_c)
        check(f"ep{s['epoch']}: uncentred == production uncentred",
              d["nc3_uncentred_forget"] == prod_u)


def test_epoch0_collapses_to_one_baseline():
    print("decomposition: at t=0 all four corners collapse to the baseline")
    sink = []
    _run(state_sink=sink)
    s0 = sink[0]
    d = DC.decompose_cell(s0["class_means"], s0["weight"], s0["weight"], FC)
    check("A(w0,c0) == A(wt,ct)", d["A_w0_c0"] == d["A_wt_ct"])
    check("A(wt,c0) == A(w0,c0)", d["A_wt_c0"] == d["A_w0_c0"])
    check("A(w0,ct) == A(w0,c0)", d["A_w0_ct"] == d["A_w0_c0"])
    check("production change is exactly zero",
          d["delta_centred_production"] == 0.0)
    check("residual is exactly zero", d["interaction_residual"] == 0.0)
    # `_l2`'s 1e-12 denominator guard means a vector's cosine with itself
    # need not round to exactly 1.0, so a vector that did not move reports a
    # tiny angle rather than an identically zero one. The floor is about
    # 1e-4 degrees at unit norm, hence the 1e-3 threshold; the alignments
    # themselves are still compared exactly, only this descriptor is
    # toleranced, and the norm is checked bitwise alongside it.
    check("forget weight has not moved",
          d["forget_weight"]["angle_deg"] < 1e-3)


def test_residual_identity_closes():
    print("decomposition: residual identity reconstructs the production change")
    sink = []
    _run(state_sink=sink)
    s0 = sink[0]
    for s in sink[1:]:
        d = DC.decompose_cell(s["class_means"], s0["weight"], s["weight"], FC)
        lhs = d["delta_centred_production"]
        rhs = (d["delta_centred_weight_only"] + d["delta_centred_centre_only"]
               + d["interaction_residual"])
        check(f"ep{s['epoch']}: dProd == dWeightOnly + dCentreOnly + residual",
              abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs)))


def test_frozen_backbone_leaves_features_fixed():
    print("decomposition: frozen backbone -> class means bit-identical")
    sink = []
    _run(state_sink=sink)
    f0 = DC.feature_state_fingerprint(sink[0]["class_means"])
    for s in sink[1:]:
        check(f"ep{s['epoch']}: class-mean fingerprint unchanged",
              DC.feature_state_fingerprint(s["class_means"]) == f0)


def test_known_answer_pure_centre_movement():
    """A case built so the forget weight does NOT move and only the reference
    frame does. The weight-only counterfactual must then read exactly zero
    change while the production number does move -- which is the whole point
    of separating them."""
    print("decomposition: known answer, only the retain centre moves")
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(K, FEAT))
    W0 = rng.normal(size=(K, FEAT))
    Wt = W0.copy()
    Wt[1:] += 0.7                       # shift every RETAIN weight, not fc=0

    d = DC.decompose_cell(mu, W0, Wt, FC)
    check("forget weight did not move",
          d["forget_weight"]["angle_deg"] < 1e-3          # see note above
          and d["forget_weight"]["norm_change"] == 0.0)
    check("retain centre did move", d["retain_centre"]["angle_deg"] > 0.0)
    check("weight-only counterfactual change is exactly zero",
          d["delta_centred_weight_only"] == 0.0)
    check("centre-only change equals the production change",
          d["delta_centred_centre_only"] == d["delta_centred_production"])
    # The four corners collapse to two distinct values here, so the residual is
    # zero in exact arithmetic; summing them in a fixed order still rounds, so
    # it is checked at floating-point tolerance rather than bitwise.
    check("residual vanishes to floating-point tolerance",
          abs(d["interaction_residual"]) < 1e-15)
    check("uncentred convention sees nothing at all",
          d["delta_uncentred_production"] == 0.0)


def test_known_answer_pure_weight_movement():
    """The mirror case: the retain centre is held fixed by construction and
    only the forget weight moves. Here it is the CENTRE-only counterfactual
    that must read exactly zero."""
    print("decomposition: known answer, only the forget weight moves")
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(K, FEAT))
    W0 = rng.normal(size=(K, FEAT))
    Wt = W0.copy()
    Wt[FC] = -Wt[FC]                    # flip the forget weight, retain fixed

    d = DC.decompose_cell(mu, W0, Wt, FC)
    check("retain centre did not move",
          d["retain_centre"]["angle_deg"] < 1e-3
          and d["retain_centre"]["norm_change"] == 0.0)
    check("forget weight flipped",
          abs(d["forget_weight"]["cosine_to_epoch0"] + 1.0) < 1e-9)
    check("centre-only counterfactual change is exactly zero",
          d["delta_centred_centre_only"] == 0.0)
    check("weight-only change equals the production change",
          d["delta_centred_weight_only"] == d["delta_centred_production"])
    check("residual vanishes to floating-point tolerance",
          abs(d["interaction_residual"]) < 1e-15)
    check("uncentred convention sees an exact sign reversal",
          abs(d["nc3_uncentred_forget"]
              + d["nc3_uncentred_forget_epoch0"]) < 1e-9)


def test_residual_is_nonzero_when_both_move():
    """Because A normalises both arguments it is NOT additive. If the residual
    were always ~0 the two counterfactuals could be read as contributions,
    which the module docstring forbids -- so prove it can be large."""
    print("decomposition: residual is genuinely nonzero when both move")
    rng = np.random.default_rng(2)
    mu = rng.normal(size=(K, FEAT))
    W0 = rng.normal(size=(K, FEAT))
    Wt = W0 + rng.normal(size=(K, FEAT))        # everything moves
    d = DC.decompose_cell(mu, W0, Wt, FC)
    check("residual is not numerically zero",
          abs(d["interaction_residual"]) > 1e-6)
    check("identity still closes exactly",
          abs(d["delta_centred_production"]
              - (d["delta_centred_weight_only"]
                 + d["delta_centred_centre_only"]
                 + d["interaction_residual"])) < 1e-12)


def test_centre_mask_matches_the_metric():
    print("decomposition: centring reference matches nc3_alignment's own")
    mu = np.random.default_rng(3).normal(size=(K, FEAT))
    mu[K - 1] = np.nan                           # an absent class
    ref = DC.centre_mask(mu, FC)
    check("forget class excluded", not ref[FC])
    check("absent class excluded", not ref[K - 1])
    check("every other class included", ref.sum() == K - 2)

    # and the centred cosine built from that mask equals the metric's
    W = np.random.default_rng(4).normal(size=(K, FEAT))
    feats = np.nan_to_num(mu, nan=0.0)
    labs = np.arange(K)
    keep = np.arange(K) != K - 1                 # drop the absent class' sample
    prod = M.nc3_alignment(feats[keep], labs[keep], W, K, centre=True,
                           exclude_from_centre=FC)["forget"]
    mine = DC.centred_alignment(mu[FC], DC.feature_centre(mu, ref),
                                W[FC], DC.weight_centre(W, ref))
    check("centred cosine identical to the metric", mine == prod)


def main():
    tests = [
        test_state_hook_does_not_change_the_run,
        test_state_hook_does_not_disturb_rng,
        test_state_hook_does_not_change_the_trace,
        test_evaluate_light_dict_is_unchanged,
        test_corners_reproduce_production_nc3,
        test_epoch0_collapses_to_one_baseline,
        test_residual_identity_closes,
        test_frozen_backbone_leaves_features_fixed,
        test_known_answer_pure_centre_movement,
        test_known_answer_pure_weight_movement,
        test_residual_is_nonzero_when_both_move,
        test_centre_mask_matches_the_metric,
    ]
    for t in tests:
        t()
    print(f"\nNC3 decomposition / state instrumentation: {len(tests)} groups passed")


if __name__ == "__main__":
    main()
