"""
Regression tests for the controlled random-label dose schedule.

Run:  python src/test_dose_schedule.py

WHAT IS BEING PROTECTED
-----------------------
`random_label` gained two optional controls -- how many of an epoch's retain
steps carry the forget term (`forget_active_steps_per_epoch`, m) and what
coefficient that term gets (`forget_loss_weight`, lambda). Three things have to
hold or the control is not a control:

  1. WITH THE CONTROLS OFF, NOTHING CHANGED. Every number already in
     notes/decisions.md came from the uncontrolled loop. If adding an optional
     branch perturbed it by one float, those numbers would silently stop being
     reproducible. `test_default_matches_legacy_bitwise` pins the default path
     against a verbatim copy of the pre-control loop -- parameters AND global
     RNG state.

  2. THE CANDIDATE STREAM IS UNCHANGED BY THE DOSE. The forget batch is still
     pulled and its random targets still drawn on inactive steps, so a dosed
     run and an undosed run at the same seed produce the SAME
     `training_trace_sha256`. That is what lets the production gate compare a
     dosed faces pair against the undosed pair's hash and conclude the sample
     stream is the same thing at a different dose. If the schedule skipped the
     draw, the two would differ and nothing could be attributed to the dose.

  3. INACTIVE MEANS NOT FORWARDED. A zero-weight forward contributes no
     gradient but still updates BatchNorm running statistics, so the model
     would keep seeing the forget class on a step whose dose is supposed to be
     zero. Asserted twice over, with a forward counter and with BatchNorm's own
     `num_batches_tracked`.
"""

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import data as D                       # noqa: E402
import unlearn as UL                   # noqa: E402
from unlearn import _clone, _optimizer, _params, _step, _unpack  # noqa: E402
from utils import set_seed             # noqa: E402


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def raises(name, fn, needle=""):
    try:
        fn()
    except Exception as e:                       # noqa: BLE001
        ok = needle.lower() in str(e).lower()
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            raise AssertionError(f"{name}: wrong message: {e}")
        return
    print(f"  FAIL  {name}")
    raise AssertionError(f"{name}: expected an exception")


# ----------------------------------------------------------------------
# fixtures: 80 samples over 4 classes, batch 8
#   forget class 0 -> 20 samples -> 3 batches per pass (8, 8, 4)
#   retain         -> 60 samples -> S = 8 steps        (8 x7, 4)
# Both partial batches and loader cycling are therefore exercised, and no
# batch has size 1 (BatchNorm in train mode rejects that).
# ----------------------------------------------------------------------

D_IN, K, N, BS = 6, 4, 80, 8


class Toy(Dataset):
    def __init__(self, n=N, seed=0):
        g = np.random.default_rng(seed)
        self.x = g.normal(size=(n, D_IN)).astype(np.float32)
        self.targets = (np.arange(n) % K).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.targets[i])


class Backbone(nn.Module):
    feat_dim = 5

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D_IN, 5)
        self.bn = nn.BatchNorm1d(5)
        self.n_forward = 0
        self.rows_seen = 0

    def forward(self, x):
        self.n_forward += 1
        self.rows_seen += int(x.shape[0])
        return self.bn(self.lin(x))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, 5) * 0.1)

    def forward(self, f, y=None):
        return f @ self.weight.T


def loaders(seed=0):
    ds = Toy()
    base = D.IndexedDataset(ds)
    tg = D.get_targets(ds)
    set_seed(seed)
    return (D.make_loader(base, np.flatnonzero(tg == 0), BS, True, 0),
            D.make_loader(base, np.flatnonzero(tg != 0), BS, True, 0))


def fresh(seed=0):
    """Model + loaders + RNG in exactly the state the method expects."""
    set_seed(seed)
    bb, hd = Backbone(), Head()
    set_seed(seed)
    fl, rl = loaders(seed)
    set_seed(seed)
    return bb, hd, fl, rl


def params_of(bb, hd):
    return [p.detach().clone()
            for p in list(bb.parameters()) + list(hd.parameters())]


def buffers_of(bb):
    return [b.detach().clone() for b in bb.buffers()]


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


# ----------------------------------------------------------------------
# a verbatim copy of the pre-control loop, to compare the default against
# ----------------------------------------------------------------------

def legacy_random_label(backbone, head, forget_loader, retain_loader, device,
                        num_classes, epochs=2, lr=1e-3, weight_decay=5e-4,
                        classifier_only=False, exclude_true=True):
    """The `random_label` body as it stood at commit 7250c70.

    Copied rather than imported on purpose: the point is to detect a change in
    the production function, and a reference that tracks the production
    function detects nothing.
    """
    backbone, head = _clone(backbone, head)
    backbone.to(device); head.to(device)
    opt = _optimizer(_params(backbone, head, classifier_only), lr, weight_decay)

    for ep in range(epochs):
        if not classifier_only:
            backbone.train()
        head.train()
        f_iter = iter(forget_loader)
        for step, rbatch in enumerate(retain_loader):
            xr, yr, ir = _unpack(rbatch)
            try:
                xf, yf, if_ = _unpack(next(f_iter))
            except StopIteration:
                f_iter = iter(forget_loader)
                xf, yf, if_ = _unpack(next(f_iter))

            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            rnd = torch.randint(0, num_classes, yf.shape, device=device)
            if exclude_true:
                clash = rnd == yf
                while clash.any():
                    rnd[clash] = torch.randint(0, num_classes,
                                               (int(clash.sum()),), device=device)
                    clash = rnd == yf

            loss = _step(backbone, head, xr, yr) + _step(backbone, head, xf, rnd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return backbone, head


# ----------------------------------------------------------------------
# recorders
# ----------------------------------------------------------------------

def dose_recorder():
    st = {"h": hashlib.sha256(), "rows": [], "active_steps": 0,
          "active_pres": 0, "candidate_pres": 0, "mass": 0.0}

    def t(ep, step, active, weight, fi, rnd, n_f):
        h = st["h"]
        h.update(np.int64([ep, step, 1 if active else 0, n_f]).tobytes())
        h.update(np.float64([weight]).tobytes())
        for a in (fi, rnd):
            h.update(b"\x00" if a is None
                     else np.ascontiguousarray(a, dtype=np.int64).tobytes())
        st["rows"].append({"ep": ep, "step": step, "active": bool(active),
                           "weight": float(weight), "n_f": int(n_f),
                           "fi": None if fi is None else fi.tolist(),
                           "rnd": rnd.tolist()})
        st["candidate_pres"] += int(n_f)
        if active:
            st["active_steps"] += 1
            st["active_pres"] += int(n_f)
            st["mass"] += float(weight)
    return st, t


def candidate_recorder():
    st = {"h": hashlib.sha256(), "steps": 0}

    def t(ep, step, ri, fi, rnd, n_r, n_f):
        h = st["h"]
        h.update(np.int64([ep, step, n_r, n_f]).tobytes())
        for a in (ri, fi, rnd):
            h.update(b"\x00" if a is None
                     else np.ascontiguousarray(a, dtype=np.int64).tobytes())
        st["steps"] += 1
    return st, t


def run(m=None, lam=1.0, classifier_only=False, epochs=2, seed=0,
        trace=None, dose_trace=None):
    bb, hd, fl, rl = fresh(seed)
    out = UL.random_label(bb, hd, forget_loader=fl, retain_loader=rl,
                          device="cpu", num_classes=K, epochs=epochs,
                          lr=0.01, weight_decay=0.0,
                          classifier_only=classifier_only,
                          trace=trace, dose_trace=dose_trace,
                          forget_active_steps_per_epoch=m,
                          forget_loss_weight=lam)
    return out


# ----------------------------------------------------------------------
# 1. the default path is bit-for-bit the pre-control loop
# ----------------------------------------------------------------------

def test_default_matches_legacy_bitwise():
    print("default (m=None, lambda=1.0) == the pre-control loop, bitwise")
    for co in (False, True):
        tag = "clfonly" if co else "full"
        bb, hd, fl, rl = fresh()
        leg_bb, leg_hd = legacy_random_label(
            bb, hd, fl, rl, "cpu", K, epochs=2, lr=0.01, weight_decay=0.0,
            classifier_only=co)
        leg_rng = torch.random.get_rng_state().clone()

        new_bb, new_hd = run(classifier_only=co)
        new_rng = torch.random.get_rng_state().clone()

        check(f"{tag}: learned parameters identical",
              same(params_of(leg_bb, leg_hd), params_of(new_bb, new_hd)))
        check(f"{tag}: BatchNorm buffers identical",
              same(buffers_of(leg_bb), buffers_of(new_bb)))
        check(f"{tag}: global torch RNG state identical",
              torch.equal(leg_rng, new_rng))

    # An explicit unit weight with no schedule is still the default path.
    a = params_of(*run())
    b = params_of(*run(m=None, lam=1.0))
    check("explicit lambda=1.0 with no schedule is the same path", same(a, b))

    # And the controls must actually be capable of changing something, or the
    # test above is vacuous.
    check("a real dose DOES change the model",
          not same(a, params_of(*run(m=3, lam=0.5))))


# ----------------------------------------------------------------------
# 2. the schedule selects exactly m, evenly spaced
# ----------------------------------------------------------------------

def test_schedule_selects_m_evenly_spaced():
    print("schedule: exactly m steps, gaps differing by at most one")
    for S in (8, 37, 100, 303, 352):
        for m in (1, 2, 3, 9, S // 2, S - 1, S):
            if not 1 <= m <= S:
                continue
            flags = UL.evenly_spaced_active_steps(S, m)
            idx = [i for i, f in enumerate(flags) if f]
            if len(flags) != S or len(idx) != m:
                raise AssertionError(f"S={S} m={m}: got {len(idx)} active")
            gaps = [b - a for a, b in zip(idx, idx[1:])]
            if gaps and max(gaps) - min(gaps) > 1:
                raise AssertionError(f"S={S} m={m}: uneven gaps {sorted(set(gaps))}")
    check("exactly m active, gaps within 1, across many (S, m)", True)

    check("m == S selects every step",
          all(UL.evenly_spaced_active_steps(11, 11)))
    check("m == 1 selects exactly one, and it is the last",
          UL.evenly_spaced_active_steps(11, 1) == [False] * 10 + [True])

    # The production setting, by hand.
    flags = UL.evenly_spaced_active_steps(303, 9)
    idx = [i for i, f in enumerate(flags) if f]
    check("S=303, m=9 -> 9 active steps", len(idx) == 9)
    check("S=303, m=9 -> the expected indices",
          idx == [33, 67, 100, 134, 168, 201, 235, 269, 302])
    check("S=303, m=9 -> gaps are only 33 and 34",
          set(b - a for a, b in zip(idx, idx[1:])) == {33, 34})

    # The formula in the prompt, evaluated independently of the implementation.
    ref = [((i + 1) * 9) // 303 > (i * 9) // 303 for i in range(303)]
    check("matches floor((step+1)*m/S) > floor(step*m/S)", flags == ref)


# ----------------------------------------------------------------------
# 3. inactive forget batches are never forwarded
# ----------------------------------------------------------------------

def test_inactive_batches_are_not_forwarded():
    print("inactive steps: no forget forward, no BatchNorm update")
    epochs, m = 2, 3
    S = math.ceil(60 / BS)                       # 8 retain steps per epoch

    bb_all, _ = run(epochs=epochs)               # every step active
    check(f"all-active: {epochs}*2*{S} forwards",
          bb_all.n_forward == epochs * 2 * S)
    check("all-active: BN tracked every forward",
          int(bb_all.bn.num_batches_tracked) == epochs * 2 * S)

    bb_dose, _ = run(m=m, lam=0.5, epochs=epochs)
    check(f"dosed: {epochs}*({S}+{m}) forwards -- retain always, forget only "
          f"when active",
          bb_dose.n_forward == epochs * (S + m))
    check("dosed: BN tracked exactly those forwards",
          int(bb_dose.bn.num_batches_tracked) == epochs * (S + m))
    check("dosed backbone saw strictly fewer rows than all-active",
          bb_dose.rows_seen < bb_all.rows_seen)

    # A zero-weight forward would leave the forward count at the all-active
    # value while contributing no gradient -- the exact failure this guards.
    check("the two forward counts genuinely differ",
          bb_dose.n_forward != bb_all.n_forward)

    # Rows seen must equal retain rows + the active forget batches' rows.
    st, t = dose_recorder()
    run(m=m, lam=0.5, epochs=epochs, dose_trace=t)
    active_rows = sum(r["n_f"] for r in st["rows"] if r["active"])
    bb2, _ = run(m=m, lam=0.5, epochs=epochs)
    check("rows forwarded == retain rows + active forget rows",
          bb2.rows_seen == 60 * epochs + active_rows)


# ----------------------------------------------------------------------
# 4. candidate batches and random targets are still drawn every step
# ----------------------------------------------------------------------

def test_candidate_stream_is_unchanged_by_the_dose():
    print("candidate stream: drawn on every step, identical hash across doses")
    epochs = 2
    S = math.ceil(60 / BS)

    base, tb = candidate_recorder()
    run(epochs=epochs, trace=tb)

    dosed, td = candidate_recorder()
    st, tdose = dose_recorder()
    run(m=3, lam=0.3128888888888889, epochs=epochs, trace=td, dose_trace=tdose)

    check("same number of traced steps", base["steps"] == dosed["steps"] == S * epochs)
    check("candidate training trace hash is IDENTICAL across doses",
          base["h"].hexdigest() == dosed["h"].hexdigest())

    check("dose trace has a row for every step, active or not",
          len(st["rows"]) == S * epochs)
    check("every step recorded a candidate forget batch",
          all(r["fi"] is not None and len(r["fi"]) == r["n_f"]
              for r in st["rows"]))
    check("every step recorded random targets, inactive ones included",
          all(len(r["rnd"]) == r["n_f"] for r in st["rows"]))
    check("inactive steps really are the majority here",
          sum(1 for r in st["rows"] if not r["active"]) == (S - 3) * epochs)
    check("random targets never equal the true forget label",
          all(v != 0 for r in st["rows"] for v in r["rnd"]))
    check("inactive rows carry weight 0.0",
          all(r["weight"] == 0.0 for r in st["rows"] if not r["active"]))
    check("active rows carry exactly lambda",
          all(r["weight"] == 0.3128888888888889
              for r in st["rows"] if r["active"]))

    # The dose trace must DISCRIMINATE, or matching hashes prove nothing.
    s2, t2 = dose_recorder(); run(m=3, lam=0.5, epochs=epochs, dose_trace=t2)
    s3, t3 = dose_recorder(); run(m=4, lam=0.3128888888888889,
                                 epochs=epochs, dose_trace=t3)
    check("a different weight changes the dose hash",
          st["h"].hexdigest() != s2["h"].hexdigest())
    check("a different m changes the dose hash",
          st["h"].hexdigest() != s3["h"].hexdigest())
    check("...while the candidate hash is blind to both",
          base["h"].hexdigest() == dosed["h"].hexdigest())


# ----------------------------------------------------------------------
# 5. predicted and observed dose accounting agree
# ----------------------------------------------------------------------

def test_dose_accounting_predicted_equals_observed():
    print("dose accounting: predicted == observed")
    from feature_movement import dose_accounting, exposure_accounting
    epochs, m, lam = 2, 3, 0.3128888888888889
    n_r, n_f = 60, 20

    pred = dose_accounting(n_r, n_f, BS, epochs, m, lam)
    st, t = dose_recorder()
    run(m=m, lam=lam, epochs=epochs, dose_trace=t)

    check("active forget-bearing steps",
          pred["total_active_forget_bearing_steps"] == st["active_steps"] == m * epochs)
    check("active forget presentations",
          pred["total_active_forget_presentations"] == st["active_pres"])
    check("candidate forget presentations",
          pred["total_candidate_forget_presentations"] == st["candidate_pres"])
    check("weighted coefficient mass",
          abs(pred["total_weighted_forget_loss_coefficient_mass"]
              - st["mass"]) < 1e-12)
    check("candidate accounting agrees with exposure_accounting",
          pred["candidate_forget_presentations_per_epoch"]
          == exposure_accounting(n_r, n_f, BS, epochs)["forget_presentations_per_epoch"])

    # Uncontrolled: candidate and active must coincide exactly.
    un = dose_accounting(n_r, n_f, BS, epochs, None, 1.0)
    su, tu = dose_recorder(); run(epochs=epochs, dose_trace=tu)
    check("uncontrolled: candidate == active presentations",
          un["total_candidate_forget_presentations"]
          == un["total_active_forget_presentations"] == su["active_pres"])
    check("uncontrolled: dose_controlled is False", un["dose_controlled"] is False)
    check("controlled: dose_controlled is True", pred["dose_controlled"] is True)

    # The production numbers, by hand, from notes/decisions.md 2026-09-15.
    f = dose_accounting(38775, 40, 128, 1, 9, (352 / 5000) / (9 / 40))
    check("faces: S = 303 retain steps", f["retain_steps_per_epoch"] == 303)
    check("faces: candidate presentations 12120",
          f["candidate_forget_presentations_per_epoch"] == 12120)
    check("faces: active presentations 360",
          f["active_forget_presentations_per_epoch"] == 360)
    check("faces: 9 active presentations per unique forget image",
          abs(f["active_presentations_per_unique_forget_image_per_epoch"] - 9.0)
          < 1e-12)
    check("faces: weighted coefficient mass 2.816",
          abs(f["total_weighted_forget_loss_coefficient_mass_per_epoch"]
              - 2.816) < 1e-12)
    check("faces: weighted coefficient per unique forget image 0.0704",
          abs(f["avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch"]
              - 0.0704) < 1e-12)
    check("faces: lambda is the documented value",
          f["forget_loss_weight"] == 0.3128888888888889)
    # The target the lambda was derived from: CIFAR's unit-weight coefficient.
    c = dose_accounting(45000, 5000, 128, 1, None, 1.0)
    check("CIFAR unit-weight coefficient per unique forget image 0.0704",
          abs(c["avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch"]
              - 0.0704) < 1e-12)
    check("the dosed faces coefficient MATCHES CIFAR's, by construction",
          abs(f["avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch"]
              - c["avg_weighted_forget_loss_coefficient_per_unique_forget_image_per_epoch"])
          < 1e-12)
    # ...while raw presentations are matched only approximately (9 vs 8.8192).
    check("raw presentations are approximately, not exactly, matched",
          abs(f["active_presentations_per_unique_forget_image_per_epoch"]
              - 44096 / 5000) > 0.1)


# ----------------------------------------------------------------------
# 6. a dosed pair shares both hashes
# ----------------------------------------------------------------------

def test_full_and_frozen_share_both_traces():
    print("dosed pair: full and classifier-only share candidate AND dose hash")
    m, lam = 3, 0.3128888888888889
    cf, tcf = candidate_recorder(); df, tdf = dose_recorder()
    run(m=m, lam=lam, classifier_only=False, trace=tcf, dose_trace=tdf)
    cc, tcc = candidate_recorder(); dc, tdc = dose_recorder()
    run(m=m, lam=lam, classifier_only=True, trace=tcc, dose_trace=tdc)

    check("candidate training trace hashes match",
          cf["h"].hexdigest() == cc["h"].hexdigest())
    check("active dose trace hashes match",
          df["h"].hexdigest() == dc["h"].hexdigest())
    check("identical dose rows", df["rows"] == dc["rows"])
    check("identical active step counts", df["active_steps"] == dc["active_steps"])
    check("identical weighted mass", df["mass"] == dc["mass"])
    check("but the two are genuinely different models",
          not same(params_of(*run(m=m, lam=lam, classifier_only=False)),
                   params_of(*run(m=m, lam=lam, classifier_only=True))))
    check("the frozen member really froze its backbone",
          same(buffers_of(run(m=m, lam=lam, classifier_only=True)[0]),
               buffers_of(fresh()[0])))


# ----------------------------------------------------------------------
# 7. invalid schedules and weights are rejected
# ----------------------------------------------------------------------

def test_invalid_schedules_and_weights_rejected():
    print("validation: bad m and bad lambda are refused, loudly")
    S = math.ceil(60 / BS)
    raises("m = 0 rejected", lambda: run(m=0), "1 <= m <= s")
    raises("m negative rejected", lambda: run(m=-1), "1 <= m <= s")
    raises("m > S rejected", lambda: run(m=S + 1), "1 <= m <= s")
    raises("m non-integer rejected", lambda: run(m=2.5), "must be an integer")
    raises("m bool rejected", lambda: run(m=True), "must be an integer")

    raises("lambda = 0 rejected", lambda: run(m=2, lam=0.0), "strictly positive")
    raises("lambda negative rejected", lambda: run(m=2, lam=-0.5),
           "strictly positive")
    raises("lambda NaN rejected", lambda: run(m=2, lam=float("nan")), "finite")
    raises("lambda inf rejected", lambda: run(m=2, lam=float("inf")), "finite")
    # Weight is validated even without a schedule, so a typo cannot slip past.
    raises("lambda validated with no schedule too",
           lambda: run(lam=-1.0), "strictly positive")

    check("m = S is accepted (every step active)", run(m=S) is not None)
    check("m = 1 is accepted", run(m=1) is not None)

    from feature_movement import dose_accounting
    raises("driver accounting rejects a bad weight",
           lambda: dose_accounting(60, 20, BS, 1, 3, 0.0), "strictly positive")
    raises("driver accounting rejects a bad schedule",
           lambda: dose_accounting(60, 20, BS, 1, 0, 1.0), "1 <= m <= s")

    raises("schedule helper rejects S < 1",
           lambda: UL.evenly_spaced_active_steps(0, 1), "positive integer")


# ----------------------------------------------------------------------
# 8. the real driver, end to end, in both modes
# ----------------------------------------------------------------------

def test_driver_end_to_end_with_dose():
    print("driver end-to-end: dosed pair writes consistent metadata")
    import test_feature_movement_driver as TD

    tmp = Path(tempfile.mkdtemp())
    try:
        base = TD.make_baseline(tmp)
        out = tmp / "dose"
        # The toy driver fixture: 30 samples/class, 5 classes, batch 8.
        # retain = 135 -> S = 17 steps; m = 4.
        m, lam = 4, 0.3128888888888889
        TD.run_driver(base, out, classifier_only=False,
                      forget_active_steps_per_epoch=m, forget_loss_weight=lam)
        TD.run_driver(base, out, classifier_only=True,
                      forget_active_steps_per_epoch=m, forget_loss_weight=lam)

        _df, full, frows = TD.load(out, "full")
        _dc, clf, crows = TD.load(out, "clfonly")

        for tag, res in (("full", full), ("clfonly", clf)):
            dp, do = res["dose_predicted"], res["dose_observed"]
            check(f"{tag}: dose recorded as controlled", dp["dose_controlled"])
            check(f"{tag}: m recorded", dp["forget_active_steps_per_epoch"] == m)
            check(f"{tag}: lambda recorded", dp["forget_loss_weight"] == lam)
            check(f"{tag}: active dose trace hash recorded",
                  len(res["active_dose_trace_sha256"]) == 64)
            check(f"{tag}: the two hashes are different quantities",
                  res["active_dose_trace_sha256"] != res["training_trace_sha256"])
            check(f"{tag}: predicted == observed active steps",
                  dp["total_active_forget_bearing_steps"]
                  == do["observed_active_forget_bearing_steps_total"])
            check(f"{tag}: predicted == observed active presentations",
                  dp["total_active_forget_presentations"]
                  == do["observed_active_forget_presentations_total"])
            check(f"{tag}: predicted == observed candidate presentations",
                  dp["total_candidate_forget_presentations"]
                  == do["observed_candidate_forget_presentations_total"])
            check(f"{tag}: predicted == observed weighted mass",
                  abs(dp["total_weighted_forget_loss_coefficient_mass"]
                      - do["observed_total_weighted_forget_loss_coefficient_mass"])
                  < 1e-9)
            check(f"{tag}: active steps are fewer than total steps",
                  do["observed_active_forget_bearing_steps_total"]
                  < do["observed_steps_total"])
            check(f"{tag}: candidate stream still covers every step",
                  do["observed_candidate_forget_presentations_total"]
                  == res["exposure_observed"]["observed_forget_presentations_total"])
            check(f"{tag}: accounting text refuses the gradient reading",
                  "not observed gradient" in dp["not_a_gradient_magnitude"])

        check("dosed pair shares the candidate trace",
              full["training_trace_sha256"] == clf["training_trace_sha256"])
        check("dosed pair shares the dose trace",
              full["active_dose_trace_sha256"] == clf["active_dose_trace_sha256"])
        check("dosed pair shares the observed index hash",
              full["splits"]["train_eval_observed_index_sha256"]
              == clf["splits"]["train_eval_observed_index_sha256"])
        check("frozen member still moved no features",
              all(r["raw_forget"]["mean"] == 0.0 for r in crows))
        check("full member still moved features",
              frows[-1]["raw_forget"]["mean"] > 0.0)
        check("roles are recorded as before",
              full["scientific_role"] == "new_experiment"
              and clf["scientific_role"] == "paired_control")

        # An UNDOSED run of the same fixture must share the candidate trace
        # with the dosed pair -- that is the production gate, in miniature.
        out2 = tmp / "undosed"
        TD.run_driver(base, out2, classifier_only=False)
        _d2, undosed, _r2 = TD.load(out2, "full")
        check("undosed run shares the dosed run's CANDIDATE trace",
              undosed["training_trace_sha256"] == full["training_trace_sha256"])
        check("...but not its dose trace",
              undosed["active_dose_trace_sha256"]
              != full["active_dose_trace_sha256"])
        check("undosed run is recorded as uncontrolled",
              undosed["dose_predicted"]["dose_controlled"] is False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [
        test_default_matches_legacy_bitwise,
        test_schedule_selects_m_evenly_spaced,
        test_inactive_batches_are_not_forwarded,
        test_candidate_stream_is_unchanged_by_the_dose,
        test_dose_accounting_predicted_equals_observed,
        test_full_and_frozen_share_both_traces,
        test_invalid_schedules_and_weights_rejected,
        test_driver_end_to_end_with_dose,
    ]
    for t in tests:
        t()
    print(f"\ndose schedule: {len(tests)} groups passed")


if __name__ == "__main__":
    main()
