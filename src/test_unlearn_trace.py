"""
Regression tests for the random_label training trace, indexed loaders, mode
selection, run classification and exposure accounting.

Run:  python src/test_unlearn_trace.py

The trace exists so that a full-model run and its frozen-backbone twin can be
PROVEN to have trained on the same sample stream. Two things must hold or the
proof is worthless:

  1. enabling the trace must not change the run in any way;
  2. the trace must actually differ when the stream differs.

Both are asserted below on tiny CPU models, along with the exposure arithmetic
for the loader-cycling behaviour that makes CIFAR and faces incomparable
without a correction.
"""

import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import data as D                       # noqa: E402
import movement as MV                  # noqa: E402
import train as TR                     # noqa: E402
import unlearn as UL                   # noqa: E402
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
# tiny fixtures
# ----------------------------------------------------------------------

D_IN, K = 6, 4


class Toy(Dataset):
    def __init__(self, n=37, seed=0):
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

    def forward(self, x):
        return self.lin(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, 5) * 0.1)

    def forward(self, f, y=None):
        return f @ self.weight.T


def loaders(indexed, bs=8, seed=0):
    ds = Toy()
    base = D.IndexedDataset(ds) if indexed else ds
    tg = D.get_targets(ds)
    f_idx = np.flatnonzero(tg == 0)
    r_idx = np.flatnonzero(tg != 0)
    set_seed(seed)
    fl = D.make_loader(base, f_idx, bs, True, 0)
    rl = D.make_loader(base, r_idx, bs, True, 0)
    return fl, rl


def run(classifier_only, indexed, trace=None, seed=0, epochs=2):
    set_seed(seed)
    bb, hd = Backbone(), Head()
    set_seed(seed)
    fl, rl = loaders(indexed, seed=seed)
    set_seed(seed)
    return UL.random_label(bb, hd, forget_loader=fl, retain_loader=rl,
                           device="cpu", num_classes=K, epochs=epochs,
                           lr=0.01, weight_decay=0.0,
                           classifier_only=classifier_only, trace=trace)


def recorder():
    st = {"h": hashlib.sha256(), "rows": [], "steps": 0, "rp": 0, "fp": 0}

    def t(ep, step, ri, fi, rnd, n_r, n_f):
        st["h"].update(np.int64([ep, step, n_r, n_f]).tobytes())
        for a in (ri, fi, rnd):
            st["h"].update(b"\x00" if a is None
                           else np.ascontiguousarray(a, dtype=np.int64).tobytes())
        st["rows"].append((ep, step, None if ri is None else ri.tolist(),
                           None if fi is None else fi.tolist(), rnd.tolist(),
                           n_r, n_f))
        st["steps"] += 1
        st["rp"] += n_r
        st["fp"] += n_f
    return st, t


def params_of(bb, hd):
    return [p.detach().clone() for p in list(bb.parameters()) + list(hd.parameters())]


def same_params(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


# ----------------------------------------------------------------------
# 1. tracing must not change the run
# ----------------------------------------------------------------------

def test_trace_does_not_change_parameters():
    print("tracing enabled vs disabled: bitwise-identical learned parameters")
    for co in (False, True):
        tag = "clfonly" if co else "full"
        bb0, hd0 = run(co, indexed=True, trace=None)
        st, t = recorder()
        bb1, hd1 = run(co, indexed=True, trace=t)
        check(f"{tag}: parameters identical",
              same_params(params_of(bb0, hd0), params_of(bb1, hd1)))
        check(f"{tag}: trace recorded steps", st["steps"] > 0)


def test_trace_does_not_change_rng_or_outputs():
    print("tracing does not disturb RNG state or model outputs")
    x = torch.randn(5, D_IN)
    run(False, indexed=True, trace=None)
    rng_untraced = torch.random.get_rng_state().clone()
    st, t = recorder()
    run(False, indexed=True, trace=t)
    rng_traced = torch.random.get_rng_state().clone()
    check("global torch RNG state identical after the run",
          torch.equal(rng_untraced, rng_traced))
    bb0, hd0 = run(False, indexed=True, trace=None)
    st2, t2 = recorder()
    bb1, hd1 = run(False, indexed=True, trace=t2)
    with torch.no_grad():
        check("model outputs identical",
              torch.equal(hd0(bb0(x)), hd1(bb1(x))))


def test_indexed_loader_does_not_change_the_run():
    print("IndexedDataset wrapping does not change the learned model")
    for co in (False, True):
        tag = "clfonly" if co else "full"
        a = params_of(*run(co, indexed=False))
        b = params_of(*run(co, indexed=True))
        check(f"{tag}: plain vs indexed loaders give identical parameters",
              same_params(a, b))


# ----------------------------------------------------------------------
# 2. the trace must be determinate and discriminating
# ----------------------------------------------------------------------

def test_trace_determinism():
    print("trace determinism: same seed -> same hash, twice")
    s1, t1 = recorder(); run(False, indexed=True, trace=t1)
    s2, t2 = recorder(); run(False, indexed=True, trace=t2)
    check("hashes identical", s1["h"].hexdigest() == s2["h"].hexdigest())
    check("rows identical", s1["rows"] == s2["rows"])
    s3, t3 = recorder(); run(False, indexed=True, trace=t3, seed=1)
    check("a different seed changes the hash",
          s1["h"].hexdigest() != s3["h"].hexdigest())


def test_full_and_frozen_share_one_trace():
    print("paired: full and classifier-only give the SAME trace hash")
    sf, tf = recorder(); run(False, indexed=True, trace=tf)
    sc, tc = recorder(); run(True, indexed=True, trace=tc)
    check("trace hashes match", sf["h"].hexdigest() == sc["h"].hexdigest())
    check("identical step rows", sf["rows"] == sc["rows"])
    check("identical retain presentations", sf["rp"] == sc["rp"])
    check("identical forget presentations", sf["fp"] == sc["fp"])
    check("but the two runs are genuinely different models",
          not same_params(params_of(*run(False, indexed=True)),
                          params_of(*run(True, indexed=True))))


def test_trace_indices_are_real():
    print("trace indices: real, in-range, forget rows are the forget class")
    ds = Toy()
    tg = D.get_targets(ds)
    st, t = recorder(); run(False, indexed=True, trace=t)
    f_idx = set(np.flatnonzero(tg == 0).tolist())
    r_idx = set(np.flatnonzero(tg != 0).tolist())
    all_f, all_r = set(), set()
    for _ep, _s, ri, fi, rnd, n_r, n_f in st["rows"]:
        check_len = len(ri) == n_r and len(fi) == n_f and len(rnd) == n_f
        if not check_len:
            raise AssertionError("batch size disagrees with index count")
        all_f |= set(fi); all_r |= set(ri)
        if not all(0 <= v < K for v in rnd):
            raise AssertionError("random target out of range")
    check("forget indices are all forget-class rows", all_f <= f_idx)
    check("retain indices are all retain rows", all_r <= r_idx)
    check("every forget row was used", all_f == f_idx)
    check("every retain row was used", all_r == r_idx)
    check("random targets never equal the true forget label (exclude_true)",
          all(v != 0 for _e, _s, _r, _f, rnd, _a, _b in st["rows"] for v in rnd))


def test_trace_without_indices():
    print("plain (non-indexed) loaders still trace, with None indices")
    st, t = recorder(); run(False, indexed=False, trace=t)
    check("steps recorded", st["steps"] > 0)
    check("indices are None", all(r[2] is None and r[3] is None
                                  for r in st["rows"]))
    check("random targets still recorded",
          all(len(r[4]) == r[6] for r in st["rows"]))


# ----------------------------------------------------------------------
# 3. exposure accounting, including partial final batches
# ----------------------------------------------------------------------

def test_exposure_accounting_matches_reality():
    print("exposure accounting: predicted == observed, cycling forget loader")
    from feature_movement import exposure_accounting
    ds = Toy()
    tg = D.get_targets(ds)
    n_f = int((tg == 0).sum())
    n_r = int((tg != 0).sum())
    bs, epochs = 8, 2
    pred = exposure_accounting(n_r, n_f, bs, epochs)
    st, t = recorder(); run(False, indexed=True, trace=t, epochs=epochs)
    check(f"steps: predicted {pred['total_optimiser_steps']} == observed {st['steps']}",
          pred["total_optimiser_steps"] == st["steps"])
    check(f"forget presentations: predicted {pred['total_forget_presentations']} "
          f"== observed {st['fp']}",
          pred["total_forget_presentations"] == st["fp"])
    check(f"retain presentations: predicted {n_r * epochs} == observed {st['rp']}",
          n_r * epochs == st["rp"])
    check("forget loader really did restart",
          pred["forget_loader_restarts_per_epoch"] > 0)
    check("partial final retain batch exists in this fixture",
          n_r % bs != 0)
    check("partial final forget batch exists in this fixture",
          n_f % bs != 0)


def test_exposure_arithmetic_known_values():
    print("exposure arithmetic: the two production shapes, by hand")
    from feature_movement import exposure_accounting
    c = exposure_accounting(45000, 5000, 128, 1)
    check("CIFAR steps 352", c["retain_batches_per_epoch"] == 352)
    check("CIFAR forget batches/pass 40",
          c["natural_forget_batches_per_pass"] == 40)
    check("CIFAR restarts 8", c["forget_loader_restarts_per_epoch"] == 8)
    check("CIFAR forget presentations 44096",
          c["forget_presentations_per_epoch"] == 44096)
    check("CIFAR per unique forget image 8.8192",
          abs(c["presentations_per_unique_forget_image"] - 8.8192) < 1e-4)
    f = exposure_accounting(38775, 40, 128, 1)
    check("faces steps 303", f["retain_batches_per_epoch"] == 303)
    check("faces forget batches/pass 1",
          f["natural_forget_batches_per_pass"] == 1)
    check("faces restarts 302", f["forget_loader_restarts_per_epoch"] == 302)
    check("faces forget presentations 12120",
          f["forget_presentations_per_epoch"] == 12120)
    check("faces per unique forget image 303.0",
          abs(f["presentations_per_unique_forget_image"] - 303.0) < 1e-9)
    ratio = (f["presentations_per_unique_forget_image"]
             / c["presentations_per_unique_forget_image"])
    check("faces:CIFAR raw-presentation ratio ~34.4x", abs(ratio - 34.36) < 0.1)

    # The SECOND, different quantity: loss weighting, not raw presentations.
    K = "avg_unit_weight_forget_ce_coefficient_per_unique_forget_image"
    check("CIFAR coefficient 352/5000 = 0.0704", abs(c[K] - 0.0704) < 1e-9)
    check("faces coefficient 303/40 = 7.575", abs(f[K] - 7.575) < 1e-9)
    check("CIFAR forget-bearing steps == retain batches",
          c["forget_bearing_optimiser_steps"] == 352)
    check("faces forget-bearing steps == retain batches",
          f["forget_bearing_optimiser_steps"] == 303)
    cratio = f[K] / c[K]
    check("coefficient ratio ~107.6x", abs(cratio - 107.60) < 0.1)
    check("the two ratios genuinely disagree (34.4 vs 107.6)",
          abs(cratio - ratio) > 70.0)

    e = exposure_accounting(1024, 128, 128, 1)   # both divide exactly
    check("exact division: 8 steps", e["retain_batches_per_epoch"] == 8)
    check("exact division: 1 forget batch/pass",
          e["natural_forget_batches_per_pass"] == 1)
    check("exact division: 7 restarts",
          e["forget_loader_restarts_per_epoch"] == 7)
    check("exact division: 1024 forget presentations",
          e["forget_presentations_per_epoch"] == 1024)


# ----------------------------------------------------------------------
# 4. mode selection and run classification
# ----------------------------------------------------------------------

def test_mode_selection():
    print("mode selection: classifier-only really freezes the backbone")
    bb_f, _ = run(False, indexed=True)
    bb_c, _ = run(True, indexed=True)
    set_seed(0); bb0 = Backbone()
    base = bb0.lin.weight.detach().clone()
    check("full model moved the backbone",
          not torch.equal(bb_f.lin.weight.detach(), base))
    check("classifier-only left the backbone untouched",
          torch.equal(bb_c.lin.weight.detach(), base))
    _, hd_c = run(True, indexed=True)
    set_seed(0); Backbone(); hd0 = Head()
    check("classifier-only still moved the head",
          not torch.equal(hd_c.weight.detach(), hd0.weight.detach()))


def test_scientific_role_is_explicit():
    print("scientific role: explicit, validated, never inferred")
    import feature_movement as FM
    from feature_movement import resolve_scientific_role as R
    check("classifier-only is always paired_control",
          R(None, True) == "paired_control")
    check("classifier-only accepts a redundant explicit paired_control",
          R("paired_control", True) == "paired_control")
    raises("classifier-only rejects any other role",
           lambda: R("new_experiment", True), "by construction")
    check("full-model new_experiment accepted",
          R("new_experiment", False) == "new_experiment")
    check("full-model instrumented_replication accepted",
          R("instrumented_replication", False) == "instrumented_replication")
    raises("full-model REQUIRES an explicit role",
           lambda: R(None, False), "required for a full-model run")
    raises("full-model may not claim paired_control",
           lambda: R("paired_control", False), "reserved for classifier-only")
    raises("unknown role rejected",
           lambda: R("pilot", False), "unknown")
    check("dataset-name inference is gone",
          not hasattr(FM, "classify_run"))
    check("the role vocabulary is exactly three",
          FM.SCIENTIFIC_ROLES == ("new_experiment", "instrumented_replication",
                                  "paired_control"))


# ----------------------------------------------------------------------
# 5. observed-index validation and indexed extraction
# ----------------------------------------------------------------------

def test_indexed_dataset():
    print("IndexedDataset: yields position, preserves labels and order")
    ds = Toy()
    idx = D.IndexedDataset(ds)
    check("same length", len(idx) == len(ds))
    x, y, i = idx[7]
    check("third element is the position", i == 7)
    check("label preserved", y == ds[7][1])
    check("tensor preserved", torch.equal(x, ds[7][0]))
    check("get_targets sees through the wrapper",
          np.array_equal(D.get_targets(idx), D.get_targets(ds)))
    dl = DataLoader(idx, batch_size=8, shuffle=False)
    seen = np.concatenate([b[2].numpy() for b in dl])
    check("shuffle=False yields 0..n-1 in order",
          np.array_equal(seen, np.arange(len(ds))))


def test_extract_with_indices():
    print("extract_with_indices: observed stream matches the dataset")
    ds = Toy()
    set_seed(0); bb, hd = Backbone(), Head()
    dl = DataLoader(D.IndexedDataset(ds), batch_size=8, shuffle=False)
    f, y, p, i = TR.extract_with_indices(bb, hd, dl, "cpu")
    check("one row per sample", f.shape[0] == len(ds))
    check("indices are 0..n-1", np.array_equal(i, np.arange(len(ds))))
    check("labels match the dataset", np.array_equal(y, D.get_targets(ds)))
    check("predictions present", p.shape == (len(ds),))
    f2, _y2, _p2, _i2 = TR.extract_with_indices(bb, hd, dl, "cpu")
    check("deterministic across calls", np.array_equal(f, f2))


def test_observed_index_mismatch_detection():
    print("observed-index validation: order, duplication, omission, count")
    exp = np.arange(10)
    MV.check_observed_indices(exp.copy(), exp)
    print("  PASS  exact match accepted")
    raises("reordering rejected",
           lambda: MV.check_observed_indices(exp[::-1].copy(), exp), "order")
    dup = exp.copy(); dup[3] = dup[2]
    raises("duplication rejected",
           lambda: MV.check_observed_indices(dup, exp), "more than once")
    raises("short stream rejected",
           lambda: MV.check_observed_indices(exp[:8], exp), "expected")
    raises("long stream rejected",
           lambda: MV.check_observed_indices(np.arange(12), exp), "expected")
    extra = exp.copy(); extra[0] = 99
    raises("unexpected index rejected",
           lambda: MV.check_observed_indices(extra, exp), "")


def main():
    tests = [
        test_trace_does_not_change_parameters,
        test_trace_does_not_change_rng_or_outputs,
        test_indexed_loader_does_not_change_the_run,
        test_trace_determinism,
        test_full_and_frozen_share_one_trace,
        test_trace_indices_are_real,
        test_trace_without_indices,
        test_exposure_accounting_matches_reality,
        test_exposure_arithmetic_known_values,
        test_mode_selection,
        test_scientific_role_is_explicit,
        test_indexed_dataset,
        test_extract_with_indices,
        test_observed_index_mismatch_detection,
    ]
    for t in tests:
        t()
    print(f"\nunlearn trace / indexed loaders: {len(tests)} groups passed")


if __name__ == "__main__":
    main()
