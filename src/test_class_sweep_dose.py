"""
Regression tests for the dose/provenance wiring in the class-sweep driver.

Run:  python src/test_class_sweep_dose.py

WHAT IS BEING PROTECTED
-----------------------
`src/unlearn.py` already owns the dose schedule itself, and
`src/test_dose_schedule.py` pins that contract. What is new here is the
PRODUCTION PATH: `scripts/unlearn_across_classes.py` now forwards the dose
parameters, reseeds per cell on request, and writes a provenance/dose record.
Four things have to hold:

  1. THE DEFAULT PATH IS UNCHANGED. Every existing caller passes no dose
     arguments. With them unset the driver must call `random_label` with
     m=None and lambda=1.0, must NOT reseed per cell, and must not add the
     uncentred NC3 field to the trajectory. `test_default_path_*` pin that.
     If this broke, previously recorded rows would stop being reproducible.

  2. AN IMPOSSIBLE DOSE IS REFUSED BEFORE ANY WORK HAPPENS. m must satisfy
     1 <= m <= S for THIS cell's own S, which depends on the retain count and
     so on the class count. At K=100 a dose tuned at K=1000 could silently be
     unschedulable, and the whole point of the fixed-dose design is that it is
     the same dose at every K. The check must fire before the run directory is
     created, so a refused cell leaves nothing behind to clean up.

  3. PREDICTED DOSE EQUALS OBSERVED DOSE. The driver recomputes the predicted
     accounting and compares it against what the loop actually did. Integer
     counts must agree exactly.

  4. THE TWO TRACES SAY DIFFERENT THINGS. The candidate stream hash must be
     invariant to the dose (same seed, same samples drawn), while the dose
     hash must change when the schedule changes. Collapsing them would make a
     dose change look like a sample-stream change.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import train as TR                      # noqa: E402
import unlearn as UL                    # noqa: E402
import unlearn_across_classes as UAC    # noqa: E402

FAILURES = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILURES.append(name)


def raises(name, fn, needle=""):
    try:
        fn()
    except Exception as e:
        ok = needle in str(e)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"   (message lacked {needle!r}: {e})"))
        if not ok:
            FAILURES.append(name)
        return
    print(f"  FAIL  {name}   (no exception raised)")
    FAILURES.append(name)


# ---------------------------------------------------------------- fixtures

class Toy(Dataset):
    """n_per classes x per images, 4 features, deterministic."""

    def __init__(self, n_classes=4, per=16, dim=4):
        g = torch.Generator().manual_seed(1234)
        self.x = torch.randn(n_classes * per, dim, generator=g)
        self.y = torch.arange(n_classes).repeat_interleave(per)
        self.targets = self.y.numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


class Backbone(nn.Module):
    feat_dim = 4

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class Head(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, 4) * 0.1)

    def forward(self, f, y=None):
        return f @ self.weight.t()


def make_cfg(out_dir):
    return {
        "seed": 0,
        "out_dir": str(out_dir),
        "data": {"name": "toy", "batch_size": 8, "num_workers": 0},
        "unlearn": {"epochs": 2, "lr": 1e-3, "weight_decay": 5e-4},
        "backbone": {}, "head": {},
    }


def stub_eval(monkey_store):
    """Replace the two expensive evaluators with cheap deterministic stubs."""
    monkey_store["evaluate"] = TR.evaluate
    monkey_store["evaluate_light"] = TR.evaluate_light

    def fake_evaluate(*a, **k):
        return {"output_forget": 0.0, "output_retain": 1.0}

    def fake_light(*a, **k):
        point = {"output_forget": 0.0, "output_retain": 1.0,
                 "probe_forget": 0.0, "nc3_centred_forget": 0.5,
                 "nc1_angular": 0.1}
        if k.get("include_uncentred"):
            point["nc3_uncentred_forget"] = -0.5
        return point

    TR.evaluate = fake_evaluate
    TR.evaluate_light = fake_light


def unstub(monkey_store):
    TR.evaluate = monkey_store["evaluate"]
    TR.evaluate_light = monkey_store["evaluate_light"]


def run_cell(tmp, name_suffix, **kw):
    """One random_label cell through the real driver function."""
    torch.manual_seed(0)
    ds = Toy()
    bb, hd = Backbone(), Head()
    cfg = make_cfg(tmp)
    out = Path(tmp) / name_suffix
    prov = {"identity_names": ["idA", "idB", "idC", "idD"]}
    return UAC.run_condition(
        cfg, "ce", bb, hd, 4, ds, None, None, "cpu",
        forget_class=0, method="random_label", classifier_only=True,
        epochs=2, lr=1e-3, weight_decay=5e-4, trajectory_every=1,
        bs=8, nw=0, out_dir=str(out), provenance=prov, **kw)


def result_of(tmp, name_suffix):
    root = Path(tmp) / name_suffix / "toy_ce_seed0_random_label_clfonly_fc0"
    return json.load(open(root / "result.json"))


# ------------------------------------------------------------------- tests

def test_invalid_dose_is_refused_before_the_run_dir_exists():
    """m > S must stop the cell, and leave no directory behind."""
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    try:
        # Toy has 64 samples, 16 forget -> 48 retain at bs=8 -> S=6.
        raises("m=99 rejected for this cell's S",
               lambda: run_cell(tmp, "bad", forget_active_steps_per_epoch=99),
               "1 <= m <= S")
        root = Path(tmp) / "bad" / "toy_ce_seed0_random_label_clfonly_fc0"
        check("refused cell created no run directory", not root.exists())
        raises("m=0 rejected",
               lambda: run_cell(tmp, "bad0", forget_active_steps_per_epoch=0),
               "1 <= m <= S")
    finally:
        unstub(store)
        shutil.rmtree(tmp, ignore_errors=True)


def test_default_path_passes_no_dose_and_does_not_reseed():
    """Unset dose arguments must reach random_label as its own defaults."""
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    seen = {}
    real = UL.run_unlearning

    def spy(method, bb, hd, device, **kw):
        seen.update(kw)
        return real(method, bb, hd, device, **kw)

    UL.run_unlearning = spy
    try:
        run_cell(tmp, "default")
        check("default forwards m=None",
              seen.get("forget_active_steps_per_epoch", "missing") is None)
        check("default forwards lambda=1.0",
              seen.get("forget_loss_weight") == 1.0)
        r = result_of(tmp, "default")
        check("default records per_cell_seed=False", r["per_cell_seed"] is False)
        check("default dose_predicted marks uncontrolled",
              r["dose_predicted"]["dose_controlled"] is False)
    finally:
        UL.run_unlearning = real
        unstub(store)
        shutil.rmtree(tmp, ignore_errors=True)


def test_default_trajectory_has_no_uncentred_field():
    """The uncentred column is opt-in; default rows must keep their old shape."""
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    try:
        run_cell(tmp, "traj")
        root = Path(tmp) / "traj" / "toy_ce_seed0_random_label_clfonly_fc0"
        rows = [json.loads(l) for l in open(root / "trajectory.jsonl")]
        check("default trajectory omits nc3_uncentred_forget",
              all("nc3_uncentred_forget" not in r for r in rows))
        shutil.rmtree(tmp, ignore_errors=True)

        tmp2 = tempfile.mkdtemp()
        run_cell(tmp2, "traj2", forget_active_steps_per_epoch=3)
        root2 = Path(tmp2) / "traj2" / "toy_ce_seed0_random_label_clfonly_fc0"
        rows2 = [json.loads(l) for l in open(root2 / "trajectory.jsonl")]
        check("dosed trajectory includes nc3_uncentred_forget",
              all("nc3_uncentred_forget" in r for r in rows2))
        shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        unstub(store)


def test_dose_propagates_and_predicted_equals_observed():
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    try:
        run_cell(tmp, "dosed", forget_active_steps_per_epoch=3,
                 forget_loss_weight=0.5, per_cell_seed=True)
        r = result_of(tmp, "dosed")
        dp, do = r["dose_predicted"], r["dose_observed"]
        check("m recorded", dp["forget_active_steps_per_epoch"] == 3)
        check("lambda recorded", dp["forget_loss_weight"] == 0.5)
        check("per_cell_seed recorded", r["per_cell_seed"] is True)
        check("active steps predicted == observed",
              dp["total_active_forget_bearing_steps"]
              == do["observed_active_forget_bearing_steps_total"])
        check("active presentations predicted == observed",
              dp["total_active_forget_presentations"]
              == do["observed_active_forget_presentations_total"])
        check("candidate presentations predicted == observed",
              dp["total_candidate_forget_presentations"]
              == do["observed_candidate_forget_presentations_total"])
        check("weighted mass predicted == observed",
              abs(dp["total_weighted_forget_loss_coefficient_mass"]
                  - do["observed_total_weighted_forget_loss_coefficient_mass"])
              <= 1e-9 * max(1.0, dp["total_weighted_forget_loss_coefficient_mass"]))
        check("active steps == m * epochs",
              do["observed_active_forget_bearing_steps_total"] == 3 * 2)
    finally:
        unstub(store)
        shutil.rmtree(tmp, ignore_errors=True)


def test_candidate_trace_is_dose_invariant_but_dose_trace_is_not():
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    try:
        run_cell(tmp, "d_none", per_cell_seed=True)
        run_cell(tmp, "d_m3", forget_active_steps_per_epoch=3, per_cell_seed=True)
        run_cell(tmp, "d_m2", forget_active_steps_per_epoch=2, per_cell_seed=True)
        a = result_of(tmp, "d_none")
        b = result_of(tmp, "d_m3")
        c = result_of(tmp, "d_m2")
        check("candidate trace invariant to the dose",
              a["training_trace_sha256"] == b["training_trace_sha256"]
              == c["training_trace_sha256"])
        check("dose trace differs between m=3 and m=2",
              b["active_dose_trace_sha256"] != c["active_dose_trace_sha256"])
        check("dose trace differs from the uncontrolled schedule",
              a["active_dose_trace_sha256"] != b["active_dose_trace_sha256"])
        check("candidate and dose hashes are not the same value",
              b["training_trace_sha256"] != b["active_dose_trace_sha256"])
    finally:
        unstub(store)
        shutil.rmtree(tmp, ignore_errors=True)


def test_provenance_block_is_recorded():
    tmp = tempfile.mkdtemp()
    store = {}
    stub_eval(store)
    try:
        run_cell(tmp, "prov", forget_active_steps_per_epoch=3)
        r = result_of(tmp, "prov")
        check("forget identity name resolved from the manifest",
              r["forget_identity_name"] == "idA")
        check("update mode recorded", r["update_mode"] == "classifier_only")
        check("retain steps per epoch recorded",
              r["retain_steps_per_epoch"] == 6)
        check("forget/retain counts recorded",
              r["n_forget"] == 16 and r["n_retain"] == 48)
    finally:
        unstub(store)
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_light_default_shape_is_unchanged():
    """The opt-in flag must add exactly one key and change nothing else."""
    keys_default = {"output_forget", "output_retain", "probe_forget",
                    "nc3_centred_forget", "nc1_angular"}
    import inspect
    sig = inspect.signature(TR.evaluate_light)
    check("include_uncentred defaults to False",
          sig.parameters["include_uncentred"].default is False)
    src = inspect.getsource(TR.evaluate_light)
    check("uncentred branch is guarded by the flag",
          "if include_uncentred:" in src)
    check("default key set documented above matches the literal",
          all(k in src for k in keys_default))


def test_uncentred_forget_is_finite_not_nan():
    """The uncentred column must carry a real cosine, not nan.

    `nc3_alignment` uses `exclude_from_centre` for TWO things: which class to
    leave out of the centring reference, and which class to report as
    "forget". Only the first is conditional on `centre`. Calling it with
    centre=False and no `exclude_from_centre` therefore returns a dict with no
    "forget" key at all, and the column fills with nan while every structural
    check still passes. That happened once; this test is why it cannot happen
    again silently.
    """
    import numpy as np
    import metrics as M
    rng = np.random.default_rng(0)
    n_classes, dim = 4, 5
    y = np.repeat(np.arange(n_classes), 8)
    f = rng.normal(size=(len(y), dim))
    W = rng.normal(size=(n_classes, dim))

    without = M.nc3_alignment(f, y, W, n_classes, centre=False)
    check("reproduces the defect: no forget key without exclude_from_centre",
          "forget" not in without)

    withfc = M.nc3_alignment(f, y, W, n_classes, centre=False,
                             exclude_from_centre=1)
    check("uncentred forget present when the class is named",
          "forget" in withfc and np.isfinite(withfc["forget"]))

    # and the guard that matters for real trajectories
    src = __import__("inspect").getsource(TR.evaluate_light)
    check("evaluate_light names the forget class in the uncentred call",
          "centre=False,\n                                exclude_from_centre=forget_class)" in src)


def main():
    print("class-sweep dose/provenance wiring")
    for fn in (test_invalid_dose_is_refused_before_the_run_dir_exists,
               test_default_path_passes_no_dose_and_does_not_reseed,
               test_default_trajectory_has_no_uncentred_field,
               test_dose_propagates_and_predicted_equals_observed,
               test_candidate_trace_is_dose_invariant_but_dose_trace_is_not,
               test_provenance_block_is_recorded,
               test_evaluate_light_default_shape_is_unchanged,
               test_uncentred_forget_is_finite_not_nan):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all class-sweep dose/provenance tests passed")


if __name__ == "__main__":
    main()
