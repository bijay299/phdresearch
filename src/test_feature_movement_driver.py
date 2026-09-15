"""
End-to-end smoke test for scripts/feature_movement.py.

Run:  python src/test_feature_movement_driver.py

WHY THIS FILE EXISTS
--------------------
Twice now a rename left a stale reference in the driver -- once a dict key in
a log f-string, once a local variable in the result block -- and both times
`make test` was green because nothing ever EXECUTED the driver. Both bugs
surfaced only after a GPU run had completed all of its setup, wasting the
launch. Unit-testing the pieces is not enough: the driver's own control flow
has to run.

So this runs the real `main()` end to end on a tiny synthetic dataset on CPU,
in both modes, and checks the artifacts it writes. It is fast (seconds) and
needs no GPU and no real checkpoint.
"""

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import data as D                       # noqa: E402
import feature_movement as FM          # noqa: E402


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


# ----------------------------------------------------------------------
# a tiny fake dataset + backbone, injected in place of the real ones
# ----------------------------------------------------------------------

K, D_IN, FEAT, N_PER = 5, 7, 6, 30


class Toy(torch.utils.data.Dataset):
    def __init__(self, n_per=N_PER, seed=0):
        g = np.random.default_rng(seed)
        c = g.normal(size=(K, D_IN)) * 3.0
        self.x = np.vstack([c[k] + g.normal(size=(n_per, D_IN))
                            for k in range(K)]).astype(np.float32)
        self.targets = np.concatenate([np.full(n_per, k) for k in range(K)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.targets[i])


class TinyBackbone(nn.Module):
    feat_dim = FEAT

    def __init__(self, **_):
        super().__init__()
        self.lin = nn.Linear(D_IN, FEAT)

    def forward(self, x):
        return self.lin(x)


class TinyHead(nn.Module):
    def __init__(self, feat_dim=FEAT, num_classes=K, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.3)

    def forward(self, f, y=None):
        return f @ self.weight.T


def fake_build_datasets(cfg, seed=0, log=None):
    ds = Toy(seed=seed)
    return ds, Toy(seed=seed), Toy(seed=seed + 1), K


def make_baseline(tmp: Path) -> Path:
    """A RunDir-shaped baseline directory: config.json, env.json, ckpt.pt."""
    run = tmp / "baseline"
    run.mkdir(parents=True)
    cfg = {
        "seed": 0, "out_dir": str(tmp / "out"),
        "data": {"name": "toyset", "batch_size": 8, "num_workers": 0},
        "backbone": {"arch": "tiny"},
        "head": {"name": "tinyhead"},
        "train": {"epochs": 1},
        "unlearn": {"epochs": 2, "lr": 0.01, "weight_decay": 0.0,
                    "forget_fraction": 0.5},
    }
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "env.json").write_text(json.dumps({"git_commit": "testcommit"}))
    torch.manual_seed(0)
    bb, hd = TinyBackbone(), TinyHead()
    torch.save({"backbone": bb.state_dict(), "head": hd.state_dict()},
               run / "ckpt.pt")
    return run


def args(run_dir, out_dir, classifier_only, n_controls=4, epochs=2,
         scientific_role="new_experiment"):
    return types.SimpleNamespace(
        run_dir=str(run_dir), ckpt=None, forget_class=0, epochs=epochs,
        out_dir=str(out_dir), name=None, anchor_fraction=0.5, anchor_seed=0,
        classifier_only=classifier_only, n_controls=n_controls,
        scientific_role=None if classifier_only else scientific_role,
        device="cpu", allow_dirty=True, set=[],
    )


def run_driver(run_dir, out_dir, classifier_only,
               scientific_role="new_experiment"):
    """Run the real main() with the tiny model/dataset injected."""
    import backbones as BK
    import heads as H
    orig = (D.build_datasets, BK.build_backbone, H.build_head,
            FM.D.build_datasets, FM.build_backbone, FM.build_head)
    try:
        FM.D.build_datasets = fake_build_datasets
        FM.build_backbone = lambda **kw: TinyBackbone(**kw)
        FM.build_head = lambda name, **kw: TinyHead(**kw)
        FM.main(args(run_dir, out_dir, classifier_only,
                     scientific_role=scientific_role))
    finally:
        (D.build_datasets, BK.build_backbone, H.build_head,
         FM.D.build_datasets, FM.build_backbone, FM.build_head) = orig


def load(out_dir, mode):
    d = Path(out_dir) / f"toyset_tinyhead_seed0_random_label_{mode}_fc0"
    result = json.loads((d / "result.json").read_text())
    rows = [json.loads(l) for l in
            (d / "trajectory.jsonl").read_text().splitlines()]
    return d, result, rows


# ----------------------------------------------------------------------

def test_driver_runs_both_modes():
    print("driver end-to-end: both modes run and write complete artifacts")
    tmp = Path(tempfile.mkdtemp())
    try:
        base = make_baseline(tmp)
        out = tmp / "paired"
        run_driver(base, out, classifier_only=False)
        run_driver(base, out, classifier_only=True)

        for mode in ("full", "clfonly"):
            d, res, rows = load(out, mode)
            check(f"{mode}: run directory written", d.is_dir())
            check(f"{mode}: no checkpoint saved", not (d / "ckpt.pt").exists())
            check(f"{mode}: epochs 0..2 present",
                  [r["epoch"] for r in rows] == [0, 1, 2])
            # Not exactly 0.0: at epoch 0 base == later, so Procrustes fits
            # R ~ I up to float error and the angle is ~1e-7 deg, not a hard
            # zero. The GPU runs print 0.000 at 3 dp for the same reason.
            check(f"{mode}: epoch-0 displacement ~ 0",
                  rows[0]["aligned_forget"]["mean"] < 1e-6)
            check(f"{mode}: epoch-0 raw displacement is exactly 0",
                  rows[0]["raw_forget"]["mean"] == 0.0)
            check(f"{mode}: epoch-0 CKA is 1",
                  abs(rows[0]["cka_linear_secondary"] - 1.0) < 1e-9)
            check(f"{mode}: trace hash recorded",
                  len(res["training_trace_sha256"]) == 64)
            check(f"{mode}: observed index hash recorded",
                  len(res["splits"]["train_eval_observed_index_sha256"]) == 64)
            check(f"{mode}: nc3 both conventions present",
                  all(k in rows[0] for k in
                      ("nc3_centred_forget", "nc3_uncentred_forget",
                       "nc3_centred_forget_epoch0",
                       "nc3_uncentred_forget_epoch0")))
            check(f"{mode}: control identities recorded",
                  len(res["controls"]["control_class_identities"])
                  == res["controls"]["n_control_classes"])
            check(f"{mode}: exposure predicted == observed",
                  res["exposure_predicted"]["total_optimiser_steps"]
                  == res["exposure_observed"]["observed_optimiser_steps_total"])
            check(f"{mode}: no NaN in the trajectory",
                  not [k for r in rows for k, v in r.items()
                       if isinstance(v, float) and v != v])
            # The defect that reached four published artifacts: every row
            # carried the full-model method label regardless of mode.
            expected_method = f"random_label_{mode}"
            check(f"{mode}: every row's method == {expected_method}",
                  all(r["method"] == expected_method for r in rows))
            check(f"{mode}: rows agree with the top-level result",
                  all(r["method"] == res["method"] for r in rows))
            check(f"{mode}: update_mode recorded orthogonally",
                  res["update_mode"] ==
                  ("classifier_only" if mode == "clfonly" else "full_model"))
            check(f"{mode}: scientific_role recorded",
                  res["scientific_role"] in FM.SCIENTIFIC_ROLES)
            check(f"{mode}: no inferred run_classification field",
                  "run_classification" not in res)
            interp = res["controls"]["interpretation"].lower()
            check(f"{mode}: control text claims no direction",
                  "not established" in interp
                  and "inflating" not in interp.replace(
                      "described as inflating", ""))
            check(f"{mode}: exposure records BOTH dose summaries",
                  "presentations_per_unique_forget_image"
                  in res["exposure_predicted"]
                  and "avg_unit_weight_forget_ce_coefficient_per_unique"
                      "_forget_image" in res["exposure_predicted"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_pairs_share_a_trace():
    print("driver end-to-end: full and clfonly produce the same trace hash")
    tmp = Path(tempfile.mkdtemp())
    try:
        base = make_baseline(tmp)
        out = tmp / "paired"
        run_driver(base, out, classifier_only=False)
        run_driver(base, out, classifier_only=True)
        _d, full, frows = load(out, "full")
        _d2, clf, crows = load(out, "clfonly")
        check("trace hashes match",
              full["training_trace_sha256"] == clf["training_trace_sha256"])
        check("observed index hashes match",
              full["splits"]["train_eval_observed_index_sha256"]
              == clf["splits"]["train_eval_observed_index_sha256"])
        check("exposure identical",
              full["exposure_observed"] == clf["exposure_observed"])
        check("update modes differ",
              full["update_mode"] != clf["update_mode"])
        check("clfonly is the paired control",
              clf["scientific_role"] == "paired_control")
        check("full-model role is the one that was requested",
              full["scientific_role"] == "new_experiment")
        check("method labels differ between the pair",
              full["method"] != clf["method"])
        check("clfonly method label is correct",
              clf["method"] == "random_label_clfonly")
        check("frozen backbone really froze: zero RAW feature movement",
              all(r["raw_forget"]["mean"] == 0.0 for r in crows))
        check("frozen backbone: aligned movement ~ 0 too",
              all(r["aligned_forget"]["mean"] < 1e-6 for r in crows))
        check("full model really moved features",
              frows[-1]["raw_forget"]["mean"] > 0.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_requires_explicit_role():
    print("driver end-to-end: a full-model run without a role is refused")
    tmp = Path(tempfile.mkdtemp())
    try:
        base = make_baseline(tmp)
        out = tmp / "paired"
        try:
            run_driver(base, out, classifier_only=False, scientific_role=None)
        except ValueError as e:
            check("raised ValueError naming the flag",
                  "scientific-role" in str(e))
        else:
            raise AssertionError("expected a ValueError")
        check("nothing was written", not out.exists() or not any(out.iterdir()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_driver_refuses_nonempty_output():
    print("driver end-to-end: refuses to overwrite an existing run directory")
    tmp = Path(tempfile.mkdtemp())
    try:
        base = make_baseline(tmp)
        out = tmp / "paired"
        run_driver(base, out, classifier_only=False)
        try:
            run_driver(base, out, classifier_only=False)
        except FileExistsError as e:
            check("raised FileExistsError", "refusing to start" in str(e))
        else:
            raise AssertionError("expected FileExistsError on reuse")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [test_driver_runs_both_modes,
             test_driver_pairs_share_a_trace,
             test_driver_requires_explicit_role,
             test_driver_refuses_nonempty_output]
    for t in tests:
        t()
    print(f"\nfeature_movement driver: {len(tests)} groups passed")


if __name__ == "__main__":
    main()
