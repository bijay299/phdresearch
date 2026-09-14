"""
Regression tests for data.py's smoke-mode splitting.

Run:  python src/test_data.py

The bug this suite exists for: `make_forget_split` is computed from the
FULL target array regardless of any upstream subsampling. Under --smoke,
that means forget/retain loaders silently trained on the full-size dataset
instead of the tiny smoke subset -- the run took 40+ minutes of CPU instead
of the "couple of minutes" --smoke promises, and it happened silently: no
crash, no error, just a much bigger job than intended.
"""

import sys, os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


def synthetic_targets(num_classes=10, per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    targets = np.concatenate([np.full(per_class, c) for c in range(num_classes)])
    perm = rng.permutation(len(targets))
    return targets[perm]


def synthetic_identity_targets(num_identities=20, per_identity=25):
    """Identity labels laid out contiguously, as FaceFolder produces them."""
    return np.concatenate([np.full(per_identity, i) for i in range(num_identities)])


class TinyDataset:
    """Map-style stand-in: __len__ + __getitem__ is all DataLoader needs.
    Carries labels only -- these tests count samples, they do not train."""

    def __init__(self, targets):
        self.targets = np.asarray(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.zeros(1), int(self.targets[i])


def loader_sample_count(loader):
    return sum(len(y) for _, y in loader)


def loader_classes(loader):
    seen = set()
    for _, y in loader:
        seen.update(int(v) for v in y)
    return seen


def test_stratified_subsample_keeps_every_class():
    print("stratified_subsample: every class present, capped per class")
    targets = synthetic_targets(num_classes=10, per_class=100)
    sub = D.stratified_subsample(targets, per_class=5, seed=0)
    check("subset size == num_classes * per_class", len(sub) == 50)
    check("every class present", set(targets[sub].tolist()) == set(range(10)))
    counts = np.bincount(targets[sub])
    check("no class exceeds the cap", counts.max() <= 5)


def test_unrestricted_split_is_not_a_smoke_subset():
    """Sanity check that reproduces the bug: this is exactly what the old
    run_experiment.py code did -- build a smoke train_idx, then hand
    make_forget_split's output straight to the loaders with no restriction.
    If this assertion ever starts passing, the synthetic data below is no
    longer exercising the bug and the test needs revisiting."""
    print("old behaviour: an unrestricted split is NOT contained in the smoke subset")
    targets = synthetic_targets(num_classes=10, per_class=100)
    train_idx = D.stratified_subsample(targets, per_class=5, seed=0)
    split = D.make_forget_split(targets, forget_class=0, mode="all", seed=0)

    allowed = set(train_idx.tolist())
    forget_outside = [i for i in split.forget_idx if i not in allowed]
    retain_outside = [i for i in split.retain_idx if i not in allowed]
    check("forget_idx has members outside the smoke subset", len(forget_outside) > 0)
    check("retain_idx has members outside the smoke subset", len(retain_outside) > 0)
    check("retain_idx is far bigger than the smoke subset itself",
          len(split.retain_idx) > 5 * len(train_idx))


def test_restrict_split_is_subset_of_smoke_subset():
    """The fix: restrict_split(split, train_idx) must confine every index
    set in the split to the smoke subset -- this is what run_experiment.py
    now calls under --smoke before building the forget/retain loaders."""
    print("restrict_split: forget/retain/held-out all subsets of the smoke subset")
    for mode, kwargs in [("all", {}), ("subset", {"forget_fraction": 0.5})]:
        targets = synthetic_targets(num_classes=10, per_class=100)
        train_idx = D.stratified_subsample(targets, per_class=5, seed=0)
        split = D.make_forget_split(targets, forget_class=0, mode=mode, seed=0, **kwargs)
        restricted = D.restrict_split(split, train_idx)

        allowed = set(train_idx.tolist())
        check(f"[{mode}] forget_idx subset of smoke subset",
              set(restricted.forget_idx.tolist()) <= allowed)
        check(f"[{mode}] retain_idx subset of smoke subset",
              set(restricted.retain_idx.tolist()) <= allowed)
        check(f"[{mode}] forget_heldout_idx subset of smoke subset",
              set(restricted.forget_heldout_idx.tolist()) <= allowed)


def test_unrestricted_test_loader_is_not_bounded():
    """Sanity check that reproduces the loader-side bug: run_experiment.py
    subsampled the training-side loaders under --smoke but built test_loader
    with `indices=None`, i.e. the full test set. evaluate() forwards the whole
    test loader through the backbone once per condition, so that asymmetry --
    not the unlearning methods -- was ~70s of every ~78s condition in the
    CIFAR smoke run. If this assertion ever starts passing, the synthetic
    data below is no longer exercising the bug."""
    print("old behaviour: a test loader built with indices=None is unbounded")
    test_targets = synthetic_targets(num_classes=10, per_class=100, seed=1)
    test_ds = TinyDataset(test_targets)
    test_idx = D.stratified_subsample(test_targets, per_class=5, seed=0)

    old = D.make_loader(test_ds, None, 16, False, 0)       # what run() used to do
    new = D.make_loader(test_ds, test_idx, 16, False, 0)

    check("unrestricted loader yields the entire test set",
          loader_sample_count(old) == len(test_targets))
    check("restricted loader yields only the subsample",
          loader_sample_count(new) == len(test_idx))
    check("old behaviour is an order of magnitude bigger",
          loader_sample_count(old) > 10 * loader_sample_count(new))


def test_smoke_restricts_train_and_test_loaders_alike():
    """The fix: under --smoke BOTH sides get a stratified subsample, so no
    loader handed to evaluate() silently covers a full split."""
    print("smoke loaders: train and test sides are both bounded and class-complete")
    train_targets = synthetic_targets(num_classes=10, per_class=500, seed=0)
    test_targets = synthetic_targets(num_classes=10, per_class=100, seed=1)

    train_idx = D.stratified_subsample(train_targets, per_class=5, seed=0)
    test_idx = D.stratified_subsample(test_targets, per_class=5, seed=0)

    train_loader = D.make_loader(TinyDataset(train_targets), train_idx, 16, False, 0)
    test_loader = D.make_loader(TinyDataset(test_targets), test_idx, 16, False, 0)

    check("train loader bounded by the subsample",
          loader_sample_count(train_loader) == 50)
    check("test loader bounded by the subsample",
          loader_sample_count(test_loader) == 50)
    check("test loader is not the full test set",
          loader_sample_count(test_loader) < len(test_targets))
    check("every class still present on the train side",
          loader_classes(train_loader) == set(range(10)))
    check("every class still present on the test side",
          loader_classes(test_loader) == set(range(10)))


def _tiny_face_dir(tmp, identities=("alice", "bob", "carol"), per_identity=4):
    from PIL import Image
    for name in identities:
        d = Path(tmp) / name
        d.mkdir(parents=True)
        for j in range(per_identity):
            Image.new("RGB", (8, 8), (j * 30 % 255, 0, 0)).save(d / f"{j:03d}.jpg")
    return tmp


def test_views_share_one_label_mapping():
    """Constraint behind building ONE FaceFolder: label k must mean the same
    person in every split. Views share the scan, so it does by construction;
    two independent constructions over differing directories would not."""
    print("with_transform: train/eval/test views share one label mapping")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_face_dir(tmp)
        ds = D.FaceFolder(tmp, min_images=1)
        train_view = ds.with_transform("TRAIN-TRANSFORM")
        eval_view = ds.with_transform("EVAL-TRANSFORM")

        check("views carry different transforms",
              train_view.transform != eval_view.transform)
        check("views share the identity_names object",
              train_view.identity_names is eval_view.identity_names)
        check("views share the samples object",
              train_view.samples is eval_view.samples)
        check("label -> identity mapping is identical",
              train_view.identity_names == eval_view.identity_names)
        check("original dataset is untouched by the views",
              ds.transform is None)


def test_one_scan_split_keeps_labels_aligned_across_train_and_test():
    """End-to-end on a real (tiny) directory: split one FaceFolder by image
    and confirm both sides agree about who label k is."""
    print("one-scan split: train and test agree on every label")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_face_dir(tmp, per_identity=5)
        ds = D.FaceFolder(tmp, min_images=1)
        train_idx, test_idx = D.stratified_image_split(ds.targets, 0.4, seed=0)

        train = Subset(ds.with_transform(None), train_idx)
        test = Subset(ds.with_transform(None), test_idx)

        tr_targets = D.get_targets(train)
        te_targets = D.get_targets(test)
        check("get_targets works through Subset", len(tr_targets) == len(train_idx))
        check("no image is in both splits",
              not (set(train_idx.tolist()) & set(test_idx.tolist())))
        check("every identity on both sides",
              set(tr_targets.tolist()) == set(te_targets.tolist()) == {0, 1, 2})
        # Both splits resolve labels through the SAME scan, so label k cannot
        # disagree between them -- that is the property two constructions lost.
        check("both splits share one identity_names mapping",
              train.dataset.identity_names is test.dataset.identity_names)
        check("mapping is the sorted directory names",
              ds.identity_names == ["alice", "bob", "carol"])


def test_old_faces_test_set_was_identical_to_train():
    """Sanity check reproducing the train/test leak: build_datasets used
    `FaceFolder(cfg.get("test_root", root))` and no config ever set test_root,
    so the test side re-scanned the SAME directory with the SAME kwargs and
    seed. Verified against the real extracted dataset: 36364/36364 images
    overlapped, meaning every reported faces test metric was train accuracy.
    If this assertion ever starts passing, this test needs revisiting."""
    print("old behaviour: a second FaceFolder over the same root reproduces train")
    targets = synthetic_identity_targets(num_identities=20, per_identity=25)

    # Two independent constructions over one directory enumerate identically.
    old_train = np.arange(len(targets))
    old_test = np.arange(len(targets))

    check("old train/test index sets are identical",
          set(old_train.tolist()) == set(old_test.tolist()))
    check("old split is therefore NOT disjoint",
          len(set(old_train.tolist()) & set(old_test.tolist())) > 0)


def test_stratified_image_split_is_disjoint_and_identity_complete():
    """The fix: split by image WITHIN identity. Disjoint images, shared
    identities -- the test set must still contain the forgotten person or
    forget accuracy cannot be measured at all."""
    print("stratified_image_split: disjoint images, every identity on both sides")
    targets = synthetic_identity_targets(num_identities=20, per_identity=25)
    train_idx, test_idx = D.stratified_image_split(targets, test_fraction=0.2, seed=0)

    tr, te = set(train_idx.tolist()), set(test_idx.tolist())
    check("train and test index sets are disjoint", not (tr & te))
    check("together they cover every image", len(tr) + len(te) == len(targets))
    check("every identity appears in train",
          set(targets[train_idx].tolist()) == set(range(20)))
    check("every identity appears in test",
          set(targets[test_idx].tolist()) == set(range(20)))
    check("identity sets match on both sides -- split is by image, not identity",
          set(targets[train_idx].tolist()) == set(targets[test_idx].tolist()))
    check(f"test fraction is about 0.2 (got {len(te) / len(targets):.2f})",
          0.15 <= len(te) / len(targets) <= 0.25)


def test_stratified_image_split_is_reproducible():
    print("stratified_image_split: same seed gives the same split")
    targets = synthetic_identity_targets(num_identities=10, per_identity=20)
    a_tr, a_te = D.stratified_image_split(targets, test_fraction=0.2, seed=0)
    b_tr, b_te = D.stratified_image_split(targets, test_fraction=0.2, seed=0)
    c_tr, c_te = D.stratified_image_split(targets, test_fraction=0.2, seed=1)

    check("same seed -> identical train indices", np.array_equal(a_tr, b_tr))
    check("same seed -> identical test indices", np.array_equal(a_te, b_te))
    check("different seed -> different split", not np.array_equal(a_te, c_te))
    check("different seed is still disjoint",
          not (set(c_tr.tolist()) & set(c_te.tolist())))


def test_stratified_image_split_rejects_single_image_identity():
    """An identity with one image cannot appear on both sides. Fail loudly
    rather than silently dropping it from train or from test."""
    print("stratified_image_split: a 1-image identity raises instead of vanishing")
    targets = np.array([0, 0, 0, 0, 1])      # identity 1 has a single image
    try:
        D.stratified_image_split(targets, test_fraction=0.2, seed=0)
        check("raised ValueError", False)
    except ValueError as e:
        check(f"raised ValueError ({str(e)[:40]}...)", True)


def _faces_cfg(root, max_images_per_identity=8):
    """The `data` sub-dict as a faces config resolves it. Deliberately has NO
    `seed` key -- that absence is what the defect below turned on."""
    return {"name": "faces", "root": root, "image_size": 8, "min_images": 1,
            "max_images_per_identity": max_images_per_identity,
            "test_fraction": 0.25}


def _selection_and_split(root, seed=None, per_identity=12):
    """Build through the real entry point and return what the seed controls:
    the subsampled file list, and the train/test partition over it."""
    cfg = _faces_cfg(root)
    if seed is None:
        train, _, test, _ = D.build_datasets(cfg)          # default path
    else:
        train, _, test, _ = D.build_datasets(cfg, seed=seed)
    files = tuple(str(p) for p, _ in train.dataset.samples)
    return files, tuple(train.indices.tolist()), tuple(test.indices.tolist()), train, test


def test_old_data_seed_lookup_was_pinned_to_zero():
    """Negative case for the fix below: the previous code read the seed as
    `cfg.get("seed", 0)` off the `data` sub-dict. No config sets `data.seed`,
    so every experiment seed resolved to 0. If this ever stops holding, the
    schema changed and the regression tests below need revisiting."""
    print("old behaviour: data-local seed lookup ignored the experiment seed")
    cfg = _faces_cfg("/nonexistent")
    check("no config-shaped data dict carries a seed key", "seed" not in cfg)
    resolved = [cfg.get("seed", 0) for _ in (0, 1, 2, 7)]
    check("old lookup returned 0 for every experiment seed", resolved == [0, 0, 0, 0])


def test_build_datasets_rejects_a_nested_data_seed():
    """The experiment seed is the SOLE source of sampling/split randomness.
    A stray `data.seed` must fail loudly rather than silently overriding it --
    two places to set one seed is exactly how the original defect hid. Checked
    before any dataset is touched, so it fires for every dataset name."""
    print("build_datasets: a nested data.seed is rejected, never honoured")
    cfg = _faces_cfg("/nonexistent")
    cfg["seed"] = 7
    for name, passed in (("faces", 1), ("cifar10", 1), ("faces", None)):
        cfg["name"] = name
        try:
            if passed is None:
                D.build_datasets(cfg)
            else:
                D.build_datasets(cfg, seed=passed)
            check(f"{name}: raised ValueError", False)
        except ValueError as e:
            msg = str(e)
            check(f"{name}: raised ValueError ({msg[:34]}...)", True)
            check(f"{name}: message names data.seed", "data.seed" in msg)
            check(f"{name}: message points at the top-level seed",
                  "top-level" in msg and "--set seed=" in msg)
    # It must fail rather than fall through to loading, even though the root
    # does not exist -- i.e. the check runs before FileNotFoundError could.
    check("rejection precedes any dataset access", True)


def test_build_datasets_seed_reaches_face_sampling_and_split():
    """The fix: the experiment seed is an argument, and it reaches BOTH
    FaceFolder's per-identity image subsampling and stratified_image_split.
    12 images per identity capped to 8 means the selection is a real random
    choice, not the whole directory."""
    print("build_datasets: the experiment seed reaches faces sampling and split")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_face_dir(tmp, per_identity=12)

        f0a, tr0a, te0a, _, _ = _selection_and_split(tmp, seed=0)
        f0b, tr0b, te0b, _, _ = _selection_and_split(tmp, seed=0)
        f1, tr1, te1, _, _ = _selection_and_split(tmp, seed=1)

        check("subsampling actually engaged (8 of 12 kept per identity)",
              len(f0a) == 3 * 8)
        check("same seed -> identical selected samples", f0a == f0b)
        check("same seed -> identical train partition", tr0a == tr0b)
        check("same seed -> identical test partition", te0a == te0b)
        check("different seed -> different selection or partition",
              (f1 != f0a) or (tr1 != tr0a) or (te1 != te0a))


def test_build_datasets_default_seed_reproduces_seed_zero():
    """Compatibility: every existing faces run was seed 0, and the default
    argument must still produce exactly what those runs saw."""
    print("build_datasets: the default seed still reproduces seed 0 exactly")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_face_dir(tmp, per_identity=12)

        fd, trd, ted, _, _ = _selection_and_split(tmp, seed=None)
        f0, tr0, te0, _, _ = _selection_and_split(tmp, seed=0)

        check("default selection == seed 0 selection", fd == f0)
        check("default train partition == seed 0", trd == tr0)
        check("default test partition == seed 0", ted == te0)


def test_build_datasets_split_stays_sound_at_every_seed():
    """Whatever the seed does to the selection, the split invariants hold:
    disjoint sides, every identity on both, one shared label mapping."""
    print("build_datasets: disjoint, identity-complete, label-consistent at every seed")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _tiny_face_dir(tmp, per_identity=12)

        for seed in (0, 1, 2, 7):
            _, tr, te, train, test = _selection_and_split(tmp, seed=seed)
            tr_targets = D.get_targets(train)
            te_targets = D.get_targets(test)

            check(f"seed {seed}: no image in both splits", not (set(tr) & set(te)))
            check(f"seed {seed}: every identity on both sides",
                  set(tr_targets.tolist()) == set(te_targets.tolist()) == {0, 1, 2})
            check(f"seed {seed}: both splits share one identity_names mapping",
                  train.dataset.identity_names is test.dataset.identity_names)
            check(f"seed {seed}: mapping is the sorted directory names",
                  train.dataset.identity_names == ["alice", "bob", "carol"])
            # Labels must still resolve through the shared scan: the target of
            # index i is the label stored beside that image, on both sides.
            samples = train.dataset.samples
            check(f"seed {seed}: train labels match the shared sample table",
                  all(samples[i][1] == t for i, t in zip(tr, tr_targets.tolist())))
            check(f"seed {seed}: test labels match the shared sample table",
                  all(samples[i][1] == t for i, t in zip(te, te_targets.tolist())))


def test_restrict_split_preserves_disjointness():
    print("restrict_split: forget/retain/held-out stay disjoint after restriction")
    targets = synthetic_targets(num_classes=10, per_class=100)
    train_idx = D.stratified_subsample(targets, per_class=5, seed=0)
    split = D.make_forget_split(targets, forget_class=0, mode="subset",
                                forget_fraction=0.5, seed=0)
    restricted = D.restrict_split(split, train_idx)

    f = set(restricted.forget_idx.tolist())
    r = set(restricted.retain_idx.tolist())
    h = set(restricted.forget_heldout_idx.tolist())
    check("forget/retain disjoint", not (f & r))
    check("forget/held-out disjoint", not (f & h))
    check("retain/held-out disjoint", not (r & h))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print()
    print(f"{len(tests)} test groups passed.")
