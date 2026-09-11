"""
Datasets and splits.

Two sources:
  * CIFAR-10/100  -- the pilot. No access dependency, downloads itself.
  * FaceFolder    -- ImageFolder layout, one directory per identity.

The important part of this file is not loading images. It is
`make_forget_split`, which decides what "forget this person" actually means.

Two split modes
---------------
`all`   every image of the identity goes in the forget set.
        This is class-level unlearning, matching the AISTATS setting.

`subset` only SOME images of the identity are handed over; the rest are
        held out and never shown to the unlearning method.
        This is what lets us ask whether forgetting GENERALISES -- unlearn
        on the given images, then probe on the held-out ones. Use this for
        the week 5-8 experiments.

Keep the held-out images out of every training and unlearning path. If they
leak in, the generalisation result is meaningless.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
FACE_MEAN, FACE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


# ----------------------------------------------------------------------
# transforms
# ----------------------------------------------------------------------

def cifar_transforms(train: bool):
    if train:
        return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def face_transforms(train: bool, size: int = 112):
    """Horizontal flip only. Aggressive augmentation (rotation, colour jitter)
    changes the appearance distribution, which is the thing we are trying to
    measure -- do not add it without a reason."""
    if train:
        return T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(FACE_MEAN, FACE_STD)])
    return T.Compose([T.Resize((size, size)), T.ToTensor(),
                      T.Normalize(FACE_MEAN, FACE_STD)])


# ----------------------------------------------------------------------
# face dataset
# ----------------------------------------------------------------------

class FaceFolder(Dataset):
    """
    ImageFolder layout:

        root/
          identity_0001/  img1.jpg  img2.jpg  ...
          identity_0002/  ...

    `max_identities` and `min_images` subset the dataset. Do subset -- you do
    not need a competitive face model, you need one where the geometry is
    measurable and unlearning runs in minutes.
    """

    def __init__(self, root: str | Path, transform=None,
                 max_identities: int | None = None,
                 min_images: int = 10,
                 max_images_per_identity: int | None = None,
                 seed: int = 0):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} not found. Point `data.root` at your extracted "
                f"face dataset (one folder per identity)."
            )
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        dirs = sorted(d for d in self.root.iterdir() if d.is_dir())

        rng = np.random.default_rng(seed)
        kept: List[Tuple[Path, int]] = []
        self.identity_names: List[str] = []

        for d in dirs:
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in exts)
            if len(files) < min_images:
                continue
            if max_images_per_identity and len(files) > max_images_per_identity:
                idx = rng.choice(len(files), max_images_per_identity, replace=False)
                files = [files[i] for i in sorted(idx)]
            label = len(self.identity_names)
            self.identity_names.append(d.name)
            kept.extend((f, label) for f in files)
            if max_identities and len(self.identity_names) >= max_identities:
                break

        if not kept:
            raise RuntimeError(
                f"no identities in {self.root} had at least {min_images} images"
            )

        self.samples = kept
        self.targets = np.array([lab for _, lab in kept])
        self.num_classes = len(self.identity_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def with_transform(self, transform) -> "FaceFolder":
        """A view of this dataset under a different transform.

        Shallow copy: `samples` and `identity_names` are shared, not rebuilt.
        That sharing is the point. Labels here are assigned by enumeration
        order of the sorted identity directories, so two independent
        FaceFolder constructions over different directory contents would
        silently disagree about which person label k refers to. Taking views
        of ONE scan makes that impossible by construction.
        """
        view = copy.copy(self)
        view.transform = transform
        return view


# ----------------------------------------------------------------------
# splits
#
# Three index sets exist in this project and they are NOT interchangeable:
#
#   test          images never trained on, spanning EVERY identity.
#                 Built by `stratified_image_split`. Measures how the model
#                 generalises. Read by `train.evaluate`.
#
#   forget/retain images that WERE trained on, partitioned by identity for
#                 the unlearning step. Built by `make_forget_split`.
#
#   forget-heldout  images of the forget identity that WERE trained on but are
#                 withheld from the unlearning method (subset mode only).
#                 Built by `make_forget_split`. Measures whether forgetting
#                 GENERALISES to images of that person the unlearner never
#                 touched. This is not a test set and must never be used as
#                 one -- the original model saw these images.
# ----------------------------------------------------------------------

def stratified_image_split(targets: Sequence[int], test_fraction: float = 0.2,
                           seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split image indices WITHIN each identity. Returns (train_idx, test_idx).

    By image, never by identity. A disjoint-identity split would leave the
    test set with no images of the forgotten person at all, and forget
    accuracy is precisely the quantity this project measures -- there would
    be nothing to compute it on. So every identity appears on both sides.

    Each identity contributes at least one image to each side, so an identity
    can never end up train-only or test-only.
    """
    targets = np.asarray(targets)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        if len(idx) < 2:
            raise ValueError(
                f"identity {c} has {len(idx)} image(s); needs at least 2 to "
                f"appear in both train and test"
            )
        perm = rng.permutation(idx)
        n_test = int(round(test_fraction * len(perm)))
        n_test = max(1, min(len(perm) - 1, n_test))   # both sides non-empty
        test_idx.extend(perm[:n_test])
        train_idx.extend(perm[n_test:])

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


@dataclass
class ForgetSplit:
    """Index sets over the TRAIN split. Everything downstream keys off this.

    These index the training data, never the test set -- see the section
    comment above for why the three sets are not interchangeable.
    """
    forget_class: int
    forget_idx: np.ndarray            # given to the unlearning method
    retain_idx: np.ndarray            # everything else in train
    forget_heldout_idx: np.ndarray    # forget identity, withheld from unlearning
    mode: str

    def summary(self) -> str:
        return (f"class {self.forget_class} | mode={self.mode} | "
                f"forget {len(self.forget_idx)} | retain {len(self.retain_idx)} | "
                f"forget-heldout {len(self.forget_heldout_idx)}")


def make_forget_split(targets: Sequence[int], forget_class: int,
                      mode: str = "all", forget_fraction: float = 0.5,
                      seed: int = 0) -> ForgetSplit:
    """
    mode='all'     -> every training image of the class is forgotten,
                      forget-heldout is empty.
    mode='subset'  -> a fraction is handed to the unlearning method, the rest
                      becomes forget-heldout: images of the same person that
                      the ORIGINAL model trained on but the unlearner never
                      sees. Probing those answers "did forgetting generalise
                      within this identity?".

    forget-heldout is NOT a test set. The original model trained on it. The
    test set is built separately by `stratified_image_split` and is never
    trained on by anything.
    """
    targets = np.asarray(targets)
    cls_idx = np.where(targets == forget_class)[0]
    if cls_idx.size == 0:
        raise ValueError(f"class {forget_class} has no samples")
    other_idx = np.where(targets != forget_class)[0]

    if mode == "all":
        return ForgetSplit(forget_class, cls_idx, other_idx,
                           np.array([], dtype=int), mode)

    if mode == "subset":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(cls_idx)
        n = max(1, int(round(forget_fraction * len(perm))))
        if n >= len(perm):
            raise ValueError(
                "forget_fraction leaves no forget-heldout images; "
                "use mode='all' or lower the fraction"
            )
        return ForgetSplit(forget_class, perm[:n], other_idx, perm[n:], mode)

    raise ValueError(f"unknown mode '{mode}' (expected 'all' or 'subset')")


def stratified_subsample(targets: Sequence[int], per_class: int, seed: int = 0) -> np.ndarray:
    """Keep every class present, capped at `per_class` samples each.

    Used by --smoke to build a tiny training subset that still has every
    class present. Anything derived from the full `targets` array downstream
    (a forget/retain split) needs `restrict_split` against this subset too --
    see restrict_split's docstring.
    """
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        keep.extend(rng.choice(idx, min(per_class, len(idx)), replace=False))
    return np.array(sorted(keep))


def restrict_split(split: ForgetSplit, allowed_idx: Sequence[int]) -> ForgetSplit:
    """Intersect every index set in `split` with `allowed_idx`.

    `make_forget_split` is computed from the full target array regardless of
    any upstream subsampling. Without this, a forget/retain split built under
    --smoke silently falls back to the full-size forget/retain sets the
    moment unlearning is enabled -- "smoke" stops meaning "tiny subset."
    """
    allowed = set(int(i) for i in allowed_idx)

    def _restrict(idx: np.ndarray) -> np.ndarray:
        return np.array([i for i in idx if int(i) in allowed], dtype=idx.dtype)

    return ForgetSplit(
        forget_class=split.forget_class,
        forget_idx=_restrict(split.forget_idx),
        retain_idx=_restrict(split.retain_idx),
        forget_heldout_idx=_restrict(split.forget_heldout_idx),
        mode=split.mode,
    )


def nearest_identities(class_means: np.ndarray, target: int, k: int = 5) -> np.ndarray:
    """
    The k identities closest to `target` in feature space.

    Used for the similar-identity control: forgetting one person should not
    damage recognition of people who look like them. Without this control,
    the research question is trivially satisfiable -- you can always forget
    someone by degrading the model for everyone.
    """
    mu = class_means / (np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-12)
    sims = mu @ mu[target]
    sims[target] = -np.inf
    valid = np.where(np.isfinite(sims))[0]
    return valid[np.argsort(-sims[valid])[:k]]


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------

def build_datasets(cfg: dict, log: Optional[Callable] = None):
    """Returns (train, train_eval, test, num_classes).

    `train_eval` is the training set with EVAL transforms -- needed for clean
    feature extraction. Extracting features under random crops and flips
    would add noise to every geometry measurement. It indexes the SAME images
    as `train`, in the same order, so index i means one image under both.
    """
    name = cfg["name"].lower()
    root = cfg.get("root", "./data")
    seed = cfg.get("seed", 0)

    if name in ("cifar10", "cifar100"):
        cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        train = cls(root, train=True, download=True, transform=cifar_transforms(True))
        train_eval = cls(root, train=True, download=False, transform=cifar_transforms(False))
        test = cls(root, train=False, download=True, transform=cifar_transforms(False))
        return train, train_eval, test, (10 if name == "cifar10" else 100)

    if name == "faces":
        size = cfg.get("image_size", 112)
        # ONE scan of the directory. Every split below is a view of it, so the
        # label->identity mapping is shared by construction rather than by
        # coincidence. Building a second FaceFolder for the test side used to
        # produce a test set 100% identical to train, silently.
        ds = FaceFolder(root,
                        max_identities=cfg.get("max_identities"),
                        min_images=cfg.get("min_images", 10),
                        max_images_per_identity=cfg.get("max_images_per_identity"),
                        seed=seed)

        train_idx, test_idx = stratified_image_split(
            ds.targets, test_fraction=cfg.get("test_fraction", 0.2), seed=seed)

        # train and train_eval: IDENTICAL indices, different transforms.
        train = Subset(ds.with_transform(face_transforms(True, size)), train_idx)
        train_eval = Subset(ds.with_transform(face_transforms(False, size)), train_idx)
        test = Subset(ds.with_transform(face_transforms(False, size)), test_idx)

        if log:
            log(f"faces: {ds.num_classes} identities, {len(ds)} images -> "
                f"train {len(train_idx)} / test {len(test_idx)}, split by image "
                f"within identity (test_fraction="
                f"{cfg.get('test_fraction', 0.2)}, seed={seed})")
        return train, train_eval, test, ds.num_classes

    raise ValueError(f"unknown dataset '{name}'")


def get_targets(ds) -> np.ndarray:
    # Subsets carry no targets of their own; take the underlying dataset's and
    # reindex, so returned targets line up with the subset's own index space.
    if isinstance(ds, Subset):
        return get_targets(ds.dataset)[np.asarray(ds.indices)]
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    raise AttributeError("dataset exposes no targets/labels")


def make_loader(ds, indices: Sequence[int] | None, batch_size: int,
                shuffle: bool, num_workers: int = 4, drop_last: bool = False) -> DataLoader:
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)
