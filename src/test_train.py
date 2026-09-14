"""
Regression tests for train.py's train_model, specifically its LR schedule.

Run:  python src/test_train.py

The bug this suite exists for: `CosineAnnealingLR` was built with
`T_max=epochs` (the raw epoch count), but `sched.step()` is only called on
post-warmup epochs -- so with `warmup_epochs > 0` the schedule only ever
receives `epochs - warmup_epochs` steps against a cycle calibrated for the
full `epochs`. Confirmed directly from real run logs: CIFAR ArcFace
(epochs=30, warmup=5) ended at lr=0.00670, not 0.00000; faces ArcFace
(epochs=40, warmup=5) ended at lr=0.00381. Every `warmup_epochs > 0` run in
the project trained with a schedule that never fully decayed. See
notes/decisions.md, 2026-09-14.
"""
import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train as TR


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise AssertionError(name)


class TinyBackbone(nn.Module):
    """Minimal stand-in fulfilling train_model's interface: features in,
    features out. Real dimensions don't matter -- only the LR schedule is
    under test, not anything about representation quality."""
    def __init__(self, feat_dim: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(4, feat_dim)

    def forward(self, x):
        return self.fc(x)


class TinyHead(nn.Module):
    """Ignores labels -- every real head is called as head(features, labels)
    during training (margin heads use labels; CE doesn't), so match that
    signature even though this head doesn't need labels itself."""
    def __init__(self, feat_dim: int = 8, num_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features, labels=None):
        return self.fc(features)


def tiny_loader(n: int = 16, batch_size: int = 4, num_classes: int = 3) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def _lr_from_log_line(line: str) -> float:
    return float(re.search(r"lr ([\d.]+)", line).group(1))


def final_lr(epochs: int, warmup_epochs: int, lr: float = 0.1) -> float:
    logs = []
    TR.train_model(TinyBackbone(), TinyHead(), tiny_loader(), device="cpu",
                   epochs=epochs, lr=lr, warmup_epochs=warmup_epochs,
                   log=lambda m: logs.append(m))
    return _lr_from_log_line(logs[-1])


def test_cosine_schedule_reaches_its_minimum_with_warmup():
    print("train_model: cosine schedule reaches ~0 by the final epoch, even with warmup")
    got = final_lr(epochs=10, warmup_epochs=3)
    check(f"final lr with warmup_epochs=3 is near 0 (got {got})", got < 1e-3)


def test_cosine_schedule_reaches_its_minimum_without_warmup():
    print("train_model: cosine schedule still reaches ~0 with no warmup (unaffected by the fix)")
    got = final_lr(epochs=10, warmup_epochs=0)
    check(f"final lr with warmup_epochs=0 is near 0 (got {got})", got < 1e-3)


def test_cosine_schedule_scales_with_epoch_budget():
    print("train_model: a longer run still reaches ~0 (not just a coincidence at epochs=10)")
    got = final_lr(epochs=40, warmup_epochs=5)
    check(f"final lr at epochs=40, warmup_epochs=5 is near 0 (got {got})", got < 1e-3)


def test_warmup_ramps_up_linearly():
    print("train_model: warmup phase ramps lr up linearly toward the target lr")
    logs = []
    TR.train_model(TinyBackbone(), TinyHead(), tiny_loader(), device="cpu",
                   epochs=3, lr=0.9, warmup_epochs=3,
                   log=lambda m: logs.append(m))
    first_ep_lr = _lr_from_log_line(logs[0])
    check(f"epoch 1 lr is target*(1/warmup_epochs) (got {first_ep_lr}, want 0.3)",
          abs(first_ep_lr - 0.3) < 1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print()
    print(f"{len(tests)} test groups passed.")
