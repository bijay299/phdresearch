"""
Classification heads with matched interfaces.

The experiment varies ONE thing: which head sits on top of the backbone.
Everything else -- backbone, data, schedule, seed -- is held fixed.

All heads expose:
    forward(features, labels=None) -> logits
    weight                          -> (num_classes, feat_dim) tensor

`weight` matters as much as `forward`: the NC3 alignment metric compares
each row of `weight` against the corresponding class-mean feature. Keeping
the attribute name identical across heads means metrics.py never needs to
know which loss produced the model.

Note on normalisation
---------------------
ArcFace and CosFace L2-normalise BOTH the features and the weights, so the
logit is a cosine. Plain CE does not. This is exactly the asymmetry the
project is testing -- do not "fix" it for consistency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Plain linear classifier + softmax CE. The AISTATS regime."""

    def __init__(self, feat_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feat_dim, num_classes, bias=bias)
        self.normalised = False

    @property
    def weight(self) -> torch.Tensor:
        return self.fc.weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        return self.fc(features)


class ArcFaceHead(nn.Module):
    """
    Additive angular margin (Deng et al., ArcFace).

        logit_y = s * cos(theta_y + m)
        logit_j = s * cos(theta_j)          for j != y

    The margin is applied to the angle, so it penalises angular distance
    between a sample and its own class weight. That is the mechanism that
    should suppress classifier drift during unlearning.
    """

    def __init__(self, feat_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5,
                 easy_margin: bool = False):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.normalised = True

        self.W = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.W)

        # precomputed constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    @property
    def weight(self) -> torch.Tensor:
        return self.W

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.W)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # At eval time (no labels) return plain scaled cosine -- no margin.
        if labels is None:
            return self.s * cosine

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m          # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # keep the function monotonic past theta = pi - m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return self.s * logits


class CosFaceHead(nn.Module):
    """
    Additive cosine margin (Wang et al., CosFace / AM-Softmax).

        logit_y = s * (cos(theta_y) - m)

    Same normalisation as ArcFace; the margin is subtracted in cosine space
    rather than added in angular space. Including both separates "margin
    losses in general" from "ArcFace specifically".
    """

    def __init__(self, feat_dim: int, num_classes: int, s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.normalised = True

        self.W = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.W)

    @property
    def weight(self) -> torch.Tensor:
        return self.W

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.W)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return self.s * cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return self.s * (cosine - one_hot * self.m)


HEADS = {
    "ce": LinearHead,
    "arcface": ArcFaceHead,
    "cosface": CosFaceHead,
}


def build_head(name: str, feat_dim: int, num_classes: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in HEADS:
        raise ValueError(f"unknown head '{name}'; choose from {sorted(HEADS)}")
    return HEADS[name](feat_dim=feat_dim, num_classes=num_classes, **kwargs)
