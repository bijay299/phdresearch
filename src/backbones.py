"""
Backbones. These return FEATURES only -- never logits.

The classification head is a separate object (see heads.py) so that the
loss function can be swapped without touching anything else. That separation
is the experimental design, not a style choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):
    """
    ResNet feature extractor.

    `small_input=True` swaps the 7x7 stride-2 stem for a 3x3 stride-1 stem
    and drops the maxpool -- standard practice for 32x32 CIFAR images, where
    the ImageNet stem would throw away most of the spatial resolution before
    the first block.

    `feat_dim` inserts a linear embedding layer after pooling. Face
    recognition conventionally uses 512-d embeddings; leave it None to use
    the backbone's native width (512 for ResNet-18, 2048 for ResNet-50).
    """

    def __init__(self, arch: str = "resnet18", small_input: bool = False,
                 feat_dim: int | None = None, pretrained: bool = False):
        super().__init__()
        if not hasattr(torchvision.models, arch):
            raise ValueError(f"unknown arch '{arch}'")

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, arch)(weights=weights)
        native = net.fc.in_features
        net.fc = nn.Identity()

        if small_input:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()

        self.net = net
        self.native_dim = native

        if feat_dim is None or feat_dim == native:
            self.embed = nn.Identity()
            self.feat_dim = native
        else:
            # BN after the embedding is standard in face recognition pipelines;
            # it stabilises training when the head normalises features.
            self.embed = nn.Sequential(
                nn.Linear(native, feat_dim, bias=False),
                nn.BatchNorm1d(feat_dim),
            )
            self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.net(x))


def build_backbone(arch: str = "resnet18", small_input: bool = False,
                   feat_dim: int | None = None, pretrained: bool = False) -> ResNetBackbone:
    return ResNetBackbone(arch=arch, small_input=small_input,
                          feat_dim=feat_dim, pretrained=pretrained)
