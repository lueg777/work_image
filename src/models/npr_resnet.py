"""NPR backbone with a pluggable residual operator.

Layout is a trimmed, weight-compatible copy of
`chuangchuangtan/NPR-DeepfakeDetection/networks/resnet.py`:

    x -> residual_op -> conv1(3,64,k3,s2,p1) -> bn1 -> relu
      -> maxpool(k3,s2,p1)
      -> layer1 (Bottleneck, 64,  blocks=3)
      -> layer2 (Bottleneck, 128, blocks=4, stride 2)
      -> AdaptiveAvgPool2d(1) -> fc1 Linear(512, 1)

Two deliberate differences from torchvision:

1.  The classifier is named `fc1`, and `layer3`/`layer4` are *not*
    instantiated. The head therefore reads 512 features straight off
    `layer2`, which is exactly the 1.44M-parameter configuration reported
    in the NPR paper (and the one the released checkpoint was trained in).
2.  `forward` runs the input through `self.op` first. The operator owns the
    residual normalisation (`scale`), so the official `2/3` factor lives in
    `NearestRoundTripOp.scale` and is not duplicated here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..ops.residual_ops import make_residual_op

__all__ = ["NPRResNet", "build_model", "count_parameters"]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class NPRResNet(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (3, 4),
        num_classes: int = 1,
        op: str = "npr_nearest",
        op_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.op_name = op
        self.op = make_residual_op(op, **(op_kwargs or {}))

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [Bottleneck(self.inplanes, planes, stride)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.op(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.avgpool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict) -> NPRResNet:
    """Build from a config dict (see configs/default.yaml)."""
    m = cfg.get("model", {})
    return NPRResNet(
        layers=tuple(m.get("layers", (3, 4))),
        num_classes=int(m.get("num_classes", 1)),
        op=cfg.get("op", "npr_nearest"),
        op_kwargs=m.get("op_kwargs") or {},
    )
