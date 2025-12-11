import torch
import torch.nn as nn
from typing import Type, Callable, Optional, List
import numpy as np
import time

# ------------------------
# Basic building blocks
# ------------------------

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """ResNet BasicBlock used in ResNet-18/34."""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # First conv
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # Second conv
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ------------------------
# ResNet backbone
# ------------------------

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        global_pool: str = "avg",  # "avg" or "max"
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # Each element corresponds to if we should replace the 2x stride with dilation in layer2/3/4
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be None or a 3-element list")

        self.groups = groups
        self.base_width = width_per_group

        # Stem
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(block, channels[0],  layers[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Head
        if global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-initialize the last BN in each residual branch, so residual branch starts with zeros.
        # This improves training for very deep nets, per He et al. (optional).
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ------------------------
# Factory: ResNet-18
# ------------------------

def resnet18(num_classes: int = 1000, 
             in_channels: int = 3, 
             channels: List[int] = [64, 128, 256, 512],
             **kwargs) -> ResNet:
    """Constructs a ResNet-18 model."""
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        channels=channels,
        **kwargs,
    )


# ------------------------
# Quick sanity check
# ------------------------
if __name__ == "__main__":
    EMEDDING_DIM = 128
    N_RUNS = 100

    torch.set_num_threads(1)

    # Model setup
    model = resnet18(
        num_classes=EMEDDING_DIM, 
        in_channels=1  # changed to in_channels
    )
    model = model.eval()

    # Random input
    x = torch.randn(1, 1, 224, 224)

    times = []

    with torch.inference_mode():
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    print("ResNet-18 Inference Timing over", N_RUNS, "runs")
    print("Average inference time:", times.mean())
    print("Standard deviation:", times.std())
