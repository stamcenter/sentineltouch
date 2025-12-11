
import math
from typing import Literal, List

import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """Configurable LeNet-5.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of output classes.
        activation: 'relu' (modern default) or 'tanh' (closer to the 1998 paper).
        pool: 'max' (modern default) or 'avg' (closer to the 1998 paper).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: List[int] = [6, 16, 256, 120, 84],
        activation: Literal['relu', 'tanh'] = 'relu',
        pool: Literal['max', 'avg'] = 'avg',
    ) -> None:
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=5, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=5, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])

        if activation == 'relu':
            self.act: nn.Module = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

        if pool == 'max':
            self.pool: nn.Module = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool must be 'max' or 'avg'")

        # Classifier with BatchNorm1d
        self.fc1 = nn.Linear(channels[2], channels[3], bias=False)
        self.bn3 = nn.BatchNorm1d(channels[3])

        self.fc2 = nn.Linear(channels[3], channels[4], bias=False)
        self.bn4 = nn.BatchNorm1d(channels[4])

        self.fc3 = nn.Linear(channels[4], num_classes)


        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        # Kaiming for ReLU, Xavier for Tanh
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if activation == 'relu':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if activation == 'relu':
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, 32, 32]
        x = self.pool(self.act(self.bn1(self.conv1(x))))  # -> [B, 6, 14, 14]
        x = self.pool(self.act(self.bn2(self.conv2(x))))  # -> [B, 16, 5, 5]
        x = torch.flatten(x, 1)                           # -> [B, 400]

        x = self.act(self.bn3(self.fc1(x)))               # -> [B, 120]
        x = self.act(self.bn4(self.fc2(x)))               # -> [B, 84]
        x = self.fc3(x)                                   # -> [B, num_classes]
        return x

def lenet5(num_classes: int = 10, 
           in_channels=1,
           channels: List[int] = [6, 16, 256, 120, 84],
           activation: str = 'relu', 
           pool: str = 'avg') -> LeNet5:
    """LeNet-5 for MNIST-like grayscale data (1x32x32)."""
    return LeNet5(in_channels=in_channels, channels=channels, num_classes=num_classes, activation=activation, pool=pool)  # type: ignore[arg-type]

if __name__ == "__main__":
    # Quick self-test
    model = LeNet5(in_channels=1,
                num_classes=10, 
                activation='relu',
                pool='avg')
    x = torch.randn(4, 1, 32, 32)
    y = model(x)
    print(model)
    print("Output shape:", y.shape)
