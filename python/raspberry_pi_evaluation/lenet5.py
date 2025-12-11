import torch
import torch.nn as nn
import numpy as np
import time


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ------------------------
# Quick sanity check
# ------------------------
if __name__ == "__main__":
    N_RUNS = 100

    torch.set_num_threads(1)

    # Model setup
    model = LeNet5(
        num_classes=10, 
        in_channels=1  # changed to in_channels
    )
    model = model.eval()

    # Random input
    x = torch.randn(1, 1, 28, 28)

    times = []

    with torch.inference_mode():
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    print("LeNet-5 Inference Timing over", N_RUNS, "runs")
    print("Average inference time:", times.mean())
    print("Standard deviation:", times.std())
