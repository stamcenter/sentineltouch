import torch.nn as nn
import torch
from typing import Literal

from utils.resnet18 import resnet18
from utils.lenet5 import lenet5

class FingerCentroids(torch.nn.Module):
    def __init__(self, n_ids, n_dim=16):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (n_ids, n_dim)))

    def forward(self, x):
        out = torch.nn.functional.linear(
            torch.nn.functional.normalize(x), 
            torch.nn.functional.normalize(self.weight)
        )

        return out

class BlindMatchModel(nn.Module):
    def __init__(
        self,  
        num_classes=1, 
        fc1_width=7, 
        channels = [64, 128, 256, 512],
        model_type: Literal['resnet18', 'lenet5'] = 'lenet5',
        num_ids = 400
    ):
        super(BlindMatchModel, self).__init__()

        self.num_classes = num_classes
        self.fc1_width = fc1_width
        self.num_ids = num_ids
        self.model_type = model_type

        print(f"Model: {self.model_type}")

        if self.model_type == 'resnet18':
            self.embeddings = resnet18(
                in_channels=1, 
                num_classes=num_classes,
                channels=channels
            )
        else:
            self.embeddings = lenet5(
                in_channels=1, 
                num_classes=num_classes,
                channels=channels,
                pool='avg'
            )

        self.logits = FingerCentroids(n_ids=self.num_ids, n_dim=num_classes)

    def forward(self, x):
        embeddings = self.embeddings(x)
        norm_embeddings =  torch.nn.functional.normalize(embeddings)
        return self.logits(norm_embeddings)

    def get_embedding(self, x):
        embeddings = self.embeddings(x)
        return torch.nn.functional.normalize(embeddings)
