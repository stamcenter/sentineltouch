import torch
import random
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1, _ = self.dataset[idx]

        # Create x2
        if random.random() > 0.5:
            x2, _ = self.dataset[idx]
            y = 1.0
        else:
            not_chosen_idx = random.choice([i for i in range(len(self.dataset)) if i != idx])
            x2, _ = self.dataset[not_chosen_idx]
            y = 0.0

        y = torch.tensor([y], dtype=torch.float32)

        return x1, x2, y