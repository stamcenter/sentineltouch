
import os
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SOCOFingDataset(Dataset):
    def __init__(self, root_dir, transform=None, size: tuple = (224, 224)):
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith('.bmp')
        ]

        self.transform = transform
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)
        x1 = Image.open(image_path).convert('L')

        if self.transform:
            x1 = self.transform(x1)
        else:
            x1 = self.resize(x1)
            x1 = self.to_tensor(x1)

        # Create x2
        if random.random() > 0.5:
            x2 = Image.open(image_path).convert('L')
            y = 1.0
        else:
            not_chosen_idx = random.choice([i for i in range(len(self.image_files)) if i != idx])
            x2 = Image.open(os.path.join(self.root_dir, self.image_files[not_chosen_idx])).convert('L')
            y = 0.0

        if self.transform:
            x2 = self.transform(x2)
        else:
            x2 = self.resize(x2)
            x2 = self.to_tensor(x2)

        y = torch.tensor([y], dtype=torch.float32)

        return (x1, x2), y
