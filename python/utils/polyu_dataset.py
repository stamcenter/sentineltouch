
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import re
import torch

class PolyUDatasetContact(Dataset):
    def __init__(self, root_dir, type: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(root_dir) if f.endswith('.jpg')
        ])

        if type == 'train1':
            # only take images with a prefix converted into an int less than < 136
            self.image_files = [f for f in self.image_files if int(f.split('_')[0]) < 136]
        elif type == 'train2':
            self.image_files = self.image_files
        elif type == 'val':
            # only take images with a prefix converted into an int greater than or equal to 136
            self.image_files = [f for f in self.image_files if int(f.split('_')[0]) >= 136]
        
        self.labels = [int(f.split('_')[0]) for f in self.image_files]

        # Normalize labels to [0, N-1]
        unique_ids = sorted(set(self.labels))
        self.label_map = {orig: i for i, orig in enumerate(unique_ids)}
        self.labels = [self.label_map[label] for label in self.labels]
        # print(f'labels:{self.labels}\n')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # image = Image.open(img_path).convert("RGB")  # or .convert("L") for grayscale
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


class PolyUDatasetContactless(Dataset):
    def __init__(self, root_dir, type: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_path = root_dir
        if type == 'train':
            file = "./metadata/polyu_meta.txt"
        elif type == 'val':
            file = "./metadata/polyu_meta_eval.txt"

        with open(file, 'r') as f:
            self.x = [x for x in f]

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        entry = self.x[idx].split("|")
        # replace "/" with "/p" and replace "_" with "/"
        parts = entry[0].split('/')
        parts[1] = re.sub(r'_', '/p', parts[1])
        parts[1] = "p" + parts[1]

        entry[0] = '/'.join(parts)
        
        
        image_with_path = os.path.join(self.root_dir, entry[0])

        image = Image.open(image_with_path).convert('L')

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        return image, torch.as_tensor(int(entry[1]), dtype=torch.long)
