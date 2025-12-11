
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class SOKOTODataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None, resize=(40,40)):
        
        self.root_dir = root_dir
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith('.bmp')
        ]
        self.transform = transform
        self.resize = resize

        self.filenames = []
        self.labels = []
        with open(labels_file, "r") as f:
            for line in f:
                fname, label = line.strip().split("|")
                self.filenames.append(fname)
                self.labels.append(int(label))

        # Optionally: store original user IDs
        self.user_ids = [label + 1 for label in self.labels]

        # Total number of classes
        self.num_classes = len(set(self.user_ids))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, torch.as_tensor(label, dtype=torch.long)


def build_sokoto_meta(image_dir, output_file="sokoto_meta.txt"):
    """
    Build metadata file (TXT) mapping original filenames to user IDs.
    Sorted by user ID.
    Format: filename|user_id
    """
    records = []

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(".bmp"):
            continue
        try:
            # Extract user ID before "__"
            user_id = int(fname.split("__")[0])-1
            records.append((user_id, fname))
        except Exception as e:
            print(f"Skipping malformed file: {fname} ({e})")

    # Sort by user_id, then filename
    records.sort(key=lambda x: (x[0], x[1]))

    with open(output_file, "w") as f:
        for user_id, fname in records:
            f.write(f"{fname}|{user_id}\n")

    print(f"Metadata written to {output_file}, total records: {len(records)}")



def reorder_pca_to_metadata(X_pca, image_dir, labels_file):
    """
    Reorder PCA data so that each row matches the order in labels_file.

    Args:
        X_pca: np.ndarray of shape (num_images, n_components)
        image_dir: directory containing original images
        labels_file: path to metadata file (filename|label)

    Returns:
        X_pca_ordered: reordered PCA array
        filenames_ordered: list of filenames in the metadata order
    """
    # Load metadata
    filenames_meta = []
    with open(labels_file, "r") as f:
        for line in f:
            fname, _ = line.strip().split("|")
            filenames_meta.append(fname)

    # Build mapping from filename to row index in X_pca
    filenames_pca = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(".bmp")
    ]
    fname_to_idx = {fname: i for i, fname in enumerate(filenames_pca)}

    # Reorder PCA rows
    X_pca_ordered = np.zeros_like(X_pca)
    for i, fname in enumerate(filenames_meta):
        if fname not in fname_to_idx:
            raise ValueError(f"Filename `{fname}` from metadata not found in {image_dir}")
        X_pca_ordered[i] = X_pca[fname_to_idx[fname]]

    return X_pca_ordered, filenames_meta
