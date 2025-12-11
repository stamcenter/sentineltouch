import torch
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define transform for raw data (no normalization for PCA)
def pca_data_construction(dataset, batch_size=128):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_images = []
    all_labels = []

    for images, labels in dataloader:
        batch_size = images.shape[0]
        flattened = images.view(batch_size, -1)
        all_images.append(flattened.numpy())
        all_labels.append(labels.numpy())
    
    X = np.vstack(all_images)
    y = np.concatenate(all_labels)
    return X, y

def pca_generation(x_input, scaler=StandardScaler(), pca_components=0.99):
    
    # Step 1: Standardize input
    X_scaled = scaler.fit_transform(x_input)

    # Step 2: Apply PCA with desired number of components
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Explained variance summary
     # Step 3: Explained variance summary
    explained_variance = pca.explained_variance_ratio_ * 100
    total_explained = np.sum(explained_variance)
    num_pca_components = pca.n_components_
    print(f"Sum of variance explained by top {num_pca_components} components: {total_explained:.2f}%")
    return X_pca, pca, num_pca_components, scaler


class PCAReconstructedSokotoDataset(Dataset):
    def __init__(self, X_pca, pca_model, labels_file, transform, scaler, image_shape):
        self.image_shape = image_shape
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        # Load metadata
        self.filenames = []
        self.labels = []
        with open(labels_file, "r") as f:
            for line in f:
                fname, label = line.strip().split("|")
                self.filenames.append(fname)
                self.labels.append(int(label))

        self.num_classes = len(set(self.labels))

        # PCA inverse transform
        self.X_reconstructed = pca_model.inverse_transform(X_pca)
        self.X_unscaled = scaler.inverse_transform(self.X_reconstructed)

        # Sanity check
        assert len(self.X_unscaled) == len(self.labels), "Mismatch between PCA data and labels!"

    def __len__(self):
        return len(self.X_unscaled)

    def __getitem__(self, idx):
        img_array = self.X_unscaled[idx].reshape(self.image_shape).astype(np.float32)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        pil_image = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')

        if self.transform:
            image = self.transform(pil_image)
        else:
            image = self.to_tensor(pil_image)

        label = self.labels[idx]

        # Ensure label is valid
        if not (0 <= label < self.num_classes):
            raise ValueError(f"Label {label} out of range [0, {self.num_classes-1}]")

        return image, torch.as_tensor(label, dtype=torch.long)

class PCAReconstructedPolyUContactless(Dataset):
    def __init__(self, data, pca_model, scaler, resize_shape, root_dir, type: str, return_components: bool = False):
        """
        Args:
            data: Normal, non-PCA dataset
            pca_model: fitted PCA object
            scaler: fitted scaler used before PCA
            root_dir: root directory (used only to keep consistency with original class)
            type: 'train' or 'val' (loads same metadata file as original)
            transform: torchvision transforms or None
            scaler: fitted scaler used before PCA
            image_shape: (H, W) tuple for grayscale images
            return_components: whether to return PCA components or reconstructed data
        """
        self.data = data
        self.pca_model = pca_model
        self.scaler = scaler

        self.root_dir = root_dir
        self.to_tensor = transforms.ToTensor()
        # self.resize = transforms.Resize(resize_shape) if resize_shape else None
        self.resize = lambda x: F.interpolate(x.unsqueeze(0).unsqueeze(0), size=resize_shape, mode="bilinear", align_corners=False)[0,0]

        self.return_components = return_components

        # Load same metadata as original class
        if type == 'train':
            file = "./metadata/polyu_meta.txt"
        elif type == 'val':
            file = "./metadata/polyu_meta_eval.txt"
        else:
            raise ValueError("type must be 'train' or 'val'")

        with open(file, 'r') as f:
            self.x = [x.strip() for x in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This automatically applies transforms to data
        x, label = self.data[idx]

        x_pca = self.pca_model.transform(self.scaler.transform(x.reshape(1, -1)))
        if self.return_components:
            # Get PCA components for this index
            return torch.from_numpy(x_pca[0]).float(), torch.as_tensor(label, dtype=torch.long)

        # Get PCA-reconstructed image array for this index
        x_inv_pca = self.pca_model.inverse_transform(x_pca).reshape(x.shape)
        x_inv_pca = torch.from_numpy(x_inv_pca).float()
        x_inv_pca_resize = self.resize(x_inv_pca)

        return x_inv_pca_resize, torch.as_tensor(label, dtype=torch.long)
    
def pca_collate_fn(batch, pca_model, scaler, resize_shape, return_components=False):
    ####################################################################
    # Moved this functionality into the collation function so that we  #
    # can operate on batches of data and greatly speed up computations #
    ####################################################################
    # batch is a list of (x, label) pairs
    # xs, labels = zip(*batch)

    is_pair_dataset = False

    try:
        xs, labels = zip(*batch)
        B = len(xs)
        C, H, W = xs[0].shape

        # Stack into (B, D) numpy array for PCA
        X = np.stack(xs, axis=0).reshape(B, C*H*W)
    except ValueError:
        x1s, x2s, labels = zip(*batch)
        B = len(x1s)
        C, H, W = x1s[0].shape

        labels = np.stack(labels, axis=0)

        X1 = np.stack(x1s, axis=0).reshape(B, C*H*W)
        X2 = np.stack(x2s, axis=0).reshape(B, C*H*W)

        X = np.concatenate((X1, X2), axis=0)
        B = B * 2  # Concatenate pairs
        is_pair_dataset = True

    
    # Scale + PCA transform
    X_scaled = scaler.transform(X)
    X_pca = pca_model.transform(X_scaled)

    if return_components:
        # Just return PCA components
        X_out = torch.from_numpy(X_pca).float()
    else:
        # Reconstruct
        X_inv = pca_model.inverse_transform(X_pca)
        X_inv = torch.from_numpy(X_inv).float()  # (B, D)

        # reshape each back to image
        X_inv = X_inv.view(B, C, H, W)

        # Use functional interpolation to avoid needing to convert back to PIL
        # This also supports batch transforms to further accelerate pipeline
        X_out = F.interpolate(X_inv, size=resize_shape, mode="bilinear", align_corners=False)
    
    labels = torch.as_tensor(labels, dtype=torch.long)

    # Unconcat if you concated above
    if is_pair_dataset:
        x1s = X_out[:B//2]
        x2s = X_out[B//2:]
        return x1s, x2s, labels

    return X_out, labels


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

        return (x1, x2), y


