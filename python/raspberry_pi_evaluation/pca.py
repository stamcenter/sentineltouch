import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# -----------------------
# Settings
# -----------------------
torch.set_num_threads(1)  # use one CPU thread for consistency
N_SAMPLES = 300
N_COMPONENTS = 209
INPUT_SHAPE = (1, 350, 225)
RESIZE = (28, 28)

N_RUNS = 100

# -----------------------
# Create fake dataset
# -----------------------
X = torch.randn(N_SAMPLES, *INPUT_SHAPE)
X_flat = X.view(N_SAMPLES, -1).numpy()  # flatten to (N, features)

# -----------------------
# Fit scaler + PCA
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

pca = PCA(n_components=N_COMPONENTS)
X_reduced = pca.fit_transform(X_scaled)

# -----------------------
# Create a single datapoint
# -----------------------
x = torch.randn(*INPUT_SHAPE).numpy()
x_flat = x.reshape(1, -1)

# -----------------------
# Timing for transform
# -----------------------
transform_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    x_scaled = scaler.transform(x_flat)
    _ = pca.transform(x_scaled)
    end = time.perf_counter()
    transform_times.append(end - start)

transform_times = np.array(transform_times)

# -----------------------
# Timing for reconstruction
# -----------------------
reconstruct_times = []
x_scaled = scaler.transform(x_flat)
x_reduced = pca.transform(x_scaled)

for _ in range(N_RUNS):
    start = time.perf_counter()
    x_reconstructed = pca.inverse_transform(x_reduced)
    x_reconstructed = scaler.inverse_transform(x_reconstructed)
    end = time.perf_counter()
    reconstruct_times.append(end - start)

reconstruct_times = np.array(reconstruct_times)

# -----------------------
# Results
# -----------------------
print(f"PCA projection:     mean = {transform_times.mean():.6f} s, std = {transform_times.std():.6f} s")
print(f"PCA reconstruction: mean = {reconstruct_times.mean():.6f} s, std = {reconstruct_times.std():.6f} s")
print(f"Original shape: {x.shape}, Reduced shape: {x_reduced.shape}, Reconstructed shape: {x_reconstructed.shape}")
