import os
import torch
import csv
from itertools import product
from utils.blindmatch_model import BlindMatchModel
from utils.lenet5 import lenet5

# Your configuration lists
embedding_spaces = [16, 32]
datasets = ['polyu']
model_types = ['lenet5']

# Current date for file naming
# current_date = datetime.now().strftime('%Y-%m-%d')
current_date = "2025-08-24"

import torch.nn as nn

def fold_bn_into_conv(conv, bn):
    """Fold BatchNorm2d into a preceding Conv2d."""
    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    # Scale
    scale = gamma / torch.sqrt(var + eps)

    # Fold weights and bias
    W_folded = W * scale.reshape([-1, 1, 1, 1])
    b_folded = (b - mean) * scale + beta

    # Copy back into conv
    conv.weight.data.copy_(W_folded)
    conv.bias = nn.Parameter(b_folded)

    return conv

def fold_bn_into_linear(fc: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    """Fold BatchNorm1d parameters into a preceding Linear layer."""
    W = fc.weight
    b = fc.bias if fc.bias is not None else torch.zeros(W.size(0), device=W.device)

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    # Fold weights + bias
    W_folded = W * scale.unsqueeze(1)   # scale each output row
    b_folded = (b - mean) * scale + beta

    fc.weight.data.copy_(W_folded)
    fc.bias = nn.Parameter(b_folded)

    return fc


def fold_lenet5_batchnorm(model: nn.Module):
    # Conv layers
    model.conv1 = fold_bn_into_conv(model.conv1, model.bn1)
    model.bn1 = nn.Identity()

    model.conv2 = fold_bn_into_conv(model.conv2, model.bn2)
    model.bn2 = nn.Identity()

    # Linear layers (only if BN exists after them)
    if hasattr(model, "bn3"):
        model.fc1 = fold_bn_into_linear(model.fc1, model.bn3)
        model.bn3 = nn.Identity()
    if hasattr(model, "bn4"):
        model.fc2 = fold_bn_into_linear(model.fc2, model.bn4)
        model.bn4 = nn.Identity()
    if hasattr(model, "bn5"):
        model.fc3 = fold_bn_into_linear(model.fc3, model.bn5)
        model.bn5 = nn.Identity()

    return model

def fold_mlp_batchnorm(model: nn.Module) -> nn.Module:
    """Fold all Linear+BatchNorm1d pairs inside an MLP Sequential network."""
    layers = []
    skip = False
    for i, layer in enumerate(model.net):
        if skip:
            skip = False
            continue

        # Detect Linear + BN pair
        if isinstance(layer, nn.Linear) and i+1 < len(model.net) and isinstance(model.net[i+1], nn.BatchNorm1d):
            fc = layer
            bn = model.net[i+1]
            folded = fold_bn_into_linear(fc, bn)
            layers.append(folded)
            skip = True   # skip BN
        else:
            layers.append(layer)

    model.net = nn.Sequential(*layers)
    return model


# Model builder
def build_model(model_type, embedding_dim):
    if model_type == "lenet5":
        return lenet5(num_classes=embedding_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
# Main export loop
for dataset, model_type, emb_dim in product(datasets, model_types, embedding_spaces):

    # Paths
    model_dir = f'./../trained_models/sentinal_1_to_N_pca_{dataset}_{model_type}_{emb_dim}_{current_date}.pth'
    weight_dir = f'./../weights/{model_type}_{dataset}_{emb_dim}/'

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    # Build model
    
    if model_type == "lenet5":
        channels = [6, 16, 256, 120, 84]
    else:
        channels = [256, 128]

    model = BlindMatchModel(num_classes=emb_dim, model_type=model_type, channels=channels)

    # Load trained checkpoint BEFORE folding
    if os.path.exists(model_dir):
        model.load_state_dict(torch.load(model_dir,  map_location=torch.device("cpu")), strict=True)
        model.eval()
    else:
        print(f"WARNING: no checkpoint found at {model_dir}, using random init")

    # Now fold BN
    if model_type == "lenet5":
        model.embeddings = fold_lenet5_batchnorm(model.embeddings)

        # Export only conv/fc weights and biases (no BN)
        for name, param in model.state_dict().items():
            if not (name.startswith("conv") or name.startswith("fc")):
                continue
            if not ("weight" in name or "bias" in name):
                continue

            safe_name = name.replace(".", "_")   # conv1.weight -> conv1_weight
            weight_file = os.path.join(weight_dir, f"{safe_name}.csv")

            # Convert tensor -> numpy
            data = param.detach().cpu().numpy().flatten()
            with open(weight_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write as a single row
                writer.writerow(data.tolist())

    print(f"Exported {model_type} ({emb_dim}) for {dataset}")
    print(f"  Weights -> {weight_dir}")
    print(f"  Model   -> {model_dir}")
