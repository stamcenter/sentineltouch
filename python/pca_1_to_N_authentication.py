import sys
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if '__file__' in globals():
    script_dir = os.path.dirname(__file__)
else:
    script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, './'))

from utils.blindmatch_model import BlindMatchModel
from utils.training_functions_1_to_N_blind_match import train_model, evaluate_1n_model, save_embeddings
from utils.pca_module import pca_data_construction, pca_generation, pca_collate_fn

from utils.split_weights import split_weights
from utils.sokoto_dataset import SOKOTODataset
from utils.polyu_dataset import PolyUDatasetContactless
from utils.loss import ArcFace

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")

np.random.seed(seed)
random.seed(seed)

PCA_ENABLED = True

# Standardize
# input_width = 40
input_width = 28
input_dim = input_width*input_width

original_size = (350, 225)
new_size = (input_width, input_width)
fc1_width = 2
batch_size = 256

# Model's seem to be struggling or taking a long time to initally learn when using smaller embeddings spaces.
embedding_space = 16

scaler = StandardScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'lenet5'

resnet18_channels = [16, 32, 64, 128]
lenet5_channels = [6, 16, 256, 120, 84]

###########################################################################################
##################################    PolyU    ############################################
###########################################################################################

num_epochs = 100
train_transform = transforms.Compose([
    torchvision.transforms.RandomRotation(20),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (1.0)),
])

polyu_location = "./../images/Cross_Fingerprint_Images_Database/processed_contactless_2d_fingerprint_images"
trainset = PolyUDatasetContactless(root_dir=polyu_location, type='train', transform=train_transform)
validationset = PolyUDatasetContactless(root_dir=polyu_location, type='val', transform=train_transform)

dataset_name = "polyu"
num_ids = 400

# ###########################################################################################
# ##################################    Sokoto    ###########################################
# ###########################################################################################

# num_epochs = 250

# train_transform = transforms.Compose([
#     torchvision.transforms.RandomRotation(20),
#     torchvision.transforms.Resize((96, 108)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize((0.5), (1.0)),
# ])

# scoofing_db = "./../images/SCOOF_DB/SOCOFing/Real"
# labels_file="./metadata/sokoto_meta.txt"
# sokoto = SOKOTODataset(root_dir=scoofing_db, labels_file=labels_file, transform=train_transform)
# trainset, validationset = train_test_split(
#     sokoto, test_size=0.4, random_state=seed, shuffle=True
# )

# num_ids = 600
# dataset_name = "sokoto"

###########################################################################################
##################################    END    ##############################################
###########################################################################################

if PCA_ENABLED:
    pca_elements = 784
    # pca_elements = 0.95

    x, y = pca_data_construction(trainset)
    x_pca, pca_model, total_pca, scaler = pca_generation(x, scaler, pca_elements)

    def collate_fn(batch):
        return pca_collate_fn(
            batch,
            pca_model,
            scaler,
            new_size,
            return_components=False
        )
else:
    collate_fn = None

# -----------------------------------------------------
# Make data loaders
# -----------------------------------------------------
train_dataloader = DataLoader(
    trainset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate_fn
)

validation_loader = DataLoader(
    validationset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate_fn
)

if model_type == "resnet18":
    blind_match_model = BlindMatchModel(num_classes=embedding_space,
                                        model_type='resnet18',
                                        fc1_width=fc1_width, 
                                        channels=resnet18_channels,
                                        num_ids=num_ids
                                    ).to(device)
else:
    blind_match_model = BlindMatchModel(num_classes=embedding_space,
                                        model_type='lenet5',
                                        fc1_width=fc1_width, 
                                        channels=lenet5_channels,
                                        num_ids=num_ids
                                    ).to(device)

arcface = ArcFace(m=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(split_weights(blind_match_model), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
current_date = datetime.now().strftime('%Y-%m-%d')

print("Number of training images:", len(trainset))
print("Number of validation images:", len(validationset))


train_model(
    model=blind_match_model, 
    train_dataloader=train_dataloader, 
    arcface=arcface, 
    criterion=criterion, 
    optimizer=optimizer, 
    val_loader=validation_loader, 
    scheduler=scheduler, 
    num_epochs=num_epochs,
    device=device, 
    model_name=f"pca_{dataset_name}_{model_type}_{embedding_space}_{current_date}"
)
blind_match_model.eval()
evaluate_1n_model(blind_match_model, validation_loader, device)

validation_loader = DataLoader(
    validationset,
    batch_size=1,
    shuffle=False,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate_fn
)
save_dir = "../embedding_csv"
# save_embeddings(blind_match_model, train_dataloader, device, save_dir, dataset_name, embedding_space, model_type, "authenticate")

print("\n\n")
validation_loader = DataLoader(
    validationset,
    batch_size=1,
    shuffle=False,
    num_workers=6,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=collate_fn
)
# save_embeddings(blind_match_model, validation_loader, device, save_dir, dataset_name, embedding_space, model_type, "register")
