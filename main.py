# ==============================================================================
# main.py — Point d'entrée du benchmark de datation d'apparition de bâtiments
# ==============================================================================
#
# Ce script :
#   1. Charge les datasets train / val / test
#   2. Initialise un modèle (exemple : AnySat depuis torch.hub)
#   3. Lance la boucle d'entraînement et d'évaluation
#   4. Visualise quelques exemples du dataset d'entraînement
# ==============================================================================

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from data import BuildingTimeSeriesDataset
from config import BATCH_SIZE, NUM_WORKERS
from utils import visualize_time_series, compute_accuracy, compute_mae


# ------------------------------------------------------------------------------
# 1. Chargement des datasets
# ------------------------------------------------------------------------------

ROOT_PATH = "/home/LBaron/git-clones/gitlab.ign.fr/exploration-bdtopo/batiment/output_vrai"  # À adapter selon l'environnement

train_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="train")
val_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="val")
test_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="test")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


# ------------------------------------------------------------------------------
# 2. Chargement du modèle
# ------------------------------------------------------------------------------

# Chargement d'AnySat (modèle pré-entraîné sur données satellitaires/aériennes)
# flash_attn=False pour la compatibilité sans GPU optimisé
AnySat = torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False)

"""
# Vérification rapide : inférence sur un sample du train
sample = train_dataset[0]
images = sample["images"]  # (T, 4, H, W)
print(f"Forme des images d'entrée : {images.shape}")

output = AnySat(
    {"aerial": images}, patch_size=20, output="dense", output_modality="aerial"
)  # Tensor
print(f"Forme de la sortie AnySat : {output.shape}")
"""


# ------------------------------------------------------------------------------
# 3. Boucle d'entraînement (à compléter selon le modèle choisi)
# ------------------------------------------------------------------------------
#
# Schéma général attendu :
#
#   model          = <votre modèle>
#   loss_function  = <votre fonction de loss>
#   optimizer      = torch.optim.Adam(model.parameters(), lr=1e-4)
#   n_epochs       = 100
#
#   for epoch in range(n_epochs):
#
#       # -- Phase entraînement --
#       model.train()
#       epoch_loss, epoch_acc = 0.0, 0.0
#
#       for batch in train_loader:
#           images    = batch['images']    # (B, T, 4, H, W)
#           emprise   = batch['emprise']   # (B, H, W)
#           frame_id  = batch['frame_id']  # (B,) — label : indice temporel d'apparition
#
#           output = model(images)         # (B, T) — score par frame
#           loss   = loss_function(output, frame_id)
#
#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()
#
#           pred       = torch.argmax(output, dim=1)          # (B,)
#           epoch_acc += compute_accuracy(pred, frame_id)
#           epoch_loss += loss.item()
#
#       print(f"[Epoch {epoch}] Train loss: {epoch_loss / len(train_loader):.4f} "
#             f"| Acc: {epoch_acc / len(train_loader):.1f}%")
#
#       # -- Phase évaluation --
#       model.eval()
#       epoch_loss, epoch_acc, epoch_mae = 0.0, 0.0, 0.0
#
#       with torch.no_grad():
#           for batch in val_loader:
#               images   = batch['images']
#               frame_id = batch['frame_id']
#
#               output = model(images)
#               pred   = torch.argmax(output, dim=1)
#
#               epoch_acc  += compute_accuracy(pred, frame_id)
#               epoch_mae  += compute_mae(pred, frame_id)
#               epoch_loss += loss_function(output, frame_id).item()
#
#       print(f"[Epoch {epoch}] Val   loss: {epoch_loss / len(val_loader):.4f} "
#             f"| Acc: {epoch_acc / len(val_loader):.1f}% "
#             f"| MAE: {epoch_mae / len(val_loader):.2f} frames")

model = AnySat
loss_function = torch.nn.functional.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 10  # 100

for epoch in range(n_epochs):

    # -- Phase entraînement --
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for batch in tqdm(train_loader):
        images = batch["images"]  # (B, T, 4, H, W)
        emprise = batch["emprise"]  # (B, H, W)
        frame_id = batch["frame_id"]  # (B,) — label : indice temporel d'apparition

        output = model(
            {"aerial": images[0]},
            patch_size=20,
            output="dense",
            output_modality="aerial",
        )  # (B, T) — score par frame

        logits = output.mean(dim=tuple(range(1, output.ndim))).unsqueeze(0)  # (1, T)

        loss = loss_function(logits, frame_id)
        # loss = loss_function(output, frame_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=1)  # (B,)
        epoch_acc += compute_accuracy(pred, frame_id)
        epoch_loss += loss.item()

    print(
        f"[Epoch {epoch}] Train loss: {epoch_loss / len(train_loader):.4f} "
        f"| Acc: {epoch_acc / len(train_loader):.1f}%"
    )

    # -- Phase évaluation --
    model.eval()
    epoch_loss, epoch_acc, epoch_mae = 0.0, 0.0, 0.0
    dict_date_apparition = [{}]
    entetes = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"]
            frame_id = batch["frame_id"]
            years = batch["years"]  # (B, T)
            building_id = batch["building_id"]  # (B,)

            output = model(
                {"aerial": images[0]},
                patch_size=20,
                output="dense",
                output_modality="aerial",
            )  # (B, T) — score par frame
            pred = torch.argmax(output, dim=1)

            epoch_acc += compute_accuracy(pred, frame_id)
            epoch_mae += compute_mae(pred, frame_id)
            epoch_loss += loss_function(output, frame_id).item()
            dict_date_apparition[0][building_id[0]] = years[0][pred[0]].item()
            entetes.append(building_id[0])

    print(
        f"[Epoch {epoch}] Val   loss: {epoch_loss / len(val_loader):.4f} "
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames"  # Mean Absolute Error
    )

    with open("resultats_AnySat.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=entetes)
        writer.writeheader()
        writer.writerows(dict_date_apparition)

# ------------------------------------------------------------------------------
# 4. Visualisation d'exemples
# ------------------------------------------------------------------------------

# Visualisation des 3 premiers bâtiments du split d'entraînement
for idx in range(3):
    visualize_time_series(train_dataset, idx=idx)
