import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from data import BuildingTimeSeriesDataset
from config import BATCH_SIZE, NUM_WORKERS
from utils import visualize_time_series, compute_accuracy, compute_mae
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="anysat.log", encoding="utf-8", level=logging.INFO)

# ------------------------------------------------------------------------------
# 1. Chargement des datasets
# ------------------------------------------------------------------------------

ROOT_PATH = "/home/LBaron/ia/output_vrai"  # À adapter selon l'environnement

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

# ------------------------------------------------------------------------------
# 3. Boucle d'entraînement (à compléter selon le modèle choisi)
# ------------------------------------------------------------------------------
#
# Schéma général attendu :
#
model          = AnySat
loss_function  = torch.nn.functional.cross_entropy
optimizer      = torch.optim.Adam(model.parameters(), lr=1e-4)
classifier = torch.nn.Linear(1536, 1)
n_epochs       = 1 # 100

for epoch in range(n_epochs):
    
    # -- Phase entraînement --
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0

    for batch in tqdm(train_loader):
        images    = batch['images']    # (B, T, 4, H, W)
        emprise   = batch['emprise']   # (B, H, W)
        frame_id  = batch['frame_id']  # (B,) — label : indice temporel d'apparition

        # aplatir les dimensions B et T pour l'entrée du modèle
        B, T, C, H, W = images.shape
        images_flat = images.view(B * T, C, H, W)  # (B*T, C, H, W)

        output = model({"aerial": images_flat}, patch_size=20, output="dense", output_modality="aerial")
        
        # désaplatir la sortie pour retrouver les bonnes dimensions 
        H_out, W_out, C_feat = output.shape[1], output.shape[2], output.shape[3]
        out = output.contiguous().view(B, T, H_out, W_out, C_feat)             # (B,T,H_out,W_out,C)

        # redimensionner l'emprise pour correspondre à la taille de sortie du modèle
        emprise_resized = torch.nn.functional.interpolate(emprise.unsqueeze(1).float(), size=(H_out, W_out), mode="nearest")
        denom = emprise_resized.sum(dim=(2,3)).clamp(min=1e-6)                  # (B,1) nombre de pixels dans l'emprise (pour faire une moyenne)

        pooled = (out * emprise_resized.unsqueeze(-1)).sum(dim=(2,3)) / denom.unsqueeze(-1) 

        # classifier -> logits (B,T)
        logits = classifier(pooled).squeeze(-1)    
                
        loss = loss_function(logits, frame_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred       = torch.argmax(logits, dim=1)          # (B,)
        epoch_acc += compute_accuracy(pred, frame_id)
        epoch_loss += loss.item()

    logger.info(f"[Epoch {epoch}] Train loss: {epoch_loss / len(train_loader):.4f} "
        f"| Acc: {epoch_acc / len(train_loader):.1f}%")

    # -- Phase évaluation --
    model.eval()
    epoch_loss, epoch_acc, epoch_mae = 0.0, 0.0, 0.0
    dict_date_apparition = [{}]
    entetes = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images   = batch['images']
            frame_id = batch['frame_id']
            emprise   = batch['emprise']
            years = batch["years"]  # (B, T)
            building_id = batch["building_id"]  # (B,) 

             # aplatir les dimensions B et T pour l'entrée du modèle
            B, T, C, H, W = images.shape
            images_flat = images.view(B * T, C, H, W)  # (B*T, C, H, W)

            output = model({"aerial": images_flat}, patch_size=20, output="dense", output_modality="aerial")
            
            # désaplatir la sortie pour retrouver les bonnes dimensions 
            H_out, W_out, C_feat = output.shape[1], output.shape[2], output.shape[3]
            out = output.contiguous().view(B, T, H_out, W_out, C_feat)             # (B,T,H_out,W_out,C)

            # redimensionner l'emprise pour correspondre à la taille de sortie du modèle
            emprise_resized = torch.nn.functional.interpolate(emprise.unsqueeze(1).float(), size=(H_out, W_out), mode="nearest")
            denom = emprise_resized.sum(dim=(2,3)).clamp(min=1e-6)                  # (B,1) nombre de pixels dans l'emprise (pour faire une moyenne)

            pooled = (out * emprise_resized.unsqueeze(-1)).sum(dim=(2,3)) / denom.unsqueeze(-1) 

            # classifier -> logits (B,T)
            logits = classifier(pooled).squeeze(-1) 
            
            pred   = torch.argmax(logits, dim=1)

            entetes.append(building_id[0])
            dict_date_apparition[0][building_id[0]] = years[0][pred[0]].item()

            epoch_acc  += compute_accuracy(pred, frame_id)
            epoch_mae  += compute_mae(pred, frame_id)
            epoch_loss += loss_function(logits, frame_id).item()

    logger.info(f"[Epoch {epoch}] Val   loss: {epoch_loss / len(val_loader):.4f} "
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames")
        
    with open("resultats_AnySat.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=entetes)
        writer.writeheader()
        writer.writerows(dict_date_apparition)