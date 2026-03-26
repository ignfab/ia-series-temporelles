import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import BuildingTimeSeriesDataset
from config import BATCH_SIZE, NUM_WORKERS, SEUIL_BATIMENT
from utils import visualize_time_series, compute_accuracy, compute_mae
import matplotlib.pyplot as plt
import rasterio

# ------------------------------------------------------------------------------
# 1. Chargement des datasets
# ------------------------------------------------------------------------------

ROOT_PATH = "/home/LBaron/git-clones/gitlab.ign.fr/exploration-bdtopo/batiment/output_vrai"  # À adapter selon l'environnement

val_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="val", norm=True)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

model_name = "FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth"
model = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=19
)
model.load_state_dict(
    {key[16:]: val for key, val in torch.load(model_name).items() if len(key[16:]) > 0}
)


# -- Phase évaluation --
model.eval()
epoch_loss, epoch_acc, epoch_mae = 0.0, 0.0, 0.0

couleurs_rvb_19_classes = torch.tensor(
    [
        [219, 14, 154],
        [147, 142, 123],
        [248, 12, 0],
        [169, 113, 1],
        [21, 83, 174],
        [25, 74, 38],
        [70, 228, 131],
        [243, 166, 13],
        [102, 0, 130],
        [85, 255, 0],
        [255, 243, 13],
        [228, 223, 124],
        [61, 230, 235],
        [255, 255, 255],
        [138, 179, 160],
        [107, 113, 79],
        [197, 220, 66],
        [153, 153, 255],
        [0, 0, 0],
    ]
)

with torch.no_grad():
    for batch in tqdm(val_loader):
        images = batch["images"]  # (B, T, 4, H, W)
        frame_id = batch["frame_id"]
        emprise = batch["emprise"]  # (B, H, W)
        B, T, C, H, W = (
            images.shape
        )  # sauvegarder le batch size et la séquence temporelle

        outputs = model(images.float().reshape(-1, C, H, W)[:, :3])  # (B*T, 19, H, W)
        preds = torch.argmax(outputs, dim=1)  # (B*T, H, W)
        preds = preds.reshape(B, T, H, W)  # (B, T, H, W)
        preds_emprise = (preds + 1) * emprise.unsqueeze(
            1
        )  # (B, T, H, W) — ne garder que les prédictions dans l'emprise
        pred_frame_id = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            for t in range(T):
                indices_classes, hist = torch.unique(
                    preds_emprise[b, t], return_counts=True
                )
                if 1 in indices_classes:
                    proportion = (
                        hist[indices_classes == 1].item()
                        / emprise[b].sum().item()
                        * 100
                    )
                    if (
                        proportion >= SEUIL_BATIMENT
                    ):  # seuil de 50% de pixels prédits comme bâtiment
                        pred_frame_id[b] = t
                        break
        # preds_rvb = couleurs_rvb_19_classes[preds]  # (B, T, H, W, 3)
        # print(f"preds_rvb.shape = {preds_rvb.shape}")
        # plt.imshow(preds_rvb[0, 0])
        # plt.show()
        epoch_acc += compute_accuracy(pred_frame_id, frame_id)
        epoch_mae += compute_mae(pred_frame_id, frame_id)

    print(
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames"
    )
