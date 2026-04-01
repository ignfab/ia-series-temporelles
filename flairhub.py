import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import BuildingTimeSeriesDataset
from config import BATCH_SIZE, NUM_WORKERS, SEUIL_BATIMENT
from utils import visualize_time_series, compute_accuracy, compute_mae
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Chargement des datasets
# ------------------------------------------------------------------------------

ROOT_PATH = "/home/LBaron/git-clones/gitlab.ign.fr/exploration-bdtopo/batiment/output_vrai"  # À adapter selon l'environnement

val_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="val", norm=True)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

model_name_rgb = "FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth"
model_rgb = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=19
)
model_rgb.load_state_dict(
    {
        key[16:]: val
        for key, val in torch.load(model_name_rgb).items()
        if len(key[16:]) > 0
    }
)

model_name_rgbi = "FLAIR-INC_rgbi_15cl_resnet34-unet_weights.pth"
model_rgbi = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=19
)
model_rgbi.load_state_dict(
    {
        key[16:]: val
        for key, val in torch.load(model_name_rgbi).items()
        if len(key[16:]) > 0
    }
)


# -- Phase évaluation --
model_rgb.eval()
model_rgbi.eval()
epoch_loss, epoch_acc, epoch_mae, acc1, acc2 = 0.0, 0.0, 0.0, 0.0, 0.0

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

dict_date_apparition = {}
with torch.no_grad():
    for batch in tqdm(val_loader):
        images = batch["images"]  # (B, T, 4, H, W)
        frame_id = batch["frame_id"]
        emprise = batch["emprise"]  # (B, H, W)
        n_channels = batch["n_channels"]  # (B, T, 4)
        years = batch["years"]  # (B, T)
        building_id = batch["building_id"]  # (B,)

        B, T, C, H, W = (
            images.shape
        )  # sauvegarder le batch size et la séquence temporelle

        # on utilise les deux modèles pour faire des prédictions à partir des données RGB et RGBI, puis on combine les résultats en fonction du nombre de canaux disponibles
        outputs_rgb = model_rgb(
            images.float().reshape(-1, C, H, W)[:, :3]
        )  # (B*T, 19, H, W)
        outputs_rgbi = model_rgbi(
            images.float().reshape(-1, C, H, W)
        )  # (B*T, 19, H, W)
        n_channels = n_channels.reshape(-1, 4)
        outputs = torch.where(
            (n_channels.sum(-1) == 4)[..., None, None, None],
            (outputs_rgbi + outputs_rgb) / 2,
            outputs_rgb,
        )  # on regarde les canaux disponibles pour chaque image et on fait une moyenne des prédictions des deux modèles si les 4 canaux sont présents, sinon on garde les prédictions du modèle RGB
        preds = torch.argmax(outputs, dim=1)  # (B*T, H, W)
        preds = preds.reshape(B, T, H, W)  # (B, T, H, W)
        preds_emprise = (preds + 1) * emprise.unsqueeze(
            1
        )  # (B, T, H, W) — ne garder que les prédictions dans l'emprise

        masque_classe1 = preds_emprise == 1  # B x T x H x W
        nb_pixels_classe1 = masque_classe1.sum(dim=(-2, -1)).float()  # B x T
        taille_emprise = emprise.sum(dim=(-2, -1)).float().unsqueeze(1)  # B x 1
        proportions = (
            nb_pixels_classe1 / taille_emprise * 100
        )  # Proportion (%) par (b, t), shape : B x T
        au_dessus_seuil = (
            proportions >= SEUIL_BATIMENT
        )  # B x T, booléen indiquant pour chaque frame si la proportion dépasse le seuil défini pour considérer qu'il y a un bâtiment détecté

        detection_trouvee = au_dessus_seuil.any(dim=1)  # B
        # premiere_detection = au_dessus_seuil.float().argmax(dim=1)  # B -> indice du premier frame où la proportion dépasse le seuil
        starts = au_dessus_seuil & (
            ~torch.cat(
                [
                    torch.zeros(B, 1, device=au_dessus_seuil.device, dtype=torch.bool),
                    au_dessus_seuil[:, :-1],
                ],
                dim=1,
            )
        )  # B x T, True uniquement pour les frames où la proportion dépasse le seuil et où le frame précédent ne le dépassait pas (détection de "starts" d'apparition)
        last_start_idx = (T - 1) - starts.flip(dims=[1]).float().argmax(
            dim=1
        )  # B, indice du dernier "start" d'apparition (correspondant au premier frame d'apparition en partant du début de la séquence)
        premiere_detection = torch.where(
            detection_trouvee,
            last_start_idx.long(),
            torch.zeros(B, dtype=torch.long, device=au_dessus_seuil.device),
        )  # B -> indice du premier frame où la proportion dépasse le seuil, en gérant le cas où il n'y a pas de détection entre plusieurs frames (destruction/reconstruction successives)
        # premiere_detection != last_start_idx s'il n'y a pas de détection

        pred_frame_id = torch.where(
            detection_trouvee, premiere_detection, torch.zeros(B, dtype=torch.long)
        )

        dict_date_apparition[building_id[0]] = years[0][pred_frame_id[0]].item()

        # preds_rvb = couleurs_rvb_19_classes[preds]  # (B, T, H, W, 3)
        # print(f"preds_rvb.shape = {preds_rvb.shape}")
        # plt.imshow(preds_rvb[0, 0])
        # plt.show()
        epoch_acc += compute_accuracy(pred_frame_id, frame_id)
        epoch_mae += compute_mae(pred_frame_id, frame_id)
        acc1 += (pred_frame_id + 1 == frame_id).sum().item() + (
            pred_frame_id - 1 == frame_id
        ).sum().item()
        acc2 += (pred_frame_id + 2 == frame_id).sum().item() + (
            pred_frame_id - 2 == frame_id
        ).sum().item()

    print(
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames "
        f"| Acc@1: {acc1 / len(val_loader):.1f}% "
        f"| Acc@2: {acc2 / len(val_loader):.1f}%"
    )
