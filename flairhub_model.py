import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import BuildingTimeSeriesDataset, IDS_PER_SPLIT
from config import BATCH_SIZE, NUM_WORKERS, SEUIL_BATIMENT
from utils import visualize_time_series, compute_accuracy, compute_mae
import csv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="flairhub.log", encoding="utf-8", level=logging.INFO)


class DecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module, segmentation_head: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.segmentation_head = segmentation_head

    def forward(self, *features: Any) -> torch.Tensor:
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


class FlairHubModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        encoder_input_dim: int,
        decoder_input_dim: int,
        num_classes: int,
        encoder_arch: str,
        img_size: int = 512,
    ) -> None:
        super().__init__()

        encoder_name, decoder_name = encoder_arch.split("-", 1)

        # Build encoder
        encoder_model = self._build_smp_model(
            decoder_name, encoder_name, num_classes, encoder_input_dim, img_size
        )
        self.encoder = encoder_model.encoder

        # Build decoder with its own input_dim
        decoder_model = self._build_smp_model(
            decoder_name, encoder_name, num_classes, decoder_input_dim, img_size
        )
        self.decoder = DecoderWrapper(
            decoder_model.decoder, decoder_model.segmentation_head
        )
        self.ckpt_path = ckpt_path
        self._load_weights(ckpt_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_smp_model(
        arch: str,
        encoder_name: str,
        classes: int,
        in_channels: int,
        img_size: int,
    ) -> nn.Module:
        """Try several fallback strategies when creating the SMP model."""
        for enc, use_img_size in [
            (encoder_name, True),
            ("tu-" + encoder_name, True),
            ("tu-" + encoder_name, False),
        ]:
            try:
                kwargs = dict(
                    arch=arch,
                    encoder_name=enc,
                    classes=classes,
                    in_channels=in_channels,
                )
                if use_img_size:
                    kwargs["img_size"] = img_size
                return smp.create_model(**kwargs)
            except (KeyError, TypeError):
                continue
        raise ValueError(
            f"Could not instantiate SMP model for encoder='{encoder_name}', arch='{arch}'"
        )

    def _load_weights(self, ckpt_path: str) -> None:
        """Load and remap encoder + decoder weights from the checkpoint."""
        state_dict = load_file(ckpt_path)

        encoder_sd = {
            k.replace("model.encoders.AERIAL_RGBI.seg_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.encoders.AERIAL_RGBI.seg_model.")
        }
        decoder_sd = {
            k.replace("model.main_decoders.AERIAL_LABEL-COSIA.seg_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.main_decoders.AERIAL_LABEL-COSIA.seg_model.")
        }

        self.encoder.load_state_dict(encoder_sd, strict=True)
        self.decoder.load_state_dict(decoder_sd, strict=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if "RGB" in self.ckpt_path:
            features = self.encoder(imgs[:, [0, 1, 2]])
        elif "IR" in self.ckpt_path:
            features = self.encoder(imgs[:, [3, 0, 1]])
        out = self.decoder(*features)[
            :, [0, 4, 3, 5, 6, 13, 12, 14, 11, 8, 9, 10, 2, 7, 1]
        ]
        return out


# How to use:

ROOT_PATH = "/home/LBaron/git-clones/gitlab.ign.fr/exploration-bdtopo/batiment/output_vrai"  # À adapter selon l'environnement

val_dataset = BuildingTimeSeriesDataset(root_path=ROOT_PATH, split="val", norm=True)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

model_irc = FlairHubModel(
    ckpt_path="FLAIR-HUB_LC-A_IR_swinbase-upernet.safetensors",
    encoder_input_dim=3,
    decoder_input_dim=1,
    num_classes=19,
    encoder_arch="swin_base_patch4_window12_384-upernet",
    img_size=512,
)
model_irc.eval()


model_rgb = FlairHubModel(
    ckpt_path="FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors",
    encoder_input_dim=3,
    decoder_input_dim=1,
    num_classes=19,
    encoder_arch="swin_base_patch4_window12_384-upernet",
    img_size=512,
)
model_rgb.eval()
epoch_loss, epoch_acc, epoch_mae, acc1, acc2 = 0.0, 0.0, 0.0, 0.0, 0.0

tp_total = fp_total = fn_total = tn_total = 0

dict_date_apparition = [{}]
entetes = []

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

        outputs_rgb = model_rgb(images.float().reshape(-1, C, H, W))  # (B*T, 19, H, W)
        outputs_irc = model_irc(images.float().reshape(-1, C, H, W))  # (B*T, 19, H, W)
        n_channels = n_channels.reshape(-1, 4)
        outputs = torch.where(
            (n_channels.sum(-1) == 4)[..., None, None, None],
            (outputs_irc + outputs_rgb) / 2,
            outputs_rgb,
        )

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
        premiere_detection = au_dessus_seuil.float().argmax(
            dim=1
        )  # B -> indice du premier frame où la proportion dépasse le seuil

        pred_frame_id = torch.where(
            detection_trouvee, premiere_detection, torch.zeros(B, dtype=torch.long)
        )

        dict_date_apparition[0][building_id[0]] = years[0][pred_frame_id[0]].item()

        epoch_acc += compute_accuracy(pred_frame_id, frame_id)
        epoch_mae += compute_mae(pred_frame_id, frame_id)
        acc1 += (pred_frame_id + 1 == frame_id).sum().item() + (
            pred_frame_id - 1 == frame_id
        ).sum().item()  # on regarde si la prédiction est à 1 frame près de la vérité pour calculer une acc@1
        acc2 += (pred_frame_id + 2 == frame_id).sum().item() + (
            pred_frame_id - 2 == frame_id
        ).sum().item()  # on regarde si la prédiction est à 2 frames près de la vérité pour calculer une acc@2

        # on sauvegarde les résultats dans un csv pour pouvoir les analyser plus facilement
        entetes.append(building_id[0])

        # masque d'emprise broadcasté (B, T, H, W)
        mask_emprise = emprise.unsqueeze(1) > 0

        # vérité terrain binaire par frame (B, T)
        # years: (B, T), frame_id: (B,)
        apparition_year = years.gather(1, frame_id.unsqueeze(1))  # (B, 1)
        gt_presence = (
            (years >= apparition_year.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
        )  # (B, T, 1, 1)
        gt_mask = gt_presence & mask_emprise  # (B, T, H, W) bool

        # masque prédit pour la classe bâtiment (1) — restreint à l'emprise
        pred_mask = (preds == 1) & mask_emprise  # (B, T, H, W) bool

        # compter
        tp = (pred_mask & gt_mask).sum().item()
        fp = (pred_mask & (~gt_mask)).sum().item()
        fn = ((~pred_mask) & gt_mask).sum().item()
        tn = ((~pred_mask) & (~gt_mask)).sum().item()

        # accumuler pour tout le dataset
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

    confusion = torch.tensor([[tp_total, fp_total], [fn_total, tn_total]])

    print(
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames"
        f"| Acc@1: {acc1 / len(val_loader):.1f}% "
        f"| Acc@2: {acc2 / len(val_loader):.1f}%"
    )

    logger.info(
        f"Evaluation terminée "
        f"| Modèle: FLAIRHUB "
        f"| Dataset: {IDS_PER_SPLIT['val']} "
        f"|Seuil bâtiment: {SEUIL_BATIMENT}% "
        f"| Acc: {epoch_acc / len(val_loader):.1f}% "
        f"| MAE: {epoch_mae / len(val_loader):.2f} frames "
        f"| Acc@1: {acc1 / len(val_loader):.1f}% "
        f"| Acc@2: {acc2 / len(val_loader):.1f}%"
        f"|Confusion matrix (TP FP / FN TN): {confusion}"
    )

    with open("resultats_FlairHub.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=entetes)
        writer.writeheader()
        writer.writerows(dict_date_apparition)
