import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Any


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
        encoder_model = self._build_smp_model(decoder_name, encoder_name, num_classes, encoder_input_dim, img_size)
        self.encoder = encoder_model.encoder

        # Build decoder with its own input_dim
        decoder_model = self._build_smp_model(decoder_name, encoder_name, num_classes, decoder_input_dim, img_size)
        self.decoder = DecoderWrapper(decoder_model.decoder, decoder_model.segmentation_head)
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
                kwargs = dict(arch=arch, encoder_name=enc, classes=classes, in_channels=in_channels)
                if use_img_size:
                    kwargs["img_size"] = img_size
                return smp.create_model(**kwargs)
            except (KeyError, TypeError):
                continue
        raise ValueError(f"Could not instantiate SMP model for encoder='{encoder_name}', arch='{arch}'")

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
        if 'RGB' in self.ckpt_path:
            features = self.encoder(imgs[:, [0, 1, 2]])
        elif 'IR' in self.ckpt_path:
            features = self.encoder(imgs[:, [3, 0, 1]])
        out = self.decoder(*features)[:, [0, 4, 3, 5, 6, 13, 12, 14, 11, 8, 9, 10, 2, 7, 1]]
        return out
