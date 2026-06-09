import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from utils.paths import ckpt_path_flairinc_rgb, ckpt_path_flairinc_rgbi


def fuse_features(c2, c3, c4, c5):
    target = c2.shape[-2:]  # 128x128

    # Upsample coarser maps to 128x128 (bilinear, no learned weights)
    c3_up = torch.nn.functional.interpolate(c3, size=target, mode='bilinear', align_corners=False)
    c4_up = torch.nn.functional.interpolate(c4, size=target, mode='bilinear', align_corners=False)
    c5_up = torch.nn.functional.interpolate(c5, size=target, mode='bilinear', align_corners=False)

    # Option A — concat: [B, 960, 128, 128]  (64+128+256+512)
    fused = torch.cat([c2, c3_up, c4_up, c5_up], dim=1)
    return fused 


class FlairIncModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        input_dim: int,
        num_classes: int,
        encoder_only: bool = True,
        use_float16: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_only = encoder_only
        self.model = smp.Unet(encoder_name="resnet34",
                               encoder_weights="imagenet",
                               in_channels=input_dim,
                               classes=num_classes)
        self.ckpt_path = ckpt_path
        self.use_float16 = use_float16
        self.model.load_state_dict(
            {
                key[16:]: val
                for key, val in torch.load(ckpt_path).items()
                if len(key[16:]) > 0
            }
        )
        
        # Convert to float16 if requested (reduces memory, speeds up inference)
        if self.use_float16:
            self.model.half()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # Ensure input matches model dtype
        if self.use_float16:
            imgs = imgs.half()
        
        if self.encoder_only:
            if "rgb_" in self.ckpt_path:
                features = self.model.encoder(imgs[:, [0, 1, 2]])
            elif "rgbi_" in self.ckpt_path:
                features = self.model.encoder(imgs[:, [0, 1, 2, 3]])            
            _, _, c2, c3, c4, c5 = features
            fused_features = fuse_features(c2, c3, c4, c5)
            return fused_features
        
        else:
            if "rgb_" in self.ckpt_path:
                features = self.model(imgs[:, [0, 1, 2]])
            elif "rgbi_" in self.ckpt_path:
                features = self.model(imgs[:, [0, 1, 2, 3]])
            out = features[:, list(range(0, 14)) + [17]]
            return out


class FlairIncWrapper(nn.Module):
    def __init__(
        self,
        rgb_only: bool,
        encoder_only: bool=True,
        fuse_mode: str = "concat",
        use_float16: bool = False,
    ) -> None:
        super().__init__()
        self.flairinc_model_rgb = FlairIncModel(
            ckpt_path=ckpt_path_flairinc_rgb,
            input_dim=3,
            num_classes=19,
            use_float16=use_float16,
            encoder_only=encoder_only,
        )
        if not rgb_only:
            self.flairinc_model_rgbi = FlairIncModel(
                ckpt_path=ckpt_path_flairinc_rgbi,
                input_dim=4,
                num_classes=19,
                use_float16=use_float16,
                encoder_only=encoder_only
            )
        self.rgb_only = rgb_only
        self.fuse_mode = fuse_mode

    def forward(self, x: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
        if self.rgb_only:
            features_rgb = self.flairinc_model_rgb(x)
            return features_rgb
        else:
            features_rgb = self.flairinc_model_rgb(x)
            features_rgbi = self.flairinc_model_rgbi(x)
            if self.fuse_mode == "concat":
                output = torch.cat([features_rgb, features_rgbi], dim=1)
            elif self.fuse_mode == "average":
                output = torch.where(
                    (n_channels.sum(-1) == 4)[..., None, None, None],
                    (features_rgbi + features_rgb) / 2,
                    features_rgb,
                )
            else:
                raise ValueError(f"Invalid fuse_mode='{self.fuse_mode}'")
            return output
