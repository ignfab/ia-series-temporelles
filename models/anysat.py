
import torch
import torch.nn as nn

class AnySatWrapper(nn.Module):
    def __init__(self, use_float16: bool = False) -> None:
        super().__init__()
        self.model = torch.hub.load("gastruc/anysat", "anysat", pretrained=True, flash_attn=False)
        if use_float16:
            self.model.half()

    def forward(self, x):
        # add a 5th zero channel to match the expected input shape of (B, 5, H, W)
        x = torch.cat([x, torch.zeros_like(x[:, :1, :, :])], dim=1)  # (B, 5, H, W)
        output = self.model({"aerial": x}, patch_size=20, output="dense", output_modality="aerial-flair")
        return output