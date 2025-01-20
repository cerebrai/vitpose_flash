import torch
import torch.nn as nn

from vitpose_flash.onepose.models.vitpose.backbone import ViT
from vitpose_flash.onepose.models.vitpose.head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self,
                cfg: dict, flash_attention: bool = False,
                dtype: torch.dtype = torch.float32,
            ) -> None:

        super(ViTPose, self).__init__()
        self.dtype = dtype
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}

        # -- setting flags for flash attention
        if flash_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.backbone = ViT(**backbone_cfg, flash_attention=flash_attention).to(self.dtype)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg).to(self.dtype)


    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x.to(self.dtype))
        return self.keypoint_head(features)

