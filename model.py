# dual_stream_video_flow.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForImageClassification, TimesformerModel
from PIL import Image
from einops import rearrange

class VideoFlowModel(nn.Module):
    """
    Dual-stream model: per-frame 2D CNN (RGB + Flow) -> masked temporal pooling -> fusion MLP.
    """
    def __init__(
        self,
        in_channels_video: int = 3,
        in_channels_flow: int = 2,
        emb_dim: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        # self.video_enc = FrameEncoder2D(in_channels_video, out_dim=emb_dim)
        # self.flow_enc  = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18") 
        # self.video_enc = AutoModel.from_pretrained('facebook/dinov2-base').cuda()
        self.video_enc = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.text_enc = AutoModelForMaskedLM.from_pretrained("roberta-base")
    
        self.text_enc.lm_head = nn.Identity()
        self.text_enc.requires_grad_(False)
        # self.fuse = nn.Sequential(
        #     nn.Linear(emb_dim * 2, fusion_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(fusion_dim, num_classes),
        # )
    def add_magnitude(self, flow):
        # Compute sqrt(dx^2 + dy^2) along channel dim=1
        magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)  # [B, H, W]
        
        # Add channel dimension back
        magnitude = magnitude.unsqueeze(1)  # [B, 1, H, W]
        
        # Concatenate along channel dim -> [B, 3, H, W]
        out = torch.cat([flow, magnitude], dim=1)
        return out
    
    def encode_video(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        x: (B, C, T, H, W) -> per-frame features (B, T, D)
        """
        B, C, T, H, W = x.shape
        # x = rearrange(x, 'B C T H W -> B T C H W') # (B*T, C, H, W)
        # if is_flow:
        #     x = self.add_magnitude(x)
        # if processor:
        #     x = processor(images=list(x), return_tensors="pt", device=x.device)
        # x = rearrange(x, 'B C T H W -> (B T) H W C')
        f = enc(x).last_hidden_state[:,:1]                                       # (B*T, D)
        # assert False, f.shape
        # f = rearrange(f, '(B T) P D -> B T P D')                               # (B, T, D)
        return f

    def encode_text(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        # B, C, T, H, W = x.shape
        # x = rearrange(x, 'B C T H W -> B T C H W') # (B*T, C, H, W)
        # if is_flow:
        #     x = self.add_magnitude(x)
        # if processor:
        #     x = processor(images=list(x), return_tensors="pt", device=x.device)
        # x = rearrange(x, 'B C T H W -> (B T) H W C')
        f = enc(x).logits
        # f = enc(x)                                        # (B*T, D)
        # assert False, f.shape
        # f = rearrange(f, '(B T) P D -> B T P D')                               # (B, T, D)
        return f
    def forward(
        self,
        video: torch.Tensor,        # (B, C_v, T, H, W)
        text: list,
        # flows: torch.Tensor | None,        # (B, C_f, T, H, W)
        # mask: torch.Tensor | None,  # (B, T) or None
    ) -> torch.Tensor:
        # video = rearrange(video, 'B T H W C-> B C T H W')
          # (B*T, C, H, W)
        v_feats = self.encode_video(video, self.video_enc)   # (B, T, D)
        t_feats = self.encode_text(text, self.text_enc)    # (B, T, D)

        # v_pool = masked_mean(v_feats, mask)                   # (B, D)
        # f_pool = masked_mean(f_feats, mask)                   # (B, D)

        # fused = torch.cat([v_pool, f_pool], dim=-1)           # (B, 2D)
        # logits = self.fuse(fused)                             # (B, num_classes)
        return v_feats, t_feats
