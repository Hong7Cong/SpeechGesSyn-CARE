# dual_stream_video_flow.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForImageClassification, TimesformerModel, WhisperProcessor, WhisperModel
from PIL import Image
from einops import rearrange
from mamba_ssm import Mamba
# from typing import Any, Dict, Tuple, Union

# def move_to_device(batch_item: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str):
#     """Move tensors or dicts of tensors to device."""
#     if isinstance(batch_item, torch.Tensor):
#         return batch_item.to(device, non_blocking=True)
#     if isinstance(batch_item, dict):
#         return {k: v.to(device, non_blocking=True) for k, v in batch_item.items()}
#     return batch_item  # leave as-is for types we don't recognize

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

        #Freeze some weights
        self.text_enc.lm_head = nn.Identity()
        self.text_enc.requires_grad_(False)
        self.text_enc.eval()
        
        for param in self.video_enc.parameters():
            param.requires_grad = False
        for i in range(10, 12):  # base model has 12 encoder blocks         # Unfreeze the last 4 blocks
            for param in self.video_enc.encoder.layer[i].parameters():
                param.requires_grad = True    
        for i in range(12):         # Unfreeze temporal attention across *all* blocks
            for name, param in self.video_enc.encoder.layer[i].named_parameters():
                if "temporal" in name:
                    param.requires_grad = True

        # assert False, self.video_enc
        self.vid_proj = self.proj_head(768, 512)
        self.txt_proj = self.proj_head(768, 512)

    def proj_head(self, d_in, d_out):
        return nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )

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

        if x.dim() == 5 and x.shape[1] in (3, 1):  # likely (B, C, T, H, W)
            x = rearrange(x, 'b c t h w -> b t c h w')
        f = enc(x).last_hidden_state[:,:1]                                       # (B*T, D)
        return f
    
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    
    def encode_text(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        with torch.no_grad():
            f = enc(x).logits
            # out = enc(**x)
            # pooled = self._mean_pool(out.last_hidden_state, x["attention_mask"])   # [B, 768]
            # f = pooled.unsqueeze(1)

        return f
    def forward(
        self,
        video: torch.Tensor,        # (B, C_v, T, H, W)
        text: list,

    ) -> torch.Tensor:

        v_feats = self.encode_video(video, self.video_enc)   # (B, T, D)
        t_feats = self.encode_text(text, self.text_enc)    # (B, T, D)

        v_feats = self.vid_proj(v_feats)
        t_feats = self.txt_proj(t_feats)

        return v_feats, t_feats

class FlowCNN(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()

        self.encoder = nn.Sequential(
            # [2, H, W] → [64, H/2, W/2]
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # [64, H/2, W/2] → [128, H/4, W/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [128, H/4, W/4] → [256, H/8, W/8]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # → [512, 1, 1]
        self.proj = nn.Linear(256, out_dim)

    def forward(self, flows):
        """
        flows: [B, 2, T, H, W]
        returns: [B, T, out_dim]
        """
        B, _, T, H, W = flows.shape

        # merge batch and time
        x = flows.permute(0, 2, 1, 3, 4).reshape(B * T, 2, H, W)

        x = self.encoder(x)
        x = self.pool(x).flatten(1)  # [B*T, 512]
        x = self.proj(x)             # [B*T, 768]

        # restore [B, T, D]
        x = x.view(B, T, -1)
        return x


class SpeechGestureModel(nn.Module):
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
        # self.video_enc = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", use_safetensors=True)
        self.text_enc = AutoModelForMaskedLM.from_pretrained("roberta-base", use_safetensors=True)
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small", use_safetensors=True)
        self.audio_enc = WhisperModel.from_pretrained(
            "openai/whisper-small",
            output_hidden_states=True
        )
        #Freeze some weights
        self.text_enc.lm_head = nn.Identity()
        self.text_enc.requires_grad_(False)
        self.text_enc.eval()
        self.audio_enc.eval()

        # for param in self.video_enc.parameters():
        #     param.requires_grad = False
        # for i in range(10, 12):  # base model has 12 encoder blocks         # Unfreeze the last 4 blocks
        #     for param in self.video_enc.encoder.layer[i].parameters():
        #         param.requires_grad = True    
        # for i in range(12):         # Unfreeze temporal attention across *all* blocks
        #     for name, param in self.video_enc.encoder.layer[i].named_parameters():
        #         if "temporal" in name:
        #             param.requires_grad = True

        # assert False, self.video_enc
        self.vid_proj = self.proj_head(768, 512)
        self.txt_proj = self.proj_head(768, 512)

        # Sequential projection head for fusion
        self.audio_seq_proj = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=768, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.flow_enc = FlowCNN(out_dim=768)
        self.flows_seq_proj = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=768, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def proj_head(self, d_in, d_out):
        return nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )

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

        if x.dim() == 5 and x.shape[1] in (3, 1):  # likely (B, C, T, H, W)
            x = rearrange(x, 'b c t h w -> b t c h w')
        f = enc(x).last_hidden_state[:,:1]                                       # (B*T, D)
        return f
    
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    
    def encode_text(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        with torch.no_grad():
            f = enc(x).logits
            # out = enc(**x)
            # pooled = self._mean_pool(out.last_hidden_state, x["attention_mask"])   # [B, 768]
            # f = pooled.unsqueeze(1)

        return f
    
    import torch

    def compute_motion_energy_full_frame_batch(
        predicted_flows,
        normalize_mode="mean"
    ):
        """
        predicted_flows: torch.Tensor [B, 2, T, H, W]
        returns: motion_energy [B, T, 1]
        """

        # flow magnitude per pixel
        # → [B, T, H, W]
        flow_mag = torch.linalg.norm(predicted_flows, dim=1)

        if normalize_mode == "sum":
            # total motion energy per frame
            motion_energy = flow_mag.sum(dim=(2, 3))  # [B, T]

        elif normalize_mode == "mean":
            # area-normalized (resolution-invariant)
            motion_energy = flow_mag.mean(dim=(2, 3))  # [B, T]

        elif normalize_mode == "sqrt_area":
            area = flow_mag.shape[2] * flow_mag.shape[3]
            motion_energy = flow_mag.sum(dim=(2, 3)) / (area ** 0.5)

        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        # add feature dimension
        return motion_energy.unsqueeze(-1)  # [B, T, 1]

    
    def encode_speech(self, inputs: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        # inputs = self.audio_processor(
        #     audio.numpy(),
        #     sampling_rate=16000,
        #     return_tensors="pt"
        # )

        with torch.no_grad():
            enc_out = self.audio_enc.encoder(
                inputs,
                output_hidden_states=True
            )

        hidden_states = enc_out.hidden_states
        # Prosody layers
        whisper_prosody = torch.stack(hidden_states[1:4]).mean(0)
        whisper_semantic = hidden_states[-1]

        # Crop padding
        # T = int(audio.shape[0] / 16000 * 50)
        # whisper_prosody = whisper_prosody[:, :T]
        # whisper_semantic = whisper_semantic[:, :T]

        # with torch.no_grad():
        #     f = enc(x).logits
            # out = enc(**x)
            # pooled = self._mean_pool(out.last_hidden_state, x["attention_mask"])   # [B, 768]
            # f = pooled.unsqueeze(1)

        return whisper_semantic
    def mean_pool_every_k(self, a_feats, k=5):
        B, T, D = a_feats.shape
        T_new = T // k  # drop remainder

        a_feats = a_feats[:, :T_new * k, :]          # [B, T_new*k, D]
        a_feats = a_feats.view(B, T_new, k, D)       # [B, T_new, k, D]
        return a_feats.mean(dim=2)                   # [B, T_new, D]
    
    # ##--------------------- Forward method for the Prosody + Flow model Alignment ---------------------##
    # def forward(
    #     self,
    #     video,
    #     # video_mask,
    #     flows,
    #     # frames_idx,
    #     # mask,
    #     # length,
    #     transcript,
    #     # text_mask,
    #     # path,
    #     audio,
    #     audio_mask,
    # ) -> torch.Tensor:

    #     # v_feats = self.encode_video(video, self.video_enc)   # (B, T, D)
    #     # t_feats = self.encode_text(transcript, self.text_enc)    # (B, T, D)
    #     a_feats = self.encode_speech(audio, audio_mask)    # (B, T_a, D_a)
    #     f_feats = self.flow_enc(flows)                # (B, T, D_flow)
        
    #     # v_feats = self.vid_proj(v_feats)
    #     # t_feats = self.txt_proj(t_feats)
    #     a_feats = self.mean_pool_every_k(a_feats, k=30)  # (B, T_a//5, D_a)
    #     f_feats = self.mean_pool_every_k(f_feats, k=18)  # (B, T//5, D_flow)
    #     seq_max_len = min(a_feats.shape[1], f_feats.shape[1])
    #     a_feats = a_feats[:, :seq_max_len, :]
    #     f_feats = f_feats[:, :seq_max_len, :]

    #     mask = [round(n / 30) for n in audio_mask] 

    #     # assert False, f"{a_feats.shape}, {f_feats.shape}, {mask}"
    #     a_feats = self.audio_seq_proj(a_feats)  # (B, T_a, D_fusion)
    #     f_feats = self.flows_seq_proj(f_feats)  # (B, T, D_flow)
    #     assert False, f"{a_feats.shape}, {f_feats.shape}, {mask}"
    #     # B, T_a, _ = a_feats.shape

    #     # if isinstance(audio_mask, list):
    #     #     audio_mask = torch.tensor(audio_mask, device=a_feats.device)

    #     # audio_mask = audio_mask.long()
    #     # audio_mask = audio_mask.clamp(0, T_a - 1)

    #     # batch_idx = torch.arange(B, device=a_feats.device)

    #     # a_out = a_feats[batch_idx, audio_mask]
    #     # f_out = f_feats[batch_idx, mask]

    #     # flow_mask = [round(n * 5 / 3) for n in audio_mask]  # crude alignment: 50 audio frames per second, 30 flows frames per second
    #     # assert False, type(mask)
    #     # assert False, f"{a_out.shape}, {f_out.shape}, {mask}"
    #     return a_feats, f_feats




class SpeechGestureModelDev(nn.Module):
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
        # self.video_enc = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", use_safetensors=True)
        # Load DINOv3
        self.visual_enc = torch.hub.load(
            '../dinov3',
            'dinov3_vits16',
            source="local",
            weights="./dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        )
        self.text_enc = AutoModelForMaskedLM.from_pretrained("roberta-base", use_safetensors=True)
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small", use_safetensors=True)
        self.audio_enc = WhisperModel.from_pretrained(
            "openai/whisper-small",
            output_hidden_states=True
        )
        #Freeze some weights
        self.text_enc.lm_head = nn.Identity()
        self.text_enc.requires_grad_(False)
        self.visual_enc.requires_grad_(False)
        self.audio_enc.requires_grad_(False)
        self.text_enc.eval()
        self.visual_enc.eval()
        self.audio_enc.eval()

        self.temporal_embed = nn.Embedding(32, 384)
        # assert False, self.video_enc
        self.vid_proj = self.proj_head(384, 384)
        self.txt_proj = self.proj_head(768, 384)
        self.aud_proj = self.proj_head(768, 384)
        # Sequential projection head for fusion
        self.audio_seq_proj = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=768, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        # self.flow_enc = FlowCNN(out_dim=768)
        # self.flows_seq_proj = Mamba(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     d_model=768, # Model dimension d_model
        #     d_state=16,  # SSM state expansion factor
        #     d_conv=4,    # Local convolution width
        #     expand=2,    # Block expansion factor
        # )
        self.video_seq_proj = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=384, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def add_temporal_superposition(self, video_patches):
        # video_patches: (B, T, P, D)
        B, T, P, D = video_patches.shape
        device = video_patches.device

        frame_ids = torch.arange(T, device=device)
        temp_embed = self.temporal_embed(frame_ids)  # (T, D)

        # superposition happens here
        video_patches = video_patches + temp_embed.unsqueeze(0).unsqueeze(2)

        return video_patches

    @torch.no_grad()
    def enc_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) in range [0,1] float32

        Returns:
            embeddings: (B, D)
        """
        b, t = x.shape[0], x.shape[1]
        if x.dim() == 5 and x.shape[2] in (3, 1):  # likely (B, T, C, H, W)
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        # assert False, x.shape
        # x = x.to(self.device)

        # Normalize
        # x = self.normalize(x)

        # Forward
        feats = self.visual_enc.forward_features(x)

        # CLS token → (B, D)
        cls_embeddings = feats["x_norm_patchtokens"]
        # assert False, cls_embeddings.shape
        cls_embeddings = rearrange(cls_embeddings, '(b t) p d -> b t p d', b=b, t=t)
        # assert False, feats.keys()
        return cls_embeddings
    
    def proj_head(self, d_in, d_out):
        return nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )

    def add_magnitude(self, flow):
        # Compute sqrt(dx^2 + dy^2) along channel dim=1
        magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)  # [B, H, W]
        
        # Add channel dimension back
        magnitude = magnitude.unsqueeze(1)  # [B, 1, H, W]
        
        # Concatenate along channel dim -> [B, 3, H, W]
        out = torch.cat([flow, magnitude], dim=1)
        return out
    
    
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()          # [B, L, 1]
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
    
    def encode_text(self, x: torch.Tensor, enc: nn.Module) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        with torch.no_grad():
            f = enc(x).logits
            # out = enc(**x)
            # pooled = self._mean_pool(out.last_hidden_state, x["attention_mask"])   # [B, 768]
            # f = pooled.unsqueeze(1)

        return f
    
    import torch

    def compute_motion_energy_full_frame_batch(
        predicted_flows,
        normalize_mode="mean"
    ):
        """
        predicted_flows: torch.Tensor [B, 2, T, H, W]
        returns: motion_energy [B, T, 1]
        """

        # flow magnitude per pixel
        # → [B, T, H, W]
        flow_mag = torch.linalg.norm(predicted_flows, dim=1)

        if normalize_mode == "sum":
            # total motion energy per frame
            motion_energy = flow_mag.sum(dim=(2, 3))  # [B, T]

        elif normalize_mode == "mean":
            # area-normalized (resolution-invariant)
            motion_energy = flow_mag.mean(dim=(2, 3))  # [B, T]

        elif normalize_mode == "sqrt_area":
            area = flow_mag.shape[2] * flow_mag.shape[3]
            motion_energy = flow_mag.sum(dim=(2, 3)) / (area ** 0.5)

        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        # add feature dimension
        return motion_energy.unsqueeze(-1)  # [B, T, 1]

    
    def encode_speech(self, inputs: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        """
        x: list of XLMRobert tokens instances (having mask and input_id keys)
        """
        # inputs = self.audio_processor(
        #     audio.numpy(),
        #     sampling_rate=16000,
        #     return_tensors="pt"
        # )

        with torch.no_grad():
            enc_out = self.audio_enc.encoder(
                inputs,
                output_hidden_states=True
            )

        hidden_states = enc_out.hidden_states
        # Prosody layers
        whisper_prosody = torch.stack(hidden_states[1:4]).mean(0)
        whisper_semantic = hidden_states[-1]

        return whisper_semantic
    def mean_pool_every_k(self, a_feats, k=5):
        B, T, D = a_feats.shape
        T_new = T // k  # drop remainder

        a_feats = a_feats[:, :T_new * k, :]          # [B, T_new*k, D]
        a_feats = a_feats.view(B, T_new, k, D)       # [B, T_new, k, D]
        return a_feats.mean(dim=2)                   # [B, T_new, D]
    
    def masked_temporal_pool(self, x, lengths, target_len=50):
        """
        x: (B, T, D)
        lengths: list or tensor of valid lengths per sample
        target_len: output temporal length

        Returns:
            (B, target_len, D)
        """
        B, T, D = x.shape
        device = x.device

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device)

        pooled_outputs = []

        for i in range(B):
            valid_len = lengths[i].item()

            if valid_len == 0:
                # Edge case: no valid frames
                pooled = torch.zeros(target_len, D, device=device)
            else:
                valid_seq = x[i, :valid_len]  # (valid_len, D)

                # reshape to (1, D, L) for adaptive pooling
                valid_seq = valid_seq.transpose(0, 1).unsqueeze(0)

                pooled = F.adaptive_avg_pool1d(valid_seq, target_len)

                # back to (target_len, D)
                pooled = pooled.squeeze(0).transpose(0, 1)

            pooled_outputs.append(pooled)

        return torch.stack(pooled_outputs, dim=0)
    
    def forward(
        self,
        video,
        # video_mask,
        flows,
        # frames_idx,
        # mask,
        # length,
        transcript,
        # text_mask,
        # path,
        audio,
        audio_mask,
    ) -> torch.Tensor:
        
        ## Video encoding
        x = self.enc_image(video)
        x = self.add_temporal_superposition(x)
        x = rearrange(x, 'b t p d -> b (t p) d')
        x = self.video_seq_proj(x)
        x = x[:,::196,:]

        ## Audio encoding
        a_feats = self.encode_speech(audio, audio_mask)    # (B, T_a, D_a)
        a_feats = self.masked_temporal_pool(a_feats, audio_mask, target_len=x.shape[1])  # (B, T_a, D_a)
        a_feats = self.audio_seq_proj(a_feats)  # (B, T_a, D_fusion)
        a_feats = self.aud_proj(a_feats)

        #Text encoding
        t_feats = self.encode_text(transcript, self.text_enc)    # (B, T_t, D_t)
        t_feats = self.txt_proj(t_feats)
        # assert False, f"{x.shape}, {a_feats.shape}, {t_feats.shape}, {transcript}"
        return x, a_feats, t_feats


