# varlen_dataset_with_padtrim.py
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from typing import Any, Dict
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video, VideoReader
import cv2
from utils.utils import cluster_flow_time_series, flow_to_features, flow_to_image, create_image_grid, extract_frames, normalize_flow_magnitude, MaybeToTensor, AddSumChannel, Normalize, weighted_dot, expand_by_edges
from typing import List, Dict, Any
from transformers import AutoImageProcessor, AutoTokenizer, WhisperProcessor
from einops import rearrange
import ffmpeg
import io
import soundfile as sf
from decord import VideoReader, cpu

def load_audio_from_mp4(mp4_path, sr=16000):
    out, _ = (
        ffmpeg
        .input(mp4_path)
        .output("pipe:", format="wav", ac=1, ar=sr)
        .run(capture_stdout=True, capture_stderr=True)
    )

    # import soundfile as sf
    audio, _ = sf.read(io.BytesIO(out))
    return torch.tensor(audio, dtype=torch.float32)

class SEAMLESSData(Dataset):
    """
    CSV columns required: path_to_clip, transcript

    Returns per item:
      {
        "video": FloatTensor (C, max_len, H, W) in [0,1]
        "mask":  BoolTensor  (max_len,) -> True for valid frames, False for padding
        "length": LongTensor ()         -> original length clipped to max_len
        "transcript": str,
        "path": str
      }
    """
    def __init__(
        self,
        csv_path: str,
        resize: Optional[int] = 224,   # fixed H=W=resize; set None to keep original
        max_len: int = 8,            # pad/trim target along time
        transform: Optional[torch.nn.Module] = None,
        n_clusters: int = 8,
    ):
        self.resize = resize
        self.max_len = int(max_len)
        self.n_clusters = n_clusters
        df = pd.read_csv(csv_path)[:4] #max 3600 samples for debug
        # assert False, len(df)
        need = {"path_to_clip", "transcript"}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV must contain {need}, got {list(df.columns)}")

        self.video_paths: List[str] = df["path_to_clip"].astype(str).tolist()
        self.flows_paths = [str(Path(p).with_suffix('.npy')) for p in df["path_to_clip"].astype(str)]
        self.texts: List[str] = df["transcript"].fillna("").astype(str).tolist()
        if len(self.video_paths) == 0:
            raise ValueError(f"No rows found in {csv_path}")
        self.processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base', use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def get_flow_clusters(self, flow, n_clusters = 10):
        n_clusters = n_clusters
        labels, clusters, features = cluster_flow_time_series(
            flow, n_clusters,
            num_dir_bins=16, grid=8,
            smooth_window=5,      # try 0 for no smoothing, or an odd number like 3/5/7
            random_state=42
        )
        clusters = sorted(clusters, key=lambda a: (len(a) == 0, a[0] if a else float('inf')))
        frames_idx = [c[int(np.floor(len(c)/2))] if len(c) > 0 else None for c in clusters]
        frames_idx = [x for x in frames_idx if x is not None]
        corresponding_flows = [flow[i] for i in frames_idx]
        corresponding_flows = np.stack(corresponding_flows)
        return clusters, frames_idx, corresponding_flows
    
    def _read_audio_tensor(self, path: str) -> torch.Tensor:
        """Read audio from mp4 -> (T,) float32 in [-1,1]."""
        audio = load_audio_from_mp4(path)
        audio_mask = int(audio.shape[0] / 16000 * 50)
        inputs = self.audio_processor(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        return inputs["input_features"].squeeze(0), audio_mask  # (T,)

    # def _read_video_tensor(self, path: str) -> torch.Tensor:
    #     """Read video -> (C, T, H, W) float32 in [0,1]."""
    #     frames, _, _ = read_video(path, pts_unit="sec")  # (T,H,W,C) uint8
    #     if frames.numel() == 0:
    #         # fallback black frame if unreadable
    #         frames = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    #     vid = frames.permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)

    #     if self.resize is not None:
    #         # Resize spatial dims; treat time as batch for interpolate
    #         C, T, H, W = vid.shape
    #         vid = F.interpolate(
    #             vid.permute(1, 0, 2, 3),  # (T,C,H,W)
    #             size=(self.resize, self.resize),
    #             mode="bilinear",
    #             align_corners=False,
    #         ).permute(1, 0, 2, 3)        # (C,T,H,W)
    #     return vid
    def _read_video_tensor(self, path: str) -> torch.Tensor:
        """
        Read video -> (C, T, H, W) float32 in [0,1].
        Tries multiple backends before failing.
        """

        frames = None

        # -------------------------------------------------
        # 1️⃣ Try torchvision.read_video (FFmpeg backend)
        # -------------------------------------------------
        try:
            frames, _, _ = read_video(path, pts_unit="sec")
            if frames is not None and frames.numel() > 0:
                pass
            else:
                frames = None
        except Exception:
            frames = None

        # -------------------------------------------------
        # 2️⃣ Try torchvision VideoReader (streaming API)
        # -------------------------------------------------
        if frames is None:
            try:
                reader = VideoReader(path, "video")
                frame_list = [f["data"] for f in reader]
                if len(frame_list) > 0:
                    frames = torch.stack(frame_list)  # (T,H,W,C)
            except Exception:
                frames = None

        # -------------------------------------------------
        # 3️⃣ Try OpenCV fallback
        # -------------------------------------------------
        if frames is None:
            try:
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    frame_list = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_list.append(torch.from_numpy(frame))
                    cap.release()

                    if len(frame_list) > 0:
                        frames = torch.stack(frame_list)
            except Exception:
                frames = None

        # -------------------------------------------------
        # ❌ If all methods fail → raise error
        # -------------------------------------------------
        if frames is None or frames.numel() == 0:
            raise RuntimeError(f"Failed to load video: {path}")

        # -------------------------------------------------
        # Convert to (C, T, H, W)
        # -------------------------------------------------
        vid = frames.permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)

        # -------------------------------------------------
        # Optional Resize
        # -------------------------------------------------
        if self.resize is not None:
            C, T, H, W = vid.shape
            vid = F.interpolate(
                vid.permute(1, 0, 2, 3),  # (T,C,H,W)
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)

        # Final safety check
        if torch.isnan(vid).any():
            raise RuntimeError(f"NaNs detected in video: {path}")

        return vid
    
    # -------- the pad/trim helper (as requested) --------
    def pad_or_trim(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        video: (C, T, H, W) -> returns dict with
          - video: (C, max_len, H, W)
          - mask:  (max_len,) bool
        """
        C, T, H, W = video.shape
        max_len = self.max_len
        out = video.new_zeros((C, max_len, H, W))
        mask = torch.zeros((max_len,), dtype=torch.bool, device=video.device)

        use = min(T, max_len)
        if use > 0:
            out[:, :use] = video[:, :use]  # trim if longer, copy if shorter
            mask[:use] = True

        length = torch.as_tensor(use, dtype=torch.long, device=video.device)
        return {"out": out, "mask": mask}
    
    def pad_or_trim_dim(self, x: torch.Tensor, pad_dim: int, max_len: int, pad_value=0) -> Dict[str, torch.Tensor]:
        """
        Pads (with pad_value) or trims a tensor along a chosen dimension to max_len.

        Args:
            x:        input tensor of arbitrary shape
            pad_dim:  dimension index to pad/trim (can be negative)
            max_len:  desired length along pad_dim (must be > 0)
            pad_value: value used for padding (default 0)

        Returns:
            {
            "out":  tensor with size(x)[pad_dim] == max_len,
            "mask": (max_len,) bool tensor where True marks original (un-padded) positions
            }
        """
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        ndim = x.dim()
        if pad_dim < 0:
            pad_dim += ndim
        if not (0 <= pad_dim < ndim):
            raise IndexError(f"pad_dim {pad_dim} out of range for tensor with {ndim} dims")

        cur_len = x.size(pad_dim)
        use = min(cur_len, max_len)

        # Build output shape (same as x, except pad_dim -> max_len)
        out_shape = list(x.shape)
        out_shape[pad_dim] = max_len
        # Allocate output filled with pad_value, and the boolean mask
        out = x.new_full(out_shape, pad_value)
        mask = torch.zeros((max_len,), dtype=torch.bool, device=x.device)
        if use > 0:
            # Prepare slice objects to copy the "used" region
            slc_out = [slice(None)] * ndim
            slc_in  = [slice(None)] * ndim
            slc_out[pad_dim] = slice(0, use)
            slc_in[pad_dim]  = slice(0, use)

            out[tuple(slc_out)] = x[tuple(slc_in)]
            mask[:use] = True
        return {"out": out, "mask": mask}
    
    def add_magnitude(self, flow):
       # input: flow [C, B, H, W] where C=2 (dx, dy)
        # Compute sqrt(dx^2 + dy^2) along channel dim=1
        magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)  # [B, H, W]
        # Add channel dimension back
        magnitude = magnitude.unsqueeze(0)  # [1, B, H, W]
        # Concatenate along channel dim -> [3, B, H, W]
        out = torch.cat([flow, magnitude], dim=0)
        return out
    
    def __getitem__(self, idx: int) -> Dict[str, object]:
        # FLOWS
        path = self.video_paths[idx]
        # assert False, np.load(self.flows_paths[idx])
        flow = np.load(self.flows_paths[idx], allow_pickle=True).float().transpose(0, 1)  # (T, H, W, C) -> (C, T, H, W)
        packed = self.pad_or_trim(flow)
        _, frames_idx, _ = self.get_flow_clusters(packed['out'].transpose(0, 1), n_clusters=self.n_clusters)
        flow_3c = self.add_magnitude(packed["out"])  # (2, T, H, W) -> (3, T, H, W)
        flow_3c = rearrange(flow_3c, 'C T H W -> T C H W')  # (3, T, H, W) -> (T, H, W, 3)
        
        audio, audio_mask = self._read_audio_tensor(path)  # (T_audio, D_audio)

        # VIDEO FRAMES
        video_frames = extract_frames(path, frames_idx) # should have length = n_clusters
        if len(video_frames) < self.n_clusters:
            n = self.n_clusters - len(video_frames)
            last = video_frames[-1:].expand(n, *video_frames.shape[1:])
            video_frames = torch.cat([video_frames, last], dim=0)
        
        if self.processor:
            video_frames = self.processor(images=list(video_frames), return_tensors="pt")['pixel_values'][0] #[T, C, H, W]
        padded_video = self.pad_or_trim_dim(video_frames, pad_dim=0, max_len=self.max_len, pad_value=0.0)  # (T, C, H, W) -> (max_len, C, H, W)
            
        # TEXT TOKENS
        text = self.tokenizer(self.texts[idx], return_tensors='pt')['input_ids'][0]  # (T, )
        padded_text = self.pad_or_trim_dim(text, pad_dim=0, max_len=self.max_len, pad_value=0.0)
        return {"video": padded_video['out'],
                "video_mask": padded_video['mask'],
                "flows": flow,
                "frames_idx": frames_idx,
                "mask": packed["mask"],
                "length": 0,
                "transcript": padded_text['out'],
                "text_mask": padded_text['mask'],
                "path": path,
                "audio": audio,
                "audio_mask": audio_mask,
                }

def pad_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Pads the time dimension to the max length in the batch.
    Returns:
      video: (B, C, T_max, H, W)
      lengths: (B,) original T per sample
      mask: (B, T_max) True for valid frames, False for padding
      transcript: list[str]
      path: list[str]
    """
    videos = [b["video"] for b in batch]  # each (C,T,H,W)
    len_np = np.stack([v.shape[1] for v in videos])
    lengths = torch.Tensor(len_np, dtype=torch.long)
    T_max = int(lengths.max())
    B = len(videos)
    C, H, W = videos[0].shape[0], videos[0].shape[2], videos[0].shape[3]

    out = torch.zeros((B, C, T_max, H, W), dtype=videos[0].dtype)
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    for i, v in enumerate(videos):
        t = v.shape[1]
        out[i, :, :t] = v
        mask[i, :t] = True

    transcripts = [b["transcript"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"video": out, "lengths": lengths, "mask": mask,
            "transcript": transcripts, "path": paths}

def pad_time_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expects each sample from SEAMLESSData to contain:
      - "flows":  (C, T, H, W)  (un-padded)
      - "mask":   (L,) bool     (L == max_len used in dataset)
      - "length": ()   long
      - "video":  key-frame frames (tensor or list; we handle both)
      - "frames_idx": list[int]
      - "transcript": str
      - "path": str

    Returns:
      flows:   (B, C, L, H, W)  # time-padded/trimmed to L
      mask:    (B, L) bool
      lengths: (B,)
      video:   stacked tensor (B, ...) if shapes match, else list
      frames_idx, transcript, path: lists
    """
    B = len(batch)

    # Use the dataset's fixed max_len from the first sample's mask length.
    L = 1000#int(batch[0]["mask"].shape[0])
    Lv = int(batch[0]["video_mask"].shape[0])
    Lt = int(batch[0]["text_mask"].shape[0])

    # Infer tensor shapes from the first sample's flows
    f0 = batch[0]["flows"]
    assert isinstance(f0, torch.Tensor) and f0.ndim == 4, "flows must be (C,T,H,W) tensor"
    C, T0, H, W = f0.shape
    dtype = f0.dtype
    device = torch.device("cpu")  # collate runs on CPU; move later in your training step

    flows = torch.zeros((B, C, L, H, W), dtype=dtype, device=device)
    mask  = torch.zeros((B, L), dtype=torch.bool, device=device)
    video_mask  = torch.zeros((B, Lv), dtype=torch.bool, device=device)
    text_mask  = torch.zeros((B, Lt), dtype=torch.bool, device=device)
    lengths = torch.empty((B,), dtype=torch.long, device=device)

    videos_raw = []
    frames_idx = []
    transcripts = []
    paths = []
    audio_masks = []

    # ---------- AUDIO (1D waveform) ----------
    a0 = batch[0]["audio"]
    assert torch.is_tensor(a0) and a0.ndim == 2, "audio must be (T, D)"
    D = a0.shape[1]
    La = max(sample["audio"].shape[0] for sample in batch)

    audio = torch.zeros((B, La, D), dtype=a0.dtype, device=device)
    # audio_mask = torch.zeros((B, La), dtype=torch.bool, device=device)

    for i, sample in enumerate(batch):
        x = sample["flows"]  # (C,T,H,W)
        assert x.shape[0] == C and x.shape[2] == H and x.shape[3] == W, \
            "All flows must have the same C/H/W. Ensure dataset.resize is fixed."

        # Use either provided length/mask or recompute quickly
        use = int(min(x.shape[1], L))
        flows[i, :, :use] = x[:, :use]
        lengths[i] = use

        # Prefer sample["mask"] if present (already L long)
        if "mask" in sample and sample["mask"].numel() == L:
            mask[i] = sample["mask"].to(torch.bool)
        else:
            mask[i, :use] = True

        if "video_mask" in sample and sample["video_mask"].numel() == L:
            video_mask[i] = sample["video_mask"].to(torch.bool)
        else:
            video_mask[i, :use] = True
            
        if "text_mask" in sample and sample["text_mask"].numel() == L:
            text_mask[i] = sample["text_mask"].to(torch.bool)
        else:
            text_mask[i, :use] = True
        
        videos_raw.append(sample["video"])
        frames_idx.append(sample.get("frames_idx", []))
        transcripts.append(sample["transcript"])
        paths.append(sample["path"])

        # ---------- AUDIO ----------
        a = sample["audio"]      # (Ta, D)
        Ta = a.shape[0]

        audio[i, :Ta] = a
        # audio_mask[i, :Ta] = True
        audio_masks.append(sample["audio_mask"])

    # Try to stack videos if they are tensors with consistent shapes
    video_first = videos_raw[0]
    if torch.is_tensor(video_first):
        try:
            video = torch.stack(videos_raw, dim=0)  # (B, ...)
        except Exception:
            video = videos_raw  # fall back to list
    else:
        video = videos_raw  # keep as list (e.g., list of PIL or numpy)

    text_first = transcripts[0]
    if torch.is_tensor(text_first):
        try:
            transcripts = torch.stack(transcripts, dim=0)  # (B, ...)
        except Exception:
            transcripts = transcripts  # fall back to list
    else:
        transcripts = transcripts  # keep as list (e.g., list of PIL or numpy)


    return {
        "flows": flows,          # (B,C,L,H,W)
        "mask": mask,            # (B,L)
        "length": lengths,      # (B,)
        "video": video,          # stacked tensor or list (if shapes differ)
        "video_mask": video_mask,
        "frames_idx": frames_idx,
        "transcript": transcripts,
        "text_mask": text_mask,
        "path": paths,
        # 🔊 audio
        "audio": audio,            # (B, La)
        "audio_mask": audio_masks,  # (B, La)
    }


class SEAMLESSDataDEV(Dataset):
    """
    CSV columns required: path_to_clip, transcript

    Returns per item:
      {
        "video": FloatTensor (C, max_len, H, W) in [0,1]
        "mask":  BoolTensor  (max_len,) -> True for valid frames, False for padding
        "length": LongTensor ()         -> original length clipped to max_len
        "transcript": str,
        "path": str
      }
    """
    def __init__(
        self,
        csv_path: str,
        resize: Optional[int] = 224,   # fixed H=W=resize; set None to keep original
        max_len: int = 32,            # pad/trim target along time
        transform: Optional[torch.nn.Module] = None,
        n_clusters: int = 32,
    ):
        self.resize = resize
        self.max_len = int(max_len)
        self.n_clusters = max_len
        df = pd.read_csv(csv_path)[:64] #max 3600 samples for debug
        # assert False, len(df)
        need = {"path_to_clip", "transcript"}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV must contain {need}, got {list(df.columns)}")

        self.video_paths: List[str] = df["path_to_clip"].astype(str).tolist()
        self.flows_paths = [str(Path(p).with_suffix('.npy')) for p in df["path_to_clip"].astype(str)]
        self.texts: List[str] = df["transcript"].fillna("").astype(str).tolist()
        if len(self.video_paths) == 0:
            raise ValueError(f"No rows found in {csv_path}")
        self.processor = AutoImageProcessor.from_pretrained('MCG-NJU/videomae-base', use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def get_flow_clusters(self, flow, n_clusters = 10):
        n_clusters = n_clusters
        labels, clusters, features = cluster_flow_time_series(
            flow, n_clusters,
            num_dir_bins=16, grid=8,
            smooth_window=5,      # try 0 for no smoothing, or an odd number like 3/5/7
            random_state=42
        )
        clusters = sorted(clusters, key=lambda a: (len(a) == 0, a[0] if a else float('inf')))
        frames_idx = [c[int(np.floor(len(c)/2))] if len(c) > 0 else None for c in clusters]
        frames_idx = [x for x in frames_idx if x is not None]
        corresponding_flows = [flow[i] for i in frames_idx]
        corresponding_flows = np.stack(corresponding_flows)
        return clusters, frames_idx, corresponding_flows
    
    def _read_audio_tensor(self, path: str) -> torch.Tensor:
        """Read audio from mp4 -> (T,) float32 in [-1,1]."""
        audio = load_audio_from_mp4(path)
        audio_mask = int(audio.shape[0] / 16000 * 50)
        inputs = self.audio_processor(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        return inputs["input_features"].squeeze(0), audio_mask  # (T,)

    def _read_video_tensor(self, path: str) -> torch.Tensor:
        """Read video -> (C, T, H, W) float32 in [0,1]."""
        frames, _, _ = read_video(path, pts_unit="sec")  # (T,H,W,C) uint8
        if frames.numel() == 0:
            # fallback black frame if unreadable
            frames = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
        vid = frames.permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)

        if self.resize is not None:
            # Resize spatial dims; treat time as batch for interpolate
            C, T, H, W = vid.shape
            vid = F.interpolate(
                vid.permute(1, 0, 2, 3),  # (T,C,H,W)
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).permute(1, 0, 2, 3)        # (C,T,H,W)
        return vid

    # -------- the pad/trim helper (as requested) --------
    def pad_or_trim(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        video: (C, T, H, W) -> returns dict with
          - video: (C, max_len, H, W)
          - mask:  (max_len,) bool
        """
        C, T, H, W = video.shape
        max_len = self.max_len
        out = video.new_zeros((C, max_len, H, W))
        mask = torch.zeros((max_len,), dtype=torch.bool, device=video.device)

        use = min(T, max_len)
        if use > 0:
            out[:, :use] = video[:, :use]  # trim if longer, copy if shorter
            mask[:use] = True

        length = torch.as_tensor(use, dtype=torch.long, device=video.device)
        return {"out": out, "mask": mask}
    
    def pad_or_trim_dim(self, x: torch.Tensor, pad_dim: int, max_len: int, pad_value=0) -> Dict[str, torch.Tensor]:
        """
        Pads (with pad_value) or trims a tensor along a chosen dimension to max_len.

        Args:
            x:        input tensor of arbitrary shape
            pad_dim:  dimension index to pad/trim (can be negative)
            max_len:  desired length along pad_dim (must be > 0)
            pad_value: value used for padding (default 0)

        Returns:
            {
            "out":  tensor with size(x)[pad_dim] == max_len,
            "mask": (max_len,) bool tensor where True marks original (un-padded) positions
            }
        """
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        ndim = x.dim()
        if pad_dim < 0:
            pad_dim += ndim
        if not (0 <= pad_dim < ndim):
            raise IndexError(f"pad_dim {pad_dim} out of range for tensor with {ndim} dims")

        cur_len = x.size(pad_dim)
        use = min(cur_len, max_len)

        # Build output shape (same as x, except pad_dim -> max_len)
        out_shape = list(x.shape)
        out_shape[pad_dim] = max_len
        # Allocate output filled with pad_value, and the boolean mask
        out = x.new_full(out_shape, pad_value)
        mask = torch.zeros((max_len,), dtype=torch.bool, device=x.device)
        if use > 0:
            # Prepare slice objects to copy the "used" region
            slc_out = [slice(None)] * ndim
            slc_in  = [slice(None)] * ndim
            slc_out[pad_dim] = slice(0, use)
            slc_in[pad_dim]  = slice(0, use)

            out[tuple(slc_out)] = x[tuple(slc_in)]
            mask[:use] = True
        return {"out": out, "mask": mask}
    
    def add_magnitude(self, flow):
       # input: flow [C, B, H, W] where C=2 (dx, dy)
        # Compute sqrt(dx^2 + dy^2) along channel dim=1
        magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)  # [B, H, W]
        # Add channel dimension back
        magnitude = magnitude.unsqueeze(0)  # [1, B, H, W]
        # Concatenate along channel dim -> [3, B, H, W]
        out = torch.cat([flow, magnitude], dim=0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # FLOWS
        path = self.video_paths[idx]
        flow = torch.Tensor(np.load(self.flows_paths[idx], allow_pickle=True)).float().transpose(0, 1)  # (T, H, W, C) -> (C, T, H, W)
        packed = self.pad_or_trim(flow)
        _, frames_idx, _ = self.get_flow_clusters(packed['out'].transpose(0, 1), n_clusters=self.n_clusters) ## THIS CODE MAY WRONG
        flow_3c = self.add_magnitude(packed["out"])  # (2, T, H, W) -> (3, T, H, W)
        flow_3c = rearrange(flow_3c, 'C T H W -> T C H W')  # (3, T, H, W) -> (T, H, W, 3)
        
        audio, audio_mask = self._read_audio_tensor(path)  # (T_audio, D_audio)
        # assert False, f"{self.n_clusters}, {frames_idx}, {packed['out'].shape}, {self.max_len}, {flow.shape}, {audio_mask}"
        # VIDEO FRAMES
        video_frames = extract_frames(path, frames_idx) # should have length = n_clusters
        if len(video_frames) < self.n_clusters:
            n = self.n_clusters - len(video_frames)
            last = video_frames[-1:].expand(n, *video_frames.shape[1:])
            video_frames = torch.cat([video_frames, last], dim=0)
        
        if self.processor:
            video_frames = self.processor(images=list(video_frames), return_tensors="pt")['pixel_values'][0] #[T, C, H, W]
            # assert False, video_frames.shape
            # video_frames = torch.stack([self.processor(images=frame, return_tensors="pt")['pixel_values'][0] for frame in video_frames])
        padded_video = self.pad_or_trim_dim(video_frames, pad_dim=0, max_len=self.max_len, pad_value=0.0)  # (T, C, H, W) -> (max_len, C, H, W)
            
        # TEXT TOKENS
        text = self.tokenizer(self.texts[idx], return_tensors='pt')['input_ids'][0]  # (T, )
        padded_text = self.pad_or_trim_dim(text, pad_dim=0, max_len=self.max_len, pad_value=0.0)
        return {"video": padded_video['out'],
                "video_mask": padded_video['mask'],
                "flows": flow,
                "frames_idx": frames_idx,
                "mask": packed["mask"],
                "length": 0,
                "transcript": padded_text['out'],
                "text_mask": padded_text['mask'],
                "path": path,
                "audio": audio,
                "audio_mask": audio_mask,
                }
    
from torchvision.transforms import v2

def make_transform(resize_size=(224, 224)):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(resize_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

class SEAMLESSDataSimple(Dataset):
    """
    Returns:
      {
        "video":      (T, 3, H, W)
        "transcript": (T_text,)
        "text_mask":  (T_text,)
        "audio":      (T_audio, D)
        "audio_mask": int (Whisper frame length)
      }
    """

    def __init__(
        self,
        csv_path: str,
        resize: int = 224,
        max_len: int = 32,
    ):
        self.resize = resize
        self.max_len = max_len

        df = pd.read_csv(csv_path)[:2000]
        self.video_paths = df["path_to_clip"].astype(str).tolist()
        self.texts = df["transcript"].fillna("").astype(str).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        # self.tokenizer.requires_grad_(False)
        # self.audio_processor.requires_grad_(False)
        self.transform = make_transform((resize, resize))

    def __len__(self):
        return len(self.video_paths)

    # ---------------- VIDEO ----------------
    # def _read_video_tensor(self, path: str) -> torch.Tensor:
    #     frames, _, _ = read_video(path, pts_unit="sec")  # (T,H,W,C)
    #     if frames.numel() == 0:
    #         frames = torch.zeros((1, self.resize, self.resize, 3), dtype=torch.uint8)

    #     # Convert to (T,C,H,W)
    #     frames = frames.permute(0, 3, 1, 2)

    #     # Apply transform frame-wise
    #     frames = torch.stack([self.transform(f) for f in frames])  # (T,3,H,W)

    #     return frames
    
    def _read_video_tensor(self, path: str):

        vr = VideoReader(path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames == 0:
            raise ValueError(f"Empty video: {path}")

        # Uniform sampling
        step = total_frames / self.max_len
        indices = (torch.arange(self.max_len) * step).long()
        indices = torch.clamp(indices, max=total_frames - 1)

        # Decode selected frames
        frames = vr.get_batch(indices.tolist())   # decord NDArray (T,H,W,3)

        # ✅ Convert to torch tensor
        frames = torch.from_numpy(frames.asnumpy())

        # (T,H,W,3) -> (T,3,H,W)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        if self.transform is not None:
            frames = torch.stack([self.transform(f) for f in frames])

        return frames

    # def _pad_video(self, video: torch.Tensor):
    #     T = video.shape[0]
    #     out = torch.zeros(
    #         (self.max_len, *video.shape[1:]),
    #         dtype=video.dtype
    #     )

    #     use = min(T, self.max_len)
    #     out[:use] = video[:use]
    #     return out
    
    def _sample_video(self, video):
        T = video.shape[0]
        step = T / self.max_len
        indices = (torch.arange(self.max_len) * step).long()
        assert len(indices)==self.max_len, f"indices length is not equal max_len {len(indices)}, {self.max_len}, {T}, {step}"

        return video[indices.clamp(max=T-1)]
    
    # ---------------- AUDIO ----------------
    def _read_audio_tensor(self, path: str):
        audio = load_audio_from_mp4(path)

        inputs = self.audio_processor(
            audio.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        audio_feat = inputs["input_features"].squeeze(0)
        audio_mask = int(audio.shape[0] / 16000 * 50)

        return audio_feat, audio_mask

    # ---------------- GET ITEM ----------------
    def __getitem__(self, idx):

        path = self.video_paths[idx]

        # VIDEO
        video = self._read_video_tensor(path)      # (T,3,H,W)
        video = self._sample_video(video)             # (max_len,3,H,W)

        # TEXT
        text_ids = self.tokenizer(
            self.texts[idx],
            return_tensors="pt"
        )["input_ids"][0]

        padded_text = torch.zeros(self.max_len, dtype=torch.long)
        text_mask = torch.zeros(self.max_len, dtype=torch.bool)

        use = min(len(text_ids), self.max_len)
        padded_text[:use] = text_ids[:use]
        text_mask[:use] = True

        # AUDIO
        audio, audio_mask = self._read_audio_tensor(path)

        return {
            "video": video,                 # (T,3,H,W)
            "transcript": padded_text,      # (T,)
            "text_mask": text_mask,         # (T,)
            "audio": audio,                 # (T_audio, D)
            "audio_mask": audio_mask,       # int
        }
    
def simple_collate(batch):

    B = len(batch)

    # ---------------- VIDEO ----------------
    video = torch.stack([x["video"] for x in batch])  # (B,T,3,H,W)

    # ---------------- TEXT ----------------
    transcript = torch.stack([x["transcript"] for x in batch])
    text_mask = torch.stack([x["text_mask"] for x in batch])

    # ---------------- AUDIO ----------------
    max_audio_len = max(x["audio"].shape[0] for x in batch)
    D = batch[0]["audio"].shape[1]

    audio = torch.zeros((B, max_audio_len, D))
    audio_mask = torch.zeros((B,), dtype=torch.long)

    for i, sample in enumerate(batch):
        a = sample["audio"]
        audio[i, :a.shape[0]] = a
        audio_mask[i] = sample["audio_mask"]

    return {
        "video": video,
        "transcript": transcript,
        "text_mask": text_mask,
        "audio": audio,
        "audio_mask": audio_mask,
    }