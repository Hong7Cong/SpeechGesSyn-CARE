import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any

from torch.utils.data import DataLoader

from CAREdataset import SEAMLESSDataDEV, pad_time_collate
from model import SpeechGestureModelDev


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def move_to_device(batch_item, device: str):
    if isinstance(batch_item, torch.Tensor):
        return batch_item.to(device, non_blocking=True)
    if isinstance(batch_item, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch_item.items()}
    return batch_item


# ---------------------------------------------------------
# Build validation dataloader
# ---------------------------------------------------------

def build_val_loader(cfg: Dict[str, Any]) -> DataLoader:
    data_cfg = cfg["data"]
    val_csv = data_cfg.get("test_csv", None)

    if val_csv is None:
        raise ValueError("test_csv must be specified in config for evaluation.")

    ds_val = SEAMLESSDataDEV(
        val_csv,
        resize=data_cfg.get("resize", 224),
        max_len=data_cfg.get("max_len", 8),
    )

    loader = DataLoader(
        ds_val,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=pad_time_collate,
        pin_memory=data_cfg.get("pin_memory", False),
        drop_last=False,
    )

    return loader


# ---------------------------------------------------------
# Retrieval Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(model, loader, device: str, topk=(1, 5, 10)):

    model.eval()

    all_speech = []
    all_video = []

    for batch in loader:

        vids       = move_to_device(batch["video"], device)
        transcript = move_to_device(batch["transcript"], device)
        audio_mask = move_to_device(batch.get("audio_mask", None), device)
        audio      = move_to_device(batch.get("audio", None), device)
        flows      = move_to_device(batch.get("flows", None), device)

        video_feats, speech_feats, _ = model(
            video=vids,
            flows=flows,
            transcript=transcript,
            audio=audio,
            audio_mask=audio_mask,
        )

        # Use last token embeddings
        z_s = speech_feats[:, -1, :]
        z_v = video_feats[:, -1, :]

        all_speech.append(z_s)
        all_video.append(z_v)

    z_s = torch.cat(all_speech, dim=0)
    z_v = torch.cat(all_video, dim=0)

    # Normalize
    z_s = F.normalize(z_s, dim=-1)
    z_v = F.normalize(z_v, dim=-1)

    # Similarity matrix
    sim_matrix = torch.matmul(z_s, z_v.T)
    N = sim_matrix.size(0)

    gt = torch.arange(N, device=device)
    ranks = sim_matrix.argsort(dim=1, descending=True)

    results = {}

    for k in topk:
        correct = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float().mean()
        results[f"R@{k}"] = correct.item()

    # mean rank
    rank_positions = (ranks == gt.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    results["mean_rank"] = rank_positions.float().mean().item() + 1

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))

    # Build model
    model_cfg = cfg.get("model", {})
    model = SpeechGestureModelDev(**model_cfg).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # Build val loader
    loader = build_val_loader(cfg)

    # Evaluate
    results = evaluate_retrieval(model, loader, device)

    print("\n=== Speech ↔ Gesture Retrieval ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()