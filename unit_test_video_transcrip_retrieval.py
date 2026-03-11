import torch
import torch.nn.functional as F
from typing import Optional
import argparse
from train import load_config, ensure_dir, move_to_device, resolve_device, format_seconds, build_dataloaders, build_model

def masked_mean(x, mask, dim: int):
    # mask: (B, T) or (B, T, 1); x: (B, T, D) or (B, D)
    # ensure mask is float and broadcastable
    if mask.dtype != x.dtype:
        mask = mask.to(x.dtype)

    # If mask has one fewer dim than x, add the last dim for multiplication.
    if mask.ndim == x.ndim - 1:
        mask_for_mul = mask.unsqueeze(-1)
    else:
        mask_for_mul = mask

    masked = x * mask_for_mul
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)

    # IMPORTANT: don't squeeze denom; keepdim=True makes it broadcastable
    return masked.sum(dim=dim) / denom

# --------- your loss (unchanged) ---------
def global_clip_loss(v_feats, t_feats, v_mask: Optional[torch.Tensor] = None, t_mask: Optional[torch.Tensor] = None, temperature: float = 0.07):
    if v_feats.ndim == 3:
        v_pool = masked_mean(v_feats, v_mask, dim=1)
    else:
        v_pool = v_feats
    if t_feats.ndim == 3:
        t_pool = masked_mean(t_feats, t_mask, dim=1)
    else:
        t_pool = t_feats

    v = F.normalize(v_pool, dim=-1)   # [B, D]
    t = F.normalize(t_pool, dim=-1)   # [B, D]

    logits = (v @ t.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

# --------- retrieval + probability computation ---------
@torch.no_grad()
def vt_probabilities(
    v_feats: torch.Tensor,
    t_feats: torch.Tensor,
    v_mask: Optional[torch.Tensor] = None,
    t_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
):
    if v_feats.ndim == 3:
        v_pool = masked_mean(v_feats, v_mask, dim=1)
    else:
        v_pool = v_feats
    if t_feats.ndim == 3:
        t_pool = masked_mean(t_feats, t_mask, dim=1)
    else:
        t_pool = t_feats

    v_pool = F.normalize(v_pool, dim=-1)
    t_pool = F.normalize(t_pool, dim=-1)

    logits = (v_pool @ t_pool.t()) / temperature          # [Nv, Nt]
    p_text_given_vid = F.softmax(logits, dim=1)           # row-wise over texts
    p_vid_given_text = F.softmax(logits.t(), dim=1)       # row-wise over videos

    return logits, p_text_given_vid, p_vid_given_text, v_pool, t_pool

@torch.no_grad()
def topk_retrieve_from_videos(p_text_given_vid: torch.Tensor, k: int = 5):
    probs, idx = p_text_given_vid.topk(k, dim=1)
    return idx, probs

@torch.no_grad()
def topk_retrieve_from_texts(p_vid_given_text: torch.Tensor, k: int = 5):
    probs, idx = p_vid_given_text.topk(k, dim=1)
    return idx, probs

@torch.no_grad()
def recall_at_k(logits: torch.Tensor, k: int = 1):
    B = logits.shape[0]
    topk_t = logits.topk(k, dim=1).indices
    r_i2t = (topk_t == torch.arange(B, device=logits.device).unsqueeze(1)).any(dim=1).float().mean()
    topk_v = logits.t().topk(k, dim=1).indices
    r_t2i = (topk_v == torch.arange(B, device=logits.device).unsqueeze(1)).any(dim=1).float().mean()
    return r_i2t.item(), r_t2i.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/retrieval_debug.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", default="./runs/debug_run/run20250928/best.pt", help="Optional checkpoint to resume from.")
    args = parser.parse_args(["--config", "./configs/retrieval_debug.yaml"])
    args = parser.parse_args(["--resume", "./runs/debug_run/run20250928/best.pt"])
    cfg = load_config(args.config)

    loader_train, loader_val = build_dataloaders(cfg)
    model = build_model(cfg, "cuda:0")

    ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    _ = next(iter(loader_train))
    batch = next(iter(loader_train))
    device = "cuda:0"

    vids       = move_to_device(batch["video"], device)
    transcript = move_to_device(batch["transcript"], device)
    v_mask     = move_to_device(batch.get("video_mask", None), device) if "video_mask" in batch else None
    t_mask     = move_to_device(batch.get("text_mask", None), device)  if "text_mask"  in batch else None

    v_feats, t_feats = model(vids, transcript)

    logits, p_t_given_v, p_v_given_t, v_pool, t_pool = vt_probabilities(
        v_feats, t_feats, v_mask=v_mask, t_mask=t_mask, temperature=0.07
    )

    loss = global_clip_loss(v_feats, t_feats, v_mask, t_mask, temperature=0.07)
    print(f"Contrastive loss: {loss.item():.4f}")

    # Top-3 texts for each video (indices and probs)
    ti, tp = topk_retrieve_from_videos(p_t_given_v, k=4)
    print("Top-3 texts for each video:\n", ti, "\nprobs:\n", tp)

    # Top-3 videos for each text (indices and probs)
    vi, vp = topk_retrieve_from_texts(p_vid_given_text=p_v_given_t, k=4)
    print("Top-3 videos for each text:\n", vi, "\nprobs:\n", vp)

    # If pairs are aligned (i == j), compute Recall@K
    r1_i2t, r1_t2i = recall_at_k(logits, k=1)
    print(f"R@1 i→t: {r1_i2t:.3f} | t→i: {r1_t2i:.3f}")