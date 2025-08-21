import torch
import torch.nn.functional as F

def masked_mean(x, mask, dim):
    """
    x:   [B, T, D]
    mask:[B, T] with 1 for valid, 0 for pad (or None)
    returns mean over valid positions, shape [B, D]
    """
    if mask is None:
        return x.mean(dim=dim)
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom

def pairwise_cosine(v, t):
    """
    v: [B, T1, D]  (already normalized)
    t: [B, T2, D]  (already normalized)
    returns sim: [B, T1, T2]
    """
    # cosine since v and t are L2-normalized
    return torch.einsum('btd,bsd->bts', v, t)

def temporal_alignment_loss(v_feats, t_feats, v_mask=None, t_mask=None):
    """
    Bidirectional best-match cosine alignment.
    v_feats: [B, T1, D], t_feats: [B, T2, D]
    masks  : [B, T1], [B, T2] with 1=valid, 0=pad (or None)
    """
    # L2-normalize along feature dim
    v = F.normalize(v_feats, dim=-1)
    t = F.normalize(t_feats, dim=-1)

    sim = pairwise_cosine(v, t)  # [B, T1, T2]

    # If masks are provided, exclude padded positions from the max
    if t_mask is not None:
        # invalid text positions -> -inf so they never win max
        sim = sim.masked_fill(~t_mask[:, None, :].bool(), float('-inf'))
    # For v->t: best text for each video step
    v2t_max = sim.max(dim=2).values  # [B, T1]

    sim_t = sim
    if v_mask is not None:
        sim_t = sim_t.masked_fill(~v_mask[:, :, None].bool(), float('-inf'))
    # For t->v: best video for each text step
    t2v_max = sim_t.max(dim=1).values  # [B, T2]

    # Average only over valid positions
    if v_mask is not None:
        v2t_mean = (v2t_max * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp_min(1e-6)
    else:
        v2t_mean = v2t_max.mean(dim=1)

    if t_mask is not None:
        t2v_mean = (t2v_max * t_mask).sum(dim=1) / t_mask.sum(dim=1).clamp_min(1e-6)
    else:
        t2v_mean = t2v_max.mean(dim=1)

    # We want to maximize cosine similarity -> minimize (1 - mean_sim)
    bidir_mean = 0.5 * (v2t_mean + t2v_mean)  # [B]
    loss_local = (1.0 - bidir_mean).mean()
    return loss_local

def global_clip_loss(v_feats, t_feats, v_mask=None, t_mask=None, temperature=0.07):
    """
    Mean-pool over time, then batchwise contrastive (CLIP-style).
    """
    v_pool = masked_mean(v_feats, v_mask, dim=1)  # [B, D]
    t_pool = masked_mean(t_feats, t_mask, dim=1)  # [B, D]

    v = F.normalize(v_pool, dim=-1)
    t = F.normalize(t_pool, dim=-1)
    # assert False, "v.shape={}, t.shape={}".format(v.shape, t.shape)
    logits = (v @ t.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

def multimodal_alignment_loss(v_feats, t_feats, v_mask=None, t_mask=None,
                              alpha=0.5, temperature=0.07):
    """
    Total loss = alpha * local (temporal) + (1-alpha) * global contrastive.
    """
    loss_local = temporal_alignment_loss(v_feats, t_feats, v_mask, t_mask)
    loss_global = global_clip_loss(v_feats, t_feats, v_mask, t_mask, temperature)
    return alpha * loss_local + (1 - alpha) * loss_global, (loss_local, loss_global)
