import torch
import torch.nn.functional as F

# def masked_mean(x, mask, dim):
#     """
#     x:   [B, T, D]
#     mask:[B, T] with 1 for valid, 0 for pad (or None)
#     returns mean over valid positions, shape [B, D]
#     """
#     if mask is None:
#         return x.mean(dim=dim)
#     denom = mask.sum(dim=dim, keepdim=True).clamp_min(1e-6)
#     return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom

def masked_mean(x, mask=None, dim=1, eps=1e-6):
    """
    x:    [B, T, D]
    mask:
        - None
        - Bool/0-1 tensor [B, T]
        - Index tensor [B]  (0..idx-1 valid)
    returns: [B, D]
    """
    B, T, _ = x.shape

    if mask is None:
        return x.mean(dim=dim)

    # Convert list → tensor
    if isinstance(mask, list):
        mask = torch.tensor(mask, device=x.device)

    # Case 1: index mask [B]
    if mask.ndim == 1:
        mask = mask.long().clamp(0, T)
        arange = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        mask = arange < mask.unsqueeze(1)                       # [B, T]

    # Case 2: boolean or 0/1 mask [B, T]
    mask = mask.to(dtype=x.dtype)

    x = x * mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp(min=eps)

    return x.sum(dim=dim) / denom.unsqueeze(-1)

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
    # v_pool = v_feats
    # t_pool = t_feats
    v = F.normalize(v_pool, dim=-1)
    t = F.normalize(t_pool, dim=-1)
    # assert False, "v.shape={}, t.shape={}".format(v.shape, t.shape)
    logits = (v @ t.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)

def global_clip_loss_af(v_feats, t_feats, temperature=0.07):
    """
    CLIP loss for already pooled features
    v_feats, t_feats: [B, D]
    """

    v = F.normalize(v_feats, dim=-1)
    t = F.normalize(t_feats, dim=-1)

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


def trajectory_alignment_loss(
    speech_feats,   # [B, T, D]
    video_feats,    # [B, T, D]
    text_feats,     # [B, T_text, D]
    lambda_energy=0.5,
):
    """
    Temporal alignment between speech prosody and gesture motion.
    Conditioned on text embedding.
    """

    # ---- Temporal derivatives ----
    d_s = speech_feats[:, 1:] - speech_feats[:, :-1]   # [B, T-1, D]
    d_v = video_feats[:, 1:] - video_feats[:, :-1]

    # ---- Text conditioning (semantic gating) ----
    # Use global text embedding to modulate dynamics
    z_t = text_feats.mean(dim=1)        # [B, D]
    gate = torch.sigmoid(z_t)          # [B, D]

    gate = gate.unsqueeze(1)           # [B, 1, D]
    d_s = d_s * gate
    d_v = d_v * gate

    # ---- Velocity cosine alignment ----
    d_s_norm = F.normalize(d_s, dim=-1)
    d_v_norm = F.normalize(d_v, dim=-1)

    vel_loss = 1 - (d_s_norm * d_v_norm).sum(dim=-1)
    vel_loss = vel_loss.mean()

    # ---- Energy alignment ----
    e_s = torch.norm(d_s, dim=-1)   # [B, T-1]
    e_v = torch.norm(d_v, dim=-1)

    energy_loss = F.mse_loss(e_s, e_v)

    return vel_loss + lambda_energy * energy_loss

def pairwise_contrastive(z1, z2, temperature=0.07):
    """
    z1, z2: [B, D]
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    return F.cross_entropy(logits, labels)


# -------------------------------------------------
# Tri-Semantic Loss
# -------------------------------------------------

def tri_semantic_loss(
    speech_feats,      # [B, T_a, D]
    video_feats,       # [B, T_a, D]
    text_feats,        # [B, T_text, D]
    temperature=0.07,
):
    """
    Uses:
        - last token of speech
        - last token of video
        - mean pooled text
    """

    z_s = speech_feats[:, -1, :]           # [B, D]
    z_v = video_feats[:, -1, :]            # [B, D]
    z_t = text_feats.mean(dim=1)           # [B, D]

    loss_sv = pairwise_contrastive(z_s, z_v, temperature)
    loss_st = pairwise_contrastive(z_s, z_t, temperature)
    loss_vt = pairwise_contrastive(z_v, z_t, temperature)

    return (loss_sv + loss_st + loss_vt) / 3.0