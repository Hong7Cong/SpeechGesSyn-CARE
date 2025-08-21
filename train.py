import argparse
import os
import time
from typing import Any, Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import yaml

# your code
from loss import multimodal_alignment_loss, global_clip_loss
from CAREdataset import SEAMLESSData, pad_time_collate
from model import VideoFlowModel


# ---------------------------
# Utils
# ---------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def move_to_device(batch_item: Union[torch.Tensor, Dict[str, torch.Tensor]], device: str):
    """Move tensors or dicts of tensors to device."""
    if isinstance(batch_item, torch.Tensor):
        return batch_item.to(device, non_blocking=True)
    if isinstance(batch_item, dict):
        return {k: v.to(device, non_blocking=True) for k, v in batch_item.items()}
    return batch_item  # leave as-is for types we don't recognize

def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg

def format_seconds(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:d}h {m:d}m {s:d}s"
    if m: return f"{m:d}m {s:d}s"
    return f"{s:d}s"

# ---------------------------
# Data
# ---------------------------
def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg.get("val_csv", None)
    resize = data_cfg.get("resize", 224)
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", False)

    ds_train = SEAMLESSData(train_csv, resize=resize)

    if val_csv:
        ds_val = SEAMLESSData(val_csv, resize=resize)
    else:
        # Fallback: small validation split from training data
        val_split = data_cfg.get("val_split", 0.05)
        val_len = max(1, int(len(ds_train) * val_split))
        train_len = len(ds_train) - val_len
        ds_train, ds_val = random_split(ds_train, [train_len, val_len],
                                        generator=torch.Generator().manual_seed(cfg.get("seed", 42)))

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_time_collate,
        pin_memory=pin_memory,
        drop_last=False,
    )

    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_time_collate,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return loader_train, loader_val

# ---------------------------
# Model / Optim / Loss
# ---------------------------
def build_model(cfg: Dict[str, Any], device: str) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    # If your VideoFlowModel takes kwargs, pass them from model_cfg
    model = VideoFlowModel(**model_cfg).to(device)
    return model

def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    lr = optim_cfg.get("lr", 1e-5)
    weight_decay = optim_cfg.get("weight_decay", 0.0)
    betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay, betas=betas
    )

def build_scaler(cfg: Dict[str, Any]) -> GradScaler:
    use_amp = cfg["train"].get("amp", False) and torch.cuda.is_available()
    return GradScaler(enabled=use_amp)

def get_loss_fn(cfg: Dict[str, Any]):
    loss_cfg = cfg["loss"]
    name = loss_cfg.get("name", "global_clip").lower()
    if name in ["global_clip", "global_clip_loss"]:
        def _loss(v_feats, t_feats, v_mask=None, t_mask=None):
            return global_clip_loss(
                v_feats, t_feats,
                v_mask=v_mask, t_mask=t_mask,
                temperature=loss_cfg.get("temperature", 0.07)
            )
        return _loss
    elif name in ["multimodal_alignment", "multimodal_alignment_loss", "alignment"]:
        def _loss(v_feats, t_feats, v_mask=None, t_mask=None):
            return multimodal_alignment_loss(
                v_feats, t_feats,
                v_mask=v_mask, t_mask=t_mask,
                **loss_cfg.get("kwargs", {})
            )
        return _loss
    else:
        raise ValueError(f"Unknown loss name: {name}")

# ---------------------------
# Train / Validate
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device: str, loss_fn, cfg: Dict[str, Any]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        vids = move_to_device(batch["video"], device)  # (B, C, T, H, W) after pad/collate
        transcript = move_to_device(batch["transcript"], device)  # tensor or dict->tensor(s)

        v_mask = move_to_device(batch.get("video_mask", None), device) if "video_mask" in batch else None
        t_mask = move_to_device(batch.get("text_mask", None), device) if "text_mask" in batch else None

        v_feats, t_feats = model(vids, transcript)  # v:[B,T1,D], t:[B,T2,D]
        loss = loss_fn(v_feats, t_feats, v_mask=v_mask, t_mask=t_mask)

        total_loss += float(loss.item())
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return {"val_loss": avg_loss}

def train_one_epoch(model, loader, optimizer, scaler, device: str, loss_fn, epoch: int, cfg: Dict[str, Any]) -> Dict[str, float]:
    model.train()
    log_interval = cfg.get("logger", {}).get("log_interval", 50)
    grad_clip = cfg["train"].get("grad_clip_norm", None)
    use_amp = cfg["train"].get("amp", False) and torch.cuda.is_available()

    running = 0.0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, 1):
        vids = move_to_device(batch["video"], device)
        transcript = move_to_device(batch["transcript"], device)
        v_mask = move_to_device(batch.get("video_mask", None), device) if "video_mask" in batch else None
        t_mask = move_to_device(batch.get("text_mask", None), device) if "text_mask" in batch else None

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            v_feats, t_feats = model(vids, transcript)
            loss = loss_fn(v_feats, t_feats, v_mask=v_mask, t_mask=t_mask)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running += float(loss.item())
        n += 1

        if step % log_interval == 0:
            avg = running / max(1, n)
            elapsed = time.time() - t0
            print(f"[Epoch {epoch:03d} | Step {step:05d}] loss={avg:.4f} ({format_seconds(elapsed)})")

    return {"train_loss": running / max(1, n)}

# ---------------------------
# Checkpointing
# ---------------------------
def save_checkpoint(state: Dict[str, Any], path: str):
    torch.save(state, path)

def maybe_save_best(best_metric: float, current: float, output_dir: str, model, optimizer, scaler, epoch: int, cfg: Dict[str, Any]) -> float:
    if current < best_metric:
        best_metric = current
        ckpt_path = os.path.join(output_dir, "best.pt")
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "best_val_loss": best_metric,
        }, ckpt_path)
        print(f"✓ Saved best checkpoint to: {ckpt_path} (val_loss={current:.4f})")
    return best_metric

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = resolve_device(cfg.get("device", "auto"))
    out_dir = cfg.get("output_dir", "runs/default")
    ensure_dir(out_dir)

    # persist a copy of the config for reproducibility
    with open(os.path.join(out_dir, "config.used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    loader_train, loader_val = build_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scaler = build_scaler(cfg)
    loss_fn = get_loss_fn(cfg)

    start_epoch = 1
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))

    epochs = cfg["train"].get("epochs", 30)
    val_every = cfg["train"].get("val_every", 1)

    wall0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(model, loader_train, optimizer, scaler, device, loss_fn, epoch, cfg)
        msg = f"Epoch {epoch:03d} | train_loss={train_stats['train_loss']:.4f}"

        if (epoch % val_every) == 0:
            val_stats = evaluate(model, loader_val, device, loss_fn, cfg)
            msg += f" | val_loss={val_stats['val_loss']:.4f}"
            best_val = maybe_save_best(best_val, val_stats["val_loss"], out_dir, model, optimizer, scaler, epoch, cfg)

        print(msg)

        # Save "last" every epoch
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg,
            "best_val_loss": best_val,
        }, os.path.join(out_dir, "last.pt"))

    print(f"Done. Total time: {format_seconds(time.time() - wall0)}")
    

if __name__ == "__main__":
    main()
