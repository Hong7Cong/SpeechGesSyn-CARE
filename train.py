from loss import multimodal_alignment_loss, global_clip_loss
from CAREdataset import SEAMLESSData, pad_time_collate
from torch.utils.data import DataLoader
from model import VideoFlowModel
import torch 

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
ds = SEAMLESSData("/data1/open_data/seamless_interaction/preprocessed/train/clips.csv", resize=224)  # fixed H,W; variable T
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1, collate_fn=pad_time_collate, pin_memory=False)
model = VideoFlowModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for e in range(EPOCHS):
    model.train()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        
        vids = batch["video"].to(DEVICE)       # (B, C, T_max, H, W)
        transcript = batch["transcript"].to(DEVICE)                     # (B,)
        v_mask = None               # (B, T1) optional
        t_mask = batch["text_mask"].to(DEVICE)                # (B, T2) optional
        if v_mask is not None: v_mask = v_mask.to(DEVICE)
        if t_mask is not None: t_mask = t_mask.to(DEVICE)
        
        v_feats, t_feats = model(vids, transcript)           # v:[B,T1,768], t:[B,T2,768]
        
        loss = global_clip_loss(
            v_feats, t_feats, v_mask=v_mask, t_mask=t_mask,
            temperature=0.07
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # (optional) log pieces
    print(f"loss={loss.item():.4f}")