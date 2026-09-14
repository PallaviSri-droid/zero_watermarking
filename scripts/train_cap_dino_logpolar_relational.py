from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from zero_watermarking.cap_dino_logpolar_relational import CAPDinoLogPolarRelational, RELATIONAL_VERSION, RelationalConfig, relational_objective
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.protocol import seed_everything
from zero_watermarking.training import PairAttackDataset, _update_memory

ATTACKS = ("gaussian_noise", "salt_pepper", "gaussian_blur", "median_blur", "jpeg", "brightness", "contrast", "rotation", "crop_resize", "translation", "compound")

def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad)

def _load_warm_start(model: torch.nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location=model.device if hasattr(model, "device") else "cpu", weights_only=False)
    state = payload.get("model", payload)
    current = model.state_dict(); matched = {}
    for key, value in state.items():
        if key in current and current[key].shape == value.shape: matched[key] = value
    missing = [k for k in current if k not in matched]
    model.load_state_dict(matched, strict=False)
    print(f"Warm-start loaded={len(matched)} missing_new={len(missing)} from={checkpoint}")

def main() -> int:
    ap = argparse.ArgumentParser(description="Train relational CAP-DINO-LogPolar candidate from stable repair.")
    ap.add_argument("--manifest", default="data/manifests/medical_manifest.csv"); ap.add_argument("--split", default="train_val"); ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--bits", type=int, default=128); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--stage1-epochs", type=int, default=1); ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--attack-views", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="auto", choices=("auto","cpu","cuda"))
    ap.add_argument("--init-checkpoint", default="experiments/checkpoints/cap_dino_logpolar_stable_seed42.pt"); ap.add_argument("--checkpoint", default="experiments/checkpoints/cap_dino_logpolar_relational_seed42.pt")
    ap.add_argument("--history", default="experiments/results/cap_dino_logpolar_relational_seed42_training_history.csv")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    frame = validate_manifest(load_manifest(args.manifest)); frame = frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns: frame = frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame = frame.head(args.limit)
    if len(frame) < 8: raise SystemExit("Need at least 8 images.")
    images = np.stack([load_image(row.path, args.size) for row in frame.itertuples(index=False)])
    labels = np.arange(len(images), dtype=np.int64)
    dataset = PairAttackDataset(images, labels, attack_names=ATTACKS, attack_views=args.attack_views)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device=="cuda"), generator=torch.Generator().manual_seed(args.seed))
    cfg = RelationalConfig(bits=args.bits, device=device, attack_views=args.attack_views); cfg.epochs=args.epochs; cfg.lr=args.lr
    model=CAPDinoLogPolarRelational(cfg); init=Path(args.init_checkpoint)
    if init.exists(): _load_warm_start(model, init)
    else: print(f"Warm-start checkpoint not found: {init}")
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=2e-4, eps=1e-8)
    memory_codes=torch.empty((0,args.bits),dtype=torch.float32,device=device); memory_labels=torch.empty((0,),dtype=torch.long,device=device)
    history=[]; skipped=0
    for epoch in range(args.epochs):
        model.train(); sums={}; batches=0; stage=1 if epoch < min(args.stage1_epochs,args.epochs) else 2
        for batch in loader:
            clean_x, attacked_x, labels_t, _ = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            pair=model.forward_pair(clean_x,attacked_x); mem=memory_codes if memory_codes.numel() else None; mem_labels=memory_labels if memory_labels.numel() else None
            loss,terms=relational_objective(pair,labels_t,mem,mem_labels,cfg,stage=stage)
            if not torch.isfinite(loss) or not all(torch.isfinite(v).all().item() for v in terms.values()): optimizer.zero_grad(set_to_none=True); skipped+=1; continue
            optimizer.zero_grad(set_to_none=True); loss.backward(); bad=any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters() if p.requires_grad)
            if bad: optimizer.zero_grad(set_to_none=True); skipped+=1; continue
            torch.nn.utils.clip_grad_norm_(model.parameters(),3.0,error_if_nonfinite=True); optimizer.step()
            if not _finite_parameters(model): raise RuntimeError("Non-finite parameters after optimizer.step().")
            memory_codes,memory_labels=_update_memory(memory_codes,memory_labels,pair["clean_hard"].detach(),labels_t,cfg.memory_size)
            row_terms={k:float(v.detach()) for k,v in terms.items()}; row_terms.update({"loss":float(loss.detach()),"stage":float(stage),"relation_scale_learned":float(model.relation_scale.detach())})
            for k,v in row_terms.items(): sums[k]=sums.get(k,0.0)+v
            batches+=1
        d=max(batches,1); row={"epoch":float(epoch+1),"stage":float(stage),"skipped_batches":float(skipped)}; row.update({k:v/d for k,v in sums.items()}); history.append(row)
        out=Path(args.checkpoint); out.parent.mkdir(parents=True,exist_ok=True); torch.save({"version":RELATIONAL_VERSION,"model":model.state_dict(),"config":cfg.__dict__,"optimizer":optimizer.state_dict(),"epoch":epoch+1,"history":history,"init_checkpoint":str(init)},out)
        hp=Path(args.history); hp.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(history).to_csv(hp,index=False)
        print(f"epoch={epoch+1} stage={stage} loss={row.get('loss',0):.5f} robust={row.get('robustness',0):.5f} q={row.get('robustness_q',0):.5f} contrastive={row.get('contrastive',0):.5f} entropy={row.get('observed_entropy',0):.4f} relation_consistency={row.get('relation_consistency',0):.5f} relation_scale={row.get('relation_scale_learned',0):.5f} skipped={skipped}")
    print(f"Training complete: {RELATIONAL_VERSION} seed={args.seed} images={len(images)} effective_samples={len(dataset)} bits={args.bits} epochs={args.epochs} device={device}")
    print(f"checkpoint={Path(args.checkpoint).resolve()}"); print(f"history={Path(args.history).resolve()}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
