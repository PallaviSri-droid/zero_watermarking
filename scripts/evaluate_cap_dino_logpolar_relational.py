from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from zero_watermarking.attacks import ATTACKS
from zero_watermarking.cap_dino_logpolar_relational import CAPDinoLogPolarRelational, RELATIONAL_VERSION, RelationalConfig
from zero_watermarking.datasets import load_image, load_manifest, validate_manifest
from zero_watermarking.metrics import bit_balance, bit_entropy, mean_abs_corr, evaluate_hash_bank

ATTACK_GRID = (("gaussian_noise", {"sigma":0.03,"seed":11}), ("gaussian_noise", {"sigma":0.08,"seed":17}), ("gaussian_blur", {"sigma":1.0}), ("gaussian_blur", {"sigma":2.0}), ("jpeg", {"quality":70}), ("jpeg", {"quality":40}), ("rotation", {"degrees":5.0}), ("rotation", {"degrees":10.0}), ("crop_resize", {"fraction":0.05}), ("translation", {"pixels":3}), ("compound", {"seed":23}))

def cfg_from(payload: dict, device: str) -> RelationalConfig:
    raw=dict(payload.get("config",{})); valid={f.name for f in fields(RelationalConfig)}; cfg=RelationalConfig(**{k:v for k,v in raw.items() if k in valid}); cfg.device=device; return cfg

def main() -> int:
    ap=argparse.ArgumentParser(description="Evaluate relational CAP-DINO-LogPolar candidate.")
    ap.add_argument("--manifest",default="data/manifests/medical_manifest.csv"); ap.add_argument("--split",default="test")
    ap.add_argument("--checkpoint",default="experiments/checkpoints/cap_dino_logpolar_relational_seed42.pt"); ap.add_argument("--out",default="experiments/results/cap_dino_logpolar_relational_eval_200")
    ap.add_argument("--limit",type=int,default=200); ap.add_argument("--size",type=int,default=128); ap.add_argument("--device",default="auto",choices=("auto","cpu","cuda")); ap.add_argument("--near-threshold",type=float,default=0.10)
    args=ap.parse_args(); device="cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device
    payload=torch.load(Path(args.checkpoint),map_location=device,weights_only=False); cfg=cfg_from(payload,device); model=CAPDinoLogPolarRelational(cfg); model.load_state_dict(payload["model"]); model.eval()
    frame=validate_manifest(load_manifest(args.manifest)); frame=frame[frame["exists"]].reset_index(drop=True)
    if "split" in frame.columns: frame=frame[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    frame=frame.head(args.limit)
    clean_bank={}; attacked_bank={}; gates={}; relation_scales=[]
    with torch.inference_mode():
        for row in frame.itertuples(index=False):
            image=load_image(row.path,args.size); x=torch.from_numpy(image[None,None]).to(device); pair=model.forward_pair(x,x)
            clean_bank[row.image_id]=pair["clean_hard"].round().to(torch.uint8).cpu().numpy()[0]; attacked_bank[row.image_id]={}; gates[row.image_id]=pair["clean_gates"].cpu().numpy()[0]; relation_scales.append(float(model.relation_scale.cpu()))
            for i,(name,kwargs) in enumerate(ATTACK_GRID):
                p=dict(kwargs)
                if name in {"gaussian_noise","compound"}: p["seed"]=int(p.get("seed",0))+i
                attacked=ATTACKS[name](image,**p); ax=torch.from_numpy(np.asarray(attacked,dtype=np.float32)[None,None]).to(device); q=model.forward_pair(x,ax)
                attacked_bank[row.image_id][f"{name}_{i}"]=q["attacked_hard"].round().to(torch.uint8).cpu().numpy()[0]
    result=evaluate_hash_bank(clean_bank,attacked_bank); ids=sorted(clean_bank); matrix=np.stack([clean_bank[k] for k in ids]); stats=result["collision_statistics"]; g=np.stack([gates[k] for k in ids]); ent,_=bit_entropy(matrix)
    dominant=np.argmax(g,axis=1); summary={"version":str(payload.get("version",RELATIONAL_VERSION)),"images":len(ids),"bits":int(matrix.shape[1]),"mean_nc":result["mean_nc"],"mean_ber":result["mean_ber"],"mean_intra_hd":result["mean_intra_hd"],"max_intra_hd":result["max_intra_hd"],"mean_inter_hd":result["mean_inter_hd"],"min_inter_hd":result["min_inter_hd"],"collision_gap":result["collision_gap"],"auc":result["auc"],"eer":result["eer"],"balance_error":bit_balance(matrix),"bit_entropy":ent,"mean_abs_corr":mean_abs_corr(matrix),"exact_collision_pairs":stats["exact_collision_pairs"],"exact_collision_rate_per_10k":stats["exact_collision_rate_per_10k"],"inter_q01":stats["inter_q01"],"inter_q05":stats["inter_q05"],"inter_q10":stats["inter_q10"],"intra_q90":stats["intra_q90"],"intra_q95":stats["intra_q95"],"q05_tail_gap":stats["q05_tail_gap"],"q10_tail_gap":stats["q10_tail_gap"],"ultra_near_pairs_le_0.05":stats["collision_pairs_le_0.05"],"ultra_near_rate_le_0.05_per_10k":stats["collision_rate_le_0.05_per_10k"],"near_collision_pairs_le_0.10":stats["collision_pairs_le_0.10"],"near_collision_rate_le_0.10_per_10k":stats["collision_rate_le_0.10_per_10k"],"gate_cap_mean":float(g[:,0].mean()),"gate_dino_mean":float(g[:,1].mean()),"gate_logpolar_mean":float(g[:,2].mean()),"fraction_cap_dominant":float(np.mean(dominant==0)),"fraction_dino_dominant":float(np.mean(dominant==1)),"fraction_logpolar_dominant":float(np.mean(dominant==2)),"relation_scale":float(np.mean(relation_scales))}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); pd.DataFrame([summary]).to_csv(out/"summary.csv",index=False); pd.DataFrame(result["details"],columns=["image_id","attack","hamming","nc"]).to_csv(out/"attack_details.csv",index=False); pd.DataFrame([{"image_id":k,"gate_cap":v[0],"gate_dino":v[1],"gate_logpolar":v[2]} for k,v in sorted(gates.items())]).to_csv(out/"gates.csv",index=False)
    pairs=[]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            d=float(np.mean(clean_bank[a]!=clean_bank[b]))
            if d<=args.near_threshold: pairs.append({"image_a":a,"image_b":b,"hamming":d,"exact_collision":d==0.0,"near_collision":True})
    pd.DataFrame(pairs,columns=["image_a","image_b","hamming","exact_collision","near_collision"]).to_csv(out/"collision_pairs.csv",index=False); print(pd.Series(summary).to_string()); print(f"Results written to {out.resolve()}"); return 0
if __name__=="__main__": raise SystemExit(main())
