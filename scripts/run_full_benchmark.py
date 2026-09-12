from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from zero_watermarking.baselines import method_registry
from zero_watermarking.deep_baselines import build_deep_registry
from zero_watermarking.datasets import load_manifest, load_image, validate_manifest
from zero_watermarking.protocol import DEFAULT_ATTACK_GRID, seed_everything
from zero_watermarking.attacks import ATTACKS
from zero_watermarking.metrics import hamming, nc, bit_balance, bit_entropy, mean_abs_corr, roc_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='data/manifests/medical_manifest.csv')
    ap.add_argument('--out', default='experiments/results')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--bits', type=int, default=256)
    args = ap.parse_args()
    seed_everything(42)
    frame = validate_manifest(load_manifest(args.manifest))
    frame = frame[frame.exists].head(args.limit)
    images = {r.image_id: load_image(r.path, args.size) for r in load_manifest(args.manifest) if r.image_id in set(frame.image_id)}
    registry = method_registry(args.bits)
    registry.update({k: v for k, v in build_deep_registry(args.bits).items() if v is not None})
    attacks = [(s.name, s.parameter, v) for s in DEFAULT_ATTACK_GRID for v in s.values]
    rows=[]
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, fn in registry.items():
        clean={i:fn(x) for i,x in images.items()}
        bank=[]
        for image_id,x in images.items():
            for attack, param, value in attacks:
                attacked=ATTACKS[attack](x, **{param:value})
                hb=fn(attacked); bank.append({'method':name,'image_id':image_id,'attack':attack,'strength':value,'hamming':hamming(clean[image_id],hb),'nc':nc(clean[image_id],hb)})
        detail=pd.DataFrame(bank); detail.to_csv(out/f'{name.replace("/","_")}_detail.csv',index=False)
        bits=pd.DataFrame([clean[i] for i in sorted(clean)]).to_numpy(np.uint8) if False else __import__('numpy').stack([clean[i] for i in sorted(clean)])
        genuine=(1-detail.hamming).to_numpy(); impostor=[]
        ids=sorted(clean)
        for p,i in enumerate(ids):
            for j in ids[p+1:]: impostor.append(1-hamming(clean[i],clean[j]))
        roc=roc_stats(genuine, np.asarray(impostor))
        ent,_=bit_entropy(bits)
        rows.append({'method':name,'mean_nc':detail.nc.mean(),'mean_ber':detail.hamming.mean(),'max_intra_hd':detail.hamming.max(),'min_inter_score':np.min(impostor),'auc':roc['auc'],'eer':roc['eer'],'balance_error':bit_balance(bits),'bit_entropy':ent,'mean_abs_corr':mean_abs_corr(bits)})
    pd.DataFrame(rows).to_csv(out/'journal_benchmark_summary.csv',index=False)
    print(pd.DataFrame(rows).sort_values(['auc','mean_nc'],ascending=False).to_string(index=False))

if __name__ == '__main__': main()
