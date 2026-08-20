"""Pre-freeze input: FLTrust's weight ratio and regime placement.

Needed to assign C3 / regime (A: ordering-preserving, S*r>1; B: attenuation, S*r<<1)
for the fltrust->rank-aggregator pairs in the prospective suite. Measured BEFORE
predictions are frozen. Output: results/fltrust_regime.json
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from torch.utils.data import Subset
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack

N, K, F_ADV, ROUNDS, SEEDS = 10, 5, 0.2, 10, [42, 43, 44]
S_EXTREME = 10.0  # model-scaling factor

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset("cifar10", N, 0.5, seed)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    atk = get_attack("model_scaling"); adv = set(range(int(N*F_ADV)))
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev) for i in range(N)]
    out = []
    for _ in range(ROUNDS):
        sel = np.random.choice(N, K, replace=False)
        ups, mask = [], []
        for cid in sel:
            u = cl[cid].train(srv.global_model, 1, 0.01, 64)
            if cid in adv: u = atk.manipulate_update(u, srv.global_model); mask.append(True)
            else: mask.append(False)
            ups.append(u)
        if not any(mask) or all(mask): srv.aggregate(ups); continue
        keys = list(ups[0].keys())
        g = srv._compute_server_update()
        gf = torch.cat([g[k].flatten().float() for k in keys]); gn = gf.norm().clamp(min=1e-8)
        flats = torch.stack([torch.cat([u[k].flatten().float() for k in keys]) for u in ups])
        w = F.relu(F.cosine_similarity(flats, gf.unsqueeze(0), dim=1))
        tot = w.sum()
        if tot.item() < 1e-8: srv.aggregate(ups); continue
        w = w / tot
        c = (w * K * gn / flats.norm(dim=1).clamp(min=1e-8))   # combined positive scale factor
        ai = [i for i,m in enumerate(mask) if m]; bi = [i for i,m in enumerate(mask) if not m]
        pos = c[c > 1e-12]
        out.append({
            "n_adv": len(ai),
            "rho": float((pos.max()/pos.min()).item()) if len(pos) >= 2 else None,
            "c_adv_mean": float(c[ai].mean()), "c_benign_max": float(c[bi].max()),
            "r": float((c[ai].mean()/c[bi].max().clamp(min=1e-12)).item()),
            "n_zeroed": int((c <= 1e-12).sum()),
            "n_adv_zeroed": int((c[ai] <= 1e-12).sum()),
        })
        srv.aggregate(ups)
    return out

if __name__ == "__main__":
    allm = []
    for s in SEEDS:
        m = run(s); allm.extend(m); print(f"  seed {s}: {len(m)} adversary-present rounds", flush=True)
    rho = [x["rho"] for x in allm if x["rho"] is not None]
    r = [x["r"] for x in allm]
    print(f"\n=== FLTrust regime placement ({len(allm)} rounds) ===")
    print(f"  rho (positive weights): mean={np.mean(rho):.3g} max={np.max(rho):.3g}")
    print(f"  r = c_adv/max c_benign: mean={np.mean(r):.3g} max={np.max(r):.3g}")
    print(f"  S*r with S={S_EXTREME:.0f}:    mean={S_EXTREME*np.mean(r):.3g}")
    print(f"  rounds with ALL adversaries zeroed: {sum(x['n_adv_zeroed']==x['n_adv'] for x in allm)}/{len(allm)}")
    print(f"  zeroed clients per round: mean={np.mean([x['n_zeroed'] for x in allm]):.2f}")
    reg = "B (attenuation)" if S_EXTREME*np.mean(r) < 0.5 else ("A (ordering)" if S_EXTREME*np.mean(r) > 1 else "AMBIGUOUS (S*r~1)")
    print(f"  => regime {reg}")
    json.dump({"description":"FLTrust weight ratio / regime, measured before freezing predictions",
               "rho_mean":float(np.mean(rho)),"rho_max":float(np.max(rho)),
               "r_mean":float(np.mean(r)),"S_times_r_mean":float(S_EXTREME*np.mean(r)),
               "regime":reg,"n_rounds":len(allm),"per_round":allm},
              open(os.path.join(base,"results","fltrust_regime.json"),"w"), indent=2)
    print("\nSaved to results/fltrust_regime.json")
