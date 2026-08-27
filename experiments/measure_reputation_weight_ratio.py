"""
Measure the Reputation defense's weight ratio rho_rep = max_i w_i / min_i w_i.

Theorem 1 Part 1 (bounded reweighting preserves coordinate ordering) is applied in
the paper to Reputation -> CoordMedian, one of the three C1^C2^C3 PASS pairs, but
rho has only ever been measured for FoolsGold (results/fg_weight_ratio.json).
This closes that gap.

Reputation weights (fl_core/federated.py:_reputation):
    consensus = coordinate-wise median of the K participating updates
    d_i       = ||u_i - consensus||_2
    sigma     = median_i(d_i)                      (adaptive bandwidth)
    w_i       = exp(-d_i / sigma), normalized to sum 1
Weights are strictly positive, so there is no zero-vector case (unlike FoolsGold).

Output: results/reputation_weight_ratio.json
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack

output_path = os.path.join(base_dir, "results", "reputation_weight_ratio.json")

SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 20
N, K, F_ADV = 10, 5, 0.2
ATTACK = "model_scaling"


def reputation_weights(client_stack):
    """Identical math to fl_core.federated._reputation / apply_d1_transform's
    reputation branch."""
    consensus = client_stack.median(dim=0).values
    dists = (client_stack - consensus.unsqueeze(0)).norm(dim=1)
    scale = float(dists.median().clamp(min=1e-6).item())
    w = torch.exp(-dists / scale)
    w = w / w.sum().clamp(min=1e-8)
    return w, dists


def flatten(updates):
    keys = list(updates[0].keys())
    return torch.stack([torch.cat([u[k].flatten().float() for k in keys]) for u in updates])


def run_seed(seed):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    client_datasets, _, num_classes = get_federated_dataset("cifar10", N, 0.5, seed)
    server = FederatedServer(get_model("cifar_cnn", num_classes), device)
    attack = get_attack(ATTACK)
    n_adv = int(N * F_ADV)
    adv_ids = set(range(n_adv))
    clients = [
        FederatedClient(
            i, attack.poison_dataset(client_datasets[i]) if i in adv_ids else client_datasets[i], device
        )
        for i in range(N)
    ]

    per_round = []
    for _ in range(ROUNDS):
        selected = np.random.choice(N, K, replace=False)
        updates, adv_mask = [], []
        for cid in selected:
            u = clients[cid].train(server.global_model, 1, 0.01, 64)
            if cid in adv_ids:
                u = attack.manipulate_update(u, server.global_model)
                adv_mask.append(True)
            else:
                adv_mask.append(False)
            updates.append(u)

        stack = flatten(updates)
        w, dists = reputation_weights(stack)
        adv_idx = [i for i, m in enumerate(adv_mask) if m]
        ben_idx = [i for i, m in enumerate(adv_mask) if not m]

        rec = {
            "n_adv_in_round": int(len(adv_idx)),
            "rho": float((w.max() / w.min().clamp(min=1e-30)).item()),
            "w_min": float(w.min().item()),
            "w_max": float(w.max().item()),
        }
        if adv_idx and ben_idx:
            rec["w_adv_mean"] = float(w[adv_idx].mean().item())
            rec["w_benign_mean"] = float(w[ben_idx].mean().item())
            # ratio the theorem actually needs: max_b w_b / w_a
            rec["max_wb_over_wa"] = float((w[ben_idx].max() / w[adv_idx].min().clamp(min=1e-30)).item())
        per_round.append(rec)

        server.aggregate(updates)
    return per_round


if __name__ == "__main__":
    print(f"=== Reputation weight ratio (N={N}, K={K}, f={F_ADV}, {ATTACK}) ===")
    print(f"  {len(SEEDS)} seeds x {ROUNDS} rounds\n")

    all_rounds = []
    for seed in SEEDS:
        rs = run_seed(seed)
        all_rounds.extend(rs)
        rhos = [r["rho"] for r in rs]
        print(f"  seed {seed}: rho mean={np.mean(rhos):.2f} max={np.max(rhos):.2f}")

    rhos = [r["rho"] for r in all_rounds]
    ratios = [r["max_wb_over_wa"] for r in all_rounds if "max_wb_over_wa" in r]
    n_adv_counts = [r["n_adv_in_round"] for r in all_rounds]

    # Empirical adversary-count distribution (should match hypergeometric:
    # P(0)=P(2)=56/252=0.2222, P(1)=140/252=0.5556 for N=10, 2 adversaries, K=5)
    counts = {c: n_adv_counts.count(c) / len(n_adv_counts) for c in sorted(set(n_adv_counts))}

    print(f"\n=== SUMMARY ({len(all_rounds)} rounds) ===")
    print(f"rho = max w / min w      : mean={np.mean(rhos):.2f} max={np.max(rhos):.2f} "
          f"p95={np.percentile(rhos,95):.2f}")
    print(f"max_b w_b / w_a          : mean={np.mean(ratios):.2f} max={np.max(ratios):.2f}")
    print(f"adversary count per round: {counts}")
    print("  (hypergeometric expectation: {0: 0.222, 1: 0.556, 2: 0.222})")

    results = {
        "description": "Reputation weight ratio for Theorem 1 Part 1 applied to Reputation->CoordMedian",
        "config": {"N": N, "K": K, "f": F_ADV, "attack": ATTACK,
                   "seeds": SEEDS, "rounds_per_seed": ROUNDS},
        "rho_mean": float(np.mean(rhos)),
        "rho_max": float(np.max(rhos)),
        "rho_p95": float(np.percentile(rhos, 95)),
        "max_wb_over_wa_mean": float(np.mean(ratios)) if ratios else None,
        "max_wb_over_wa_max": float(np.max(ratios)) if ratios else None,
        "adv_count_empirical_distribution": counts,
        "adv_count_hypergeometric": {"0": 56 / 252, "1": 140 / 252, "2": 56 / 252},
        "n_rounds": len(all_rounds),
        "per_round": all_rounds,
    }
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {output_path}")
