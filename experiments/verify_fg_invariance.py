"""
Verify Proposition 1(a): FoolsGold's weight vector is exactly invariant under any
per-client positive rescaling T(u_i) = c_i * u_i, and in particular under NormClip.

This is the empirical check that NormClip -> FoolsGold cannot be a C2 failure:
the upstream transform leaves FoolsGold's discriminative signal bit-identical.

Two tests:
  (A) Synthetic: random updates, random positive scalings (incl. NormClip's
      c_i = min(1, tau/||u_i||)), 1000 trials.
  (B) Real: updates from live FL rounds under model_scaling, passed through the
      shipped apply_d1_transform(..., "norm_clip") code path.

Output: results/fg_invariance_check.json
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform

output_path = os.path.join(base_dir, "results", "fg_invariance_check.json")
TAU = 5.0


def fg_weights_from_stack(client_stack):
    """FoolsGold weights — identical math to fl_core.federated._foolsgold and to
    apply_d1_transform's foolsgold branch."""
    normed = F.normalize(client_stack, dim=1)
    sim = normed @ normed.T
    sim.fill_diagonal_(0.0)
    max_sim, _ = sim.abs().max(dim=1)
    max_sim = max_sim.clamp(0.0, 1.0 - 1e-6)
    max_overall = max_sim.max().clamp(min=1e-8)
    max_sim = max_sim / max_overall
    weights = torch.log((1.0 - max_sim) / (max_sim + 1e-5) + 1e-5)
    weights = weights - weights.min()
    weights = weights.clamp(min=0.0)
    total = weights.sum()
    if total.item() < 1e-8:
        return torch.ones(client_stack.shape[0]) / client_stack.shape[0]
    return weights / total


def flatten(updates):
    keys = list(updates[0].keys())
    return torch.stack([torch.cat([u[k].flatten().float() for k in keys]) for u in updates])


def test_synthetic(n_trials=1000, K=5, d=2000, seed=0):
    """Random updates x random positive scalings."""
    g = torch.Generator().manual_seed(seed)
    diffs_general, diffs_normclip = [], []
    for _ in range(n_trials):
        U = torch.randn(K, d, generator=g)
        # inject a colluding pair so FG's similarity signal is non-degenerate
        U[1] = U[0] + 0.05 * torch.randn(d, generator=g)
        w_raw = fg_weights_from_stack(U)

        # (i) arbitrary positive scaling, spanning 4 orders of magnitude
        c = torch.exp(torch.randn(K, generator=g) * 2.0)
        w_scaled = fg_weights_from_stack(U * c.unsqueeze(1))
        diffs_general.append((w_raw - w_scaled).abs().max().item())

        # (ii) exactly NormClip's scaling
        norms = U.norm(dim=1)
        c_nc = torch.clamp(TAU / norms.clamp(min=1e-8), max=1.0)
        w_nc = fg_weights_from_stack(U * c_nc.unsqueeze(1))
        diffs_normclip.append((w_raw - w_nc).abs().max().item())

    return {
        "n_trials": n_trials,
        "max_abs_weight_diff_arbitrary_positive_scaling": float(np.max(diffs_general)),
        "max_abs_weight_diff_normclip": float(np.max(diffs_normclip)),
        "n_normclip_actually_clipped": int(sum(1 for x in diffs_normclip if True)),
    }


def test_real(seeds=(42, 43, 44), rounds=3, N=10, K=5, f=0.2):
    """Real FL updates through the shipped apply_d1_transform norm_clip branch."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    per_round = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        client_datasets, _, num_classes = get_federated_dataset("cifar10", N, 0.5, seed)
        server = FederatedServer(get_model("cifar_cnn", num_classes), device)
        attack = get_attack("model_scaling")
        n_adv = int(N * f)
        adv_ids = set(range(n_adv))
        clients = [
            FederatedClient(
                i, attack.poison_dataset(client_datasets[i]) if i in adv_ids else client_datasets[i], device
            )
            for i in range(N)
        ]

        for _ in range(rounds):
            selected = np.random.choice(N, K, replace=False)
            updates = []
            for cid in selected:
                u = clients[cid].train(server.global_model, 1, 0.01, 64)
                if cid in adv_ids:
                    u = attack.manipulate_update(u, server.global_model)
                updates.append(u)

            w_raw = fg_weights_from_stack(flatten(updates))
            clipped = apply_d1_transform(updates, "norm_clip", tau=TAU)
            w_clipped = fg_weights_from_stack(flatten(clipped))

            raw_norms = flatten(updates).norm(dim=1)
            per_round.append({
                "seed": int(seed),
                "max_abs_weight_diff": float((w_raw - w_clipped).abs().max().item()),
                "n_clients_clipped": int((raw_norms > TAU).sum().item()),
                "raw_norm_range": [float(raw_norms.min()), float(raw_norms.max())],
                "n_zero_weight_clients": int((w_raw <= 0).sum().item()),
            })
            server.aggregate(updates)
    return per_round


if __name__ == "__main__":
    print("=== Proposition 1(a): FoolsGold weight invariance under positive rescaling ===\n")

    syn = test_synthetic()
    print("(A) Synthetic, 1000 trials:")
    print(f"    max |w(U) - w(cU)|, arbitrary c>0 : {syn['max_abs_weight_diff_arbitrary_positive_scaling']:.3e}")
    print(f"    max |w(U) - w(NormClip(U))|      : {syn['max_abs_weight_diff_normclip']:.3e}")

    print("\n(B) Real FL updates through shipped apply_d1_transform('norm_clip'):")
    real = test_real()
    max_real = max(r["max_abs_weight_diff"] for r in real)
    for r in real:
        print(f"    seed {r['seed']}: diff={r['max_abs_weight_diff']:.3e}  "
              f"clipped {r['n_clients_clipped']}/5 clients  "
              f"norms [{r['raw_norm_range'][0]:.1f}, {r['raw_norm_range'][1]:.1f}]  "
              f"zero-weight clients={r['n_zero_weight_clients']}")
    print(f"    max over all rounds: {max_real:.3e}")

    results = {
        "description": "Proposition 1(a) verification: FG weights invariant under per-client positive rescaling",
        "tau": TAU,
        "synthetic": syn,
        "real_per_round": real,
        "max_abs_weight_diff_real": float(max_real),
        "invariance_holds": bool(max_real < 1e-6
                                 and syn["max_abs_weight_diff_normclip"] < 1e-6
                                 and syn["max_abs_weight_diff_arbitrary_positive_scaling"] < 1e-6),
    }
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nInvariance holds: {results['invariance_holds']}")
    print(f"Saved to {output_path}")
