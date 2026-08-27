#!/usr/bin/env python3
"""Measure FoolsGold's max weight ratio (rho) across rounds in FG->RFA setting.
Quick 5-seed measurement — only computes FG weights on real updates, doesn't need full experiment."""
import sys, json
sys.path.insert(0, '/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game')

import torch
import torch.nn.functional as F
import numpy as np
from fl_core.data_loader import get_federated_dataset
from fl_core.models import CifarCNN
from fl_core.federated import FederatedClient, FederatedServer
from attacks.attack_strategies import get_attack

def compute_fg_weights(updates):
    """Compute FoolsGold weights and return weight ratio rho."""
    keys = list(updates[0].keys())
    flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
    client_stack = torch.stack(flats)
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
        return None
    weights = weights / total
    positive_weights = weights[weights > 1e-10]
    if len(positive_weights) < 2:
        return None
    w_max = positive_weights.max().item()
    w_min = positive_weights.min().item()
    return w_max / w_min

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def run_one_seed(seed, attack_name='model_scaling'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, K, f, rounds = 10, 5, 0.2, 20
    n_adv = int(N * f)

    client_datasets, _, _ = get_federated_dataset('cifar10', N, alpha=0.5, seed=seed)
    model = CifarCNN()
    server = FederatedServer(model, device=DEVICE)

    clients = []
    for i in range(N):
        clients.append(FederatedClient(i, client_datasets[i], device=DEVICE))

    attack = get_attack(attack_name)
    adv_ids = set(range(n_adv))
    for i in adv_ids:
        clients[i] = FederatedClient(i, attack.poison_dataset(client_datasets[i]), device=DEVICE)

    rhos = []
    for r in range(rounds):
        selected = np.random.choice(N, K, replace=False)
        updates = []
        for idx in selected:
            update = clients[idx].train(server.global_model, epochs=1, lr=0.01, batch_size=64)
            if idx in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        rho = compute_fg_weights(updates)
        if rho is not None:
            rhos.append(rho)

        # Apply FG weights then RFA aggregate (simplified: just use RFA for global model update)
        server.aggregate(updates)

    return rhos

if __name__ == '__main__':
    all_rhos = []
    for seed in range(5):
        print(f"Seed {seed}...", end=' ', flush=True)
        rhos = run_one_seed(seed, 'model_scaling')
        all_rhos.extend(rhos)
        print(f"max_rho={max(rhos):.3f}, mean={np.mean(rhos):.3f}")

    print(f"\n=== SUMMARY ===")
    print(f"Total rounds measured: {len(all_rhos)}")
    print(f"Max rho across all seeds/rounds: {max(all_rhos):.4f}")
    print(f"Mean rho: {np.mean(all_rhos):.4f}")
    print(f"95th percentile rho: {np.percentile(all_rhos, 95):.4f}")
    print(f"99th percentile rho: {np.percentile(all_rhos, 99):.4f}")

    results = {
        'max_rho': float(max(all_rhos)),
        'mean_rho': float(np.mean(all_rhos)),
        'p95_rho': float(np.percentile(all_rhos, 95)),
        'p99_rho': float(np.percentile(all_rhos, 99)),
        'n_seeds': 5,
        'n_rounds': 20,
    }

    with open('/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game/results/fg_weight_ratio.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/fg_weight_ratio.json")
