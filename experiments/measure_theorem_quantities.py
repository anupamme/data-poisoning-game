#!/usr/bin/env python3
"""Measure theorem quantities: r* (coordinate-wise ratio), R_B, delta, Delta_sep.
Quick 3-seed measurement on MPS."""
import sys, json
sys.path.insert(0, '/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game')

import torch
import torch.nn.functional as F
import numpy as np
from fl_core.data_loader import get_federated_dataset
from fl_core.models import CifarCNN
from fl_core.federated import FederatedClient, FederatedServer
from attacks.attack_strategies import get_attack

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def weiszfeld(points, max_iter=50, tol=1e-6):
    """Compute geometric median via Weiszfeld algorithm."""
    mu = points.mean(dim=0)
    for _ in range(max_iter):
        dists = torch.norm(points - mu.unsqueeze(0), dim=1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / dists
        mu_new = (weights * points).sum(dim=0) / weights.sum()
        if torch.norm(mu_new - mu) < tol:
            break
        mu = mu_new
    return mu

def measure_round(updates, adv_mask):
    """Measure r*, R_B, delta, Delta_sep for one round's updates."""
    keys = list(updates[0].keys())
    flats = [torch.cat([u[k].flatten().float().cpu() for k in keys]) for u in updates]
    stacked = torch.stack(flats)

    benign_idx = [i for i, m in enumerate(adv_mask) if not m]
    adv_idx = [i for i, m in enumerate(adv_mask) if m]

    if not adv_idx or not benign_idx:
        return None

    benign_updates = stacked[benign_idx]
    adv_updates = stacked[adv_idx]

    # R_B: max distance from benign-only geometric median to any benign point
    mu_B = weiszfeld(benign_updates)
    benign_residuals = torch.norm(benign_updates - mu_B.unsqueeze(0), dim=1)
    R_B = benign_residuals.max().item()

    # mu': geometric median of ALL updates
    mu_all = weiszfeld(stacked)

    # delta: displacement
    delta = torch.norm(mu_all - mu_B).item()

    # Delta_sep: min distance between any adversarial and any benign update
    dists = torch.cdist(adv_updates.unsqueeze(0), benign_updates.unsqueeze(0)).squeeze(0)
    Delta_sep = dists.min().item()

    # r*: minimum coordinate-wise |u_a,k| / |u_b,k| over coordinates where |u_a,k| > |u_b,k|
    # Only over coordinates where adversarial is more extreme (suppression-relevant)
    r_stars = []
    for a_i in range(len(adv_idx)):
        u_a = adv_updates[a_i].abs()
        for b_i in range(len(benign_idx)):
            u_b = benign_updates[b_i].abs()
            # Only look at coords where adversarial IS more extreme
            mask = u_a > u_b
            if mask.sum() == 0:
                continue
            # Avoid division by zero
            u_b_safe = u_b[mask].clamp(min=1e-10)
            ratios = u_a[mask] / u_b_safe
            # r* is the minimum ratio (worst-case coordinate)
            # But we only care about the top-k most extreme coords (those that trigger trimming)
            # Use percentile 1 to avoid noise from near-zero coordinates
            r_star = ratios.quantile(0.01).item()
            r_stars.append(r_star)

    r_star_min = min(r_stars) if r_stars else float('inf')

    # Verify condition: Delta_sep > 2(R_B + delta)?
    condition_met = Delta_sep > 2 * (R_B + delta)

    return {
        'R_B': R_B,
        'delta': delta,
        'delta_over_R_B': delta / max(R_B, 1e-10),
        'Delta_sep': Delta_sep,
        'r_star_p01': r_star_min,
        'condition_ratio': Delta_sep / (2 * (R_B + delta)) if (R_B + delta) > 0 else float('inf'),
        'condition_met': condition_met,
    }

def run_one_seed(seed, attack_name='model_scaling'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, K, f, rounds = 10, 5, 0.2, 10
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

    measurements = []
    for r in range(rounds):
        selected = np.random.choice(N, K, replace=False)
        updates = []
        adv_mask = []
        for idx in selected:
            update = clients[idx].train(server.global_model, epochs=1, lr=0.01, batch_size=64)
            if idx in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
                adv_mask.append(True)
            else:
                adv_mask.append(False)
            updates.append(update)

        if any(adv_mask):
            m = measure_round(updates, adv_mask)
            if m is not None:
                measurements.append(m)

        server.aggregate(updates)

    return measurements

if __name__ == '__main__':
    all_measurements = []
    for seed in range(3):
        print(f"Seed {seed}...", end=' ', flush=True)
        ms = run_one_seed(seed, 'model_scaling')
        all_measurements.extend(ms)
        if ms:
            avg_delta_rb = np.mean([m['delta_over_R_B'] for m in ms])
            avg_rstar = np.mean([m['r_star_p01'] for m in ms])
            cond_frac = np.mean([m['condition_met'] for m in ms])
            print(f"delta/R_B={avg_delta_rb:.3f}, r*={avg_rstar:.2f}, cond_met={cond_frac:.0%}")
        else:
            print("no adversarial rounds")

    print(f"\n=== SUMMARY ({len(all_measurements)} rounds) ===")
    deltas = [m['delta_over_R_B'] for m in all_measurements]
    rstars = [m['r_star_p01'] for m in all_measurements]
    cond_ratios = [m['condition_ratio'] for m in all_measurements]

    print(f"delta/R_B: mean={np.mean(deltas):.4f}, max={np.max(deltas):.4f}")
    print(f"r* (p01):  mean={np.mean(rstars):.2f}, min={np.min(rstars):.2f}, max={np.max(rstars):.2f}")
    print(f"Delta_sep / 2(R_B+delta): mean={np.mean(cond_ratios):.2f}, min={np.min(cond_ratios):.2f}")
    print(f"Condition met: {np.mean([m['condition_met'] for m in all_measurements]):.0%}")

    results = {
        'delta_over_R_B_mean': float(np.mean(deltas)),
        'delta_over_R_B_max': float(np.max(deltas)),
        'r_star_p01_mean': float(np.mean(rstars)),
        'r_star_p01_min': float(np.min(rstars)),
        'r_star_p01_max': float(np.max(rstars)),
        'condition_ratio_mean': float(np.mean(cond_ratios)),
        'condition_ratio_min': float(np.min(cond_ratios)),
        'condition_met_frac': float(np.mean([m['condition_met'] for m in all_measurements])),
        'n_seeds': 3,
        'n_rounds_per_seed': 10,
    }

    with open('/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game/results/theorem_quantities.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/theorem_quantities.json")
