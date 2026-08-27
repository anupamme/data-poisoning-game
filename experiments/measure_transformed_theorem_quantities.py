#!/usr/bin/env python3
"""Measure theorem quantities on TRANSFORMED points u'_i = w_i * u_i.
This validates the corrected RFA theorem (Part 2) which reasons about
the points that RFA actually receives after FoolsGold reweighting.

Measures: R'_B, delta', Delta'_sep on transformed updates."""
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


def compute_fg_weights(updates_flat):
    """Compute FoolsGold weights from flattened updates."""
    normed = F.normalize(updates_flat, dim=1)
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
        return torch.ones(updates_flat.shape[0]) / updates_flat.shape[0]
    return weights / total


def measure_round_transformed(updates, adv_mask):
    """Measure R'_B, delta', Delta'_sep on transformed points u'_i = w_i * u_i."""
    keys = list(updates[0].keys())
    flats = [torch.cat([u[k].flatten().float().cpu() for k in keys]) for u in updates]
    stacked = torch.stack(flats)

    benign_idx = [i for i, m in enumerate(adv_mask) if not m]
    adv_idx = [i for i, m in enumerate(adv_mask) if m]

    if not adv_idx or not benign_idx:
        return None

    # Compute FG weights
    fg_weights = compute_fg_weights(stacked)

    # Transform: u'_i = w_i * u_i
    transformed = stacked * fg_weights.unsqueeze(1)

    benign_transformed = transformed[benign_idx]
    adv_transformed = transformed[adv_idx]

    # R'_B: max distance from benign-only geometric median to any benign transformed point
    mu_B_prime = weiszfeld(benign_transformed)
    benign_residuals = torch.norm(benign_transformed - mu_B_prime.unsqueeze(0), dim=1)
    R_B_prime = benign_residuals.max().item()

    # mu': geometric median of ALL transformed updates
    mu_all_prime = weiszfeld(transformed)

    # delta': displacement
    delta_prime = torch.norm(mu_all_prime - mu_B_prime).item()

    # Delta'_sep: min distance between any adversarial and any benign transformed update
    dists = torch.cdist(adv_transformed.unsqueeze(0), benign_transformed.unsqueeze(0)).squeeze(0)
    Delta_sep_prime = dists.min().item()

    # Condition check: Delta'_sep > 2(R'_B + delta')?
    condition_met = Delta_sep_prime > 2 * (R_B_prime + delta_prime)

    # --- Lemma 1 (zero-weight annihilation) ---
    # At the Weiszfeld fixed point, mu' = sum_i lambda_i u'_i with
    # lambda_i = ||u'_i - mu'||^{-1} / sum_j ||u'_j - mu'||^{-1}.
    # When w_a = 0 the adversarial term is lambda_a * 0 and drops out identically,
    # so mu' = (1 - sum_a lambda_a) * (convex combination of benign updates):
    # the aggregate is a CONTRACTION of a benign-only average by factor
    # contraction = 1 - sum_a lambda_a. Measure that factor.
    resid_all = torch.norm(transformed - mu_all_prime.unsqueeze(0), dim=1).clamp(min=1e-8)
    lam = (1.0 / resid_all)
    lam = lam / lam.sum()
    lambda_adv_total = lam[adv_idx].sum().item()
    contraction = 1.0 - lambda_adv_total

    # Also report weight ratio rho
    positive_w = fg_weights[fg_weights > 1e-10]
    rho = (positive_w.max() / positive_w.min()).item() if len(positive_w) >= 2 else float('inf')

    # Report per-group weights
    w_benign = fg_weights[benign_idx].mean().item()
    w_adv = fg_weights[adv_idx].mean().item()

    return {
        'R_B_prime': R_B_prime,
        'delta_prime': delta_prime,
        'delta_over_R_B_prime': delta_prime / max(R_B_prime, 1e-10),
        'Delta_sep_prime': Delta_sep_prime,
        'condition_ratio_prime': Delta_sep_prime / (2 * (R_B_prime + delta_prime)) if (R_B_prime + delta_prime) > 0 else float('inf'),
        'condition_met_prime': condition_met,
        'rho': rho,
        'w_benign_mean': w_benign,
        'w_adv_mean': w_adv,
        'w_ratio_benign_over_adv': w_benign / max(w_adv, 1e-10),
        # Lemma 1 quantities
        'w_adv_is_zero': bool(fg_weights[adv_idx].max().item() <= 0.0),
        'lambda_adv_total': lambda_adv_total,
        'contraction': contraction,
        'n_adv_in_round': len(adv_idx),
        'n_zero_weight_clients': int((fg_weights <= 0.0).sum().item()),
    }


def run_one_seed(seed, attack_name='model_scaling', alpha=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, K, f, rounds = 10, 5, 0.2, 10
    n_adv = int(N * f)

    client_datasets, _, _ = get_federated_dataset('cifar10', N, alpha=alpha, seed=seed)
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
            m = measure_round_transformed(updates, adv_mask)
            if m is not None:
                measurements.append(m)

        server.aggregate(updates)

    return measurements


if __name__ == '__main__':
    import os
    ALPHA = float(os.environ.get('SWEEP_ALPHA', '0.5'))
    OUT = os.environ.get('SWEEP_OUT', 'results/theorem_quantities_transformed.json')
    all_measurements = []
    for seed in range(3):
        print(f"Seed {seed}...", end=' ', flush=True)
        ms = run_one_seed(seed, 'model_scaling', alpha=ALPHA)
        all_measurements.extend(ms)
        if ms:
            avg_cond = np.mean([m['condition_ratio_prime'] for m in ms])
            avg_rho = np.mean([m['rho'] for m in ms])
            cond_frac = np.mean([m['condition_met_prime'] for m in ms])
            print(f"cond_ratio={avg_cond:.2f}, rho={avg_rho:.2f}, cond_met={cond_frac:.0%}")
        else:
            print("no adversarial rounds")

    print(f"\n=== TRANSFORMED-POINT SUMMARY ({len(all_measurements)} rounds) ===")
    deltas = [m['delta_over_R_B_prime'] for m in all_measurements]
    cond_ratios = [m['condition_ratio_prime'] for m in all_measurements]
    rhos = [m['rho'] for m in all_measurements]
    w_ratios = [m['w_ratio_benign_over_adv'] for m in all_measurements]

    print(f"delta'/R'_B: mean={np.mean(deltas):.4f}, max={np.max(deltas):.4f}")
    print(f"Delta'_sep / 2(R'_B+delta'): mean={np.mean(cond_ratios):.2f}, min={np.min(cond_ratios):.2f}")
    print(f"Condition met: {np.mean([m['condition_met_prime'] for m in all_measurements]):.0%}")
    print(f"Weight ratio rho: mean={np.mean(rhos):.2f}, max={np.max(rhos):.2f}")
    print(f"w_benign/w_adv: mean={np.mean(w_ratios):.2f}")

    # --- Lemma 1: contraction factor in w_a = 0 rounds ---
    zero_rounds = [m for m in all_measurements if m['w_adv_is_zero']]
    pos_rounds = [m for m in all_measurements if not m['w_adv_is_zero']]
    print(f"\n=== LEMMA 1 (zero-weight annihilation) ===")
    print(f"Rounds with w_a = 0: {len(zero_rounds)}/{len(all_measurements)} "
          f"({len(zero_rounds)/len(all_measurements):.0%})")
    if zero_rounds:
        contr = [m['contraction'] for m in zero_rounds]
        print(f"  contraction 1 - lambda_a: mean={np.mean(contr):.3f}, "
              f"min={np.min(contr):.3f}, max={np.max(contr):.3f}")
        print(f"  (aggregate is this fraction of a benign-only weighted average;")
        print(f"   the shortfall is the utility cost, NOT an attack pathway)")
    zw = [m['n_zero_weight_clients'] for m in all_measurements]
    print(f"Zero-weight clients per round: mean={np.mean(zw):.2f}, min={np.min(zw)}, max={np.max(zw)}")
    nadv = [m['n_adv_in_round'] for m in all_measurements]
    print(f"Adversaries per round (adv-present rounds only): {sorted(set(nadv))}, mean={np.mean(nadv):.2f}")

    results = {
        'delta_over_R_B_prime_mean': float(np.mean(deltas)),
        'delta_over_R_B_prime_max': float(np.max(deltas)),
        'condition_ratio_prime_mean': float(np.mean(cond_ratios)),
        'condition_ratio_prime_min': float(np.min(cond_ratios)),
        'condition_met_prime_frac': float(np.mean([m['condition_met_prime'] for m in all_measurements])),
        'rho_mean': float(np.mean(rhos)),
        'rho_max': float(np.max(rhos)),
        'w_ratio_benign_over_adv_mean': float(np.mean(w_ratios)),
        'n_seeds': 3,
        'n_rounds_per_seed': 10,
        # Lemma 1
        'n_rounds_total': len(all_measurements),
        'n_rounds_w_adv_zero': len(zero_rounds),
        'frac_rounds_w_adv_zero': len(zero_rounds) / len(all_measurements),
        'contraction_mean_zero_rounds': float(np.mean([m['contraction'] for m in zero_rounds])) if zero_rounds else None,
        'contraction_min_zero_rounds': float(np.min([m['contraction'] for m in zero_rounds])) if zero_rounds else None,
        'contraction_max_zero_rounds': float(np.max([m['contraction'] for m in zero_rounds])) if zero_rounds else None,
        'condition_ratio_prime_min_positive_w': float(np.min([m['condition_ratio_prime'] for m in pos_rounds])) if pos_rounds else None,
        'zero_weight_clients_per_round_mean': float(np.mean(zw)),
        'zero_weight_clients_per_round_min': int(np.min(zw)),
        # Corollary (bounded aggregate adversarial mass): keep per-round records so the
        # measured Lambda_a can be checked against the derived bound n_a/(n_b*(2m-1)+n_a).
        'per_round': all_measurements,
    }

    with open(os.path.join('/Users/mediratta/code/paper_writing/AI-Researcher/code/data_poisoning_game', OUT), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/theorem_quantities_transformed.json")
