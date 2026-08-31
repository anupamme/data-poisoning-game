#!/usr/bin/env python3
"""Measure the quantities needed by the two *checkable* forms of Theorem 2(2).

Theorem 2(2) as stated is  Delta'_sep > 2(R'_B + delta').  The quantity delta'
= ||mu' - mu'_B|| cannot be evaluated without actually running the composition,
which is what makes the condition unscreenable.  Two reductions are measured here.

STEP A -- eliminate delta'.  mu' minimises F(x) = sum_i ||x - u'_i||.  Comparing
F(mu') <= F(mu'_B), the benign terms lose at least n_b*delta' - 2*S'_B and the
adversarial terms gain at most n_a*delta', where S'_B = sum_b ||u'_b - mu'_B||.
Hence

    (n_b - n_a) delta'  <=  2 S'_B        i.e.   delta' <= C_A * Rbar'_B,
    C_A := 2 n_b / (n_b - n_a),                  Rbar'_B := S'_B / n_b,

so honest majority n_b > n_a is exactly what is required, and

    Delta'_sep > 2 R'_B + 2 C_A Rbar'_B          (STEP A, mean-radius form)
    Delta'_sep > 2 (1 + C_A) R'_B                (STEP A, max-radius form, weaker)

is sufficient.  Both are checkable without evaluating delta'.

STEP B -- reduce further to the *untransformed* round geometry and the weight
ratio rho, so the condition is checkable from the C1 baseline runs alone:

    Delta'_sep >= w_min [ Delta_sep - (rho-1) N_A ]
    R'_B       <= 2 w_min [ rho R_B + (rho-1) ||mu_B|| ]

(the second because every benign transformed point, and hence their geometric
median, lies in a ball of radius w_min*r about w_min*mu_B).  w_min cancels, and
the max-radius Step A form becomes a linear inequality in rho with critical value

    rho* = 1 + (Delta_sep - B Rb) / (N_A + B (Rb + ||mu_B||)),   B := 4(1 + C_A),

defined only when Delta_sep > B*R_B.  Step B is expected to be conservative; the
verdict is reported either way and a vacuous result is reported as vacuous.

Writes results/theorem_margins.json.  Does NOT touch the frozen artifacts
results/theorem_quantities_transformed.json or results/theorem_quantities_alpha0.1.json;
the shared per-round fields are recomputed by importing the same code path, so
they double as a reproduction check on those files.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)

import json
import numpy as np
import torch

# Reuse the exact primitives the frozen artifacts were produced with, so the
# shared fields are bit-comparable rather than a reimplementation.
import measure_transformed_theorem_quantities as mt
from fl_core.data_loader import get_federated_dataset
from fl_core.models import CifarCNN
from fl_core.federated import FederatedClient, FederatedServer
from attacks.attack_strategies import get_attack

DEVICE = mt.DEVICE


def _flatten(updates):
    keys = list(updates[0].keys())
    return torch.stack([torch.cat([u[k].flatten().float().cpu() for k in keys])
                        for u in updates])


def measure_round_margins(updates, adv_mask):
    """Extend mt.measure_round_transformed with the Step A / Step B quantities."""
    base = mt.measure_round_transformed(updates, adv_mask)
    if base is None:
        return None

    stacked = _flatten(updates)
    benign_idx = [i for i, m in enumerate(adv_mask) if not m]
    adv_idx = [i for i, m in enumerate(adv_mask) if m]
    n_b, n_a = len(benign_idx), len(adv_idx)

    fg_weights = mt.compute_fg_weights(stacked)
    transformed = stacked * fg_weights.unsqueeze(1)

    # ---- transformed benign spread: the mean residual Step A needs ----
    bt = transformed[benign_idx]
    mu_B_prime = mt.weiszfeld(bt)
    resid_prime = torch.norm(bt - mu_B_prime.unsqueeze(0), dim=1)
    S_B_prime = resid_prime.sum().item()
    Rbar_B_prime = resid_prime.mean().item()
    R_B_prime = base['R_B_prime']
    Delta_sep_prime = base['Delta_sep_prime']
    delta_prime = base['delta_prime']

    # ---- untransformed round geometry: what Step B is allowed to use ----
    braw, araw = stacked[benign_idx], stacked[adv_idx]
    mu_B = mt.weiszfeld(braw)
    resid_raw = torch.norm(braw - mu_B.unsqueeze(0), dim=1)
    R_B = resid_raw.max().item()
    Rbar_B = resid_raw.mean().item()
    Delta_sep = torch.cdist(araw.unsqueeze(0), braw.unsqueeze(0)).squeeze(0).min().item()
    mu_B_norm = torch.norm(mu_B).item()
    N_A = torch.norm(araw, dim=1).max().item()
    N_B = torch.norm(braw, dim=1).max().item()

    rho = base['rho']

    # ---- Step A ----
    if n_b > n_a:
        C_A = 2.0 * n_b / (n_b - n_a)
        delta_bound_mean = C_A * Rbar_B_prime
        delta_bound_max = C_A * R_B_prime
        need_A_mean = 2.0 * R_B_prime + 2.0 * delta_bound_mean
        need_A_max = 2.0 * (1.0 + C_A) * R_B_prime
    else:
        C_A = float('inf')
        delta_bound_mean = delta_bound_max = float('inf')
        need_A_mean = need_A_max = float('inf')

    # ---- Step B ----
    if np.isfinite(C_A):
        Bc = 4.0 * (1.0 + C_A)
        denom = N_A + Bc * (R_B + mu_B_norm)
        rho_star = 1.0 + (Delta_sep - Bc * R_B) / denom if denom > 0 else float('-inf')
    else:
        Bc = float('inf')
        rho_star = float('-inf')

    out = dict(base)
    out.update({
        'n_benign_in_round': n_b,
        # transformed spread
        'S_B_prime': S_B_prime,
        'Rbar_B_prime': Rbar_B_prime,
        'Rbar_over_R_B_prime': Rbar_B_prime / max(R_B_prime, 1e-12),
        # untransformed geometry
        'R_B': R_B,
        'Rbar_B': Rbar_B,
        'Delta_sep': Delta_sep,
        'mu_B_norm': mu_B_norm,
        'N_A': N_A,
        'N_B': N_B,
        'Delta_sep_over_R_B': Delta_sep / max(R_B, 1e-12),
        # Step A
        'C_A': C_A,
        'delta_bound_mean_radius': delta_bound_mean,
        'delta_bound_max_radius': delta_bound_max,
        'delta_bound_mean_holds': bool(delta_prime <= delta_bound_mean),
        'delta_bound_slack_mean': delta_bound_mean / max(delta_prime, 1e-12),
        'stepA_need_mean_radius': need_A_mean,
        'stepA_need_max_radius': need_A_max,
        'stepA_met_mean_radius': bool(Delta_sep_prime > need_A_mean),
        'stepA_met_max_radius': bool(Delta_sep_prime > need_A_max),
        # Step B
        'stepB_B_const': Bc,
        'stepB_rho_star': rho_star,
        'stepB_met': bool(np.isfinite(rho_star) and rho < rho_star),
        'stepB_vacuous_round': bool(not np.isfinite(rho_star) or rho_star <= 1.0),
    })
    return out


def run_one_seed(seed, attack_name='model_scaling', alpha=0.5):
    """Mirrors mt.run_one_seed exactly, with the extended per-round measurement."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, K, f, rounds = 10, 5, 0.2, 10
    n_adv = int(N * f)

    client_datasets, _, _ = get_federated_dataset('cifar10', N, alpha=alpha, seed=seed)
    server = FederatedServer(CifarCNN(), device=DEVICE)

    clients = [FederatedClient(i, client_datasets[i], device=DEVICE) for i in range(N)]
    attack = get_attack(attack_name)
    adv_ids = set(range(n_adv))
    for i in adv_ids:
        clients[i] = FederatedClient(i, attack.poison_dataset(client_datasets[i]), device=DEVICE)

    measurements = []
    for _ in range(rounds):
        selected = np.random.choice(N, K, replace=False)
        updates, adv_mask = [], []
        for idx in selected:
            update = clients[idx].train(server.global_model, epochs=1, lr=0.01, batch_size=64)
            if idx in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
                adv_mask.append(True)
            else:
                adv_mask.append(False)
            updates.append(update)

        if any(adv_mask):
            m = measure_round_margins(updates, adv_mask)
            if m is not None:
                measurements.append(m)

        server.aggregate(updates)

    return measurements


def _frac(rounds, key):
    return (sum(1 for r in rounds if r[key]), len(rounds))


if __name__ == '__main__':
    ALPHA = float(os.environ.get('SWEEP_ALPHA', '0.5'))
    OUT = os.environ.get('SWEEP_OUT', 'results/theorem_margins.json')

    all_m = []
    for seed in range(3):
        print(f"Seed {seed}...", end=' ', flush=True)
        ms = run_one_seed(seed, 'model_scaling', alpha=ALPHA)
        all_m.extend(ms)
        print(f"{len(ms)} adversarial rounds")

    pos = [m for m in all_m if not m['w_adv_is_zero']]
    print(f"\n=== alpha={ALPHA}: {len(all_m)} rounds, {len(pos)} with w_a > 0 ===")

    # Step A validity: the derived bound on delta' must hold in EVERY round.
    bad = [m for m in pos if not m['delta_bound_mean_holds']]
    print(f"delta' <= C_A * Rbar'_B holds: {len(pos)-len(bad)}/{len(pos)} positive-w rounds")
    if bad:
        print("  *** VIOLATION -- the derivation is wrong, stop and re-derive ***")
    else:
        sl = [m['delta_bound_slack_mean'] for m in pos]
        print(f"  bound/realised slack: min={min(sl):.1f}x  mean={np.mean(sl):.1f}x  "
              f"(the bound is conservative by this factor)")

    for tag, key in [('mean-radius', 'stepA_met_mean_radius'),
                     ('max-radius ', 'stepA_met_max_radius'),
                     ('Step B     ', 'stepB_met')]:
        k, n = _frac(pos, key)
        print(f"{tag} condition met: {k}/{n} positive-w rounds")

    by_na = {}
    for m in pos:
        by_na.setdefault(m['n_adv_in_round'], []).append(m)
    for na in sorted(by_na):
        g = by_na[na]
        print(f"  n_a={na} (n_b={g[0]['n_benign_in_round']}, C_A={g[0]['C_A']:.3f}): "
              f"mean-radius {_frac(g,'stepA_met_mean_radius')[0]}/{len(g)}, "
              f"max-radius {_frac(g,'stepA_met_max_radius')[0]}/{len(g)}, "
              f"StepB {_frac(g,'stepB_met')[0]}/{len(g)}")

    vac = sum(1 for m in pos if m['stepB_vacuous_round'])
    print(f"Step B vacuous (rho* <= 1, no rho can satisfy it): {vac}/{len(pos)} rounds")
    if pos:
        rr = [m['Rbar_over_R_B_prime'] for m in pos]
        print(f"Rbar'_B / R'_B: mean={np.mean(rr):.3f} min={min(rr):.3f} max={max(rr):.3f}")

    results = {
        'alpha': ALPHA,
        'n_rounds_total': len(all_m),
        'n_rounds_positive_w': len(pos),
        'delta_bound_holds_all_positive_w': len(bad) == 0,
        'delta_bound_slack_min': float(min(m['delta_bound_slack_mean'] for m in pos)) if pos else None,
        'delta_bound_slack_mean': float(np.mean([m['delta_bound_slack_mean'] for m in pos])) if pos else None,
        'stepA_mean_radius_met': _frac(pos, 'stepA_met_mean_radius')[0] if pos else None,
        'stepA_max_radius_met': _frac(pos, 'stepA_met_max_radius')[0] if pos else None,
        'stepB_met': _frac(pos, 'stepB_met')[0] if pos else None,
        'stepB_vacuous_rounds': vac,
        'Rbar_over_R_B_prime_mean': float(np.mean([m['Rbar_over_R_B_prime'] for m in pos])) if pos else None,
        'per_na': {str(na): {
            'n_rounds': len(g),
            'n_benign': g[0]['n_benign_in_round'],
            'C_A': g[0]['C_A'],
            'stepA_mean_radius_met': _frac(g, 'stepA_met_mean_radius')[0],
            'stepA_max_radius_met': _frac(g, 'stepA_met_max_radius')[0],
            'stepB_met': _frac(g, 'stepB_met')[0],
        } for na, g in sorted(by_na.items())},
        'per_round': all_m,
    }
    with open(os.path.join(BASE, OUT), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}")
