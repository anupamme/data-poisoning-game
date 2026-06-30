"""
Round 48 — solve static NE on augmented {fedavg, norm_clip, fltrust, foolsgold, reputation}
defense menu × {no_attack, backdoor_pixel, model_scaling, dba} attack menu.

For each seed in 42-46: build per-seed payoff matrices, solve Nash via nashpy support
enumeration, report whether FoolsGold or reputation enters any equilibrium support.

Output: results/cifar10_orthogonal_suite/ne_summary.json
"""
import json
import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

import nashpy as nash

ATTACKS  = ["no_attack", "backdoor_pixel", "model_scaling", "dba"]
DEFENSES = ["fedavg", "norm_clip", "foolsgold", "reputation"]  # FLTrust covered separately (Appendix L)
SEEDS    = [42, 43, 44, 45, 46]

# Defense costs (from defenses/defense_costs.py)
DEF_COSTS = {"fedavg": 0.0, "norm_clip": 0.03, "fltrust": 0.10,
             "foolsgold": 0.06, "reputation": 0.04}
ATK_COSTS = {"no_attack": 0.0, "backdoor_pixel": 0.05,
             "model_scaling": 0.05, "dba": 0.10}
LAMBDA_C = 0.1
LAMBDA_F = 0.05
LAMBDA_A = 0.1

out_dir = os.path.join(base_dir, "results", "cifar10_orthogonal_suite")
os.makedirs(out_dir, exist_ok=True)


def get_cell(seed, attack, defense):
    """Get (asr, accuracy, worst_class_acc) for one cell."""
    path = os.path.join(base_dir, "results", "cifar10_10seeds",
                         f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    key = f"{attack}_{defense}"
    # Try main cache
    if key in data:
        for e in data[key]:
            if e["seed"] == seed:
                return (e["attack_success_rate"], e["accuracy"],
                        e.get("worst_class_accuracy", 0))
    # FLTrust fall-back from pilot summary
    if defense == "fltrust":
        pilot_path = os.path.join(base_dir, "results", "cifar10_fltrust_pilot", "summary.json")
        if os.path.exists(pilot_path):
            with open(pilot_path) as f:
                pilot = json.load(f)
            for e in pilot.get("per_seed", []):
                if (e.get("seed") == seed and e.get("attack") == attack and
                        e.get("defense") == defense):
                    return (e["asr"], e["accuracy"], e.get("worst_class_acc", 0))
    return None


def build_payoffs(seed):
    """Build U_A and U_D matrices for one seed."""
    n_attacks, n_defenses = len(ATTACKS), len(DEFENSES)
    U_A = np.zeros((n_attacks, n_defenses))
    U_D = np.zeros((n_attacks, n_defenses))
    missing = []
    for i, a in enumerate(ATTACKS):
        for j, d in enumerate(DEFENSES):
            cell = get_cell(seed, a, d)
            if cell is None:
                missing.append((a, d))
                continue
            asr, acc, wca = cell
            # U_A = ASR - lambda_a * cost_attack
            U_A[i, j] = asr - LAMBDA_A * ATK_COSTS[a]
            # U_D = accuracy - lambda_c * cost_defense - lambda_f * (1 - worst_class_acc)
            U_D[i, j] = acc - LAMBDA_C * DEF_COSTS[d] - LAMBDA_F * (1 - wca)
    return U_A, U_D, missing


per_seed_results = []
for seed in SEEDS:
    U_A, U_D, missing = build_payoffs(seed)
    if missing:
        print(f"Seed {seed}: MISSING cells: {missing}")
        continue
    # Server is the column player; adversary is row player.
    # nashpy: Game(A, B) where A = row payoffs, B = col payoffs.
    # Row player (adversary) wants to MAX U_A. Col player (server) wants to MAX U_D.
    game = nash.Game(U_A, U_D)
    eqs = list(game.support_enumeration())
    print(f"\n=== Seed {seed}: {len(eqs)} equilibria ===")
    # Argmax check per defense
    argmax_per_defense = {DEFENSES[j]: ATTACKS[U_A[:, j].argmax()] for j in range(len(DEFENSES))}
    print(f"  Argmax per defense: {argmax_per_defense}")
    seed_info = {"seed": seed, "argmax_per_defense": argmax_per_defense,
                 "equilibria": []}
    for k, (sigma_A, sigma_D) in enumerate(eqs):
        ua = float(sigma_A @ U_A @ sigma_D)
        ud = float(sigma_A @ U_D @ sigma_D)
        # Full info VoPD: E_d[max_a U_A(a, d)] - U_A(NE)
        col_maxes = U_A.max(axis=0)
        full_info = float(col_maxes @ sigma_D)
        vopd = full_info - ua
        adv_support = [(ATTACKS[i], float(sigma_A[i])) for i in range(len(ATTACKS)) if sigma_A[i] > 0.01]
        srv_support = [(DEFENSES[j], float(sigma_D[j])) for j in range(len(DEFENSES)) if sigma_D[j] > 0.01]
        is_mixed = len(adv_support) > 1 or len(srv_support) > 1
        print(f"  NE{k+1}: {'MIXED' if is_mixed else 'PURE'}  U_A={ua:.3f} U_D={ud:.3f} VoPD={vopd:.4f}")
        print(f"    Adv support: {adv_support}")
        print(f"    Srv support: {srv_support}")
        seed_info["equilibria"].append({
            "is_mixed": is_mixed, "U_A": ua, "U_D": ud, "VoPD": vopd,
            "adv_support": adv_support, "srv_support": srv_support,
        })
    per_seed_results.append(seed_info)

# Aggregate
summary = {"seeds": SEEDS, "per_seed": per_seed_results,
            "lambda_c": LAMBDA_C, "lambda_f": LAMBDA_F, "lambda_a": LAMBDA_A}

# Count: in how many seeds does foolsgold or reputation appear in any NE support?
fg_in_support = 0
rep_in_support = 0
fltrust_in_support = 0
for s in per_seed_results:
    fg_seen = any(any(d == "foolsgold" for d, _ in eq["srv_support"]) for eq in s["equilibria"])
    rep_seen = any(any(d == "reputation" for d, _ in eq["srv_support"]) for eq in s["equilibria"])
    ft_seen = any(any(d == "fltrust" for d, _ in eq["srv_support"]) for eq in s["equilibria"])
    if fg_seen: fg_in_support += 1
    if rep_seen: rep_in_support += 1
    if ft_seen: fltrust_in_support += 1
summary["fg_in_support_count"] = fg_in_support
summary["rep_in_support_count"] = rep_in_support
summary["fltrust_in_support_count"] = fltrust_in_support
summary["n_seeds_analyzed"] = len(per_seed_results)

print(f"\n=== AUGMENTED-MENU VERDICT ===")
print(f"  Seeds analyzed: {len(per_seed_results)}")
print(f"  FoolsGold in NE support: {fg_in_support}/{len(per_seed_results)}")
print(f"  Reputation in NE support: {rep_in_support}/{len(per_seed_results)}")
print(f"  FLTrust in NE support: {fltrust_in_support}/{len(per_seed_results)}")

# Pre-committed criterion
positive = (fg_in_support >= 1) or (rep_in_support >= 1)
if positive:
    print(f"\n  *** POSITIVE: at least one orthogonal-signal defense enters NE support ***")
else:
    distinct_argmax = any(
        s["argmax_per_defense"]["foolsgold"] != s["argmax_per_defense"]["fedavg"]
        or s["argmax_per_defense"]["reputation"] != s["argmax_per_defense"]["fedavg"]
        for s in per_seed_results
    )
    if distinct_argmax:
        print(f"\n  *** PARTIAL: argmax divergence yes, but no support entry (similar to FLTrust) ***")
    else:
        print(f"\n  *** NEGATIVE: no argmax divergence introduced by new defenses ***")

out_path = os.path.join(out_dir, "ne_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_path}")
