"""
Round 51 — lambda_c sweep on the augmented orthogonal-signal menu (Part C).

The Round 48 augmented-menu NE analysis (analyze_augmented_ne.py) used a fixed
lambda_c = 0.1. The reviewer asks: under what server cost-benefit profile does
the natural static NE land in the persistence-survival region (reputation NE
weight >= 70%)? This sweep re-solves the Nash equilibrium at lambda_c values
{0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3} on the {fedavg, norm_clip,
foolsgold, reputation} menu, using cached payoff cells -- no new compute.

Output: results/cifar10_orthogonal_suite/lambda_c_sweep.json
"""
import json
import os
import sys
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

import nashpy as nash

ATTACKS  = ["no_attack", "backdoor_pixel", "model_scaling", "dba"]
DEFENSES = ["fedavg", "norm_clip", "foolsgold", "reputation"]
SEEDS    = [42, 43, 44, 45, 46]
LAMBDA_C_VALUES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]

DEF_COSTS = {"fedavg": 0.0, "norm_clip": 0.03, "fltrust": 0.10,
             "foolsgold": 0.06, "reputation": 0.04}
ATK_COSTS = {"no_attack": 0.0, "backdoor_pixel": 0.05,
             "model_scaling": 0.05, "dba": 0.10}
LAMBDA_F = 0.05
LAMBDA_A = 0.1

out_dir = os.path.join(base_dir, "results", "cifar10_orthogonal_suite")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "lambda_c_sweep.json")


def get_cell(seed, attack, defense):
    path = os.path.join(base_dir, "results", "cifar10_10seeds",
                         f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    key = f"{attack}_{defense}"
    if key in data:
        for e in data[key]:
            if e["seed"] == seed:
                return (e["attack_success_rate"], e["accuracy"],
                        e.get("worst_class_accuracy", 0))
    return None


def build_payoffs(seed, lambda_c):
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
            U_A[i, j] = asr - LAMBDA_A * ATK_COSTS[a]
            U_D[i, j] = acc - lambda_c * DEF_COSTS[d] - LAMBDA_F * (1 - wca)
    return U_A, U_D, missing


def analyze_seed(seed, lambda_c):
    U_A, U_D, missing = build_payoffs(seed, lambda_c)
    if missing:
        return None
    game = nash.Game(U_A, U_D)
    eqs = list(game.support_enumeration())
    rep_idx = DEFENSES.index("reputation")
    fg_idx = DEFENSES.index("foolsgold")
    eq_records = []
    for sigma_A, sigma_D in eqs:
        ua = float(sigma_A @ U_A @ sigma_D)
        ud = float(sigma_A @ U_D @ sigma_D)
        col_maxes = U_A.max(axis=0)
        full_info = float(col_maxes @ sigma_D)
        vopd = full_info - ua
        rep_w = float(sigma_D[rep_idx])
        fg_w  = float(sigma_D[fg_idx])
        adv_supp = [(ATTACKS[i], float(sigma_A[i])) for i in range(len(ATTACKS)) if sigma_A[i] > 0.01]
        srv_supp = [(DEFENSES[j], float(sigma_D[j])) for j in range(len(DEFENSES)) if sigma_D[j] > 0.01]
        eq_records.append({
            "U_A": ua, "U_D": ud, "VoPD": vopd,
            "reputation_weight": rep_w,
            "foolsgold_weight": fg_w,
            "adv_support": adv_supp,
            "srv_support": srv_supp,
        })
    # Pick the max-VoPD equilibrium for headline
    if eq_records:
        best = max(eq_records, key=lambda e: e["VoPD"])
        max_rep_weight = max((e["reputation_weight"] for e in eq_records), default=0.0)
    else:
        best = None
        max_rep_weight = 0.0
    return {
        "n_equilibria": len(eq_records),
        "max_rep_weight_across_eqs": max_rep_weight,
        "best_eq": best,
        "all_eqs": eq_records,
    }


sweep = {"lambda_c_values": LAMBDA_C_VALUES, "seeds": SEEDS,
         "menu": DEFENSES, "attacks": ATTACKS,
         "per_lambda_c": {}}

print(f"=== lambda_c sweep on orthogonal menu {DEFENSES} ===")
print(f"{'λ_c':>6s} {'seed':>5s} {'NEs':>4s} {'max_rep':>8s} {'best_VoPD':>9s} {'srv_support'}")
print("-" * 80)

for lc in LAMBDA_C_VALUES:
    per_seed = []
    for seed in SEEDS:
        r = analyze_seed(seed, lc)
        if r is None:
            continue
        r["seed"] = seed
        per_seed.append(r)
        best = r["best_eq"]
        srv_str = "{" + ", ".join(f"{d}:{w:.2f}" for d, w in best["srv_support"]) + "}" if best else "-"
        bvopd_str = f"{best['VoPD']:.3f}" if best else "-"
        print(f"{lc:>6.3f} {seed:>5d} {r['n_equilibria']:>4d} {r['max_rep_weight_across_eqs']:>8.3f} "
              f"{bvopd_str:>9s} {srv_str}")
    # Aggregate per lambda_c
    rep_weights = [r["max_rep_weight_across_eqs"] for r in per_seed]
    sweep["per_lambda_c"][f"{lc:.3f}"] = {
        "lambda_c": lc,
        "n_seeds_analyzed": len(per_seed),
        "max_rep_weight_per_seed": rep_weights,
        "mean_max_rep_weight": float(np.mean(rep_weights)) if rep_weights else 0.0,
        "n_seeds_with_rep_70pct": sum(1 for w in rep_weights if w >= 0.70),
        "n_seeds_with_rep_50pct": sum(1 for w in rep_weights if w >= 0.50),
        "per_seed": per_seed,
    }
    print(f"  -> lc={lc:.3f}: mean max-rep-weight = {np.mean(rep_weights):.3f}, "
          f"{sum(1 for w in rep_weights if w>=0.70)}/5 with rep_w>=70%, "
          f"{sum(1 for w in rep_weights if w>=0.50)}/5 with rep_w>=50%")
    print()

# Final summary
print(f"\n=== SUMMARY: max reputation NE weight by lambda_c ===")
print(f"{'lambda_c':>9s}  {'mean_max_rep':>13s}  {'seeds_>=50%':>11s}  {'seeds_>=70%':>11s}")
for lc in LAMBDA_C_VALUES:
    key = f"{lc:.3f}"
    a = sweep["per_lambda_c"][key]
    print(f"{lc:>9.3f}  {a['mean_max_rep_weight']:>13.3f}  "
          f"{a['n_seeds_with_rep_50pct']:>11d}  {a['n_seeds_with_rep_70pct']:>11d}")

with open(out_path, "w") as f:
    json.dump(sweep, f, indent=2)
print(f"\nSaved: {out_path}")
