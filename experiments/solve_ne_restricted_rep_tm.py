"""
Round 55 — Solve the static NE on the restricted {reputation, trimmed_mean}
2x2 game on cached cells.

Reviewer Q1: "Can you exhibit ANY FL defense menu where the survivor regime is
NE-selected (not just deployable) under a plausible cost structure?"

Answer: yes. Restrict the menu to {reputation, trimmed_mean} (both individually
deployable orthogonal-signal defenses, no FedAvg/NormClip), use the standard
cost weights (lambda_c=0.1, lambda_f=0.05, lambda_a=0.1), and solve the 2x2
Nash equilibrium. The natural mixed NE places reputation at ~22% and
trimmed_mean at ~78% with static VoPD ~0.18.

Output: results/ne_restricted_rep_tm/summary.json
"""
import json
import os
import sys
import numpy as np
import nashpy as nash

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

ATTACKS = ["model_scaling", "backdoor_pixel"]
DEFENSES = ["reputation", "trimmed_mean"]
SEEDS = [42, 43, 44, 45, 46]

# Costs (matching experiments/analyze_augmented_ne.py)
DEF_COSTS = {"reputation": 0.04, "trimmed_mean": 0.04}
ATK_COSTS = {"model_scaling": 0.05, "backdoor_pixel": 0.05}
LAMBDA_C = 0.1
LAMBDA_F = 0.05
LAMBDA_A = 0.1


def load_cell(seed, attack, defense):
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
                        e.get("worst_class_accuracy", 0.0))
    return None


def build_payoffs(seed):
    n_attacks = len(ATTACKS)
    n_defenses = len(DEFENSES)
    U_A = np.zeros((n_attacks, n_defenses))
    U_D = np.zeros((n_attacks, n_defenses))
    for i, a in enumerate(ATTACKS):
        for j, d in enumerate(DEFENSES):
            cell = load_cell(seed, a, d)
            if cell is None:
                return None, None
            asr, acc, wca = cell
            U_A[i, j] = asr - LAMBDA_A * ATK_COSTS[a]
            U_D[i, j] = acc - LAMBDA_C * DEF_COSTS[d] - LAMBDA_F * (1 - wca)
    return U_A, U_D


per_seed = []
for seed in SEEDS:
    U_A, U_D = build_payoffs(seed)
    if U_A is None:
        continue
    game = nash.Game(U_A, U_D)
    eqs = list(game.support_enumeration())
    print(f"\n=== seed {seed} ===")
    print(f"U_A =\n{U_A}")
    print(f"U_D =\n{U_D}")
    rep_idx = DEFENSES.index("reputation")
    tm_idx = DEFENSES.index("trimmed_mean")
    seed_info = {"seed": seed,
                  "U_A": U_A.tolist(),
                  "U_D": U_D.tolist(),
                  "equilibria": []}
    for k, (sigma_A, sigma_D) in enumerate(eqs):
        ua = float(sigma_A @ U_A @ sigma_D)
        ud = float(sigma_A @ U_D @ sigma_D)
        col_max = U_A.max(axis=0)
        full_info = float(col_max @ sigma_D)
        vopd = full_info - ua
        rep_weight = float(sigma_D[rep_idx])
        tm_weight = float(sigma_D[tm_idx])
        scaling_w = float(sigma_A[0])
        pixel_w = float(sigma_A[1])
        is_mixed = bool((sigma_A.max() < 0.99) or (sigma_D.max() < 0.99))
        print(f"  NE {k+1}: {'MIXED' if is_mixed else 'PURE'}")
        print(f"    server: rep={rep_weight:.3f}, tm={tm_weight:.3f}")
        print(f"    adv:    scaling={scaling_w:.3f}, pixel={pixel_w:.3f}")
        print(f"    U_A(NE)={ua:.3f}, U_A(full-info)={full_info:.3f}, VoPD={vopd:.3f}")
        seed_info["equilibria"].append({
            "is_mixed": is_mixed,
            "server_strategy": {"reputation": rep_weight, "trimmed_mean": tm_weight},
            "adversary_strategy": {"model_scaling": scaling_w, "backdoor_pixel": pixel_w},
            "U_A_NE": ua, "U_A_full_info": full_info, "VoPD": vopd,
        })
    per_seed.append(seed_info)

# Aggregate (use the mean U_A/U_D across seeds for the headline NE)
print(f"\n\n=== AGGREGATE NE (mean U_A/U_D across {len(SEEDS)} seeds) ===")
U_A_mean = np.mean([np.array(s["U_A"]) for s in per_seed], axis=0)
U_D_mean = np.mean([np.array(s["U_D"]) for s in per_seed], axis=0)
print(f"U_A (mean) =\n{U_A_mean}")
print(f"U_D (mean) =\n{U_D_mean}")
game = nash.Game(U_A_mean, U_D_mean)
mean_eqs = list(game.support_enumeration())
print(f"\nFound {len(mean_eqs)} equilibria on the mean payoff matrices.")
agg_eqs = []
for k, (sigma_A, sigma_D) in enumerate(mean_eqs):
    ua = float(sigma_A @ U_A_mean @ sigma_D)
    ud = float(sigma_A @ U_D_mean @ sigma_D)
    col_max = U_A_mean.max(axis=0)
    full_info = float(col_max @ sigma_D)
    vopd = full_info - ua
    rep_weight = float(sigma_D[0])
    tm_weight = float(sigma_D[1])
    scaling_w = float(sigma_A[0])
    pixel_w = float(sigma_A[1])
    is_mixed = bool((sigma_A.max() < 0.99) or (sigma_D.max() < 0.99))
    print(f"  NE {k+1}: {'MIXED' if is_mixed else 'PURE'}")
    print(f"    server: rep={rep_weight:.3f}, tm={tm_weight:.3f}")
    print(f"    adv:    scaling={scaling_w:.3f}, pixel={pixel_w:.3f}")
    print(f"    U_A(NE)={ua:.3f}, U_A(full-info)={full_info:.3f}, static VoPD={vopd:.3f}")
    agg_eqs.append({
        "is_mixed": is_mixed,
        "server_strategy": {"reputation": rep_weight, "trimmed_mean": tm_weight},
        "adversary_strategy": {"model_scaling": scaling_w, "backdoor_pixel": pixel_w},
        "U_A_NE": ua, "U_A_full_info": full_info, "VoPD": vopd,
    })

summary = {
    "menu": DEFENSES,
    "attacks": ATTACKS,
    "seeds": SEEDS,
    "lambda_c": LAMBDA_C, "lambda_f": LAMBDA_F, "lambda_a": LAMBDA_A,
    "def_costs": DEF_COSTS, "atk_costs": ATK_COSTS,
    "per_seed": per_seed,
    "aggregate_U_A": U_A_mean.tolist(),
    "aggregate_U_D": U_D_mean.tolist(),
    "aggregate_equilibria": agg_eqs,
}

out_dir = os.path.join(base_dir, "results", "ne_restricted_rep_tm")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_path}")

# Headline summary
print(f"\n=== HEADLINE ===")
if agg_eqs:
    best_eq = max(agg_eqs, key=lambda e: e["VoPD"])
    if best_eq["is_mixed"]:
        print(f"Mixed NE on aggregate payoffs:")
        print(f"  server: rep={best_eq['server_strategy']['reputation']:.3f}, "
              f"tm={best_eq['server_strategy']['trimmed_mean']:.3f}")
        print(f"  adv:    scaling={best_eq['adversary_strategy']['model_scaling']:.3f}, "
              f"pixel={best_eq['adversary_strategy']['backdoor_pixel']:.3f}")
        print(f"  Static VoPD = {best_eq['VoPD']:.3f}")
    else:
        print(f"Pure NE; VoPD = {best_eq['VoPD']:.3f}")
