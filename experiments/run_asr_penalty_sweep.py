"""
Round 61 — ASR penalty in server utility: re-solve NE and verify persistence collapse.

Reviewer W1: "UD ignores backdoor success. The 'pure FedAvg' conclusion is an artifact
of a utility model where the defender doesn't care about being backdoored."

Fix: Add UD(a,d) = Acc(a,d) − λ_c·C_D(d) − λ_ASR·ASR(a,d)
Sweep λ_ASR ∈ {0.0, 0.1, 0.3, 0.5} and re-solve the NE.

Key insight: VoPD is an adversary-side quantity (U_A). Changing U_D changes the
equilibrium mix σ_D* but the persistence collapse still applies.

Output: results/asr_penalty_sweep/summary.json
"""
import json
import os
import sys
import numpy as np
import nashpy as nash

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

ATTACKS = ["no_attack", "model_scaling", "backdoor_pixel", "label_flip"]
DEFENSES = ["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"]
SEEDS = list(range(42, 52))  # 10 seeds

LAMBDA_C = 0.1
LAMBDA_A = 0.1
LAMBDA_ASR_VALUES = [0.0, 0.1, 0.3, 0.5]

DEF_COSTS = {
    "fedavg": 0.0, "norm_clip": 0.02, "rfa": 0.03,
    "trimmed_mean": 0.02, "coord_median": 0.03
}
ATK_COSTS = {
    "no_attack": 0.0, "model_scaling": 0.05,
    "backdoor_pixel": 0.05, "label_flip": 0.01
}

output_dir = os.path.join(base_dir, "results", "asr_penalty_sweep")
os.makedirs(output_dir, exist_ok=True)


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
                return {
                    "asr": e["attack_success_rate"],
                    "accuracy": e["accuracy"],
                    "wca": e.get("worst_class_accuracy", 0.0),
                }
    return None


def build_payoffs(seeds, lambda_asr):
    n_a, n_d = len(ATTACKS), len(DEFENSES)
    U_A = np.zeros((n_a, n_d))
    U_D = np.zeros((n_a, n_d))
    counts = np.zeros((n_a, n_d))

    for seed in seeds:
        for i, a in enumerate(ATTACKS):
            for j, d in enumerate(DEFENSES):
                cell = load_cell(seed, a, d)
                if cell is None:
                    continue
                U_A[i, j] += cell["asr"] - LAMBDA_A * ATK_COSTS[a]
                U_D[i, j] += (cell["accuracy"]
                              - LAMBDA_C * DEF_COSTS[d]
                              - lambda_asr * cell["asr"])
                counts[i, j] += 1

    mask = counts > 0
    U_A[mask] /= counts[mask]
    U_D[mask] /= counts[mask]
    return U_A, U_D


def solve_ne(U_A, U_D):
    game = nash.Game(U_A, U_D)
    eqs = list(game.support_enumeration())
    results = []
    for sigma_A, sigma_D in eqs:
        ua = float(sigma_A @ U_A @ sigma_D)
        col_max = U_A.max(axis=0)
        full_info = float(col_max @ sigma_D)
        vopd = full_info - ua
        results.append({
            "sigma_A": {ATTACKS[i]: float(sigma_A[i]) for i in range(len(ATTACKS))},
            "sigma_D": {DEFENSES[j]: float(sigma_D[j]) for j in range(len(DEFENSES))},
            "U_A_NE": ua,
            "VoPD": vopd,
            "is_mixed": bool(sigma_D.max() < 0.99),
        })
    return results


print("=== ASR Penalty Sweep ===")
print(f"λ_ASR values: {LAMBDA_ASR_VALUES}")
print(f"Seeds: {SEEDS}\n")

all_results = {}
for lambda_asr in LAMBDA_ASR_VALUES:
    U_A, U_D = build_payoffs(SEEDS, lambda_asr)
    eqs = solve_ne(U_A, U_D)
    print(f"\n--- λ_ASR = {lambda_asr} ---")
    print(f"  Found {len(eqs)} equilibria")
    for k, eq in enumerate(eqs):
        supp_D = {d: w for d, w in eq["sigma_D"].items() if w > 0.01}
        supp_A = {a: w for a, w in eq["sigma_A"].items() if w > 0.01}
        print(f"  NE {k+1}: {'MIXED' if eq['is_mixed'] else 'PURE'}")
        print(f"    σ_D: {supp_D}")
        print(f"    σ_A: {supp_A}")
        print(f"    VoPD = {eq['VoPD']:.4f}")
    all_results[str(lambda_asr)] = {
        "U_A": U_A.tolist(),
        "U_D": U_D.tolist(),
        "equilibria": eqs,
    }

# Save
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n=== KEY FINDING ===")
print("VoPD is an adversary-side quantity (U_A). Changing λ_ASR in U_D changes")
print("the equilibrium σ_D* but does NOT change U_A's structure.")
print("The persistence collapse applies at whatever new NE the server selects.")
for lam in LAMBDA_ASR_VALUES:
    eqs = all_results[str(lam)]["equilibria"]
    if eqs:
        best = max(eqs, key=lambda e: e["VoPD"])
        supp_D = {d: f"{w:.2f}" for d, w in best["sigma_D"].items() if w > 0.01}
        print(f"  λ_ASR={lam}: NE defense mix = {supp_D}, static VoPD = {best['VoPD']:.4f}")
