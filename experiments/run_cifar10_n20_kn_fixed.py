"""
K/N-fixed contrast experiment: N=20, K=10 (K/N=0.5).

Complements the existing N=20, K=4 (K/N=0.2) run by holding K/N=0.5 constant
(same as N=10 K=5 baseline) while varying N.

Expected: P(>=1 adv/round) at N=20, K=10 ≈ 0.957 (vs 0.624 at K=4).
Predicted: higher VoPD than K/N=0.2 at same N, validating participation-probability mechanism.

Output: results/cifar10_20clients_k10/
"""
import sys, os, json
import numpy as np
from scipy.special import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(num_clients=20, clients_per_round=10, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46]
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_20clients_k10")
os.makedirs(output_dir, exist_ok=True)

# Compute participation probability
N, K, f = 20, 10, 0.2
n_adv = int(N * f)  # 4
n_ben = N - n_adv   # 16
p_none = comb(n_ben, K, exact=True) / comb(N, K, exact=True)
p_part = 1.0 - p_none
print(f"N=20 K=10 CIFAR-10 K/N-fixed contrast: 4x5 game, {len(SEEDS)} seeds")
print(f"K/N = {K/N:.2f}, P(>=1 adv/round) = {p_part:.4f}")
print(f"Compare: N=20 K=4, K/N=0.20, P=0.6242 (existing)")
print(f"Attacks: {game_config.attacks}")
print(f"Defenses: {game_config.defenses}")
print(f"Output: {output_dir}\n")

vopds = []
per_seed_nes = []

for seed in SEEDS:
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    if os.path.exists(os.path.join(seed_dir, "payoff_results.json")):
        print(f"Seed {seed}: already complete, loading...")
    else:
        exp_config = ExperimentConfig(
            dataset="cifar10",
            model="cifar_cnn",
            dirichlet_alpha=0.5,
            adversarial_fraction=0.2,
            num_trials=1,
            seed=seed,
            device="mps",
        )
        print(f"\n--- Seed {seed} ---")
        run_full_payoff_matrix(fl_config, exp_config, game_config, seed_dir)

    with open(os.path.join(seed_dir, "per_seed_results.json")) as f:
        psr = json.load(f)

    results = {}
    for a in game_config.attacks:
        for d in game_config.defenses:
            key = f"{a}_{d}"
            if key in psr:
                for e in psr[key]:
                    if e["seed"] == seed:
                        results[(a, d)] = e
                        break

    pm = PayoffMatrix.from_experiment_results(results, game_config.attacks, game_config.defenses)
    solver = GameSolver(pm)
    equilibria = solver.solve_nash()

    best_vopd = max((ne.value_of_information(pm) for ne in equilibria), default=0.0)
    is_mixed = any(
        (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
        for ne in equilibria
    )
    vopds.append(best_vopd)

    complementarity_correct = []
    seed_nes = []
    for i, ne in enumerate(equilibria):
        adv = [game_config.attacks[j] for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
        srv = [game_config.defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01]
        vopd_val = float(ne.value_of_information(pm))

        srv_indices = [game_config.defenses.index(d) for d in srv]
        best_attacks_per_def = [
            set(np.where(pm.adversary_payoffs[:, j] == pm.adversary_payoffs[:, j].max())[0])
            for j in srv_indices
        ]
        intersection = best_attacks_per_def[0]
        for s in best_attacks_per_def[1:]:
            intersection = intersection & s
        has_common_best_response = len(intersection) > 0
        theorem_predicts_null = has_common_best_response
        actual_null = vopd_val < 1e-4
        complementarity_correct.append(theorem_predicts_null == actual_null)

        seed_nes.append({
            "ne_idx": i + 1,
            "adversary_utility": float(ne.adversary_utility),
            "server_utility": float(ne.server_utility),
            "vopd": vopd_val,
            "adversary_support": adv,
            "server_support": srv,
            "complementarity_correct": theorem_predicts_null == actual_null,
        })
        print(f"    NE{i+1}: U_A={ne.adversary_utility:.4f}, VoPD={vopd_val:.4f}")
        print(f"      Adv: {adv}  Srv: {srv}")

    asr_binding = None
    for e in psr.get("model_scaling_fedavg", []):
        if e["seed"] == seed:
            asr_binding = e["attack_success_rate"]
            break

    all_correct = all(complementarity_correct)
    asr_str = f"{asr_binding:.3f}" if asr_binding is not None else "N/A"
    per_seed_nes.append({
        "seed": seed,
        "best_vopd": float(best_vopd),
        "mixed": is_mixed,
        "complementarity_all_correct": all_correct,
        "asr_model_scaling_fedavg": asr_binding,
        "equilibria": seed_nes,
    })
    print(f"  Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, "
          f"ASR(scaling,fedavg)={asr_str}, diagnostic_correct={all_correct}")

print(f"\n=== N=20 K=10 (K/N=0.50) Summary ===")
print(f"VoPDs: {[round(v, 3) for v in vopds]}")
mixed_count = sum(v > 1e-4 for v in vopds)
diagnostic_correct = sum(s["complementarity_all_correct"] for s in per_seed_nes)
asrs = [s["asr_model_scaling_fedavg"] for s in per_seed_nes if s["asr_model_scaling_fedavg"] is not None]
print(f"Mixed NE: {mixed_count}/{len(vopds)} seeds")
print(f"Complementarity diagnostic correct: {diagnostic_correct}/{len(vopds)} seeds")
print(f"Mean VoPD: {np.mean(vopds):.4f} ± {np.std(vopds):.4f}")
if asrs:
    print(f"ASR(model_scaling, fedavg): mean={np.mean(asrs):.3f}, std={np.std(asrs):.3f}")
print(f"P(>=1 adv/round) = {p_part:.4f}")
print(f"\nContrast with N=20 K=4 (K/N=0.20):")
print(f"  K=4: P=0.6242, expected VoPD≈0.044 (load results/cifar10_20clients/summary.json)")

summary = {
    "num_clients": 20,
    "clients_per_round": 10,
    "kn_ratio": K / N,
    "participation_prob": float(p_part),
    "seeds": SEEDS,
    "attacks": game_config.attacks,
    "defenses": game_config.defenses,
    "vopds": [float(v) for v in vopds],
    "mixed_count": int(mixed_count),
    "n_seeds": len(vopds),
    "pct_mixed": float(100 * mixed_count / len(vopds)),
    "mean_vopd": float(np.mean(vopds)),
    "std_vopd": float(np.std(vopds)),
    "median_vopd": float(np.median(vopds)),
    "diagnostic_correct": int(diagnostic_correct),
    "asr_binding_cell": {
        "cell": "model_scaling_fedavg",
        "per_seed": asrs,
        "mean": float(np.mean(asrs)) if asrs else None,
        "std": float(np.std(asrs)) if asrs else None,
    },
    "per_seed": per_seed_nes,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved to {output_dir}/summary.json")
