"""
N=100 client CIFAR-10 rich scale validation (4x5 game, 5 seeds).
Attacks: no_attack, backdoor_pixel, model_scaling, dba
Defenses: fedavg, norm_clip, rfa, trimmed_mean, coord_median
Output: results/cifar10_100clients_rich/
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(num_clients=100, clients_per_round=20, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46]
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_100clients_rich")
os.makedirs(output_dir, exist_ok=True)

print(f"N=100 CIFAR-10 rich scale validation: 4x5 game, {len(SEEDS)} seeds")
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

    # Check complementarity diagnostic: for each NE, does the theorem correctly predict VoPD sign?
    complementarity_correct = []
    seed_nes = []
    for i, ne in enumerate(equilibria):
        adv = [game_config.attacks[j] for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
        srv = [game_config.defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01]
        vopd_val = float(ne.value_of_information(pm))

        # Complementarity condition: intersection of argmax_a U_A(a,d) for d in srv support
        srv_indices = [game_config.defenses.index(d) for d in srv]
        best_attacks_per_def = [
            set(np.where(pm.adversary_payoffs[:, j] == pm.adversary_payoffs[:, j].max())[0])
            for j in srv_indices
        ]
        intersection = best_attacks_per_def[0]
        for s in best_attacks_per_def[1:]:
            intersection = intersection & s
        has_common_best_response = len(intersection) > 0
        # Theorem: VoPD=0 iff common best response exists
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

    all_correct = all(complementarity_correct)
    per_seed_nes.append({
        "seed": seed,
        "best_vopd": float(best_vopd),
        "mixed": is_mixed,
        "complementarity_all_correct": all_correct,
        "equilibria": seed_nes,
    })
    print(f"  Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}, diagnostic_correct={all_correct}")

print(f"\n=== N=100 Rich CIFAR-10 Summary ===")
print(f"VoPDs: {[round(v, 3) for v in vopds]}")
mixed_count = sum(v > 1e-4 for v in vopds)
diagnostic_correct = sum(s["complementarity_all_correct"] for s in per_seed_nes)
print(f"Mixed NE: {mixed_count}/{len(vopds)} seeds")
print(f"Complementarity diagnostic correct: {diagnostic_correct}/{len(vopds)} seeds")
print(f"Mean VoPD: {np.mean(vopds):.4f}, Median: {np.median(vopds):.4f}")

summary = {
    "seeds": SEEDS,
    "attacks": game_config.attacks,
    "defenses": game_config.defenses,
    "vopds": [float(v) for v in vopds],
    "mixed_count": int(mixed_count),
    "n_seeds": len(vopds),
    "pct_mixed": float(100 * mixed_count / len(vopds)),
    "mean_vopd": float(np.mean(vopds)),
    "median_vopd": float(np.median(vopds)),
    "diagnostic_correct": int(diagnostic_correct),
    "per_seed": per_seed_nes,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved to {output_dir}/summary.json")
