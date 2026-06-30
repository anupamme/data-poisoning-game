"""
Stateless FL pilot: Gaussian-noise availability attack.

Runs a 4×5 payoff matrix (gaussian_noise replaces model_scaling) on CIFAR-10,
N=10, K=5, 5 seeds. Then solves the game's NE and runs the NE mixed policy
to measure realized VoPD.

Attack semantics note:
  GaussianNoiseAttack replaces malicious updates with scaled Gaussian noise.
  Effect: degrades global accuracy each round the noise enters aggregation.
  The attack is stateless — each round's noise is independent.
  ASR for this attack is measured as ACCURACY DEGRADATION (1 - accuracy),
  not trigger-based ASR (which would always be 0).

Output: results/cifar10_stateless/
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "gaussian_noise", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]  # 8 seeds total

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_stateless")
os.makedirs(output_dir, exist_ok=True)

print(f"Stateless FL pilot: 4×5 game (gaussian_noise replaces model_scaling)")
print(f"N=10, K=5, {len(SEEDS)} seeds, 50 rounds")
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

    seed_nes = []
    for i, ne in enumerate(equilibria):
        adv = [game_config.attacks[j] for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
        srv = [game_config.defenses[j] for j, p in enumerate(ne.server_strategy) if p > 0.01]
        vopd_val = float(ne.value_of_information(pm))
        seed_nes.append({
            "ne_idx": i + 1,
            "vopd": vopd_val,
            "adversary_support": adv,
            "server_support": srv,
        })
        print(f"    NE{i+1}: VoPD={vopd_val:.4f}  Adv:{adv}  Srv:{srv}")

    per_seed_nes.append({
        "seed": seed,
        "best_vopd": float(best_vopd),
        "mixed": is_mixed,
        "equilibria": seed_nes,
    })
    print(f"  Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}")

# Summary
print(f"\n=== Stateless Pilot Summary ===")
print(f"VoPDs: {[round(v, 3) for v in vopds]}")
mixed_count = sum(v > 1e-4 for v in vopds)
print(f"Mixed NE: {mixed_count}/{len(vopds)}")
print(f"Mean VoPD: {np.mean(vopds):.4f} ± {np.std(vopds):.4f}")

summary = {
    "attacks": game_config.attacks,
    "defenses": game_config.defenses,
    "seeds": SEEDS,
    "vopds": [float(v) for v in vopds],
    "mixed_count": int(mixed_count),
    "n_seeds": len(vopds),
    "mean_vopd": float(np.mean(vopds)),
    "std_vopd": float(np.std(vopds)),
    "per_seed": per_seed_nes,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {output_dir}/summary.json")
