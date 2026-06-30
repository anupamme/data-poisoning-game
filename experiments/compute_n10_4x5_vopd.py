"""
Compute N=10 VoPD on the 4×5 game (same subset as the N=20/50/100 sweep).

Uses cached per_seed_results.json from results/cifar10_10seeds/ — no new FL training.

Output: results/cifar10_10seeds/n10_4x5_summary.json
"""
import sys, os, json
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

ATTACKS  = ["no_attack", "backdoor_pixel", "model_scaling", "dba"]
DEFENSES = ["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"]
SEEDS    = list(range(42, 52))

seed_dir_base = os.path.join(base_dir, "results", "cifar10_10seeds")
output_path   = os.path.join(seed_dir_base, "n10_4x5_summary.json")

vopds, asrs, mixed_count = [], [], 0
per_seed = []

for seed in SEEDS:
    path = os.path.join(seed_dir_base, f"seed_{seed}", "per_seed_results.json")
    with open(path) as f:
        psr = json.load(f)

    results = {}
    for a in ATTACKS:
        for d in DEFENSES:
            key = f"{a}_{d}"
            if key in psr:
                for e in psr[key]:
                    if e["seed"] == seed:
                        results[(a, d)] = e
                        break

    # ASR binding cell
    asr_val = None
    for e in psr.get("model_scaling_fedavg", []):
        if e["seed"] == seed:
            asr_val = e["attack_success_rate"]
            break

    pm = PayoffMatrix.from_experiment_results(results, ATTACKS, DEFENSES)
    solver = GameSolver(pm)
    equilibria = solver.solve_nash()
    best_v = max((ne.value_of_information(pm) for ne in equilibria), default=0.0)
    is_mixed = any(
        (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
        for ne in equilibria
    )

    vopds.append(float(best_v))
    asrs.append(float(asr_val) if asr_val is not None else None)
    if best_v > 1e-4:
        mixed_count += 1

    per_seed.append({
        "seed": seed,
        "best_vopd": float(best_v),
        "mixed": bool(is_mixed),
        "asr_model_scaling_fedavg": float(asr_val) if asr_val is not None else None,
    })
    asr_str = f"{asr_val:.3f}" if asr_val is not None else "N/A"
    print(f"  Seed {seed}: VoPD={best_v:.4f}, mixed={is_mixed}, ASR={asr_str}")

asrs_valid = [a for a in asrs if a is not None]
summary = {
    "n": 10,
    "clients_per_round": 5,
    "game": "4x5",
    "attacks": ATTACKS,
    "defenses": DEFENSES,
    "seeds": SEEDS,
    "vopds": vopds,
    "mean_vopd": float(np.mean(vopds)),
    "std_vopd": float(np.std(vopds)),
    "mixed_count": int(mixed_count),
    "n_seeds": len(SEEDS),
    "asr_binding_cell": {
        "cell": "model_scaling_fedavg",
        "per_seed": asrs_valid,
        "mean": float(np.mean(asrs_valid)) if asrs_valid else None,
        "std": float(np.std(asrs_valid)) if asrs_valid else None,
    },
    "per_seed": per_seed,
}

with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nN=10 (4×5): mixed={mixed_count}/{len(SEEDS)}, "
      f"mean VoPD={np.mean(vopds):.4f}±{np.std(vopds):.4f}, "
      f"ASR={np.mean(asrs_valid):.3f}±{np.std(asrs_valid):.3f}")
print(f"Saved: {output_path}")
