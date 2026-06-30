"""
Run CIFAR-10 payoff matrix with 10 seeds (42-51) to characterize
the distribution of per-seed VoPD and mixed-NE frequency.
Output: results/cifar10_10seeds/
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

fl_config = FLConfig(
    num_clients=10,
    clients_per_round=5,
    num_rounds=50,
)

game_config = GameConfig()
attacks = game_config.attacks
defenses = game_config.defenses

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_10seeds")
os.makedirs(output_dir, exist_ok=True)

# Seeds 42-51; run one trial per seed (total 10 seeds x 42 pairs = 420 FL runs)
SEEDS = list(range(42, 52))

print(f"Running CIFAR-10 payoff matrix for {len(SEEDS)} seeds (seeds {SEEDS[0]}-{SEEDS[-1]})...")
print(f"Each seed: 42 (attack,defense) pairs, 1 trial, 50 rounds on MPS")
print(f"Output: {output_dir}")

all_per_seed = {}

for seed in SEEDS:
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Check if already done
    if os.path.exists(os.path.join(seed_dir, "payoff_results.json")):
        print(f"Seed {seed}: already complete, loading...")
        with open(os.path.join(seed_dir, "per_seed_results.json")) as f:
            seed_psr = json.load(f)
        for k, v in seed_psr.items():
            if k not in all_per_seed:
                all_per_seed[k] = []
            all_per_seed[k].extend(v)
        continue

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
        seed_psr = json.load(f)
    for k, v in seed_psr.items():
        if k not in all_per_seed:
            all_per_seed[k] = []
        all_per_seed[k].extend(v)

# Save combined per-seed results
combined_path = os.path.join(output_dir, "per_seed_results_all.json")
with open(combined_path, "w") as f:
    json.dump(all_per_seed, f, indent=2)

# Compute VoPD for each seed
print("\n=== Per-seed VoPD distribution (10 seeds) ===")
vopds = []
mixed_count = 0
for seed in SEEDS:
    seed_psr_path = os.path.join(output_dir, f"seed_{seed}", "per_seed_results.json")
    payoff_path = os.path.join(output_dir, f"seed_{seed}", "payoff_results.json")
    if not os.path.exists(seed_psr_path):
        print(f"Seed {seed}: missing results, skipping")
        continue
    with open(seed_psr_path) as f:
        psr = json.load(f)

    results = {}
    for a in attacks:
        for d in defenses:
            key = f"{a}_{d}"
            if key in psr:
                for e in psr[key]:
                    if e["seed"] == seed:
                        results[(a, d)] = e
                        break

    try:
        pm = PayoffMatrix.from_experiment_results(results, attacks, defenses)
        solver = GameSolver(pm)
        equilibria = solver.solve_nash()
        best_vopd = max((ne.value_of_information(pm) for ne in equilibria), default=0.0)
        is_mixed = any(
            (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
            for ne in equilibria
        )
        vopds.append(best_vopd)
        if is_mixed:
            mixed_count += 1
        print(f"  Seed {seed}: VoPD={best_vopd:.4f}, mixed={is_mixed}")
    except Exception as e:
        print(f"  Seed {seed}: solver error {e}")
        vopds.append(0.0)

print(f"\nSummary over {len(vopds)} seeds:")
print(f"  Mixed NE: {mixed_count}/{len(vopds)} seeds ({100*mixed_count/len(vopds):.0f}%)")
print(f"  VoPD distribution: {[round(v,3) for v in vopds]}")
print(f"  VoPD > 0: {sum(v > 1e-4 for v in vopds)}/{len(vopds)}")
print(f"  VoPD mean: {np.mean(vopds):.4f}, median: {np.median(vopds):.4f}")

summary = {
    "seeds": SEEDS,
    "vopds": [float(v) for v in vopds],
    "mixed_count": mixed_count,
    "n_seeds": len(vopds),
    "pct_mixed": 100 * mixed_count / max(len(vopds), 1),
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {output_dir}/summary.json")
