"""
Positive FL pilot v2 (Round 42 — reviewer-designed recipe).

The reviewer's exact prescription: "a low-K/N, higher-f stateless availability
setting where admission variance is restored."

Setup:
- N=20, K=4, f=0.4 (8 adversarial clients of 20, K/N=0.20)
- Attack menu: no_attack, label_flip (high poison), gaussian_noise, dba
  (NO pixel — it dominated the Round 39 pilot's equilibrium and prevented
  gaussian_noise from entering support)
- Defense menu: fedavg, norm_clip, rfa, trimmed_mean, coord_median
- 4×5 game, 5 seeds (42-46), 50 rounds

Pre-committed interpretation:
- STRONG positive: ≥3/5 seeds mixed NE under metric (i), VoPD > 0.05
- WEAK positive: 1-2/5 seeds; or mixed NE only under metric (ii)
- NEGATIVE: 0/5 seeds mixed NE under either metric

Output: results/cifar10_positive_pilot_v2/
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
from defenses.defense_costs import get_defense_cost
from attacks import get_attack

# Config
fl_config = FLConfig(num_clients=20, clients_per_round=4, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "label_flip", "gaussian_noise", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median"],
)
SEEDS = [42, 43, 44, 45, 46]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_positive_pilot_v2")
os.makedirs(output_dir, exist_ok=True)

print(f"Positive FL pilot v2: N=20, K=4, f=0.4 (low K/N, restored admission variance)")
print(f"Attacks: {game_config.attacks}")
print(f"Defenses: {game_config.defenses}")
print(f"Seeds: {SEEDS}, 50 rounds, 4×5 game\n")

# Run payoff matrices
for seed in SEEDS:
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    if os.path.exists(os.path.join(seed_dir, "payoff_results.json")):
        print(f"Seed {seed}: already complete, skipping")
        continue

    exp_config = ExperimentConfig(
        dataset="cifar10",
        model="cifar_cnn",
        dirichlet_alpha=0.5,
        adversarial_fraction=0.4,    # KEY: raised from 0.2
        num_trials=1,
        seed=seed,
        device="mps",
    )
    print(f"\n--- Seed {seed} ---")
    run_full_payoff_matrix(fl_config, exp_config, game_config, seed_dir)


# Analyze under BOTH metrics
def compute_vopd_two_metrics(seed):
    """Compute VoPD under (i) primary ASR-only and (ii) availability-aware utility."""
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    with open(os.path.join(seed_dir, "payoff_results.json")) as f:
        results_raw = json.load(f)

    attacks = game_config.attacks
    defenses = game_config.defenses
    baseline_acc = results_raw["no_attack_fedavg"]["accuracy"]

    # Metric (i): primary, ASR-only
    U_A_i = np.zeros((len(attacks), len(defenses)))
    U_D_i = np.zeros((len(attacks), len(defenses)))
    for i, a in enumerate(attacks):
        for j, d in enumerate(defenses):
            r = results_raw[f"{a}_{d}"]
            asr = r["attack_success_rate"]
            acc = r["accuracy"]
            U_A_i[i, j] = (0 if a == "no_attack" else asr - 0.1 * get_attack(a).cost)
            U_D_i[i, j] = acc - 0.1 * get_defense_cost(d)

    pm_i = PayoffMatrix(attacks=attacks, defenses=defenses,
                        adversary_payoffs=U_A_i, server_payoffs=U_D_i)
    eqs_i = GameSolver(pm_i).solve_nash()
    best_vopd_i = max((ne.value_of_information(pm_i) for ne in eqs_i), default=0.0)
    mixed_i = any((ne.adversary_strategy > 0.01).sum() > 1 for ne in eqs_i)

    # Metric (ii): availability-aware (use max(ASR, accuracy_drop))
    U_A_ii = np.zeros((len(attacks), len(defenses)))
    for i, a in enumerate(attacks):
        for j, d in enumerate(defenses):
            r = results_raw[f"{a}_{d}"]
            asr = r["attack_success_rate"]
            acc = r["accuracy"]
            acc_drop = max(0, baseline_acc - acc)
            impact = max(asr, acc_drop) if a != "no_attack" else 0
            U_A_ii[i, j] = impact - 0.1 * get_attack(a).cost

    pm_ii = PayoffMatrix(attacks=attacks, defenses=defenses,
                         adversary_payoffs=U_A_ii, server_payoffs=U_D_i)
    eqs_ii = GameSolver(pm_ii).solve_nash()
    best_vopd_ii = max((ne.value_of_information(pm_ii) for ne in eqs_ii), default=0.0)
    mixed_ii = any((ne.adversary_strategy > 0.01).sum() > 1 for ne in eqs_ii)

    # Get NE supports
    def get_support(eqs, pm, attacks, defenses):
        best = max(eqs, key=lambda ne: ne.value_of_information(pm), default=None)
        if best is None:
            return None
        adv = {attacks[k]: round(p, 3) for k, p in enumerate(best.adversary_strategy) if p > 0.01}
        srv = {defenses[k]: round(p, 3) for k, p in enumerate(best.server_strategy) if p > 0.01}
        return {"adv": adv, "srv": srv}

    return {
        "seed": seed,
        "metric_i": {
            "vopd": float(best_vopd_i),
            "mixed": mixed_i,
            "support": get_support(eqs_i, pm_i, attacks, defenses),
            "U_A": U_A_i.tolist(),
        },
        "metric_ii": {
            "vopd": float(best_vopd_ii),
            "mixed": mixed_ii,
            "support": get_support(eqs_ii, pm_ii, attacks, defenses),
            "U_A": U_A_ii.tolist(),
        },
    }


print("\n\n=== ANALYSIS UNDER BOTH UTILITY METRICS ===\n")
per_seed_results = []
for seed in SEEDS:
    if not os.path.exists(os.path.join(output_dir, f"seed_{seed}", "payoff_results.json")):
        print(f"Seed {seed}: payoff missing, skipping")
        continue
    r = compute_vopd_two_metrics(seed)
    per_seed_results.append(r)
    print(f"Seed {seed}:")
    print(f"  Metric (i):  VoPD={r['metric_i']['vopd']:.4f}, mixed={r['metric_i']['mixed']}, support={r['metric_i']['support']}")
    print(f"  Metric (ii): VoPD={r['metric_ii']['vopd']:.4f}, mixed={r['metric_ii']['mixed']}, support={r['metric_ii']['support']}")

vopds_i = [r["metric_i"]["vopd"] for r in per_seed_results]
vopds_ii = [r["metric_ii"]["vopd"] for r in per_seed_results]
mixed_i = sum(1 for r in per_seed_results if r["metric_i"]["mixed"])
mixed_ii = sum(1 for r in per_seed_results if r["metric_ii"]["mixed"])

print(f"\n=== Summary ({len(per_seed_results)} seeds) ===")
print(f"Metric (i):  mean VoPD = {np.mean(vopds_i):.4f} ± {np.std(vopds_i):.4f}, mixed-NE: {mixed_i}/{len(per_seed_results)}")
print(f"Metric (ii): mean VoPD = {np.mean(vopds_ii):.4f} ± {np.std(vopds_ii):.4f}, mixed-NE: {mixed_ii}/{len(per_seed_results)}")

# Pre-committed verdict
strong_positive = (mixed_i >= 3 and np.mean(vopds_i) > 0.05)
weak_positive = (mixed_i >= 1 or mixed_ii >= 2)
verdict = "STRONG positive" if strong_positive else ("WEAK positive" if weak_positive else "NEGATIVE")
print(f"\n=== VERDICT (pre-committed): {verdict} ===")
if strong_positive:
    print("Paper category change available: 'negative + positive FL characterization'")
elif weak_positive:
    print("Marginal improvement; pilot v2 strengthens stateless-FL section.")
else:
    print("Negative: complementarity does not emerge even at restored admission variance.")
    print("This strengthens the typicality argument (Part C).")

summary = {
    "fl_config": {"N": 20, "K": 4, "f": 0.4, "num_rounds": 50},
    "attacks": game_config.attacks,
    "defenses": game_config.defenses,
    "seeds": SEEDS,
    "per_seed": per_seed_results,
    "metric_i_mean_vopd": float(np.mean(vopds_i)) if vopds_i else None,
    "metric_i_std_vopd": float(np.std(vopds_i)) if vopds_i else None,
    "metric_i_mixed_count": mixed_i,
    "metric_ii_mean_vopd": float(np.mean(vopds_ii)) if vopds_ii else None,
    "metric_ii_std_vopd": float(np.std(vopds_ii)) if vopds_ii else None,
    "metric_ii_mixed_count": mixed_ii,
    "n_seeds": len(per_seed_results),
    "verdict": verdict,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {output_dir}/summary.json")
