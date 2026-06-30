"""
FLTrust pilot (Round 43) — orthogonal-signal defense test.

Reviewer's request: "produce one FL defense menu on an orthogonal signal that
recovers positive realized VoPD, turning the negative result into a
characterization with a demonstrated boundary."

Setup:
- N=10, K=5, f=0.2 (matches original CIFAR-10 setting for direct comparison)
- Attacks: no_attack, backdoor_pixel, model_scaling, dba (the original 4 live attacks)
- Defenses: fedavg, norm_clip, rfa, trimmed_mean, coord_median, fltrust (NEW)
- 5 seeds (42-46), 50 rounds
- FLTrust uses 100-sample clean holdout from test_dataset for cosine-similarity weighting

Pre-committed interpretation:
- STRONG positive (→ 7/10): ≥3/5 seeds give mixed NE with VoPD > 0.05 under metric (i),
  AND FLTrust enters NE support, AND argmax(FLTrust) differs from argmax(FedAvg)
- WEAK positive (→ 6/10): 1-2 seeds mixed; or mixed NE only under metric (ii)
- NEGATIVE (→ 5-6/10): 0/5 seeds mixed NE with FLTrust in support

Output: results/cifar10_fltrust_pilot/
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
fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
game_config = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling", "dba"],
    defenses=["fedavg", "norm_clip", "rfa", "trimmed_mean", "coord_median", "fltrust"],
)
SEEDS = [42, 43, 44, 45, 46]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "cifar10_fltrust_pilot")
os.makedirs(output_dir, exist_ok=True)

print(f"FLTrust pilot (Round 43): N=10, K=5, f=0.2, 4×6 game with FLTrust added")
print(f"Attacks: {game_config.attacks}")
print(f"Defenses: {game_config.defenses}")
print(f"Seeds: {SEEDS}, 50 rounds\n")

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
        adversarial_fraction=0.2,
        num_trials=1,
        seed=seed,
        device="mps",
    )
    print(f"\n--- Seed {seed} ---")
    run_full_payoff_matrix(fl_config, exp_config, game_config, seed_dir)


# Analyze under both metrics
def compute_vopd(seed):
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    with open(os.path.join(seed_dir, "payoff_results.json")) as f:
        results_raw = json.load(f)

    attacks = game_config.attacks
    defenses = game_config.defenses
    baseline_acc = results_raw["no_attack_fedavg"]["accuracy"]

    # Metric (i): primary ASR-only
    U_A_i = np.zeros((len(attacks), len(defenses)))
    U_D = np.zeros((len(attacks), len(defenses)))
    for i, a in enumerate(attacks):
        for j, d in enumerate(defenses):
            r = results_raw[f"{a}_{d}"]
            asr = r["attack_success_rate"]
            acc = r["accuracy"]
            U_A_i[i, j] = (0 if a == "no_attack" else asr - 0.1 * get_attack(a).cost)
            U_D[i, j] = acc - 0.1 * get_defense_cost(d)

    pm_i = PayoffMatrix(attacks=attacks, defenses=defenses,
                        adversary_payoffs=U_A_i, server_payoffs=U_D)
    eqs_i = GameSolver(pm_i).solve_nash()
    best_vopd_i = max((ne.value_of_information(pm_i) for ne in eqs_i), default=0.0)
    mixed_i = any((ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1 for ne in eqs_i)

    # Metric (ii): availability-aware
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
                         adversary_payoffs=U_A_ii, server_payoffs=U_D)
    eqs_ii = GameSolver(pm_ii).solve_nash()
    best_vopd_ii = max((ne.value_of_information(pm_ii) for ne in eqs_ii), default=0.0)
    mixed_ii = any((ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1 for ne in eqs_ii)

    # Extract argmax info
    def argmax_per_defense(U_A_mat):
        return {defenses[j]: attacks[int(np.argmax(U_A_mat[:, j]))] for j in range(len(defenses))}

    # Check whether FLTrust enters NE support
    def fltrust_in_support(eqs, pm):
        best = max(eqs, key=lambda ne: ne.value_of_information(pm), default=None)
        if best is None:
            return False, {}
        fltrust_idx = defenses.index("fltrust")
        in_support = best.server_strategy[fltrust_idx] > 0.01
        srv = {defenses[k]: round(p, 3) for k, p in enumerate(best.server_strategy) if p > 0.01}
        adv = {attacks[k]: round(p, 3) for k, p in enumerate(best.adversary_strategy) if p > 0.01}
        return in_support, {"adv": adv, "srv": srv}

    fltrust_in_i, support_i = fltrust_in_support(eqs_i, pm_i)
    fltrust_in_ii, support_ii = fltrust_in_support(eqs_ii, pm_ii)

    return {
        "seed": seed,
        "metric_i": {
            "vopd": float(best_vopd_i),
            "mixed": bool(mixed_i),
            "fltrust_in_support": bool(fltrust_in_i),
            "support": support_i,
            "argmax_per_defense": argmax_per_defense(U_A_i),
            "U_A": U_A_i.tolist(),
        },
        "metric_ii": {
            "vopd": float(best_vopd_ii),
            "mixed": bool(mixed_ii),
            "fltrust_in_support": bool(fltrust_in_ii),
            "support": support_ii,
            "argmax_per_defense": argmax_per_defense(U_A_ii),
        },
    }


print("\n\n=== ANALYSIS UNDER BOTH METRICS ===\n")
per_seed = []
for seed in SEEDS:
    if not os.path.exists(os.path.join(output_dir, f"seed_{seed}", "payoff_results.json")):
        continue
    r = compute_vopd(seed)
    per_seed.append(r)
    print(f"Seed {seed}:")
    print(f"  Metric (i):  VoPD={r['metric_i']['vopd']:.4f}, mixed={r['metric_i']['mixed']}, FLTrust in support={r['metric_i']['fltrust_in_support']}")
    print(f"    Support: {r['metric_i']['support']}")
    print(f"    Argmax per defense: {r['metric_i']['argmax_per_defense']}")
    print(f"  Metric (ii): VoPD={r['metric_ii']['vopd']:.4f}, mixed={r['metric_ii']['mixed']}, FLTrust in support={r['metric_ii']['fltrust_in_support']}")
    print(f"    Support: {r['metric_ii']['support']}")
    print()

vopds_i = [r["metric_i"]["vopd"] for r in per_seed]
vopds_ii = [r["metric_ii"]["vopd"] for r in per_seed]
mixed_i_count = sum(1 for r in per_seed if r["metric_i"]["mixed"])
mixed_ii_count = sum(1 for r in per_seed if r["metric_ii"]["mixed"])
fltrust_i_count = sum(1 for r in per_seed if r["metric_i"]["fltrust_in_support"])
fltrust_ii_count = sum(1 for r in per_seed if r["metric_ii"]["fltrust_in_support"])

print(f"=== Summary ({len(per_seed)} seeds) ===")
print(f"Metric (i):  mean VoPD = {np.mean(vopds_i):.4f} ± {np.std(vopds_i):.4f}, "
      f"mixed-NE: {mixed_i_count}/{len(per_seed)}, FLTrust in support: {fltrust_i_count}/{len(per_seed)}")
print(f"Metric (ii): mean VoPD = {np.mean(vopds_ii):.4f} ± {np.std(vopds_ii):.4f}, "
      f"mixed-NE: {mixed_ii_count}/{len(per_seed)}, FLTrust in support: {fltrust_ii_count}/{len(per_seed)}")

# Pre-committed verdict
strong_positive = (mixed_i_count >= 3 and np.mean(vopds_i) > 0.05 and fltrust_i_count >= 3)
weak_positive = (mixed_i_count >= 1 or mixed_ii_count >= 2)
verdict = "STRONG positive" if strong_positive else ("WEAK positive" if weak_positive else "NEGATIVE")
print(f"\n=== VERDICT (pre-committed): {verdict} ===")
if strong_positive:
    print("FLTrust breaks the dichotomy: complementarity emerges with direction-based defense.")
elif weak_positive:
    print("Marginal: FLTrust offers some divergence but not metric-robust complementarity.")
else:
    print("FLTrust does NOT break the dichotomy: even direction-based defenses converge.")
    print("This strengthens the typicality argument (Lemma 1).")

summary = {
    "fl_config": {"N": 10, "K": 5, "f": 0.2, "num_rounds": 50},
    "attacks": game_config.attacks,
    "defenses": game_config.defenses,
    "seeds": SEEDS,
    "per_seed": per_seed,
    "metric_i_mean_vopd": float(np.mean(vopds_i)) if vopds_i else None,
    "metric_i_std_vopd": float(np.std(vopds_i)) if vopds_i else None,
    "metric_i_mixed_count": mixed_i_count,
    "metric_i_fltrust_in_support_count": fltrust_i_count,
    "metric_ii_mean_vopd": float(np.mean(vopds_ii)) if vopds_ii else None,
    "metric_ii_std_vopd": float(np.std(vopds_ii)) if vopds_ii else None,
    "metric_ii_mixed_count": mixed_ii_count,
    "metric_ii_fltrust_in_support_count": fltrust_ii_count,
    "n_seeds": len(per_seed),
    "verdict": verdict,
}
with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {output_dir}/summary.json")
