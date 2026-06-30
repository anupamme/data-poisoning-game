"""
Compute 10-seed averaged payoff matrices and solve the game.
Outputs: LaTeX-formatted table rows for Tables 1, 2, appendix tables.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game_theory.game_solver import GameSolver
from game_theory.payoff_matrix import PayoffMatrix

SEEDS = list(range(42, 52))
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tenseed_dir = os.path.join(base_dir, "results", "cifar10_10seeds")

# Load attacks/defenses from first seed
attacks = defenses = None
for seed in SEEDS:
    payoff_path = os.path.join(tenseed_dir, f"seed_{seed}", "payoff_results.json")
    if os.path.exists(payoff_path):
        with open(payoff_path) as f:
            payoff = json.load(f)
        attack_set, defense_set = set(), set()
        for v in payoff.values():
            attack_set.add(v["attack"])
            defense_set.add(v["defense"])
        attacks = sorted(attack_set)
        defenses = sorted(defense_set)
        break

print(f"Attacks: {attacks}")
print(f"Defenses: {defenses}")

# Load per-seed data for all 10 seeds
mats = {}
for seed in SEEDS:
    seed_path = os.path.join(tenseed_dir, f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(seed_path):
        print(f"Missing seed {seed}, skipping")
        continue
    with open(seed_path) as f:
        per_seed = json.load(f)
    results = {}
    for a in attacks:
        for d in defenses:
            key = f"{a}_{d}"
            if key in per_seed:
                for e in per_seed[key]:
                    if e["seed"] == seed:
                        results[(a, d)] = e
                        break
    mats[seed] = results

print(f"Loaded {len(mats)} seeds")

# Compute 10-seed averages
avg_results = {}
std_acc = {}
for a in attacks:
    for d in defenses:
        key = (a, d)
        accs, asrs, worsts = [], [], []
        for seed in sorted(mats.keys()):
            e = mats[seed].get(key)
            if e:
                accs.append(e["accuracy"])
                asrs.append(e["attack_success_rate"])
                worsts.append(e["worst_class_accuracy"])
        if accs:
            avg_results[key] = {
                "accuracy": np.mean(accs),
                "attack_success_rate": np.mean(asrs),
                "worst_class_accuracy": np.mean(worsts),
            }
            std_acc[key] = np.std(accs)

# Build payoff matrix and solve
pm = PayoffMatrix.from_experiment_results(avg_results, attacks, defenses)
solver = GameSolver(pm)
equilibria = solver.solve_nash()

print(f"\n=== 10-seed averaged game equilibria ===")
for i, ne in enumerate(equilibria):
    v = ne.value_of_information(pm)
    adv_support = [(attacks[j], f"{p:.3f}") for j, p in enumerate(ne.adversary_strategy) if p > 0.01]
    srv_support = [(defenses[j], f"{p:.3f}") for j, p in enumerate(ne.server_strategy) if p > 0.01]
    mixed = (ne.adversary_strategy > 0.01).sum() > 1 or (ne.server_strategy > 0.01).sum() > 1
    print(f"  NE{i+1}: VoPD={v:.4f}, mixed={mixed}")
    print(f"    Adversary: {adv_support}")
    print(f"    Server:    {srv_support}")
    print(f"    U_A={ne.adversary_utility:.4f}, U_D={ne.server_utility:.4f}")

# Print accuracy table (Table 1 format) — 10-seed averages
print("\n=== Table 1: Clean accuracy and ASR (10-seed mean) ===")
print("Attack", end="")
for d in defenses:
    print(f" & {d[:6]}", end="")
print(" \\\\")

clean_attacks = [a for a in attacks if a == "no_attack"] + [a for a in attacks if a != "no_attack"]
for a in attacks:
    row = avg_results.get((a, defenses[0]))
    if row is None:
        continue
    print(f"  {a:<20}", end="")
    for d in defenses:
        r = avg_results.get((a, d))
        if r:
            print(f" & {r['accuracy']:.2f}", end="")
        else:
            print(" & --", end="")
    print(" \\\\")

print("\n--- ASR ---")
backdoor_attacks = [a for a in attacks if a in ["backdoor_pixel", "model_scaling", "dba", "DBA"]]
for a in backdoor_attacks:
    print(f"  {a:<20}", end="")
    for d in defenses:
        r = avg_results.get((a, d))
        if r:
            print(f" & {r['attack_success_rate']:.2f}", end="")
        else:
            print(" & --", end="")
    print(" \\\\")

# Print adversary payoff matrix
print("\n=== Adversary payoff matrix (10-seed mean) ===")
for a in attacks:
    print(f"  {a:<20}", end="")
    for d in defenses:
        r = avg_results.get((a, d))
        if r:
            print(f" & {pm.adversary_payoffs[attacks.index(a), defenses.index(d)]:+.3f}", end="")
        else:
            print(" & --", end="")
    print(" \\\\")

# Print server payoff matrix
print("\n=== Server payoff matrix (10-seed mean) ===")
for a in attacks:
    print(f"  {a:<20}", end="")
    for d in defenses:
        r = avg_results.get((a, d))
        if r:
            print(f" & {pm.server_payoff[attacks.index(a), defenses.index(d)]:.3f}", end="")
        else:
            print(" & --", end="")
    print(" \\\\")

# Std check
print("\n=== Per-entry accuracy std (10 seeds) ===")
for a in attacks:
    for d in defenses:
        s = std_acc.get((a, d), 0)
        if s > 0.05:
            print(f"  {a} vs {d}: std={s:.3f}")

# Save output
out = {
    "attacks": attacks,
    "defenses": defenses,
    "avg_results": {f"{a}_{d}": avg_results[(a, d)] for (a, d) in avg_results},
    "equilibria": [
        {
            "adversary_strategy": ne.adversary_strategy.tolist(),
            "server_strategy": ne.server_strategy.tolist(),
            "adversary_utility": float(ne.adversary_utility),
            "server_utility": float(ne.server_utility),
            "vopd": float(ne.value_of_information(pm)),
        }
        for ne in equilibria
    ],
    "payoff_matrix": {
        "adversary": pm.adversary_payoffs.tolist(),
        "server": pm.server_payoff.tolist(),
    },
}
out_path = os.path.join(tenseed_dir, "averaged_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
