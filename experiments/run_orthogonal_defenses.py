"""
Round 48 — pure-strategy cells for orthogonal-signal defense suite.

Adds FoolsGold (pairwise-similarity) and reputation (distance-from-consensus) to the
defense menu and runs the standard cells × attacks × seeds grid:
  defenses: foolsgold, reputation
  attacks:  no_attack, model_scaling, backdoor_pixel, dba
  seeds:    42-46 (5 seeds)

= 8 cells × 5 seeds = 40 runs, ~7 hours MPS at ~10 min/run.

Output: extends results/cifar10_10seeds/seed_{42..46}/per_seed_results.json
with new keys "{attack}_{defense}".

Per-seed checkpointing: writes after each cell so re-runs skip done work.
"""
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig, ExperimentConfig
from experiments.run_payoff_matrix import run_single_experiment

NEW_DEFENSES = ["foolsgold", "reputation"]
ATTACKS = ["no_attack", "model_scaling", "backdoor_pixel", "dba"]
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

ps_root = os.path.join(base_dir, "results", "cifar10_10seeds")


def has_cached(seed: int, attack: str, defense: str) -> bool:
    path = os.path.join(ps_root, f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    key = f"{attack}_{defense}"
    if key not in data:
        return False
    return any(e.get("seed") == seed for e in data[key])


def save_cell(seed: int, attack: str, defense: str, accuracy: float,
              worst_class_accuracy: float, asr: float) -> None:
    seed_dir = os.path.join(ps_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    path = os.path.join(seed_dir, "per_seed_results.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    key = f"{attack}_{defense}"
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "worst_class_accuracy": float(worst_class_accuracy),
        "attack_success_rate": float(asr),
    }
    existing = data.get(key, [])
    existing = [e for e in existing if e.get("seed") != seed]
    existing.append(entry)
    data[key] = existing
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


print(f"Round 48 orthogonal-defense PS sweep:")
print(f"  Defenses: {NEW_DEFENSES}")
print(f"  Attacks:  {ATTACKS}")
print(f"  Seeds:    {SEEDS}")
print(f"  Expected runs: {len(NEW_DEFENSES) * len(ATTACKS) * len(SEEDS)}\n")

t0 = time.time()
runs_done = 0

for seed in SEEDS:
    for defense in NEW_DEFENSES:
        for attack in ATTACKS:
            if has_cached(seed, attack, defense):
                print(f"  [skip] seed {seed} {attack}-{defense} cached", flush=True)
                continue
            t_run = time.time()
            exp_config = ExperimentConfig(
                dataset="cifar10", model="cifar_cnn",
                dirichlet_alpha=0.5, adversarial_fraction=ADV_FRACTION,
                num_trials=1, seed=seed, device="mps",
            )
            r = run_single_experiment(attack, defense, FL_CONFIG, exp_config)
            save_cell(seed, attack, defense, r["accuracy"],
                      r["worst_class_accuracy"], r["attack_success_rate"])
            runs_done += 1
            dt = time.time() - t_run
            print(f"  seed {seed} {attack}-{defense}: acc={r['accuracy']:.3f} "
                  f"ASR={r['attack_success_rate']:.3f} ({dt:.0f}s, "
                  f"total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)

print(f"\n=== ORTHOGONAL DEFENSE SWEEP COMPLETE ===")
print(f"  Total runs: {runs_done}")
print(f"  Wall time:  {(time.time()-t0)/60:.1f} min")
