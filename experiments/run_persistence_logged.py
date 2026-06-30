"""
Per-round ASR logging for the model_scaling attack under three defense conditions.

This produces the data Theorem 3 (persistence-aware VoPD) fits against.

Configurations:
    (A) model_scaling vs. pure FedAvg          → maximally admitting (p=1.0)
    (B) model_scaling vs. pure NormClip τ=5    → minimally admitting (p=0.0)
    (C) model_scaling vs. NE3 mix              → p=0.26 FedAvg, p=0.74 NormClip per round

Output: results/persistence/{condition}/seed_{seed}/asr_timeline.json
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig
from experiments.run_payoff_matrix import run_single_experiment

SEEDS = [42, 43, 44, 45, 46]
LOG_EVERY = 1  # log per-round ASR every round

fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_base = os.path.join(base_dir, "results", "persistence")

CONFIGS = [
    {"name": "fedavg_pure",     "attack": "model_scaling", "defense": "fedavg",    "p_admit": 1.0,  "schedule": None},
    {"name": "normclip_pure",   "attack": "model_scaling", "defense": "norm_clip", "p_admit": 0.0,  "schedule": None},
    {"name": "ne3_mix",         "attack": "model_scaling", "defense": "fedavg",    "p_admit": 0.26, "schedule": "ne3"},
]


def make_ne3_schedule(num_rounds, seed):
    """NE3 policy: each round independently FedAvg with prob 0.26, NormClip with prob 0.74."""
    rng = np.random.default_rng(seed)
    draws = rng.uniform(0, 1, num_rounds)
    return ["fedavg" if d < 0.26 else "norm_clip" for d in draws]


print(f"Persistence logging: {len(CONFIGS)} configs × {len(SEEDS)} seeds, {fl_config.num_rounds} rounds each")
print(f"Output base: {out_base}\n")

for cfg in CONFIGS:
    print(f"\n=== {cfg['name']} (attack={cfg['attack']}, defense={cfg['defense']}, p_admit={cfg['p_admit']}) ===")
    cfg_dir = os.path.join(out_base, cfg["name"])
    os.makedirs(cfg_dir, exist_ok=True)

    for seed in SEEDS:
        seed_dir = os.path.join(cfg_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        out_path = os.path.join(seed_dir, "asr_timeline.json")

        if os.path.exists(out_path):
            print(f"  seed {seed}: already complete, skipping")
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

        schedule = make_ne3_schedule(fl_config.num_rounds, seed) if cfg["schedule"] == "ne3" else None
        print(f"  --- seed {seed} ---")
        result = run_single_experiment(
            cfg["attack"], cfg["defense"],
            fl_config, exp_config,
            norm_clip_tau=5.0,
            log_every=LOG_EVERY,
            defense_schedule=schedule,
        )

        record = {
            "condition": cfg["name"],
            "seed": seed,
            "p_admit": cfg["p_admit"],
            "schedule": schedule,
            "final_asr": result["attack_success_rate"],
            "final_accuracy": result["accuracy"],
            "asr_timeline": result["asr_timeline"],
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  saved: {out_path}, final_asr={result['attack_success_rate']:.3f}")

# Summary
print("\n=== SUMMARY ===")
for cfg in CONFIGS:
    final_asrs = []
    for seed in SEEDS:
        path = os.path.join(out_base, cfg["name"], f"seed_{seed}", "asr_timeline.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            final_asrs.append(d["final_asr"])
    if final_asrs:
        print(f"{cfg['name']:>15} (p={cfg['p_admit']:.2f}): final ASR mean={np.mean(final_asrs):.3f}±{np.std(final_asrs):.3f}, n={len(final_asrs)}")
