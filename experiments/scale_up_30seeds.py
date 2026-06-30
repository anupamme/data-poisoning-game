"""
Round 46 — Scale headline NE3 realized-VoPD result from 10 seeds to 30.

The current headline -0.0001 +/- 0.083 uses 10 cached pure-strategy seeds (42-51)
matched with 10 NE3 BR realized ASRs. We have 15-seed BR data (42-56) but PS only
for 42-51. To extend to 30 matched seeds we need:

  (a) PS baselines for seeds 52-71 (20 new) on the 4 binding cells:
      model_scaling x {fedavg, norm_clip} and backdoor_pixel x {fedavg, norm_clip}
  (b) NE3 BR realized ASR for seeds 57-71 (15 new) on attacks
      {model_scaling, backdoor_pixel} (drop no_attack -- not needed for VoPD)

Per-seed checkpointing: writes after each cell so re-running skips done work.

Output:
  results/cifar10_10seeds/seed_{52..71}/per_seed_results.json   (PS schema match)
  results/randomized_defense/summary_extended.json              (BR for new seeds)
"""
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig, ExperimentConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor, run_single_experiment

# ── Configuration ──────────────────────────────────────────────────────────────
NEW_SEEDS_PS = list(range(52, 72))   # 20 new seeds for PS baselines
NEW_SEEDS_BR = list(range(57, 72))   # 15 new seeds for BR (52-56 already in 15-seed summary)
BINDING_CELLS = [
    ("model_scaling", "fedavg"),
    ("model_scaling", "norm_clip"),
    ("backdoor_pixel", "fedavg"),
    ("backdoor_pixel", "norm_clip"),
]
BR_ATTACKS = ["model_scaling", "backdoor_pixel"]  # drop no_attack for compute economy
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

results_dir = os.path.join(base_dir, "results")
ps_root = os.path.join(results_dir, "cifar10_10seeds")
br_path = os.path.join(results_dir, "randomized_defense", "summary_extended.json")
os.makedirs(os.path.dirname(br_path), exist_ok=True)


def has_cached_ps(seed: int, attack: str, defense: str) -> bool:
    path = os.path.join(ps_root, f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        data = json.load(f)
    key = f"{attack}_{defense}"
    if key not in data:
        return False
    return any(e.get("seed") == seed for e in data[key])


def save_ps_cell(seed: int, attack: str, defense: str, accuracy: float,
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


def run_ne3_br(seed: int, attack_name: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    exp_config = ExperimentConfig(
        dataset="cifar10", model="cifar_cnn",
        dirichlet_alpha=0.5, adversarial_fraction=ADV_FRACTION,
        num_trials=1, seed=seed, device="mps",
    )
    client_datasets, test_dataset, num_classes = get_federated_dataset(
        exp_config.dataset, FL_CONFIG.num_clients, exp_config.dirichlet_alpha, seed
    )
    model = get_model(exp_config.model, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))
    attack = get_attack(attack_name)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 1000)
    current_lr = FL_CONFIG.learning_rate

    fedavg_count = 0
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)
        d_this = rng.choice(defenses, p=probs)
        if d_this == "fedavg":
            fedavg_count += 1
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "attack_success_rate": float(asr),
        "fedavg_rounds": int(fedavg_count),
    }


def load_br_summary() -> dict:
    if os.path.exists(br_path):
        with open(br_path) as f:
            return json.load(f)
    return {
        "policy": DEFENSE_DIST,
        "seeds": [],
        "attacks": {a: {"realized_asr": [], "realized_accuracy": [],
                         "per_seed": []}
                    for a in BR_ATTACKS},
    }


def save_br_result(seed: int, attack: str, accuracy: float, asr: float) -> None:
    s = load_br_summary()
    a = s["attacks"].setdefault(attack, {"realized_asr": [], "realized_accuracy": [],
                                          "per_seed": []})
    # Remove any prior entry for this seed
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    a["per_seed"].append({"seed": seed, "accuracy": float(accuracy),
                           "attack_success_rate": float(asr)})
    a["realized_asr"] = [e["attack_success_rate"] for e in a["per_seed"]]
    a["realized_accuracy"] = [e["accuracy"] for e in a["per_seed"]]
    if seed not in s["seeds"]:
        s["seeds"].append(seed)
    with open(br_path, "w") as f:
        json.dump(s, f, indent=2)


# ── Execute ────────────────────────────────────────────────────────────────────
print(f"Round 46 scale-up: PS for seeds {NEW_SEEDS_PS[0]}-{NEW_SEEDS_PS[-1]} "
      f"({len(NEW_SEEDS_PS)} seeds, {len(BINDING_CELLS)} cells each), "
      f"BR for seeds {NEW_SEEDS_BR[0]}-{NEW_SEEDS_BR[-1]} ({len(NEW_SEEDS_BR)} seeds, "
      f"{len(BR_ATTACKS)} attacks each)")
print(f"Expected total runs: {len(NEW_SEEDS_PS) * len(BINDING_CELLS) + len(NEW_SEEDS_BR) * len(BR_ATTACKS)}")
print(f"Output:\n  PS: {ps_root}/seed_{{52..71}}/per_seed_results.json\n  BR: {br_path}\n")

t0 = time.time()
runs_done = 0

# Phase 1: PS binding cells
for seed in NEW_SEEDS_PS:
    for attack, defense in BINDING_CELLS:
        if has_cached_ps(seed, attack, defense):
            print(f"  [skip] seed {seed} {attack}-{defense} cached", flush=True)
            continue
        t_run = time.time()
        exp_config = ExperimentConfig(
            dataset="cifar10", model="cifar_cnn",
            dirichlet_alpha=0.5, adversarial_fraction=ADV_FRACTION,
            num_trials=1, seed=seed, device="mps",
        )
        result = run_single_experiment(attack, defense, FL_CONFIG, exp_config)
        save_ps_cell(seed, attack, defense, result["accuracy"],
                     result["worst_class_accuracy"], result["attack_success_rate"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {attack}-{defense}: acc={result['accuracy']:.3f} "
              f"ASR={result['attack_success_rate']:.3f}  ({dt:.0f}s, "
              f"total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)

print(f"\nPS phase done in {(time.time()-t0)/60:.1f} min.")

# Phase 2: BR extension for new seeds
t1 = time.time()
for seed in NEW_SEEDS_BR:
    for attack in BR_ATTACKS:
        # Check if already saved
        s = load_br_summary()
        if any(e["seed"] == seed for e in s["attacks"].get(attack, {}).get("per_seed", [])):
            print(f"  [skip] BR seed {seed} {attack} cached", flush=True)
            continue
        t_run = time.time()
        r = run_ne3_br(seed, attack)
        save_br_result(seed, attack, r["accuracy"], r["attack_success_rate"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  BR seed {seed} {attack}: acc={r['accuracy']:.3f} "
              f"ASR={r['attack_success_rate']:.3f}  ({dt:.0f}s, "
              f"total {(time.time()-t1)/60:.1f}min phase2)", flush=True)

print(f"\nBR phase done in {(time.time()-t1)/60:.1f} min.")
print(f"\n=== SCALE-UP COMPLETE ===")
print(f"  Total runs: {runs_done}")
print(f"  Wall time: {(time.time()-t0)/60:.1f} min")
print(f"  PS dir: {ps_root}")
print(f"  BR extended: {br_path}")
