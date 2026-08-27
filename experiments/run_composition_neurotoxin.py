"""
Neurotoxin-style adaptive attack vs rep+tm composition.

Addresses reviewer's request for an adaptive attack on the composition defense.
Neurotoxin (Zhang et al., 2022): projects the malicious update onto the top-k
coordinates of the running gradient mean, making the backdoor reside in the same
subspace as benign training.

We test this against rep+tm composition at 3 top_frac values (0.05, 0.10, 0.20)
and 2 scale_factors (1.0, 2.0), 5 seeds each.

Comparison baselines (from run_composition_rep_tm.py, n=30):
  - committed_pixel: ASR = 0.679 ± 0.118
  - oracle: ASR = 0.316 ± 0.162

Output: results/composition_neurotoxin/summary.json
"""
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

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks.attack_strategies import NeurotoxinAttack, PoisonedDataset, backdoor_pixel_fn
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_composition_rep_tm import compose_reputation_trimmed_mean

CONFIGS = [
    {"top_frac": 0.05, "scale_factor": 1.0},
    {"top_frac": 0.10, "scale_factor": 1.0},
    {"top_frac": 0.20, "scale_factor": 1.0},
    {"top_frac": 0.10, "scale_factor": 2.0},
    {"top_frac": 0.20, "scale_factor": 2.0},
]
SEEDS = list(range(42, 47))  # 5-seed pilot
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "composition_neurotoxin")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def config_key(cfg):
    return f"top{cfg['top_frac']}_scale{cfg['scale_factor']}"


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"method": "neurotoxin_vs_composition", "configs": {}}


def has_run(cfg, seed):
    s = load_or_init()
    key = config_key(cfg)
    if key not in s["configs"]:
        return False
    return any(r["seed"] == seed for r in s["configs"][key]["per_seed"])


def save_one(cfg, seed, accuracy, asr):
    s = load_or_init()
    key = config_key(cfg)
    if key not in s["configs"]:
        s["configs"][key] = {"top_frac": cfg["top_frac"],
                             "scale_factor": cfg["scale_factor"], "per_seed": []}
    runs = s["configs"][key]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({"seed": seed, "accuracy": float(accuracy), "asr_final": float(asr)})
    s["configs"][key]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack = NeurotoxinAttack(top_frac=cfg["top_frac"],
                              scale_factor=cfg["scale_factor"])

    clean_datasets = {}
    poisoned_datasets = {}
    for i in range(FL_CONFIG.num_clients):
        clean_datasets[i] = client_datasets[i]
        if i in adv_ids:
            poisoned_datasets[i] = attack.poison_dataset(client_datasets[i])

    current_lr = FL_CONFIG.learning_rate

    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        updates = []
        for cid in participant_ids:
            if cid in adv_ids:
                poison_client = FederatedClient(cid, poisoned_datasets[cid], device)
                poisoned_update = poison_client.train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )
                projected_update = attack.manipulate_update(poisoned_update, server.global_model)
                updates.append(projected_update)
            else:
                client = FederatedClient(cid, clean_datasets[cid], device)
                update = client.train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )
                updates.append(update)

        aggregated = compose_reputation_trimmed_mean(server, updates)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


print("=== Neurotoxin Attack vs Rep+TM Composition ===")
print(f"  Configs: {CONFIGS}")
print(f"  Seeds: {SEEDS}")
print(f"  Baselines: committed_pixel=0.679, oracle=0.316\n")

t0 = time.time()
runs_done = 0

for cfg in CONFIGS:
    for seed in SEEDS:
        if has_run(cfg, seed):
            print(f"  [skip] {config_key(cfg)} seed {seed}", flush=True)
            continue
        t_run = time.time()
        acc, asr = run_one(seed, cfg)
        save_one(cfg, seed, acc, asr)
        runs_done += 1
        dt = time.time() - t_run
        print(f"  {config_key(cfg)} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)",
              flush=True)

print(f"\n=== NEUROTOXIN RESULTS ===")
s = load_or_init()
best_key = None
best_asr = -1
for key, data in sorted(s["configs"].items()):
    runs = data["per_seed"]
    if runs:
        asrs = [r["asr_final"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        mean_asr = np.mean(asrs)
        print(f"  {key}: ASR={mean_asr:.3f}±{np.std(asrs):.3f}, "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f} (n={len(runs)})")
        if mean_asr > best_asr:
            best_asr = mean_asr
            best_key = key

print(f"\n  Best config: {best_key} (ASR={best_asr:.3f})")
print(f"  Does Neurotoxin beat composition?")
print(f"    vs committed-pixel (0.679): {'YES' if best_asr > 0.679 else 'NO'} ({best_asr - 0.679:+.3f})")
print(f"    vs oracle (0.316): {'YES' if best_asr > 0.316 else 'NO'} ({best_asr - 0.316:+.3f})")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
