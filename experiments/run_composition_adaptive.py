"""
Adaptive attack (projected backdoor) vs rep+tm composition.

Addresses reviewer concern: composition is the paper's affirmative recommendation,
but tested only against committed/fixed adversaries. A Shejwalkar-Houmansadr-style
adaptive adversary that constrains its updates to pass both filters could break it.

Attack: ProjectedBackdoorAttack
  - Per round, adversarial client trains on CLEAN data -> honest_update
  - Trains on POISONED data -> poisoned_update
  - Projects poisoned_update onto epsilon-ball around honest_update
  - This bounds L2 distance from consensus (preserving reputation weight)
    and limits per-coordinate deviation (reducing trimmed_mean clipping)

Sweep epsilon in [0.5, 1.0, 2.0, 3.0, 5.0] at 5 seeds each.
Then scale to 30 seeds at the best epsilon.

Defense: reputation_then_trimmed_mean composition (same as run_composition_rep_tm.py)
FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn

Output: results/composition_adaptive/summary.json
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
from attacks import get_attack
from attacks.attack_strategies import ProjectedBackdoorAttack, PoisonedDataset, backdoor_pixel_fn
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_composition_rep_tm import compose_reputation_trimmed_mean

EPSILONS = [0.5, 1.0, 2.0, 3.0, 5.0]
SEEDS = list(range(42, 47))  # 5-seed pilot; extend to range(42, 72) after selecting best epsilon
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "composition_adaptive")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "method": "projected_backdoor_vs_composition",
        "description": "Adaptive projected-backdoor attack against rep+tm composition",
        "epsilons": {},
    }


def has_run(eps, seed):
    s = load_or_init()
    eps_key = f"eps_{eps}"
    if eps_key not in s["epsilons"]:
        return False
    return any(r["seed"] == seed for r in s["epsilons"][eps_key]["per_seed"])


def save_one(eps, seed, accuracy, asr, proj_ratio_mean):
    s = load_or_init()
    eps_key = f"eps_{eps}"
    if eps_key not in s["epsilons"]:
        s["epsilons"][eps_key] = {"epsilon": eps, "per_seed": []}
    runs = s["epsilons"][eps_key]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_final": float(asr),
        "proj_ratio_mean": float(proj_ratio_mean),
    })
    s["epsilons"][eps_key]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, epsilon):
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

    attack = ProjectedBackdoorAttack(epsilon=epsilon)

    # Pre-poison adversarial datasets
    clean_datasets = {}
    poisoned_datasets = {}
    for i in range(FL_CONFIG.num_clients):
        clean_datasets[i] = client_datasets[i]
        if i in adv_ids:
            poisoned_datasets[i] = attack.poison_dataset(client_datasets[i])

    current_lr = FL_CONFIG.learning_rate
    proj_ratios = []

    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        updates = []
        for cid in participant_ids:
            if cid in adv_ids:
                # Two-pass training for adaptive attack
                # Pass 1: train on clean data -> honest_update
                clean_client = FederatedClient(cid, clean_datasets[cid], device)
                honest_update = clean_client.train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )

                # Pass 2: train on poisoned data -> poisoned_update
                poison_client = FederatedClient(cid, poisoned_datasets[cid], device)
                poisoned_update = poison_client.train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )

                # Project poisoned onto epsilon-ball around honest
                attack.set_honest_update(honest_update)
                projected_update = attack.manipulate_update(poisoned_update, server.global_model)

                # Track projection ratio (how much we had to clip)
                keys = list(poisoned_update.keys())
                diff = torch.cat([(poisoned_update[k] - honest_update[k]).flatten() for k in keys])
                honest_norm = torch.cat([honest_update[k].flatten() for k in keys]).norm().item()
                radius = epsilon * max(honest_norm, 1e-8)
                ratio = min(1.0, radius / max(diff.norm().item(), 1e-8))
                proj_ratios.append(ratio)

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

    # Evaluate
    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    proj_ratio_mean = float(np.mean(proj_ratios)) if proj_ratios else 0.0

    return float(eval_result["accuracy"]), float(asr), proj_ratio_mean


print("=== Adaptive Attack (Projected Backdoor) vs Rep+TM Composition ===")
print(f"  Epsilons: {EPSILONS}")
print(f"  Seeds: {SEEDS}")
print(f"  Comparison: committed_pixel ASR=0.679, oracle ASR=0.316 (n=30)")
print(f"  Two-pass training: clean (honest ref) + poisoned (backdoor signal)\n")

t0 = time.time()
runs_done = 0

for eps in EPSILONS:
    for seed in SEEDS:
        if has_run(eps, seed):
            print(f"  [skip] eps={eps} seed {seed}", flush=True)
            continue
        t_run = time.time()
        acc, asr, proj_ratio = run_one(seed, eps)
        save_one(eps, seed, acc, asr, proj_ratio)
        runs_done += 1
        dt = time.time() - t_run
        print(f"  eps={eps} seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
              f"proj_ratio={proj_ratio:.3f} ({dt:.0f}s)", flush=True)

print(f"\n=== ADAPTIVE ATTACK RESULTS ===")
s = load_or_init()
best_eps = None
best_asr = -1
for eps_key, data in sorted(s["epsilons"].items()):
    runs = data["per_seed"]
    if runs:
        asrs = [r["asr_final"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        proj_ratios = [r["proj_ratio_mean"] for r in runs]
        mean_asr = np.mean(asrs)
        print(f"  eps={data['epsilon']}: ASR={mean_asr:.3f}±{np.std(asrs):.3f}, "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
              f"proj_ratio={np.mean(proj_ratios):.3f}")
        if mean_asr > best_asr:
            best_asr = mean_asr
            best_eps = data["epsilon"]

print(f"\n  Best epsilon: {best_eps} (ASR={best_asr:.3f})")
print(f"  Committed-pixel baseline: 0.679 ± 0.118")
print(f"  Oracle baseline: 0.316 ± 0.162")
print(f"  Improvement over oracle: {best_asr - 0.316:+.3f}")
print(f"  Improvement over committed-pixel: {best_asr - 0.679:+.3f}")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
