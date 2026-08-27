"""
FG→RFA Flagship Composition — Full validation at n=30 + adaptive attack suite.

The composability criterion identifies FoolsGold→RFA as a deployable defense stack
(max ASR 0.045 at n=3). This script validates with:
  1. Base composition at n=30 (committed_scaling, committed_pixel, oracle)
  2. Projection-based adaptive attack (5 epsilons × 5 seeds)
  3. Neurotoxin-style top-k attack (5 configs × 5 seeds)
  4. Consensus-shift attack (2 rates × 5 seeds)

FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn

Output: results/fg_rfa_flagship/summary.json
"""
import json
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import apply_d1_transform, generic_compose

FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
BASE_SEEDS = list(range(42, 72))  # 30 seeds
ADAPTIVE_SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for adaptive attacks

output_dir = os.path.join(base_dir, "results", "fg_rfa_flagship")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "FG→RFA flagship composition validation",
        "base_composition": {
            "committed_scaling": {"per_seed": []},
            "committed_pixel": {"per_seed": []},
            "oracle": {"per_seed": []},
        },
        "adaptive_projection": {},
        "adaptive_neurotoxin": {},
        "adaptive_consensus_shift": {},
    }


def has_run(section, key, seed):
    s = load_or_init()
    if section not in s:
        return False
    if key not in s[section]:
        return False
    return any(r["seed"] == seed for r in s[section][key].get("per_seed", []))


def save_run(section, key, seed, accuracy, asr, extra=None):
    s = load_or_init()
    if section not in s:
        s[section] = {}
    if key not in s[section]:
        s[section][key] = {"per_seed": []}
    runs = s[section][key]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    entry = {"seed": seed, "accuracy": float(accuracy), "asr": float(asr)}
    if extra:
        entry.update(extra)
    runs.append(entry)
    s[section][key]["per_seed"] = runs
    asrs = [r["asr"] for r in runs]
    s[section][key]["mean_asr"] = float(np.mean(asrs))
    s[section][key]["std_asr"] = float(np.std(asrs))
    s[section][key]["n"] = len(runs)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Base FG→RFA composition ---
def run_base(seed, adv_policy):
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

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    clients_scaling = []
    clients_pixel = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            clients_scaling.append(FederatedClient(i, attack_scaling.poison_dataset(ds), device))
            clients_pixel.append(FederatedClient(i, attack_pixel.poison_dataset(ds), device))
        else:
            clients_scaling.append(FederatedClient(i, ds, device))
            clients_pixel.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
        if adv_policy == "committed_scaling":
            attack_obj, clients = attack_scaling, clients_scaling
        elif adv_policy == "committed_pixel":
            attack_obj, clients = attack_pixel, clients_pixel
        elif adv_policy == "oracle":
            if r % 2 == 0:
                attack_obj, clients = attack_scaling, clients_scaling
            else:
                attack_obj, clients = attack_pixel, clients_pixel
        else:
            raise ValueError(f"Unknown policy: {adv_policy}")

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
            if cid in adv_ids:
                update = attack_obj.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, "foolsgold", "rfa", tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Adaptive: Projection-based ---
def run_projection_adaptive(seed, epsilon):
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

    attack = get_attack("backdoor_pixel")
    clean_datasets = [client_datasets[i] for i in range(FL_CONFIG.num_clients)]
    poisoned_datasets = []
    for i in range(FL_CONFIG.num_clients):
        if i in adv_ids:
            poisoned_datasets.append(attack.poison_dataset(client_datasets[i]))
        else:
            poisoned_datasets.append(client_datasets[i])

    clients_clean = [FederatedClient(i, clean_datasets[i], device) for i in range(FL_CONFIG.num_clients)]
    clients_poisoned = [FederatedClient(i, poisoned_datasets[i], device) for i in range(FL_CONFIG.num_clients)]

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            if cid in adv_ids:
                honest_update = clients_clean[cid].train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )
                poisoned_update = clients_poisoned[cid].train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )
                projected = project_update(poisoned_update, honest_update, epsilon)
                updates.append(projected)
            else:
                update = clients_clean[cid].train(
                    server.global_model, FL_CONFIG.local_epochs,
                    current_lr, FL_CONFIG.local_batch_size
                )
                updates.append(update)

        aggregated = generic_compose(server, updates, "foolsgold", "rfa", tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


def project_update(poisoned, honest, epsilon):
    keys = list(poisoned.keys())
    p_flat = torch.cat([poisoned[k].flatten().float() for k in keys])
    h_flat = torch.cat([honest[k].flatten().float() for k in keys])
    diff = p_flat - h_flat
    norm = diff.norm().item()
    if norm > epsilon:
        diff = diff * (epsilon / norm)
    projected_flat = h_flat + diff
    result = {}
    offset = 0
    for k in keys:
        numel = poisoned[k].numel()
        result[k] = projected_flat[offset:offset+numel].reshape(poisoned[k].shape).to(poisoned[k].dtype)
        offset += numel
    return result


# --- Adaptive: Neurotoxin-style ---
def run_neurotoxin_adaptive(seed, top_frac, scale_factor):
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

    attack = get_attack("backdoor_pixel")

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    grad_history = None
    current_lr = FL_CONFIG.learning_rate

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
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
                update = neurotoxin_project(update, grad_history, top_frac, scale_factor)
            updates.append(update)

        # Update gradient history from honest updates
        honest_updates = [u for i, u in enumerate(updates)
                         if participant_ids[i] not in adv_ids]
        if honest_updates:
            keys = list(honest_updates[0].keys())
            avg_flat = torch.zeros_like(
                torch.cat([honest_updates[0][k].flatten().float() for k in keys])
            )
            for u in honest_updates:
                avg_flat += torch.cat([u[k].flatten().float() for k in keys])
            avg_flat /= len(honest_updates)
            if grad_history is None:
                grad_history = avg_flat
            else:
                grad_history = 0.9 * grad_history + 0.1 * avg_flat

        aggregated = generic_compose(server, updates, "foolsgold", "rfa", tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


def neurotoxin_project(update, grad_history, top_frac, scale_factor):
    if grad_history is None:
        return update
    keys = list(update.keys())
    flat = torch.cat([update[k].flatten().float() for k in keys])
    k = int(len(grad_history) * top_frac)
    _, top_indices = grad_history.abs().topk(k)
    mask = torch.zeros_like(flat)
    mask[top_indices] = 1.0
    projected = flat * mask * scale_factor
    result = {}
    offset = 0
    for key in keys:
        numel = update[key].numel()
        result[key] = projected[offset:offset+numel].reshape(update[key].shape).to(update[key].dtype)
        offset += numel
    return result


# --- Adaptive: Consensus-shift ---
def run_consensus_shift(seed, shift_rate):
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

    attack = get_attack("backdoor_pixel")

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    shift_rounds = 25
    current_lr = FL_CONFIG.learning_rate

    for r in range(FL_CONFIG.num_rounds):
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
            if cid in adv_ids:
                if r < shift_rounds:
                    # Phase 1: subtle shift toward target
                    backdoor_update = attack.manipulate_update(update, server.global_model)
                    keys = list(update.keys())
                    shifted = {}
                    for k in keys:
                        shifted[k] = update[k] + shift_rate * (backdoor_update[k] - update[k])
                    update = shifted
                else:
                    # Phase 2: full backdoor
                    update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, "foolsgold", "rfa", tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Main ---
if __name__ == "__main__":
    print("=" * 70)
    print("  FG→RFA FLAGSHIP COMPOSITION VALIDATION")
    print("=" * 70)

    t0 = time.time()
    runs_done = 0

    # --- Part 1: Base composition (n=30) ---
    print("\n--- Part 1: Base Composition (n=30) ---")
    for policy in ["committed_scaling", "committed_pixel", "oracle"]:
        print(f"\n  Policy: {policy}")
        for seed in BASE_SEEDS:
            if has_run("base_composition", policy, seed):
                runs_done += 1
                continue
            t_run = time.time()
            acc, asr = run_base(seed, policy)
            save_run("base_composition", policy, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run
            print(f"    seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    # --- Part 2: Projection adaptive ---
    print("\n--- Part 2: Projection Adaptive Attack ---")
    for eps in [0.5, 1.0, 2.0, 3.0, 5.0]:
        key = f"eps_{eps:.1f}"
        print(f"\n  epsilon={eps}")
        for seed in ADAPTIVE_SEEDS:
            if has_run("adaptive_projection", key, seed):
                runs_done += 1
                continue
            t_run = time.time()
            acc, asr = run_projection_adaptive(seed, eps)
            save_run("adaptive_projection", key, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run
            print(f"    seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    # --- Part 3: Neurotoxin adaptive ---
    print("\n--- Part 3: Neurotoxin Adaptive Attack ---")
    for top_frac, scale in [(0.05, 1.0), (0.10, 1.0), (0.20, 1.0), (0.10, 2.0), (0.20, 2.0)]:
        key = f"top{top_frac:.2f}_scale{scale:.1f}"
        print(f"\n  top_frac={top_frac}, scale={scale}")
        for seed in ADAPTIVE_SEEDS:
            if has_run("adaptive_neurotoxin", key, seed):
                runs_done += 1
                continue
            t_run = time.time()
            acc, asr = run_neurotoxin_adaptive(seed, top_frac, scale)
            save_run("adaptive_neurotoxin", key, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run
            print(f"    seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    # --- Part 4: Consensus-shift ---
    print("\n--- Part 4: Consensus-Shift Attack ---")
    for rate in [0.01, 0.05]:
        key = f"rate_{rate:.2f}"
        print(f"\n  shift_rate={rate}")
        for seed in ADAPTIVE_SEEDS:
            if has_run("adaptive_consensus_shift", key, seed):
                runs_done += 1
                continue
            t_run = time.time()
            acc, asr = run_consensus_shift(seed, rate)
            save_run("adaptive_consensus_shift", key, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run
            print(f"    seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    s = load_or_init()

    print("\n  Base composition (FG→RFA):")
    for policy in ["committed_scaling", "committed_pixel", "oracle"]:
        data = s["base_composition"].get(policy, {})
        if "mean_asr" in data:
            print(f"    {policy:<20s}: ASR = {data['mean_asr']:.3f} ± {data['std_asr']:.3f} (n={data['n']})")

    print("\n  Projection adaptive:")
    for key in sorted(s.get("adaptive_projection", {}).keys()):
        data = s["adaptive_projection"][key]
        if "mean_asr" in data:
            print(f"    {key:<15s}: ASR = {data['mean_asr']:.3f} ± {data['std_asr']:.3f}")

    print("\n  Neurotoxin adaptive:")
    for key in sorted(s.get("adaptive_neurotoxin", {}).keys()):
        data = s["adaptive_neurotoxin"][key]
        if "mean_asr" in data:
            print(f"    {key:<20s}: ASR = {data['mean_asr']:.3f} ± {data['std_asr']:.3f}")

    print("\n  Consensus-shift:")
    for key in sorted(s.get("adaptive_consensus_shift", {}).keys()):
        data = s["adaptive_consensus_shift"][key]
        if "mean_asr" in data:
            print(f"    {key:<15s}: ASR = {data['mean_asr']:.3f} ± {data['std_asr']:.3f}")

    wall_time = (time.time() - t0) / 60
    print(f"\n  Total runs: {runs_done}")
    print(f"  Wall time: {wall_time:.1f} min ({wall_time/60:.1f} hours)")
    print(f"  Output: {output_path}")
