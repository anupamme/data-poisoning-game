"""
Round 61 — Defense composition baseline (W2).

Reviewer W2: "The obvious alternative — simultaneous composition — is never evaluated.
A practitioner would compose defenses (NormClip AND reputation applied jointly every round)."

This experiment applies BOTH NormClip and reputation every round (deterministic composition,
not randomization). This trivially avoids the temporal-mixture/persistence problem because
the suppressing defense is always active.

Setup: N=10, K=5, f=0.2, 50 rounds, cifar_cnn, 5 seeds
Attacks: committed_scaling, committed_pixel
Defense: NormClip(τ=5) then reputation (applied sequentially to same updates)

Output: results/composition_baseline/summary.json
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
from experiments.run_payoff_matrix import evaluate_backdoor

SEEDS = list(range(42, 47))  # 5 seeds
ATTACKS_TO_TEST = ["model_scaling", "backdoor_pixel"]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "composition_baseline")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"method": "norm_clip_then_reputation", "per_run": []}


def has_run(seed, attack):
    s = load_or_init()
    return any(r["seed"] == seed and r["attack"] == attack for r in s["per_run"])


def save_one(seed, attack, accuracy, asr):
    s = load_or_init()
    s["per_run"] = [r for r in s["per_run"] if not (r["seed"] == seed and r["attack"] == attack)]
    s["per_run"].append({"seed": seed, "attack": attack,
                         "accuracy": float(accuracy), "asr": float(asr)})
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def compose_norm_clip_reputation(server, updates, tau=5.0):
    """Apply NormClip first, then reputation weighting on the clipped updates."""
    # Step 1: Norm clip
    clipped = []
    for u in updates:
        flat = torch.cat([u[name].flatten() for name in u])
        norm = flat.norm()
        scale = min(1.0, tau / (norm.item() + 1e-8))
        clipped.append({name: u[name] * scale for name in u})
    # Step 2: Apply reputation to the clipped updates
    return server._reputation(clipped)


def run_composition(seed, attack_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
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
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        # COMPOSITION: NormClip + reputation every round
        aggregated = compose_norm_clip_reputation(server, updates, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


print("=== Defense Composition Baseline ===")
print(f"  Method: NormClip(τ=5) + reputation (applied jointly every round)")
print(f"  Seeds: {SEEDS}, Attacks: {ATTACKS_TO_TEST}\n")

t0 = time.time()
for seed in SEEDS:
    for attack_name in ATTACKS_TO_TEST:
        if has_run(seed, attack_name):
            print(f"  [skip] seed {seed} {attack_name} cached", flush=True)
            continue
        t_run = time.time()
        acc, asr = run_composition(seed, attack_name)
        save_one(seed, attack_name, acc, asr)
        dt = time.time() - t_run
        print(f"  seed {seed} {attack_name}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

print(f"\n=== COMPOSITION BASELINE COMPLETE ===")
s = load_or_init()
for attack_name in ATTACKS_TO_TEST:
    runs = [r for r in s["per_run"] if r["attack"] == attack_name]
    if runs:
        asrs = [r["asr"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        print(f"  {attack_name}: ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}, "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

print(f"\nComparison points (from cached data):")
print(f"  NE3 mix (FedAvg 26% + NormClip 74%): scaling ASR = 0.961±0.052 (n=30)")
print(f"  Pure FedAvg: scaling ASR = ~0.50 (binary), pixel ASR = ~0.82")
print(f"  Pure NormClip: scaling ASR = ~0.95")
print(f"  Pure reputation: scaling ASR = 0.017, pixel ASR = ~0.84")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
