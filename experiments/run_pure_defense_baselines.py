"""
Pure-defense baselines for composition comparison table.

Addresses reviewer Q4: "§5.4 lacks the most natural baseline — composition is
compared against the temporal mix, but not against the best *single* defense."

Tests pure reputation, pure trimmed_mean, and pure coord_median each against
committed_scaling, committed_pixel, and oracle (same FL config as composition).

Config: N=10, K=5, f=0.2, 50 rounds, 5 seeds
Oracle logic: plays model_scaling on trimmed_mean/coord_median rounds,
             backdoor_pixel on reputation rounds (same as rep_tm_survivor)

Output: results/pure_defense_baselines/summary.json
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

SEEDS = list(range(42, 47))
DEFENSES = ["reputation", "trimmed_mean", "coord_median"]
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

ORACLE_BEST = {
    "reputation": "backdoor_pixel",
    "trimmed_mean": "model_scaling",
    "coord_median": "model_scaling",
}

output_dir = os.path.join(base_dir, "results", "pure_defense_baselines")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"defenses": {d: {"adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES}} for d in DEFENSES}}


def has_run(defense, adv_pol, seed):
    s = load_or_init()
    if defense not in s["defenses"]:
        return False
    if adv_pol not in s["defenses"][defense]["adversary_policies"]:
        return False
    return any(r["seed"] == seed for r in s["defenses"][defense]["adversary_policies"][adv_pol]["per_seed"])


def save_one(defense, adv_pol, seed, accuracy, asr):
    s = load_or_init()
    if defense not in s["defenses"]:
        s["defenses"][defense] = {"adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES}}
    if adv_pol not in s["defenses"][defense]["adversary_policies"]:
        s["defenses"][defense]["adversary_policies"][adv_pol] = {"per_seed": []}
    runs = s["defenses"][defense]["adversary_policies"][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["defenses"][defense]["adversary_policies"][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, defense, adv_pol):
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

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack_scaling.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        if adv_pol == "oracle":
            current_attack = attack_scaling if ORACLE_BEST[defense] == "model_scaling" else attack_pixel
        elif adv_pol == "committed_scaling":
            current_attack = attack_scaling
        else:
            current_attack = attack_pixel

        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = current_attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = server.aggregate(updates, method=defense)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


if __name__ == "__main__":
    print("=== Pure-Defense Baselines (Composition Comparison) ===")
    print(f"  Defenses: {DEFENSES}")
    print(f"  Adversary policies: {ADV_POLICIES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Comparison: composition oracle ASR = 0.316 ± 0.162 (n=30)\n")

    t0 = time.time()
    for defense in DEFENSES:
        for adv_pol in ADV_POLICIES:
            for seed in SEEDS:
                if has_run(defense, adv_pol, seed):
                    print(f"  [skip] {defense}/{adv_pol} seed {seed}", flush=True)
                    continue
                t_run = time.time()
                acc, asr = run_one(seed, defense, adv_pol)
                save_one(defense, adv_pol, seed, acc, asr)
                dt = time.time() - t_run
                print(f"  {defense}/{adv_pol} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    print(f"\n=== PURE-DEFENSE BASELINES COMPLETE ===")
    s = load_or_init()
    for defense in DEFENSES:
        print(f"\n  {defense}:")
        for adv_pol in ADV_POLICIES:
            runs = s["defenses"][defense]["adversary_policies"][adv_pol]["per_seed"]
            if runs:
                asrs = [r["asr"] for r in runs]
                accs = [r["accuracy"] for r in runs]
                print(f"    {adv_pol}: ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}, "
                      f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    print(f"\n  Comparison:")
    print(f"    Composition (rep+tm): scaling=0.023, pixel=0.679, oracle=0.316")
    print(f"    Temporal mix (rep30/tm70): scaling=0.871, pixel=0.747, oracle=0.915")
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
