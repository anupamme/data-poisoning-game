"""
Compute-matched mixing control experiment.

Addresses the "compute-budget confound" concern: standard temporal mixing compares
composition (both defenses every round) against mixing (one defense per round). This
confounds "two mechanisms vs one" with "deterministic vs randomized order."

This experiment holds compute constant: both conditions apply BOTH defenses every
round, but in a FIXED vs RANDOMIZED order.

Conditions:
  fg_cm_fixed:      FG->CM every round (criterion PASS)
  randorder_fg_cm:  each round 50/50 sample FG->CM or CM->FG
  fg_rfa_fixed:     FG->RFA every round (flagship, theorem-backed)
  randorder_fg_rfa: each round 50/50 sample FG->RFA or RFA->FG

The prediction from the criterion (C2/C3 are order-sensitive):
  CM->FG: apply_d1_transform passes CM updates unchanged (CM is an aggregator,
          not a per-client weighting transform); d2=FG sees unmodified updates
          -> FG-only aggregation, doesn't suppress pixel backdoor.
  FG->CM: C1^C2^C3 PASS -> suppresses both attacks.
  -> Randomized-order: FG->CM half rounds, CM->FG (=FG alone) half rounds
     -> pixel ASR intermediate, not as low as fixed FG->CM.

Config: N=10, K=5, f=0.2, cifar_cnn, 50 rounds, seeds 42-46 (n=5)
Attacks: committed_scaling, committed_pixel
Output: results/compute_matched_mixing/summary.json
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
from experiments.run_all_compositions import apply_d1_transform

SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

CONDITIONS = [
    ("fg_cm_fixed",      "foolsgold", "coord_median", False),
    ("randorder_fg_cm",  "foolsgold", "coord_median", True),
    ("fg_rfa_fixed",     "foolsgold", "rfa",          False),
    ("randorder_fg_rfa", "foolsgold", "rfa",          True),
]
ATTACKS = ["committed_scaling", "committed_pixel"]

output_dir = os.path.join(base_dir, "results", "compute_matched_mixing")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"description": "Compute-matched mixing: fixed vs randomized order", "conditions": {}}


def has_run(cond_name, attack, seed):
    s = load_or_init()
    c = s["conditions"].get(cond_name, {})
    per_seed = c.get(attack, {}).get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(cond_name, attack, seed, accuracy, asr):
    s = load_or_init()
    if cond_name not in s["conditions"]:
        s["conditions"][cond_name] = {}
    if attack not in s["conditions"][cond_name]:
        s["conditions"][cond_name][attack] = {"per_seed": []}
    per_seed = s["conditions"][cond_name][attack]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["conditions"][cond_name][attack]["per_seed"] = per_seed
    asrs = [r["asr"] for r in per_seed]
    s["conditions"][cond_name][attack]["mean_asr"] = float(np.mean(asrs))
    s["conditions"][cond_name][attack]["std_asr"] = float(np.std(asrs))
    s["conditions"][cond_name][attack]["n"] = len(asrs)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, d1, d2, randomize_order, attack_name):
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

    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    elif attack_name == "committed_pixel":
        internal_attack = "backdoor_pixel"
    else:
        internal_attack = attack_name

    attack = get_attack(internal_attack)
    clients = []
    for i in range(FL_CONFIG.num_clients):
        dataset = client_datasets[i]
        if i in adversarial_ids:
            dataset = attack.poison_dataset(dataset)
        clients.append(FederatedClient(i, dataset, device))

    rng = np.random.default_rng(seed + 1000)
    current_lr = FL_CONFIG.learning_rate

    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = rng.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False
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

        # Determine order this round
        if randomize_order and rng.random() < 0.5:
            actual_d1, actual_d2 = d2, d1
        else:
            actual_d1, actual_d2 = d1, d2

        # Apply d1's per-client transform, then aggregate with d2
        transformed = apply_d1_transform(updates, actual_d1)
        aggregated = server.aggregate(transformed, method=actual_d2)
        server.apply_update(aggregated)

        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


def main():
    t0 = time.time()
    total_runs = len(CONDITIONS) * len(ATTACKS) * len(SEEDS)
    done = 0

    for cond_name, d1, d2, randomize_order in CONDITIONS:
        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(cond_name, attack, seed):
                    done += 1
                    print(f"  [cached] {cond_name} | {attack} | seed={seed}")
                    continue
                t1 = time.time()
                acc, asr = run_one(seed, d1, d2, randomize_order, attack)
                save_one(cond_name, attack, seed, acc, asr)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done}/{total_runs}] {cond_name} | {attack} | seed={seed} "
                      f"acc={acc:.3f} asr={asr:.3f} ({time.time()-t1:.1f}s, total {elapsed:.0f}s)",
                      flush=True)

    s = load_or_init()
    print("\n=== COMPUTE-MATCHED MIXING SUMMARY ===")
    for cond_name, d1, d2, randomize_order in CONDITIONS:
        cond = s["conditions"].get(cond_name, {})
        row = f"{cond_name:25s}"
        for atk in ATTACKS:
            atk_data = cond.get(atk, {})
            mean = atk_data.get("mean_asr", float("nan"))
            std = atk_data.get("std_asr", float("nan"))
            row += f"  {atk.replace('committed_',''):10s} ASR={mean:.3f}+/-{std:.3f}"
        print(row)


if __name__ == "__main__":
    main()
