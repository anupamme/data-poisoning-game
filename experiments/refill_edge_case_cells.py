"""
Round 52 — refill cached backdoor_edge_case cells with the correct evaluator.

The existing cached cells for backdoor_edge_case in
results/cifar10_10seeds/seed_*/per_seed_results.json were measured with the pixel
trigger evaluator, producing artifact zeros. This script re-runs all
(edge_case, defense) cells for 5 seeds and 7 defenses (the FoolsGold and
reputation cells weren't cached at all and are added here).

The new evaluator (evaluate_edge_case_backdoor in run_payoff_matrix) applies the
top-corner -1.0 trigger matching BackdoorEdgeCaseAttack.poison_dataset.

Usage:
  python3 experiments/refill_edge_case_cells.py            # full run
  python3 experiments/refill_edge_case_cells.py --sanity   # one cell (42, fedavg) only

Output: updated cells in results/cifar10_10seeds/seed_*/per_seed_results.json
"""
import argparse
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
from experiments.run_payoff_matrix import evaluate_edge_case_backdoor

SEEDS = [42, 43, 44, 45, 46]
DEFENSES = ["fedavg", "norm_clip", "foolsgold", "reputation",
            "rfa", "trimmed_mean", "coord_median"]
ATTACK = "backdoor_edge_case"
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2


def cell_path(seed):
    return os.path.join(base_dir, "results", "cifar10_10seeds",
                         f"seed_{seed}", "per_seed_results.json")


def has_correct_cell(seed, defense):
    """Check if cell has an ASR > 0 (heuristic: the broken evaluator gave zeros)."""
    p = cell_path(seed)
    if not os.path.exists(p):
        return False
    with open(p) as f:
        d = json.load(f)
    key = f"{ATTACK}_{defense}"
    if key not in d:
        return False
    entry = d[key][0] if isinstance(d[key], list) else d[key]
    # If ASR > 0.01 we treat it as correct; otherwise assume artifact zero.
    return entry.get("attack_success_rate", 0) > 0.01


def save_cell(seed, defense, asr, accuracy, worst_class_acc):
    p = cell_path(seed)
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
    else:
        d = {}
    key = f"{ATTACK}_{defense}"
    entry = {
        "seed": seed,
        "attack": ATTACK,
        "defense": defense,
        "accuracy": float(accuracy),
        "attack_success_rate": float(asr),
        "worst_class_accuracy": float(worst_class_acc),
    }
    d[key] = [entry]  # list format matching existing cells
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def run_pure_strategy(seed, defense):
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
    attack = get_attack(ATTACK)

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
        aggregated = server.aggregate(updates, method=defense)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_edge_case_backdoor(server.global_model, test_dataset, device=device)
    accuracy = float(eval_result["accuracy"])
    # Worst-class accuracy
    per_class = eval_result.get("per_class_accuracy", None)
    if per_class is not None and len(per_class) > 0:
        worst = float(min(per_class))
    else:
        worst = accuracy  # fallback
    return asr, accuracy, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="Run a single (seed 42, fedavg) cell only")
    args = parser.parse_args()

    if args.sanity:
        plan = [(42, "fedavg")]
    else:
        plan = [(s, d) for s in SEEDS for d in DEFENSES]

    print(f"Round 52 edge_case cell refill:")
    print(f"  Attack: {ATTACK}")
    print(f"  Defenses: {DEFENSES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Planned cells: {len(plan)}")
    t0 = time.time()
    runs_done = 0
    for seed, defense in plan:
        if not args.sanity and has_correct_cell(seed, defense):
            print(f"  [skip] seed={seed} defense={defense} (already has ASR>0.01)", flush=True)
            continue
        t_run = time.time()
        asr, acc, wca = run_pure_strategy(seed, defense)
        save_cell(seed, defense, asr, acc, wca)
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed={seed} defense={defense}: ASR={asr:.3f} acc={acc:.3f} wca={wca:.3f} "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)

    print(f"\n=== EDGE_CASE CELL REFILL {'SANITY' if args.sanity else 'COMPLETE'} ===")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")

    if args.sanity:
        # Read back the cell to print verdict
        with open(cell_path(42)) as f:
            d = json.load(f)
        key = f"{ATTACK}_fedavg"
        asr = d[key][0]["attack_success_rate"]
        print(f"Sanity verdict: ASR_42_fedavg = {asr:.3f}")
        if asr > 0.1:
            print(f"PASS -- edge_case is functional under FedAvg; safe to launch full refill.")
        else:
            print(f"FAIL -- edge_case ASR is suspiciously low; investigate before full refill.")


if __name__ == "__main__":
    main()
