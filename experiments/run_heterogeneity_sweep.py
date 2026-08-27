"""
Heterogeneity sweep for the three criterion-PASS compositions.

Reviewer concern (Round 8): every composition experiment in the paper uses Dirichlet
alpha=0.5. This matters most for FoolsGold, whose discriminative signal is pairwise
cosine similarity -- at high alpha (near-IID) benign clients produce genuinely similar
updates, so FG may down-weight benign clients and a PASS pair could fail. That would be
a real boundary condition for the framework, so we test it rather than assume it.

NOTE: the pre-existing results/sweep_new_alpha*/ directories are single-defense
payoff matrices (fedavg / multi_krum / rfa only) -- they contain no compositions.
This is the first composition-level heterogeneity measurement.

Config: N=10, K=5, f=0.2, 50 rounds, cifar_cnn. alpha=0.5 is NOT re-run; reuse
results/all_compositions/summary.json for that column.

Output: results/heterogeneity_sweep/summary.json
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
from experiments.run_all_compositions import generic_compose

PAIRS = [
    ("foolsgold", "rfa"),
    ("foolsgold", "coord_median"),
    ("reputation", "coord_median"),
]
ALPHAS = [0.1, 1.0, 10.0]          # 0.5 already measured in all_compositions
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "heterogeneity_sweep")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")

ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}


def run_one(seed, d1, d2, attack_name, alpha):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, alpha, seed
    )
    server = FederatedServer(get_model("cifar_cnn", num_classes), device)

    adversarial_ids = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    attack = get_attack(ATTACK_MAP.get(attack_name, attack_name))

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
        aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


if __name__ == "__main__":
    total = len(PAIRS) * len(ALPHAS) * len(ATTACKS) * len(SEEDS)
    print("=== Heterogeneity sweep: criterion-PASS compositions vs Dirichlet alpha ===")
    print(f"  pairs={len(PAIRS)} alphas={ALPHAS} attacks={len(ATTACKS)} seeds={SEEDS}")
    print(f"  total runs = {total}\n", flush=True)

    # resume support: reload partial results if the job is restarted
    results = {}
    if os.path.exists(output_path):
        try:
            results = json.load(open(output_path)).get("cells", {})
            print(f"  resuming: {len(results)} cells already complete\n", flush=True)
        except Exception:
            results = {}

    t0 = time.time()
    done = 0
    for d1, d2 in PAIRS:
        for alpha in ALPHAS:
            for attack in ATTACKS:
                key = f"{d1}_then_{d2}|alpha{alpha}|{attack}"
                if key in results and len(results[key].get("per_seed", [])) == len(SEEDS):
                    done += len(SEEDS)
                    continue
                per_seed = []
                for seed in SEEDS:
                    t = time.time()
                    acc, asr = run_one(seed, d1, d2, attack, alpha)
                    per_seed.append({"seed": seed, "accuracy": acc, "asr": asr})
                    done += 1
                    print(f"  [{done}/{total}] {d1}->{d2} a={alpha} {attack} s{seed}: "
                          f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
                asrs = [r["asr"] for r in per_seed]
                accs = [r["accuracy"] for r in per_seed]
                results[key] = {
                    "d1": d1, "d2": d2, "alpha": alpha, "attack": attack,
                    "per_seed": per_seed,
                    "mean_asr": float(np.mean(asrs)), "std_asr": float(np.std(asrs)),
                    "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
                }
                with open(output_path, "w") as f:
                    json.dump({"description": "Criterion-PASS compositions vs Dirichlet alpha",
                               "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                                          "f": ADV_FRACTION, "rounds": FL_CONFIG.num_rounds,
                                          "model": "cifar_cnn", "seeds": SEEDS},
                               "note": "alpha=0.5 not re-run; see results/all_compositions/summary.json",
                               "cells": results}, f, indent=2)

    # max-committed ASR per (pair, alpha)
    print("\n=== MAX-COMMITTED ASR (max over the two attacks of the per-attack seed mean) ===")
    print(f"{'pair':32s} " + " ".join(f"a={a:<6}" for a in ALPHAS))
    for d1, d2 in PAIRS:
        row = []
        for alpha in ALPHAS:
            vals = [results[f"{d1}_then_{d2}|alpha{alpha}|{atk}"]["mean_asr"]
                    for atk in ATTACKS if f"{d1}_then_{d2}|alpha{alpha}|{atk}" in results]
            row.append(f"{max(vals):.3f} " if vals else "  --   ")
        print(f"{d1+'->'+d2:32s} " + " ".join(f"{v:<8}" for v in row))
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h")
    print(f"Saved to {output_path}")
