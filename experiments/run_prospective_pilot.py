"""
PHASE 0 of the prospective composition test: single-defense baselines for the three
defenses that were never part of the paper's 7-defense menu.

These baselines are INPUTS to condition C1, not outcomes. They must therefore be
measured BEFORE predictions are frozen in pre_registration_prospective.md. Nothing in
this script touches composition performance.

Also records the FedAvg sanity baseline for both committed attacks. This is the gate the
DBA out-of-sample experiment lacked: a predicted-FAIL control is only informative if the
attack actually embeds against no defense. If FedAvg ASR < 0.5 here, the corresponding
FAIL predictions must be reported as uninformative rather than confirmed.

Config matches the rest of the paper: N=10, K=5, f=0.2, 50 rounds, cifar_cnn, alpha=0.5.
Output: results/prospective_pilot/summary.json
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

from torch.utils.data import Subset
from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor

DEFENSES = ["fltrust", "krum", "multi_krum", "fedavg"]   # fedavg = sanity gate
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}

out_dir = os.path.join(base_dir, "results", "prospective_pilot")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary.json")


def run_one(seed, defense, attack_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed)
    # FLTrust needs a small clean server holdout; harmless for the others.
    clean_holdout = Subset(test_dataset, list(range(100)))
    server = FederatedServer(get_model("cifar_cnn", num_classes), device,
                             clean_holdout_dataset=clean_holdout, holdout_batch_size=32)

    adv_ids = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    attack = get_attack(ATTACK_MAP[attack_name])
    clients = [FederatedClient(i, attack.poison_dataset(client_datasets[i]) if i in adv_ids
                               else client_datasets[i], device)
               for i in range(FL_CONFIG.num_clients)]

    lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        pids = np.random.choice(FL_CONFIG.num_clients,
                                size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
                                replace=False)
        updates = []
        for cid in pids:
            u = clients[cid].train(server.global_model, FL_CONFIG.local_epochs,
                                   lr, FL_CONFIG.local_batch_size)
            if cid in adv_ids:
                u = attack.manipulate_update(u, server.global_model)
            updates.append(u)
        server.apply_update(server.aggregate(updates, method=defense))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    acc = float(server.evaluate(test_dataset)["accuracy"])
    asr = float(evaluate_backdoor(server.global_model, test_dataset, device=device))
    return acc, asr


if __name__ == "__main__":
    total = len(DEFENSES) * len(ATTACKS) * len(SEEDS)
    print("=== PHASE 0: prospective-suite pilot (C1 inputs + FedAvg sanity gate) ===")
    print(f"  defenses={DEFENSES}  attacks={ATTACKS}  seeds={SEEDS}  total={total}\n", flush=True)

    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path)).get("cells", {})
            print(f"  resuming, {len(results)} cells done\n", flush=True)
        except Exception:
            results = {}

    t0 = time.time(); done = 0
    for d in DEFENSES:
        for atk in ATTACKS:
            key = f"{d}|{atk}"
            if key in results and len(results[key]["per_seed"]) == len(SEEDS):
                done += len(SEEDS); continue
            per_seed = []
            for seed in SEEDS:
                t = time.time()
                acc, asr = run_one(seed, d, atk)
                per_seed.append({"seed": seed, "accuracy": acc, "asr": asr})
                done += 1
                print(f"  [{done}/{total}] {d:12s} {atk:18s} s{seed}: "
                      f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
            results[key] = {"defense": d, "attack": atk, "per_seed": per_seed,
                            "mean_asr": float(np.mean([r["asr"] for r in per_seed])),
                            "std_asr": float(np.std([r["asr"] for r in per_seed])),
                            "mean_acc": float(np.mean([r["accuracy"] for r in per_seed]))}
            json.dump({"description": "Phase 0 pilot: single-defense baselines (C1 inputs) "
                                      "+ FedAvg sanity gate, measured BEFORE predictions frozen",
                       "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                                  "f": ADV_FRACTION, "rounds": FL_CONFIG.num_rounds,
                                  "alpha": 0.5, "seeds": SEEDS},
                       "cells": results}, open(out_path, "w"), indent=2)

    print("\n=== C1 INPUTS (max-committed ASR per defense) ===")
    for d in DEFENSES:
        vals = [results[f"{d}|{a}"]["mean_asr"] for a in ATTACKS if f"{d}|{a}" in results]
        per = {a: round(results[f'{d}|{a}']['mean_asr'], 3) for a in ATTACKS if f"{d}|{a}" in results}
        if vals:
            mx = max(vals)
            print(f"  {d:12s} max={mx:.3f} {per}  -> suppresses both: {mx < 0.5}")
    print("\n=== SANITY GATE ===")
    for a in ATTACKS:
        k = f"fedavg|{a}"
        if k in results:
            v = results[k]["mean_asr"]
            print(f"  fedavg vs {a}: ASR={v:.3f} -> "
                  f"{'INFORMATIVE (>=0.5)' if v >= 0.5 else 'UNINFORMATIVE (<0.5): FAIL controls void'}")
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
