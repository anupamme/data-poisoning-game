"""
PHASE 0 of the metric-swap ablation: standalone baselines for the two new variants.

cos_krum and cos_reputation have never been run. Their standalone ASR is a C0/C1 INPUT,
so it must be measured BEFORE predictions are frozen in
experiments/pre_registration_metric_swap.md -- otherwise the labels for Phase 2 would be
unfalsifiable. Same reason experiments/run_prospective_pilot.py preceded the Round 9 freeze.

Run as `fedavg_then_X` so the numbers are directly comparable to the single-defense
baselines already in the paper (reputation 0.017/0.842, krum 0.061/0.583).

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn, seeds 42/43/44, both committed attacks.
Output: results/metric_swap_baselines/summary.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from torch.utils.data import Subset
from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import generic_compose

DEFENSES = ["cos_reputation", "cos_krum"]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}

out_dir = os.path.join(base, "results", "metric_swap_baselines"); os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "summary.json")


def run_one(seed, d1, d2, attack_name):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset("cifar10", FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[attack_name])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(FL_CONFIG.num_clients)]
    lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        pids = np.random.choice(FL_CONFIG.num_clients,
                               size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
                               replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, FL_CONFIG.local_epochs, lr, FL_CONFIG.local_batch_size)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=5.0))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return float(srv.evaluate(td)["accuracy"]), float(evaluate_backdoor(srv.global_model, td, device=dev))


if __name__ == "__main__":
    total = len(DEFENSES) * len(ATTACKS) * len(SEEDS)
    print(f"=== PHASE 0: metric-swap standalone baselines ({total} runs) ===")
    print("    C0/C1 inputs -- must complete BEFORE the Phase 2 predictions are frozen\n", flush=True)
    cells = {}
    if os.path.exists(out_path):
        try:
            cells = json.load(open(out_path)).get("cells", {}); print(f"  resuming: {len(cells)} cells\n", flush=True)
        except Exception:
            cells = {}
    t0 = time.time(); done = 0
    for d2 in DEFENSES:
        for atk in ATTACKS:
            key = f"fedavg_then_{d2}|{atk}"
            if key in cells and len(cells[key]["per_seed"]) == len(SEEDS):
                done += len(SEEDS); continue
            ps = []
            for seed in SEEDS:
                t = time.time(); acc, asr = run_one(seed, "fedavg", d2, atk)
                ps.append({"seed": seed, "accuracy": acc, "asr": asr}); done += 1
                print(f"  [{done}/{total}] fedavg->{d2:15s} {atk:18s} s{seed}: "
                      f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
            cells[key] = {"d1": "fedavg", "d2": d2, "attack": atk, "per_seed": ps,
                          "mean_asr": float(np.mean([r["asr"] for r in ps])),
                          "std_asr": float(np.std([r["asr"] for r in ps])),
                          "mean_acc": float(np.mean([r["accuracy"] for r in ps]))}
            json.dump({"description": "Phase 0: standalone baselines for cos_reputation / cos_krum "
                                      "(C0/C1 inputs, measured before the Phase 2 freeze)",
                       "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50, "seeds": SEEDS},
                       "cells": cells}, open(out_path, "w"), indent=2)

    print("\n=== STANDALONE BASELINES (max-committed ASR) ===")
    print(f"{'defense':18s} {'scaling':>16s} {'pixel':>16s} {'max':>7s}  C1 gate")
    for d2 in DEFENSES:
        ks = {a: f"fedavg_then_{d2}|{a}" for a in ATTACKS}
        if not all(k in cells for k in ks.values()):
            continue
        sc, px = cells[ks["committed_scaling"]], cells[ks["committed_pixel"]]
        mx = max(sc["mean_asr"], px["mean_asr"])
        # C1 is per-attack: does this defense suppress the attack norm_clip/rfa fail (scaling)?
        gate = "suppresses scaling" if sc["mean_asr"] < 0.5 else "FAILS scaling -> C1 fails, no positive test"
        print(f"{d2:18s} {sc['mean_asr']:.3f} (acc {sc['mean_acc']:.2f}) "
              f"{px['mean_asr']:.3f} (acc {px['mean_acc']:.2f}) {mx:7.3f}  {gate}")
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
