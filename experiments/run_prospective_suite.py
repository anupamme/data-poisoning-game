"""
PHASE 2 of the prospective composition test.

Runs the 20 pairs whose predictions were FROZEN in
experiments/pre_registration_prospective.md (git commit 262cf35), committed before
this script's output directory existed.

Do not edit the pair list or the predictions to match observed outcomes. Misses are
reported as misses.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn, seeds 42/43/44, both committed attacks.
Output: results/prospective_suite/summary.json
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

# (d1, d2, predicted_label, category) -- verbatim from the frozen pre-registration
PAIRS = [
    ("fltrust", "rfa",           "LOW",  "certified"),
    ("foolsgold", "fltrust",     "LOW",  "certified"),
    ("reputation", "fltrust",    "LOW",  "certified"),
    ("norm_clip", "fltrust",     "LOW",  "certified"),
    ("rfa", "fltrust",           "LOW",  "certified"),
    ("fltrust", "coord_median",  "LOW",  "certified-uncertain"),
    ("fltrust", "reputation",    "HIGH", "C2-fail"),
    ("fltrust", "norm_clip",     "HIGH", "C2-fail"),
    ("fltrust", "krum",          "HIGH", "C2-fail"),
    ("fltrust", "multi_krum",    "HIGH", "C2-fail"),
    ("fltrust", "trimmed_mean",  "HIGH", "C3-fail-uncertain"),
    ("foolsgold", "krum",        "HIGH", "C1-fail"),
    ("reputation", "krum",       "HIGH", "C1-fail"),
    ("norm_clip", "multi_krum",  "HIGH", "C1-fail"),
    ("rfa", "multi_krum",        "HIGH", "C1-fail"),
    ("krum", "rfa",              "HIGH", "DEGEN"),
    ("krum", "coord_median",     "HIGH", "DEGEN"),
    ("multi_krum", "rfa",        "HIGH", "DEGEN"),
    ("krum", "fltrust",          "LOW",  "DEGEN"),
    ("multi_krum", "fltrust",    "LOW",  "DEGEN"),
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}

out_dir = os.path.join(base, "results", "prospective_suite"); os.makedirs(out_dir, exist_ok=True)
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
    total = len(PAIRS) * len(ATTACKS) * len(SEEDS)
    print(f"=== PHASE 2: prospective suite ({len(PAIRS)} pairs, {total} runs) ===")
    print("    predictions frozen in pre_registration_prospective.md @ 262cf35\n", flush=True)
    cells = {}
    if os.path.exists(out_path):
        try:
            cells = json.load(open(out_path)).get("cells", {}); print(f"  resuming: {len(cells)} cells\n", flush=True)
        except Exception:
            cells = {}
    t0 = time.time(); done = 0
    for d1, d2, pred, cat in PAIRS:
        for atk in ATTACKS:
            key = f"{d1}_then_{d2}|{atk}"
            if key in cells and len(cells[key]["per_seed"]) == len(SEEDS):
                done += len(SEEDS); continue
            ps = []
            for seed in SEEDS:
                t = time.time(); acc, asr = run_one(seed, d1, d2, atk)
                ps.append({"seed": seed, "accuracy": acc, "asr": asr}); done += 1
                print(f"  [{done}/{total}] {d1}->{d2:13s} {atk:18s} s{seed}: "
                      f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
            cells[key] = {"d1": d1, "d2": d2, "attack": atk, "predicted": pred, "category": cat,
                          "per_seed": ps, "mean_asr": float(np.mean([r["asr"] for r in ps])),
                          "std_asr": float(np.std([r["asr"] for r in ps])),
                          "mean_acc": float(np.mean([r["accuracy"] for r in ps]))}
            json.dump({"description": "Phase 2: prospective suite; predictions frozen at 262cf35",
                       "prereg_commit": "262cf35",
                       "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50, "seeds": SEEDS},
                       "cells": cells}, open(out_path, "w"), indent=2)

    print("\n=== RESULTS vs FROZEN PREDICTIONS ===")
    print(f"{'pair':30s} {'cat':20s} {'pred':>5} {'maxASR':>7} {'actual':>7}  hit")
    tp = fp = tn = fn = 0
    for d1, d2, pred, cat in PAIRS:
        ks = [f"{d1}_then_{d2}|{a}" for a in ATTACKS]
        if not all(k in cells for k in ks):
            continue
        mx = max(cells[k]["mean_asr"] for k in ks)
        actual = "LOW" if mx < 0.5 else "HIGH"
        hit = actual == pred
        if not cat.startswith("DEGEN"):
            if pred == "LOW" and hit: tp += 1
            elif pred == "LOW" and not hit: fp += 1
            elif pred == "HIGH" and hit: tn += 1
            else: fn += 1
        print(f"{d1+'->'+d2:30s} {cat:20s} {pred:>5} {mx:7.3f} {actual:>7}  {'OK' if hit else 'MISS'}")
    n = tp + fp + tn + fn
    if n:
        print(f"\n  Non-DEGEN confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}  ({tp+tn}/{n} correct)")
        if tp + fp: print(f"  precision (predicted LOW): {tp}/{tp+fp}")
        if tn + fp: print(f"  specificity: {tn}/{tn+fp}")
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
