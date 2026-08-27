"""
3-Way composition experiments (Move 3, Round-2 review).

Tests the composability criterion on ordered triples (d1->d2->d3):
  d1 transform -> d2 transform -> d3 aggregate

With 7 defenses, there are P(7,3)=210 ordered triples. Exhaustive search is
infeasible; the criterion guides selection. We pre-registered 10 triples
(6 predicted LOW, 4 predicted HIGH) in pre_registration_three_way.md.

Config: N=10, K=5, f=0.2, cifar_cnn, 50 rounds, seeds 42-46
Output: results/three_way_compositions/summary.json
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

# --- Configuration ---
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACKS = ["committed_scaling", "committed_pixel"]

# Pre-registered triples: (label, d1, d2, d3, predicted)
TRIPLES = [
    ("fg_rep_rfa", "foolsgold", "reputation", "rfa", "LOW"),
    ("fg_rep_cm", "foolsgold", "reputation", "coord_median", "LOW"),
    ("rep_fg_rfa", "reputation", "foolsgold", "rfa", "LOW"),
    ("rep_fg_cm", "reputation", "foolsgold", "coord_median", "LOW"),
    ("fg_nc_cm", "foolsgold", "norm_clip", "coord_median", "LOW"),
    ("nc_fg_tm", "norm_clip", "foolsgold", "trimmed_mean", "HIGH"),
    ("rfa_nc_rep", "rfa", "norm_clip", "reputation", "HIGH"),
    ("fg_rfa_cm", "foolsgold", "rfa", "coord_median", "LOW"),
    ("fedavg_nc_rep", "fedavg", "norm_clip", "reputation", "HIGH"),
    ("nc_rep_tm", "norm_clip", "reputation", "trimmed_mean", "HIGH"),
]

output_dir = os.path.join(base_dir, "results", "three_way_compositions")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "3-way composition criterion validation (Move 3)",
        "config": {
            "num_clients": FL_CONFIG.num_clients,
            "clients_per_round": FL_CONFIG.clients_per_round,
            "adversarial_fraction": ADV_FRACTION,
            "num_rounds": FL_CONFIG.num_rounds,
            "model": "cifar_cnn",
        },
        "triples": {},
    }


def has_run(label, attack, seed):
    s = load_or_init()
    triple_data = s["triples"].get(label, {})
    atk_data = triple_data.get(attack, {})
    return any(r["seed"] == seed for r in atk_data.get("per_seed", []))


def save_one(label, d1, d2, d3, predicted, attack, seed, acc, asr):
    s = load_or_init()
    if label not in s["triples"]:
        s["triples"][label] = {
            "d1": d1, "d2": d2, "d3": d3,
            "predicted": predicted,
        }
    if attack not in s["triples"][label]:
        s["triples"][label][attack] = {"per_seed": []}
    ps = [r for r in s["triples"][label][attack]["per_seed"] if r["seed"] != seed]
    ps.append({"seed": seed, "accuracy": float(acc), "asr": float(asr)})
    s["triples"][label][attack]["per_seed"] = ps
    asrs = [r["asr"] for r in ps]
    s["triples"][label][attack]["mean_asr"] = float(np.mean(asrs))
    s["triples"][label][attack]["std_asr"] = float(np.std(asrs))
    # Compute max_committed_asr across attacks
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["triples"][label] and "mean_asr" in s["triples"][label][atk]:
            max_asr = max(max_asr, s["triples"][label][atk]["mean_asr"])
    s["triples"][label]["max_committed_asr"] = float(max_asr)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def three_way_compose(server, updates, d1, d2, d3, tau=5.0):
    """Chain d1 transform -> d2 transform -> d3 aggregate."""
    t1 = apply_d1_transform(updates, d1, tau=tau)
    t2 = apply_d1_transform(t1, d2, tau=tau)
    aggregated = server.aggregate(t2, method=d3, tau=tau)
    return aggregated


def run_one(seed, d1, d2, d3, attack_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed)
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    else:
        internal_attack = "backdoor_pixel"
    attack = get_attack(internal_attack)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = list(np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False))

        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size)
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = three_way_compose(server, updates, d1, d2, d3, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    acc = server.evaluate(test_dataset)["accuracy"]
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(acc), float(asr)


if __name__ == "__main__":
    total = len(TRIPLES) * len(ATTACKS) * len(SEEDS)
    print("=" * 70)
    print("  3-WAY COMPOSITION CRITERION VALIDATION")
    print("=" * 70)
    print(f"  Config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}")
    print(f"  Triples: {len(TRIPLES)}  Attacks: {ATTACKS}  Seeds: {SEEDS}")
    print(f"  Total runs: {total}")
    print()

    t0 = time.time()
    done = 0
    for label, d1, d2, d3, predicted in TRIPLES:
        print(f"\n--- [{predicted}] {d1}->{d2}->{d3} ({label}) ---")
        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(label, attack, seed):
                    print(f"  [skip] {attack} seed {seed}", flush=True)
                    done += 1
                    continue
                tr = time.time()
                acc, asr = run_one(seed, d1, d2, d3, attack)
                save_one(label, d1, d2, d3, predicted, attack, seed, acc, asr)
                done += 1
                eta = (time.time() - t0) / done * (total - done) / 3600
                print(f"  {attack} seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({(time.time()-tr)/60:.1f}min, {done}/{total}, ETA {eta:.1f}h)",
                      flush=True)

    # --- Results ---
    print("\n" + "=" * 70)
    print("  3-WAY COMPOSITION — RESULTS")
    print("=" * 70)

    s = load_or_init()
    correct = 0
    total_scored = 0
    print(f"\n  {'Triple':<20s} {'Predict':<8s} {'Max ASR':<10s} {'Scaling':<12s} {'Pixel':<12s} {'Verdict'}")
    print(f"  {'-'*72}")

    for label, d1, d2, d3, predicted in TRIPLES:
        td = s["triples"].get(label, {})
        max_asr = td.get("max_committed_asr", None)
        sc = td.get("committed_scaling", {}).get("mean_asr", None)
        px = td.get("committed_pixel", {}).get("mean_asr", None)
        if max_asr is not None:
            if predicted == "LOW":
                ok = max_asr < 0.5
            else:
                ok = max_asr >= 0.5
            correct += int(ok)
            total_scored += 1
            sc_str = f"{sc:.3f}" if sc is not None else "?"
            px_str = f"{px:.3f}" if px is not None else "?"
            print(f"  {label:<20s} {predicted:<8s} {max_asr:<10.3f} {sc_str:<12s} "
                  f"{px_str:<12s} {'CORRECT' if ok else 'MISS'}")

    print(f"\n  Accuracy: {correct}/{total_scored}")
    low_triples = [l for l, _, _, _, p in TRIPLES if p == "LOW"]
    low_correct = sum(1 for l, _, _, _, p in TRIPLES
                      if p == "LOW" and s["triples"].get(l, {}).get("max_committed_asr", 1) < 0.5)
    high_triples = [l for l, _, _, _, p in TRIPLES if p == "HIGH"]
    high_correct = sum(1 for l, _, _, _, p in TRIPLES
                       if p == "HIGH" and s["triples"].get(l, {}).get("max_committed_asr", 0) >= 0.5)
    print(f"  LOW predictions: {low_correct}/{len(low_triples)}")
    print(f"  HIGH predictions: {high_correct}/{len(high_triples)}")
    print(f"\n  Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  Output: {output_path}")
