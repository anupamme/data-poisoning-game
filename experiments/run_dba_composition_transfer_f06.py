"""
Powered DBA attack transfer at f=0.6 (N1 response, Round-2 review).

The f=0.4 DBA transfer (run_dba_composition_transfer.py) had no power to fail:
the no-composition sanity baseline (DBA vs fedavg) never cleared the 0.5 verdict
threshold, so FAIL predictions could not be falsified and the criterion scored
3/6 (identical to a constant "predict LOW" rule). This re-run raises f to 0.6
(num_adv=6, expected adversarial co-participation K*f=3.0) so the sanity baseline
clears 0.5 and FAIL predictions become falsifiable.

All six pre-registered pairs (predictions from experiments/pre_registration_oos_pass.md,
UNCHANGED) are scored symmetrically:
  PASS predictions: fg->rfa, fg->cm, rep->cm   (predict ASR < 0.5)
  FAIL predictions: nc->rep, rfa->rep, fg->tm  (predict ASR >= 0.5)
  Sanity baseline:  DBA vs fedavg (must clear 0.5 for the test to have power)

Config: N=10, K=5, f=0.6, cifar_cnn, 50 rounds, seeds 42-46
Output: results/dba_composition_transfer_f06/summary.json
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

# --- Configuration ---
PASS_PAIRS = [
    ("foolsgold", "rfa"),
    ("foolsgold", "coord_median"),
    ("reputation", "coord_median"),
]
FAIL_PAIRS = [
    ("norm_clip", "reputation"),
    ("rfa", "reputation"),
    ("foolsgold", "trimmed_mean"),
]
SANITY_PAIRS = [
    ("fedavg", "fedavg"),  # DBA vs no defense (power check: must clear 0.5)
]
ALL_PAIRS = PASS_PAIRS + FAIL_PAIRS + SANITY_PAIRS

SEEDS = [42, 43, 44, 45, 46]
ADV_FRACTION = 0.6
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "dba_composition_transfer_f06")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "Powered OOS PASS validation: DBA attack transfer at f=0.6 (N1)",
        "attack": "dba",
        "adversarial_fraction": ADV_FRACTION,
        "num_clients": FL_CONFIG.num_clients,
        "clients_per_round": FL_CONFIG.clients_per_round,
        "num_rounds": FL_CONFIG.num_rounds,
        "note": "Predictions pre-registered in pre_registration_oos_pass.md (unchanged from f=0.4 run).",
        "pairs": {},
    }


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def has_run(d1, d2, seed):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        return False
    per_seed = s["pairs"][pk].get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(d1, d2, seed, accuracy, asr):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2, "per_seed": []}
    per_seed = s["pairs"][pk]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["pairs"][pk]["per_seed"] = per_seed
    asrs = [r["asr"] for r in per_seed]
    s["pairs"][pk]["mean_asr"] = float(np.mean(asrs))
    s["pairs"][pk]["std_asr"] = float(np.std(asrs))
    s["pairs"][pk]["min_asr"] = float(np.min(asrs))
    s["pairs"][pk]["max_asr"] = float(np.max(asrs))
    s["pairs"][pk]["n_seeds"] = len(per_seed)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Training Loop ---
def run_one(seed, d1, d2):
    """Run one DBA composition experiment at f=0.6."""
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

    attack = get_attack("dba")

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

        if d1 == "fedavg" and d2 == "fedavg":
            aggregated = server.aggregate(updates, "fedavg")
        else:
            aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Main ---
if __name__ == "__main__":
    total_runs = len(ALL_PAIRS) * len(SEEDS)
    print("=" * 70)
    print("  POWERED OOS PASS VALIDATION: DBA Attack Transfer (f=0.6)")
    print("=" * 70)
    print(f"  Config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}")
    print(f"  num_adv = {int(FL_CONFIG.num_clients * ADV_FRACTION)}, "
          f"expected co-participation K*f = {FL_CONFIG.clients_per_round * ADV_FRACTION:.1f}")
    print(f"  Seeds: {SEEDS}")
    print(f"  PASS pairs: {PASS_PAIRS}")
    print(f"  FAIL pairs: {FAIL_PAIRS}")
    print(f"  Total runs: {total_runs}")
    print()

    t0 = time.time()
    runs_done = 0

    for d1, d2 in ALL_PAIRS:
        pk = pair_key(d1, d2)
        category = "PASS" if (d1, d2) in PASS_PAIRS else "FAIL" if (d1, d2) in FAIL_PAIRS else "SANITY"
        print(f"\n--- [{category}] {d1}->{d2} ---")

        for seed in SEEDS:
            if has_run(d1, d2, seed):
                print(f"  [skip] seed {seed}", flush=True)
                runs_done += 1
                continue

            t_run = time.time()
            acc, asr = run_one(seed, d1, d2)
            save_one(d1, d2, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run

            elapsed = time.time() - t0
            avg = elapsed / runs_done
            remaining = (total_runs - runs_done) * avg
            eta_h = remaining / 3600

            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                  f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

    # --- Symmetric scoring summary ---
    print("\n" + "=" * 70)
    print("  DBA COMPOSITION TRANSFER f=0.6 — SYMMETRIC SCORING")
    print("=" * 70)

    s = load_or_init()
    print(f"\n  {'Pair':<25s} {'Predict':<8s} {'Mean ASR':<10s} {'[min,max]':<16s} {'Verdict'}")
    print(f"  {'-'*72}")

    correct = 0
    total = 0
    for d1, d2 in PASS_PAIRS + FAIL_PAIRS:
        pk = pair_key(d1, d2)
        if pk in s["pairs"] and "mean_asr" in s["pairs"][pk]:
            mean_asr = s["pairs"][pk]["mean_asr"]
            lo, hi = s["pairs"][pk]["min_asr"], s["pairs"][pk]["max_asr"]
            pred = "PASS" if (d1, d2) in PASS_PAIRS else "FAIL"
            if pred == "PASS":
                ok = mean_asr < 0.5
            else:
                ok = mean_asr >= 0.5
            total += 1
            correct += int(ok)
            print(f"  {d1}->{d2:<17s} {pred:<8s} {mean_asr:<10.3f} "
                  f"[{lo:.3f},{hi:.3f}]    {'CORRECT' if ok else 'MISS'}")

    print(f"\n  Symmetric accuracy (all 6 pairs): {correct}/{total}")

    sanity_pk = pair_key("fedavg", "fedavg")
    if sanity_pk in s["pairs"] and "mean_asr" in s["pairs"][sanity_pk]:
        baseline = s["pairs"][sanity_pk]["mean_asr"]
        blo, bhi = s["pairs"][sanity_pk]["min_asr"], s["pairs"][sanity_pk]["max_asr"]
        print(f"\n  Power check: DBA vs fedavg baseline ASR = {baseline:.3f} "
              f"[{blo:.3f}, {bhi:.3f}]")
        if baseline >= 0.5:
            print("  OK: baseline clears 0.5 -- FAIL predictions are falsifiable.")
        else:
            print("  WARNING: baseline still < 0.5 -- test remains under-powered even at f=0.6.")

    wall = (time.time() - t0) / 60
    print(f"\n  Wall time: {wall:.1f} min")
    print(f"  Output: {output_path}")
