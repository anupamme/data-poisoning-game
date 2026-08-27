"""
Out-of-sample PASS-direction validation: DBA attack transfer at f=0.4.

Tests whether compositions predicted to PASS by the Signal-Preservation Criterion
also suppress the Distributed Backdoor Attack (DBA) — a categorically different
attack mechanism from the development-set attacks (model_scaling, backdoor_pixel).

DBA uses distributed sub-triggers across 4 image corners, requiring inter-client
coordination. At f=0.4 with K=5, expected adversarial co-participation = 2.0,
enabling the distributed mechanism to function.

Pairs tested:
  PASS predictions: fg→rfa, fg→cm, rep→cm
  FAIL controls: nc→rep, rfa→rep, fg→tm
  Sanity baseline: DBA vs fedavg (no composition)

Config: N=10, K=5, f=0.4, cifar_cnn, 50 rounds, seeds 42-46
Output: results/dba_composition_transfer/summary.json
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import apply_d1_transform, generic_compose

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
    ("fedavg", "fedavg"),  # DBA vs no defense (baseline)
]
ALL_PAIRS = PASS_PAIRS + FAIL_PAIRS + SANITY_PAIRS

SEEDS = [42, 43, 44, 45, 46]
ADV_FRACTION = 0.4
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "dba_composition_transfer")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "OOS PASS validation: DBA attack transfer at f=0.4",
        "attack": "dba",
        "adversarial_fraction": ADV_FRACTION,
        "num_clients": FL_CONFIG.num_clients,
        "clients_per_round": FL_CONFIG.clients_per_round,
        "num_rounds": FL_CONFIG.num_rounds,
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
    s["pairs"][pk]["n_seeds"] = len(per_seed)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Training Loop ---
def run_one(seed, d1, d2):
    """Run one DBA composition experiment."""
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

    # Poison adversarial client datasets with DBA sub-triggers
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

        # Apply composition (or single defense for sanity)
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
    print("  OOS PASS VALIDATION: DBA Attack Transfer (f=0.4)")
    print("=" * 70)
    print(f"  Config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}")
    print(f"  num_adv = {int(FL_CONFIG.num_clients * ADV_FRACTION)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  PASS pairs: {[(d1, d2) for d1, d2 in PASS_PAIRS]}")
    print(f"  FAIL pairs: {[(d1, d2) for d1, d2 in FAIL_PAIRS]}")
    print(f"  Total runs: {total_runs}")
    print()

    t0 = time.time()
    runs_done = 0

    for d1, d2 in ALL_PAIRS:
        pk = pair_key(d1, d2)
        category = "PASS" if (d1, d2) in PASS_PAIRS else "FAIL" if (d1, d2) in FAIL_PAIRS else "SANITY"
        print(f"\n--- [{category}] {d1}→{d2} ---")

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

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  DBA COMPOSITION TRANSFER — RESULTS")
    print("=" * 70)

    s = load_or_init()
    print(f"\n  {'Pair':<25s} {'Category':<8s} {'Mean ASR':<12s} {'Std':<8s} {'Verdict'}")
    print(f"  {'-'*70}")

    pass_correct = 0
    pass_total = 0
    fail_correct = 0
    fail_total = 0

    for d1, d2 in ALL_PAIRS:
        pk = pair_key(d1, d2)
        if pk in s["pairs"] and "mean_asr" in s["pairs"][pk]:
            mean_asr = s["pairs"][pk]["mean_asr"]
            std_asr = s["pairs"][pk]["std_asr"]
            category = "PASS" if (d1, d2) in PASS_PAIRS else "FAIL" if (d1, d2) in FAIL_PAIRS else "SANITY"

            if category == "PASS":
                verdict = "CORRECT" if mean_asr < 0.5 else "MISS"
                pass_total += 1
                if mean_asr < 0.5:
                    pass_correct += 1
            elif category == "FAIL":
                verdict = "CORRECT" if mean_asr >= 0.5 else "MISS"
                fail_total += 1
                if mean_asr >= 0.5:
                    fail_correct += 1
            else:
                verdict = f"baseline ({mean_asr:.3f})"

            print(f"  {d1}→{d2:<15s} {category:<8s} {mean_asr:<12.3f} {std_asr:<8.3f} {verdict}")

    print(f"\n  PASS accuracy: {pass_correct}/{pass_total}")
    print(f"  FAIL accuracy: {fail_correct}/{fail_total}")
    if pass_total + fail_total > 0:
        print(f"  Overall: {pass_correct + fail_correct}/{pass_total + fail_total}")

    # Sanity check
    sanity_pk = pair_key("fedavg", "fedavg")
    if sanity_pk in s["pairs"] and "mean_asr" in s["pairs"][sanity_pk]:
        baseline = s["pairs"][sanity_pk]["mean_asr"]
        print(f"\n  Sanity: DBA vs fedavg baseline ASR = {baseline:.3f}")
        if baseline < 0.3:
            print("  WARNING: DBA baseline < 0.3 — attack may be too weak at f=0.4")
            print("  Consider re-running at f=0.6 for stronger differentiation")

    wall = (time.time() - t0) / 60
    print(f"\n  Wall time: {wall:.1f} min")
    print(f"  Output: {output_path}")
