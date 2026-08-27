"""
Out-of-sample PASS-direction validation: Scale transfer (N=100, ResNet18).

Tests whether compositions predicted to PASS by the criterion (validated at
N=10/cifar_cnn) still suppress backdoors at N=100, K=10, f=0.05, ResNet18.

Compositions: fg→rfa, fg→cm
Attacks: committed_scaling, committed_pixel

Config: N=100, K=10, f=0.05, ResNet18, CIFAR-10, alpha=0.5, 50 rounds, seeds 42-46
Output: results/scale_composition_fg_rfa_cm/summary.json
"""
import json
import os
import sys
import time
import gc
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
PAIRS = [
    ("foolsgold", "rfa"),
    ("foolsgold", "coord_median"),
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44, 45, 46]

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ADV_FRACTION = 0.05
NUM_ROUNDS = 50
MODEL_NAME = "resnet18"

FL_CONFIG = FLConfig(
    num_clients=NUM_CLIENTS,
    clients_per_round=CLIENTS_PER_ROUND,
    num_rounds=NUM_ROUNDS,
)

output_dir = os.path.join(base_dir, "results", "scale_composition_fg_rfa_cm")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- MPS memory management ---
def clear_mps_cache():
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    gc.collect()


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "OOS PASS: scale transfer for fg→rfa, fg→cm at N=100/ResNet18",
        "config": {
            "num_clients": NUM_CLIENTS,
            "clients_per_round": CLIENTS_PER_ROUND,
            "adversarial_fraction": ADV_FRACTION,
            "model": MODEL_NAME,
            "num_rounds": NUM_ROUNDS,
        },
        "pairs": {},
    }


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def has_run(d1, d2, attack, seed):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        return False
    if attack not in s["pairs"][pk]:
        return False
    per_seed = s["pairs"][pk][attack].get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(d1, d2, attack, seed, accuracy, asr):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2}
    if attack not in s["pairs"][pk]:
        s["pairs"][pk][attack] = {"per_seed": []}
    per_seed = s["pairs"][pk][attack]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["pairs"][pk][attack]["per_seed"] = per_seed
    asrs = [r["asr"] for r in per_seed]
    s["pairs"][pk][attack]["mean_asr"] = float(np.mean(asrs))
    s["pairs"][pk][attack]["std_asr"] = float(np.std(asrs))
    # Max ASR across attacks for this pair
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
            max_asr = max(max_asr, s["pairs"][pk][atk]["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Training Loop ---
def run_one(seed, d1, d2, attack_name):
    """Run one N=100 composition experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", NUM_CLIENTS, 0.5, seed
    )
    model = get_model(MODEL_NAME, num_classes)
    server = FederatedServer(model, device)

    num_adv = int(NUM_CLIENTS * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    elif attack_name == "committed_pixel":
        internal_attack = "backdoor_pixel"
    else:
        internal_attack = attack_name

    attack = get_attack(internal_attack)

    clients = []
    for i in range(NUM_CLIENTS):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for r in range(NUM_ROUNDS):
        participant_ids = np.random.choice(
            NUM_CLIENTS,
            size=min(CLIENTS_PER_ROUND, NUM_CLIENTS),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Main ---
if __name__ == "__main__":
    total_runs = len(PAIRS) * len(ATTACKS) * len(SEEDS)
    print("=" * 70)
    print("  OOS PASS VALIDATION: Scale Transfer (N=100, ResNet18)")
    print("=" * 70)
    print(f"  Config: N={NUM_CLIENTS}, K={CLIENTS_PER_ROUND}, f={ADV_FRACTION}, "
          f"rounds={NUM_ROUNDS}, model={MODEL_NAME}")
    print(f"  num_adv = {int(NUM_CLIENTS * ADV_FRACTION)}")
    print(f"  Pairs: {PAIRS}")
    print(f"  Attacks: {ATTACKS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated: ~50 min/run → ~{total_runs * 50 / 60:.0f}h total")
    print()

    t0 = time.time()
    runs_done = 0

    for d1, d2 in PAIRS:
        for attack_name in ATTACKS:
            print(f"\n--- {d1}→{d2} | {attack_name} ---")
            for seed in SEEDS:
                if has_run(d1, d2, attack_name, seed):
                    print(f"  [skip] seed {seed}", flush=True)
                    runs_done += 1
                    continue

                t_run = time.time()
                acc, asr = run_one(seed, d1, d2, attack_name)
                save_one(d1, d2, attack_name, seed, acc, asr)
                runs_done += 1
                dt = time.time() - t_run

                elapsed = time.time() - t0
                avg = elapsed / runs_done
                remaining = (total_runs - runs_done) * avg
                eta_h = remaining / 3600

                print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

                clear_mps_cache()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SCALE COMPOSITION TRANSFER — RESULTS")
    print("=" * 70)

    s = load_or_init()
    print(f"\n  {'Pair':<25s} {'Attack':<20s} {'Mean ASR':<12s} {'Std':<8s} {'n':<5s} {'Verdict'}")
    print(f"  {'-'*75}")

    all_pass = True
    for d1, d2 in PAIRS:
        pk = pair_key(d1, d2)
        if pk not in s["pairs"]:
            continue
        for atk in ATTACKS:
            if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
                mean = s["pairs"][pk][atk]["mean_asr"]
                std = s["pairs"][pk][atk]["std_asr"]
                n = len(s["pairs"][pk][atk]["per_seed"])
                verdict = "PASS" if mean < 0.5 else "FAIL"
                if mean >= 0.5:
                    all_pass = False
                print(f"  {d1}→{d2:<15s} {atk:<20s} {mean:<12.3f} {std:<8.3f} {n:<5d} {verdict}")

        if "max_committed_asr" in s["pairs"][pk]:
            print(f"  {'':25s} {'MAX':<20s} {s['pairs'][pk]['max_committed_asr']:<12.3f}")

    print(f"\n  All predictions correct: {'YES' if all_pass else 'NO'}")

    wall = (time.time() - t0) / 60
    print(f"\n  Wall time: {wall:.1f} min ({wall/60:.1f}h)")
    print(f"  Output: {output_path}")
