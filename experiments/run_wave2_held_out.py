"""
Wave 2 Held-Out Composability Criterion Validation — 24 pre-registered pairs.

Pre-registration: experiments/pre_registration_wave2.md (committed before this runs).
All predictions made BEFORE experiments. This validates the criterion on a held-out set.

Defenses: fedavg, norm_clip, trimmed_mean, coord_median, rfa, foolsgold, reputation
Pairs: 24 held-out ordered pairs (not in the Wave 1 development set of 18 pairs)
Attacks: committed_scaling, committed_pixel
Seeds: 42-46 (5 seeds)
FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn

Output: results/wave2_held_out/summary.json
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
PAIRS = [
    # DEGEN: coord_median as d1 (passes through unchanged → d2 alone)
    ("coord_median", "fedavg"),
    ("coord_median", "foolsgold"),
    ("coord_median", "norm_clip"),
    ("coord_median", "reputation"),
    ("coord_median", "rfa"),
    ("coord_median", "trimmed_mean"),
    # DEGEN: trimmed_mean as d1 (passes through unchanged → d2 alone)
    ("trimmed_mean", "coord_median"),
    ("trimmed_mean", "fedavg"),
    ("trimmed_mean", "foolsgold"),
    ("trimmed_mean", "norm_clip"),
    ("trimmed_mean", "reputation"),
    ("trimmed_mean", "rfa"),
    # DEGEN: fedavg as d1 (remaining untested)
    ("fedavg", "coord_median"),
    ("fedavg", "foolsgold"),
    ("fedavg", "reputation"),
    ("fedavg", "rfa"),
    # Weighting d1 pairs
    ("foolsgold", "fedavg"),
    ("foolsgold", "norm_clip"),
    ("foolsgold", "reputation"),
    # RFA as d1
    ("rfa", "fedavg"),
    ("rfa", "foolsgold"),
    ("rfa", "norm_clip"),
    # NC/rep as d1
    ("norm_clip", "fedavg"),
    ("reputation", "fedavg"),
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for held-out validation
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "wave2_held_out")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "Wave 2 held-out composability validation (24 pre-registered pairs)",
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
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
            max_asr = max(max_asr, s["pairs"][pk][atk]["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Training Loop (reuse from run_all_compositions) ---
def run_one(seed, d1, d2, attack_name):
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


# --- Main ---
if __name__ == "__main__":
    total_runs = len(PAIRS) * len(ATTACKS) * len(SEEDS)
    print("=== Wave 2 Held-Out Composability Validation (24 pairs, pre-registered) ===")
    print(f"  Pairs: {len(PAIRS)} held-out ordered pairs")
    print(f"  Attacks: {ATTACKS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print(f"  FL config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}, cifar_cnn")
    print()

    t0 = time.time()
    completed = 0
    skipped = 0

    for pair_idx, (d1, d2) in enumerate(PAIRS):
        pk = pair_key(d1, d2)
        print(f"[{pair_idx+1}/{len(PAIRS)}] {pk}", flush=True)

        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(d1, d2, attack, seed):
                    skipped += 1
                    continue
                t_run = time.time()
                acc, asr = run_one(seed, d1, d2, attack)
                save_one(d1, d2, attack, seed, acc, asr)
                dt = time.time() - t_run
                completed += 1
                done_total = completed + skipped
                print(f"    {attack} seed={seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({dt:.0f}s) [{done_total}/{total_runs}]", flush=True)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("WAVE 2 HELD-OUT RESULTS")
    print(f"{'='*70}")
    print(f"{'Pair':<35} {'scaling ASR':>12} {'pixel ASR':>12} {'max ASR':>10}")
    print(f"{'-'*70}")

    s = load_or_init()
    for d1, d2 in PAIRS:
        pk = pair_key(d1, d2)
        if pk in s["pairs"]:
            pair_data = s["pairs"][pk]
            scaling_asr = pair_data.get("committed_scaling", {}).get("mean_asr", float("nan"))
            pixel_asr = pair_data.get("committed_pixel", {}).get("mean_asr", float("nan"))
            max_asr = pair_data.get("max_committed_asr", float("nan"))
            print(f"  {pk:<33} {scaling_asr:>10.3f}   {pixel_asr:>10.3f}   {max_asr:>8.3f}")

    print(f"\nCompleted: {completed} new runs, {skipped} cached")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"Output: {output_path}")
