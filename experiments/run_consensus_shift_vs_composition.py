"""
Consensus-Shift Attack vs Rep+TM Composition Defense.

Tests whether a slow, coordinated consensus-shifting attack can break the
reputation + trimmed-mean composition defense. The attack operates in two phases:

Phase 1 (rounds 1..shift_rounds): adversarial clients submit updates biased
toward a target direction at a small fraction (shift_rate) of the honest update
magnitude, staying within the trimmed-mean acceptance range and gradually
shifting the consensus reference.

Phase 2 (rounds shift_rounds+1..50): adversarial clients submit full backdoor
updates that now appear normal relative to the shifted reference.

Sweep: shift_rate in [0.01, 0.05, 0.1, 0.2], 10 seeds each.
FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn.

Comparison baselines (from composition_rep_tm results):
  committed_pixel vs composition:  0.679
  committed_scaling vs composition: 0.023
  temporal_mix oracle:              0.915

Output: results/consensus_shift_vs_composition/summary.json
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
from experiments.run_composition_rep_tm import compose_reputation_trimmed_mean
from attacks.consensus_shift_attack import ConsensusShiftAttack

# Config
SEEDS = list(range(42, 52))  # 10 seeds
SHIFT_RATES = [0.01, 0.05, 0.1, 0.2]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

# Output
output_dir = os.path.join(base_dir, "results", "consensus_shift_vs_composition")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "method": "consensus_shift_vs_rep_tm_composition",
        "description": "consensus-shift attack (two-phase) vs reputation+trimmed_mean composition",
        "shift_rates": {f"rate_{r}": {"per_seed": []} for r in SHIFT_RATES},
        "comparison": {
            "committed_pixel_vs_composition": 0.679,
            "committed_scaling_vs_composition": 0.023,
            "temporal_mix_oracle": 0.915,
        },
    }


def has_run(shift_rate, seed):
    s = load_or_init()
    key = f"rate_{shift_rate}"
    return any(r["seed"] == seed for r in s["shift_rates"][key]["per_seed"])


def save_one(shift_rate, seed, accuracy, asr_final):
    s = load_or_init()
    key = f"rate_{shift_rate}"
    runs = s["shift_rates"][key]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_final": float(asr_final),
    })
    s["shift_rates"][key]["per_seed"] = runs

    # Update summary statistics
    asrs = [r["asr_final"] for r in runs]
    s["shift_rates"][key]["mean_asr"] = float(np.mean(asrs))
    s["shift_rates"][key]["std_asr"] = float(np.std(asrs))

    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, shift_rate):
    """Run a single (seed, shift_rate) experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack = ConsensusShiftAttack(
        shift_rate=shift_rate,
        shift_rounds=25,
        scale_factor=1.0,
    )

    # Build clients; poison adversarial datasets
    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
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
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = compose_reputation_trimmed_mean(server, updates)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr_final)


if __name__ == "__main__":
    print("=== Consensus-Shift Attack vs Rep+TM Composition ===")
    print(f"  Shift rates: {SHIFT_RATES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  FL config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"rounds={FL_CONFIG.num_rounds}, f={ADV_FRACTION}")
    print(f"  Attack: ConsensusShiftAttack (shift_rounds=25, scale_factor=1.0)")
    print(f"  Defense: reputation + trimmed_mean composition\n")

    t0 = time.time()
    for shift_rate in SHIFT_RATES:
        for seed in SEEDS:
            if has_run(shift_rate, seed):
                print(f"  [skip] rate={shift_rate} seed {seed}", flush=True)
                continue
            t_run = time.time()
            acc, asr = run_one(seed, shift_rate)
            save_one(shift_rate, seed, acc, asr)
            dt = time.time() - t_run
            print(f"  rate={shift_rate} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)",
                  flush=True)

    # Final summary
    print(f"\n=== CONSENSUS-SHIFT vs COMPOSITION SUMMARY ===")
    s = load_or_init()
    for shift_rate in SHIFT_RATES:
        key = f"rate_{shift_rate}"
        runs = s["shift_rates"][key]["per_seed"]
        if runs:
            asrs = [r["asr_final"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            print(f"  shift_rate={shift_rate}: ASR={np.mean(asrs):.3f}+/-{np.std(asrs):.3f}, "
                  f"acc={np.mean(accs):.3f}+/-{np.std(accs):.3f} (n={len(runs)})")

    print(f"\nBaseline comparison (committed attacks vs same composition):")
    print(f"  committed_pixel:   ASR=0.679")
    print(f"  committed_scaling: ASR=0.023")
    print(f"  temporal_mix oracle: ASR=0.915")
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
