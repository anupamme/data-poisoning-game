"""
Round 58 — Shakespeare LSTM collapse experiment.

Tests whether the persistence collapse replicates on a text/LSTM benchmark.
NE3-equivalent BR: FedAvg 26% + NormClip 74%, committed model_scaling adversary.
15 seeds, 50 rounds, N=10, K=5, f=0.2, CharLSTM.

Output: results/shakespeare_collapse/summary.json
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
from fl_core import FederatedServer, FederatedClient
from fl_core.shakespeare import (
    get_shakespeare_federated, CharLSTM, ShakespearePoisonedDataset,
    get_trigger_info, evaluate_shakespeare_backdoor, evaluate_shakespeare_accuracy
)

DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
SEEDS = list(range(42, 57))  # 15 seeds
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
SCALE_FACTOR = 10.0
MAX_SAMPLES_PER_CLIENT = 2000  # Cap dataset size for speed; original is ~100K

output_dir = os.path.join(base_dir, "results", "shakespeare_collapse")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"policy": DEFENSE_DIST, "seeds": SEEDS, "per_seed": []}


def has_run(seed):
    s = load_or_init()
    return any(e["seed"] == seed for e in s["per_seed"])


def save_one(seed, accuracy, asr):
    s = load_or_init()
    s["per_seed"] = [e for e in s["per_seed"] if e["seed"] != seed]
    s["per_seed"].append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_collapse(seed, log_trajectory=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, vocab_size, char_to_idx = get_shakespeare_federated(
        FL_CONFIG.num_clients, seed=seed
    )
    trigger, target = get_trigger_info(char_to_idx)
    model = CharLSTM(vocab_size)
    server = FederatedServer(model, device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))

    # Cap dataset size for speed
    from torch.utils.data import Subset
    all_datasets = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if len(ds) > MAX_SAMPLES_PER_CLIENT:
            ds = Subset(ds, list(range(MAX_SAMPLES_PER_CLIENT)))
        if i in adversarial_ids:
            ds = ShakespearePoisonedDataset(ds, trigger, target, 0.5)
        all_datasets.append(ds)

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 11000)
    current_lr = 0.5

    asr_timeline = []
    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            client = FederatedClient(cid, all_datasets[cid], device)
            update = client.train(
                server.global_model, 1,
                current_lr, 128
            )
            if cid in adversarial_ids:
                update = {k: v * SCALE_FACTOR for k, v in update.items()}
            updates.append(update)
        d_this = rng.choice(defenses, p=probs)
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= 0.99

        if log_trajectory:
            round_asr = evaluate_shakespeare_backdoor(
                server.global_model, test_dataset, char_to_idx, device=device, num_samples=200
            )
            asr_timeline.append({"round": r + 1, "asr": float(round_asr), "defense": d_this})

    acc = evaluate_shakespeare_accuracy(server.global_model, test_dataset, device=device)
    asr = evaluate_shakespeare_backdoor(server.global_model, test_dataset, char_to_idx, device=device)

    if log_trajectory:
        # Save trajectory
        traj_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(traj_dir, exist_ok=True)
        with open(os.path.join(traj_dir, "asr_timeline.json"), "w") as f:
            json.dump({"seed": seed, "policy": DEFENSE_DIST, "final_asr": float(asr),
                        "final_accuracy": float(acc), "asr_timeline": asr_timeline}, f, indent=2)

    return acc, asr


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trajectory", action="store_true",
                    help="Log per-round ASR for seeds 42-44 (mechanism ablation)")
args = parser.parse_args()

TRAJECTORY_SEEDS = [42, 43, 44]

print(f"Shakespeare collapse experiment:")
print(f"  Policy: {DEFENSE_DIST}, Seeds: {TRAJECTORY_SEEDS if args.trajectory else SEEDS}")
print(f"  N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, f={ADV_FRACTION}")
print(f"  Model: CharLSTM, Scale: {SCALE_FACTOR}x")
if args.trajectory:
    print(f"  Mode: TRAJECTORY (per-round ASR logging)\n")
else:
    print()

t0 = time.time()
runs_done = 0
run_seeds = TRAJECTORY_SEEDS if args.trajectory else SEEDS
for seed in run_seeds:
    if not args.trajectory and has_run(seed):
        print(f"  [skip] seed {seed} cached", flush=True)
        continue
    t_run = time.time()
    acc, asr = run_collapse(seed, log_trajectory=args.trajectory)
    if not args.trajectory:
        save_one(seed, acc, asr)
    runs_done += 1
    dt = time.time() - t_run
    traj_str = " (trajectory logged)" if args.trajectory else ""
    print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s, {runs_done} runs, "
          f"total {(time.time()-t0)/60:.1f}min){traj_str}", flush=True)

print(f"\n=== SHAKESPEARE COLLAPSE COMPLETE ===")
s = load_or_init()
asrs = [e["asr"] for e in s["per_seed"]]
accs = [e["accuracy"] for e in s["per_seed"]]
if asrs:
    print(f"  Scaling ASR: {np.mean(asrs):.3f} +/- {np.std(asrs):.3f}")
    print(f"  Accuracy:    {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  per-seed ASRs: {[round(x, 3) for x in asrs]}")
    if np.mean(asrs) > 0.85:
        print(f"  PERSISTENCE REPLICATES ON LSTM!")
    elif np.mean(asrs) > 0.5:
        print(f"  PARTIAL: persistence present but weaker than CNN")
    else:
        print(f"  NEGATIVE: persistence does not replicate on LSTM")
print(f"Wall time: {(time.time()-t0)/60:.1f} min")
