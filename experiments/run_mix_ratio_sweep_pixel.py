"""
Round 51 — Pixel attack on the 5 Round-50 NormClip+reputation mix ratios.

Round 50 measured only committed-scaling ASR across the mix ratios. The realized
VoPD computation needs the realized committed-pixel ASR at each mix too: the
"committed-best" attack at each ratio is whichever of {scaling, pixel} gives
higher realized ASR. From the cached cells, pixel under NC averages 0.81 and
pixel under reputation averages 0.84 -- both admitted -- so pixel may persist
across all mix ratios.

Setup mirrors run_mix_ratio_sweep.py exactly, only ATTACK changes.

Output: results/cifar10_mix_ratio_sweep_pixel/summary.json
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

from config import FLConfig, ExperimentConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor

MIX_RATIOS = [
    ("NC90_rep10", {"norm_clip": 0.90, "reputation": 0.10}),
    ("NC70_rep30", {"norm_clip": 0.70, "reputation": 0.30}),
    ("NC50_rep50", {"norm_clip": 0.50, "reputation": 0.50}),
    ("NC30_rep70", {"norm_clip": 0.30, "reputation": 0.70}),
    ("NC10_rep90", {"norm_clip": 0.10, "reputation": 0.90}),
]
ATTACK = "backdoor_pixel"
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_mix_ratio_sweep_pixel")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "attack": ATTACK,
        "seeds": SEEDS,
        "mix_ratios": {name: {"policy": pol, "per_seed": []} for name, pol in MIX_RATIOS},
    }


def save_one(ratio_name: str, seed: int, accuracy: float, asr: float,
              nc_rounds: int, rep_rounds: int):
    s = load_or_init()
    r = s["mix_ratios"][ratio_name]
    r["per_seed"] = [e for e in r["per_seed"] if e["seed"] != seed]
    r["per_seed"].append({
        "seed": seed, "accuracy": float(accuracy),
        "attack_success_rate": float(asr),
        "norm_clip_rounds": int(nc_rounds),
        "reputation_rounds": int(rep_rounds),
    })
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(ratio_name: str, seed: int) -> bool:
    s = load_or_init()
    r = s["mix_ratios"].get(ratio_name, {})
    return any(e["seed"] == seed for e in r.get("per_seed", []))


def run_br(seed: int, defense_dist: dict) -> dict:
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
    attack = get_attack(ATTACK)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defenses = list(defense_dist.keys())
    probs = list(defense_dist.values())
    rng = np.random.default_rng(seed + 3000)
    current_lr = FL_CONFIG.learning_rate

    nc_count = 0
    rep_count = 0
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
        d_this = rng.choice(defenses, p=probs)
        if d_this == "norm_clip":
            nc_count += 1
        else:
            rep_count += 1
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "attack_success_rate": float(asr),
        "nc_rounds": nc_count,
        "rep_rounds": rep_count,
    }


print(f"Round 51 pixel mix-ratio sweep:")
for name, pol in MIX_RATIOS:
    print(f"  {name}: {pol}")
print(f"  Attack: {ATTACK}, Seeds: {SEEDS}")
print(f"  Expected runs: {len(MIX_RATIOS) * len(SEEDS)}\n")

t0 = time.time()
runs_done = 0

for ratio_name, policy in MIX_RATIOS:
    for seed in SEEDS:
        if has_run(ratio_name, seed):
            print(f"  [skip] {ratio_name} seed {seed} cached", flush=True)
            continue
        t_run = time.time()
        r = run_br(seed, policy)
        save_one(ratio_name, seed, r["accuracy"], r["attack_success_rate"],
                  r["nc_rounds"], r["rep_rounds"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {ratio_name}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"(nc={r['nc_rounds']}/rep={r['rep_rounds']}) "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)


print(f"\n=== PIXEL MIX-RATIO SWEEP COMPLETE ===")
s = load_or_init()
for name, _ in MIX_RATIOS:
    asrs = [e["attack_success_rate"] for e in s["mix_ratios"][name]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        print(f"  {name}: pixel ASR = {m:.3f} +/- {sd:.3f}  per-seed: {[round(x,3) for x in asrs]}")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
