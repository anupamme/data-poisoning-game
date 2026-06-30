"""
Round 51 — Per-round-switching oracle adversary on the NC+rep mix-ratio sweep.

Oracle observes the sampled defense each round and plays the per-defense optimal
attack: scaling on NC rounds (argmax(NC) = scaling), pixel on rep rounds
(argmax(rep) = pixel). At end of training, both attacks have contributed to the
trigger embedding; we evaluate the shared pixel trigger ASR.

The realized full-info ASR = trigger ASR at end of T rounds; realized VoPD =
oracle_ASR - max(committed_scaling_ASR, committed_pixel_ASR).

Setup mirrors run_mix_ratio_sweep.py; per-round attack chosen by defense.

Output: results/cifar10_mix_ratio_sweep_oracle/summary.json
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
ORACLE_BEST_ATTACK = {"norm_clip": "model_scaling", "reputation": "backdoor_pixel"}
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_mix_ratio_sweep_oracle")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "oracle_best_attack_per_defense": ORACLE_BEST_ATTACK,
        "seeds": SEEDS,
        "mix_ratios": {name: {"policy": pol, "per_seed": []} for name, pol in MIX_RATIOS},
    }


def save_one(ratio_name: str, seed: int, accuracy: float, asr: float,
              nc_rounds: int, rep_rounds: int, scaling_rounds: int, pixel_rounds: int):
    s = load_or_init()
    r = s["mix_ratios"][ratio_name]
    r["per_seed"] = [e for e in r["per_seed"] if e["seed"] != seed]
    r["per_seed"].append({
        "seed": seed, "accuracy": float(accuracy),
        "attack_success_rate": float(asr),
        "norm_clip_rounds": int(nc_rounds),
        "reputation_rounds": int(rep_rounds),
        "scaling_rounds": int(scaling_rounds),
        "pixel_rounds": int(pixel_rounds),
    })
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(ratio_name: str, seed: int) -> bool:
    s = load_or_init()
    r = s["mix_ratios"].get(ratio_name, {})
    return any(e["seed"] == seed for e in r.get("per_seed", []))


def run_oracle(seed: int, defense_dist: dict) -> dict:
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

    # Pre-poison datasets for both attacks
    scaling_attack = get_attack("model_scaling")
    pixel_attack = get_attack("backdoor_pixel")
    clean_datasets = {}
    scaling_poisoned = {}
    pixel_poisoned = {}
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        clean_datasets[i] = ds
        if i in adversarial_ids:
            scaling_poisoned[i] = scaling_attack.poison_dataset(ds)
            pixel_poisoned[i] = pixel_attack.poison_dataset(ds)

    defenses = list(defense_dist.keys())
    probs = list(defense_dist.values())
    rng = np.random.default_rng(seed + 6000)
    current_lr = FL_CONFIG.learning_rate

    nc_count = 0
    rep_count = 0
    scaling_rounds = 0
    pixel_rounds = 0
    for _ in range(FL_CONFIG.num_rounds):
        d_this = rng.choice(defenses, p=probs)
        if d_this == "norm_clip":
            nc_count += 1
        else:
            rep_count += 1
        attack_this = ORACLE_BEST_ATTACK[d_this]
        if attack_this == "model_scaling":
            scaling_rounds += 1
            poisoned = scaling_poisoned
            attack_obj = scaling_attack
        else:
            pixel_rounds += 1
            poisoned = pixel_poisoned
            attack_obj = pixel_attack

        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            ds = poisoned[cid] if cid in adversarial_ids else clean_datasets[cid]
            client = FederatedClient(cid, ds, device)
            update = client.train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack_obj.manipulate_update(update, server.global_model)
            updates.append(update)
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
        "scaling_rounds": scaling_rounds,
        "pixel_rounds": pixel_rounds,
    }


print(f"Round 51 oracle mix-ratio sweep:")
for name, pol in MIX_RATIOS:
    print(f"  {name}: {pol}")
print(f"  Oracle attack table: {ORACLE_BEST_ATTACK}")
print(f"  Seeds: {SEEDS}")
print(f"  Expected runs: {len(MIX_RATIOS) * len(SEEDS)}\n")

t0 = time.time()
runs_done = 0

for ratio_name, policy in MIX_RATIOS:
    for seed in SEEDS:
        if has_run(ratio_name, seed):
            print(f"  [skip] {ratio_name} seed {seed} cached", flush=True)
            continue
        t_run = time.time()
        r = run_oracle(seed, policy)
        save_one(ratio_name, seed, r["accuracy"], r["attack_success_rate"],
                  r["nc_rounds"], r["rep_rounds"], r["scaling_rounds"], r["pixel_rounds"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {ratio_name}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"(nc={r['nc_rounds']}/rep={r['rep_rounds']} scaling={r['scaling_rounds']}/pixel={r['pixel_rounds']}) "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)


print(f"\n=== ORACLE MIX-RATIO SWEEP COMPLETE ===")
s = load_or_init()
for name, _ in MIX_RATIOS:
    asrs = [e["attack_success_rate"] for e in s["mix_ratios"][name]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        print(f"  {name}: oracle ASR = {m:.3f} +/- {sd:.3f}  per-seed: {[round(x,3) for x in asrs]}")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
