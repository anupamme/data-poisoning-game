"""
Round 53 — ResNet18 mini mix-ratio sweep on NormClip+reputation (Path 2: architecture replication).

Mirrors experiments/run_mix_ratio_sweep.py exactly, with two changes:
  - Model: resnet18 (with GroupNorm head) instead of cifar_cnn.
  - Mix ratios: 3 points (NC90/rep10, NC50/rep50, NC10/rep90) instead of 5,
    to keep compute budget at ~25 hr instead of ~42 hr.

Total: 15 runs (3 mixes x 5 seeds) at ~100 min/run = ~25 hr MPS.
Output: results/cifar10_mix_ratio_sweep_resnet18/summary.json

Pre-committed verdict:
  PASS (STRONG): ASR > 0.85 at NC90/rep10 AND ASR < 0.4 at NC10/rep90 AND monotonic
  MEDIUM: monotonic but one endpoint criterion fails
  FAIL: NC90/rep10 ASR < 0.5 (scaling does not persist on ResNet18 here)
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

MIX_RATIOS = [
    ("NC90_rep10", {"norm_clip": 0.90, "reputation": 0.10}),
    ("NC50_rep50", {"norm_clip": 0.50, "reputation": 0.50}),
    ("NC10_rep90", {"norm_clip": 0.10, "reputation": 0.90}),
]
ATTACK = "model_scaling"
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
MODEL_NAME = "resnet18"

output_dir = os.path.join(base_dir, "results", "cifar10_mix_ratio_sweep_resnet18")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "model": MODEL_NAME,
        "attack": ATTACK,
        "seeds": SEEDS,
        "mix_ratios": {name: {"policy": pol, "per_seed": []} for name, pol in MIX_RATIOS},
        "pre_committed": {
            "PASS": "ASR > 0.85 at NC90/rep10 AND ASR < 0.4 at NC10/rep90 AND monotonic",
            "MEDIUM": "monotonic but one endpoint criterion fails",
            "FAIL": "NC90/rep10 ASR < 0.5 (scaling doesn't persist on ResNet18)",
        },
    }


def save_one(ratio_name: str, seed: int, accuracy: float, asr: float,
              nc_rounds: int, rep_rounds: int, wall_time_s: float):
    s = load_or_init()
    r = s["mix_ratios"][ratio_name]
    r["per_seed"] = [e for e in r["per_seed"] if e["seed"] != seed]
    r["per_seed"].append({
        "seed": seed, "accuracy": float(accuracy),
        "attack_success_rate": float(asr),
        "norm_clip_rounds": int(nc_rounds),
        "reputation_rounds": int(rep_rounds),
        "wall_time_s": float(wall_time_s),
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
    model = get_model(MODEL_NAME, num_classes)
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
    rng = np.random.default_rng(seed + 8000)
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true",
                    help="Run only (seed 42, NC90_rep10) for wall-time smoke test")
args = parser.parse_args()

if args.smoke:
    plan = [("NC90_rep10", dict(MIX_RATIOS)["NC90_rep10"], 42)]
else:
    plan = [(name, pol, s) for name, pol in MIX_RATIOS for s in SEEDS]

print(f"Round 53 ResNet18 {'SMOKE' if args.smoke else 'mini-boundary'} sweep:")
print(f"  Model: {MODEL_NAME}")
for name, pol in MIX_RATIOS:
    print(f"  {name}: {pol}")
print(f"  Attack: {ATTACK}, Seeds: {SEEDS}")
print(f"  Planned runs: {len(plan)}\n")

t0 = time.time()
runs_done = 0

for ratio_name, policy, seed in plan:
    if has_run(ratio_name, seed):
        print(f"  [skip] {ratio_name} seed {seed} cached", flush=True)
        continue
    t_run = time.time()
    r = run_br(seed, policy)
    dt = time.time() - t_run
    save_one(ratio_name, seed, r["accuracy"], r["attack_success_rate"],
              r["nc_rounds"], r["rep_rounds"], dt)
    runs_done += 1
    print(f"  seed {seed} {ratio_name}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
          f"(nc={r['nc_rounds']}/rep={r['rep_rounds']}) "
          f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)


print(f"\n=== ResNet18 {'SMOKE' if args.smoke else 'MINI-SWEEP'} COMPLETE ===")
s = load_or_init()
ratio_means = {}
for name, _ in MIX_RATIOS:
    asrs = [e["attack_success_rate"] for e in s["mix_ratios"][name]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        ratio_means[name] = m
        print(f"  {name}: scaling ASR = {m:.3f} +/- {sd:.3f}  per-seed: {[round(x,3) for x in asrs]}")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")

if args.smoke:
    if "NC90_rep10" in ratio_means:
        asr = ratio_means["NC90_rep10"]
        if asr > 0.1:
            print(f"\nSMOKE PASS: ASR={asr:.3f} > 0.1 on ResNet18; safe to launch full sweep")
        else:
            print(f"\nSMOKE WARN: ASR={asr:.3f} suspiciously low; verify before full launch")
else:
    # Pre-committed verdict
    if all(name in ratio_means for name, _ in MIX_RATIOS):
        nc90 = ratio_means["NC90_rep10"]
        nc50 = ratio_means["NC50_rep50"]
        nc10 = ratio_means["NC10_rep90"]
        monotonic = nc90 >= nc50 >= nc10
        print(f"\n=== Pre-committed verdict ===")
        if nc90 > 0.85 and nc10 < 0.4 and monotonic:
            print(f"  PASS (STRONG): {nc90:.3f} -> {nc50:.3f} -> {nc10:.3f}, "
                  f"endpoints + monotonicity met")
        elif monotonic and nc90 < 0.5:
            print(f"  FAIL: NC90/rep10 ASR = {nc90:.3f} < 0.5 -> scaling doesn't persist on ResNet18")
        elif monotonic:
            print(f"  MEDIUM: {nc90:.3f} -> {nc50:.3f} -> {nc10:.3f} monotonic but endpoints don't both pass")
        else:
            print(f"  MEDIUM/AMBIGUOUS: {nc90:.3f} -> {nc50:.3f} -> {nc10:.3f} non-monotonic")
