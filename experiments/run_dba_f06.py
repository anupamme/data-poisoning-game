"""
Round 50 — DBA at f=0.6 second-attack-family pilot (reviewer C3).

DBA at f=0.4 (Round 45) was suggestive but bimodal: with K=5, expected adversarial
co-participation per round = K*f = 2.0, sometimes <2 adversarial clients sampled
together, blocking the distributed trigger. At f=0.6: expected co-participation
3.0, making >=2 adversarial co-participation near-certain. This pilot tests
whether the persistence collapse replicates on DBA when co-sampling is reliable.

Run DBA NE3 BR against TWO deployed policies:
  - NE3 mix (FedAvg 26% + NormClip 74%) - original equilibrium
  - Reputation-equilibrium (NormClip 83% + reputation 17%) - orthogonal menu

Setup:
  - 5 seeds (42-46), 50 rounds, cifar_cnn
  - N=10, K=5, f=0.6 (6 adversarial of 10)
  - Attack: dba only (focus on second persistent attack family)

Total: 5 seeds * 2 policies = 10 runs at ~10 min each = ~1.7 hr MPS.

Pre-committed interpretation:
  STRONG: DBA ASR > 0.7 across >=4/5 seeds AND > 0.5 on at least one mix
    => "DBA at f=0.6 replicates persistence collapse on a second attack family;
        mechanism is not specific to model scaling."
  PARTIAL: DBA ASR mean > 0.5 but per-seed bimodal
    => "Improved over f=0.4 but co-sampling variance remains; partial replication."
  NEGATIVE: DBA ASR < 0.3
    => "Even at f=0.6 DBA does not persist; suggests DBA's per-round trigger-alignment
        mechanism is fundamentally different from scaling's amplitude-amplification."

Output: results/randomized_defense_dba_f06/summary.json
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

POLICIES = [
    ("ne3_mix", {"fedavg": 0.26, "norm_clip": 0.74}),
    ("reputation_eq", {"norm_clip": 0.83, "reputation": 0.17}),
]
ATTACK = "dba"
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.6  # KEY CHANGE: raised from 0.4

output_dir = os.path.join(base_dir, "results", "randomized_defense_dba_f06")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "attack": ATTACK,
        "adversarial_fraction": ADV_FRACTION,
        "seeds": SEEDS,
        "policies": {name: {"policy": pol, "per_seed": []} for name, pol in POLICIES},
        "pre_committed": {
            "STRONG": "DBA ASR > 0.7 in >=4/5 seeds AND > 0.5 on at least one mix",
            "PARTIAL": "DBA ASR mean > 0.5 but per-seed bimodal",
            "NEGATIVE": "DBA ASR < 0.3",
        },
    }


def save_one(policy_name: str, seed: int, accuracy: float, asr: float, breakdown: dict):
    s = load_or_init()
    p = s["policies"][policy_name]
    p["per_seed"] = [e for e in p["per_seed"] if e["seed"] != seed]
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "attack_success_rate": float(asr),
    }
    entry.update({k: int(v) for k, v in breakdown.items()})
    p["per_seed"].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(policy_name: str, seed: int) -> bool:
    s = load_or_init()
    p = s["policies"].get(policy_name, {})
    return any(e["seed"] == seed for e in p.get("per_seed", []))


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
    rng = np.random.default_rng(seed + 4000)
    current_lr = FL_CONFIG.learning_rate

    breakdown = {f"{d}_rounds": 0 for d in defenses}
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
        breakdown[f"{d_this}_rounds"] += 1
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "attack_success_rate": float(asr),
        "breakdown": breakdown,
    }


print(f"Round 50 DBA at f=0.6 pilot (reviewer C3):")
for name, pol in POLICIES:
    print(f"  {name}: {pol}")
print(f"  Attack: {ATTACK}, Seeds: {SEEDS}")
print(f"  f={ADV_FRACTION}, N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}")
print(f"  Expected runs: {len(POLICIES) * len(SEEDS)}\n")

t0 = time.time()
runs_done = 0

for policy_name, policy in POLICIES:
    for seed in SEEDS:
        if has_run(policy_name, seed):
            print(f"  [skip] {policy_name} seed {seed} cached", flush=True)
            continue
        t_run = time.time()
        r = run_br(seed, policy)
        save_one(policy_name, seed, r["accuracy"], r["attack_success_rate"], r["breakdown"])
        runs_done += 1
        dt = time.time() - t_run
        breakdown_str = " ".join(f"{k.replace('_rounds','')}={v}" for k, v in r["breakdown"].items())
        print(f"  seed {seed} {policy_name}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"({breakdown_str}) ({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)


print(f"\n=== DBA f=0.6 PILOT COMPLETE ===")
s = load_or_init()
policy_means = {}
for name, _ in POLICIES:
    asrs = [e["attack_success_rate"] for e in s["policies"][name]["per_seed"]]
    accs = [e["accuracy"] for e in s["policies"][name]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        am, asd = float(np.mean(accs)), float(np.std(accs))
        policy_means[name] = m
        print(f"  {name}: DBA ASR = {m:.3f} +/- {sd:.3f}  acc = {am:.3f} +/- {asd:.3f}")
        print(f"    per-seed ASRs: {[round(x,3) for x in asrs]}")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")

# Pre-committed verdict
print(f"\n=== Pre-committed verdict ===")
all_asrs = []
for name, _ in POLICIES:
    all_asrs.extend([e["attack_success_rate"] for e in s["policies"][name]["per_seed"]])
if all_asrs:
    overall_mean = float(np.mean(all_asrs))
    any_mix_strong = any(m > 0.5 for m in policy_means.values())
    n_above_70 = sum(1 for x in all_asrs if x > 0.7)
    if n_above_70 >= 4 and any_mix_strong:
        print(f"  STRONG: {n_above_70}/{len(all_asrs)} seeds > 0.7 AND some mix mean > 0.5 -> persistence replicates on DBA")
    elif overall_mean > 0.5:
        print(f"  PARTIAL: overall mean ASR = {overall_mean:.3f} > 0.5 (bimodal?) -> partial replication")
    elif overall_mean < 0.3:
        print(f"  NEGATIVE: overall mean ASR = {overall_mean:.3f} < 0.3 -> DBA does not persist even at f=0.6")
    else:
        print(f"  INTERMEDIATE: overall mean ASR = {overall_mean:.3f} in [0.3, 0.5]")
