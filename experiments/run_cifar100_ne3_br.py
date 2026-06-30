"""
Round 47 — NE3 BR on CIFAR-100 to test second-dataset replication of the persistence collapse.

Reviewer Q1 asks for replication on another benchmark. CIFAR-100 shares the input domain
with CIFAR-10 but has 100 classes; if persistence is dataset-agnostic, the realized
scaling ASR should remain near-saturation (>0.85) under NE3 mix.

Setup mirrors the CIFAR-10 headline:
  N=10, K=5, f=0.2, 50 rounds, cifar_cnn (with 100-class output)
  NE3: FedAvg 26% + NormClip 74% per round (sampled)
  Attacks: model_scaling, backdoor_pixel
  Seeds: 42-46 (5 seeds)

Output: results/cifar100_ne3_br/summary.json

Pre-committed interpretation criteria:
  POSITIVE: realized scaling ASR > 0.85 AND realized VoPD (matched seeds) <= 0.05
  PARTIAL : realized scaling ASR in [0.6, 0.85] OR realized VoPD in [0.05, 0.15]
  NEGATIVE: realized scaling ASR < 0.6 OR realized VoPD > 0.15
"""
import copy
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

DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
ATTACKS = ["model_scaling", "backdoor_pixel"]
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar100_ne3_br")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init_summary():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "dataset": "cifar100",
        "model": "cifar_cnn",
        "policy": DEFENSE_DIST,
        "seeds": [],
        "attacks": {a: {"per_seed": []} for a in ATTACKS},
    }


def save_one(seed: int, attack: str, accuracy: float, asr: float):
    s = load_or_init_summary()
    a = s["attacks"].setdefault(attack, {"per_seed": []})
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    a["per_seed"].append({"seed": seed, "accuracy": float(accuracy),
                           "attack_success_rate": float(asr)})
    if seed not in s["seeds"]:
        s["seeds"].append(seed)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(seed: int, attack: str) -> bool:
    s = load_or_init_summary()
    a = s["attacks"].get(attack, {})
    return any(e["seed"] == seed for e in a.get("per_seed", []))


def run_ne3_br(seed: int, attack_name: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    exp_config = ExperimentConfig(
        dataset="cifar100", model="cifar_cnn",
        dirichlet_alpha=0.5, adversarial_fraction=ADV_FRACTION,
        num_trials=1, seed=seed, device="mps",
    )
    client_datasets, test_dataset, num_classes = get_federated_dataset(
        exp_config.dataset, FL_CONFIG.num_clients, exp_config.dirichlet_alpha, seed
    )
    model = get_model(exp_config.model, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))
    attack = get_attack(attack_name)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 1000)
    current_lr = FL_CONFIG.learning_rate

    fedavg_count = 0
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
        if d_this == "fedavg":
            fedavg_count += 1
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "attack_success_rate": float(asr),
        "fedavg_rounds": int(fedavg_count),
    }


print(f"CIFAR-100 NE3 BR experiment (Round 47 second-dataset replication):")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Attacks: {ATTACKS}, Seeds: {SEEDS}")
print(f"  Dataset: cifar100, N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, T={FL_CONFIG.num_rounds}")
print(f"  Output: {output_path}\n")

t0 = time.time()
runs_done = 0

for seed in SEEDS:
    for attack in ATTACKS:
        if has_run(seed, attack):
            print(f"  [skip] seed {seed} {attack} cached", flush=True)
            continue
        t_run = time.time()
        r = run_ne3_br(seed, attack)
        save_one(seed, attack, r["accuracy"], r["attack_success_rate"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {attack}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)


print(f"\n=== CIFAR-100 NE3 BR SUMMARY (n={len(SEEDS)} seeds) ===")
s = load_or_init_summary()
for attack in ATTACKS:
    a = s["attacks"][attack]
    asrs = [e["attack_success_rate"] for e in a["per_seed"]]
    accs = [e["accuracy"] for e in a["per_seed"]]
    print(f"  {attack}: ASR={np.mean(asrs):.3f}+/-{np.std(asrs):.3f}  acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}")
    print(f"    per-seed ASRs: {[round(x, 3) for x in asrs]}")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
print(f"Saved: {output_path}")

# Pre-committed interpretation
scaling_asrs = [e["attack_success_rate"] for e in s["attacks"]["model_scaling"]["per_seed"]]
scaling_mean = float(np.mean(scaling_asrs))
print(f"\n=== Pre-committed interpretation ===")
if scaling_mean > 0.85:
    print(f"  Realized scaling ASR = {scaling_mean:.3f} > 0.85 -> POSITIVE replication likely.")
elif scaling_mean > 0.6:
    print(f"  Realized scaling ASR = {scaling_mean:.3f} in [0.6, 0.85] -> PARTIAL replication.")
else:
    print(f"  Realized scaling ASR = {scaling_mean:.3f} < 0.6 -> NEGATIVE; persistence does NOT cleanly replicate on CIFAR-100.")
