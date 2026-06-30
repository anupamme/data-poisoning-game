"""
Round 49 — dynamic BR experiment on the reputation-based equilibrium.

The reviewer's named lever to 8/10. Round 48 showed reputation enters NE support in
3/5 seeds with mean weight 17% (alongside NormClip at ~83%). We deploy the average
of these reputation-bearing equilibria as the server's mixed policy and run NE3-style
BR: a committed adversary plays one attack against this mix.

Pre-committed interpretation:
  COLLAPSES (strengthens persistence claim):
    realized scaling ASR > 0.85 AND realized VoPD <= 0.05
    => "Even when complementarity is generated through genuinely orthogonal-signal
        defenses, persistence still destroys realized value."
  SURVIVES (identifies operative boundary):
    realized scaling ASR < 0.5 OR realized VoPD > 0.10
    => "Reputation-based defense crosses the persistence boundary; the operative
        condition is 'static complementarity + dynamic suppression of persistent BR'."
  MIXED (partial):
    realized scaling ASR in [0.5, 0.85] or VoPD in [0.05, 0.10]
    => "Partial survival; softening of the negative result."

Setup:
  Defense mix: NormClip 83% + reputation 17% (cross-seed average from Round 48)
  Attacks: model_scaling, backdoor_pixel
  Seeds: 42-51 (10 seeds)
  N=10, K=5, f=0.2, 50 rounds, cifar_cnn

Output: results/cifar10_reputation_eq_br/summary.json
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

DEFENSE_DIST = {"norm_clip": 0.83, "reputation": 0.17}  # Reputation equilibrium average
ATTACKS = ["model_scaling", "backdoor_pixel"]
SEEDS = list(range(42, 52))  # 10 seeds
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_reputation_eq_br")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init_summary():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "policy": DEFENSE_DIST,
        "seeds": [],
        "attacks": {a: {"per_seed": []} for a in ATTACKS},
        "pre_committed": {
            "COLLAPSES": "realized scaling ASR > 0.85 AND realized VoPD <= 0.05",
            "SURVIVES": "realized scaling ASR < 0.5 OR realized VoPD > 0.10",
            "MIXED": "in between",
        },
    }


def save_one(seed: int, attack: str, accuracy: float, asr: float, fedavg_rounds: int):
    s = load_or_init_summary()
    a = s["attacks"].setdefault(attack, {"per_seed": []})
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    a["per_seed"].append({"seed": seed, "accuracy": float(accuracy),
                           "attack_success_rate": float(asr),
                           "fedavg_rounds": int(fedavg_rounds)})
    if seed not in s["seeds"]:
        s["seeds"].append(seed)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(seed: int, attack: str) -> bool:
    s = load_or_init_summary()
    a = s["attacks"].get(attack, {})
    return any(e["seed"] == seed for e in a.get("per_seed", []))


def run_br(seed: int, attack_name: str) -> dict:
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
    attack = get_attack(attack_name)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 2000)
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
    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "attack_success_rate": float(asr),
        "norm_clip_rounds": nc_count,
        "reputation_rounds": rep_count,
    }


print(f"Round 49 dynamic BR on reputation equilibrium:")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Attacks: {ATTACKS}, Seeds: {SEEDS}")
print(f"  Output: {output_path}")
print(f"  Pre-committed interpretation in summary.json\n")

t0 = time.time()
runs_done = 0

for seed in SEEDS:
    for attack in ATTACKS:
        if has_run(seed, attack):
            print(f"  [skip] seed {seed} {attack} cached", flush=True)
            continue
        t_run = time.time()
        r = run_br(seed, attack)
        save_one(seed, attack, r["accuracy"], r["attack_success_rate"],
                  r["norm_clip_rounds"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {attack}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"(nc={r['norm_clip_rounds']}/rep={r['reputation_rounds']}) "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)", flush=True)

print(f"\n=== DYNAMIC BR ON REPUTATION EQUILIBRIUM COMPLETE ===")
s = load_or_init_summary()
for attack in ATTACKS:
    a = s["attacks"][attack]
    asrs = [e["attack_success_rate"] for e in a["per_seed"]]
    accs = [e["accuracy"] for e in a["per_seed"]]
    print(f"  {attack}: ASR={np.mean(asrs):.3f}+/-{np.std(asrs):.3f}  "
          f"acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}")
    print(f"    per-seed ASRs: {[round(x, 3) for x in asrs]}")
print(f"\nWall time: {(time.time()-t0)/60:.1f} min")

# Pre-committed interpretation
scaling_asrs = [e["attack_success_rate"] for e in s["attacks"]["model_scaling"]["per_seed"]]
scaling_mean = float(np.mean(scaling_asrs))
print(f"\n=== Pre-committed interpretation ===")
if scaling_mean > 0.85:
    print(f"  Scaling ASR = {scaling_mean:.3f} > 0.85 -> COLLAPSES likely")
elif scaling_mean < 0.5:
    print(f"  Scaling ASR = {scaling_mean:.3f} < 0.5 -> SURVIVES")
else:
    print(f"  Scaling ASR = {scaling_mean:.3f} in [0.5, 0.85] -> MIXED")
