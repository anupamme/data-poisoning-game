"""
Round 62 — Low-f / long-horizon probe (Part C).

Reviewer gap: "The negative headline is demonstrated only at f=0.2, 50 rounds.
At realistic low f (0.01–0.05) with long horizons (200–500 rounds), adversarial
participation is sparse and suppression windows are long — precisely the regime
where the collapse might not hold."

Tγ ≈ 7 rounds implies backdoors decay quickly under sustained suppression.
At f=0.05 (1/10 clients adversarial) with 200 rounds, a committed adversary
participates ~K*f = 0.5 rounds/round on average, leaving long suppression windows.
The persistence mechanism requires τ_embed < τ_suppress; at f=0.05 the adversary
gets far fewer rounds to embed the trigger.

Config: f=0.05, 200 rounds, N=10 (1 adversarial client), 5 seeds
Menu: NE3 mix (FedAvg 26% + NormClip 74%), committed model_scaling adversary
(same as the main collapse experiment at f=0.2, 50 rounds: realized ASR 0.961)

Output: results/low_f_probe/summary.json
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

SEEDS = list(range(42, 47))
ADV_FRACTION = 0.05   # 1 adversarial client out of 10
NUM_ROUNDS = 200
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}  # NE3 mix

FL_CONFIG = FLConfig(
    num_clients=10,
    clients_per_round=5,
    num_rounds=NUM_ROUNDS,
)

output_dir = os.path.join(base_dir, "results", "low_f_probe")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "adv_fraction": ADV_FRACTION,
        "num_rounds": NUM_ROUNDS,
        "defense_dist": DEFENSE_DIST,
        "attack": "model_scaling",
        "per_seed": [],
    }


def has_run(seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s["per_seed"])


def save_one(seed, accuracy, asr):
    s = load_or_init()
    s["per_seed"] = [r for r in s["per_seed"] if r["seed"] != seed]
    s["per_seed"].append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_low_f(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = max(1, int(FL_CONFIG.num_clients * ADV_FRACTION))  # at least 1
    adv_ids = set(range(num_adv))
    attack = get_attack("model_scaling")

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    defense_rng = np.random.default_rng(seed + 99000)

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

        d_this = defense_rng.choice(defenses, p=probs)
        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


print("=== Low-f / Long-Horizon Probe (Round 62) ===")
print(f"  f={ADV_FRACTION} ({int(FL_CONFIG.num_clients * ADV_FRACTION)} adversarial clients)")
print(f"  rounds={NUM_ROUNDS}, defense={DEFENSE_DIST}, attack=model_scaling")
print(f"  Comparison: f=0.2, 50 rounds, NE3 mix: scaling ASR = 0.961 ± 0.052 (n=30)")
print(f"  Seeds: {SEEDS}\n")

t0 = time.time()
for seed in SEEDS:
    if has_run(seed):
        print(f"  [skip] seed {seed} cached", flush=True)
        continue
    t_run = time.time()
    acc, asr = run_low_f(seed)
    save_one(seed, acc, asr)
    dt = time.time() - t_run
    print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

print(f"\n=== LOW-F PROBE COMPLETE ===")
s = load_or_init()
asrs = [r["asr"] for r in s["per_seed"]]
accs = [r["accuracy"] for r in s["per_seed"]]
if asrs:
    print(f"  Scaling ASR: {np.mean(asrs):.3f} ± {np.std(asrs):.3f}")
    print(f"  Accuracy:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  Per-seed ASRs: {[round(x,3) for x in asrs]}")
    if np.mean(asrs) > 0.85:
        print(f"  VERDICT: Persistence HOLDS at f=0.05, 200 rounds.")
    elif np.mean(asrs) > 0.4:
        print(f"  VERDICT: PARTIAL — persistence weakened but not broken.")
    else:
        print(f"  VERDICT: Persistence FAILS at f=0.05, 200 rounds — headline limited to high-f regime.")
print(f"Wall time: {(time.time()-t0)/60:.1f} min")
