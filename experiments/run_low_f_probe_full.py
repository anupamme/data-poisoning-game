"""
Low-f / long-horizon probe with all three arms: committed_scaling, committed_pixel, oracle.

Reviewer gap: §5.5 only measures committed_scaling ASR at f=0.05. High committed ASR
demonstrates persistence, but doesn't measure realized VoPD (which requires oracle arm).

Arms:
  - committed_scaling: same as run_low_f_probe.py (cached results reused)
  - committed_pixel: backdoor_pixel only, no scaling
  - oracle: plays model_scaling on NormClip rounds, backdoor_pixel on FedAvg rounds

Config: f=0.05, N=10 (1 adversarial), 200 rounds, NE3 mix (FedAvg 26% + NormClip 74%)
Seeds: 42-46 (5-seed pilot)

Output: results/low_f_probe_full/summary.json
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
ADV_FRACTION = 0.05
NUM_ROUNDS = 200
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]

FL_CONFIG = FLConfig(
    num_clients=10,
    clients_per_round=5,
    num_rounds=NUM_ROUNDS,
)

output_dir = os.path.join(base_dir, "results", "low_f_probe_full")
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
        "adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES},
    }


def has_run(adv_pol, seed):
    s = load_or_init()
    if adv_pol not in s["adversary_policies"]:
        return False
    return any(r["seed"] == seed for r in s["adversary_policies"][adv_pol]["per_seed"])


def save_one(adv_pol, seed, accuracy, asr):
    s = load_or_init()
    if adv_pol not in s["adversary_policies"]:
        s["adversary_policies"][adv_pol] = {"per_seed": []}
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["adversary_policies"][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = max(1, int(FL_CONFIG.num_clients * ADV_FRACTION))
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Poison datasets based on policy
    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            if adv_pol == "committed_pixel":
                ds = attack_pixel.poison_dataset(ds)
            else:
                # committed_scaling and oracle both use scaling's poisoned data
                # (same trigger as pixel for shared-trigger setup)
                ds = attack_scaling.poison_dataset(ds)
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

        d_this = defense_rng.choice(defenses, p=probs)

        # Oracle selects attack based on defense drawn this round
        if adv_pol == "oracle":
            if d_this == "norm_clip":
                current_attack = attack_scaling  # scaling evades norm_clip better
            else:
                current_attack = attack_pixel  # pixel evades fedavg better
        elif adv_pol == "committed_scaling":
            current_attack = attack_scaling
        else:
            current_attack = attack_pixel

        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = current_attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


print("=== Low-f / Long-Horizon Probe — Full (All 3 Arms) ===")
print(f"  f={ADV_FRACTION} ({max(1, int(FL_CONFIG.num_clients * ADV_FRACTION))} adversarial client)")
print(f"  rounds={NUM_ROUNDS}, defense={DEFENSE_DIST}")
print(f"  Arms: {ADV_POLICIES}")
print(f"  Oracle: model_scaling on NormClip rounds, backdoor_pixel on FedAvg rounds")
print(f"  Seeds: {SEEDS}\n")

t0 = time.time()
for adv_pol in ADV_POLICIES:
    for seed in SEEDS:
        if has_run(adv_pol, seed):
            print(f"  [skip] {adv_pol} seed {seed}", flush=True)
            continue
        t_run = time.time()
        acc, asr = run_one(seed, adv_pol)
        save_one(adv_pol, seed, acc, asr)
        dt = time.time() - t_run
        print(f"  {adv_pol} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

print(f"\n=== LOW-F PROBE FULL RESULTS ===")
s = load_or_init()
asrs_by_pol = {}
for adv_pol in ADV_POLICIES:
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    if runs:
        asrs = [r["asr"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        asrs_by_pol[adv_pol] = asrs
        print(f"  {adv_pol}: ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}, "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

# Compute realized VoPD if all arms complete
if all(p in asrs_by_pol for p in ADV_POLICIES):
    oracle_asrs = np.array(asrs_by_pol["oracle"])
    scaling_asrs = np.array(asrs_by_pol["committed_scaling"])
    pixel_asrs = np.array(asrs_by_pol["committed_pixel"])
    best_committed = np.maximum(scaling_asrs, pixel_asrs)
    vopd = oracle_asrs - best_committed
    print(f"\n  Realized VoPD (oracle - max(committed)): {np.mean(vopd):.3f}±{np.std(vopd):.3f}")
    print(f"  Per-seed VoPD: {[round(x, 3) for x in vopd]}")
    positive_frac = np.mean(vopd > 0)
    print(f"  Fraction positive: {positive_frac:.1%}")
    if np.mean(vopd) > 0.02:
        print(f"  VERDICT: Oracle retains value at f=0.05 — persistence alone insufficient for collapse.")
    elif np.mean(vopd) > -0.02:
        print(f"  VERDICT: VoPD ≈ 0 at f=0.05 — collapse extends to oracle (persistence dominates).")
    else:
        print(f"  VERDICT: VoPD < 0 — oracle HURTS at f=0.05 (switching is strictly worse).")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
