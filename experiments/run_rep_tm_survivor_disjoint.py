"""
Disjoint-trigger replication of the rep+tm survivor experiment.

Addresses reviewer concern: model_scaling and backdoor_pixel share the same trigger
(bottom-right 4x4 = 1.0). The oracle per-round switching accumulates onto a single
backdoor pattern, plausibly inflating oracle ASR and the +0.032 VoPD.

FIX: model_scaling_disjoint uses a DIFFERENT trigger (bottom-LEFT 4x4 = 1.0).
Each trigger is evaluated separately at end-of-training:
  - asr_pixel: bottom-right trigger (backdoor_pixel only)
  - asr_scaling: bottom-left trigger (model_scaling_disjoint only)
  - oracle realized ASR: weighted sum by fraction of rounds each attack was active

If VoPD holds with disjoint triggers, the shared-trigger concern is dismissed.
If VoPD collapses, the survivor was an artifact of union-embedding.

Setup: Same as run_rep_tm_survivor.py (rep30/tm70, 50 rounds, N=10, K=5, f=0.2).
Seeds: 42-46 (5-seed pilot); extend to 42-71 if directional.
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
from experiments.run_payoff_matrix import evaluate_backdoor, evaluate_scaling_disjoint_backdoor

DEFENSE_DIST = {"reputation": 0.30, "trimmed_mean": 0.70}
ORACLE_BEST_ATTACK = {"reputation": "backdoor_pixel", "trimmed_mean": "model_scaling_disjoint"}
ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
SEEDS = list(range(42, 72))  # n=30 scale-up (reviewer Q3)
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_rep_tm_survivor_disjoint")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "policy": DEFENSE_DIST,
        "oracle_best_attack_per_defense": ORACLE_BEST_ATTACK,
        "seeds": SEEDS,
        "disjoint_trigger": True,
        "description": "Disjoint triggers: pixel=bottom-right 4x4, scaling=bottom-left 4x4",
        "adversary_policies": {p: {"per_seed": []} for p in ADVERSARY_POLICIES},
    }


def save_one(adv_pol, seed, accuracy, asr_scaling, asr_pixel, defense_counts, attack_counts):
    s = load_or_init()
    a = s["adversary_policies"][adv_pol]
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_scaling_final": float(asr_scaling),
        "asr_pixel_final": float(asr_pixel),
        "defense_counts": {k: int(v) for k, v in defense_counts.items()},
        "attack_counts": {k: int(v) for k, v in attack_counts.items()},
    }
    a["per_seed"].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(adv_pol, seed):
    s = load_or_init()
    a = s["adversary_policies"].get(adv_pol, {})
    return any(e["seed"] == seed for e in a.get("per_seed", []))


def run_adversary(seed, adv_pol):
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

    # DISJOINT triggers: scaling uses bottom-left, pixel uses bottom-right
    scaling_attack = get_attack("model_scaling_disjoint")
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

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 5000)
    current_lr = FL_CONFIG.learning_rate

    defense_counts = {d: 0 for d in defenses}
    attack_counts = {"model_scaling_disjoint": 0, "backdoor_pixel": 0}

    for _ in range(FL_CONFIG.num_rounds):
        d_this = rng.choice(defenses, p=probs)
        defense_counts[d_this] += 1

        if adv_pol == "committed_scaling":
            attack_this = "model_scaling_disjoint"
        elif adv_pol == "committed_pixel":
            attack_this = "backdoor_pixel"
        elif adv_pol == "oracle":
            attack_this = ORACLE_BEST_ATTACK[d_this]
        else:
            raise ValueError(adv_pol)
        attack_counts[attack_this] += 1

        if attack_this == "model_scaling_disjoint":
            poisoned = scaling_poisoned
            attack_obj = scaling_attack
        else:
            poisoned = pixel_poisoned
            attack_obj = pixel_attack

        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            if cid in adversarial_ids:
                ds = poisoned[cid]
            else:
                ds = clean_datasets[cid]
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

    # Evaluate BOTH triggers SEPARATELY
    eval_result = server.evaluate(test_dataset)
    asr_pixel_final = evaluate_backdoor(server.global_model, test_dataset, device=device)
    asr_scaling_final = evaluate_scaling_disjoint_backdoor(server.global_model, test_dataset, device=device)

    return {
        "accuracy": float(eval_result["accuracy"]),
        "asr_scaling_final": float(asr_scaling_final),
        "asr_pixel_final": float(asr_pixel_final),
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }


print("=== Disjoint-Trigger Rep+TM Survivor ===")
print(f"  Triggers: pixel=bottom-right 4x4, scaling=bottom-LEFT 4x4")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Oracle: {ORACLE_BEST_ATTACK}")
print(f"  Seeds: {SEEDS}, Policies: {ADVERSARY_POLICIES}\n")

t0 = time.time()
runs_done = 0

for adv_pol in ADVERSARY_POLICIES:
    for seed in SEEDS:
        if has_run(adv_pol, seed):
            print(f"  [skip] {adv_pol} seed {seed}", flush=True)
            continue
        t_run = time.time()
        r = run_adversary(seed, adv_pol)
        save_one(adv_pol, seed, r["accuracy"], r["asr_scaling_final"],
                 r["asr_pixel_final"], r["defense_counts"], r["attack_counts"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  {adv_pol} seed {seed}: acc={r['accuracy']:.3f} "
              f"ASR_scaling={r['asr_scaling_final']:.3f} ASR_pixel={r['asr_pixel_final']:.3f} "
              f"({dt:.0f}s)", flush=True)

print(f"\n=== DISJOINT-TRIGGER SURVIVOR COMPLETE ===")
s = load_or_init()
for adv_pol in ADVERSARY_POLICIES:
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    if runs:
        asr_s = [r["asr_scaling_final"] for r in runs]
        asr_p = [r["asr_pixel_final"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        print(f"\n  {adv_pol} (n={len(runs)}):")
        print(f"    ASR_scaling (bottom-left): {np.mean(asr_s):.3f} +/- {np.std(asr_s):.3f}")
        print(f"    ASR_pixel (bottom-right):  {np.mean(asr_p):.3f} +/- {np.std(asr_p):.3f}")
        print(f"    accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

# Compute realized VoPD with disjoint triggers
print(f"\n=== VoPD Computation (Disjoint Triggers) ===")
all_have_data = all(
    len(s["adversary_policies"][p]["per_seed"]) > 0 for p in ADVERSARY_POLICIES
)
if all_have_data:
    # For committed_scaling: only scaling trigger matters
    committed_scaling_asrs = [r["asr_scaling_final"]
                              for r in s["adversary_policies"]["committed_scaling"]["per_seed"]]
    # For committed_pixel: only pixel trigger matters
    committed_pixel_asrs = [r["asr_pixel_final"]
                            for r in s["adversary_policies"]["committed_pixel"]["per_seed"]]
    # For oracle: weighted average of both triggers by round fraction
    oracle_runs = s["adversary_policies"]["oracle"]["per_seed"]
    oracle_realized_asrs = []
    for r in oracle_runs:
        ac = r["attack_counts"]
        total_rounds = ac["model_scaling_disjoint"] + ac["backdoor_pixel"]
        frac_scaling = ac["model_scaling_disjoint"] / total_rounds
        frac_pixel = ac["backdoor_pixel"] / total_rounds
        realized = frac_scaling * r["asr_scaling_final"] + frac_pixel * r["asr_pixel_final"]
        oracle_realized_asrs.append(realized)

    mean_cs = np.mean(committed_scaling_asrs)
    mean_cp = np.mean(committed_pixel_asrs)
    mean_oracle = np.mean(oracle_realized_asrs)
    committed_max = max(mean_cs, mean_cp)

    print(f"  committed_scaling (scaling trigger only): {mean_cs:.3f} +/- {np.std(committed_scaling_asrs):.3f}")
    print(f"  committed_pixel (pixel trigger only):     {mean_cp:.3f} +/- {np.std(committed_pixel_asrs):.3f}")
    print(f"  oracle (weighted avg of both triggers):   {mean_oracle:.3f} +/- {np.std(oracle_realized_asrs):.3f}")
    print(f"  committed_max: {committed_max:.3f}")
    print(f"  REALIZED VoPD (disjoint) = {mean_oracle - committed_max:.3f}")

    # Per-seed paired VoPD
    n_seeds = min(len(committed_scaling_asrs), len(committed_pixel_asrs), len(oracle_realized_asrs))
    per_seed_vopd = []
    for i in range(n_seeds):
        best_committed = max(committed_scaling_asrs[i], committed_pixel_asrs[i])
        vopd_i = oracle_realized_asrs[i] - best_committed
        per_seed_vopd.append(vopd_i)
    print(f"  Per-seed VoPD: {[round(v,3) for v in per_seed_vopd]}")
    print(f"  Mean per-seed VoPD: {np.mean(per_seed_vopd):.3f} +/- {np.std(per_seed_vopd):.3f}")

    # Compare to shared-trigger baseline
    print(f"\n  Comparison (shared-trigger, n=30):")
    print(f"    committed_scaling: 0.871 +/- 0.095")
    print(f"    committed_pixel:   0.747 +/- 0.106")
    print(f"    oracle:            0.915 +/- 0.065")
    print(f"    realized VoPD:    +0.032 +/- 0.054")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
