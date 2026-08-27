"""
Independent survivor validation: fg→cm composition dominates temporal mixing.

Tests whether fg→cm (selected purely by criterion PASS status, NOT from the
189-config payoff-matrix scan) dominates temporal mixing of fg/cm, providing
an independent replication of the composition-dominates-mixing claim.

Two conditions:
  1. Composed: always apply fg→cm (FoolsGold transform then CoordMedian aggregate)
  2. Mixed: per-round random selection from {foolsgold, coord_median} with p=(0.5, 0.5)

Three adversary policies: committed_scaling, committed_pixel, oracle
Oracle: plays model_scaling regardless (best single attack vs both defenses)

Config: N=10, K=5, f=0.2, cifar_cnn, 50 rounds, seeds 42-46
Output: results/fg_cm_survivor/summary.json
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
from experiments.run_all_compositions import generic_compose

# --- Configuration ---
SEEDS = [42, 43, 44, 45, 46]
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
CONDITIONS = ["composed", "mixed"]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

# Oracle table: best attack against each pure defense
# fg alone: scaling ASR moderate, pixel ASR lower → play scaling
# cm alone: scaling ASR 0.450, pixel ASR 0.377 → play scaling
ORACLE_BEST_ATTACK = {
    "foolsgold": "model_scaling",
    "coord_median": "model_scaling",
}

output_dir = os.path.join(base_dir, "results", "fg_cm_survivor")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "fg→cm independent survivor: composition vs temporal mixing",
        "config": {
            "num_clients": FL_CONFIG.num_clients,
            "clients_per_round": FL_CONFIG.clients_per_round,
            "adversarial_fraction": ADV_FRACTION,
            "num_rounds": FL_CONFIG.num_rounds,
            "model": "cifar_cnn",
        },
        "composed": {
            "committed_scaling": {"per_seed": []},
            "committed_pixel": {"per_seed": []},
            "oracle": {"per_seed": []},
        },
        "mixed": {
            "defense_policy": {"foolsgold": 0.5, "coord_median": 0.5},
            "oracle_best_attack": ORACLE_BEST_ATTACK,
            "committed_scaling": {"per_seed": []},
            "committed_pixel": {"per_seed": []},
            "oracle": {"per_seed": []},
        },
    }


def has_run(condition, adv_pol, seed):
    s = load_or_init()
    per_seed = s[condition][adv_pol].get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(condition, adv_pol, seed, accuracy, asr, extra=None):
    s = load_or_init()
    per_seed = s[condition][adv_pol]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    entry = {"seed": seed, "accuracy": float(accuracy), "asr": float(asr)}
    if extra:
        entry.update(extra)
    per_seed.append(entry)
    s[condition][adv_pol]["per_seed"] = per_seed
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Composed condition: always fg→cm ---
def run_composed(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    # Determine attack
    if adv_pol == "committed_scaling":
        attack = get_attack("model_scaling")
    elif adv_pol == "committed_pixel":
        attack = get_attack("backdoor_pixel")
    elif adv_pol == "oracle":
        # Oracle always plays model_scaling (best vs both fg and cm)
        attack = get_attack("model_scaling")
    else:
        raise ValueError(f"Unknown policy: {adv_pol}")

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
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
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, "foolsgold", "coord_median", tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Mixed condition: per-round random fg or cm ---
def run_mixed(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Prepare clients for both attacks
    clients_scaling = []
    clients_pixel = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds_s = attack_scaling.poison_dataset(ds)
            ds_p = attack_pixel.poison_dataset(ds)
        else:
            ds_s = ds
            ds_p = ds
        clients_scaling.append(FederatedClient(i, ds_s, device))
        clients_pixel.append(FederatedClient(i, ds_p, device))

    # Defense schedule: random 50/50 fg/cm
    defense_rng = np.random.default_rng(seed + 77777)
    defense_schedule = [defense_rng.choice(["foolsgold", "coord_median"])
                        for _ in range(FL_CONFIG.num_rounds)]

    defense_counts = {"foolsgold": 0, "coord_median": 0}
    current_lr = FL_CONFIG.learning_rate

    for r in range(FL_CONFIG.num_rounds):
        defense = defense_schedule[r]
        defense_counts[defense] += 1

        # Determine attack for this round
        if adv_pol == "committed_scaling":
            attack = attack_scaling
            clients = clients_scaling
        elif adv_pol == "committed_pixel":
            attack = attack_pixel
            clients = clients_pixel
        elif adv_pol == "oracle":
            # Oracle: pick best attack for drawn defense
            best = ORACLE_BEST_ATTACK[defense]
            if best == "model_scaling":
                attack = attack_scaling
                clients = clients_scaling
            else:
                attack = attack_pixel
                clients = clients_pixel
        else:
            raise ValueError(f"Unknown policy: {adv_pol}")

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

        # Apply single defense (not composition — this IS temporal mixing)
        aggregated = server.aggregate(updates, method=defense, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr), defense_counts


# --- Main ---
if __name__ == "__main__":
    total_runs = len(CONDITIONS) * len(ADV_POLICIES) * len(SEEDS)
    print("=" * 70)
    print("  FG→CM INDEPENDENT SURVIVOR: Composition vs Temporal Mixing")
    print("=" * 70)
    print(f"  Config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}")
    print(f"  Composed: always fg→cm")
    print("  Mixed: 50/50 random {foolsgold, coord_median} per round")
    print(f"  Oracle: always model_scaling (best vs both defenses)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print()

    t0 = time.time()
    runs_done = 0

    # --- Composed condition ---
    print("\n" + "=" * 70)
    print("  COMPOSED CONDITION: fg→cm every round")
    print("=" * 70)

    for adv_pol in ADV_POLICIES:
        print(f"\n--- Composed | {adv_pol} ---")
        for seed in SEEDS:
            if has_run("composed", adv_pol, seed):
                print(f"  [skip] seed {seed}", flush=True)
                runs_done += 1
                continue

            t_run = time.time()
            acc, asr = run_composed(seed, adv_pol)
            save_one("composed", adv_pol, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run

            elapsed = time.time() - t0
            avg = elapsed / runs_done
            remaining = (total_runs - runs_done) * avg
            eta_h = remaining / 3600

            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                  f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

    # --- Mixed condition ---
    print("\n" + "=" * 70)
    print("  MIXED CONDITION: 50/50 random fg/cm per round")
    print("=" * 70)

    for adv_pol in ADV_POLICIES:
        print(f"\n--- Mixed | {adv_pol} ---")
        for seed in SEEDS:
            if has_run("mixed", adv_pol, seed):
                print(f"  [skip] seed {seed}", flush=True)
                runs_done += 1
                continue

            t_run = time.time()
            acc, asr, d_counts = run_mixed(seed, adv_pol)
            save_one("mixed", adv_pol, seed, acc, asr, extra={"defense_counts": d_counts})
            runs_done += 1
            dt = time.time() - t_run

            elapsed = time.time() - t0
            avg = elapsed / runs_done
            remaining = (total_runs - runs_done) * avg
            eta_h = remaining / 3600

            d_str = " ".join(f"{k}:{v}" for k, v in d_counts.items())
            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} [{d_str}] "
                  f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  FG→CM SURVIVOR — RESULTS")
    print("=" * 70)

    s = load_or_init()

    print("\n  COMPOSED (fg→cm every round):")
    print(f"  {'Policy':<25s} {'ASR (mean±std)':<20s} {'Accuracy'}")
    print(f"  {'-'*60}")
    composed_asrs = {}
    for pol in ADV_POLICIES:
        runs = s["composed"][pol]["per_seed"]
        if runs:
            asrs = [r["asr"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            composed_asrs[pol] = np.mean(asrs)
            print(f"  {pol:<25s} {np.mean(asrs):.3f}±{np.std(asrs):.3f}"
                  f"        {np.mean(accs):.3f}±{np.std(accs):.3f} (n={len(runs)})")

    print("\n  MIXED (50/50 fg/cm per round):")
    print(f"  {'Policy':<25s} {'ASR (mean±std)':<20s} {'Accuracy'}")
    print(f"  {'-'*60}")
    mixed_asrs = {}
    for pol in ADV_POLICIES:
        runs = s["mixed"][pol]["per_seed"]
        if runs:
            asrs = [r["asr"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            mixed_asrs[pol] = np.mean(asrs)
            print(f"  {pol:<25s} {np.mean(asrs):.3f}±{np.std(asrs):.3f}"
                  f"        {np.mean(accs):.3f}±{np.std(accs):.3f} (n={len(runs)})")

    # Comparison
    if composed_asrs and mixed_asrs:
        composed_max = max(composed_asrs.get("committed_scaling", 0),
                          composed_asrs.get("committed_pixel", 0))
        mixed_max = max(mixed_asrs.get("committed_scaling", 0),
                       mixed_asrs.get("committed_pixel", 0))
        mixed_oracle = mixed_asrs.get("oracle", 0)
        mixed_vopd = mixed_oracle - mixed_max

        print(f"\n  COMPARISON:")
        print(f"    Composed max committed ASR: {composed_max:.3f}")
        print(f"    Mixed max committed ASR:    {mixed_max:.3f}")
        print(f"    Mixed oracle ASR:           {mixed_oracle:.3f}")
        print(f"    Mixed VoPD:                 {mixed_vopd:+.3f}")
        print(f"    Composition dominates:      {'YES' if composed_max < mixed_max else 'NO'}")

    wall = (time.time() - t0) / 60
    print(f"\n  Wall time: {wall:.1f} min")
    print(f"  Output: {output_path}")
