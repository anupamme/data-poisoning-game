"""
Wave 3: the two cells that discriminate a statistic-preservation screen from the criterion.

Pre-registration: experiments/pre_registration_wave3_emergent.md, which MUST be committed
before this runs. The guard below refuses to start otherwise, so the ordering is a
property of the code and not of anyone's discipline.

Why these four pairs and no others. Restricted to the region where invariance is both
algebraically settled and code-verified -- d1 emitting strictly positive per-client
coefficients, d2 reading an exactly scale-invariant statistic -- the space of ordered
pairs is exactly nine, and seven of them are already measured. The two unspent ones are
pairs 1 and 2 here, so this wave closes a census rather than sampling one. Pairs 3 and 4
are the controls that census cannot supply: foolsgold as d1 forces at least one client
coefficient to EXACTLY zero every round (measured, 2 of 5), which takes it outside the
strict hypothesis of the invariance proposition, and its downstream statistics were
measured moving in 9 of 9 rounds. They are therefore measured-C2-fail, not assumed-C2-fail.

    #  pair                          C1(gated)  C2            role
    1  reputation -> cos_krum        holds      holds  (0/9)  certified: precision
    2  reputation -> cos_reputation  fails      holds  (0/9)  C1-fail & C2-true: recall
    3  foolsgold  -> cos_krum        fails      FAILS  (9/9)  control
    4  foolsgold  -> cos_reputation  fails      FAILS  (9/9)  control

Mechanism evidence for the C2 column: results/wave3_invariance_check.json, written by
experiments/verify_wave3_invariance.py BEFORE the predictions were frozen.

No standalone baselines are run here. Every C1 and C0 input is already frozen at this
exact protocol in results/metric_swap_baselines/summary.json and
results/pure_defense_baselines/summary.json; re-running them would spend 6 h to reproduce
numbers the pre-registration already cites.

Protocol, identical to run_wave2_held_out.py and the metric_swap suite:
    cifar10 / cifar_cnn, N=10, K=5, f=0.2, 50 rounds, Dirichlet 0.5, tau=5.0
    attacks committed_scaling, committed_pixel; seeds 42, 43, 44
    4 pairs x 2 attacks x 3 seeds = 24 runs, ~13 min/run, ~5 h

Output: results/wave3_emergent/summary.json
"""
import json
import os
import subprocess
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import generic_compose

# --- Configuration ---
PAIRS = [
    # census, unspent: strictly-positive d1 x exactly-invariant d2
    ("reputation", "cos_krum"),          # predicted LOW  (the only predicted LOW)
    ("reputation", "cos_reputation"),    # predicted HIGH (C1 fails on pixel)
    # measured-C2-fail controls
    ("foolsgold", "cos_krum"),           # predicted HIGH
    ("foolsgold", "cos_reputation"),     # predicted HIGH
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
TAU = 5.0

PREREG = os.path.join(base_dir, "experiments", "pre_registration_wave3_emergent.md")
output_dir = os.path.join(base_dir, "results", "wave3_emergent")
output_path = os.path.join(output_dir, "summary.json")


# --- Pre-registration gate ---
def prereg_commit():
    """The commit that added the pre-registration, or None if it is not committed.

    This is the whole non-post-hoc guarantee of the wave, so it is enforced rather than
    asserted in prose: an uncommitted or dirty pre-registration aborts the run. Reported
    the same way analyze_comparability.py reports its own frozen-file commit.
    """
    rel = os.path.relpath(PREREG, base_dir)
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", rel],
            cwd=base_dir, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not out:
        return None
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", rel],
        cwd=base_dir, capture_output=True, text=True).stdout.strip()
    return None if dirty else out


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "Wave 3: certified and C1-fail-and-C2-true cells, plus "
                       "measured-C2-fail controls, all with an admissible d1",
        "pre_registration": os.path.relpath(PREREG, base_dir),
        "protocol": {
            "dataset": "cifar10", "model": "cifar_cnn",
            "num_clients": FL_CONFIG.num_clients,
            "clients_per_round": FL_CONFIG.clients_per_round,
            "adv_fraction": ADV_FRACTION, "num_rounds": FL_CONFIG.num_rounds,
            "dirichlet_alpha": 0.5, "tau": TAU, "seeds": SEEDS, "attacks": ATTACKS,
        },
        "pairs": {},
    }


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def has_run(d1, d2, attack, seed):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"] or attack not in s["pairs"][pk]:
        return False
    return any(r["seed"] == seed for r in s["pairs"][pk][attack].get("per_seed", []))


def save_one(d1, d2, attack, seed, accuracy, asr):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2}
    if attack not in s["pairs"][pk]:
        s["pairs"][pk][attack] = {"per_seed": []}
    per_seed = [r for r in s["pairs"][pk][attack]["per_seed"] if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    per_seed.sort(key=lambda r: r["seed"])
    s["pairs"][pk][attack]["per_seed"] = per_seed
    s["pairs"][pk][attack]["mean_asr"] = float(np.mean([r["asr"] for r in per_seed]))
    s["pairs"][pk][attack]["std_asr"] = float(np.std([r["asr"] for r in per_seed]))
    s["pairs"][pk][attack]["mean_accuracy"] = float(
        np.mean([r["accuracy"] for r in per_seed]))
    max_asr = 0.0
    for atk in ATTACKS:
        cell = s["pairs"][pk].get(atk, {})
        if "mean_asr" in cell:
            max_asr = max(max_asr, cell["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- One run ---
def run_one(seed, d1, d2, attack_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed)
    server = FederatedServer(get_model("cifar_cnn", num_classes), device)

    adversarial_ids = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    internal = {"committed_scaling": "model_scaling",
                "committed_pixel": "backdoor_pixel"}.get(attack_name, attack_name)
    attack = get_attack(internal)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False)
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size)
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        server.apply_update(generic_compose(server, updates, d1, d2, tau=TAU))
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Main ---
if __name__ == "__main__":
    commit = prereg_commit()
    if commit is None:
        raise SystemExit(
            f"REFUSING TO RUN: {os.path.relpath(PREREG, base_dir)} is not committed "
            "(or has uncommitted changes).\nThe wave's predictions must be frozen in git "
            "before any of its data exists; see the non-negotiables in that file.")
    print("=== Wave 3: emergent-suppression cells with an admissible d1 ===")
    print(f"  Pre-registration committed at {commit}")
    print(f"  Pairs: {len(PAIRS)}   attacks: {ATTACKS}   seeds: {SEEDS}")
    total_runs = len(PAIRS) * len(ATTACKS) * len(SEEDS)
    print(f"  Total runs: {total_runs}")
    print(f"  FL config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}, cifar_cnn, tau={TAU}")
    print()

    t0 = time.time()
    completed = skipped = 0
    for pair_idx, (d1, d2) in enumerate(PAIRS):
        print(f"[{pair_idx+1}/{len(PAIRS)}] {pair_key(d1, d2)}", flush=True)
        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(d1, d2, attack, seed):
                    skipped += 1
                    continue
                t_run = time.time()
                acc, asr = run_one(seed, d1, d2, attack)
                save_one(d1, d2, attack, seed, acc, asr)
                completed += 1
                print(f"    {attack} seed={seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({time.time()-t_run:.0f}s) "
                      f"[{completed+skipped}/{total_runs}]", flush=True)

    # --- Summary. Scoring is NOT done here; see analyze_wave3_emergent.py ---
    print(f"\n{'='*78}")
    print("WAVE 3 RESULTS (raw; scored against the pre-registration separately)")
    print(f"{'='*78}")
    print(f"{'Pair':<34}{'scaling':>9}{'pixel':>9}{'max':>8}{'min acc':>9}")
    print("-" * 78)
    s = load_or_init()
    for d1, d2 in PAIRS:
        pk = pair_key(d1, d2)
        if pk not in s["pairs"]:
            continue
        p = s["pairs"][pk]
        accs = [p[a]["mean_accuracy"] for a in ATTACKS if a in p]
        print(f"  {pk:<32}"
              f"{p.get('committed_scaling', {}).get('mean_asr', float('nan')):>9.3f}"
              f"{p.get('committed_pixel', {}).get('mean_asr', float('nan')):>9.3f}"
              f"{p.get('max_committed_asr', float('nan')):>8.3f}"
              f"{(min(accs) if accs else float('nan')):>9.3f}")

    print(f"\nCompleted: {completed} new runs, {skipped} cached")
    print(f"Wall time: {(time.time()-t0)/3600:.2f} h")
    print(f"Output: {output_path}")
