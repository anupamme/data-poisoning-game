"""
N2 FAIL control at N=100/ResNet18 (Round-2 review).

The §4.1(B) scale transfer tests only PASS predictions (fg→rfa, fg→cm). This
adds ≥1 FAIL-predicted composition to show the criterion's FAIL predictions
also transfer. We run nc→rep (C2-FAIL) and fg→tm (C3-FAIL).

Config: N=100, K=10, f=0.05, ResNet18, CIFAR-10, alpha=0.5, 50 rounds, seeds 42-46
Output: results/scale_fail_control/summary.json
"""
import json
import os
import sys
import time
import gc
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
FAIL_PAIRS = [
    ("norm_clip", "reputation"),     # C2-FAIL
    ("foolsgold", "trimmed_mean"),   # C3-FAIL
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44, 45, 46]

NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ADV_FRACTION = 0.05
NUM_ROUNDS = 50
MODEL_NAME = "resnet18"

FL_CONFIG = FLConfig(
    num_clients=NUM_CLIENTS,
    clients_per_round=CLIENTS_PER_ROUND,
    num_rounds=NUM_ROUNDS,
)

output_dir = os.path.join(base_dir, "results", "scale_fail_control")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def clear_mps_cache():
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    gc.collect()


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "N2 FAIL control: scale transfer for FAIL-predicted pairs at N=100/ResNet18",
        "config": {
            "num_clients": NUM_CLIENTS,
            "clients_per_round": CLIENTS_PER_ROUND,
            "adversarial_fraction": ADV_FRACTION,
            "model": MODEL_NAME,
            "num_rounds": NUM_ROUNDS,
        },
        "pairs": {},
    }


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def has_run(d1, d2, attack, seed):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        return False
    if attack not in s["pairs"][pk]:
        return False
    return any(r["seed"] == seed for r in s["pairs"][pk][attack].get("per_seed", []))


def save_one(d1, d2, attack, seed, accuracy, asr):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2}
    if attack not in s["pairs"][pk]:
        s["pairs"][pk][attack] = {"per_seed": []}
    ps = [r for r in s["pairs"][pk][attack]["per_seed"] if r["seed"] != seed]
    ps.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["pairs"][pk][attack]["per_seed"] = ps
    asrs = [r["asr"] for r in ps]
    s["pairs"][pk][attack]["mean_asr"] = float(np.mean(asrs))
    s["pairs"][pk][attack]["std_asr"] = float(np.std(asrs))
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
            max_asr = max(max_asr, s["pairs"][pk][atk]["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, d1, d2, attack_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", NUM_CLIENTS, 0.5, seed)
    model = get_model(MODEL_NAME, num_classes)
    server = FederatedServer(model, device)

    num_adv = int(NUM_CLIENTS * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    else:
        internal_attack = "backdoor_pixel"
    attack = get_attack(internal_attack)

    clients = []
    for i in range(NUM_CLIENTS):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(NUM_ROUNDS):
        participant_ids = np.random.choice(
            NUM_CLIENTS,
            size=min(CLIENTS_PER_ROUND, NUM_CLIENTS),
            replace=False)
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size)
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    clear_mps_cache()
    return float(eval_result["accuracy"]), float(asr)


if __name__ == "__main__":
    total = len(FAIL_PAIRS) * len(ATTACKS) * len(SEEDS)
    print("=" * 70)
    print("  N2 FAIL CONTROL: Scale Transfer (N=100, ResNet18)")
    print("=" * 70)
    print(f"  Config: N={NUM_CLIENTS}, K={CLIENTS_PER_ROUND}, "
          f"f={ADV_FRACTION}, rounds={NUM_ROUNDS}, model={MODEL_NAME}")
    print(f"  FAIL pairs: {FAIL_PAIRS}")
    print(f"  Seeds: {SEEDS}   Total runs: {total}")
    print()

    t0 = time.time()
    done = 0
    for d1, d2 in FAIL_PAIRS:
        pk = pair_key(d1, d2)
        print(f"\n--- [FAIL] {d1}->{d2} ---")
        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(d1, d2, attack, seed):
                    print(f"  [skip] {attack} seed {seed}", flush=True)
                    done += 1
                    continue
                tr = time.time()
                acc, asr = run_one(seed, d1, d2, attack)
                save_one(d1, d2, attack, seed, acc, asr)
                done += 1
                eta = (time.time() - t0) / done * (total - done) / 3600
                print(f"  {attack} seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({(time.time()-tr)/60:.1f}min, {done}/{total}, ETA {eta:.1f}h)",
                      flush=True)

    print("\n" + "=" * 70)
    print("  N2 FAIL CONTROL — RESULTS")
    print("=" * 70)
    s = load_or_init()
    for d1, d2 in FAIL_PAIRS:
        pk = pair_key(d1, d2)
        pd = s["pairs"].get(pk, {})
        max_asr = pd.get("max_committed_asr", "?")
        sc = pd.get("committed_scaling", {}).get("mean_asr", "?")
        px = pd.get("committed_pixel", {}).get("mean_asr", "?")
        print(f"  {d1}->{d2}: max={max_asr:.3f}, scaling={sc:.3f}, pixel={px:.3f}")
    print(f"\n  Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"  Output: {output_path}")
