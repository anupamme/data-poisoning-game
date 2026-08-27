"""
Cross-distribution validation: test criterion predictions on CIFAR-100 and ResNet18 at f=0.2.

The criterion was developed on CIFAR-10/CNN/N=10/f=0.2. We test whether PASS/FAIL
predictions transfer to:
  (a) CIFAR-100/CNN (different output distribution, same architecture)
  (b) CIFAR-10/ResNet18/N=10/f=0.2 (different architecture, same dataset/f)

Key pairs tested (5 each condition):
  PASS: fg->rfa, fg->cm, rep->cm
  C2-FAIL: nc->rep, nc->fg
  C3-FAIL: fg->tm, nc->rfa

Seeds: 42-46 (5 seeds per pair)
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform, generic_compose
from experiments.run_payoff_matrix import evaluate_backdoor

# --- Configuration ---
PAIRS = [
    # PASS predictions
    ("foolsgold", "rfa"),            # PASS: C1+C2+C3
    ("foolsgold", "coord_median"),   # PASS: C1+C2+C3
    ("reputation", "coord_median"),  # PASS: C1+C2+C3
    # FAIL predictions
    ("norm_clip", "reputation"),     # C2-FAIL
    ("norm_clip", "foolsgold"),      # C2-FAIL
    ("foolsgold", "trimmed_mean"),   # C3-FAIL
    ("norm_clip", "rfa"),            # C3-FAIL
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]  # 3 seeds for speed
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

# Start with ResNet18 (faster, same 10-class output), then CIFAR-100
CONDITIONS = [
    {"dataset": "cifar10", "model": "resnet18", "label": "cifar10_resnet18"},
    {"dataset": "cifar100", "model": "cifar_cnn", "label": "cifar100_cnn"},
]


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def get_output_path(condition_label):
    output_dir = os.path.join(base_dir, "results", "cross_distribution", condition_label)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "summary.json")


def load_or_init(condition_label):
    path = get_output_path(condition_label)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"description": f"Cross-distribution validation: {condition_label}", "pairs": {}}


def has_run(condition_label, d1, d2, attack, seed):
    s = load_or_init(condition_label)
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        return False
    if attack not in s["pairs"][pk]:
        return False
    per_seed = s["pairs"][pk][attack].get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(condition_label, d1, d2, attack, seed, accuracy, asr):
    s = load_or_init(condition_label)
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2}
    if attack not in s["pairs"][pk]:
        s["pairs"][pk][attack] = {"per_seed": []}
    per_seed = s["pairs"][pk][attack]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["pairs"][pk][attack]["per_seed"] = per_seed
    asrs = [r["asr"] for r in per_seed]
    s["pairs"][pk][attack]["mean_asr"] = float(np.mean(asrs))
    s["pairs"][pk][attack]["std_asr"] = float(np.std(asrs))
    # Max committed ASR
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
            max_asr = max(max_asr, s["pairs"][pk][atk]["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    with open(get_output_path(condition_label), "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, d1, d2, attack_name, dataset, model_name):
    """Run one composition experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        dataset, FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model(model_name, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))

    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    elif attack_name == "committed_pixel":
        internal_attack = "backdoor_pixel"
    else:
        internal_attack = attack_name

    attack = get_attack(internal_attack)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for round_num in range(FL_CONFIG.num_rounds):
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
            updates.append(update)

        aggregated = generic_compose(server, updates, d1, d2)
        server.apply_update(aggregated)

    eval_result = server.evaluate(test_dataset)
    accuracy = float(eval_result["accuracy"])
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)

    return accuracy, float(asr)


def main():
    total_runs = len(CONDITIONS) * len(PAIRS) * len(ATTACKS) * len(SEEDS)
    done = 0
    skipped = 0

    for cond in CONDITIONS:
        label = cond["label"]
        dataset = cond["dataset"]
        model_name = cond["model"]
        print(f"\n{'='*60}")
        print(f"Condition: {label} (dataset={dataset}, model={model_name})")
        print(f"{'='*60}")

        for d1, d2 in PAIRS:
            for attack in ATTACKS:
                for seed in SEEDS:
                    if has_run(label, d1, d2, attack, seed):
                        skipped += 1
                        continue
                    done += 1
                    print(f"  [{done+skipped}/{total_runs}] {d1}->{d2} | {attack} | seed={seed}")
                    try:
                        accuracy, asr = run_one(seed, d1, d2, attack, dataset, model_name)
                        save_one(label, d1, d2, attack, seed, accuracy, asr)
                        print(f"    -> acc={accuracy:.3f}, asr={asr:.3f}")
                    except Exception as e:
                        print(f"    -> ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {done} runs executed, {skipped} skipped (cached)")
    print(f"{'='*60}")

    # Print summary
    for cond in CONDITIONS:
        label = cond["label"]
        s = load_or_init(label)
        print(f"\n--- {label} ---")
        for pk, info in sorted(s["pairs"].items()):
            max_asr = info.get("max_committed_asr", "?")
            predicted = "PASS" if pk in ["foolsgold_then_rfa", "foolsgold_then_coord_median",
                                         "reputation_then_coord_median"] else "FAIL"
            correct = (predicted == "PASS" and max_asr < 0.5) or \
                      (predicted == "FAIL" and max_asr >= 0.5)
            print(f"  {pk}: max_asr={max_asr:.3f} | predicted={predicted} | "
                  f"{'CORRECT' if correct else 'MISS'}")


if __name__ == "__main__":
    main()
