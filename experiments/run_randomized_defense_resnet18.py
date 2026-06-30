"""
NE3 BR run on ResNet-18 (Round 45 — second architecture).

Reviewer Q1: "Show the persistence collapse on at least one more persistent
attack or architecture so the now-broader title is earned."

Setup mirrors the headline CIFAR-10 NE3 BR experiment but with ResNet-18.

- N=10, K=5, f=0.2 (matches headline setting)
- Model: resnet18
- 50 rounds, 5 seeds (42-46)
- NE3 mixed policy: FedAvg 26% + NormClip 74% per round
- Attacks tested: no_attack, model_scaling, backdoor_pixel
- For each attack, run BR experiment under NE3 mix

Output: results/randomized_defense_resnet18/summary.json
"""
import copy
import json
import os
import sys
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
ATTACKS = ["no_attack", "model_scaling", "backdoor_pixel"]
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "randomized_defense_resnet18")
os.makedirs(output_dir, exist_ok=True)


def run_randomized_experiment(attack_name, defense_dist, fl_config, exp_config, rng):
    """Run one FL experiment sampling defense per round from defense_dist."""
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        exp_config.dataset, fl_config.num_clients, exp_config.dirichlet_alpha, exp_config.seed
    )
    model = get_model(exp_config.model, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(fl_config.num_clients * exp_config.adversarial_fraction)
    adversarial_ids = set(range(num_adversarial))
    attack = get_attack(attack_name)

    clients = []
    for i in range(fl_config.num_clients):
        dataset = client_datasets[i]
        if i in adversarial_ids:
            dataset = attack.poison_dataset(dataset)
        clients.append(FederatedClient(i, dataset, device))

    defenses = list(defense_dist.keys())
    probs = list(defense_dist.values())
    defenses_used = []
    current_lr = fl_config.learning_rate

    for _ in range(fl_config.num_rounds):
        participant_ids = np.random.choice(
            fl_config.num_clients,
            size=min(fl_config.clients_per_round, fl_config.num_clients),
            replace=False
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, fl_config.local_epochs,
                current_lr, fl_config.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        defense_this_round = rng.choice(defenses, p=probs)
        defenses_used.append(defense_this_round)
        aggregated = server.aggregate(updates, method=defense_this_round)
        server.apply_update(aggregated)
        current_lr *= getattr(fl_config, 'lr_decay', 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)

    fedavg_count = defenses_used.count("fedavg")
    return {
        "accuracy": eval_result["accuracy"],
        "attack_success_rate": asr,
        "fedavg_rounds": fedavg_count,
    }


print(f"ResNet-18 NE3 BR experiment (Round 45 second architecture):")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Attacks: {ATTACKS}, Seeds: {SEEDS}")
print(f"  Model: resnet18, N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, T={FL_CONFIG.num_rounds}\n")

results = {attack: {"realized_accuracy": [], "realized_asr": [], "fedavg_fraction": []}
           for attack in ATTACKS}

for seed in SEEDS:
    exp_config = ExperimentConfig(
        dataset="cifar10", model="resnet18",
        dirichlet_alpha=0.5, adversarial_fraction=0.2,
        num_trials=1, seed=seed, device="mps",
    )
    rng = np.random.default_rng(seed + 1000)
    for attack in ATTACKS:
        print(f"  Seed {seed}, attack={attack}...", end=" ", flush=True)
        r = run_randomized_experiment(attack, DEFENSE_DIST, FL_CONFIG, exp_config, rng)
        results[attack]["realized_accuracy"].append(r["accuracy"])
        results[attack]["realized_asr"].append(r["attack_success_rate"])
        results[attack]["fedavg_fraction"].append(r["fedavg_rounds"] / FL_CONFIG.num_rounds)
        print(f"acc={r['accuracy']:.3f}, ASR={r['attack_success_rate']:.3f}")

summary = {"model": "resnet18", "policy": DEFENSE_DIST, "seeds": SEEDS, "attacks": {}}
for attack in ATTACKS:
    r = results[attack]
    summary["attacks"][attack] = {
        "realized_accuracy": {"mean": float(np.mean(r["realized_accuracy"])),
                              "std": float(np.std(r["realized_accuracy"])),
                              "per_seed": r["realized_accuracy"]},
        "realized_asr": {"mean": float(np.mean(r["realized_asr"])),
                         "std": float(np.std(r["realized_asr"])),
                         "per_seed": r["realized_asr"]},
        "fedavg_fraction": {"mean": float(np.mean(r["fedavg_fraction"])),
                            "per_seed": r["fedavg_fraction"]},
    }

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Print results
print(f"\n=== ResNet-18 NE3 BR Summary ===")
for attack in ATTACKS:
    a = summary["attacks"][attack]
    print(f"  {attack}: acc={a['realized_accuracy']['mean']:.3f}±{a['realized_accuracy']['std']:.3f}, "
          f"ASR={a['realized_asr']['mean']:.3f}±{a['realized_asr']['std']:.3f}")

# Pre-committed interpretation
ms_asr = summary["attacks"]["model_scaling"]["realized_asr"]["mean"]
bp_asr = summary["attacks"]["backdoor_pixel"]["realized_asr"]["mean"]
gap = ms_asr - bp_asr
print(f"\nGap (scaling - pixel): {gap:+.3f}")
print(f"\n=== Pre-committed interpretation ===")
if ms_asr > 0.85:
    print(f"  REPLICATES on ResNet-18: realized scaling ASR={ms_asr:.3f} > 0.85, persistence collapse confirmed.")
elif ms_asr > 0.5:
    print(f"  PARTIAL: realized scaling ASR={ms_asr:.3f} in [0.5, 0.85]; weaker but not zero.")
else:
    print(f"  DOES NOT REPLICATE: realized scaling ASR={ms_asr:.3f} < 0.5; ResNet-18 breaks persistence dynamics.")

print(f"\nSaved: {output_dir}/summary.json")
