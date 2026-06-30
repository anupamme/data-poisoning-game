"""
NE3 BR run on DBA at f=0.4 (Round 45 — second persistent attack).

Reviewer Q1: "Show the persistence collapse on at least one more persistent
attack or architecture so the now-broader title is earned."

DBA at f=0.2 was effectively dead (ASR≈0.02) due to co-sampling failure. At f=0.4
with K=5, expected adversarial co-participation per round = K·f = 2.0 — co-sampling
is essentially guaranteed. DBA becomes a working persistent attack distinct from
model_scaling (distributed trigger pattern rather than amplified magnitude).

- N=10, K=5, f=0.4 (4 adversarial clients of 10)
- Model: cifar_cnn (same as headline)
- 50 rounds, 5 seeds (42-46)
- NE3 mixed policy: FedAvg 26% + NormClip 74% per round
- Attacks: no_attack, dba, backdoor_pixel (drop model_scaling — DBA is the focus)

Output: results/randomized_defense_dba_f04/summary.json
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
ATTACKS = ["no_attack", "dba", "backdoor_pixel"]
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADVERSARIAL_FRACTION = 0.4  # KEY CHANGE: raised from 0.2

output_dir = os.path.join(base_dir, "results", "randomized_defense_dba_f04")
os.makedirs(output_dir, exist_ok=True)


def run_randomized_experiment(attack_name, defense_dist, fl_config, exp_config, rng):
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


print(f"DBA at f=0.4 NE3 BR experiment (Round 45 second persistent attack):")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Attacks: {ATTACKS}, Seeds: {SEEDS}")
print(f"  f={ADVERSARIAL_FRACTION}, N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, T={FL_CONFIG.num_rounds}\n")

results = {attack: {"realized_accuracy": [], "realized_asr": [], "fedavg_fraction": []}
           for attack in ATTACKS}

for seed in SEEDS:
    exp_config = ExperimentConfig(
        dataset="cifar10", model="cifar_cnn",
        dirichlet_alpha=0.5, adversarial_fraction=ADVERSARIAL_FRACTION,
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

summary = {"model": "cifar_cnn", "adversarial_fraction": ADVERSARIAL_FRACTION,
           "policy": DEFENSE_DIST, "seeds": SEEDS, "attacks": {}}
for attack in ATTACKS:
    r = results[attack]
    summary["attacks"][attack] = {
        "realized_accuracy": {"mean": float(np.mean(r["realized_accuracy"])),
                              "std": float(np.std(r["realized_accuracy"])),
                              "per_seed": r["realized_accuracy"]},
        "realized_asr": {"mean": float(np.mean(r["realized_asr"])),
                         "std": float(np.std(r["realized_asr"])),
                         "per_seed": r["realized_asr"]},
    }

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== DBA at f=0.4 NE3 BR Summary ===")
for attack in ATTACKS:
    a = summary["attacks"][attack]
    print(f"  {attack}: acc={a['realized_accuracy']['mean']:.3f}±{a['realized_accuracy']['std']:.3f}, "
          f"ASR={a['realized_asr']['mean']:.3f}±{a['realized_asr']['std']:.3f}")

# Pre-committed interpretation
dba_asr = summary["attacks"]["dba"]["realized_asr"]["mean"]
print(f"\n=== Pre-committed interpretation ===")
if dba_asr > 0.5:
    print(f"  DBA at f=0.4 ACTIVE: realized DBA ASR={dba_asr:.3f} > 0.5.")
    print(f"  If DBA under NE3 ≈ pure-defense DBA: persistence collapse confirmed for second attack.")
elif dba_asr > 0.1:
    print(f"  PARTIAL: realized DBA ASR={dba_asr:.3f} in [0.1, 0.5]; DBA partially works at f=0.4.")
else:
    print(f"  DBA STILL DEAD: realized DBA ASR={dba_asr:.3f} < 0.1; co-sampling not sufficient.")

print(f"\nSaved: {output_dir}/summary.json")
