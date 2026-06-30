"""
τ=2 disentanglement experiment for Round 41.

Re-run the NE3 mixed policy (FedAvg 26% + NormClip 74%) on CIFAR-10 with
NormClip τ=2 instead of τ=5. This tests whether the realized VoPD collapse
is driven by persistence (γ near 1) or by high admission (p_eff).

At τ=2, NormClip clips scaling updates more aggressively, lowering p^scaling_NC
from ≈0.95 (τ=5) to ≈0.74 (estimated from τ-sweep at τ=1 giving 0.74).
This drops p_eff from 0.83 to ≈0.78.

Pre-committed interpretation (Round 41 plan):
  - If realized scaling ASR > 0.85: persistence dominates; Theorem 4 strengthens
  - If realized scaling ASR < 0.70: admission dominates; Theorem 4 weakens
  - Mixed (0.70–0.85): inconclusive but informative

Output: results/randomized_defense_tau2/summary.json
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


# Configuration
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
ATTACKS = ["no_attack", "model_scaling", "backdoor_pixel"]
SEEDS = [42, 43, 44, 45, 46]
NORM_CLIP_TAU = 2.0   # CHANGED FROM 5.0
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "randomized_defense_tau2")
os.makedirs(output_dir, exist_ok=True)


def run_randomized_experiment(attack_name, defense_dist, fl_config, exp_config,
                              rng, norm_clip_tau=2.0):
    """Run one FL experiment sampling defense per round; uses τ=norm_clip_tau for NormClip."""
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

    for round_idx in range(fl_config.num_rounds):
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
        if defense_this_round == "norm_clip":
            aggregated = server.aggregate(updates, method=defense_this_round, tau=norm_clip_tau)
        else:
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
        "norm_clip_rounds": fl_config.num_rounds - fedavg_count,
    }


# Run
print(f"τ=2 disentanglement: NE3 BR on CIFAR-10 (5 seeds, NormClip τ={NORM_CLIP_TAU})")
print(f"Attacks: {ATTACKS}, Seeds: {SEEDS}, Policy: {DEFENSE_DIST}\n")

results = {attack: {"realized_accuracy": [], "realized_asr": [], "fedavg_fraction": []}
           for attack in ATTACKS}

for seed in SEEDS:
    exp_config = ExperimentConfig(
        dataset="cifar10", model="cifar_cnn",
        dirichlet_alpha=0.5, adversarial_fraction=0.2,
        num_trials=1, seed=seed, device="mps",
    )
    for attack in ATTACKS:
        print(f"  Seed {seed}, attack={attack}... ", end="", flush=True)
        rng = np.random.default_rng(seed * 31 + hash(attack) % 1000)
        r = run_randomized_experiment(attack, DEFENSE_DIST, FL_CONFIG, exp_config, rng,
                                       norm_clip_tau=NORM_CLIP_TAU)
        results[attack]["realized_accuracy"].append(r["accuracy"])
        results[attack]["realized_asr"].append(r["attack_success_rate"])
        results[attack]["fedavg_fraction"].append(r["fedavg_rounds"] / FL_CONFIG.num_rounds)
        print(f"acc={r['accuracy']:.3f}, ASR={r['attack_success_rate']:.3f}")

# Summarize
summary = {"policy": DEFENSE_DIST, "norm_clip_tau": NORM_CLIP_TAU, "seeds": SEEDS, "attacks": {}}
for attack in ATTACKS:
    r = results[attack]
    realized_asr = np.array(r["realized_asr"])
    realized_acc = np.array(r["realized_accuracy"])
    summary["attacks"][attack] = {
        "realized_asr": {"mean": float(realized_asr.mean()), "std": float(realized_asr.std()),
                          "per_seed": realized_asr.tolist()},
        "realized_accuracy": {"mean": float(realized_acc.mean()), "std": float(realized_acc.std()),
                               "per_seed": realized_acc.tolist()},
    }

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== τ=2 Disentanglement Summary ===")
ms = summary["attacks"]["model_scaling"]["realized_asr"]
bp = summary["attacks"]["backdoor_pixel"]["realized_asr"]
print(f"Model scaling realized ASR: mean={ms['mean']:.3f}, std={ms['std']:.3f}, per-seed={[round(x,3) for x in ms['per_seed']]}")
print(f"Backdoor pixel realized ASR: mean={bp['mean']:.3f}, std={bp['std']:.3f}, per-seed={[round(x,3) for x in bp['per_seed']]}")
print(f"\nGap (scaling - pixel): {ms['mean'] - bp['mean']:+.3f}")

# Pre-committed interpretation
ms_mean = ms['mean']
print(f"\n=== Pre-committed interpretation ===")
if ms_mean > 0.85:
    print(f"  Persistence DOMINATES (ms_ASR={ms_mean:.3f} > 0.85): Theorem 4 strengthens.")
elif ms_mean < 0.70:
    print(f"  Admission DOMINATES (ms_ASR={ms_mean:.3f} < 0.70): Theorem 4 weakens; persistence claim softens.")
else:
    print(f"  Mixed (ms_ASR={ms_mean:.3f} in [0.70, 0.85]): inconclusive but informative.")

print(f"\nSaved: {output_dir}/summary.json")
