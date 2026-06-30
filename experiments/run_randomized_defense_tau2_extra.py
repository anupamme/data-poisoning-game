"""
Round 54 — τ=2 disentanglement scale to 15 seeds (extends Round 41's 5-seed pilot).

Reviewer comment: "the τ=2 experiment (Appendix H) is the sole empirical separation
[of γ vs. p_eff] and is 5 seeds." Scaling to 15 seeds tests falsification prediction
(iii) (low-p_eff via menu reduces realized VoPD) with statistical power.

Mirrors experiments/run_randomized_defense_tau2.py exactly; only SEEDS differs
(seeds 47-56 here; original 42-46 already cached).

Output: results/randomized_defense_tau2/summary_seeds_47_56.json (separate file
to preserve the original 5-seed pilot; aggregation happens at integration time).
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
SEEDS = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56]  # 10 NEW seeds
NORM_CLIP_TAU = 2.0
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "randomized_defense_tau2")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary_seeds_47_56.json")


def run_randomized_experiment(attack_name, defense_dist, fl_config, exp_config, rng,
                                norm_clip_tau=2.0):
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


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "policy": DEFENSE_DIST,
        "norm_clip_tau": NORM_CLIP_TAU,
        "seeds": SEEDS,
        "attacks": {a: [] for a in ATTACKS},
    }


def has_run(seed, attack):
    s = load_or_init()
    return any(r.get("seed") == seed for r in s["attacks"].get(attack, []))


def save_one(seed, attack, result):
    s = load_or_init()
    s["attacks"].setdefault(attack, [])
    s["attacks"][attack] = [r for r in s["attacks"][attack] if r.get("seed") != seed]
    entry = {"seed": seed}
    entry.update(result)
    s["attacks"][attack].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


import time
print(f"τ=2 scale to 15 seeds (Round 54 extension): seeds 47-56, NormClip τ={NORM_CLIP_TAU}")
print(f"Attacks: {ATTACKS}, Seeds: {SEEDS}, Policy: {DEFENSE_DIST}\n")

t0 = time.time()
runs_done = 0
for seed in SEEDS:
    exp_config = ExperimentConfig(
        dataset="cifar10", model="cifar_cnn", dirichlet_alpha=0.5,
        adversarial_fraction=0.2, num_trials=1, seed=seed, device="mps",
    )
    rng = np.random.default_rng(seed + 9000)
    for attack in ATTACKS:
        if has_run(seed, attack):
            print(f"  [skip] seed {seed} attack={attack} cached", flush=True)
            continue
        t_run = time.time()
        r = run_randomized_experiment(attack, DEFENSE_DIST, FL_CONFIG, exp_config, rng,
                                        norm_clip_tau=NORM_CLIP_TAU)
        save_one(seed, attack, r)
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {attack}: acc={r['accuracy']:.3f} ASR={r['attack_success_rate']:.3f} "
              f"(fedavg_rounds={r['fedavg_rounds']}) "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)


print(f"\n=== τ=2 EXTENSION COMPLETE ===")
s = load_or_init()
for attack in ATTACKS:
    asrs = [r["attack_success_rate"] for r in s["attacks"][attack]]
    accs = [r["accuracy"] for r in s["attacks"][attack]]
    if asrs:
        print(f"  {attack}: ASR={np.mean(asrs):.3f}+/-{np.std(asrs):.3f} (n={len(asrs)})  "
              f"acc={np.mean(accs):.3f}")
        print(f"    per-seed: {[round(x, 3) for x in asrs]}")
print(f"Wall time: {(time.time()-t0)/60:.1f} min")
