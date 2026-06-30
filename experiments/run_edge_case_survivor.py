"""
Round 52 — 3-attack survivor pilot at the new max-static-VoPD menu.

Generalizes run_rep_tm_survivor.py to four adversary policies: committed_scaling,
committed_pixel, committed_edge_case, and an oracle that observes the round's
sampled defense and plays the argmax attack on that defense.

Setup (filled in after Part C identifies the new max-VoPD menu):
  - Defense mix: <to be set from the 3-attack static-VoPD scan>
  - Oracle's per-defense argmax table: <to be set from the same scan>

Each round, oracle picks attack = ORACLE_BEST_ATTACK[d_this]. committed_*
policies play one fixed attack across all rounds.

End-of-training ASR evaluation uses the trigger of whichever attack was played
this round. Because scaling/pixel share the pixel trigger but edge_case has a
distinct top-corner trigger, we evaluate BOTH triggers at end-of-training and
report:
  - asr_pixel_final (relevant for committed_scaling/pixel and pixel-rounds of oracle)
  - asr_edge_case_final (relevant for committed_edge_case and edge_case-rounds of oracle)

Realized VoPD = oracle_realized_ASR - max(committed_scaling, committed_pixel,
                                          committed_edge_case)
where each committed_* realized ASR is measured against its own trigger.

Output: results/cifar10_edge_case_survivor/summary.json
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
from experiments.run_payoff_matrix import (
    evaluate_backdoor, evaluate_edge_case_backdoor
)

# === MENU / ORACLE CONFIG (filled after Part C) ===
# Placeholder values; will be overwritten by main() if --menu CLI args are given.
DEFENSE_DIST = {"reputation": 0.30, "trimmed_mean": 0.70}
ORACLE_BEST_ATTACK = {
    "reputation": "backdoor_pixel",
    "trimmed_mean": "model_scaling",
    "norm_clip": "model_scaling",
    "fedavg": "model_scaling",
    "foolsgold": "backdoor_pixel",
    "rfa": "model_scaling",
    "coord_median": "model_scaling",
}
# === / ===

ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel",
                      "committed_edge_case", "oracle"]
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2


def output_path():
    return os.path.join(base_dir, "results", "cifar10_edge_case_survivor", "summary.json")


def load_or_init():
    p = output_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {
        "policy": DEFENSE_DIST,
        "oracle_best_attack_per_defense": ORACLE_BEST_ATTACK,
        "seeds": SEEDS,
        "adversary_policies": {pol: {"per_seed": []} for pol in ADVERSARY_POLICIES},
        "pre_committed": {
            "STRONG": "realized VoPD >= 0.05 AND edge_case oracle-contributes",
            "MEDIUM": "realized VoPD in [0.02, 0.05] OR edge_case in NE but not oracle",
            "NEGATIVE": "realized VoPD < 0.02 OR edge_case not contributing",
        },
    }


def save_one(adv_pol, seed, accuracy, asr_pixel, asr_edge_case, realized_asr,
              defense_counts, attack_counts):
    s = load_or_init()
    a = s["adversary_policies"][adv_pol]
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    a["per_seed"].append({
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_pixel_final": float(asr_pixel),
        "asr_edge_case_final": float(asr_edge_case),
        "realized_asr": float(realized_asr),
        "defense_counts": {k: int(v) for k, v in defense_counts.items()},
        "attack_counts": {k: int(v) for k, v in attack_counts.items()},
    })
    os.makedirs(os.path.dirname(output_path()), exist_ok=True)
    with open(output_path(), "w") as f:
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

    scaling_attack = get_attack("model_scaling")
    pixel_attack = get_attack("backdoor_pixel")
    edge_case_attack = get_attack("backdoor_edge_case")

    # Pre-poison three copies of each adversarial client's dataset
    clean_datasets = {}
    scaling_poisoned = {}
    pixel_poisoned = {}
    edge_poisoned = {}
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        clean_datasets[i] = ds
        if i in adversarial_ids:
            scaling_poisoned[i] = scaling_attack.poison_dataset(ds)
            pixel_poisoned[i] = pixel_attack.poison_dataset(ds)
            edge_poisoned[i] = edge_case_attack.poison_dataset(ds)

    defenses = list(DEFENSE_DIST.keys())
    probs = list(DEFENSE_DIST.values())
    rng = np.random.default_rng(seed + 7000)
    current_lr = FL_CONFIG.learning_rate

    defense_counts = {d: 0 for d in defenses}
    attack_counts = {"model_scaling": 0, "backdoor_pixel": 0, "backdoor_edge_case": 0}
    # Per-round attack chosen (for realized-ASR computation under oracle)
    per_round_attack = []

    for _ in range(FL_CONFIG.num_rounds):
        d_this = rng.choice(defenses, p=probs)
        defense_counts[d_this] += 1

        if adv_pol == "committed_scaling":
            attack_this = "model_scaling"
        elif adv_pol == "committed_pixel":
            attack_this = "backdoor_pixel"
        elif adv_pol == "committed_edge_case":
            attack_this = "backdoor_edge_case"
        elif adv_pol == "oracle":
            attack_this = ORACLE_BEST_ATTACK[d_this]
        else:
            raise ValueError(adv_pol)
        attack_counts[attack_this] += 1
        per_round_attack.append(attack_this)

        if attack_this == "model_scaling":
            poisoned = scaling_poisoned
            attack_obj = scaling_attack
        elif attack_this == "backdoor_pixel":
            poisoned = pixel_poisoned
            attack_obj = pixel_attack
        else:
            poisoned = edge_poisoned
            attack_obj = edge_case_attack

        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            ds = poisoned[cid] if cid in adversarial_ids else clean_datasets[cid]
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

    eval_result = server.evaluate(test_dataset)
    asr_pixel_final = evaluate_backdoor(server.global_model, test_dataset, device=device)
    asr_edge_case_final = evaluate_edge_case_backdoor(
        server.global_model, test_dataset, device=device
    )

    # Realized ASR per attack-mechanism, weighted by the rounds the attack was played
    # For committed policies, realized ASR = the trigger matching their committed attack.
    # For oracle, realized ASR = weighted average over per-round triggers.
    if adv_pol == "committed_scaling" or adv_pol == "committed_pixel":
        realized_asr = asr_pixel_final
    elif adv_pol == "committed_edge_case":
        realized_asr = asr_edge_case_final
    else:  # oracle
        pixel_round_frac = (
            attack_counts["model_scaling"] + attack_counts["backdoor_pixel"]
        ) / FL_CONFIG.num_rounds
        edge_round_frac = attack_counts["backdoor_edge_case"] / FL_CONFIG.num_rounds
        realized_asr = pixel_round_frac * asr_pixel_final + edge_round_frac * asr_edge_case_final

    return {
        "accuracy": float(eval_result["accuracy"]),
        "asr_pixel_final": asr_pixel_final,
        "asr_edge_case_final": asr_edge_case_final,
        "realized_asr": realized_asr,
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }


def main():
    print(f"Round 52 edge_case survivor pilot:")
    print(f"  Policy: {DEFENSE_DIST}")
    print(f"  Oracle attack table: {ORACLE_BEST_ATTACK}")
    print(f"  Adversary policies: {ADVERSARY_POLICIES}, Seeds: {SEEDS}")
    print(f"  Expected runs: {len(ADVERSARY_POLICIES) * len(SEEDS)}\n")

    t0 = time.time()
    runs_done = 0
    for adv_pol in ADVERSARY_POLICIES:
        for seed in SEEDS:
            if has_run(adv_pol, seed):
                print(f"  [skip] {adv_pol} seed {seed} cached", flush=True)
                continue
            t_run = time.time()
            r = run_adversary(seed, adv_pol)
            save_one(adv_pol, seed, r["accuracy"], r["asr_pixel_final"],
                      r["asr_edge_case_final"], r["realized_asr"],
                      r["defense_counts"], r["attack_counts"])
            runs_done += 1
            dt = time.time() - t_run
            dc_str = " ".join(f"{k}={v}" for k, v in r["defense_counts"].items())
            ac_str = " ".join(f"{k}={v}" for k, v in r["attack_counts"].items())
            print(f"  seed {seed} {adv_pol}: acc={r['accuracy']:.3f} "
                  f"realized_asr={r['realized_asr']:.3f} "
                  f"(asr_pixel={r['asr_pixel_final']:.3f} asr_edge={r['asr_edge_case_final']:.3f}) "
                  f"({dc_str}) ({ac_str}) "
                  f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
                  flush=True)

    print(f"\n=== EDGE_CASE SURVIVOR PILOT COMPLETE ===")
    s = load_or_init()
    means = {}
    for adv_pol in ADVERSARY_POLICIES:
        asrs = [e["realized_asr"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
        accs = [e["accuracy"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
        if asrs:
            m, sd = float(np.mean(asrs)), float(np.std(asrs))
            means[adv_pol] = m
            print(f"  {adv_pol}: realized ASR = {m:.3f} +/- {sd:.3f}  "
                  f"acc = {np.mean(accs):.3f}")
            print(f"    per-seed: {[round(x,3) for x in asrs]}")

    if all(p in means for p in ADVERSARY_POLICIES):
        committed_max = max(means["committed_scaling"], means["committed_pixel"],
                             means["committed_edge_case"])
        realized_vopd = means["oracle"] - committed_max
        print(f"\n  realized full-info (oracle): {means['oracle']:.3f}")
        print(f"  realized committed-max:      {committed_max:.3f}")
        print(f"  REALIZED VoPD = {realized_vopd:.3f}")
        print(f"\n=== Pre-committed verdict ===")
        if realized_vopd >= 0.05:
            print(f"  STRONG: realized VoPD = {realized_vopd:.3f} >= 0.05")
        elif realized_vopd >= 0.02:
            print(f"  MEDIUM: realized VoPD = {realized_vopd:.3f} in [0.02, 0.05]")
        else:
            print(f"  NEGATIVE: realized VoPD = {realized_vopd:.3f} < 0.02")

    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
