"""
Round 55 — 15-seed pilot at the EXACT NE point on the restricted
{reputation, trimmed_mean} menu.

The aggregate NE (experiments/solve_ne_restricted_rep_tm.py) places reputation
at 21.8% and trimmed_mean at 78.2%, with static VoPD = 0.180. This pilot
confirms empirically whether the survivor regime holds at this NE point under
dynamic best-response play, answering reviewer Q1 (NE-reachable AND realized
survivor).

Mirrors experiments/run_rep_tm_survivor.py exactly; only DEFENSE_DIST and SEEDS
change.

Total: 15 seeds × 3 policies = 45 runs × ~12 min = ~9 hr MPS.
Output: results/cifar10_rep_tm_ne_point/summary.json

Pre-committed verdict:
  STRONG: realized VoPD >= 0.03 AND >= 70% seeds positive
  MEDIUM: realized VoPD in [0.01, 0.03]
  NEGATIVE: realized VoPD < 0.01 (NE point is OUTSIDE realized survivor band)
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

# THE NE POINT from solve_ne_restricted_rep_tm.py
DEFENSE_DIST = {"reputation": 0.218, "trimmed_mean": 0.782}
ORACLE_BEST_ATTACK = {
    "reputation": "backdoor_pixel",   # argmax(rep) U_A = pixel (0.842 vs 0.017)
    "trimmed_mean": "model_scaling",  # argmax(tm) U_A = scaling (0.846 vs 0.616)
}
ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
SEEDS = list(range(42, 72))  # 30 seeds 42-71 (Round 56 extension; has_run skips cached)
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_rep_tm_ne_point")
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
        "adversary_policies": {p: {"per_seed": []} for p in ADVERSARY_POLICIES},
        "static_VoPD_at_NE": 0.180,
        "NE_strategy_server": DEFENSE_DIST,
        "NE_strategy_adversary": {"model_scaling": 0.027, "backdoor_pixel": 0.973},
        "pre_committed": {
            "STRONG": "realized VoPD >= 0.03 AND >=70% seeds positive",
            "MEDIUM": "realized VoPD in [0.01, 0.03]",
            "NEGATIVE": "realized VoPD < 0.01",
        },
    }


def save_one(adv_pol, seed, accuracy, asr_scaling, asr_pixel,
              defense_counts, attack_counts):
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

    scaling_attack = get_attack("model_scaling")
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
    rng = np.random.default_rng(seed + 10000)
    current_lr = FL_CONFIG.learning_rate

    defense_counts = {d: 0 for d in defenses}
    attack_counts = {"model_scaling": 0, "backdoor_pixel": 0}

    for _ in range(FL_CONFIG.num_rounds):
        d_this = rng.choice(defenses, p=probs)
        defense_counts[d_this] += 1

        if adv_pol == "committed_scaling":
            attack_this = "model_scaling"
        elif adv_pol == "committed_pixel":
            attack_this = "backdoor_pixel"
        elif adv_pol == "oracle":
            attack_this = ORACLE_BEST_ATTACK[d_this]
        else:
            raise ValueError(adv_pol)
        attack_counts[attack_this] += 1

        if attack_this == "model_scaling":
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

    # Same trigger (pixel pattern) for scaling and pixel attacks
    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "asr_scaling_final": asr_final,
        "asr_pixel_final": asr_final,
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }


print(f"Round 55 NE-point pilot:")
print(f"  Policy (NE on restricted menu): {DEFENSE_DIST}")
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
        save_one(adv_pol, seed, r["accuracy"], r["asr_scaling_final"],
                  r["asr_pixel_final"], r["defense_counts"], r["attack_counts"])
        runs_done += 1
        dt = time.time() - t_run
        dc_str = " ".join(f"{k}={v}" for k, v in r["defense_counts"].items())
        ac_str = " ".join(f"{k}={v}" for k, v in r["attack_counts"].items())
        print(f"  seed {seed} {adv_pol}: acc={r['accuracy']:.3f} "
              f"ASR={r['asr_pixel_final']:.3f}  ({dc_str}) ({ac_str}) "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)

print(f"\n=== NE-POINT PILOT COMPLETE ===")
s = load_or_init()
means = {}
for adv_pol in ADVERSARY_POLICIES:
    asrs = [e["asr_pixel_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        means[adv_pol] = m
        print(f"  {adv_pol}: n={len(asrs)} mean={m:.3f}±{sd:.3f}")
        print(f"    per-seed: {[round(x, 3) for x in asrs]}")

if all(p in means for p in ADVERSARY_POLICIES):
    cmax = max(means["committed_scaling"], means["committed_pixel"])
    vp = means["oracle"] - cmax
    sc = {e["seed"]: e["asr_pixel_final"] for e in s["adversary_policies"]["committed_scaling"]["per_seed"]}
    px = {e["seed"]: e["asr_pixel_final"] for e in s["adversary_policies"]["committed_pixel"]["per_seed"]}
    ora = {e["seed"]: e["asr_pixel_final"] for e in s["adversary_policies"]["oracle"]["per_seed"]}
    per_seed_vopds = [(sd, ora[sd] - max(sc[sd], px[sd])) for sd in sorted(ora) if sd in sc and sd in px]
    if per_seed_vopds:
        ps_mean = float(np.mean([v for _, v in per_seed_vopds]))
        ps_std = float(np.std([v for _, v in per_seed_vopds]))
        ps_pos = sum(1 for _, v in per_seed_vopds if v > 0)
        ps_strong = sum(1 for _, v in per_seed_vopds if v >= 0.05)
        print(f"\n  Per-seed (n={len(per_seed_vopds)}): mean VoPD = {ps_mean:.3f}±{ps_std:.3f}")
        print(f"  positive: {ps_pos}/{len(per_seed_vopds)}")
        print(f"  STRONG (>=0.05): {ps_strong}/{len(per_seed_vopds)}")
        print(f"\n=== Pre-committed verdict ===")
        positive_frac = ps_pos / len(per_seed_vopds)
        if ps_mean >= 0.03 and positive_frac >= 0.7:
            print(f"  STRONG: realized VoPD = {ps_mean:.3f} >= 0.03 AND {positive_frac:.0%} positive")
        elif ps_mean >= 0.01:
            print(f"  MEDIUM: realized VoPD = {ps_mean:.3f} in [0.01, 0.03]")
        else:
            print(f"  NEGATIVE: realized VoPD = {ps_mean:.3f} < 0.01")

print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
