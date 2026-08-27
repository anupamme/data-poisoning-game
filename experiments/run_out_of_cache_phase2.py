"""
Phase 2: Realized VoPD deployment for N=50, K=5, f=0.05 (low participation).

Phase 1 (run_out_of_cache_prediction.py) computed static NE and VoPD.
4/5 seeds had positive static VoPD, with mean δ_NE = 0.043.

This script deploys the best mixed NE from Phase 1 as the server's policy
and measures realized VoPD under three adversary strategies.

The best-VoPD NE from Phase 1:
- Seed 44: VoPD=0.166, Adv=[pixel, scaling], Srv=[fedavg, norm_clip]  ← use this
  (adv 50/50 pixel+scaling, server 50/50 fedavg+norm_clip approximately)
- Seeds 43, 45, 46 had smaller VoPD; we average across available mixed NE.

We run the best-VoPD NE as a fixed policy across 5 seeds (42-46, same FL config),
and report oracle vs. committed ASRs.

Output: results/out_of_cache_n50_f005_phase2/
"""
import sys, os, json, time
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

FL_CONFIG = FLConfig(num_clients=50, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.05
SEEDS = list(range(42, 72))  # 30 seeds for adequate power

# Load Phase 1 NE from the cached summary
phase1_path = os.path.join(base_dir, "results", "out_of_cache_n50_f005", "summary.json")
with open(phase1_path) as f:
    phase1 = json.load(f)

# Extract the best mixed NE: seed 44 has VoPD=0.166
# Its NE1: Adv=[pixel, scaling], Srv=[fedavg, norm_clip], VoPD=0.166
# We'll use this as our deployed policy
best_seed_ne = None
best_vopd = 0.0
for ps in phase1["per_seed"]:
    if ps["best_vopd"] > best_vopd and ps["mixed"]:
        best_vopd = ps["best_vopd"]
        # Find the equilibrium with max VoPD
        best_ne = max(ps["equilibria"], key=lambda e: e["vopd"])
        best_seed_ne = {"seed": ps["seed"], "ne": best_ne, "vopd": ps["best_vopd"]}

print(f"Best Phase 1 NE: seed={best_seed_ne['seed']}, VoPD={best_seed_ne['vopd']:.4f}")
print(f"  Adv support: {best_seed_ne['ne']['adversary_support']}")
print(f"  Srv support: {best_seed_ne['ne']['server_support']}")

# Build server policy from NE support (equal weight over support elements)
srv_support = best_seed_ne["ne"]["server_support"]
adv_support = best_seed_ne["ne"]["adversary_support"]
DEFENSE_DIST = {d: 1.0/len(srv_support) for d in srv_support}

# Oracle best attack: for fedavg (permissive) → scaling; for norm_clip (clipping) → pixel
# From Phase 1 payoffs: at N=50/f=0.05, scaling beats pixel against fedavg,
# pixel beats scaling against norm_clip
ORACLE_BEST_ATTACK = {"fedavg": "model_scaling", "norm_clip": "backdoor_pixel",
                      "trimmed_mean": "model_scaling", "rfa": "model_scaling",
                      "coord_median": "backdoor_pixel"}

ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]

output_dir = os.path.join(base_dir, "results", "out_of_cache_n50_f005_phase2")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "config": {
            "num_clients": 50, "clients_per_round": 5, "adversarial_fraction": ADV_FRACTION,
            "p_present": phase1["config"]["p_present"],
            "phase1_static_vopd_best": best_seed_ne["vopd"],
        },
        "deployed_policy": DEFENSE_DIST,
        "oracle_best_attack": ORACLE_BEST_ATTACK,
        "seeds": SEEDS,
        "adversary_policies": {p: {"per_seed": []} for p in ADVERSARY_POLICIES},
    }


def save_one(adv_pol, seed, accuracy, asr, defense_counts, attack_counts=None):
    s = load_or_init()
    a = s["adversary_policies"][adv_pol]
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    entry = {"seed": seed, "accuracy": float(accuracy), "asr_final": float(asr),
             "defense_counts": {k: int(v) for k, v in defense_counts.items()}}
    if attack_counts:
        entry["attack_counts"] = {k: int(v) for k, v in attack_counts.items()}
    a["per_seed"].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(adv_pol, seed):
    s = load_or_init()
    return any(e["seed"] == seed for e in s["adversary_policies"].get(adv_pol, {}).get("per_seed", []))


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
    rng = np.random.default_rng(seed + 5000)
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
            attack_this = ORACLE_BEST_ATTACK.get(d_this, "model_scaling")
        else:
            raise ValueError(adv_pol)

        attack_counts[attack_this] += 1
        poisoned = scaling_poisoned if attack_this == "model_scaling" else pixel_poisoned
        attack_obj = scaling_attack if attack_this == "model_scaling" else pixel_attack

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
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return {
        "accuracy": float(eval_result["accuracy"]),
        "asr_final": float(asr_final),
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }


print(f"=== PHASE 2: REALIZED VoPD at N=50, K=5, f=0.05 ===")
print(f"  Deployed policy: {DEFENSE_DIST}")
print(f"  p_present = {phase1['config']['p_present']:.4f}")
print(f"  Static VoPD (Phase 1 best): {best_seed_ne['vopd']:.4f}")
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
        save_one(adv_pol, seed, r["accuracy"], r["asr_final"],
                 r["defense_counts"], r["attack_counts"])
        runs_done += 1
        dt = time.time() - t_run
        print(f"  seed {seed} {adv_pol}: ASR={r['asr_final']:.3f} "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)

print(f"\n=== PHASE 2 COMPLETE ===")
s = load_or_init()
policy_asrs = {}
for adv_pol in ADVERSARY_POLICIES:
    asrs = [e["asr_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        policy_asrs[adv_pol] = m
        print(f"  {adv_pol}: ASR={m:.3f}±{sd:.3f} (n={len(asrs)})")

if "oracle" in policy_asrs:
    best_committed = max(policy_asrs.get("committed_scaling", 0),
                         policy_asrs.get("committed_pixel", 0))
    print(f"\n  Realized VoPD = {policy_asrs['oracle']:.3f} - {best_committed:.3f} = "
          f"{policy_asrs['oracle'] - best_committed:.3f}")

print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
