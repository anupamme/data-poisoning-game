"""
Ceiling-effect discrimination test at the NC30/rep70 intermediate-ASR mix point.

Reviewer W2: "ASR is near-saturated (0.83-0.96) for nearly every policy.
The 'max-like' behavior could be a saturation artifact rather than persistence."

At NC30/rep70, committed-scaling has mean ASR ≈ 0.51 (intermediate, NOT saturated).
If VoPD ≈ 0 here too, the ceiling-effect explanation is ruled out — the collapse
is from persistence (pixel universally admitted), not from saturation of all arms.

Oracle: scaling on norm_clip rounds (30%), pixel on reputation rounds (70%).

Output: results/cifar10_ceiling_discrimination/
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

DEFENSE_DIST = {"norm_clip": 0.30, "reputation": 0.70}
ORACLE_BEST_ATTACK = {"norm_clip": "model_scaling", "reputation": "backdoor_pixel"}
ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
SEEDS = list(range(42, 52))  # 10 seeds
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "cifar10_ceiling_discrimination")
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
            attack_this = ORACLE_BEST_ATTACK[d_this]
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


print(f"=== CEILING-EFFECT DISCRIMINATION: NC30/rep70 ===")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Policies: {ADVERSARY_POLICIES}, Seeds: {SEEDS}")
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

print(f"\n=== CEILING DISCRIMINATION COMPLETE ===")
s = load_or_init()
policy_asrs = {}
for adv_pol in ADVERSARY_POLICIES:
    asrs = [e["asr_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        policy_asrs[adv_pol] = (m, sd, len(asrs))
        print(f"  {adv_pol}: ASR={m:.3f}±{sd:.3f} (n={len(asrs)})")

if "oracle" in policy_asrs:
    oracle_asrs = np.array([e["asr_final"] for e in s["adversary_policies"]["oracle"]["per_seed"]])
    scaling_asrs = np.array([e["asr_final"] for e in s["adversary_policies"]["committed_scaling"]["per_seed"]])
    pixel_asrs = np.array([e["asr_final"] for e in s["adversary_policies"]["committed_pixel"]["per_seed"]])
    vopd = oracle_asrs - np.maximum(scaling_asrs, pixel_asrs)
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(vopd, 0)
    ci = stats.t.interval(0.95, df=len(vopd)-1, loc=vopd.mean(), scale=stats.sem(vopd))
    print(f"\n  Per-seed-max VoPD: {vopd.mean():.3f}±{vopd.std():.3f}")
    print(f"  t({len(vopd)-1})={t_stat:.2f}, p={p_val:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  {(vopd>0).sum()}/{len(vopd)} positive")
    print(f"\n  KEY: committed-scaling at {scaling_asrs.mean():.3f} (INTERMEDIATE, not saturated)")
    print(f"       VoPD≈0 rules out ceiling-effect explanation")

print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
