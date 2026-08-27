"""
Held-out survivor menu test: FoolsGold(0.3) + TrimmedMean(0.7).

Reviewer Q6: "The held-out menu test is mentioned but never run."

This tests a structurally different defense family from rep+tm:
- FoolsGold uses pairwise-similarity weighting (kills scaling in 4/5 seeds)
- TrimmedMean uses coordinate-wise trimming
- Static VoPD from cached scan = 0.114

Oracle: scaling on TM rounds, pixel on FoolsGold rounds
(FoolsGold suppresses scaling → pixel is the best response there)

Output: results/held_out_foolsgold_tm/
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

DEFENSE_DIST = {"foolsgold": 0.30, "trimmed_mean": 0.70}
ORACLE_BEST_ATTACK = {"foolsgold": "backdoor_pixel", "trimmed_mean": "model_scaling"}
ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
SEEDS = list(range(42, 72))  # 30 seeds for statistical confidence
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "held_out_foolsgold_tm")
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
        "pre_committed": {
            "STRONG": "realized VoPD >= 0.05 -> survivor regime",
            "MEDIUM": "realized VoPD in [0.01, 0.05] -> weak survivor",
            "NEGATIVE": "realized VoPD < 0.01 -> universal collapse",
        },
    }


def save_one(adv_pol: str, seed: int, accuracy: float, asr_scaling: float,
              asr_pixel: float, defense_counts: dict, attack_counts: dict | None = None):
    s = load_or_init()
    a = s["adversary_policies"][adv_pol]
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_scaling_final": float(asr_scaling),
        "asr_pixel_final": float(asr_pixel),
        "defense_counts": {k: int(v) for k, v in defense_counts.items()},
    }
    if attack_counts is not None:
        entry["attack_counts"] = {k: int(v) for k, v in attack_counts.items()}
    a["per_seed"].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(adv_pol: str, seed: int) -> bool:
    s = load_or_init()
    a = s["adversary_policies"].get(adv_pol, {})
    return any(e["seed"] == seed for e in a.get("per_seed", []))


def run_adversary(seed: int, adv_pol: str) -> dict:
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
            if cid in adversarial_ids:
                ds = poisoned[cid]
            else:
                ds = clean_datasets[cid]
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
    asr_scaling_final = asr_final
    asr_pixel_final = asr_final

    return {
        "accuracy": float(eval_result["accuracy"]),
        "asr_scaling_final": float(asr_scaling_final),
        "asr_pixel_final": float(asr_pixel_final),
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }


print(f"=== HELD-OUT MENU: FoolsGold(0.3) + TrimmedMean(0.7) ===")
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


print(f"\n=== HELD-OUT MENU COMPLETE ===")
s = load_or_init()
policy_means = {}
for adv_pol in ADVERSARY_POLICIES:
    asrs = [e["asr_pixel_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    accs = [e["accuracy"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    if asrs:
        m, sd = float(np.mean(asrs)), float(np.std(asrs))
        am, asd = float(np.mean(accs)), float(np.std(accs))
        policy_means[adv_pol] = m
        print(f"  {adv_pol}: ASR={m:.3f}±{sd:.3f}, Acc={am:.3f}±{asd:.3f} (n={len(asrs)})")

if "oracle" in policy_means and "committed_scaling" in policy_means:
    best_committed = max(policy_means.get("committed_scaling", 0),
                         policy_means.get("committed_pixel", 0))
    realized_vopd = policy_means["oracle"] - best_committed
    print(f"\n  Realized VoPD = ASR(oracle) - max(ASR(committed)) "
          f"= {policy_means['oracle']:.3f} - {best_committed:.3f} = {realized_vopd:.3f}")
    if realized_vopd >= 0.05:
        print(f"  VERDICT: STRONG survivor (VoPD >= 0.05)")
    elif realized_vopd >= 0.01:
        print(f"  VERDICT: MEDIUM/weak survivor (VoPD in [0.01, 0.05])")
    else:
        print(f"  VERDICT: NEGATIVE — persistence collapses this menu")

print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
print(f"Runs completed this session: {runs_done}")
