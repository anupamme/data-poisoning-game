"""
Observer-adaptive attack against the rep+tm temporal mix.

Reviewer W5: "No adaptive attack is designed against the temporal mix itself —
an attacker that modulates scaling factor based on inferred defense from
observed global-model dynamics."

The adversary observes the global model state before and after each round.
If the L2 norm of the aggregate update is small relative to what a scaling
attack would produce, the defense was likely clipping/filtering (reputation
or norm-clip). If the update norm is large (near what scaling injects),
the defense was permissive (trimmed_mean/fedavg).

Inference rule: compute the ratio of ||Δw_global|| to expected ||Δw_scaling||.
- Ratio < threshold → clipping defense active → play pixel next round
- Ratio >= threshold → permissive defense → play scaling next round

Run on rep+tm temporal mix (reputation 30% + trimmed_mean 70%), same as §5.3.
Compare: committed_scaling, committed_pixel, oracle, observer_adaptive.

Expected: observer_adaptive ASR ≈ oracle ASR (0.915), confirming oracle is tight.

Output: results/observer_adaptive_attack/
"""
import sys, os, json, time
import numpy as np
import torch
import copy
import warnings

warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor

DEFENSE_DIST = {"reputation": 0.30, "trimmed_mean": 0.70}
ORACLE_BEST_ATTACK = {"reputation": "backdoor_pixel", "trimmed_mean": "model_scaling"}
ADVERSARY_POLICIES = ["committed_scaling", "committed_pixel", "oracle", "observer_adaptive"]
SEEDS = list(range(42, 52))  # 10 seeds
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

# Inference threshold: if ||Δw_global||/||Δw_scaling_expected|| < this, infer clipping
# Calibrated to distinguish rep (clipping) from TM (permissive) via update norm
INFERENCE_THRESHOLD = 0.3

output_dir = os.path.join(base_dir, "results", "observer_adaptive_attack")
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
        "inference_threshold": INFERENCE_THRESHOLD,
    }


def save_one(adv_pol, seed, accuracy, asr, defense_counts, attack_counts=None,
             inference_accuracy=None):
    s = load_or_init()
    a = s["adversary_policies"][adv_pol]
    a["per_seed"] = [e for e in a["per_seed"] if e["seed"] != seed]
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_final": float(asr),
        "defense_counts": {k: int(v) for k, v in defense_counts.items()},
    }
    if attack_counts:
        entry["attack_counts"] = {k: int(v) for k, v in attack_counts.items()}
    if inference_accuracy is not None:
        entry["inference_accuracy"] = float(inference_accuracy)
    a["per_seed"].append(entry)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def has_run(adv_pol, seed):
    s = load_or_init()
    return any(e["seed"] == seed for e in s["adversary_policies"].get(adv_pol, {}).get("per_seed", []))


def compute_model_l2(state_dict):
    return float(sum(v.float().norm()**2 for v in state_dict.values())**0.5)


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

    # For observer-adaptive: track inferred defense each round
    inferred_defense = "trimmed_mean"  # initial guess: permissive
    n_correct_inferences = 0
    current_attack = "model_scaling"  # start with scaling (permissive default)

    for round_idx in range(FL_CONFIG.num_rounds):
        # Sample the actual defense (hidden from observer)
        d_this = rng.choice(defenses, p=probs)
        defense_counts[d_this] += 1

        # Determine attack for this round
        if adv_pol == "committed_scaling":
            attack_this = "model_scaling"
        elif adv_pol == "committed_pixel":
            attack_this = "backdoor_pixel"
        elif adv_pol == "oracle":
            attack_this = ORACLE_BEST_ATTACK[d_this]
        elif adv_pol == "observer_adaptive":
            # Use inferred defense from previous round's observation
            attack_this = ORACLE_BEST_ATTACK.get(inferred_defense, "model_scaling")
        else:
            raise ValueError(adv_pol)

        attack_counts[attack_this] += 1

        if attack_this == "model_scaling":
            poisoned = scaling_poisoned
            attack_obj = scaling_attack
        else:
            poisoned = pixel_poisoned
            attack_obj = pixel_attack

        # Record global model state before aggregation (for observer-adaptive)
        w_before = copy.deepcopy(server.global_model.state_dict())

        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        adv_update_norm = None
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
                if adv_update_norm is None:
                    # Record norm of adversary's manipulated update
                    adv_update_norm = float(sum(v.float().norm()**2 for v in update.values())**0.5)
            updates.append(update)

        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

        # Observer-adaptive: infer defense from global model change
        if adv_pol == "observer_adaptive" and adv_update_norm is not None and adv_update_norm > 0:
            w_after = server.global_model.state_dict()
            # Compute ||Δw_global|| = ||w_after - w_before||
            delta_norm = float(sum(
                (w_after[k].float() - w_before[k].float()).norm()**2
                for k in w_after
            )**0.5)
            # Ratio of global change to adversary's submitted norm
            ratio = delta_norm / adv_update_norm
            # Low ratio → clipping suppressed adversary update → reputation
            inferred_defense = "reputation" if ratio < INFERENCE_THRESHOLD else "trimmed_mean"
            if inferred_defense == d_this:
                n_correct_inferences += 1

    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)

    result = {
        "accuracy": float(eval_result["accuracy"]),
        "asr_final": float(asr_final),
        "defense_counts": defense_counts,
        "attack_counts": attack_counts,
    }
    if adv_pol == "observer_adaptive":
        result["inference_accuracy"] = n_correct_inferences / FL_CONFIG.num_rounds
    return result


print(f"=== OBSERVER-ADAPTIVE ATTACK vs REP+TM TEMPORAL MIX ===")
print(f"  Policy: {DEFENSE_DIST}")
print(f"  Policies: {ADVERSARY_POLICIES}, Seeds: {SEEDS}")
print(f"  Inference threshold: {INFERENCE_THRESHOLD}")
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
        inf_acc = r.get("inference_accuracy")
        save_one(adv_pol, seed, r["accuracy"], r["asr_final"],
                 r["defense_counts"], r["attack_counts"], inf_acc)
        runs_done += 1
        dt = time.time() - t_run
        inf_str = f" inf_acc={inf_acc:.2f}" if inf_acc is not None else ""
        print(f"  seed {seed} {adv_pol}: ASR={r['asr_final']:.3f}{inf_str} "
              f"({dt:.0f}s, total {(time.time()-t0)/60:.1f}min, {runs_done} runs)",
              flush=True)

print(f"\n=== OBSERVER-ADAPTIVE COMPLETE ===")
s = load_or_init()
import numpy as np
for adv_pol in ADVERSARY_POLICIES:
    asrs = [e["asr_final"] for e in s["adversary_policies"][adv_pol]["per_seed"]]
    if asrs:
        m = float(np.mean(asrs))
        sd = float(np.std(asrs))
        print(f"  {adv_pol}: ASR={m:.3f}±{sd:.3f} (n={len(asrs)})")

print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")
