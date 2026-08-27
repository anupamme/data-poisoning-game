"""
Low-f / long-horizon probe - extended to 15 seeds.
Config: f=0.05, N=10 (1 adversarial), 200 rounds, NE3 mix (FedAvg 26% + NormClip 74%)
Seeds: 42-56 (15 seeds, extending original 5-seed pilot)
Output: results/low_f_probe_extended/summary.json
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

SEEDS = list(range(42, 57))  # 15 seeds
ADV_FRACTION = 0.05
NUM_ROUNDS = 200
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]

FL_CONFIG = FLConfig(
    num_clients=10,
    clients_per_round=5,
    num_rounds=NUM_ROUNDS,
)

output_dir = os.path.join(base_dir, "results", "low_f_probe_extended")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "adv_fraction": ADV_FRACTION,
        "num_rounds": NUM_ROUNDS,
        "defense_dist": DEFENSE_DIST,
        "adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES},
    }


def has_run(adv_pol, seed):
    s = load_or_init()
    if adv_pol not in s["adversary_policies"]:
        return False
    return any(r["seed"] == seed for r in s["adversary_policies"][adv_pol]["per_seed"])


def save_one(adv_pol, seed, accuracy, asr):
    s = load_or_init()
    if adv_pol not in s["adversary_policies"]:
        s["adversary_policies"][adv_pol] = {"per_seed": []}
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["adversary_policies"][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = max(1, int(FL_CONFIG.num_clients * ADV_FRACTION))
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Poison datasets based on policy
    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            if adv_pol == "committed_pixel":
                ds = attack_pixel.poison_dataset(ds)
            else:
                # committed_scaling and oracle both use scaling's poisoned data
                ds = attack_scaling.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    defense_rng = np.random.default_rng(seed + 77777)
    defense_schedule = []
    for _ in range(NUM_ROUNDS):
        if defense_rng.random() < 0.26:
            defense_schedule.append("fedavg")
        else:
            defense_schedule.append("norm_clip")

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        d_this = defense_schedule[r]

        # Oracle selects attack based on defense drawn this round
        if adv_pol == "oracle":
            if d_this == "norm_clip":
                current_attack = attack_scaling
            else:
                current_attack = attack_pixel
        elif adv_pol == "committed_scaling":
            current_attack = attack_scaling
        else:
            current_attack = attack_pixel

        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = current_attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = server.aggregate(updates, method=d_this)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


if __name__ == "__main__":
    from scipy import stats

    print("=== Low-f / Long-Horizon Probe — Extended (15 Seeds) ===")
    print(f"  f={ADV_FRACTION} ({max(1, int(FL_CONFIG.num_clients * ADV_FRACTION))} adversarial client)")
    print(f"  rounds={NUM_ROUNDS}, defense={DEFENSE_DIST}")
    print(f"  Arms: {ADV_POLICIES}")
    print(f"  Oracle: model_scaling on NormClip rounds, backdoor_pixel on FedAvg rounds")
    print(f"  Seeds: {SEEDS}\n")

    t0 = time.time()
    for adv_pol in ADV_POLICIES:
        for seed in SEEDS:
            if has_run(adv_pol, seed):
                print(f"  [skip] {adv_pol} seed {seed}", flush=True)
                continue
            t_run = time.time()
            acc, asr = run_one(seed, adv_pol)
            save_one(adv_pol, seed, acc, asr)
            dt = time.time() - t_run
            print(f"  {adv_pol} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    print(f"\n=== LOW-F PROBE EXTENDED RESULTS (15 seeds) ===")
    s = load_or_init()
    asrs_by_pol = {}
    for adv_pol in ADV_POLICIES:
        runs = s["adversary_policies"][adv_pol]["per_seed"]
        if runs:
            asrs = [r["asr"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            asrs_by_pol[adv_pol] = asrs
            print(f"  {adv_pol}: ASR={np.mean(asrs):.3f} +/- {np.std(asrs):.3f}, "
                  f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f} (n={len(runs)})")

    # Compute VoPD = oracle_asr - committed_scaling_asr (per-seed)
    if "oracle" in asrs_by_pol and "committed_scaling" in asrs_by_pol:
        oracle_asrs = np.array(asrs_by_pol["oracle"])
        scaling_asrs = np.array(asrs_by_pol["committed_scaling"])
        n_paired = min(len(oracle_asrs), len(scaling_asrs))
        vopd = oracle_asrs[:n_paired] - scaling_asrs[:n_paired]

        print(f"\n  VoPD (oracle - committed_scaling):")
        print(f"    Mean: {np.mean(vopd):.4f} +/- {np.std(vopd):.4f}")
        print(f"    Per-seed: {[round(x, 4) for x in vopd]}")

        # One-sided t-test: H0: VoPD >= 0, H1: VoPD < 0
        t_stat, p_two = stats.ttest_1samp(vopd, 0)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        print(f"    t-stat: {t_stat:.3f}, p-value (one-sided, VoPD < 0): {p_one:.4f}")

        if p_one < 0.05:
            print(f"    VERDICT: VoPD significantly < 0 at p={p_one:.4f} — oracle hurts at f=0.05.")
        elif np.mean(vopd) > 0.02:
            print(f"    VERDICT: Oracle retains value at f=0.05 — persistence alone insufficient.")
        else:
            print(f"    VERDICT: VoPD not significantly different from 0 (inconclusive).")

    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
