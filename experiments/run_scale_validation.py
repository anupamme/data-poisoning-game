"""
Scale validation: N=100, K=10, f=0.05, ResNet18, 50 rounds.

Validates the paper's main claims at practitioner-relevant scale:
  - Experiment A (NE2 Collapse): VoPD collapse under randomized 50/50 FedAvg/NormClip
  - Experiment B (Composition Dominance): rep+tm composition still dominates

Config: N=100, K=10, f=0.05 (num_adv=5), ResNet18, CIFAR-10, alpha=0.5
Seeds: 42-51 (10 seeds)
Adversary policies: committed_scaling, committed_pixel, oracle (per-round switch)

Timing estimate: ResNet18 at N=100 is ~50 min per run on MPS (60 runs total ~50 hours).

Output: results/scale_validation_n100_resnet18/summary.json
"""
import copy
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
from experiments.run_composition_rep_tm import compose_reputation_trimmed_mean

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # n=10 power-up; cached seeds auto-skipped
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
ADV_FRACTION = 0.05  # num_adv = int(100 * 0.05) = 5
NUM_ROUNDS = 50
MODEL_NAME = "resnet18"

FL_CONFIG = FLConfig(
    num_clients=NUM_CLIENTS,
    clients_per_round=CLIENTS_PER_ROUND,
    num_rounds=NUM_ROUNDS,
)

# Estimated per-run time in minutes (ResNet18 on MPS with N=100, K=10, 50 rounds)
EST_MINUTES_PER_RUN = 50

output_dir = os.path.join(base_dir, "results", "scale_validation_n100_resnet18")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# ── Checkpointing ────────────────────────────────────────────────────────────
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "config": {
            "num_clients": NUM_CLIENTS,
            "clients_per_round": CLIENTS_PER_ROUND,
            "adversarial_fraction": ADV_FRACTION,
            "model": MODEL_NAME,
            "num_rounds": NUM_ROUNDS,
        },
        "ne2_collapse": {
            "defense_policy": {"fedavg": 0.5, "norm_clip": 0.5},
            "committed_scaling": {"per_seed": []},
            "committed_pixel": {"per_seed": []},
            "oracle": {"per_seed": []},
        },
        "composition_rep_tm": {
            "committed_scaling": {"per_seed": []},
            "committed_pixel": {"per_seed": []},
            "oracle": {"per_seed": []},
        },
    }


def has_run(experiment, adv_pol, seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s[experiment][adv_pol]["per_seed"])


def save_one(experiment, adv_pol, seed, accuracy, asr_final, extra=None):
    s = load_or_init()
    runs = s[experiment][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    entry = {
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_final": float(asr_final),
    }
    if extra:
        entry.update(extra)
    runs.append(entry)
    s[experiment][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# ── Experiment A: NE2 Collapse (FedAvg/NormClip 50/50) ──────────────────────
def run_one_ne2(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model(MODEL_NAME, num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Prepare client datasets for both attacks
    clients_scaling = []
    clients_pixel = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds_scaling = attack_scaling.poison_dataset(ds)
            ds_pixel = attack_pixel.poison_dataset(ds)
        else:
            ds_scaling = ds
            ds_pixel = ds
        clients_scaling.append(FederatedClient(i, ds_scaling, device))
        clients_pixel.append(FederatedClient(i, ds_pixel, device))

    # Defense schedule: random 50/50 FedAvg/NormClip
    defense_rng = np.random.default_rng(seed + 99999)
    defense_schedule = [defense_rng.choice(["fedavg", "norm_clip"]) for _ in range(NUM_ROUNDS)]

    defense_counts = {"fedavg": 0, "norm_clip": 0}
    attack_counts = {"model_scaling": 0, "backdoor_pixel": 0}
    current_lr = FL_CONFIG.learning_rate

    for r in range(NUM_ROUNDS):
        defense = defense_schedule[r]
        defense_counts[defense] += 1

        # Determine attack for this round
        if adv_pol == "committed_scaling":
            attack_this_round = "model_scaling"
        elif adv_pol == "committed_pixel":
            attack_this_round = "backdoor_pixel"
        elif adv_pol == "oracle":
            # Oracle: model_scaling on even rounds, backdoor_pixel on odd rounds
            attack_this_round = "model_scaling" if (r % 2 == 0) else "backdoor_pixel"
        else:
            raise ValueError(f"Unknown policy: {adv_pol}")

        attack_counts[attack_this_round] += 1

        # Select attack object and clients
        if attack_this_round == "model_scaling":
            attack_obj = attack_scaling
            clients = clients_scaling
        else:
            attack_obj = attack_pixel
            clients = clients_pixel

        # Select K clients uniformly
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        # Train and collect updates
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = attack_obj.manipulate_update(update, server.global_model)
            updates.append(update)

        # Apply defense
        if defense == "fedavg":
            aggregated = server.aggregate(updates, "fedavg")
        elif defense == "norm_clip":
            aggregated = server.aggregate(updates, "norm_clip", tau=5.0)
        else:
            raise ValueError(f"Unknown defense: {defense}")

        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    # Final evaluation
    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)

    return (
        float(eval_result["accuracy"]),
        float(asr_final),
        defense_counts,
        attack_counts,
    )


# ── Experiment B: Composition Dominance (rep+tm) ─────────────────────────────
def run_one_composition(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model(MODEL_NAME, num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Poison adversarial clients for both attacks (both share same trigger)
    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack_scaling.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate

    for r in range(NUM_ROUNDS):
        # Determine attack for this round
        if adv_pol == "committed_scaling":
            current_attack = attack_scaling
        elif adv_pol == "committed_pixel":
            current_attack = attack_pixel
        elif adv_pol == "oracle":
            # Oracle for composition: pixel on even rounds, scaling on odd rounds
            current_attack = attack_pixel if (r % 2 == 0) else attack_scaling
        else:
            raise ValueError(f"Unknown policy: {adv_pol}")

        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = current_attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = compose_reputation_trimmed_mean(server, updates)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    # Final evaluation
    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)

    return float(eval_result["accuracy"]), float(asr_final)


# ── MPS memory management ────────────────────────────────────────────────────
def clear_mps_cache():
    """Free MPS memory between runs to avoid OOM with ResNet18."""
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    import gc
    gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  SCALE VALIDATION: N=100, K=10, f=0.05, ResNet18, 50 rounds")
    print("=" * 70)
    print(f"  Config: N={NUM_CLIENTS}, K={CLIENTS_PER_ROUND}, f={ADV_FRACTION}, "
          f"rounds={NUM_ROUNDS}, model={MODEL_NAME}")
    print(f"  num_adv = {int(NUM_CLIENTS * ADV_FRACTION)}, "
          f"selection_prob = {CLIENTS_PER_ROUND}/{NUM_CLIENTS} = "
          f"{CLIENTS_PER_ROUND/NUM_CLIENTS:.2f}")
    print(f"  Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"  Adversary policies: {ADV_POLICIES}")
    print(f"  Estimated time per run: ~{EST_MINUTES_PER_RUN} min")
    total_runs = len(ADV_POLICIES) * len(SEEDS) * 2  # 2 experiments
    print(f"  Total runs: {total_runs} ({total_runs * EST_MINUTES_PER_RUN / 60:.0f} hours estimated)")
    print()

    t0 = time.time()
    runs_done = 0

    # ── Experiment A: NE2 Collapse ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: VoPD Collapse on NE2 (FedAvg/NormClip 50/50)")
    print("=" * 70)

    for adv_pol in ADV_POLICIES:
        print(f"\n--- NE2 | Policy: {adv_pol} ---")
        for seed in SEEDS:
            if has_run("ne2_collapse", adv_pol, seed):
                print(f"  [skip] ne2/{adv_pol} seed {seed}", flush=True)
                runs_done += 1
                continue

            t_run = time.time()
            acc, asr, d_counts, a_counts = run_one_ne2(seed, adv_pol)
            save_one("ne2_collapse", adv_pol, seed, acc, asr, extra={
                "defense_counts": {k: int(v) for k, v in d_counts.items()},
                "attack_counts": {k: int(v) for k, v in a_counts.items()},
            })
            runs_done += 1
            dt = time.time() - t_run

            # ETA calculation
            elapsed_total = time.time() - t0
            avg_per_run = elapsed_total / runs_done if runs_done > 0 else EST_MINUTES_PER_RUN * 60
            remaining = (total_runs - runs_done) * avg_per_run
            eta_h = remaining / 3600

            a_str = " ".join(f"{k}:{v}" for k, v in a_counts.items() if v > 0)
            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} [{a_str}] "
                  f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

            clear_mps_cache()

    # ── Experiment B: Composition Dominance ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Composition Dominance (rep+tm)")
    print("=" * 70)

    for adv_pol in ADV_POLICIES:
        print(f"\n--- Composition | Policy: {adv_pol} ---")
        for seed in SEEDS:
            if has_run("composition_rep_tm", adv_pol, seed):
                print(f"  [skip] composition/{adv_pol} seed {seed}", flush=True)
                runs_done += 1
                continue

            t_run = time.time()
            acc, asr = run_one_composition(seed, adv_pol)
            save_one("composition_rep_tm", adv_pol, seed, acc, asr)
            runs_done += 1
            dt = time.time() - t_run

            # ETA calculation
            elapsed_total = time.time() - t0
            avg_per_run = elapsed_total / runs_done if runs_done > 0 else EST_MINUTES_PER_RUN * 60
            remaining = (total_runs - runs_done) * avg_per_run
            eta_h = remaining / 3600

            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                  f"({dt/60:.1f}min, {runs_done}/{total_runs}, ETA {eta_h:.1f}h)", flush=True)

            clear_mps_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCALE VALIDATION COMPLETE")
    print("=" * 70)

    s = load_or_init()

    print("\n  Experiment A — NE2 Collapse (FedAvg/NormClip 50/50):")
    print(f"  {'Policy':<25s} {'ASR (mean +/- std)':<25s} {'Accuracy'}")
    print(f"  {'-'*70}")
    ne2_asrs = {}
    for adv_pol in ADV_POLICIES:
        runs = s["ne2_collapse"][adv_pol]["per_seed"]
        if runs:
            asrs = [r["asr_final"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            ne2_asrs[adv_pol] = (np.mean(asrs), np.std(asrs))
            print(f"  {adv_pol:<25s} {np.mean(asrs):.3f} +/- {np.std(asrs):.3f}"
                  f"        {np.mean(accs):.3f} +/- {np.std(accs):.3f} (n={len(runs)})")

    if "committed_scaling" in ne2_asrs and "oracle" in ne2_asrs:
        vopd_ne2 = ne2_asrs["oracle"][0] - ne2_asrs["committed_scaling"][0]
        print(f"\n  VoPD (oracle - committed_scaling) = {vopd_ne2:+.4f}")
        print(f"  Claim: VoPD should collapse (near 0 or negative) at scale")

    print("\n  Experiment B — Composition Dominance (rep+tm):")
    print(f"  {'Policy':<25s} {'ASR (mean +/- std)':<25s} {'Accuracy'}")
    print(f"  {'-'*70}")
    comp_asrs = {}
    for adv_pol in ADV_POLICIES:
        runs = s["composition_rep_tm"][adv_pol]["per_seed"]
        if runs:
            asrs = [r["asr_final"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            comp_asrs[adv_pol] = (np.mean(asrs), np.std(asrs))
            print(f"  {adv_pol:<25s} {np.mean(asrs):.3f} +/- {np.std(asrs):.3f}"
                  f"        {np.mean(accs):.3f} +/- {np.std(accs):.3f} (n={len(runs)})")

    if "committed_scaling" in comp_asrs and "oracle" in comp_asrs:
        vopd_comp = comp_asrs["oracle"][0] - max(
            comp_asrs["committed_scaling"][0], comp_asrs["committed_pixel"][0]
        )
        print(f"\n  VoPD (oracle - best_committed) = {vopd_comp:+.4f}")
        print(f"  Claim: composition dominates (all ASRs suppressed)")

    # Comparison to N=10 baselines
    print("\n  Comparison to N=10 baselines (from paper Table 3):")
    print(f"    N=10 NE2: committed_scaling ASR ~0.511, oracle ~0.450")
    print(f"    N=10 rep+tm: committed_scaling ASR ~0.871, oracle ~0.915")
    print(f"    N=100 results above should show similar or stronger suppression")

    wall_time = (time.time() - t0) / 60
    print(f"\n  Wall time: {wall_time:.1f} min ({wall_time/60:.1f} hours)")
    print(f"  Output: {output_path}")
