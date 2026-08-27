"""
Strong adversary experiment for N=50, f=0.05 configuration.

Reviewer gap: The current oracle (per-round switching) gets ASR 0.450 vs
committed_scaling 0.511 — the oracle hurts itself by switching. The saturation
bound (epsilon <= 0.489) is uninformative. We need to show that even a smarter
adversary that infers the defense and switches adaptively cannot do much better
than committed_scaling.

Three adversary policies:
  A) committed_scaling — baseline, always model_scaling (known ASR ~0.511)
  B) commit_then_probe — embed trigger for 30 rounds, then switch based on
     inferred defense (large model change => FedAvg => play pixel)
  C) belief_state — maintain P(FedAvg) belief, switch to pixel only when
     belief is high AND trigger is already saturated (ASR > 0.5)

Defense schedule: NE2 mix (50% FedAvg, 50% NormClip), sampled randomly.
The adversary does NOT know the schedule — must infer from observations.

Config: N=50, K=5, f=0.05, 50 rounds, cifar_cnn, cifar10, alpha=0.5
Seeds: 42-71 (30 seeds)

Output: results/strong_adversary_n50/summary.json
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

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS = list(range(42, 72))  # 30 seeds
ADV_POLICIES = ["committed_scaling", "commit_then_probe", "belief_state"]
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 5
ADV_FRACTION = 0.05  # num_adv = int(50 * 0.05) = 2
NUM_ROUNDS = 50

FL_CONFIG = FLConfig(
    num_clients=NUM_CLIENTS,
    clients_per_round=CLIENTS_PER_ROUND,
    num_rounds=NUM_ROUNDS,
)

# Defense NE2: 50/50 FedAvg / NormClip
DEFENSE_DIST = {"fedavg": 0.5, "norm_clip": 0.5}

# commit_then_probe: switch phase begins at round 30
PROBE_PHASE_START = 30

# belief_state: thresholds
BELIEF_THRESHOLD = 0.7     # P(FedAvg) must exceed this to switch
ASR_THRESHOLD = 0.5        # trigger must be saturated above this
ASR_CHECK_INTERVAL = 5     # check ASR every N rounds

output_dir = os.path.join(base_dir, "results", "strong_adversary_n50")
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
            "num_rounds": NUM_ROUNDS,
        },
        "defense_policy": DEFENSE_DIST,
        "adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES},
    }


def has_run(adv_pol, seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s["adversary_policies"][adv_pol]["per_seed"])


def save_one(adv_pol, seed, accuracy, asr_final, defense_counts, attack_counts):
    s = load_or_init()
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_final": float(asr_final),
        "defense_counts": {k: int(v) for k, v in defense_counts.items()},
        "attack_counts": {k: int(v) for k, v in attack_counts.items()},
    })
    s["adversary_policies"][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# ── Training loop ─────────────────────────────────────────────────────────────
def run_one(seed, adv_pol):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    attack_scaling = get_attack("model_scaling")
    attack_pixel = get_attack("backdoor_pixel")

    # Prepare client datasets: adversarial clients get poisoned data for both attacks
    # (both use the same pixel trigger, so poison_dataset is equivalent)
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

    # Tracking
    defense_counts = {"fedavg": 0, "norm_clip": 0}
    attack_counts = {"model_scaling": 0, "backdoor_pixel": 0}

    # State for adaptive policies
    norm_history = []          # ||w_new - w_old|| per round
    cached_asr = 0.0          # last evaluated ASR (for belief_state)
    belief_p_fedavg = 0.5     # running belief P(FedAvg)

    current_lr = FL_CONFIG.learning_rate

    for r in range(NUM_ROUNDS):
        defense = defense_schedule[r]
        defense_counts[defense] += 1

        # ── Determine attack for this round based on policy ──
        if adv_pol == "committed_scaling":
            attack_this_round = "model_scaling"

        elif adv_pol == "commit_then_probe":
            if r < PROBE_PHASE_START:
                # Embedding phase: always model_scaling
                attack_this_round = "model_scaling"
            else:
                # Probing phase: infer defense from previous round's norm change
                if len(norm_history) >= 2:
                    median_norm = float(np.median(norm_history))
                    last_norm = norm_history[-1]
                    if last_norm > median_norm:
                        # Large change => infer FedAvg was used => play pixel
                        attack_this_round = "backdoor_pixel"
                    else:
                        # Small change => infer NormClip => stay on scaling
                        attack_this_round = "model_scaling"
                else:
                    attack_this_round = "model_scaling"

        elif adv_pol == "belief_state":
            # Maintain running belief; switch only when conditions are met
            should_switch = (
                belief_p_fedavg > BELIEF_THRESHOLD
                and cached_asr > ASR_THRESHOLD
            )
            if should_switch:
                attack_this_round = "backdoor_pixel"
            else:
                attack_this_round = "model_scaling"

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

        # Record global model state before aggregation
        w_before = copy.deepcopy(server.global_model.state_dict())

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

        # ── Adversary observation: compute ||w_new - w_old|| ──
        w_after = server.global_model.state_dict()
        delta_norm = float(sum(
            (w_after[k].float() - w_before[k].float()).norm() ** 2
            for k in w_after
        ) ** 0.5)
        norm_history.append(delta_norm)

        # ── Update belief state ──
        if adv_pol == "belief_state" and len(norm_history) >= 5:
            # Estimate threshold from first 5 rounds of observation
            baseline_norms = norm_history[:5]
            threshold = float(np.median(baseline_norms))

            # Update belief: large norm => likely FedAvg, small => likely NormClip
            if delta_norm > threshold:
                # Evidence for FedAvg
                belief_p_fedavg = 0.8 * belief_p_fedavg + 0.2 * 1.0
            else:
                # Evidence for NormClip
                belief_p_fedavg = 0.8 * belief_p_fedavg + 0.2 * 0.0

            # Periodically evaluate ASR to check trigger saturation
            if (r + 1) % ASR_CHECK_INTERVAL == 0:
                cached_asr = evaluate_backdoor(
                    server.global_model, test_dataset, device=device
                )

    # Final evaluation
    eval_result = server.evaluate(test_dataset)
    asr_final = evaluate_backdoor(server.global_model, test_dataset, device=device)

    return float(eval_result["accuracy"]), float(asr_final), defense_counts, attack_counts


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Strong Adversary Experiment: N=50, f=0.05 ===")
    print(f"  Config: N={NUM_CLIENTS}, K={CLIENTS_PER_ROUND}, f={ADV_FRACTION}, "
          f"rounds={NUM_ROUNDS}")
    print(f"  num_adv = {int(NUM_CLIENTS * ADV_FRACTION)}, "
          f"selection_prob = {CLIENTS_PER_ROUND}/{NUM_CLIENTS} = "
          f"{CLIENTS_PER_ROUND/NUM_CLIENTS:.2f}")
    print(f"  Defense: {DEFENSE_DIST}")
    print(f"  Adversary policies: {ADV_POLICIES}")
    print(f"  Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"  Baseline: committed_scaling ~0.511, oracle ~0.450")
    print(f"  Goal: show smarter adversaries cannot beat committed_scaling\n")

    t0 = time.time()
    total_runs = len(ADV_POLICIES) * len(SEEDS)
    runs_done = 0

    for adv_pol in ADV_POLICIES:
        print(f"\n--- Policy: {adv_pol} ---")
        for seed in SEEDS:
            if has_run(adv_pol, seed):
                print(f"  [skip] {adv_pol} seed {seed}", flush=True)
                runs_done += 1
                continue
            t_run = time.time()
            acc, asr, d_counts, a_counts = run_one(seed, adv_pol)
            save_one(adv_pol, seed, acc, asr, d_counts, a_counts)
            runs_done += 1
            dt = time.time() - t_run
            a_str = " ".join(f"{k}:{v}" for k, v in a_counts.items() if v > 0)
            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} [{a_str}] "
                  f"({dt:.0f}s, {runs_done}/{total_runs})", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"=== STRONG ADVERSARY N=50 COMPLETE ===")
    print(f"{'='*60}")
    s = load_or_init()
    for adv_pol in ADV_POLICIES:
        runs = s["adversary_policies"][adv_pol]["per_seed"]
        if runs:
            asrs = [r["asr_final"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            print(f"  {adv_pol:25s}: ASR={np.mean(asrs):.3f} +/- {np.std(asrs):.3f}, "
                  f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f} (n={len(runs)})")

    print(f"\nComparison:")
    print(f"  committed_scaling (existing): 0.511")
    print(f"  oracle (per-round switch):    0.450  (hurts itself by switching)")
    print(f"  saturation bound:             0.489  (uninformative)")
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
