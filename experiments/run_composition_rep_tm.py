"""
Round 62 — Rep+TM composition baseline (Part B).

Reviewer question: Does composing reputation+trimmed_mean every round dominate
the paper's survivor result (temporal mix at rep30/tm70, realized VoPD +0.032)?
If yes, the temporal mix loses operational meaning (composition is simpler and better).

Defense: reputation weighting THEN trimmed_mean, applied jointly every round.
Both defenses run on the same update set: reputation first (to down-weight outliers
by distance from median), then trimmed_mean on those updates (coordinate-wise trim).

Attacks: committed_scaling, committed_pixel, oracle (switches each round based on
which committed attack has higher ASR against the composition's static suppression).

Seeds: 42-46 (5 seeds, same as composition_baseline and existing pilots).
FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn — identical to §5.8.

Comparison:
  Temporal mix rep30/tm70 (§5.8, n=30):
    committed_scaling: 0.871 ± 0.095
    committed_pixel:   0.747 ± 0.106
    oracle:            0.915 ± 0.065
    realized VoPD:    +0.032 ± 0.054

Output: results/composition_rep_tm/summary.json
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

SEEDS = list(range(42, 72))  # Round 63: scale 5->30 seeds to match temporal-mix baseline; has_run() skips cached
ADV_POLICIES = ["committed_scaling", "committed_pixel", "oracle"]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

# Oracle: play the attack that has higher committed-composition ASR.
# We'll determine this empirically on seed 42 pilot, then hard-code.
# Reputation strongly suppresses scaling (pure rep ASR ~0.02 on scaling),
# trimmed_mean partially suppresses pixel (pure tm ASR ~0.62 on pixel).
# In composition the oracle should try both; we run oracle as:
# each round, use model_scaling (since reputation is always active, it suppresses
# it; but the oracle can observe which attack the composition fails to catch).
# Actually: oracle plays the best COMMITTED attack (not per-round switching),
# determined after we see committed results.
# For safety, we implement oracle as: play model_scaling on even rounds, pixel on odd —
# this is the per-round oracle as in rep_tm_survivor.
ORACLE_BEST_ATTACKS = {
    "reputation": "backdoor_pixel",   # reputation kills scaling (ASR 0.02), not pixel
    "trimmed_mean": "model_scaling",  # trimmed_mean less effective vs scaling
}

output_dir = os.path.join(base_dir, "results", "composition_rep_tm")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "method": "reputation_then_trimmed_mean",
        "description": "reputation weighting then trimmed_mean aggregation, every round",
        "oracle_best_attack_per_defense": ORACLE_BEST_ATTACKS,
        "adversary_policies": {p: {"per_seed": []} for p in ADV_POLICIES},
    }


def has_run(adv_pol, seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s["adversary_policies"][adv_pol]["per_seed"])


def save_one(adv_pol, seed, accuracy, asr_scaling, asr_pixel):
    s = load_or_init()
    runs = s["adversary_policies"][adv_pol]["per_seed"]
    runs = [r for r in runs if r["seed"] != seed]
    runs.append({
        "seed": seed,
        "accuracy": float(accuracy),
        "asr_scaling_final": float(asr_scaling),
        "asr_pixel_final": float(asr_pixel),
    })
    s["adversary_policies"][adv_pol]["per_seed"] = runs
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def compose_reputation_trimmed_mean(server, updates, beta=0.2):
    """Reputation weighting followed by trimmed_mean on the same update set.

    Step 1: Compute reputation weights (exp(-dist_from_median / scale)).
    Step 2: Apply those weights to scale each client's update.
    Step 3: Apply trimmed_mean on the reputation-scaled updates.

    This means both defenses see the original updates: reputation down-weights
    adversarial outliers, then trimmed_mean trims the coordinate extremes.
    """
    from fl_core.federated import FederatedServer as FS
    keys = list(updates[0].keys())
    n = len(updates)

    # Step 1: compute reputation weights (mirrors _reputation logic)
    import torch
    import torch.nn.functional as F
    flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
    client_stack = torch.stack(flats)
    consensus = client_stack.median(dim=0).values
    dists = (client_stack - consensus.unsqueeze(0)).norm(dim=1)
    scale = float(dists.median().clamp(min=1e-6).item())
    rep_weights = torch.exp(-dists / scale)
    rep_weights = rep_weights / rep_weights.sum().clamp(min=1e-8)

    # Step 2: scale each update by its reputation weight
    scaled_updates = []
    for i, u in enumerate(updates):
        w = rep_weights[i].item() * n  # multiply by n so trimmed_mean denominator works out
        scaled_updates.append({k: u[k] * w for k in keys})

    # Step 3: trimmed_mean on the reputation-scaled updates
    trim_count = int(n * beta)
    result = {}
    for k in keys:
        shape = updates[0][k].shape
        device = updates[0][k].device
        stacked = torch.stack([u[k].flatten().cpu().float() for u in scaled_updates])
        sorted_vals, _ = stacked.sort(dim=0)
        trimmed = sorted_vals[trim_count:n - trim_count]
        result[k] = trimmed.mean(dim=0).reshape(shape).to(device)
    return result


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

    # Poison adversarial clients for both attacks (both share same trigger)
    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack_scaling.poison_dataset(ds)  # same trigger as pixel
        clients.append(FederatedClient(i, ds, device))

    # Oracle RNG — use same pattern as rep_tm_survivor.py
    oracle_rng = np.random.default_rng(seed + 77777)

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        # Oracle: alternate scaling/pixel each round to mimic per-round oracle
        if adv_pol == "oracle":
            # Simpler: use scaling on rep-dominant rounds, pixel on tm-dominant
            # Since composition always has both active, oracle just picks best;
            # we'll run oracle as pixel (since rep kills scaling, pixel is harder)
            current_attack = attack_pixel if (r % 2 == 0) else attack_scaling
        elif adv_pol == "committed_scaling":
            current_attack = attack_scaling
        else:  # committed_pixel
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

        aggregated = compose_reputation_trimmed_mean(server, updates)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr_scaling = evaluate_backdoor(server.global_model, test_dataset, device=device)
    # Both attacks share the same trigger, so asr_scaling == asr_pixel in evaluate_backdoor
    asr_pixel = asr_scaling
    return float(eval_result["accuracy"]), float(asr_scaling), float(asr_pixel)


if __name__ == "__main__":
    print("=== Rep+TM Composition Baseline (Round 62) ===")
    print(f"  Method: reputation weighting then trimmed_mean, every round")
    print(f"  Seeds: {SEEDS}")
    print(f"  Adversary policies: {ADV_POLICIES}")
    print(f"  Comparison: temporal mix rep30/tm70 oracle ASR = 0.915 ± 0.065 (n=30)\n")

    t0 = time.time()
    for adv_pol in ADV_POLICIES:
        for seed in SEEDS:
            if has_run(adv_pol, seed):
                print(f"  [skip] {adv_pol} seed {seed}", flush=True)
                continue
            t_run = time.time()
            acc, asr_s, asr_p = run_one(seed, adv_pol)
            save_one(adv_pol, seed, acc, asr_s, asr_p)
            dt = time.time() - t_run
            print(f"  {adv_pol} seed {seed}: acc={acc:.3f} ASR={asr_s:.3f} ({dt:.0f}s)", flush=True)

    print(f"\n=== COMPOSITION REP+TM COMPLETE ===")
    s = load_or_init()
    for adv_pol in ADV_POLICIES:
        runs = s["adversary_policies"][adv_pol]["per_seed"]
        if runs:
            asrs = [r["asr_scaling_final"] for r in runs]
            accs = [r["accuracy"] for r in runs]
            print(f"  {adv_pol}: ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}, "
                  f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    print(f"\nComparison (rep30/tm70 temporal mix, n=30):")
    print(f"  committed_scaling: 0.871 ± 0.095")
    print(f"  committed_pixel:   0.747 ± 0.106")
    print(f"  oracle:            0.915 ± 0.065")
    print(f"  realized VoPD:    +0.032 ± 0.054")
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
