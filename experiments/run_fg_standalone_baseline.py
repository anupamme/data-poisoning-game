"""
FoolsGold standalone baseline — same FL config as pure_defense_baselines.
Tests FG alone (no composition) against model_scaling and pixel attacks.
Config: N=10, K=5, f=0.2, 50 rounds, 5 seeds.
Output: results/fg_standalone_baseline.json
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor

SEEDS = list(range(42, 47))
ADV_POLICIES = ["committed_scaling", "committed_pixel"]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_path = os.path.join(base_dir, "results", "fg_standalone_baseline.json")


def compute_fg_weights(updates):
    """Compute FoolsGold weights."""
    keys = list(updates[0].keys())
    flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
    client_stack = torch.stack(flats)
    normed = F.normalize(client_stack, dim=1)
    sim = normed @ normed.T
    sim.fill_diagonal_(0.0)
    max_sim, _ = sim.abs().max(dim=1)
    max_sim = max_sim.clamp(0.0, 1.0 - 1e-6)
    max_overall = max_sim.max().clamp(min=1e-8)
    max_sim = max_sim / max_overall
    weights = torch.log((1.0 - max_sim) / (max_sim + 1e-5) + 1e-5)
    weights = weights - weights.min()
    weights = weights.clamp(min=0.0)
    total = weights.sum()
    if total.item() < 1e-8:
        weights = torch.ones(len(updates)) / len(updates)
    else:
        weights = weights / total
    return weights


def fg_aggregate(updates, weights):
    """FoolsGold weighted average aggregation."""
    keys = list(updates[0].keys())
    agg = {}
    for k in keys:
        stacked = torch.stack([u[k].float() for u in updates])
        agg[k] = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
    return agg


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

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            if adv_pol == "committed_scaling":
                ds = attack_scaling.poison_dataset(ds)
            else:
                ds = attack_pixel.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for r in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )

        if adv_pol == "committed_scaling":
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

        # FoolsGold aggregation
        weights = compute_fg_weights(updates)
        aggregated = fg_aggregate(updates, weights)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


if __name__ == "__main__":
    print("=== FoolsGold Standalone Baseline ===")
    print(f"  Adversary policies: {ADV_POLICIES}")
    print(f"  Seeds: {SEEDS}\n")

    results = {}
    t0 = time.time()
    for adv_pol in ADV_POLICIES:
        results[adv_pol] = {"per_seed": []}
        for seed in SEEDS:
            t_run = time.time()
            acc, asr = run_one(seed, adv_pol)
            results[adv_pol]["per_seed"].append({
                "seed": seed, "accuracy": acc, "asr": asr
            })
            dt = time.time() - t_run
            print(f"  {adv_pol} seed {seed}: acc={acc:.3f} ASR={asr:.3f} ({dt:.0f}s)", flush=True)

    # Summary
    print(f"\n=== SUMMARY ===")
    for adv_pol in ADV_POLICIES:
        asrs = [r["asr"] for r in results[adv_pol]["per_seed"]]
        accs = [r["accuracy"] for r in results[adv_pol]["per_seed"]]
        results[adv_pol]["mean_asr"] = float(np.mean(asrs))
        results[adv_pol]["std_asr"] = float(np.std(asrs))
        results[adv_pol]["mean_acc"] = float(np.mean(accs))
        results[adv_pol]["std_acc"] = float(np.std(accs))
        print(f"  {adv_pol}: ASR={np.mean(asrs):.3f}±{np.std(asrs):.3f}, "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")
