"""
Rich defense-inference sanity check: can an adversary infer the realized defense
from per-round signals including update norms, cosine similarity between
adversarial/benign updates, per-layer norm ratios, and update sparsity?

Runs backdoor_pixel x {fedavg, norm_clip, rfa} x 3 seeds x 50 rounds,
logging 4 feature types per round. Trains logistic regression (LOO) on windows.

Output: results/defense_inference_rich/inference_results_rich.json
"""
import sys, os, json, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack

DEFENSES = ["fedavg", "norm_clip", "rfa"]
ATTACK = "backdoor_pixel"
SEEDS = [42, 43, 44]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "defense_inference_rich")
os.makedirs(output_dir, exist_ok=True)

fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
exp_base = dict(dataset="cifar10", model="cifar_cnn", dirichlet_alpha=0.5,
                adversarial_fraction=0.2, num_trials=1)


def run_with_rich_logging(attack_name, defense_name, fl_config, exp_config):
    """Run one FL experiment and return per-round rich feature vectors."""
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        exp_config.dataset, fl_config.num_clients, exp_config.dirichlet_alpha, exp_config.seed
    )
    model = get_model(exp_config.model, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(fl_config.num_clients * exp_config.adversarial_fraction)
    adversarial_ids = set(range(num_adversarial))
    attack = get_attack(attack_name)

    clients = []
    for i in range(fl_config.num_clients):
        dataset = client_datasets[i]
        if i in adversarial_ids:
            dataset = attack.poison_dataset(dataset)
        clients.append(FederatedClient(i, dataset, device))

    clients_per_round = getattr(fl_config, "clients_per_round", max(10, fl_config.num_clients // 10))
    current_lr = fl_config.learning_rate

    per_round_features = []

    for _ in range(fl_config.num_rounds):
        participant_ids = np.random.choice(fl_config.num_clients,
                                           size=min(clients_per_round, fl_config.num_clients),
                                           replace=False)
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(server.global_model, fl_config.local_epochs,
                                        current_lr, fl_config.local_batch_size)
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append((cid, update))

        cids = [u[0] for u in updates]
        raw_updates = [u[1] for u in updates]

        # Feature 1: aggregated update L2 norm
        aggregated = server.aggregate(raw_updates, method=defense_name)
        agg_flat = torch.cat([v.flatten().float() for v in aggregated.values()])
        norm = agg_flat.norm().item()

        # Feature 2: cosine similarity between mean adversarial and mean benign updates
        adv_updates = [raw_updates[i] for i, cid in enumerate(cids) if cid in adversarial_ids]
        ben_updates = [raw_updates[i] for i, cid in enumerate(cids) if cid not in adversarial_ids]

        cos_sim = 0.0
        if adv_updates and ben_updates:
            adv_flat = torch.stack([
                torch.cat([v.flatten().float() for v in u.values()])
                for u in adv_updates
            ]).mean(dim=0)
            ben_flat = torch.stack([
                torch.cat([v.flatten().float() for v in u.values()])
                for u in ben_updates
            ]).mean(dim=0)
            cos_sim = F.cosine_similarity(adv_flat.unsqueeze(0), ben_flat.unsqueeze(0)).item()

        # Feature 3: per-layer norm ratio (first layer / last layer of aggregated update)
        param_names = list(aggregated.keys())
        norm_ratio = 1.0
        if len(param_names) >= 2:
            first_norm = aggregated[param_names[0]].float().norm().item()
            last_norm = aggregated[param_names[-1]].float().norm().item()
            norm_ratio = first_norm / (last_norm + 1e-8)

        # Feature 4: update sparsity (fraction of agg update entries with |v| < 1e-4)
        sparsity = (agg_flat.abs() < 1e-4).float().mean().item()

        per_round_features.append({
            "norm": norm,
            "cos_sim": cos_sim,
            "norm_ratio": norm_ratio,
            "sparsity": sparsity,
        })

        server.apply_update(aggregated)

    return per_round_features


# Collect per-round rich traces for each (defense, seed)
rich_data_path = os.path.join(output_dir, "rich_traces.json")

if os.path.exists(rich_data_path):
    print("Loading cached rich traces...")
    with open(rich_data_path) as f:
        saved = json.load(f)
    all_traces = [(t["defense"], t["features"]) for t in saved]
else:
    print(f"Running {len(DEFENSES)} defenses x {len(SEEDS)} seeds x 50 rounds (rich logging)...")
    saved_traces = []
    all_traces = []
    for defense in DEFENSES:
        for seed in SEEDS:
            print(f"  {ATTACK} vs {defense}, seed={seed}...")
            exp_config = ExperimentConfig(**exp_base, seed=seed, device="mps")
            features = run_with_rich_logging(ATTACK, defense, fl_config, exp_config)
            all_traces.append((defense, features))
            saved_traces.append({"defense": defense, "seed": seed, "features": features})

    with open(rich_data_path, "w") as f:
        json.dump(saved_traces, f, indent=2)
    print(f"Saved rich traces to {rich_data_path}")

# Train classifier for each window size using rich features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score

n_rounds = len(all_traces[0][1])
window_sizes = [1, 5, 10, 20, 50]
feature_names = ["norm", "cos_sim", "norm_ratio", "sparsity"]

print(f"\n=== Rich Defense-Inference Classifier (7 features) ===")
print(f"Features per round: mean/std/min/max of norm + mean of cos_sim, norm_ratio, sparsity")
print(f"Classes: {DEFENSES}, {len(all_traces)} traces total")
print(f"Windows tested: {window_sizes} rounds\n")

results_by_window = {}
for w in window_sizes:
    trace_features, trace_labels = [], []
    for defense, feat_list in all_traces:
        feat_arr = {k: np.array([f[k] for f in feat_list]) for k in feature_names}
        window_norm = feat_arr["norm"][:w]
        feat = [
            window_norm.mean(), window_norm.std(), window_norm.min(), window_norm.max(),
            feat_arr["cos_sim"][:w].mean(),
            feat_arr["norm_ratio"][:w].mean(),
            feat_arr["sparsity"][:w].mean(),
        ]
        trace_features.append(feat)
        trace_labels.append(defense)

    X_trace = np.array(trace_features)
    le = LabelEncoder()
    y_trace = le.fit_transform(trace_labels)

    clf = LogisticRegression(max_iter=500, random_state=0)
    loo_scores = cross_val_score(clf, X_trace, y_trace, cv=LeaveOneOut(), scoring="accuracy")
    acc = loo_scores.mean()

    results_by_window[w] = float(acc)
    print(f"  Window={w:2d} rounds: LOO accuracy = {acc:.1%}  (random baseline: {1/len(DEFENSES):.0%})")

# Also run with norm-only features for comparison
print(f"\n=== Norm-only features (baseline comparison) ===")
norm_only_results = {}
for w in window_sizes:
    trace_features, trace_labels = [], []
    for defense, feat_list in all_traces:
        norms = np.array([f["norm"] for f in feat_list])
        window = norms[:w]
        feat = [window.mean(), window.std(), window.min(), window.max()]
        trace_features.append(feat)
        trace_labels.append(defense)

    X = np.array(trace_features)
    le = LabelEncoder()
    y = le.fit_transform(trace_labels)
    clf = LogisticRegression(max_iter=500, random_state=0)
    loo_scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
    norm_only_results[w] = float(loo_scores.mean())
    print(f"  Window={w:2d} rounds: LOO accuracy (norm-only) = {loo_scores.mean():.1%}")

print(f"\nRich (7-feat) single-round: {results_by_window[1]:.1%}  |  Norm-only: {norm_only_results[1]:.1%}")

# Per-defense mean norms for reporting
print("\nPer-defense mean norms (round 1):")
for defense in DEFENSES:
    r0_norms = [feat_list[0]["norm"] for d, feat_list in all_traces if d == defense]
    print(f"  {defense}: mean={np.mean(r0_norms):.4f}, std={np.std(r0_norms):.4f}")

out = {
    "defenses": DEFENSES,
    "seeds": SEEDS,
    "n_rounds": n_rounds,
    "features_used": ["norm_mean", "norm_std", "norm_min", "norm_max",
                      "cos_sim_mean", "norm_ratio_mean", "sparsity_mean"],
    "window_accuracies_rich": results_by_window,
    "window_accuracies_norm_only": norm_only_results,
    "single_round_accuracy_rich": results_by_window[1],
    "single_round_accuracy_norm_only": norm_only_results[1],
    "random_baseline": round(1.0 / len(DEFENSES), 4),
    "per_defense_mean_norm_r0": {
        defense: float(np.mean([feat_list[0]["norm"] for d, feat_list in all_traces if d == defense]))
        for defense in DEFENSES
    },
}
out_path = os.path.join(output_dir, "inference_results_rich.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
