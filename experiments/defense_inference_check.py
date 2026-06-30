"""
Defense-inference sanity check: can an adversary infer the realized defense
from observable per-round signals (aggregated update norms)?

Runs backdoor_pixel x {fedavg, norm_clip, rfa} x 3 seeds x 50 rounds,
logging per-round update norms. Then trains a logistic regression classifier
on sliding windows of norms and reports accuracy vs. window size.

Output: results/defense_inference/inference_results.json
"""
import sys, os, json, copy
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack

DEFENSES = ["fedavg", "norm_clip", "rfa"]
ATTACK = "backdoor_pixel"
SEEDS = [42, 43, 44]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(base_dir, "results", "defense_inference")
os.makedirs(output_dir, exist_ok=True)

fl_config = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
exp_base = dict(dataset="cifar10", model="cifar_cnn", dirichlet_alpha=0.5,
                adversarial_fraction=0.2, num_trials=1)


def run_with_norm_logging(attack_name, defense_name, fl_config, exp_config):
    """Run one FL experiment and return per-round aggregated update norms."""
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

    update_norms = []
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
            updates.append(update)

        aggregated = server.aggregate(updates, method=defense_name)

        # Log the L2 norm of the aggregated update
        norm = torch.cat([v.flatten() for v in aggregated.values()]).norm().item()
        update_norms.append(norm)

        server.apply_update(aggregated)

    return update_norms


# Collect per-round norm traces for each (defense, seed)
all_traces = []  # list of (defense_label, norm_vector)
norm_data_path = os.path.join(output_dir, "norm_traces.json")

if os.path.exists(norm_data_path):
    print("Loading cached norm traces...")
    with open(norm_data_path) as f:
        saved = json.load(f)
    all_traces = [(t["defense"], t["norms"]) for t in saved]
else:
    print(f"Running {len(DEFENSES)} defenses x {len(SEEDS)} seeds x 50 rounds...")
    saved_traces = []
    for defense in DEFENSES:
        for seed in SEEDS:
            print(f"  {ATTACK} vs {defense}, seed={seed}...")
            exp_config = ExperimentConfig(**exp_base, seed=seed, device="mps")
            norms = run_with_norm_logging(ATTACK, defense, fl_config, exp_config)
            all_traces.append((defense, norms))
            saved_traces.append({"defense": defense, "seed": seed, "norms": norms})

    with open(norm_data_path, "w") as f:
        json.dump(saved_traces, f, indent=2)
    print(f"Saved norm traces to {norm_data_path}")

# Train classifier for each window size
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score

n_rounds = len(all_traces[0][1])
window_sizes = [1, 5, 10, 20, 50]

print(f"\n=== Defense-Inference Classifier ===")
print(f"Classes: {DEFENSES}, {len(all_traces)} traces total")
print(f"Windows tested: {window_sizes} rounds\n")

results_by_window = {}
for w in window_sizes:
    features = []
    labels = []
    for defense, norms in all_traces:
        norms_arr = np.array(norms)
        # Use sliding windows across the 50 rounds for each trace
        for start in range(0, n_rounds - w + 1, max(1, w)):
            window = norms_arr[start:start + w]
            feat = [window.mean(), window.std(), window.min(), window.max()]
            features.append(feat)
            labels.append(defense)

    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=500, random_state=0)
    # Use leave-one-out on the original traces to avoid leakage across windows of same trace
    trace_features, trace_labels = [], []
    for defense, norms in all_traces:
        norms_arr = np.array(norms)
        window = norms_arr[:w]
        feat = [window.mean(), window.std(), window.min(), window.max()]
        trace_features.append(feat)
        trace_labels.append(defense)

    X_trace = np.array(trace_features)
    y_trace = le.transform(trace_labels)
    loo_scores = cross_val_score(clf, X_trace, y_trace, cv=LeaveOneOut(), scoring="accuracy")
    acc = loo_scores.mean()

    results_by_window[w] = float(acc)
    print(f"  Window={w:2d} rounds: LOO accuracy = {acc:.1%}")

# Find minimum window to exceed 70%
threshold = 0.70
min_window_70 = None
for w in sorted(results_by_window):
    if results_by_window[w] >= threshold:
        min_window_70 = w
        break

print(f"\nSingle-round (w=1) accuracy: {results_by_window[1]:.1%}")
if min_window_70:
    print(f"Min rounds to exceed {threshold:.0%}: {min_window_70}")
else:
    print(f"Accuracy never exceeds {threshold:.0%} within {max(window_sizes)} rounds")

# Also compute per-defense mean norm statistics
print("\nPer-defense mean update norm (first round):")
for defense in DEFENSES:
    first_round_norms = [norms[0] for d, norms in all_traces if d == defense]
    print(f"  {defense}: mean={np.mean(first_round_norms):.4f}, std={np.std(first_round_norms):.4f}")

out = {
    "defenses": DEFENSES,
    "seeds": SEEDS,
    "n_rounds": n_rounds,
    "window_accuracies": results_by_window,
    "single_round_accuracy": results_by_window[1],
    "min_window_to_70pct": min_window_70,
    "per_defense_mean_norm_r0": {
        defense: float(np.mean([norms[0] for d, norms in all_traces if d == defense]))
        for defense in DEFENSES
    },
}
out_path = os.path.join(output_dir, "inference_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
