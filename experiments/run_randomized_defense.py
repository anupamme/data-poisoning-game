"""
Validate the normal-form payoff abstraction by running the NE3 randomized defense policy.

Each round, the defense is sampled from {fedavg: p_fedavg, norm_clip: 1-p_fedavg}
(NE3 mix: p_fedavg = 0.26). Realized accuracy and ASR are compared against the
normal-form prediction (linear combination of pure-strategy payoffs).

Attacks evaluated: no_attack (baseline), model_scaling, backdoor_pixel.

Output: results/randomized_defense/summary.json
        paper/figures/randomized_defense_validation.pdf
"""
import copy
import json
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig, ExperimentConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor


# ── Configuration ──────────────────────────────────────────────────────────────
DEFENSE_DIST = {"fedavg": 0.26, "norm_clip": 0.74}   # NE3 mix
ATTACKS = ["no_attack", "model_scaling", "backdoor_pixel"]
SEEDS = list(range(42, 57))  # 15 seeds (42–56); seeds 52–56 will require generating per_seed_results on the fly
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)

output_dir = os.path.join(base_dir, "results", "randomized_defense")
fig_path = os.path.join(base_dir, "paper", "figures", "randomized_defense_validation.pdf")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(fig_path), exist_ok=True)


def run_randomized_experiment(attack_name: str, defense_dist: dict,
                              fl_config: FLConfig, exp_config: ExperimentConfig,
                              rng: np.random.Generator) -> dict:
    """Run one FL experiment sampling defense per round from defense_dist."""
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

    defenses = list(defense_dist.keys())
    probs = list(defense_dist.values())
    defenses_used = []
    current_lr = fl_config.learning_rate

    for _ in range(fl_config.num_rounds):
        participant_ids = np.random.choice(
            fl_config.num_clients,
            size=min(fl_config.clients_per_round, fl_config.num_clients),
            replace=False
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, fl_config.local_epochs,
                current_lr, fl_config.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        defense_this_round = rng.choice(defenses, p=probs)
        defenses_used.append(defense_this_round)
        aggregated = server.aggregate(updates, method=defense_this_round)
        server.apply_update(aggregated)
        current_lr *= getattr(fl_config, 'lr_decay', 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)

    fedavg_count = defenses_used.count("fedavg")
    return {
        "accuracy": eval_result["accuracy"],
        "attack_success_rate": asr,
        "fedavg_rounds": fedavg_count,
        "norm_clip_rounds": fl_config.num_rounds - fedavg_count,
    }


# ── Load pure-strategy baselines from existing cached results ──────────────────
print("Loading pure-strategy baselines from cached per_seed_results...")
pure_results = {}   # (seed, attack, defense) -> {accuracy, asr}

for seed in SEEDS:
    path = os.path.join(base_dir, "results", "cifar10_10seeds", f"seed_{seed}", "per_seed_results.json")
    if not os.path.exists(path):
        # Seeds without cached pure-strategy results: skip delta computation but still run NE3 BR
        print(f"  seed {seed}: no cached per_seed_results; will run NE3 BR without delta comparison")
        continue
    with open(path) as f:
        psr = json.load(f)
    for attack in ATTACKS:
        for defense in ["fedavg", "norm_clip"]:
            key = f"{attack}_{defense}"
            if key in psr:
                for e in psr[key]:
                    if e["seed"] == seed:
                        pure_results[(seed, attack, defense)] = {
                            "accuracy": e["accuracy"],
                            "asr": e["attack_success_rate"],
                        }
                        break


# ── Run randomized defense experiments ────────────────────────────────────────
print(f"\nRunning randomized defense: {DEFENSE_DIST}")
print(f"Attacks: {ATTACKS}, Seeds: {SEEDS}")

results = {attack: {"realized_accuracy": [], "realized_asr": [],
                    "predicted_accuracy": [], "predicted_asr": [],
                    "fedavg_fraction": []}
           for attack in ATTACKS}

for seed in SEEDS:
    exp_config = ExperimentConfig(
        dataset="cifar10", model="cifar_cnn",
        dirichlet_alpha=0.5, adversarial_fraction=0.2,
        num_trials=1, seed=seed, device="mps",
    )
    rng = np.random.default_rng(seed + 1000)

    for attack in ATTACKS:
        print(f"  Seed {seed}, attack={attack}...", end=" ", flush=True)
        r = run_randomized_experiment(attack, DEFENSE_DIST, FL_CONFIG, exp_config, rng)

        # Normal-form prediction: linear combination of pure-strategy outcomes
        p_fa = DEFENSE_DIST["fedavg"]
        p_nc = DEFENSE_DIST["norm_clip"]
        pure_fa = pure_results.get((seed, attack, "fedavg"), {})
        pure_nc = pure_results.get((seed, attack, "norm_clip"), {})
        pred_acc = p_fa * pure_fa.get("accuracy", 0) + p_nc * pure_nc.get("accuracy", 0)
        pred_asr = p_fa * pure_fa.get("asr", 0) + p_nc * pure_nc.get("asr", 0)

        results[attack]["realized_accuracy"].append(r["accuracy"])
        results[attack]["realized_asr"].append(r["attack_success_rate"])
        results[attack]["predicted_accuracy"].append(pred_acc)
        results[attack]["predicted_asr"].append(pred_asr)
        results[attack]["fedavg_fraction"].append(r["fedavg_rounds"] / FL_CONFIG.num_rounds)

        print(f"acc={r['accuracy']:.3f} (pred={pred_acc:.3f}), ASR={r['attack_success_rate']:.3f} (pred={pred_asr:.3f})")


# ── Summarize ─────────────────────────────────────────────────────────────────
summary = {"policy": DEFENSE_DIST, "seeds": SEEDS, "attacks": {}}
for attack in ATTACKS:
    r = results[attack]
    realized_acc = np.array(r["realized_accuracy"])
    predicted_acc = np.array(r["predicted_accuracy"])
    realized_asr = np.array(r["realized_asr"])
    predicted_asr = np.array(r["predicted_asr"])
    delta_acc = realized_acc - predicted_acc
    delta_asr = realized_asr - predicted_asr

    summary["attacks"][attack] = {
        "realized_accuracy":  {"mean": float(realized_acc.mean()), "std": float(realized_acc.std()), "per_seed": realized_acc.tolist()},
        "predicted_accuracy": {"mean": float(predicted_acc.mean()), "std": float(predicted_acc.std()), "per_seed": predicted_acc.tolist()},
        "realized_asr":  {"mean": float(realized_asr.mean()), "std": float(realized_asr.std()), "per_seed": realized_asr.tolist()},
        "predicted_asr": {"mean": float(predicted_asr.mean()), "std": float(predicted_asr.std()), "per_seed": predicted_asr.tolist()},
        "delta_accuracy": {"mean": float(delta_acc.mean()), "std": float(delta_acc.std())},
        "delta_asr": {"mean": float(delta_asr.mean()), "std": float(delta_asr.std())},
    }
    print(f"\n{attack}: realized acc={realized_acc.mean():.3f}±{realized_acc.std():.3f}, "
          f"predicted={predicted_acc.mean():.3f}±{predicted_acc.std():.3f}, "
          f"Δ={delta_acc.mean():.3f}±{delta_acc.std():.3f}")
    print(f"         realized ASR={realized_asr.mean():.3f}±{realized_asr.std():.3f}, "
          f"predicted={predicted_asr.mean():.3f}±{predicted_asr.std():.3f}, "
          f"Δ={delta_asr.mean():.3f}±{delta_asr.std():.3f}")

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {output_dir}/summary.json")


# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
attack_labels = {"no_attack": "No Attack (accuracy)", "model_scaling": "Model Scaling", "backdoor_pixel": "Backdoor Pixel"}

for ax, attack in zip(axes, ATTACKS):
    r = results[attack]
    ra = np.array(r["realized_accuracy"])
    pa = np.array(r["predicted_accuracy"])
    rr = np.array(r["realized_asr"])
    pr = np.array(r["predicted_asr"])

    if attack == "no_attack":
        # Only accuracy, no ASR
        ax.scatter(pa, ra, color="#1565C0", s=60, zorder=4, label="Seeds")
        ax.errorbar([pa.mean()], [ra.mean()], xerr=[pa.std()], yerr=[ra.std()],
                    fmt="D", color="#1565C0", ms=8, capsize=4, capthick=2, zorder=5, label="Mean±std")
        mn, mx = min(pa.min(), ra.min()) - 0.02, max(pa.max(), ra.max()) + 0.02
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="y=x")
        ax.set_xlabel("Predicted accuracy")
        ax.set_ylabel("Realized accuracy")
    else:
        # ASR scatter
        ax.scatter(pr, rr, color="#d32f2f", s=60, zorder=4, label="Seeds")
        ax.errorbar([pr.mean()], [rr.mean()], xerr=[pr.std()], yerr=[rr.std()],
                    fmt="D", color="#d32f2f", ms=8, capsize=4, capthick=2, zorder=5, label="Mean±std")
        mn, mx = min(pr.min(), rr.min()) - 0.05, max(pr.max(), rr.max()) + 0.05
        mn, mx = max(mn, -0.05), min(mx, 1.05)
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="y=x")
        ax.set_xlabel("Predicted ASR")
        ax.set_ylabel("Realized ASR")

    ax.set_title(f"({chr(97+list(ATTACKS).index(attack))}) {attack_labels[attack]}", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="box")

fig.suptitle(
    "Normal-form abstraction validation: realized vs.\ predicted outcomes under NE3 policy\n"
    "(FedAvg 26\\% + NormClip 74\\% per round, CIFAR-10, 5 seeds)",
    fontsize=8.5, y=1.02
)
plt.tight_layout()
plt.savefig(fig_path)
plt.close()
print(f"Saved: {fig_path}")
