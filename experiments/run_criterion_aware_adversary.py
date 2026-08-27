"""
Criterion-aware adversary against FoolsGold->RFA (Move 2, Round-2 review).

Prior adaptive attacks (projection, Neurotoxin, consensus-shift) are tuned to
individual defenses. This experiment builds the adversary that KNOWS FG->RFA is
deployed (Kerckhoffs) and solves the JOINT constraint:

    maximize backdoor signal  subject to
      (a) pairwise cosine similarity among malicious updates below FoolsGold's
          down-weighting regime  (evade FoolsGold), AND
      (b) L2 distance to the running geometric-median estimate below RFA's
          effective rejection radius  (evade RFA).

The two constraints are in tension: coordinating a backdoor raises mutual
similarity (FoolsGold flags it), while staying near the benign geometric median
caps the malicious perturbation (weakening the backdoor). The criterion predicts
FG->RFA holds precisely because no adversary threads both. We test that directly.

Construction each round (whitebox strong adversary observing the round's updates):
  1. Estimate the benign geometric median mu_hat from honest participants.
  2. Build each adversary's poisoned update p_a (pixel backdoor).
  3. RFA-evasion: project onto the eps-ball  u_a = mu_hat + Pi_eps(p_a - mu_hat),
     radius = eps * median(||h_b - mu_hat||)  (stay within benign spread).
  4. FoolsGold-evasion (optional): restrict each adversary's malicious component
     to a DISJOINT coordinate block, making the malicious parts mutually
     orthogonal (cosine ~ 0), so FoolsGold cannot down-weight via similarity.

We sweep eps in {0.5, 1, 2, 4} x decorrelate in {False, True} and, for reference,
run committed_pixel (no evasion) and an RFA-only projected baseline. If FG->RFA
keeps ASR < 0.10 across the sweep, the deployment claim upgrades from "survives
attacks built for other defenses" to "survives an attack built for this one";
if it breaks, we have located the criterion's boundary condition.

Config: N=10, K=5, f=0.2, cifar_cnn, 50 rounds, seeds 42-46
Output: results/criterion_aware_adversary/summary.json
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
from experiments.run_all_compositions import generic_compose

# --- Configuration ---
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
DEFENSE = ("foolsgold", "rfa")  # the flagship composition under attack

# Sweep: (label, eps_radius, decorrelate)
#   eps controls RFA evasion (smaller = closer to benign median = weaker backdoor)
#   decorrelate controls FoolsGold evasion (disjoint coordinate blocks)
CONDITIONS = [
    ("committed_pixel", None, False),      # reference: no evasion (plain backdoor)
    ("ca_eps0.5_decorr", 0.5, True),
    ("ca_eps1_decorr", 1.0, True),
    ("ca_eps2_decorr", 2.0, True),
    ("ca_eps4_decorr", 4.0, True),
    ("ca_eps2_nodecorr", 2.0, False),      # ablation: RFA-evasion only, no FG-evasion
]

output_dir = os.path.join(base_dir, "results", "criterion_aware_adversary")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "Criterion-aware joint-constraint adversary vs FoolsGold->RFA",
        "defense": "foolsgold_then_rfa",
        "config": {"num_clients": FL_CONFIG.num_clients, "clients_per_round": FL_CONFIG.clients_per_round,
                   "adversarial_fraction": ADV_FRACTION, "num_rounds": FL_CONFIG.num_rounds,
                   "model": "cifar_cnn"},
        "conditions": {},
    }


def has_run(label, seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s["conditions"].get(label, {}).get("per_seed", []))


def save_one(label, seed, acc, asr, eps, decorr):
    s = load_or_init()
    if label not in s["conditions"]:
        s["conditions"][label] = {"eps": eps, "decorrelate": decorr, "per_seed": []}
    ps = [r for r in s["conditions"][label]["per_seed"] if r["seed"] != seed]
    ps.append({"seed": seed, "accuracy": float(acc), "asr": float(asr)})
    s["conditions"][label]["per_seed"] = ps
    asrs = [r["asr"] for r in ps]
    s["conditions"][label]["mean_asr"] = float(np.mean(asrs))
    s["conditions"][label]["std_asr"] = float(np.std(asrs))
    s["conditions"][label]["max_asr"] = float(np.max(asrs))
    s["conditions"][label]["mean_acc"] = float(np.mean([r["accuracy"] for r in ps]))
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def _flatten(update, keys):
    return torch.cat([update[k].flatten().float() for k in keys])


def _geometric_median(stack, iters=50):
    """Weyszfeld geometric median of rows of `stack` (m x d)."""
    est = stack.mean(dim=0)
    for _ in range(iters):
        d = (stack - est.unsqueeze(0)).norm(dim=1, keepdim=True).clamp(min=1e-8)
        w = (1.0 / d)
        w = w / w.sum()
        new = (w * stack).sum(dim=0)
        if (new - est).norm() < 1e-6:
            break
        est = new
    return est


def criterion_aware_updates(honest_updates, adv_local_indices, benign_local_indices,
                            keys, eps, decorrelate):
    """Construct joint-constraint malicious updates from this round's honest updates.

    honest_updates: list of update dicts (one per participant, in participation order)
    adv_local_indices / benign_local_indices: positions within honest_updates
    Returns a new list of update dicts with adversarial entries replaced.
    """
    flats = [_flatten(u, keys) for u in honest_updates]
    if len(benign_local_indices) >= 1:
        benign_stack = torch.stack([flats[i] for i in benign_local_indices])
        mu = _geometric_median(benign_stack)
        benign_spread = (benign_stack - mu.unsqueeze(0)).norm(dim=1).median().clamp(min=1e-8).item()
    else:
        mu = torch.stack(flats).mean(dim=0)
        benign_spread = torch.stack(flats).norm(dim=1).median().clamp(min=1e-8).item()

    d = mu.numel()
    n_adv = len(adv_local_indices)
    result = [dict(u) for u in honest_updates]

    for j, li in enumerate(adv_local_indices):
        p = flats[li]                       # adversary's poisoned-training update
        delta = p - mu                      # malicious direction relative to benign median
        # (b) RFA-evasion: project delta into the eps-ball around mu
        radius = eps * benign_spread
        dn = delta.norm().clamp(min=1e-8)
        if dn.item() > radius:
            delta = delta * (radius / dn)
        # (a) FoolsGold-evasion: restrict each adversary to a disjoint coordinate block
        if decorrelate and n_adv > 1:
            mask = torch.zeros(d, device=delta.device)
            block = torch.arange(j, d, n_adv)
            mask[block] = 1.0
            delta = delta * mask
        u_flat = mu + delta
        # unflatten back into the update dict shape
        offset = 0
        newu = {}
        for k in keys:
            numel = honest_updates[li][k].numel()
            newu[k] = u_flat[offset:offset + numel].reshape(honest_updates[li][k].shape).to(
                honest_updates[li][k].device)
            offset += numel
        result[li] = newu
    return result


def run_one(seed, label, eps, decorrelate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed)
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))
    attack = get_attack("backdoor_pixel")  # pixel trigger; ASR measured on it

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = list(np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients), replace=False))

        updates = []
        adv_local, benign_local = [], []
        for pos, cid in enumerate(participant_ids):
            update = clients[cid].train(server.global_model, FL_CONFIG.local_epochs,
                                        current_lr, FL_CONFIG.local_batch_size)
            updates.append(update)
            if cid in adv_ids:
                adv_local.append(pos)
            else:
                benign_local.append(pos)

        if label == "committed_pixel":
            # plain backdoor: standard per-client manipulate (scale toward trigger)
            for pos in adv_local:
                updates[pos] = attack.manipulate_update(updates[pos], server.global_model)
        elif adv_local:
            keys = list(updates[0].keys())
            updates = criterion_aware_updates(updates, adv_local, benign_local,
                                              keys, eps, decorrelate)

        aggregated = generic_compose(server, updates, DEFENSE[0], DEFENSE[1], tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    acc = server.evaluate(test_dataset)["accuracy"]
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(acc), float(asr)


if __name__ == "__main__":
    total = len(CONDITIONS) * len(SEEDS)
    print("=" * 70)
    print("  CRITERION-AWARE ADVERSARY vs FoolsGold->RFA")
    print("=" * 70)
    print(f"  Config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}")
    print(f"  Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"  Seeds: {SEEDS}   Total runs: {total}")
    print()

    t0 = time.time()
    done = 0
    for label, eps, decorr in CONDITIONS:
        print(f"\n--- {label} (eps={eps}, decorrelate={decorr}) ---")
        for seed in SEEDS:
            if has_run(label, seed):
                print(f"  [skip] seed {seed}", flush=True); done += 1; continue
            tr = time.time()
            acc, asr = run_one(seed, label, eps, decorr)
            save_one(label, seed, acc, asr, eps, decorr)
            done += 1
            eta = (time.time() - t0) / done * (total - done) / 3600
            print(f"  seed {seed}: acc={acc:.3f} ASR={asr:.3f} "
                  f"({(time.time()-tr)/60:.1f}min, {done}/{total}, ETA {eta:.1f}h)", flush=True)

    print("\n" + "=" * 70)
    print("  CRITERION-AWARE ADVERSARY — RESULTS (vs FG->RFA)")
    print("=" * 70)
    s = load_or_init()
    print(f"  {'Condition':<22s} {'mean ASR':<12s} {'max ASR':<10s} {'mean acc'}")
    print(f"  {'-'*58}")
    worst = 0.0
    for label, eps, decorr in CONDITIONS:
        c = s["conditions"].get(label, {})
        if "mean_asr" in c:
            worst = max(worst, c["max_asr"])
            print(f"  {label:<22s} {c['mean_asr']:.3f}±{c['std_asr']:.3f}   "
                  f"{c['max_asr']:<10.3f} {c['mean_acc']:.3f}")
    print(f"\n  Worst-case ASR across all criterion-aware conditions: {worst:.3f}")
    print("  Verdict:", "FG->RFA HOLDS (<0.10)" if worst < 0.10 else
          "boundary probed -- FG->RFA admits some ASR under joint attack")
    print(f"  Wall time: {(time.time()-t0)/60:.1f} min\n  Output: {output_path}")
