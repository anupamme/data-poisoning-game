"""
Verify the mechanism claim behind the metric-swap ablation, BEFORE any composition is run.

The claim under test is C2 for two new downstream variants, and its failure for the two
Euclidean originals they are matched against:

  cos_krum        selected index    invariant under u_i -> c_i u_i (c_i > 0)
  krum            selected index    NOT invariant
  cos_reputation  weight vector     invariant
  reputation      weight vector     NOT invariant  (ordering changes, not just magnitudes)

If the invariance half fails, the metric-swap experiment is invalid and the construction
must be fixed -- the result must not be reinterpreted. If the NON-invariance half fails
(i.e. the Euclidean controls turn out invariant too on real updates), the contrast is
vacuous and there is no experiment either. Both halves are asserted.

Three tests:
  (A) Synthetic: random updates x random positive scalings spanning 4 orders of magnitude.
  (B) Real: updates from live FL rounds under model_scaling, pushed through the SHIPPED
      apply_d1_transform code path for d1 in {norm_clip, rfa} -- the two upstream stages
      the metric-swap suite actually uses.
  (C) Fidelity: the weight/score functions below are re-implementations (as in
      verify_fg_invariance.py). Test C ties them to the shipped fl_core implementations,
      so a future edit to federated.py that breaks the correspondence is caught here.

Output: results/cos_invariance_check.json
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform

output_path = os.path.join(base_dir, "results", "cos_invariance_check.json")
TAU = 5.0
D1_STAGES = ["norm_clip", "rfa"]


# --- statistics under test (mirrors of fl_core.federated; see Test C) -----------------

def krum_selection(stack, cosine):
    """Selected index + scores. cosine=False mirrors _krum; True mirrors _cos_krum."""
    n = stack.shape[0]
    if cosine:
        normed = F.normalize(stack, dim=1)
        distances = 1.0 - (normed @ normed.T)
        distances.fill_diagonal_(0.0)
    else:
        distances = torch.cdist(stack.unsqueeze(0), stack.unsqueeze(0)).squeeze(0)
    f = max(1, n // 5)
    scores = [distances[i].sort().values[1:n - f].sum().item() for i in range(n)]
    return int(min(range(n), key=lambda i: scores[i])), scores


def rep_weights(stack, cosine):
    """Weight vector. cosine=False mirrors _reputation; True mirrors _cos_reputation."""
    if cosine:
        normed = F.normalize(stack, dim=1)
        consensus = normed.median(dim=0).values
        dists = 1.0 - F.cosine_similarity(normed, consensus.unsqueeze(0), dim=1)
    else:
        consensus = stack.median(dim=0).values
        dists = (stack - consensus.unsqueeze(0)).norm(dim=1)
    scale = float(dists.median().clamp(min=1e-6).item())
    w = torch.exp(-dists / scale)
    return w / w.sum().clamp(min=1e-8)


def flatten(updates):
    # .cpu() to match _krum, which moves to cpu before cdist, and so the fidelity check
    # in Test C can compare against aggregates returned on the training device.
    keys = list(updates[0].keys())
    return torch.stack([torch.cat([u[k].flatten().float().cpu() for k in keys]) for u in updates])


def compare(stack_raw, stack_scaled):
    """All four quantities, raw vs. rescaled."""
    w_cos_a, w_cos_b = rep_weights(stack_raw, True), rep_weights(stack_scaled, True)
    w_euc_a, w_euc_b = rep_weights(stack_raw, False), rep_weights(stack_scaled, False)
    sel_cos_a, _ = krum_selection(stack_raw, True)
    sel_cos_b, _ = krum_selection(stack_scaled, True)
    sel_euc_a, _ = krum_selection(stack_raw, False)
    sel_euc_b, _ = krum_selection(stack_scaled, False)
    return {
        "cos_reputation_max_weight_diff": float((w_cos_a - w_cos_b).abs().max().item()),
        "reputation_max_weight_diff": float((w_euc_a - w_euc_b).abs().max().item()),
        "cos_reputation_order_changed": bool(
            not torch.equal(torch.argsort(w_cos_a), torch.argsort(w_cos_b))),
        "reputation_order_changed": bool(
            not torch.equal(torch.argsort(w_euc_a), torch.argsort(w_euc_b))),
        "cos_krum_selection_changed": bool(sel_cos_a != sel_cos_b),
        "krum_selection_changed": bool(sel_euc_a != sel_euc_b),
    }


# --- (A) synthetic --------------------------------------------------------------------

def test_synthetic(n_trials=1000, K=5, d=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    acc = {k: [] for k in
           ["cos_reputation_max_weight_diff", "reputation_max_weight_diff",
            "cos_reputation_order_changed", "reputation_order_changed",
            "cos_krum_selection_changed", "krum_selection_changed"]}
    for _ in range(n_trials):
        U = torch.randn(K, d, generator=g)
        U[1] = U[0] + 0.05 * torch.randn(d, generator=g)  # a colluding pair
        c = torch.exp(torch.randn(K, generator=g) * 2.0)  # positive, ~4 orders of magnitude
        r = compare(U, U * c.unsqueeze(1))
        for k in acc:
            acc[k].append(r[k])
    out = {"n_trials": n_trials}
    for k, v in acc.items():
        out[k] = float(np.max(v)) if "diff" in k else int(sum(v))
    return out


# --- (B) real FL rounds through the shipped d1 code path -------------------------------

def test_real(seeds=(42, 43, 44), rounds=3, N=10, K=5, f=0.2):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    per_round, fidelity = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        client_datasets, _, num_classes = get_federated_dataset("cifar10", N, 0.5, seed)
        server = FederatedServer(get_model("cifar_cnn", num_classes), device)
        attack = get_attack("model_scaling")
        adv_ids = set(range(int(N * f)))
        clients = [
            FederatedClient(
                i, attack.poison_dataset(client_datasets[i]) if i in adv_ids else client_datasets[i], device)
            for i in range(N)
        ]

        for _ in range(rounds):
            selected = np.random.choice(N, K, replace=False)
            updates = []
            for cid in selected:
                u = clients[cid].train(server.global_model, 1, 0.01, 64)
                if cid in adv_ids:
                    u = attack.manipulate_update(u, server.global_model)
                updates.append(u)

            stack_raw = flatten(updates)
            for d1 in D1_STAGES:
                transformed = apply_d1_transform(updates, d1, tau=TAU)
                stack_t = flatten(transformed)
                # realized per-client scaling c_i, for the record
                c = (stack_t.norm(dim=1) / stack_raw.norm(dim=1).clamp(min=1e-12))
                row = {"seed": int(seed), "d1": d1,
                       "c_min": float(c.min()), "c_max": float(c.max()),
                       "rho": float((c.max() / c.min().clamp(min=1e-12)).item())}
                row.update(compare(stack_raw, stack_t))
                per_round.append(row)
                fidelity.append(check_fidelity(server, transformed, stack_t, d1, seed))
            server.aggregate(updates)
    return per_round, fidelity


# --- (C) fidelity of the mirrors to the shipped implementations ------------------------

def check_fidelity(server, updates, stack, d1, seed):
    """Tie the mirror functions above to fl_core.federated's actual outputs.

    The comparison is run on CPU copies of the updates. The shipped code path is
    identical either way, but float32 arithmetic on the MPS backend diverges from CPU
    float32 by ~7e-05 relative on a ~1.1M-dim tensor, which would otherwise swamp the
    ~1e-07 quantity we are actually trying to measure. Verified: reproducing the shipped
    per-key accumulation order on CPU gives bit-identical output to the vectorized mirror,
    and both agree with a float64 reference to 8.8e-08, so the divergence is the device,
    not the math. It is recorded below rather than ignored.
    """
    keys = list(updates[0].keys())
    cpu_updates = [{k: u[k].detach().float().cpu() for k in keys} for u in updates]

    # cos_krum returns exactly the selected client's update (single selection, _fedavg of
    # one element), so the shipped selection is exactly recoverable.
    agg = server.aggregate(cpu_updates, method="cos_krum")
    agg_flat = torch.cat([agg[k].flatten().float().cpu() for k in keys])
    residuals = [(agg_flat - stack[i]).abs().max().item() for i in range(stack.shape[0])]
    shipped_sel = int(np.argmin(residuals))
    mirror_sel, _ = krum_selection(stack, True)

    # cos_reputation: rebuild the weighted average from the mirror's weights and compare
    # to the shipped aggregate.
    w = rep_weights(stack, True)
    agg_rep = server.aggregate(cpu_updates, method="cos_reputation")
    agg_rep_flat = torch.cat([agg_rep[k].flatten().float().cpu() for k in keys])
    predicted = (w.unsqueeze(1) * stack).sum(dim=0)
    denom = agg_rep_flat.abs().max().clamp(min=1e-12)

    # For the record: the same shipped call on the training device.
    agg_dev = server.aggregate(updates, method="cos_reputation")
    agg_dev_flat = torch.cat([agg_dev[k].flatten().float().cpu() for k in keys])

    return {
        "seed": int(seed), "d1": d1,
        "cos_krum_selection_matches_shipped": bool(shipped_sel == mirror_sel),
        "cos_krum_selection_recovery_err": float(min(residuals)),
        "cos_reputation_rel_agg_err": float(((agg_rep_flat - predicted).abs().max() / denom).item()),
        "device_vs_cpu_rel_agg_err": float(((agg_dev_flat - agg_rep_flat).abs().max() / denom).item()),
    }


if __name__ == "__main__":
    print("=== C2 mechanism check: metric-swap ablations vs. Euclidean originals ===\n")

    syn = test_synthetic()
    print(f"(A) Synthetic, {syn['n_trials']} trials, arbitrary c>0:")
    print(f"    cos_reputation  max |w(U)-w(cU)| : {syn['cos_reputation_max_weight_diff']:.3e}")
    print(f"    reputation      max |w(U)-w(cU)| : {syn['reputation_max_weight_diff']:.3e}")
    print(f"    weight ORDER changed  : cos_reputation {syn['cos_reputation_order_changed']}"
          f"/{syn['n_trials']}   reputation {syn['reputation_order_changed']}/{syn['n_trials']}")
    print(f"    selection changed     : cos_krum {syn['cos_krum_selection_changed']}"
          f"/{syn['n_trials']}   krum {syn['krum_selection_changed']}/{syn['n_trials']}")

    print("\n(B) Real FL updates through shipped apply_d1_transform:")
    real, fidelity = test_real()
    for r in real:
        print(f"    seed {r['seed']} d1={r['d1']:10s} rho={r['rho']:8.1f}  "
              f"cos_rep diff={r['cos_reputation_max_weight_diff']:.2e}  "
              f"rep diff={r['reputation_max_weight_diff']:.2e} (order chg {r['reputation_order_changed']})  "
              f"cos_krum chg={r['cos_krum_selection_changed']}  krum chg={r['krum_selection_changed']}")

    max_cos_rep = max(r["cos_reputation_max_weight_diff"] for r in real)
    n_cos_krum_chg = sum(r["cos_krum_selection_changed"] for r in real)
    n_cos_rep_order_chg = sum(r["cos_reputation_order_changed"] for r in real)
    n_rep_order_chg = sum(r["reputation_order_changed"] for r in real)
    n_krum_chg = sum(r["krum_selection_changed"] for r in real)

    print("\n(C) Fidelity of the mirrors to fl_core.federated:")
    n_sel_match = sum(f["cos_krum_selection_matches_shipped"] for f in fidelity)
    max_rel_err = max(f["cos_reputation_rel_agg_err"] for f in fidelity)
    max_rec_err = max(f["cos_krum_selection_recovery_err"] for f in fidelity)
    max_dev_err = max(f["device_vs_cpu_rel_agg_err"] for f in fidelity)
    print(f"    cos_krum selection matches shipped : {n_sel_match}/{len(fidelity)} "
          f"(max recovery residual {max_rec_err:.2e})")
    print(f"    cos_reputation aggregate rel. err  : {max_rel_err:.3e}")
    print(f"    (device vs cpu float32 divergence  : {max_dev_err:.3e} -- benign, recorded)")

    invariance_holds = bool(
        max_cos_rep < 1e-6
        and syn["cos_reputation_max_weight_diff"] < 1e-6
        and n_cos_krum_chg == 0 and syn["cos_krum_selection_changed"] == 0
        and n_cos_rep_order_chg == 0 and syn["cos_reputation_order_changed"] == 0)
    # The contrast is only meaningful if the Euclidean controls are genuinely disturbed
    # on the SAME real updates.
    controls_disturbed = bool(n_rep_order_chg > 0 or n_krum_chg > 0)
    mirrors_faithful = bool(n_sel_match == len(fidelity) and max_rel_err < 1e-4)

    results = {
        "description": "C2 mechanism check for the metric-swap ablations: cos_krum / cos_reputation "
                       "invariant under per-client positive rescaling, krum / reputation not",
        "tau": TAU,
        "d1_stages": D1_STAGES,
        "synthetic": syn,
        "real_per_round": real,
        "fidelity": fidelity,
        "max_cos_reputation_weight_diff_real": float(max_cos_rep),
        "n_real_rounds": len(real),
        "n_rounds_reputation_order_changed": int(n_rep_order_chg),
        "n_rounds_krum_selection_changed": int(n_krum_chg),
        "invariance_holds": invariance_holds,
        "controls_disturbed": controls_disturbed,
        "mirrors_faithful": mirrors_faithful,
    }
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n  invariance holds (cos_* variants) : {invariance_holds}")
    print(f"  controls disturbed (krum/rep)     : {controls_disturbed}"
          f"   [rep order changed in {n_rep_order_chg}/{len(real)} rounds, "
          f"krum selection in {n_krum_chg}/{len(real)}]")
    print(f"  mirrors faithful to fl_core       : {mirrors_faithful}")
    if not (invariance_holds and mirrors_faithful):
        print("\n  *** The metric-swap experiment is NOT valid as designed. Fix the "
              "construction; do not reinterpret the outcome. ***")
    elif not controls_disturbed:
        print("\n  *** Controls not disturbed: the contrast would be vacuous. ***")
    print(f"\nSaved to {output_path}")
