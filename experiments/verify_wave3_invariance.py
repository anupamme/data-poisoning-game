"""
Phase 0 mechanism checks for Wave 3, measured BEFORE the Wave 3 predictions are frozen.

This is the companion to verify_cos_invariance.py, which covered d1 in {norm_clip, rfa}.
Wave 3 introduces two new upstream stages, and one of them is not covered by the
invariance proposition at all, so C2 cannot be assigned from the proposition here:

  reputation  c_i = n * exp(-d_i/s) / sum(...)   strictly positive  -> Prop. invariance(a)
  foolsgold   c_i = n * w_i / sum(w)             w - w.min() makes at least one c_i
                                                 EXACTLY zero every round, so foolsgold
                                                 sits on the boundary of the strict
                                                 hypothesis and is governed by the
                                                 annihilation lemma instead

Test (D) is separate and is the reason this file exists at all. It checks a degeneracy
in the shipped menu that no C2 argument would have caught:

  multi_krum selects sorted-by-score[:k] with k defaulting to 5. Every experiment in
  this project runs clients_per_round = 5, so the selection is ALL FIVE clients and
  _krum(multi=True) reduces to _fedavg -- the same aggregation rule, differing only in
  the order the five updates are summed. "Multi-Krum" is therefore not a distinct
  downstream defense at K=5, and any d1 -> multi_krum pair measures d1 alone.

Tests:
  (A) Synthetic invariance/non-invariance, reusing the mirrors verified in
      verify_cos_invariance.py Test C (no need to re-derive fidelity here).
  (B) Real FL rounds through the SHIPPED apply_d1_transform for d1 in
      {foolsgold, reputation}, recording the realized coefficient vector, its number
      of exact zeros, and whether each downstream statistic moved.
  (D) The multi_krum degeneracy, at K=5 (the protocol) and K=10 (a control), on both
      server.aggregate and the shipped generic_compose path.
  (E) The two OTHER statistics that carry a stored C2=True label on a foolsgold-as-d1
      pair, which (B) does not cover because it measures cosine statistics only:
      RFA's final Weiszfeld weights (foolsgold -> rfa is the paper's emergent case,
      max-committed ASR 0.045) and coord_median's per-client share of selected
      coordinates (foolsgold -> coord_median is one of the two certified pairs).
      Measured on the same rounds as (B), with reputation as the strictly-positive
      control. This decides whether those C2 labels rest on invariance or on something
      else; it does not by itself relabel them.
  (F) Whether norm_clip's tau=5 clip ever BINDS, per attack. The frozen metric_swap
      artifacts show norm_clip -> cos_krum and norm_clip -> cos_reputation are per-seed
      bit-identical to the d2 alone on the pixel arm (3/3) but not on the scaling arm
      (0/3), which is only possible if d1 is the exact identity there. This measures the
      update norms against tau rather than inferring it, because a d1 that is the
      identity means the pair measures d2 alone -- the same near-identity class the paper
      already discloses for FLTrust-as-d2.

Output: results/wave3_invariance_check.json
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform, generic_compose
from experiments.verify_cos_invariance import compare, flatten

output_path = os.path.join(base_dir, "results", "wave3_invariance_check.json")
TAU = 5.0
D1_STAGES = ["foolsgold", "reputation"]
ZERO_TOL = 0.0          # exact, not approximate: the shift-and-clamp gives a true zero


# --- statistics for Test (E), mirroring fl_core.federated ------------------------------

def rfa_weights(stack, max_iter=50):
    """Final Weiszfeld weights, mirroring _rfa / apply_d1_transform's rfa branch."""
    estimate = stack.mean(dim=0)
    for _ in range(max_iter):
        norms = (stack - estimate.unsqueeze(0)).norm(dim=1, keepdim=True).clamp(min=1e-8)
        w = 1.0 / norms
        w = w / w.sum()
        new_estimate = (w * stack).sum(dim=0)
        if (new_estimate - estimate).norm() < 1e-6:
            break
        estimate = new_estimate
    norms = (stack - estimate.unsqueeze(0)).norm(dim=1).clamp(min=1e-8)
    w = 1.0 / norms
    return w / w.sum()


def coord_median_shares(stack):
    """Per-client share of coordinates at which that client supplies the median.

    This is the (P4) quantity the paper reads off coord_median (see
    experiments/build_channel_table.py). With an even client count the median lies
    between two clients and no client "supplies" it, so K=5 (odd) is required for this
    to be well defined -- which is the protocol.
    """
    n = stack.shape[0]
    med = stack.median(dim=0).values
    # index of the client whose value equals the median at each coordinate
    winner = (stack == med.unsqueeze(0)).float().argmax(dim=0)
    return torch.bincount(winner, minlength=n).float() / stack.shape[1]


# --- (A) synthetic, for the two new upstream stages' coefficient shapes ----------------

def test_synthetic(n_trials=1000, K=5, d=2000, seed=0):
    """Invariance of the cos_* statistics under the coefficient shapes d1 actually emits.

    Two shapes are drawn: strictly positive (reputation-like) and one-exact-zero
    (foolsgold-like). The second is the case the proposition does not cover.
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for shape in ("strictly_positive", "one_exact_zero"):
        acc = {}
        for _ in range(n_trials):
            U = torch.randn(K, d, generator=g)
            U[1] = U[0] + 0.05 * torch.randn(d, generator=g)      # a colluding pair
            c = torch.exp(torch.randn(K, generator=g) * 2.0)
            if shape == "one_exact_zero":
                c[int(torch.randint(K, (1,), generator=g).item())] = 0.0
            r = compare(U, U * c.unsqueeze(1))
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        out[shape] = {k: (float(np.max(v)) if "diff" in k else int(sum(v)))
                      for k, v in acc.items()}
        out[shape]["n_trials"] = n_trials
    return out


# --- (B) real FL rounds through the shipped d1 code path -------------------------------

def test_real(seeds=(42, 43, 44), rounds=3, N=10, K=5, f=0.2, attack_name="model_scaling"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    per_round = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        client_datasets, _, num_classes = get_federated_dataset("cifar10", N, 0.5, seed)
        server = FederatedServer(get_model("cifar_cnn", num_classes), device)
        attack = get_attack(attack_name)
        adv_ids = set(range(int(N * f)))
        clients = [
            FederatedClient(
                i,
                attack.poison_dataset(client_datasets[i]) if i in adv_ids else client_datasets[i],
                device)
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
                c = (stack_t.norm(dim=1) / stack_raw.norm(dim=1).clamp(min=1e-12))
                n_zero = int((c <= ZERO_TOL).sum().item())
                nonzero = c[c > ZERO_TOL]
                row = {
                    "seed": int(seed), "d1": d1,
                    "n_clients": int(K),
                    "n_exact_zero_coefficients": n_zero,
                    "n_adversaries_selected": int(sum(1 for cid in selected if cid in adv_ids)),
                    "c_min": float(c.min()), "c_max": float(c.max()),
                    # rho over the strictly positive part only: with an exact zero present
                    # c_max/c_min is +inf and carries no information.
                    "rho_nonzero": float((nonzero.max() / nonzero.min().clamp(min=1e-12)).item())
                    if nonzero.numel() else float("nan"),
                }
                row.update(compare(stack_raw, stack_t))
                row.update(compare_other_statistics(stack_raw, stack_t))
                per_round.append(row)
            server.aggregate(updates)
    return per_round


# --- (E) the other two statistics carrying a stored C2=True on a foolsgold pair --------

def compare_other_statistics(stack_raw, stack_t):
    """RFA's Weiszfeld weights and coord_median's selected-coordinate shares, raw vs. d1."""
    wr_a, wr_b = rfa_weights(stack_raw), rfa_weights(stack_t)
    cm_a, cm_b = coord_median_shares(stack_raw), coord_median_shares(stack_t)
    return {
        "rfa_max_weight_diff": float((wr_a - wr_b).abs().max().item()),
        "rfa_order_changed": bool(not torch.equal(torch.argsort(wr_a), torch.argsort(wr_b))),
        "coord_median_max_share_diff": float((cm_a - cm_b).abs().max().item()),
        "coord_median_order_changed": bool(
            not torch.equal(torch.argsort(cm_a), torch.argsort(cm_b))),
        "coord_median_argmax_changed": bool(
            int(cm_a.argmax().item()) != int(cm_b.argmax().item())),
    }


# --- (F) is norm_clip the identity? per attack ----------------------------------------

def test_norm_clip_activity(seeds=(42, 43, 44), rounds=3, N=10, K=5, f=0.2):
    """Does tau=5 ever bind, under each committed attack?

    Motivation, found in the frozen metric_swap artifacts: on the pixel arm
    norm_clip -> cos_krum and norm_clip -> cos_reputation are per-seed BIT-IDENTICAL
    (3/3) to the corresponding d2 alone, while on the scaling arm they differ (0/3).
    That can only happen if apply_d1_transform is the exact identity there, i.e. if
    every update norm is below tau so scale = min(1, tau/||u||) = 1 for all clients.
    This measures the norms directly instead of inferring it.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for attack_name in ("model_scaling", "backdoor_pixel"):
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            client_datasets, _, num_classes = get_federated_dataset("cifar10", N, 0.5, seed)
            server = FederatedServer(get_model("cifar_cnn", num_classes), device)
            attack = get_attack(attack_name)
            adv_ids = set(range(int(N * f)))
            clients = [
                FederatedClient(
                    i,
                    attack.poison_dataset(client_datasets[i]) if i in adv_ids
                    else client_datasets[i],
                    device)
                for i in range(N)
            ]
            for rnd in range(rounds):
                selected = np.random.choice(N, K, replace=False)
                updates = []
                for cid in selected:
                    u = clients[cid].train(server.global_model, 1, 0.01, 64)
                    if cid in adv_ids:
                        u = attack.manipulate_update(u, server.global_model)
                    updates.append(u)
                stack_raw = flatten(updates)
                norms = stack_raw.norm(dim=1)
                stack_t = flatten(apply_d1_transform(updates, "norm_clip", tau=TAU))
                rows.append({
                    "attack": attack_name, "seed": int(seed), "round": rnd,
                    "min_norm": float(norms.min()), "max_norm": float(norms.max()),
                    "n_clients_above_tau": int((norms > TAU).sum().item()),
                    "n_clients": int(K),
                    # the decisive quantity: is d1 the exact identity this round?
                    "norm_clip_is_identity": bool(torch.equal(stack_raw, stack_t)),
                    "norm_clip_max_abs_change": float((stack_raw - stack_t).abs().max().item()),
                })
                server.aggregate(updates)
    return rows


# --- (D) the multi_krum degeneracy ----------------------------------------------------

def test_multi_krum_degeneracy(seed=0):
    """multi_krum == fedavg whenever clients_per_round <= k (default k=5)."""
    torch.manual_seed(seed)
    model = get_model("cifar_cnn", 10)
    server = FederatedServer(model, "cpu")
    ref = model.state_dict()

    def make(n):
        # deliberately wide norm spread, so a genuine selector cannot coincide by luck
        return [{k: torch.randn_like(v.float()) * (1.0 + 3.0 * i) for k, v in ref.items()}
                for i in range(n)]

    rows = []
    for K in (5, 10):
        ups = make(K)
        a = server.aggregate(ups, method="multi_krum")
        b = server.aggregate(ups, method="fedavg")
        rows.append({
            "clients_per_round": K, "path": "server.aggregate", "d1": None,
            "max_abs_diff": max((a[k] - b[k]).abs().max().item() for k in a),
        })
    for d1 in ["norm_clip", "rfa", "foolsgold", "reputation"]:
        ups = make(5)
        a = generic_compose(server, ups, d1, "multi_krum", tau=TAU)
        b = generic_compose(server, ups, d1, "fedavg", tau=TAU)
        rows.append({
            "clients_per_round": 5, "path": "generic_compose", "d1": d1,
            "max_abs_diff": max((a[k] - b[k]).abs().max().item() for k in a),
        })
    return rows


if __name__ == "__main__":
    print("=== Wave 3 Phase 0: mechanism checks, before predictions are frozen ===\n")

    syn = test_synthetic()
    for shape, s in syn.items():
        print(f"(A) Synthetic, {s['n_trials']} trials, coefficients {shape}:")
        print(f"    cos_reputation  max |w(U)-w(cU)| : {s['cos_reputation_max_weight_diff']:.3e}"
              f"   order changed {s['cos_reputation_order_changed']}/{s['n_trials']}")
        print(f"    cos_krum        selection changed: {s['cos_krum_selection_changed']}"
              f"/{s['n_trials']}")
        print(f"    controls: reputation order changed {s['reputation_order_changed']}"
              f"/{s['n_trials']}, krum selection {s['krum_selection_changed']}/{s['n_trials']}")

    print("\n(B) Real FL updates through shipped apply_d1_transform:")
    real = test_real()
    for r in real:
        print(f"    seed {r['seed']} d1={r['d1']:11s} zeros={r['n_exact_zero_coefficients']}/"
              f"{r['n_clients']} adv_sel={r['n_adversaries_selected']} "
              f"rho+={r['rho_nonzero']:8.1f}  "
              f"cos_rep diff={r['cos_reputation_max_weight_diff']:.2e} "
              f"(order chg {r['cos_reputation_order_changed']})  "
              f"cos_krum chg={r['cos_krum_selection_changed']}  "
              f"krum chg={r['krum_selection_changed']}")

    print("\n(E) The other two statistics with a stored C2=True on a foolsgold pair:")
    for d1 in D1_STAGES:
        rows = [r for r in real if r["d1"] == d1]
        n = len(rows)
        print(f"    d1={d1:11s} rfa weights: max diff {max(r['rfa_max_weight_diff'] for r in rows):.2e}, "
              f"order changed {sum(r['rfa_order_changed'] for r in rows)}/{n}")
        print(f"    {'':15s}coord_median: max share diff "
              f"{max(r['coord_median_max_share_diff'] for r in rows):.2e}, "
              f"order changed {sum(r['coord_median_order_changed'] for r in rows)}/{n}, "
              f"argmax changed {sum(r['coord_median_argmax_changed'] for r in rows)}/{n}")

    print("\n(F) Does norm_clip's tau=5 clip bind? (is d1 the identity?)")
    nc = test_norm_clip_activity()
    for attack_name in ("model_scaling", "backdoor_pixel"):
        rows = [r for r in nc if r["attack"] == attack_name]
        n = len(rows)
        n_ident = sum(r["norm_clip_is_identity"] for r in rows)
        print(f"    {attack_name:15s} norms [{min(r['min_norm'] for r in rows):.3f}, "
              f"{max(r['max_norm'] for r in rows):8.3f}]  "
              f"clients above tau={TAU:g}: "
              f"{sum(r['n_clients_above_tau'] for r in rows)}/"
              f"{sum(r['n_clients'] for r in rows)}  "
              f"-> d1 is the exact identity in {n_ident}/{n} rounds")

    print("\n(D) multi_krum degeneracy:")
    deg = test_multi_krum_degeneracy()
    for r in deg:
        tag = "IDENTICAL to fedavg" if r["max_abs_diff"] == 0.0 else (
            "equal to summation-order error" if r["max_abs_diff"] < 1e-5 else "GENUINELY DIFFERENT")
        who = f"{r['path']}" + (f", d1={r['d1']}" if r["d1"] else "")
        print(f"    K={r['clients_per_round']:>2}  {who:32s} max|diff|={r['max_abs_diff']:.3e}"
              f"  -> {tag}")

    # --- verdicts -------------------------------------------------------------------
    by_d1 = {d1: [r for r in real if r["d1"] == d1] for d1 in D1_STAGES}

    rep_rows = by_d1["reputation"]
    rep_invariant = bool(
        all(r["n_exact_zero_coefficients"] == 0 for r in rep_rows)
        and max(r["cos_reputation_max_weight_diff"] for r in rep_rows) < 1e-6
        and not any(r["cos_krum_selection_changed"] for r in rep_rows)
        and not any(r["cos_reputation_order_changed"] for r in rep_rows))

    fg_rows = by_d1["foolsgold"]
    fg_always_zeroes = bool(all(r["n_exact_zero_coefficients"] > 0 for r in fg_rows))
    fg_cos_krum_moved = int(sum(r["cos_krum_selection_changed"] for r in fg_rows))
    fg_cos_rep_order_moved = int(sum(r["cos_reputation_order_changed"] for r in fg_rows))

    controls_disturbed = bool(
        any(r["reputation_order_changed"] for r in real)
        or any(r["krum_selection_changed"] for r in real))

    mkrum_degenerate_at_5 = bool(all(
        r["max_abs_diff"] < 1e-5 for r in deg if r["clients_per_round"] == 5))
    mkrum_distinct_at_10 = bool(all(
        r["max_abs_diff"] > 1e-5 for r in deg if r["clients_per_round"] == 10))

    results = {
        "description": "Wave 3 Phase 0: C2 mechanism check for d1 in {foolsgold, reputation}, "
                       "plus the multi_krum == fedavg degeneracy at clients_per_round <= k",
        "tau": TAU,
        "d1_stages": D1_STAGES,
        "synthetic": syn,
        "real_per_round": real,
        "multi_krum_degeneracy": deg,
        "n_real_rounds": len(real),
        "reputation_c2_holds": rep_invariant,
        "foolsgold_emits_exact_zero_every_round": fg_always_zeroes,
        "foolsgold_n_rounds_cos_krum_selection_changed": fg_cos_krum_moved,
        "foolsgold_n_rounds_cos_reputation_order_changed": fg_cos_rep_order_moved,
        "foolsgold_n_rounds": len(fg_rows),
        "controls_disturbed": controls_disturbed,
        "multi_krum_degenerate_at_K5": mkrum_degenerate_at_5,
        "multi_krum_distinct_at_K10": mkrum_distinct_at_10,
        "norm_clip_activity_per_round": nc,
        "norm_clip_identity_by_attack": {
            a: {
                "n_rounds": len([r for r in nc if r["attack"] == a]),
                "n_rounds_d1_is_exact_identity":
                    int(sum(r["norm_clip_is_identity"] for r in nc if r["attack"] == a)),
                "n_clients_above_tau":
                    int(sum(r["n_clients_above_tau"] for r in nc if r["attack"] == a)),
                "n_client_rounds":
                    int(sum(r["n_clients"] for r in nc if r["attack"] == a)),
                "max_norm": max(r["max_norm"] for r in nc if r["attack"] == a),
            }
            for a in ("model_scaling", "backdoor_pixel")
        },
        "other_statistics": {
            d1: {
                "n_rounds": len(by_d1[d1]),
                "rfa_max_weight_diff": max(r["rfa_max_weight_diff"] for r in by_d1[d1]),
                "rfa_n_rounds_order_changed":
                    int(sum(r["rfa_order_changed"] for r in by_d1[d1])),
                "coord_median_max_share_diff":
                    max(r["coord_median_max_share_diff"] for r in by_d1[d1]),
                "coord_median_n_rounds_order_changed":
                    int(sum(r["coord_median_order_changed"] for r in by_d1[d1])),
                "coord_median_n_rounds_argmax_changed":
                    int(sum(r["coord_median_argmax_changed"] for r in by_d1[d1])),
            }
            for d1 in D1_STAGES
        },
    }
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n  C2 for d1=reputation (strictly positive)   : "
          f"{'HOLDS' if rep_invariant else 'DOES NOT HOLD'}")
    print(f"  d1=foolsgold emits an exact zero every round: {fg_always_zeroes}")
    print(f"  ... and cos_krum's selection then moved in {fg_cos_krum_moved}/{len(fg_rows)} rounds,"
          f" cos_reputation's ordering in {fg_cos_rep_order_moved}/{len(fg_rows)}")
    print(f"  controls (krum / reputation) disturbed      : {controls_disturbed}")
    print(f"  multi_krum == fedavg at K=5                 : {mkrum_degenerate_at_5}")
    print(f"  multi_krum distinct at K=10                 : {mkrum_distinct_at_10}")
    for a in ("model_scaling", "backdoor_pixel"):
        s = results["norm_clip_identity_by_attack"][a]
        print(f"  norm_clip is the identity under {a:15s}: "
              f"{s['n_rounds_d1_is_exact_identity']}/{s['n_rounds']} rounds "
              f"({s['n_clients_above_tau']}/{s['n_client_rounds']} client-rounds above tau)")
    print(f"\nSaved to {output_path}")
