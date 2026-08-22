"""
Measure, independently of any ASR, how much each dose rung actually disturbs each arm's statistic.

The dose-response ladder's abscissa must be MEASURED, not nominal. kappa and rho = exp(2*kappa) are
what we impose; what matters to C2 is what the imposed rescaling does to the specific statistic the
downstream defense reads. This script supplies that, per (arm, kappa), on live FL updates pushed
through the SHIPPED apply_d1_transform code path -- the same design as Test B of
experiments/verify_cos_invariance.py, whose mirror functions are imported here rather than
duplicated so a future edit to fl_core cannot silently desynchronize the two.

  arm 4 statistic     what is measured                                      Prop 1 class
  krum                fraction of rounds the selected client changes        (c) not invariant
  reputation          fraction of rounds the weight ORDERING changes        (c) not invariant
  cos_krum            same as krum -- ASSERTED to be 0 at every kappa       (a) exactly invariant
  coord_median        fraction of COORDINATES whose median-attaining        (b) conditionally
                      client changes (graded, as befits class (b)),         invariant
                      plus the ordering-survival indicator below

For coord_median the theory gives a sharp quantitative prediction, not just "some threshold".
Theorem 1(1)'s bounded-ratio condition, in the two-regime form the paper states it in, is
S * r > 1 with r := w_a / max_b w_b the adversary's relative weight and S the attack's coordinate
extremeness. Under a dose transform whose coefficients are assigned independently of adversary
status, r >= 1/rho, so ordering survival is implied by

    rho  <  S        (equivalently S/rho > 1, the margin form used in the paper)

S is a property of the ATTACK, not of the dose, so it is measured here -- before the ladder runs --
and the predicted crossing rung follows from it arithmetically. The paper already reports S = 10 for
model-scaling; the pixel backdoor is a different attack and its S must not be assumed equal.

Trajectories are advanced with plain FedAvg, and every kappa is evaluated on the SAME raw updates.
That is deliberate: it isolates the quantity we want on the abscissa -- what the transform does to
the statistic -- from trajectory divergence, which is the ASR outcome the ladder measures
separately. No model is trained per (arm, kappa).

Output: results/dose_disturbance.json
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform
# Single-sourced mirrors of fl_core.federated's statistics; Test C of verify_cos_invariance.py
# ties them to the shipped implementations.
from experiments.verify_cos_invariance import krum_selection, rep_weights, flatten
from experiments.run_dose_response import ARMS, KAPPAS, SEEDS, d1_name, rho

OUT = os.path.join(base, "results", "dose_disturbance.json")
ROUNDS = 3                 # live rounds per seed, as in verify_cos_invariance.test_real
N, K, F_ADV = 10, 5, 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}
TOL = 1e-7                 # float32 rounding floor for the class-(a) invariance assertion


def argmedian(stack):
    """Per coordinate, the index of the client attaining the coordinate-wise median.

    torch.median(dim=0) returns an actual element of the input (not an average), so for odd K the
    index is exact and this is precisely the client CoordMedian's output is drawn from at that
    coordinate -- i.e. the decision the class-(b) ordering claim is about.
    """
    return stack.median(dim=0).indices


def coordinate_extremeness(stack, adv_rows):
    """S: how extreme the adversarial coordinates are against the benign scale.

    S := median_k ( |u_a,k| / median_{b benign} |u_b,k| ), then the max over adversaries present.
    Chosen to reproduce the quantity the paper already reports (S = 10 for model-scaling) and to be
    readable off the raw updates alone. Scale-free in the sense that matters: multiplying every
    client's update by one constant leaves S unchanged, so S measures the attack, not the round.
    """
    ben = [i for i in range(stack.shape[0]) if i not in adv_rows]
    if not ben or not adv_rows:
        return float("nan")
    scale = stack[ben].abs().median(dim=0).values.clamp(min=1e-12)
    return float(max((stack[a].abs() / scale).median().item() for a in adv_rows))


def measure(attack_name):
    """Per (kappa, seed, round): disturbance of every statistic on the same raw updates."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cd, _, nc = get_federated_dataset("cifar10", N, 0.5, seed)
        srv = FederatedServer(get_model("cifar_cnn", nc), dev)
        atk = get_attack(ATTACK_MAP[attack_name])
        adv = set(range(int(N * F_ADV)))
        cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
              for i in range(N)]
        for rnd in range(ROUNDS):
            pids = np.random.choice(N, K, replace=False)
            ups = []
            for cid in pids:
                u = cl[cid].train(srv.global_model, 1, 0.01, 64)
                if cid in adv:
                    u = atk.manipulate_update(u, srv.global_model)
                ups.append(u)
            raw = flatten(ups)
            adv_rows = [i for i, cid in enumerate(pids) if cid in adv]
            S = coordinate_extremeness(raw, adv_rows)
            sel_krum_0, _ = krum_selection(raw, False)
            sel_cos_0, sc_cos_0 = krum_selection(raw, True)
            sc_cos_0 = torch.tensor(sc_cos_0)
            w_rep_0 = rep_weights(raw, False)
            am_0 = argmedian(raw)
            for kappa in KAPPAS:
                t = apply_d1_transform(ups, d1_name(kappa), tau=5.0, dose_key=(seed, rnd))
                st = flatten(t)
                c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
                sel_krum, _ = krum_selection(st, False)
                sel_cos, sc_cos = krum_selection(st, True)
                w_rep = rep_weights(st, False)
                am = argmedian(st)
                # r = c_a / max_b c_b: the adversary's relative weight actually realized this
                # round. The dose is assigned by a permutation independent of client identity, so
                # r fluctuates around 1/rho rather than being pinned to it; both are recorded.
                r = (float(min(c[a].item() for a in adv_rows)
                           / max(c[b].item() for b in range(len(c)) if b not in adv_rows))
                     if adv_rows and len(adv_rows) < len(c) else float("nan"))
                rows.append({
                    "attack": attack_name, "seed": int(seed), "round": int(rnd),
                    "kappa": kappa, "rho_nominal": rho(kappa),
                    "rho_realized": float((c.max() / c.min().clamp(min=1e-12)).item()),
                    "c_mean": float(c.mean().item()),
                    "S": S, "r": r, "S_times_r": (S * r if r == r else float("nan")),
                    "rho_lt_S": bool(rho(kappa) < S),
                    "krum_selection_changed": bool(sel_krum != sel_krum_0),
                    "cos_krum_selection_changed": bool(sel_cos != sel_cos_0),
                    "cos_krum_score_max_rel_diff": float(
                        ((torch.tensor(sc_cos) - sc_cos_0).abs().max()
                         / sc_cos_0.abs().max().clamp(min=1e-12)).item()),
                    "reputation_order_changed": bool(
                        not torch.equal(torch.argsort(w_rep_0), torch.argsort(w_rep))),
                    "reputation_max_weight_diff": float((w_rep_0 - w_rep).abs().max().item()),
                    "coord_median_frac_argmedian_changed": float((am != am_0).float().mean().item()),
                })
            srv.apply_update(srv.aggregate(ups))
    return rows


def arm4_survival(rows):
    """Theorem 1(1)'s ordering-survival condition rho < S, per attack, from the measured S.

    Rounds that sampled no adversary carry no S (K=5 of N=10 with 2 adversaries leaves a 22% chance
    of an all-benign round), so they are excluded rather than allowed to propagate NaN through the
    median. The count of usable rounds is reported alongside, so the exclusion is visible.
    """
    out = {}
    for atk in sorted({r["attack"] for r in rows}):
        first = [r for r in rows if r["attack"] == atk and r["kappa"] == KAPPAS[0]]
        Ss = [r["S"] for r in first if r["S"] == r["S"]]          # drop NaN: no adversary sampled
        if not Ss:
            out[atk] = {"S_median": float("nan"), "n_rounds": len(first),
                        "n_rounds_with_adversary": 0}
            continue
        S = float(np.median(Ss))
        out[atk] = {
            "S_median": S, "S_min": float(np.min(Ss)), "S_max": float(np.max(Ss)),
            "n_rounds": len(first), "n_rounds_with_adversary": len(Ss),
            "kappas_predicted_order_preserving": [k for k in KAPPAS if rho(k) < S],
            "first_kappa_predicted_to_break": next((k for k in KAPPAS if rho(k) >= S), None),
            "regime_A_applies_at_identity": bool(S > 1.0),
        }
    return out


KEY = {"krum": "krum_selection_changed",
       "cos_krum": "cos_krum_selection_changed",
       "reputation": "reputation_order_changed",
       "coord_median": "coord_median_frac_argmedian_changed"}


def main():
    print("=== Measured disturbance per dose rung (no ASR, no per-arm training) ===")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} live rounds, every kappa on the same raw updates\n",
          flush=True)
    rows = []
    for atk in sorted({a[1] for a in ARMS}):
        print(f"-- {atk} --", flush=True)
        rows += measure(atk)

    # --- the class-(a) assertion: cos_krum's SELECTION must be untouched at every kappa ---
    # The selection is the decision CoordMedian-style class-(a) invariance is about and is what the
    # negative control rests on, so the assertion is on the selection. The scores are compared in
    # RELATIVE terms and recorded rather than asserted: float32 normalization of a ~1.1M-dim vector
    # at rho up to 54.6 legitimately perturbs them at the ~1e-6 level, which would make an absolute
    # assertion a test of the floating-point unit rather than of the mechanism.
    viol = [r for r in rows if r["cos_krum_selection_changed"]]
    max_score_diff = max(r["cos_krum_score_max_rel_diff"] for r in rows)
    invariance_holds = not viol

    print("\n=== DISTURBANCE BY ARM AND RUNG ===")
    print(f"  {'arm':26s} {'class':22s} " + " ".join(f"{'k='+str(k):>10s}" for k in KAPPAS))
    summary = {}
    for d2, atk, cls, shape in ARMS:
        vals = []
        for kappa in KAPPAS:
            sub = [r for r in rows if r["attack"] == atk and r["kappa"] == kappa]
            k = KEY[d2]
            if d2 == "coord_median":
                v = float(np.mean([r[k] for r in sub])); vals.append(f"{v:.4f}")
            else:
                n = sum(bool(r[k]) for r in sub); vals.append(f"{n}/{len(sub)}")
                v = n / len(sub)
            summary[f"{d2}|{atk}|kappa{kappa}"] = v
        print(f"  {d2 + '/' + atk.replace('committed_',''):26s} {'Prop 1 ' + cls:22s} "
              + " ".join(f"{v:>10s}" for v in vals))
    print("\n  krum / cos_krum: rounds the selected client changed."
          "\n  reputation: rounds the weight ordering changed."
          "\n  coord_median: mean fraction of COORDINATES whose median-attaining client changed.")

    # --- monotonicity of the dose: if disturbance does not grow with kappa there is no dose ---
    print("\n=== IS THERE ACTUALLY A DOSE? (disturbance must grow with kappa) ===")
    dose_is_monotone = {}
    for d2, atk, _, _ in ARMS:
        seq = [summary[f"{d2}|{atk}|kappa{k}"] for k in KAPPAS]
        mono = all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))
        dose_is_monotone[f"{d2}|{atk}"] = bool(mono)
        print(f"  {d2 + '/' + atk.replace('committed_',''):26s} "
              + " -> ".join(f"{v:.3f}" for v in seq)
              + f"   {'nondecreasing' if mono else 'NOT MONOTONE'}")
    print("  (cos_krum is expected to be flat at 0.000 -- that is the negative control, not a"
          "\n   failure of the dose; the class-(c) arms are the ones that must rise.)")

    # --- arm 4: the ordering-survival prediction, from the MEASURED S ---
    print("\n=== ARM 4 (coord_median): predicted crossing from Theorem 1(1), rho < S ===")
    cross = arm4_survival(rows)
    for atk, d in cross.items():
        print(f"  {atk.replace('committed_',''):9s} measured S = {d['S_median']:.2f} "
              f"[{d['S_min']:.2f}, {d['S_max']:.2f}] over {d['n_rounds_with_adversary']} of "
              f"{d['n_rounds']} rounds that sampled an adversary")
        print(f"    rho < S at kappa in {d['kappas_predicted_order_preserving']};  "
              f"first rung predicted to break ordering: {d['first_kappa_predicted_to_break']}")
    print("  S < 1 means the attack's coordinates are not extreme against the benign scale at all,")
    print("  so Theorem 1(1)'s Regime-A condition rho < S fails even at the identity rung and the")
    print("  theorem offers that arm no protection to lose. DIAGNOSTIC ONLY: as the min/max show,")
    print("  S is not a stable per-attack constant under this definition -- it swings by orders of")
    print("  magnitude across rounds -- so NO frozen prediction is built on it. The paper's S=10 is")
    print("  a stipulated illustrative value, not a measurement, and is not used here either. Arm 4's")
    print("  predicted shape comes from the measured graded disturbance above instead.")

    print("\n=== REALIZED DOSE (sanity: rho_realized must equal exp(2*kappa), mean c = 1) ===")
    for kappa in KAPPAS:
        sub = [r for r in rows if r["kappa"] == kappa]
        rr = [r["rho_realized"] for r in sub]
        cm = [r["c_mean"] for r in sub]
        print(f"  kappa={kappa}: rho nominal {rho(kappa):8.3f}  realized "
              f"[{min(rr):.3f}, {max(rr):.3f}]   mean(c) [{min(cm):.6f}, {max(cm):.6f}]")
    print("  (rho_realized is read back from the transformed update NORMS, so it also verifies that")
    print("   the shipped code path applies the coefficients it is documented to apply.)")

    out = {
        "description": "Measured disturbance of each arm's statistic per dose rung; the abscissa of "
                       "the dose-response curves, and the class-(a) invariance assertion",
        "rounds_per_seed": ROUNDS, "seeds": SEEDS, "kappas": KAPPAS,
        "arms": [{"d2": d2, "attack": a, "prop1_class": c, "predicted_shape": s}
                 for d2, a, c, s in ARMS],
        "per_round": rows,
        "summary_disturbance": summary,
        "dose_is_monotone": dose_is_monotone,
        "arm4_ordering_survival": cross,
        "cos_krum_invariance_holds": invariance_holds,
        "cos_krum_max_score_rel_diff": float(max_score_diff),
        "n_cos_krum_violations": len(viol),
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"\n  class-(a) invariance (cos_krum selection unchanged at every kappa): "
          f"{invariance_holds}")
    print(f"    violations {len(viol)}/{len(rows)}; max RELATIVE cos-score deviation "
          f"{max_score_diff:.2e} (recorded, not asserted; float32 floor ~{TOL:g})")
    if not invariance_holds:
        print("\n  *** Arm 3 is NOT a valid negative control as constructed. Fix the")
        print("      construction; do not reinterpret the ladder's arm-3 outcome. ***")
    if not any(dose_is_monotone[f"{d2}|{a}"] and summary[f"{d2}|{a}|kappa{KAPPAS[-1]}"] > 0
               for d2, a, _, _ in ARMS if d2 != "cos_krum"):
        print("\n  *** No class-(c)/(b) arm shows disturbance growing with kappa: there is no dose")
        print("      and no experiment. Fix the transform before running the ladder. ***")
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
