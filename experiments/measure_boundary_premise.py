"""Premise measurement for the decomposition arm: does the frozen instrument meet its own share gate?

WHY THIS EXISTS. experiments/pre_registration_oracle_free_decomposition.md (commit bed6562) freezes a
share gate -- per round, the adversarial coefficient share gap across the arm pair must be at or below
the imported SHARE_TOL of 1e-6 -- and freezes an instrument, the 1001-point float64 Gram grid, that it
asserts can meet it. Its section 3 offers ONE measured pair as the existence proof: a share gap of
5.109e-07 on seed 42's single flip round.

This script asks whether that holds generally, BEFORE any ASR for either arm exists. It reads no
outcome: no accuracy, no ASR, no adversary-conditioned quantity enters any decision here. It is a
premise check of exactly the kind App. D.7 should have run and did not, run in the same order.

THREE ROUTES ARE MEASURED, on the same live stacks, per round:

  route 1  FROZEN         the pair boundary_pair() returns at the frozen grid, h = 1e-3.
  route 2  WRONG-PREDICATE  bracket with the float64 Gram predicate, then bisect THAT predicate to a
                          tight pair, and test the pair with the SHIPPED float32 statistic. This is
                          the obvious repair -- "the gap is linear in h, so shrink h" -- and it is
                          measured here because it does not work, for a reason worth recording.
  route 3  REFINED        bisect the SHIPPED float32 predicate directly, with the stopping rule set on
                          the gated quantity itself (the mask-free supremum share gap), not on a proxy.

TWO GAP QUANTITIES ARE REPORTED, and the second is the one that matters:

  realized    |share_B - share_A| for the round's true adversary rows. This is the freeze's literal text.
  supremum    max over all 30 nonempty proper subsets S of |share_B(S) - share_A(S)|. Mask-free, so it
              bounds the realized gap for EVERY adversary set. A construction that passes on the
              realized gap and fails on the supremum is neutral only for the mask it happened to draw,
              which is weaker than what the freeze's section 3 argues for the blend ("uniformly").

Run:  PYTHONPATH=. python3 -m experiments.measure_boundary_premise
"""

import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.boundary_blend import (  # noqa: E402
    GRID_POINTS, H, SHARE_TOL, SHIPPED_BISECT_MAX_STEPS, SHARE_GAP_TARGET_DIVISOR,
    rfa_coefficients, blend, gram_matrix, gram_krum_selection, first_flip,
    boundary_pair, refined_boundary_pair, apply_coefficients, adv_share, share_gap_sup,
    shipped_selection,
)
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
from experiments.verify_cos_invariance import krum_selection, flatten  # noqa: E402

OUT = "results/oracle_free_decomposition_premise.json"
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 5
TAU = 5.0

# The frozen tolerance and the stopping divisor are both imported, never restated here.
TARGET_DIVISOR = SHARE_GAP_TARGET_DIVISOR


def float64_bracket_bisect(ups, c_rfa, G, steps=40):
    """Route 2: bisect the FLOAT64 Gram predicate to an arbitrarily tight pair.

    This is the repair the freeze's own linearity argument implies: the coefficient gap is exactly
    h * (c_rfa - 1), so shrink h. It is measured rather than assumed because the pair it produces is
    tight around the crossing of the float64 statistic, and the arm is typed by the float32 one.
    """
    t_star, sel0, _, _ = first_flip(G, c_rfa, GRID_POINTS)
    if t_star is None:
        return None
    lo, hi = max(0.0, t_star - H), t_star
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        s, _ = gram_krum_selection(G, blend(c_rfa, mid))
        if s == sel0:
            lo = mid
        else:
            hi = mid
    return blend(c_rfa, lo), blend(c_rfa, hi), float(lo), float(hi)


def pair_row(ups, c_a, c_b, adv_rows):
    """Everything a share-gate verdict could rest on, for one candidate pair, measured both ways."""
    sel_a = shipped_selection(ups, c_a)
    sel_b = shipped_selection(ups, c_b)
    sa, sb = adv_share(c_a, adv_rows), adv_share(c_b, adv_rows)
    realized = (abs(sb - sa) if not (np.isnan(sa) or np.isnan(sb)) else float("nan"))
    sup, arg = share_gap_sup(c_a, c_b)
    return {
        "sel_A_shipped": sel_a, "sel_B_shipped": sel_b,
        "selections_differ_under_shipped_statistic": bool(sel_a != sel_b),
        "coeff_gap_linf": float(np.max(np.abs(np.asarray(c_b) - np.asarray(c_a)))),
        "share_A": sa, "share_B": sb,
        "share_gap_realized": realized,
        "share_gap_sup": float(sup), "argmax_subset": arg,
        "realized_within_tol": bool(not np.isnan(realized) and realized <= SHARE_TOL),
        "sup_within_tol": bool(not np.isnan(sup) and sup <= SHARE_TOL),
    }


def measure():
    from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
    from attacks import get_attack
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, seed)
        srv = FederatedServer(get_model("cifar_cnn", nc), dev)
        atk = get_attack("backdoor_pixel")
        adv = set(range(2))
        cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
              for i in range(10)]
        for rnd in range(ROUNDS):
            pids = np.random.choice(10, 5, replace=False)
            ups = []
            for cid in pids:
                u = cl[cid].train(srv.global_model, 1, 0.01, 64)
                if cid in adv:
                    u = atk.manipulate_update(u, srv.global_model)
                ups.append(u)
            adv_rows = [i for i, cid in enumerate(pids) if cid in adv]
            c_rfa, raw = rfa_coefficients(ups, tau=TAU)
            G = gram_matrix(raw)

            r = {"seed": int(seed), "round": int(rnd),
                 "n_adv_in_round": len(adv_rows),
                 "rho_rfa": float(np.max(c_rfa) / max(np.min(c_rfa), 1e-12))}

            # ---- route 1: the frozen grid, exactly as bed6562 describes it ----
            p1 = boundary_pair(ups, tau=TAU, grid_points=GRID_POINTS)
            r["route1_frozen"] = {"flip": p1["flip"], "t_star": p1["t_star"], "h": p1["h"]}
            if p1["flip"]:
                r["route1_frozen"].update(pair_row(ups, p1["c_A"], p1["c_B"], adv_rows))

            # ---- route 2: bracket and bisect the float64 predicate, then test with float32 ----
            b2 = float64_bracket_bisect(ups, c_rfa, G)
            if b2 is None:
                r["route2_wrong_predicate"] = {"flip": False}
            else:
                ca2, cb2, lo2, hi2 = b2
                r["route2_wrong_predicate"] = {"flip": True, "t_lo": lo2, "t_hi": hi2}
                r["route2_wrong_predicate"].update(pair_row(ups, ca2, cb2, adv_rows))

            # ---- route 3: bisect the shipped predicate, stopping on the gated quantity ----
            p3 = refined_boundary_pair(ups, tau=TAU, share_tol=SHARE_TOL,
                                       gap_target_divisor=TARGET_DIVISOR,
                                       max_steps=SHIPPED_BISECT_MAX_STEPS)
            r["route3_refined"] = {"flip": p3["flip"], "t_star": p3["t_star"],
                                   "n_bisect_steps": p3["n_bisect_steps"],
                                   "n_shipped_evaluations": p3["n_shipped_evaluations"]}
            if p3["flip"]:
                r["route3_refined"].update(pair_row(ups, p3["c_A"], p3["c_B"], adv_rows))

            rows.append(r)
            f1, f3 = r["route1_frozen"], r["route3_refined"]
            print(f"  seed {seed} rnd {rnd}: "
                  f"r1 flip={f1['flip']} realized={f1.get('share_gap_realized', float('nan')):.3e} "
                  f"sup={f1.get('share_gap_sup', float('nan')):.3e} | "
                  f"r2 differ={r['route2_wrong_predicate'].get('selections_differ_under_shipped_statistic')} | "
                  f"r3 flip={f3['flip']} differ={f3.get('selections_differ_under_shipped_statistic')} "
                  f"sup={f3.get('share_gap_sup', float('nan')):.3e} steps={f3['n_bisect_steps']}")
            srv.apply_update(srv.aggregate(ups))
    return rows


def summarize(rows):
    def dosed(key):
        """Rounds where the route produced a flip AND the round has at least one adversary, which is
        the exact population the freeze's share gate quantifies over."""
        return [r for r in rows if r[key].get("flip") and r["n_adv_in_round"] > 0]

    out = {}
    for key in ("route1_frozen", "route2_wrong_predicate", "route3_refined"):
        d = dosed(key)
        rl = [r[key]["share_gap_realized"] for r in d
              if not np.isnan(r[key].get("share_gap_realized", float("nan")))]
        sp = [r[key]["share_gap_sup"] for r in d]
        differ = [r[key]["selections_differ_under_shipped_statistic"] for r in d]
        out[key] = {
            "n_dosed_adversary_rounds": len(d),
            "n_realized_exceeding_tol": int(sum(1 for g in rl if g > SHARE_TOL)),
            "n_sup_exceeding_tol": int(sum(1 for g in sp if g > SHARE_TOL)),
            "max_realized_gap": float(max(rl)) if rl else float("nan"),
            "max_sup_gap": float(max(sp)) if sp else float("nan"),
            "min_realized_gap": float(min(rl)) if rl else float("nan"),
            "n_selections_differ_under_shipped": int(sum(1 for b in differ if b)),
            "n_selections_identical_under_shipped": int(sum(1 for b in differ if not b)),
            "share_gate_passes_every_dosed_round": bool(
                len(d) > 0 and all(g <= SHARE_TOL for g in sp)),
            "flip_gate_passes_every_dosed_round": bool(len(d) > 0 and all(differ)),
        }
    r3 = out["route3_refined"]
    steps = [r["route3_refined"]["n_bisect_steps"] for r in rows if r["route3_refined"]["flip"]]
    evals = [r["route3_refined"]["n_shipped_evaluations"] for r in rows]
    r3["bisect_steps_min_max"] = [int(min(steps)), int(max(steps))] if steps else None
    r3["shipped_evaluations_per_round_min_max"] = [int(min(evals)), int(max(evals))]
    both = (out["route3_refined"]["share_gate_passes_every_dosed_round"]
            and out["route3_refined"]["flip_gate_passes_every_dosed_round"])
    out["verdict"] = {
        "frozen_instrument_meets_its_own_share_gate": bool(
            out["route1_frozen"]["share_gate_passes_every_dosed_round"]),
        "wrong_predicate_repair_retains_a_decision_change": bool(
            out["route2_wrong_predicate"]["flip_gate_passes_every_dosed_round"]),
        "refined_instrument_meets_both_gates": bool(both),
        "label": ("FROZEN_INSTRUMENT_PREMISE_FAILS__REFINED_INSTRUMENT_PASSES" if both and not
                  out["route1_frozen"]["share_gate_passes_every_dosed_round"] else "SEE_FIELDS"),
    }
    return out


def main():
    print("=== decomposition arm premise measurement (no ASR, no accuracy, no outcome) ===")
    print(f"frozen grid {GRID_POINTS} points, h = {H:g};  SHARE_TOL = {SHARE_TOL:g} (imported);  "
          f"route 3 target = SHARE_TOL/{TARGET_DIVISOR:g}")
    print(f"seeds {SEEDS} x {ROUNDS} live rounds\n")
    rows = measure()
    summary = summarize(rows)
    print("\n--- summary ---")
    for k in ("route1_frozen", "route2_wrong_predicate", "route3_refined"):
        s = summary[k]
        print(f"{k}: dosed rounds {s['n_dosed_adversary_rounds']}, "
              f"realized over tol {s['n_realized_exceeding_tol']}, "
              f"sup over tol {s['n_sup_exceeding_tol']}, "
              f"max sup {s['max_sup_gap']:.3e}, "
              f"selections differ {s['n_selections_differ_under_shipped']}/"
              f"{s['n_dosed_adversary_rounds']}")
    print(f"verdict: {summary['verdict']['label']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {
        "purpose": ("Premise check for the decomposition arm, run before any ASR for either arm "
                    "exists. Reads no outcome. Asks whether the instrument frozen in "
                    "experiments/pre_registration_oracle_free_decomposition.md (commit bed6562) "
                    "meets that document's own share gate, and measures two alternatives."),
        "frozen_prereg": "experiments/pre_registration_oracle_free_decomposition.md",
        "frozen_prereg_commit": "bed6562",
        "share_tol": SHARE_TOL,
        "share_tol_source": "experiments/measure_admission.py (imported, not restated)",
        "grid_points": GRID_POINTS, "h": H,
        "route3_target_share_gap_sup": SHARE_TOL / float(TARGET_DIVISOR),
        "bisect_max_steps": SHIPPED_BISECT_MAX_STEPS,
        "seeds": SEEDS, "rounds_per_seed": ROUNDS, "tau": TAU,
        "gap_definitions": {
            "share_gap_realized": ("|share_B - share_A| over the round's true adversary rows. This "
                                   "is the freeze's literal text."),
            "share_gap_sup": ("supremum over all 30 nonempty proper subsets of |share_B(S) - "
                              "share_A(S)|. Mask-free, so it bounds the realized gap for every "
                              "adversary set and matches what the freeze's section 3 argues for the "
                              "blend, namely UNIFORM neutrality rather than neutrality for one mask."),
        },
        "summary": summary,
        "per_round": rows,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
