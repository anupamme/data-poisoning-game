"""
Admission-level measurement of the dose transforms: what they do to the adversary, not just to the
statistic. No ASR is computed here and no model is trained per rung, so this precedes the freeze.

WHY THIS EXISTS. Round 11 measured disturbance as "the fraction of rounds d2's decision changes"
and found the two quantities the causal story needs are not the same thing: Krum's selection changed
in 86.7% of rounds at rho = 54.6 and its suppression did not deteriorate at all. A decision that
flips from one benign client to another benign client disturbs the statistic maximally and the
mechanism not at all. So this script separates the levels the paper had been sliding between:

  statistic invariance      S(T(U)) = S(U)                        cos_krum: exact
  decision invariance       d2's selection/ordering unchanged     krum: fails at 0.867
  ADMISSION invariance      the adversarial mass d2 admits is unchanged   <- measured here
  suppression preservation  ASR stays below threshold             (the ladders' outcome)

and measures the third for every transform, alongside the adversarial coefficient share that the
Round-12 instruments are built to control.

WHAT IS MEASURED, per (transform, rung, seed, round), all on the SAME raw updates:

  krum, cos_krum   selected index; whether that index is adversarial (admission), before and after
  reputation       adversarial weight share sum_A w / sum w, before and after
  coord_median     fraction of coordinates whose median-attaining client is adversarial, before/after
  every arm        relative displacement of the aggregate, ||agg(T(U)) - agg(U)|| / ||agg(U)||
  the transform    adversarial coefficient share sum_A c / sum c, realized rho, mean(c), c_adv

THREE TRANSFORM FAMILIES:

  dose_kappa<K>    Round 11's ladder, replayed. Its admission numbers are POST-HOC/EXPLORATORY --
                   they do not rescore any frozen rule, and the Round-11 abscissa in
                   results/dose_disturbance.json is left exactly as it was.
  doseS_kappa<K>   Round 12 mode S. The assertion this instrument rests on: the adversarial
                   coefficient share must be CONSTANT across rungs. If it is not, the instrument is
                   still confounded and must be fixed before any ASR is run.
  doseA_nu<V>      Round 12 mode A. Its coefficient share must move monotonically in nu.

Output: results/admission_measurement.json
Run: python3 experiments/measure_admission.py
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform
# Same single-sourced statistic mirrors the Round-11 measurement uses, so the two scripts cannot
# drift apart in what they call "the statistic".
from experiments.verify_cos_invariance import krum_selection, rep_weights, flatten
from experiments.measure_dose_disturbance import argmedian

OUT = os.path.join(base, "results", "admission_measurement.json")
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 3
N, K, F_ADV = 10, 5, 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}

KAPPAS = [0.0, 0.5, 1.0, 2.0]
NUS = [-2.0, -1.0, 0.0, 1.0, 2.0]
# (family, rung value, d1 name). The rung value is the family's own dial: kappa for the two
# dispersion families, nu for the payload family. They are NOT interchangeable and are never pooled.
RUNGS = ([("dose", k, f"dose_kappa{k}") for k in KAPPAS]
         + [("doseS", k, f"doseS_kappa{k}") for k in KAPPAS]
         + [("doseA", v, f"doseA_nu{v}") for v in NUS])


def rel_disp(a, b):
    """||a - b|| / ||b||, the displacement of an aggregate relative to its untransformed value."""
    d = float((a - b).norm().item())
    n = float(b.norm().clamp(min=1e-12).item())
    return d / n


def aggregates(stack, sel_krum, sel_cos, w_rep):
    """Each arm's aggregate output, from the mirrors rather than from a second code path."""
    return {"krum": stack[sel_krum], "cos_krum": stack[sel_cos],
            "coord_median": stack.median(dim=0).values,
            "reputation": (w_rep[:, None] * stack).sum(0) / w_rep.sum().clamp(min=1e-12)}


def admission(stack, adv_rows, sel_krum, sel_cos, w_rep, am):
    """The adversarial mass each arm admits into its aggregate at this round."""
    n = stack.shape[0]
    adv = set(adv_rows)
    return {
        "krum_admits_adv": bool(sel_krum in adv),
        "cos_krum_admits_adv": bool(sel_cos in adv),
        "reputation_adv_weight_share": float(
            sum(w_rep[a].item() for a in adv_rows) / max(float(w_rep.sum().item()), 1e-12)),
        "coord_median_adv_argmedian_frac": float(
            np.mean([1.0 if int(i) in adv else 0.0 for i in am.tolist()])) if n else float("nan"),
    }


def measure(attack_name):
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
            adv_mask = [bool(cid in adv) for cid in pids]

            sel_krum_0, _ = krum_selection(raw, False)
            sel_cos_0, _ = krum_selection(raw, True)
            w_rep_0 = rep_weights(raw, False)
            am_0 = argmedian(raw)
            adm_0 = admission(raw, adv_rows, sel_krum_0, sel_cos_0, w_rep_0, am_0)
            agg_0 = aggregates(raw, sel_krum_0, sel_cos_0, w_rep_0)

            for family, val, d1 in RUNGS:
                t = apply_d1_transform(ups, d1, tau=5.0, dose_key=(seed, rnd), adv_mask=adv_mask)
                st = flatten(t)
                # Coefficients read back from the transformed NORMS, so this also verifies that the
                # shipped code path applied what the builder documents.
                c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
                sel_krum, _ = krum_selection(st, False)
                sel_cos, _ = krum_selection(st, True)
                w_rep = rep_weights(st, False)
                am = argmedian(st)
                adm = admission(st, adv_rows, sel_krum, sel_cos, w_rep, am)
                agg = aggregates(st, sel_krum, sel_cos, w_rep)
                c_adv = [float(c[a].item()) for a in adv_rows]
                row = {
                    "family": family, "rung": float(val), "d1": d1, "attack": attack_name,
                    "seed": int(seed), "round": int(rnd),
                    "n_adv_in_round": len(adv_rows), "n_benign_in_round": K - len(adv_rows),
                    "degenerate": bool(len(adv_rows) == 0 or K - len(adv_rows) < 2),
                    "rho_realized": float((c.max() / c.min().clamp(min=1e-12)).item()),
                    "c_mean": float(c.mean().item()),
                    "c_adv_min": min(c_adv) if c_adv else float("nan"),
                    "c_adv_max": max(c_adv) if c_adv else float("nan"),
                    "adv_coeff_share": (float(sum(c_adv) / float(c.sum().item()))
                                        if c_adv else float("nan")),
                    # decision level, as Round 11 measured it
                    "krum_selection_changed": bool(sel_krum != sel_krum_0),
                    "cos_krum_selection_changed": bool(sel_cos != sel_cos_0),
                    "reputation_order_changed": bool(
                        not torch.equal(torch.argsort(w_rep_0), torch.argsort(w_rep))),
                    "coord_median_frac_argmedian_changed": float((am != am_0).float().mean().item()),
                    # admission level, the distinction this script exists to make
                    "krum_admission_changed": bool(
                        adm["krum_admits_adv"] != adm_0["krum_admits_adv"]),
                    "cos_krum_admission_changed": bool(
                        adm["cos_krum_admits_adv"] != adm_0["cos_krum_admits_adv"]),
                    "reputation_adv_share_delta": (adm["reputation_adv_weight_share"]
                                                   - adm_0["reputation_adv_weight_share"]),
                    "coord_median_adv_frac_delta": (adm["coord_median_adv_argmedian_frac"]
                                                    - adm_0["coord_median_adv_argmedian_frac"]),
                }
                row.update({f"base_{k}": v for k, v in adm_0.items()})
                row.update({f"post_{k}": v for k, v in adm.items()})
                row.update({f"agg_disp_{a}": rel_disp(agg[a], agg_0[a]) for a in agg})
                rows.append(row)
            srv.apply_update(srv.aggregate(ups))
    return rows


ARM_ATTACK = {"krum": "committed_scaling", "reputation": "committed_scaling",
              "cos_krum": "committed_pixel", "coord_median": "committed_pixel"}
DECISION_KEY = {"krum": "krum_selection_changed", "cos_krum": "cos_krum_selection_changed",
                "reputation": "reputation_order_changed",
                "coord_median": "coord_median_frac_argmedian_changed"}
ADMISSION_KEY = {"krum": "krum_admission_changed", "cos_krum": "cos_krum_admission_changed",
                 "reputation": "reputation_adv_share_delta",
                 "coord_median": "coord_median_adv_frac_delta"}


def rate(sub, key):
    """Mean of a boolean or graded per-round field; abs() for the signed share deltas so that a
    change in either direction counts as a change, which is what 'disturbance' has always meant."""
    vals = [abs(float(r[key])) if not isinstance(r[key], bool) else float(r[key]) for r in sub]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    print("=== Admission-level measurement (no ASR, no per-rung training) ===")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} live rounds x {len(RUNGS)} rungs, "
          "every rung on the same raw updates\n", flush=True)
    rows = []
    for atk in ("committed_scaling", "committed_pixel"):
        print(f"-- {atk} --", flush=True)
        rows += measure(atk)

    summary = {}

    def cells(family, arm, val):
        atk = ARM_ATTACK[arm]
        return [r for r in rows if r["family"] == family and r["attack"] == atk
                and abs(r["rung"] - val) < 1e-12]

    print("\n=== DECISION CHANGE vs ADMISSION CHANGE ===")
    print("  (Round 11 put the left number on the abscissa. The right number is the one the")
    print("   mechanism claim is actually about.)\n")
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        dial = "nu" if family == "doseA" else "kappa"
        print(f"  --- {family} ---")
        print(f"  {'arm':26s} " + " ".join(f"{dial+'='+str(v):>15s}" for v in vals))
        for arm in ("krum", "reputation", "cos_krum", "coord_median"):
            cols = []
            for v in vals:
                sub = cells(family, arm, v)
                d, a = rate(sub, DECISION_KEY[arm]), rate(sub, ADMISSION_KEY[arm])
                summary[f"{family}|{arm}|{v}|decision"] = d
                summary[f"{family}|{arm}|{v}|admission"] = a
                summary[f"{family}|{arm}|{v}|agg_disp"] = rate(sub, f"agg_disp_{arm}")
                cols.append(f"{d:.3f}/{a:.3f}")
            print(f"  {arm:26s} " + " ".join(f"{c:>15s}" for c in cols))
        print()

    print("=== THE MODE-S ASSERTION: adversarial coefficient share must be CONSTANT ===")
    share = {}
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        dial = "nu" if family == "doseA" else "kappa"
        seq = []
        for v in vals:
            sub = [r for r in rows if r["family"] == family and abs(r["rung"] - v) < 1e-12
                   and r["n_adv_in_round"] > 0]
            s = float(np.mean([r["adv_coeff_share"] for r in sub]))
            share[f"{family}|{v}"] = s
            seq.append(s)
        spread = float(max(seq) - min(seq))
        verdict = ("CONSTANT" if spread < 1e-9 else
                   f"varies by {spread:.4f}" + (" <- INTENDED" if family == "doseA" else
                                                " <- INSTRUMENT STILL CONFOUNDED"))
        print(f"  {family:6s} ({dial}) " + "  ".join(f"{v:>5}:{s:.4f}" for v, s in zip(vals, seq))
              + f"   {verdict}")
    print("\n  dose  : Round 11. The share moves with the rung -- this is the confound the paper")
    print("          reported: the ladder varies the adversary's weight as well as the statistic.")
    print("  doseS : must be flat. That is what makes a rise in ASR attributable to C2 alone.")
    print("  doseA : must move monotonically. That is the payload channel, isolated.")

    print("\n=== c_adv UNDER MODE S (must be exactly 1.0 at every rung) ===")
    for v in KAPPAS:
        sub = [r for r in rows if r["family"] == "doseS" and abs(r["rung"] - v) < 1e-12
               and r["n_adv_in_round"] > 0]
        lo = min(r["c_adv_min"] for r in sub); hi = max(r["c_adv_max"] for r in sub)
        print(f"  kappa={v}: c_adv in [{lo:.12f}, {hi:.12f}]   max deviation from 1: "
              f"{max(abs(lo - 1), abs(hi - 1)):.2e}")

    print("\n=== REALIZED DOSE ===")
    for family, vals in (("doseS", KAPPAS), ("doseA", NUS)):
        for v in vals:
            sub = [r for r in rows if r["family"] == family and abs(r["rung"] - v) < 1e-12
                   and not r["degenerate"]]
            if not sub:
                continue
            rr = [r["rho_realized"] for r in sub]; cm = [r["c_mean"] for r in sub]
            target = np.exp(2 * v) if family == "doseS" else max(np.exp(v), 1 / np.exp(v))
            print(f"  {family} {v:+.1f}: rho realized [{min(rr):.4f}, {max(rr):.4f}] "
                  f"target {target:8.4f}   mean(c) [{min(cm):.8f}, {max(cm):.8f}]")

    ndegen = sum(r["degenerate"] for r in rows if r["family"] == "doseS")
    ntot = sum(1 for r in rows if r["family"] == "doseS")
    print(f"\n=== DEGENERATE ROUNDS (mode S has no dispersion to impose) ===")
    print(f"  {ndegen}/{ntot} rung-rounds have 0 adversaries or <2 benign participants "
          f"({100.0 * ndegen / max(ntot, 1):.1f}%)")

    json.dump({
        "description": "Admission-level measurement of the Round-11 dose and the two Round-12 "
                       "targeted-dose instruments. No ASR. Round-11 admission figures are POST-HOC.",
        "seeds": SEEDS, "rounds_per_seed": ROUNDS, "kappas": KAPPAS, "nus": NUS,
        "families": ["dose", "doseS", "doseA"],
        "arm_attack": ARM_ATTACK,
        "per_round": rows,
        "summary": summary,
        "adv_coeff_share": share,
        "n_degenerate_modeS_rung_rounds": int(ndegen),
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
