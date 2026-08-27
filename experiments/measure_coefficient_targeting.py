"""
Adversary-correlated weight dispersion: Delta_c, measured for the REAL upstream transforms.

WHY THIS EXISTS. The Round-11 ladder's abscissa is dispersion, rho = max_i c_i / min_i c_i. Two of the
six real upstream anchors miss the synthetic curve badly at matched statistic disturbance (RFA->Krum
+0.767, NormClip->Reputation +0.789), and the paper's explanation is that rho says how much the weights
spread but not WHO gets the spread. This script measures the quantity that does say who:

    Delta_c := E[c_i | i adversarial] - E[c_i | i benign]

rho is a property of the coefficient vector's range; Delta_c is a property of its correlation with
adversary status. Proposition 1's invariance classes are statements about rho. Lemma 1's attenuation
channel is a statement about Delta_c. They are different quantities and the paper had only ever
reported the first.

NO ASR IS COMPUTED HERE and no model is trained per rung: one raw update stack per (seed, round) is
shared by every transform, so the transforms are compared on identical inputs and adding transforms
costs no training. These figures are POST-HOC AND EXPLORATORY -- they rescore no frozen rule.

WHAT IS MEASURED, per (transform, seed, round):

    c                  coefficients read back from transformed norms, ||T(u_i)|| / ||u_i||, so this
                       measures the shipped code path rather than the builder's documentation
    c_adv_mean         E[c | adversarial]
    c_benign_mean      E[c | benign]
    delta_c            c_adv_mean - c_benign_mean
    delta_c_norm       delta_c / mean(c), the scale-free form. Required for cross-transform
                       comparison: the synthetic families normalize mean(c)=1 but norm_clip,
                       reputation, foolsgold and rfa do not.
    rho_realized       max(c)/min(c), the Round-11 abscissa, for the contrast
    adv_coeff_share    sum_A c / sum c, the quantity the Round-12 instruments pin or sweep

TRANSFORMS. The four real per-client transforms of the defense menu, plus the three synthetic families
re-measured by this same code path so the comparison is apples to apples and so the synthetic shares
can be checked against results/admission_measurement.json WITHOUT overwriting it.

Output: results/coefficient_targeting.json  (a NEW file; admission_measurement.json is never touched)
Run: python3 experiments/measure_coefficient_targeting.py
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_all_compositions import apply_d1_transform
from experiments.verify_cos_invariance import flatten

OUT = os.path.join(base, "results", "coefficient_targeting.json")
FROZEN = os.path.join(base, "results", "admission_measurement.json")

# Identical to measure_admission.py, so the synthetic rungs are directly comparable to the frozen file.
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 3
N, K, F_ADV = 10, 5, 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}
KAPPAS = [0.0, 0.5, 1.0, 2.0]
NUS = [-2.0, -1.0, 0.0, 1.0, 2.0]

# (family, rung, d1 name). "real" is the addition this script exists for: the four upstream transforms
# of the defense menu that actually appear in the compositions the paper scores. Each has a per-client
# reweighting branch in apply_d1_transform -- note that function's docstring lists rfa under "pass
# through" while the code reweights it (an inverse-residual weighting); the code is what runs.
REAL_D1 = ["norm_clip", "reputation", "foolsgold", "rfa"]
RUNGS = ([("real", 0.0, d1) for d1 in REAL_D1]
         + [("dose", k, f"dose_kappa{k}") for k in KAPPAS]
         + [("doseS", k, f"doseS_kappa{k}") for k in KAPPAS]
         + [("doseA", v, f"doseA_nu{v}") for v in NUS])


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
            ben_rows = [i for i in range(len(pids)) if i not in set(adv_rows)]
            adv_mask = [bool(cid in adv) for cid in pids]

            for family, val, d1 in RUNGS:
                t = apply_d1_transform(ups, d1, tau=5.0, dose_key=(seed, rnd), adv_mask=adv_mask)
                st = flatten(t)
                c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
                c_adv = [float(c[i].item()) for i in adv_rows]
                c_ben = [float(c[i].item()) for i in ben_rows]
                cm = float(c.mean().item())
                a_mean = float(np.mean(c_adv)) if c_adv else float("nan")
                b_mean = float(np.mean(c_ben)) if c_ben else float("nan")
                dc = a_mean - b_mean
                rows.append({
                    "family": family, "rung": float(val), "d1": d1, "attack": attack_name,
                    "seed": int(seed), "round": int(rnd),
                    "n_adv_in_round": len(adv_rows), "n_benign_in_round": len(ben_rows),
                    "degenerate": bool(not adv_rows or len(ben_rows) < 2),
                    "c_mean": cm,
                    "c_adv_mean": a_mean,
                    "c_benign_mean": b_mean,
                    "delta_c": dc,
                    "delta_c_norm": dc / cm if cm > 1e-12 else float("nan"),
                    "rho_realized": float((c.max() / c.min().clamp(min=1e-12)).item()),
                    "adv_coeff_share": (float(sum(c_adv) / float(c.sum().item()))
                                        if c_adv else float("nan")),
                })
            srv.apply_update(srv.aggregate(ups))
    return rows


def agg(rows, family, rung, attack=None, field="delta_c_norm"):
    """Mean of a field over the non-degenerate rounds of one cell."""
    sub = [r for r in rows if r["family"] == family and abs(r["rung"] - rung) < 1e-12
           and not r["degenerate"] and (attack is None or r["attack"] == attack)]
    vals = [r[field] for r in sub if not np.isnan(r[field])]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def main():
    print("=== Delta_c: adversary-correlated weight dispersion (no ASR, no per-rung training) ===")
    print(f"    {len(SEEDS)} seeds x {ROUNDS} rounds x {len(RUNGS)} transforms, every transform on")
    print("    the same raw update stack. POST-HOC / EXPLORATORY: rescores no frozen rule.\n",
          flush=True)
    rows = []
    for atk in ("committed_scaling", "committed_pixel"):
        print(f"-- {atk} --", flush=True)
        rows += measure(atk)

    summary = {}

    print("\n=== THE REAL UPSTREAM TRANSFORMS: rho says how much, Delta_c says to whom ===")
    print(f"  {'d1':12s} {'attack':18s} {'rho':>10s} {'E[c|A]':>10s} {'E[c|B]':>10s} "
          f"{'Delta_c':>10s} {'Dc/mean(c)':>11s} {'advshare':>9s}  n")
    for d1 in REAL_D1:
        for atk in ("committed_scaling", "committed_pixel"):
            sub = [r for r in rows if r["d1"] == d1 and r["attack"] == atk and not r["degenerate"]]
            if not sub:
                continue
            g = lambda f: float(np.mean([r[f] for r in sub if not np.isnan(r[f])]))
            summary[f"real|{d1}|{atk}|delta_c"] = g("delta_c")
            summary[f"real|{d1}|{atk}|delta_c_norm"] = g("delta_c_norm")
            summary[f"real|{d1}|{atk}|rho"] = g("rho_realized")
            summary[f"real|{d1}|{atk}|adv_coeff_share"] = g("adv_coeff_share")
            print(f"  {d1:12s} {atk:18s} {g('rho_realized'):10.4f} {g('c_adv_mean'):10.4f} "
                  f"{g('c_benign_mean'):10.4f} {g('delta_c'):+10.4f} {g('delta_c_norm'):+11.4f} "
                  f"{g('adv_coeff_share'):9.4f}  {len(sub)}")

    print("\n=== THE SYNTHETIC FAMILIES, same code path, for the contrast ===")
    print("  dose  : Round 11's ladder. Delta_c is NOT controlled -- the dose is assigned")
    print("          independently of adversary status, so it drifts.")
    print("  doseS : Delta_c pinned by construction (c_adv == 1 exactly at every rung).")
    print("  doseA : Delta_c swept by construction. That is what the instrument is for.")
    print(f"\n  {'family':8s} {'rung':>6s} {'rho':>10s} {'Delta_c':>10s} {'Dc/mean(c)':>11s} "
          f"{'advshare':>9s}  n")
    for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
        for v in vals:
            dcn, n = agg(rows, family, v, field="delta_c_norm")
            dc, _ = agg(rows, family, v, field="delta_c")
            rho, _ = agg(rows, family, v, field="rho_realized")
            sh, _ = agg(rows, family, v, field="adv_coeff_share")
            summary[f"{family}|{v}|delta_c"] = dc
            summary[f"{family}|{v}|delta_c_norm"] = dcn
            summary[f"{family}|{v}|adv_coeff_share"] = sh
            print(f"  {family:8s} {v:>6} {rho:10.4f} {dc:+10.4f} {dcn:+11.4f} {sh:9.4f}  {n}")
        print()

    # Reproduction check against the frozen artifact. This script writes its own output file and never
    # touches admission_measurement.json; the point of the check is that the synthetic shares measured
    # here should land on the frozen ones, which is what licenses reading the real-transform numbers
    # on the same axis as the published Mode-S/Mode-A figures.
    repro = None
    if os.path.exists(FROZEN):
        frozen = json.load(open(FROZEN))["adv_coeff_share"]
        print("=== REPRODUCTION CHECK vs results/admission_measurement.json (not modified) ===")
        worst, repro = 0.0, {}
        for family, vals in (("dose", KAPPAS), ("doseS", KAPPAS), ("doseA", NUS)):
            for v in vals:
                key = f"{family}|{v}"
                if key not in frozen:
                    continue
                here, _ = agg(rows, family, v, field="adv_coeff_share")
                d = abs(here - frozen[key])
                worst = max(worst, d)
                repro[key] = {"frozen": frozen[key], "here": here, "abs_diff": d}
                print(f"  {key:14s} frozen {frozen[key]:.6f}   here {here:.6f}   "
                      f"diff {d:.2e}")
        print(f"\n  worst absolute deviation: {worst:.3e}")
        print("  (a large deviation would mean the training path is not reproducing and the real-")
        print("   transform numbers must not be read alongside the published synthetic ones)")

    json.dump({
        "description": "Adversary-correlated weight dispersion Delta_c = E[c|adv] - E[c|benign] for "
                       "the four real upstream transforms and the three synthetic dose families. "
                       "No ASR. POST-HOC AND EXPLORATORY: rescores no frozen rule. Written to its "
                       "own file; results/admission_measurement.json is never modified.",
        "seeds": SEEDS, "rounds_per_seed": ROUNDS, "kappas": KAPPAS, "nus": NUS,
        "real_d1": REAL_D1,
        "families": ["real", "dose", "doseS", "doseA"],
        "per_round": rows,
        "summary": summary,
        "reproduction_check_vs_admission_measurement": repro,
    }, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
