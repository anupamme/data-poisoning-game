"""
POST-HOC re-analysis of the Round-11 dose ladder against two predictors it did not use.

NOTHING HERE RESCORES A FROZEN RULE. The Round-11 primary test (pooled Spearman of ASR rise against
measured DECISION change, one-sided, alpha 0.05) was refuted at rho_s = +0.351, p = 0.091, and stays
refuted. This script asks a different, exploratory question that the Round-11 outcome forced:

  Krum's decision changed in 86.7% of rounds at rho = 54.6 and its suppression did not deteriorate.
  Was the wrong thing on the abscissa?

Two candidate predictors, both measured without any ASR by experiments/measure_admission.py:

  ADMISSION CHANGE   how often the transform changes whether d2 admits ADVERSARIAL input --
                     the selected client's adversary status for krum/cos_krum, the adversarial
                     weight share for reputation, the adversarial share of median-attaining
                     coordinates for coord_median. A flip from one benign client to another
                     disturbs the statistic maximally and the mechanism not at all.
  SHARE CHANGE       the change in the adversary's share of the total coefficient mass. Round 11's
                     permutation assigned the dose independently of adversary status, so a wide
                     spread DILUTES the poisoned client: this is the attenuation channel the paper
                     named as a limitation of its own instrument, here quantified.

The paper's Round-11 finding was that three arms FELL. A predictor of the rise cannot explain a
fall, so both channels are entered together as well as separately. Every number is recomputed from
results/dose_response/summary.json and results/admission_measurement.json.

Run: python3 experiments/analyze_admission.py
"""
import json, os, sys
import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(base, "results", "dose_response", "summary.json")
ADM = os.path.join(base, "results", "admission_measurement.json")

DECISION_KEY = {"krum": "krum_selection_changed", "cos_krum": "cos_krum_selection_changed",
                "reputation": "reputation_order_changed",
                "coord_median": "coord_median_frac_argmedian_changed"}
ADMISSION_KEY = {"krum": "krum_admission_changed", "cos_krum": "cos_krum_admission_changed",
                 "reputation": "reputation_adv_share_delta",
                 "coord_median": "coord_median_adv_frac_delta"}


def rate(sub, key):
    return float(np.mean([abs(float(r[key])) if not isinstance(r[key], bool) else float(r[key])
                          for r in sub])) if sub else float("nan")


def spearman(x, y, alt="greater"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if sps is None:
        return float("nan"), float("nan")
    r = sps.spearmanr(x, y)
    rho = float(r.statistic)
    n = len(x)
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    p = float(sps.t.sf(t, n - 2)) if alt == "greater" else float(sps.t.cdf(t, n - 2))
    return rho, p


def main():
    for p in (LADDER, ADM):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    ladder = json.load(open(LADDER))
    adm = json.load(open(ADM))
    rows = [r for r in adm["per_round"] if r["family"] == "dose"]
    kappas = ladder["config"]["kappas"]
    by_cell = {(c["d2"], c["attack"], c["kappa"]): c for c in ladder["cells"].values()}

    print("=== PROVENANCE CHECK: does the replay reproduce Round 11's abscissa? ===")
    print("    (same seeds, same live rounds, same shipped transform; if these differ, the replay")
    print("     is not measuring the same instrument and nothing below is comparable)\n")
    ok = True
    for a in ladder["arms"]:
        d2, atk = a["d2"], a["attack"]
        seq = [rate([r for r in rows if r["attack"] == atk and abs(r["rung"] - k) < 1e-12],
                    DECISION_KEY[d2]) for k in kappas]
        print(f"  {d2:13s} replay decision change: " + " ".join(f"{v:.3f}" for v in seq))
    print()

    cells = []
    for a in ladder["arms"]:
        d2, atk = a["d2"], a["attack"]
        base_asr = float(np.mean([r["asr"] for r in by_cell[(d2, atk, kappas[0])]["per_seed"]]))
        base_share = float(np.mean([r["adv_coeff_share"] for r in rows
                                    if r["attack"] == atk and abs(r["rung"] - kappas[0]) < 1e-12
                                    and r["n_adv_in_round"] > 0]))
        for k in kappas:
            sub = [r for r in rows if r["attack"] == atk and abs(r["rung"] - k) < 1e-12]
            sub_adv = [r for r in sub if r["n_adv_in_round"] > 0]
            asr = float(np.mean([r["asr"] for r in by_cell[(d2, atk, k)]["per_seed"]]))
            cells.append(dict(
                d2=d2, attack=atk, kappa=k, asr=asr, rise=asr - base_asr,
                decision=rate(sub, DECISION_KEY[d2]),
                admission=rate(sub, ADMISSION_KEY[d2]),
                share=float(np.mean([r["adv_coeff_share"] for r in sub_adv])),
                dshare=float(np.mean([r["adv_coeff_share"] for r in sub_adv])) - base_share))

    print("=== THE 16 CELLS ===")
    print(f"  {'arm':14s} {'kappa':>5s} {'ASR':>7s} {'rise':>8s} {'decision':>9s} "
          f"{'admission':>10s} {'d(share)':>9s}")
    for c in cells:
        print(f"  {c['d2']:14s} {c['kappa']:5.1f} {c['asr']:7.3f} {c['rise']:+8.3f} "
              f"{c['decision']:9.3f} {c['admission']:10.3f} {c['dshare']:+9.4f}")

    rise = [c["rise"] for c in cells]
    print("\n=== POOLED RANK CORRELATION OF ASR RISE WITH EACH PREDICTOR (n=16) ===")
    print("    one-sided 'greater', the direction every hypothesis here is stated in\n")
    for name, key in (("decision change  (Round 11's frozen predictor)", "decision"),
                      ("admission change (the refinement)", "admission"),
                      ("change in adversarial coefficient share", "dshare")):
        r, p = spearman([c[key] for c in cells], rise)
        print(f"  {name:46s} rho_s = {r:+.3f}   p = {p:.4f}")
    print("\n  The first line must reproduce Round 11's +0.351 / 0.091. It is printed here as a")
    print("  check on the replay, NOT as a re-test: that rule is closed and was reported refuted.")

    print("\n=== TWO-CHANNEL MODEL (rise ~ admission change + change in adversarial share) ===")
    if sps is not None:
        X = np.column_stack([np.ones(len(cells)),
                             [c["admission"] for c in cells], [c["dshare"] for c in cells]])
        y = np.asarray(rise, float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        print(f"  rise = {beta[0]:+.3f} {beta[1]:+.3f}*admission {beta[2]:+.3f}*d(share)"
              f"   R^2 = {1 - ss_res / max(ss_tot, 1e-12):.3f}")
        Xd = np.column_stack([np.ones(len(cells)), [c["decision"] for c in cells],
                              [c["dshare"] for c in cells]])
        bd, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pr = Xd @ bd
        r2d = 1 - float(((y - pr) ** 2).sum()) / max(ss_tot, 1e-12)
        print(f"  same model with DECISION change instead of admission: R^2 = {r2d:.3f}")
        print("\n  Both coefficients are expected positive if the reading is right: admitting more")
        print("  adversarial input raises ASR, and losing adversarial coefficient mass lowers it.")
        print("  A negative admission coefficient would refute the refinement on this data.")

    print("\n=== PER-ARM READING ===")
    for a in ladder["arms"]:
        d2 = a["d2"]
        sub = [c for c in cells if c["d2"] == d2]
        print(f"  {d2:14s} ASR " + " ".join(f"{c['asr']:.3f}" for c in sub)
              + "  | decision " + " ".join(f"{c['decision']:.3f}" for c in sub)
              + "  | admission " + " ".join(f"{c['admission']:.3f}" for c in sub))
    print("\n  POST-HOC AND EXPLORATORY. These predictors were measured after the Round-11 outcome")
    print("  was known. They are frozen as competing predictions for the targeted-dose suite in")
    print("  experiments/pre_registration_targeted_dose.md, which is where they get a real test.")


if __name__ == "__main__":
    main()
