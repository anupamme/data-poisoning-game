"""
Scoring of the targeted-dose suite (results/targeted_dose/summary.json) against the rules frozen in
experiments/pre_registration_targeted_dose.md.

Two instruments, each closing one of the two channels the Round-11 ladder confounded:

  MODE S  doseS_kappa<K>   adversary pinned at c = 1.0 exactly; benign spread over rho = exp(2K).
                           Adversarial coefficient share measured constant at 0.266667 across rungs
                           (spread 3.5e-07), so a rise in ASR here cannot be attenuation.
  MODE A  doseA_nu<V>      benign uniform; adversary-to-benign ratio exactly exp(V); mean(c) = 1.
                           Adversarial share sweeps 0.049 -> 0.710.

and two competing readings of C2, frozen against each other before the run:

  H-statistic   what matters is disturbance of d2's statistic, read off its decisions
  H-admission   what matters is whether the transform changes the ADVERSARIAL input d2 admits

They predict OPPOSITE shapes on mode S's krum arm, whose measured decision change is 0.533/0.800/
0.733 while its measured admission change is exactly 0.000 at every rung. That arm adjudicates.

Frozen rules, transcribed from the pre-registration and NOT adjusted here:

  S/krum        JT increasing p<0.05                      -> H-statistic confirmed, H-admission refuted
                |mean(k=2) - mean(k=0)| < 0.15 AND every rung < 0.5
                                                          -> H-admission confirmed, H-statistic refuted
                neither                                   -> indeterminate
  S/reputation  mode-S rise at k=2 significantly SMALLER than Round 11's (one-sided Welch, p<0.05)
                                                          -> H-admission confirmed
                not smaller AND JT increasing p<0.05      -> H-statistic confirmed
                neither                                   -> indeterminate
  S/cos_krum    equivalence (n=8)                         -> confirmed (both readings agree)
                JT increasing p<0.05 OR any rung >= 0.5   -> refuted (refutes BOTH readings and
                                                             Proposition 1's practical content)
  A/both        JT decreasing p<0.05 on {0,+1,+2} AND on {0,-1,-2} -> confirmed (two regimes)
                JT increasing p<0.05 on EITHER side               -> refuted
                neither                                          -> indeterminate

  GATE          any cell with mean clean accuracy < 0.35 is uninterpretable and is flagged wherever
                it feeds a verdict.

Non-negotiable 3: every number is recomputed per seed from results/targeted_dose/summary.json,
results/admission_measurement.json and results/dose_response/summary.json. Nothing is transcribed.
Non-negotiable 5: nothing in Round 11 is rescored here; it is read only as the comparison baseline
the reputation rule names.

Run: python3 experiments/analyze_targeted_dose.py
"""
import json, os, sys
import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.analyze_dose_response import ci, jonckheere       # single-sourced, not re-derived

TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
LADDER1 = os.path.join(base, "results", "dose_response", "summary.json")
ADM = os.path.join(base, "results", "admission_measurement.json")

ACC_FLOOR = 0.35
EQUIV_MARGIN = 0.15
ALPHA = 0.05
KAPPAS = [0.0, 0.5, 1.0, 2.0]
NUS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def rungs_of(cells, mode, d2, vals):
    """The arm's rungs in frozen order, each with its per-seed values and a Student-t interval."""
    out = []
    for v in vals:
        c = next((c for c in cells.values()
                  if c["mode"] == mode and c["d2"] == d2 and abs(c["rung"] - v) < 1e-12), None)
        if c is None:
            return None
        rows = sorted(c["per_seed"], key=lambda r: r["seed"])
        asrs = [r["asr"] for r in rows]; accs = [r["accuracy"] for r in rows]
        m, lo, hi = ci(asrs)
        out.append(dict(rung=v, asrs=asrs, accs=accs, seeds=[r["seed"] for r in rows],
                        mean=m, lo=lo, hi=hi, acc=float(np.mean(accs)),
                        gated=float(np.mean(accs)) < ACC_FLOOR,
                        imported=any("source" in r for r in rows)))
    return out


def welch_less(x, y):
    """One-sided Welch: is mean(x) < mean(y)?"""
    if sps is None:
        return float("nan"), float("nan")
    t, p = sps.ttest_ind(x, y, equal_var=False, alternative="less")
    return float(t), float(p)


def show(name, rungs):
    print(f"  {name:24s} ASR " + "  ".join(f"{r['mean']:.3f}[{r['lo']:+.2f},{r['hi']:+.2f}]"
                                           + ("*" if r["gated"] else " ") for r in rungs))
    print(f"  {'':24s} acc " + "  ".join(f"{r['acc']:.3f}" + " " * 12 for r in rungs))


def main():
    for p in (TARGETED, ADM):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run experiments/run_targeted_dose.py first")
    tg = json.load(open(TARGETED))
    cells = tg["cells"]
    adm = json.load(open(ADM))
    l1 = json.load(open(LADDER1))["cells"] if os.path.exists(LADDER1) else {}
    share = adm["adv_coeff_share"]

    print("=== PROVENANCE ===")
    print(f"  rules frozen at commit {tg['prereg_commit']} "
          "(experiments/pre_registration_targeted_dose.md)")
    n_new = sum(len([r for r in c["per_seed"] if "source" not in r]) for c in cells.values())
    n_imp = sum(len([r for r in c["per_seed"] if "source" in r]) for c in cells.values())
    print(f"  {n_new} runs in this suite, {n_imp} identity runs imported from Round 11 "
          "(same computation, verified by --harness-check)")
    sS = [share[f"doseS|{k}"] for k in KAPPAS]
    sA = [share[f"doseA|{v}"] for v in NUS]
    print("  mode S adversarial coefficient share: " + " ".join(f"{x:.6f}" for x in sS)
          + f"   spread {max(sS)-min(sS):.2e}   <- the assertion the C2 attribution rests on")
    print("  mode A adversarial coefficient share: " + " ".join(f"{x:.4f}" for x in sA)
          + f"   spread {max(sA)-min(sA):.4f}   <- the payload channel, isolated\n")

    verdicts = {}

    print("=== MODE S: statistic-only. Adversary pinned at c = 1.0 at every rung. ===")
    print("    '*' = below the 0.35 accuracy gate: uninterpretable, not suppression.\n")
    for d2 in ("krum", "reputation", "cos_krum"):
        r = rungs_of(cells, "S", d2, KAPPAS)
        if r is None:
            print(f"  {d2}: INCOMPLETE\n"); continue
        show(f"{d2} (n={len(r[0]['asrs'])})", r)
        _, z, p_inc, p_dec, p_perm = jonckheere([x["asrs"] for x in r])
        delta = abs(r[-1]["mean"] - r[0]["mean"])
        flat = delta < EQUIV_MARGIN and all(x["mean"] < 0.5 for x in r)
        print(f"  {'':24s} measured decision  "
              + " ".join(f"{adm['summary'][f'doseS|{d2}|{k}|decision']:.3f}" for k in KAPPAS))
        print(f"  {'':24s} measured admission "
              + " ".join(f"{adm['summary'][f'doseS|{d2}|{k}|admission']:.3f}" for k in KAPPAS))
        print(f"  {'':24s} JT z={z:+.3f} p_inc={p_inc:.4g} p_dec={p_dec:.4g} perm={p_perm:.4g}"
              f"   |Delta(k=2 vs 0)|={delta:.3f}")

        if d2 == "krum":
            v = ("H-STATISTIC CONFIRMED, H-admission REFUTED" if p_inc < ALPHA else
                 "H-ADMISSION CONFIRMED, H-statistic REFUTED" if flat else
                 "INDETERMINATE (neither frozen criterion met)")
        elif d2 == "reputation":
            c0 = next((c for c in l1.values() if c["d2"] == "reputation" and c["kappa"] == 0.0), None)
            c2 = next((c for c in l1.values() if c["d2"] == "reputation" and c["kappa"] == 2.0), None)
            if c0 and c2:
                b0 = [x["asr"] for x in c0["per_seed"]]; b2 = [x["asr"] for x in c2["per_seed"]]
                rise_s = [a - float(np.mean(r[0]["asrs"])) for a in r[-1]["asrs"]]
                rise_1 = [a - float(np.mean(b0)) for a in b2]
                t, p_less = welch_less(rise_s, rise_1)
                print(f"  {'':24s} rise(k=2): mode S {np.mean(rise_s):+.3f} vs Round 11 "
                      f"{np.mean(rise_1):+.3f}   Welch one-sided smaller: t={t:+.3f} p={p_less:.4g}")
                v = ("H-ADMISSION CONFIRMED (rise significantly smaller than Round 11's)"
                     if p_less < ALPHA else
                     "H-STATISTIC CONFIRMED (rise not smaller, and increasing)"
                     if p_inc < ALPHA else "INDETERMINATE (neither frozen criterion met)")
            else:
                v = "INDETERMINATE (Round-11 comparison unavailable)"
        else:
            v = ("REFUTED -- refutes BOTH readings and Proposition 1's practical content"
                 if (p_inc < ALPHA or any(x["mean"] >= 0.5 for x in r)) else
                 "CONFIRMED (flat: preservation survives with the magnitude channel closed)"
                 if flat else "INDETERMINATE (neither frozen criterion met)")
        gated = [f"k={x['rung']:g}" for x in r if x["gated"]]
        print(f"  {'':24s} ==> {v}" + (f"   [GATED: {','.join(gated)}]" if gated else "") + "\n")
        verdicts[f"S/{d2}"] = v

    print("=== MODE A: payload-only. Benign uniform, adversary ratio gamma = exp(nu). ===")
    print("    Frozen prediction: ASR single-peaked in gamma, maximal at or adjacent to nu = 0.\n")
    for d2 in ("reputation", "coord_median"):
        r = rungs_of(cells, "A", d2, NUS)
        if r is None:
            print(f"  {d2}: INCOMPLETE\n"); continue
        show(f"{d2} (n={len(r[0]['asrs'])})", r)
        mid = NUS.index(0.0)
        _, z_u, pi_u, pd_u, _ = jonckheere([x["asrs"] for x in r[mid:]])
        _, z_d, pi_d, pd_d, _ = jonckheere([x["asrs"] for x in r[:mid + 1][::-1]])
        peak = max(range(len(r)), key=lambda i: r[i]["mean"])
        print(f"  {'':24s} amplified  (nu 0,+1,+2): z={z_u:+.3f} p_dec={pd_u:.4g} p_inc={pi_u:.4g}")
        print(f"  {'':24s} attenuated (nu 0,-1,-2): z={z_d:+.3f} p_dec={pd_d:.4g} p_inc={pi_d:.4g}")
        print(f"  {'':24s} arg-max rung: nu = {r[peak]['rung']:+.1f} (ASR {r[peak]['mean']:.3f})")
        v = ("REFUTED (ASR rises away from the identity on at least one side)"
             if (pi_u < ALPHA or pi_d < ALPHA) else
             "CONFIRMED (single-peaked: both mechanism-preserving regimes visible)"
             if (pd_u < ALPHA and pd_d < ALPHA) else
             "INDETERMINATE (neither frozen criterion met)")
        gated = [f"nu={x['rung']:+g}" for x in r if x["gated"]]
        print(f"  {'':24s} ==> {v}" + (f"   [GATED: {','.join(gated)}]" if gated else "") + "\n")
        verdicts[f"A/{d2}"] = v

    print("=== POOLED SECONDARY (frozen): rise ~ admission change + change in adversarial share ===")
    pts = []
    for mode, arms, vals, fam in (("S", ("krum", "reputation", "cos_krum"), KAPPAS, "doseS"),
                                  ("A", ("reputation", "coord_median"), NUS, "doseA")):
        for d2 in arms:
            r = rungs_of(cells, mode, d2, vals)
            if r is None:
                continue
            zero = next(x for x in r if x["rung"] == 0.0)
            for x in r:
                pts.append(dict(rise=x["mean"] - zero["mean"],
                                admission=adm["summary"][f"{fam}|{d2}|{x['rung']}|admission"],
                                dshare=share[f"{fam}|{x['rung']}"] - share[f"{fam}|0.0"]))
    if pts and sps is not None:
        X = np.column_stack([np.ones(len(pts)), [p["admission"] for p in pts],
                             [p["dshare"] for p in pts]])
        y = np.array([p["rise"] for p in pts])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(len(pts) - X.shape[1], 1)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * float(resid @ resid) / dof)
        tvals = beta / np.maximum(se, 1e-12)
        p_adm = float(sps.t.sf(tvals[1], dof))
        r2 = 1 - float(resid @ resid) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
        print(f"  n={len(pts)} cells   rise = {beta[0]:+.3f} {beta[1]:+.3f}*admission "
              f"{beta[2]:+.3f}*d(share)   R^2 = {r2:.3f}")
        print(f"  admission coefficient t = {tvals[1]:+.3f}, one-sided p = {p_adm:.4g}")
        ok = bool(beta[1] > 0 and beta[2] > 0 and p_adm < ALPHA)
        print(f"  ==> two-channel reading {'CONFIRMED' if ok else 'NOT CONFIRMED'} "
              "(needs both coefficients positive and the admission coefficient significant)")
        verdicts["pooled/two-channel"] = "CONFIRMED" if ok else "NOT CONFIRMED"

    print("\n=== SCORECARD ===")
    for k, v in verdicts.items():
        print(f"  {k:22s} {v}")
    print("\n  Round 11 is not rescored: its pooled refutation (rho_s = +0.351, p = 0.091), its two")
    print("  refuted arms and its indeterminate arm stand exactly as reported.")


if __name__ == "__main__":
    main()
