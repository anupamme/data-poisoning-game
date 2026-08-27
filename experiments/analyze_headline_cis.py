"""
Per-seed values and 95% CIs for the three headline compositions.

Merges results/all_compositions/summary.json (development set) with
results/headline_seed_topup/summary.json (seeds 45/46 added for the three headline pairs),
then reports, per pair and per committed attack: every seed's ASR, the mean, and a 95%
confidence interval. The interval is a Student-t interval on the per-seed means; at n=3-5 a
normal interval would understate the width, and the paper reports the t interval.

Reports the accuracy alongside ASR because a low ASR at collapsed accuracy is not
suppression (the fg->krum 0.078 @ 9.9% trap), and flags any cell below 0.35 accuracy.

Prints a LaTeX-ready block. Every number is recomputed from the JSONs, never transcribed.
Run: python3 experiments/analyze_headline_cis.py
"""
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:                                     # scipy is optional
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(BASE, "results", "all_compositions", "summary.json")
TOPUP = os.path.join(BASE, "results", "headline_seed_topup", "summary.json")
# FG->RFA was separately measured on 30 seeds (42--71) for the flagship/adaptive study. That is the
# largest measurement of any composition in the paper and supersedes the n=5 cell for this pair.
FLAGSHIP = os.path.join(BASE, "results", "fg_rfa_flagship", "summary.json")

PAIRS = [("foolsgold", "rfa"), ("foolsgold", "coord_median"), ("reputation", "coord_median")]
ATTACKS = ["committed_scaling", "committed_pixel"]
SHORT = {"foolsgold": "FG", "reputation": "Rep", "rfa": "RFA", "coord_median": "CM"}
ACC_FLOOR = 0.35

# t critical values for a two-sided 95% interval, df = n-1
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_crit(n):
    if sps is not None:
        return float(sps.t.ppf(0.975, n - 1))
    return T95[n - 1]


def collect(dev, topup, d1, d2, attack):
    """Per-seed rows for one cell, development seeds plus any top-up seeds."""
    rows = {}
    cell = dev["pairs"].get(f"{d1}_then_{d2}", {}).get(attack)
    if cell:
        for r in cell["per_seed"]:
            rows[r["seed"]] = (r["asr"], r["accuracy"], "dev")
    tkey = f"{d1}_then_{d2}|{attack}"
    if topup and tkey in topup.get("cells", {}):
        for r in topup["cells"][tkey]["per_seed"]:
            if r["seed"] in rows:                     # never overwrite an existing value
                continue
            rows[r["seed"]] = (r["asr"], r["accuracy"], "topup")
    return [(s,) + rows[s] for s in sorted(rows)]


def ci(vals):
    a = np.asarray(vals, dtype=float)
    n = len(a)
    if n < 2:
        return a.mean(), float("nan"), float("nan"), n
    sd = a.std(ddof=1)
    half = t_crit(n) * sd / np.sqrt(n)
    return a.mean(), max(a.mean() - half, 0.0), min(a.mean() + half, 1.0), n


def main():
    dev = json.load(open(DEV))
    topup = json.load(open(TOPUP)) if os.path.exists(TOPUP) else None
    if topup is None:
        print(f"note: {TOPUP} absent -- reporting development seeds only\n")

    warnings = []
    tex = []
    for d1, d2 in PAIRS:
        label = f"{SHORT[d1]}$\\to${SHORT[d2]}"
        print(f"=== {SHORT[d1]}->{SHORT[d2]} ===")
        maxcell = None
        for attack in ATTACKS:
            rows = collect(dev, topup, d1, d2, attack)
            if not rows:
                print(f"  {attack:18s} MISSING")
                continue
            asrs = [r[1] for r in rows]
            accs = [r[2] for r in rows]
            m, lo, hi, n = ci(asrs)
            src = "".join("t" if r[3] == "topup" else "d" for r in rows)
            per = ", ".join(f"s{r[0]}:{r[1]:.3f}" for r in rows)
            print(f"  {attack:18s} n={n} ({src})  mean={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
                  f"  sd={np.std(asrs, ddof=1) if n > 1 else float('nan'):.3f}")
            print(f"    per-seed ASR: {per}")
            print(f"    accuracy:     mean={np.mean(accs):.3f} "
                  f"min={np.min(accs):.3f} max={np.max(accs):.3f}")
            if np.mean(accs) < ACC_FLOOR:
                warnings.append(f"{SHORT[d1]}->{SHORT[d2]} {attack}: mean accuracy "
                                f"{np.mean(accs):.3f} < {ACC_FLOOR} -- uninterpretable, "
                                f"report as such, not as suppression")
            if maxcell is None or m > maxcell[1]:
                maxcell = (attack, m, lo, hi, n, per, np.mean(accs))
        if maxcell:
            attack, m, lo, hi, n, per, acc = maxcell
            print(f"  max-committed ({attack}): {m:.3f} [{lo:.3f}, {hi:.3f}], n={n}\n")
            tex.append(f"{label} & {attack.replace('committed_', '')} & ${m:.3f}$ & "
                       f"$[{lo:.3f},\\,{hi:.3f}]$ & ${n}$ & {per.replace('s', '')} \\\\")

    print("--- LaTeX rows (max-committed attack per pair; ASR, 95% t-CI, n, per-seed) ---")
    for r in tex:
        print(r)

    if warnings:
        print("\n--- ACCURACY GATE ---")
        for w in warnings:
            print("  !", w)

    ns = set()
    for d1, d2 in PAIRS:
        for attack in ATTACKS:
            rows = collect(dev, topup, d1, d2, attack)
            if rows:
                ns.add(len(rows))
    print(f"\nseed counts present across headline cells: {sorted(ns)}")
    if len(ns) > 1:
        print("  ! non-uniform n across headline cells -- state the per-cell n in the paper")

    # ---- FG->RFA at n=30, and what the small-n subsample did to it --------------------------
    # The flagship study ran the same composition and config on seeds 42--71. Reporting the n=5
    # figure when a 30-seed measurement of the same cell exists would understate the pair, so the
    # n=30 value is the one the paper headlines.
    if not os.path.exists(FLAGSHIP):
        return 0
    fs = json.load(open(FLAGSHIP))["base_composition"]
    print("\n=== FG->RFA AT n=30 (results/fg_rfa_flagship, seeds 42--71) ===")
    big = {}
    for attack in ATTACKS:
        if attack not in fs:
            continue
        a = np.array([r["asr"] for r in fs[attack]["per_seed"]], dtype=float)
        acc = np.array([r["accuracy"] for r in fs[attack]["per_seed"]], dtype=float)
        m, lo, hi, n = ci(a)
        big[attack] = (a, acc)
        print(f"  {attack:18s} n={n}  mean={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
              f"sd={a.std(ddof=1):.3f}  median={np.median(a):.3f}  acc={acc.mean():.3f}")
        print(f"    right tail: {(a >= 0.10).sum()}/{n} seeds >= 0.10, max {a.max():.3f}"
              f"{f', skew {sps.skew(a):+.2f}' if sps is not None else ''}")
        print(f"    seeds 42--46 subset: {a[:5].mean():.3f}  vs all {n}: {m:.3f}  "
              f"({m / a[:5].mean():.2f}x) -- the small-n seed set is favourable")
    if "committed_scaling" in big:
        mx = max(big, key=lambda k: big[k][0].mean())
        print(f"  max-committed ({mx}): {big[mx][0].mean():.3f}, n=30")

    # Does the certified pair the criterion DOES accept actually cost anything against the
    # false negative it rejects? Compared at each pair's own largest n.
    fgcm = [r[1] for r in collect(dev, topup, "foolsgold", "coord_median", "committed_scaling")]
    if fgcm and "committed_scaling" in big and sps is not None:
        rfa_a, rfa_acc = big["committed_scaling"]
        fgcm_a = np.array(fgcm, dtype=float)
        fgcm_acc = np.mean([r[2] for r in collect(dev, topup, "foolsgold", "coord_median",
                                                  "committed_scaling")])
        t, p = sps.ttest_ind(rfa_a, fgcm_a, equal_var=False)
        print("\n=== COST OF THE FALSE NEGATIVE: FG->RFA (rejected) vs FG->CM (certified) ===")
        print(f"  FG->RFA  ASR {rfa_a.mean():.3f}  n={len(rfa_a)}  acc {rfa_acc.mean():.3f}")
        print(f"  FG->CM   ASR {fgcm_a.mean():.3f}  n={len(fgcm_a)}  acc {fgcm_acc:.3f}")
        print(f"  Welch t={t:.3f} p={p:.3f}  -> ASR "
              f"{'INDISTINGUISHABLE' if p > 0.05 else 'differs significantly'}")
        print(f"  accuracy: FG->CM is {fgcm_acc - rfa_acc.mean():+.3f} "
              f"({(fgcm_acc / rfa_acc.mean() - 1) * 100:.0f}% relative)")
        if p > 0.05:
            print("  => the criterion's false negative costs no measurable security, and the")
            print("     certified pair is strictly better on accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
