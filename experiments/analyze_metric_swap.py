"""
Analysis of the metric-swap suite (results/metric_swap/summary.json).

The frozen labels are scored by the runner. This script computes the one comparison the suite
supports that the cross-defense contrast does not: **within-defense preservation**.

Comparing a composition d1->d2 against d2's OWN standalone baseline holds C1 exactly fixed,
because it is the same d2 with the same standalone effectiveness on both sides. The residual
Delta = ASR(d1->d2, a) - ASR(d2, a) therefore measures only what the upstream transform did to
d2's statistic -- which is what C2 is about. This sidesteps mechanism--effectiveness confounding
by construction, at the cost of being a post-hoc analysis of a frozen suite rather than a
pre-registered prediction. It is reported as such.

The prediction being checked is quantitative and comes from the invariance check
(results/cos_invariance_check.json), which measured, per upstream defense, how often each d2's
weight ordering / selected client actually changes under that d1's realized rescaling:

    disturbance rate 0  ->  Delta ~ 0   (statistic preserved, suppression carries over)
    disturbance rate >0 ->  Delta > 0   (statistic damaged, suppression lost)

Cells whose mean accuracy is below ACC_FLOOR on either side are flagged: a Delta computed from a
collapsed baseline is not interpretable, which is exactly the cos_krum/model-scaling case.

Student-t 95% CIs on the per-seed values; n=3 throughout the frozen suite.

--with-topup additionally merges results/metric_swap_topup/summary.json (seeds 45, 46, run
post-hoc for the six matched-contrast cells only) into the within-defense and matched-contrast
analyses, and runs the significance tests on the topped-up per-seed values. It deliberately does
NOT touch the frozen label scoring or the threshold-proximity table, which stay at n=3: the
pre-registration's second non-negotiable forbids changing the seed count of the frozen test, so
the top-up may inform the post-hoc contrast and nothing else. Cells absent from the top-up keep
their n=3 values and are marked, so a mixed-n comparison is never reported as uniform.

Run: python3 experiments/analyze_metric_swap.py [--with-topup]
"""
import json
import math
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWAP = os.path.join(BASE, "results", "metric_swap", "summary.json")
INV = os.path.join(BASE, "results", "cos_invariance_check.json")
TOPUP = os.path.join(BASE, "results", "metric_swap_topup", "summary.json")

ATTACKS = ["committed_scaling", "committed_pixel"]
D2S = ["reputation", "cos_reputation", "krum", "cos_krum"]
D1S = ["norm_clip", "rfa"]
ACC_FLOOR = 0.35
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# Standalone baselines, per attack: (ASR, accuracy). Measured before the freeze; sources are
# listed in experiments/pre_registration_metric_swap.md.
STANDALONE = {
    "reputation":     {"committed_scaling": (0.017, 0.777), "committed_pixel": (0.842, 0.774)},
    "cos_reputation": {"committed_scaling": (0.982, 0.229), "committed_pixel": (0.754, 0.795)},
    "krum":           {"committed_scaling": (0.061, 0.577), "committed_pixel": (0.583, 0.485)},
    "cos_krum":       {"committed_scaling": (0.078, 0.146), "committed_pixel": (0.300, 0.598)},
}
C2 = {"reputation": "fail", "cos_reputation": "hold", "krum": "fail", "cos_krum": "hold"}


def t_crit(n):
    return float(sps.t.ppf(0.975, n - 1)) if sps is not None else T95[n - 1]


def ci(vals):
    a = np.asarray(vals, float)
    n = len(a)
    if n < 2:
        return a.mean(), float("nan"), float("nan")
    half = t_crit(n) * a.std(ddof=1) / np.sqrt(n)
    return a.mean(), a.mean() - half, a.mean() + half


def disturbance_rates(inv):
    """Per (d1, d2): fraction of live rounds on which d2's decision actually changed."""
    key = {"reputation": "reputation_order_changed",
           "cos_reputation": "cos_reputation_order_changed",
           "krum": "krum_selection_changed",
           "cos_krum": "cos_krum_selection_changed"}
    out = {}
    for d1 in D1S:
        rows = [r for r in inv["real_per_round"] if r["d1"] == d1]
        for d2 in D2S:
            out[(d1, d2)] = (sum(bool(r[key[d2]]) for r in rows), len(rows))
    return out


def load_topup():
    """Post-hoc extra seeds for the matched-contrast cells only. Returns {cell_key: [rows]}."""
    if not os.path.exists(TOPUP):
        return {}
    return {k: v["per_seed"] for k, v in json.load(open(TOPUP))["cells"].items()}


def merged_rows(cell, extra):
    """Frozen per-seed rows plus any post-hoc rows, deduplicated by seed. Frozen values win."""
    rows = {r["seed"]: r for r in cell["per_seed"]}
    for r in extra:
        rows.setdefault(r["seed"], r)
    return [rows[s] for s in sorted(rows)]


def significance(lo_r, hi_r):
    """Tests on one matched contrast. The direction was fixed before the top-up seeds ran
    (run_metric_swap_topup.py: the higher-disturbance arm is predicted higher), so the one-sided
    tests are licensed; both sidednesses are printed so the reader can apply their own standard."""
    if sps is None:
        print("      (scipy unavailable -- no tests)")
        return
    a = np.asarray(lo_r["asrs"], float)          # undisturbed arm
    b = np.asarray(hi_r["asrs"], float)          # disturbed arm
    t, p2 = sps.ttest_ind(b, a, equal_var=False)
    print(f"      Welch  t={t:+.3f}  p_2sided={p2:.3f}  p_1sided={p2 / 2 if t > 0 else 1 - p2 / 2:.3f}"
          f"  (n={len(b)} vs {len(a)})")

    shared = sorted(set(lo_r["seeds"]) & set(hi_r["seeds"]))
    if len(shared) >= 2:
        la = dict(zip(lo_r["seeds"], lo_r["asrs"]))
        lb = dict(zip(hi_r["seeds"], hi_r["asrs"]))
        d = np.array([lb[s] - la[s] for s in shared])
        tp, pp = sps.ttest_rel([lb[s] for s in shared], [la[s] for s in shared])
        print(f"      paired t={tp:+.3f}  p_2sided={pp:.3f}  p_1sided="
              f"{pp / 2 if tp > 0 else 1 - pp / 2:.3f}  on {len(shared)} shared seeds "
              f"{shared}; per-seed deltas {', '.join(f'{v:+.3f}' for v in d)}")
    u = sps.mannwhitneyu(b, a, alternative="greater")
    floor = 1.0 / math.comb(len(a) + len(b), len(a))
    print(f"      Mann-Whitney one-sided p={u.pvalue:.3f}  "
          f"(smallest attainable at this n: {floor:.3f})")
    if p2 / 2 < 0.05 and t > 0:
        print("      => gap is significant one-sided at 0.05")
    else:
        print("      => NOT significant -- report as a null, per the pre-fixed direction")


def main():
    if not os.path.exists(SWAP):
        sys.exit(f"missing {SWAP} -- run experiments/run_metric_swap_suite.py first")
    with_topup = "--with-topup" in sys.argv
    cells = json.load(open(SWAP))["cells"]
    inv = json.load(open(INV))
    dist = disturbance_rates(inv)
    topup = load_topup() if with_topup else {}
    if with_topup:
        if not topup:
            print(f"note: --with-topup given but {TOPUP} is absent -- reporting frozen n=3 only\n")
        else:
            print("=== POST-HOC TOP-UP MERGED (seeds beyond the frozen 42--44) ===")
            print("    frozen label scoring and the threshold-proximity table below stay at n=3;")
            print("    only the within-defense and matched-contrast analyses use the extra seeds.")
            for k in sorted(topup):
                seeds = sorted(r["seed"] for r in topup[k])
                print(f"    {k:44s} + seeds {seeds}")
            print()

    print("=== WITHIN-DEFENSE PRESERVATION (C1 held exactly fixed) ===")
    print("    Delta = composition ASR - that same d2's standalone ASR.")
    print("    'disturb' = rounds on which d1's rescaling changed d2's decision "
          "(results/cos_invariance_check.json).\n")
    rows = []
    for attack in ATTACKS:
        print(f"-- {attack} --")
        print(f"  {'pair':26s} {'C2':>5} {'disturb':>8} {'standalone':>11} {'composed':>9} "
              f"{'Delta':>8} {'95% CI on Delta':>20}  acc   flag")
        for d2 in D2S:
            for d1 in D1S:
                k = f"{d1}_then_{d2}|{attack}"
                if k not in cells:
                    continue
                c = cells[k]
                per = merged_rows(c, topup.get(k, []))
                asrs = [r["asr"] for r in per]
                accs = [r["accuracy"] for r in per]
                seeds = [r["seed"] for r in per]
                s_asr, s_acc = STANDALONE[d2][attack]
                m, lo, hi = ci(asrs)
                d = m - s_asr
                nd, tot = dist[(d1, d2)]
                hollow = (s_acc < ACC_FLOOR) or (np.mean(accs) < ACC_FLOOR)
                flag = "HOLLOW BASELINE" if s_acc < ACC_FLOOR else (
                    "collapsed composition" if np.mean(accs) < ACC_FLOOR else "")
                print(f"  {d1+'->'+d2:26s} {C2[d2]:>5} {str(nd)+'/'+str(tot):>8} "
                      f"{s_asr:11.3f} {m:9.3f} {d:+8.3f} "
                      f"{'['+format(lo-s_asr,'+.3f')+', '+format(hi-s_asr,'+.3f')+']':>20} "
                      f"{np.mean(accs):.2f}  {flag}")
                rows.append(dict(attack=attack, d1=d1, d2=d2, c2=C2[d2], nd=nd, tot=tot,
                                 delta=d, lo=lo - s_asr, hi=hi - s_asr, hollow=hollow,
                                 s_asr=s_asr, s_acc=s_acc, mean=m, asrs=asrs, seeds=seeds,
                                 suppressing=(s_asr < 0.5 and s_acc >= ACC_FLOOR)))
        print()

    print("=== DOES Delta TRACK THE MEASURED DISTURBANCE RATE? ===")
    print("    interpretable cells only (no hollow standalone baseline, no collapsed composition)\n")
    zero = [r for r in rows if not r["hollow"] and r["nd"] == 0]
    pos = [r for r in rows if not r["hollow"] and r["nd"] > 0]
    for label, grp in (("disturbance 0/9", zero), ("disturbance >0/9", pos)):
        if not grp:
            continue
        ds = [r["delta"] for r in grp]
        print(f"  {label:18s} n={len(grp):2d}  Delta mean {np.mean(ds):+.3f}  "
              f"range [{min(ds):+.3f}, {max(ds):+.3f}]")
        for r in grp:
            print(f"      {r['d1']+'->'+r['d2']:26s} {r['attack'].replace('committed_',''):9s} "
                  f"{r['nd']}/{r['tot']}  Delta {r['delta']:+.3f}  "
                  f"{'(no suppression to lose)' if not r['suppressing'] else ''}")
    if zero and pos:
        print(f"\n  separation: max |Delta| at zero disturbance = {max(abs(r['delta']) for r in zero):.3f}; "
              f"min Delta at nonzero disturbance = {min(r['delta'] for r in pos):+.3f}")

    # Delta only has power where the standalone d2 actually suppresses the attack. Where the
    # baseline is already above threshold (every pixel cell for reputation/krum), there is no
    # suppression left to destroy and Delta ~ 0 regardless of what the transform did -- a ceiling
    # effect, not preservation. Restricting to genuinely-suppressing baselines is what makes the
    # comparison informative.
    print("\n=== POWERED SUBSET: standalone d_2 genuinely suppresses this attack ===")
    print(f"    (standalone ASR < 0.5 at accuracy >= {ACC_FLOOR}; the only cells where a loss of")
    print("     suppression is observable at all)\n")
    powered = [r for r in rows if r["suppressing"] and not r["hollow"]]
    pz = [r for r in powered if r["nd"] == 0]
    pp = [r for r in powered if r["nd"] > 0]
    print(f"  {'pair':26s} {'attack':9s} {'disturb':>8} {'standalone':>11} {'composed':>9} {'Delta':>8}")
    for r in sorted(powered, key=lambda r: r["nd"]):
        print(f"  {r['d1']+'->'+r['d2']:26s} {r['attack'].replace('committed_',''):9s} "
              f"{str(r['nd'])+'/'+str(r['tot']):>8} {r['s_asr']:11.3f} {r['mean']:9.3f} "
              f"{r['delta']:+8.3f}")
    if pz and pp:
        print(f"\n  undisturbed (0/9): n={len(pz)}  Delta in "
              f"[{min(r['delta'] for r in pz):+.3f}, {max(r['delta'] for r in pz):+.3f}]")
        print(f"  disturbed  (>0/9): n={len(pp)}  Delta in "
              f"[{min(r['delta'] for r in pp):+.3f}, {max(r['delta'] for r in pp):+.3f}]")
        sep = min(r["delta"] for r in pp) - max(r["delta"] for r in pz)
        print(f"  separation gap: {sep:+.3f}  "
              f"({'PERFECT' if sep > 0 else 'OVERLAPPING'} split by measured disturbance)")

    # The strongest cell of all: same d_2, same attack, same standalone baseline, two upstreams that
    # differ in measured disturbance. Nothing about d_2's effectiveness varies across the pair, so
    # mechanism--effectiveness confounding cannot account for the gap.
    print("\n=== MATCHED-d_2 CONTRASTS (C1 identical by construction) ===")
    print("    same d_2, same attack, same standalone baseline; upstreams differ only in the")
    print("    disturbance they actually inflict on d_2's statistic\n")
    for d2 in D2S:
        for attack in ATTACKS:
            grp = [r for r in rows if r["d2"] == d2 and r["attack"] == attack
                   and not r["hollow"] and r["suppressing"]]
            if len(grp) < 2 or len({r["nd"] for r in grp}) < 2:
                continue
            lo_r = min(grp, key=lambda r: r["nd"])
            hi_r = max(grp, key=lambda r: r["nd"])
            print(f"  {d2} / {attack.replace('committed_','')}  "
                  f"(standalone {lo_r['s_asr']:.3f})")
            for r in (lo_r, hi_r):
                print(f"      {r['d1']:9s} disturbance {r['nd']}/{r['tot']}  "
                      f"ASR {r['mean']:.3f}  Delta {r['delta']:+.3f}  n={len(r['asrs'])}  "
                      f"per-seed {', '.join(f's{s}:{v:.3f}' for s, v in zip(r['seeds'], r['asrs']))}")
            print(f"      -> gap {hi_r['mean'] - lo_r['mean']:+.3f} attributable to C2 alone")
            significance(lo_r, hi_r)
            print()

    print("=== EXCLUDED AS UNINTERPRETABLE ===")
    for r in rows:
        if r["hollow"]:
            print(f"  {r['d1']+'->'+r['d2']:26s} {r['attack'].replace('committed_',''):9s} "
                  f"Delta {r['delta']:+.3f}  (standalone or composed accuracy < {ACC_FLOOR})")
    print("=== EXCLUDED FOR LACK OF POWER (standalone does not suppress) ===")
    for r in rows:
        if not r["suppressing"] and not r["hollow"]:
            print(f"  {r['d1']+'->'+r['d2']:26s} {r['attack'].replace('committed_',''):9s} "
                  f"standalone {r['s_asr']:.3f} >= 0.5, Delta {r['delta']:+.3f}")

    print("\n=== THRESHOLD PROXIMITY (max-committed, for the frozen scoring) ===")
    print("    frozen seeds 42--44 ONLY; --with-topup does not enter here, because the "
          "pre-registered\n    label scoring is fixed at n=3 and must not move.\n")
    for d2 in D2S:
        for d1 in D1S:
            ks = [f"{d1}_then_{d2}|{a}" for a in ATTACKS]
            if not all(k in cells for k in ks):
                continue
            worst = max((cells[k] for k in ks), key=lambda c: c["mean_asr"])
            asrs = [r["asr"] for r in worst["per_seed"]]
            m, lo, hi = ci(asrs)
            near = "  <-- within 0.10 of the 0.5 threshold" if abs(m - 0.5) < 0.10 else ""
            print(f"  {d1+'->'+d2:26s} {worst['attack'].replace('committed_',''):9s} "
                  f"{m:.3f} [{lo:.3f}, {hi:.3f}]  per-seed "
                  f"{', '.join(f'{v:.3f}' for v in asrs)}{near}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
