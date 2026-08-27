"""
Scoring of the dose-response ladder (results/dose_response/summary.json) against the decision rules
frozen in experiments/pre_registration_dose_response.md.

The design: d2 and the attack are held fixed within every arm and only the DOSE of upstream
per-client rescaling varies (d1 = dose_kappa<K>, rho = exp(2K) in {1.00, 2.72, 7.39, 54.60}). C1 --
standalone effectiveness of d2 on that attack -- is therefore numerically the same quantity at every
rung, so mechanism--effectiveness confounding is impossible by construction rather than matched
away. That is what the testability proposition licenses and what the two cross-defense suites could
not deliver.

Every rule below is transcribed from the frozen file and NOT adjusted here. In particular:

  PRIMARY   pooled Spearman (one-sided, greater, alpha 0.05) of each cell's ASR rise over its own
            arm's kappa=0 rung against that cell's MEASURED disturbance, across all 16 cells.
            Confirmed if rho > 0 at p < 0.05; refuted otherwise, including a negative correlation.
  SECONDARY per-arm Jonckheere--Terpstra, one-sided for increasing trend, 4 ordered rungs x 5 seeds.
            arms 1, 2   monotone rise      confirmed: JT increasing p < 0.05
            arm 3       FLAT               confirmed: |mean ASR(k=2) - mean ASR(k=0)| < 0.15 AND
                                           every rung < 0.5;  refuted: JT increasing p < 0.05 OR
                                           any rung >= 0.5
            arm 4       shallower rise      confirmed: JT increasing p < 0.05 AND its k=2 rise
                                           smaller than both class-(c) arms'
  GATE      any cell with mean clean accuracy < 0.35 is uninterpretable: a low ASR there is a
            collapsed model, not suppression. Reported, and flagged wherever it feeds a verdict.

An arm that satisfies neither its confirmation nor its refutation criterion is reported as
INDETERMINATE. It is not scored whichever way flatters the framework -- non-negotiable 1 forbids
revising a rule after an outcome, and that cuts both ways.

Two further sections, both frozen in advance as checks on the instrument rather than on C2:

  ECOLOGICAL VALIDITY  the two real upstream transforms already measured on three of these same
            cells (norm_clip, rfa from results/metric_swap/) are placed on the same measured-
            disturbance axis (caveat 4). If they fall far off the synthetic curve at matched
            disturbance, that is a finding about the synthetic transform and is reported as one.
  kappa=0 CROSS-CHECK  the fresh kappa=0 rung against the published standalone baselines, per seed.
            The ladder does not use those baselines as data; this only establishes whether the
            published value is the same quantity.

Non-negotiable 3: every number here is recomputed from results/dose_response/summary.json,
results/dose_disturbance.json, results/cos_invariance_check.json and results/metric_swap*/, per
seed. Nothing is transcribed from a paper, a table, or a runner's printed output.

Run: python3 experiments/analyze_dose_response.py [--tex]
"""
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = os.path.join(BASE, "results", "dose_response", "summary.json")
DIST = os.path.join(BASE, "results", "dose_disturbance.json")
INV = os.path.join(BASE, "results", "cos_invariance_check.json")
SWAP = os.path.join(BASE, "results", "metric_swap", "summary.json")
TOPUP = os.path.join(BASE, "results", "metric_swap_topup", "summary.json")

ACC_FLOOR = 0.35            # frozen gate; also recorded in the ladder's own config, asserted below
EQUIV_MARGIN = 0.15         # frozen arm-3 equivalence margin
ALPHA = 0.05
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

# Which per-round field of results/cos_invariance_check.json records "this d1 changed this d2's
# decision". Only three of the four arms have a real-transform anchor; coord_median was not in the
# metric-swap menu, so arm 4 has none and that is stated rather than filled in with a proxy.
INV_FIELD = {"krum": "krum_selection_changed",
             "cos_krum": "cos_krum_selection_changed",
             "reputation": "reputation_order_changed"}
D1S = ["norm_clip", "rfa"]

# Imported only for the kappa=0 cross-check, and only for its per-seed lookup into the three source
# suites, so that lookup is not duplicated here and cannot drift from the runner's. Guarded because
# the runner imports torch and this analysis otherwise does not need it.
try:
    sys.path.insert(0, BASE)
    from experiments.run_dose_response import STANDALONE, published_per_seed
except Exception as _e:          # noqa: BLE001 -- any import failure degrades to skipping one section
    STANDALONE, published_per_seed, _IMPORT_ERR = None, None, _e


def t_crit(n):
    return float(sps.t.ppf(0.975, n - 1)) if sps is not None else T95[n - 1]


def ci(vals):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.mean()), float("nan"), float("nan")
    half = t_crit(len(a)) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(a.mean() - half), float(a.mean() + half)


def jt_statistic(groups):
    """Jonckheere--Terpstra J: Mann--Whitney U summed over all ordered group pairs i < j."""
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for x in groups[i]:
                for y in groups[j]:
                    J += 1.0 if y > x else (0.5 if y == x else 0.0)
    return J


def jonckheere(groups, n_perm=20000):
    """One-sided JT test for trend across groups given in their pre-registered order.

    Returns (J, z, p_increasing, p_decreasing, p_perm_increasing). The normal approximation is the
    standard one; the permutation p-value is added because n=5 per group is small, is computed with a
    fixed RNG seed so it is reproducible, and is reported alongside rather than instead of z -- the
    frozen rule names the test, not the p-value machinery, so both are shown.
    """
    gs = [np.asarray(g, float) for g in groups]
    ns = np.array([len(g) for g in gs], float)
    N = ns.sum()
    J = jt_statistic(gs)
    EJ = (N ** 2 - (ns ** 2).sum()) / 4.0
    VJ = (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72.0
    z = (J - EJ) / np.sqrt(VJ)
    if sps is not None:
        p_inc, p_dec = float(sps.norm.sf(z)), float(sps.norm.cdf(z))
    else:
        p_inc = p_dec = float("nan")
    pooled = np.concatenate(gs)
    rng = np.random.default_rng(0)
    sizes = np.cumsum([len(g) for g in gs])[:-1]
    hits = 0
    for _ in range(n_perm):
        hits += jt_statistic(np.split(rng.permutation(pooled), sizes)) >= J
    return J, float(z), p_inc, p_dec, (1 + hits) / (1 + n_perm)


def load_arms(ladder, dist):
    """One record per arm, with its four rungs in frozen kappa order. Cells are indexed by their own
    recorded (d2, attack, kappa) fields rather than by reconstructing the runner's key string."""
    kappas = ladder["config"]["kappas"]
    by_cell = {(c["d2"], c["attack"], c["kappa"]): c for c in ladder["cells"].values()}
    sd = dist["summary_disturbance"]
    arms = []
    for a in ladder["arms"]:
        d2, atk = a["d2"], a["attack"]
        rungs = []
        for k in kappas:
            c = by_cell[(d2, atk, k)]
            asrs = [r["asr"] for r in sorted(c["per_seed"], key=lambda r: r["seed"])]
            accs = [r["accuracy"] for r in sorted(c["per_seed"], key=lambda r: r["seed"])]
            seeds = [r["seed"] for r in sorted(c["per_seed"], key=lambda r: r["seed"])]
            m, lo, hi = ci(asrs)
            rungs.append(dict(kappa=k, rho=c["rho"], asrs=asrs, accs=accs, seeds=seeds,
                              mean=m, lo=lo, hi=hi, acc=float(np.mean(accs)),
                              disturb=sd[f"{d2}|{atk}|kappa{k}"],
                              gated=float(np.mean(accs)) < ACC_FLOOR))
        base = rungs[0]["mean"]
        for r in rungs:
            r["rise"] = r["mean"] - base
        arms.append(dict(d2=d2, attack=atk, cls=a["prop1_class"], shape=a["predicted_shape"],
                         published=a["standalone_published"], rungs=rungs))
    return arms


def anchors(arm, inv, swap, topup):
    """Real upstream transforms measured on this arm's own (d2, attack) cell: measured disturbance
    rate, ASR at the frozen n=3, and ASR with the post-hoc top-up seeds merged in."""
    out = []
    field = INV_FIELD.get(arm["d2"])
    if field is None:
        return out
    for d1 in D1S:
        key = f"{d1}_then_{arm['d2']}|{arm['attack']}"
        if key not in swap["cells"]:
            continue
        rows = {r["seed"]: r for r in swap["cells"][key]["per_seed"]}
        n3 = [rows[s]["asr"] for s in sorted(rows)]
        for r in topup.get(key, []):
            rows.setdefault(r["seed"], r)
        n5 = [rows[s]["asr"] for s in sorted(rows)]
        accs = [rows[s]["accuracy"] for s in sorted(rows)]
        per = [r for r in inv["real_per_round"] if r["d1"] == d1]
        nd = sum(bool(r[field]) for r in per)
        rhos = [r["rho"] for r in per]
        out.append(dict(d1=d1, nd=nd, tot=len(per), disturb=nd / len(per),
                        asr3=float(np.mean(n3)), n3=len(n3),
                        asr5=float(np.mean(n5)), n5=len(n5), acc=float(np.mean(accs)),
                        rho_mean=float(np.mean(rhos)), rho_max=float(np.max(rhos))))
    return out


def main():
    for p in (LADDER, DIST):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    ladder = json.load(open(LADDER))
    dist = json.load(open(DIST))
    inv = json.load(open(INV)) if os.path.exists(INV) else None
    swap = json.load(open(SWAP)) if os.path.exists(SWAP) else None
    topup = ({k: v["per_seed"] for k, v in json.load(open(TOPUP))["cells"].items()}
             if os.path.exists(TOPUP) else {})

    cfg = ladder["config"]
    assert cfg["acc_floor"] == ACC_FLOOR, "accuracy gate differs from the frozen 0.35"
    arms = load_arms(ladder, dist)

    print("=== PROVENANCE ===")
    print(f"  shapes frozen at commit {ladder['prereg_commit']} "
          "(experiments/pre_registration_dose_response.md)")
    print(f"  {len(arms)} arms x {len(cfg['kappas'])} rungs x {len(cfg['seeds'])} seeds = "
          f"{sum(len(r['asrs']) for a in arms for r in a['rungs'])} runs recorded")
    print(f"  kappa {cfg['kappas']}  ->  rho "
          f"{[round(cfg['rhos'][str(k)], 2) for k in cfg['kappas']]}")
    print(f"  seeds {cfg['seeds']};  N={cfg['N']} K={cfg['K']} f={cfg['f']} alpha={cfg['alpha']} "
          f"rounds={cfg['rounds']};  accuracy gate {cfg['acc_floor']}")
    print("  abscissa measured before the freeze, no ASR computed by it "
          "(results/dose_disturbance.json):")
    print(f"    dose is monotone in kappa: {dist['dose_is_monotone']}")
    print(f"    cos_krum selection invariance holds: {dist['cos_krum_invariance_holds']} "
          f"({dist['n_cos_krum_violations']} violations; max relative score drift "
          f"{dist['cos_krum_max_score_rel_diff']:.2e})")
    if sps is None:
        print("  WARNING: scipy unavailable -- Spearman and the JT normal approximation are skipped")
    print()

    print("=== PER-ARM CURVES (per-attack mean ASR, n=5, Student-t 95% CI) ===")
    print("    'disturb' is the MEASURED disturbance of this arm's statistic at this rung.")
    print("    'rise' is over this arm's own kappa=0 rung -- the primary test's ordinate.\n")
    for i, a in enumerate(arms, 1):
        print(f"-- arm {i}: {a['d2']} / {a['attack'].replace('committed_', '')}  "
              f"[Prop 1 {a['cls']}]  predicted: {a['shape']}")
        print(f"   {'kappa':>6} {'rho':>7} {'disturb':>8} {'mean ASR':>9} {'95% CI':>18} "
              f"{'rise':>8} {'acc':>6}  flag")
        for r in a["rungs"]:
            print(f"   {r['kappa']:6.1f} {r['rho']:7.2f} {r['disturb']:8.3f} {r['mean']:9.3f} "
                  f"{'[' + format(r['lo'], '.3f') + ', ' + format(r['hi'], '.3f') + ']':>18} "
                  f"{r['rise']:+8.3f} {r['acc']:6.3f}  "
                  f"{'GATED (acc < ' + str(ACC_FLOOR) + ')' if r['gated'] else ''}")
        print(f"   per-seed ASR by rung (seeds {a['rungs'][0]['seeds']}):")
        for r in a["rungs"]:
            print(f"     kappa={r['kappa']:<4} " + "  ".join(f"{v:.3f}" for v in r["asrs"]))
        print()

    print("=== PRIMARY (FROZEN): pooled Spearman, ASR rise vs measured disturbance ===")
    print("    all 16 cells; one-sided (greater), alpha 0.05.")
    print("    Confirmed if rho > 0 at p < 0.05.  Refuted otherwise, including a negative rho.\n")
    x = [r["disturb"] for a in arms for r in a["rungs"]]
    y = [r["rise"] for a in arms for r in a["rungs"]]
    print(f"   {'cell':34s} {'disturb':>8} {'rise':>8}")
    for a in arms:
        for r in a["rungs"]:
            print(f"   {a['d2'] + '/' + a['attack'].replace('committed_', '') + ' k=' + str(r['kappa']):34s} "
                  f"{r['disturb']:8.3f} {r['rise']:+8.3f}")
    primary = None
    if sps is not None:
        s = sps.spearmanr(x, y, alternative="greater")
        s2 = sps.spearmanr(x, y, alternative="less")
        primary = bool(s.statistic > 0 and s.pvalue < ALPHA)
        print(f"\n   Spearman rho = {s.statistic:+.3f}   p(greater) = {s.pvalue:.3f}   "
              f"p(less) = {s2.pvalue:.3f}   n = {len(x)}")
        print(f"   => PRIMARY {'CONFIRMED' if primary else 'REFUTED'} "
              f"({'rho > 0 at p < 0.05' if primary else 'the frozen criterion is not met'})")
        # Post-hoc sensitivity, labelled as such and NOT part of the frozen rule. The four kappa=0
        # cells sit at (0, 0) by construction rather than by measurement, so they are ties that
        # carry no dose information; dropping them is a defensible analysis that we did not
        # pre-register, so it cannot rescue the primary test and is reported only as a diagnostic.
        keep = [i for i in range(len(x)) if not (x[i] == 0 and y[i] == 0)]
        s3 = sps.spearmanr([x[i] for i in keep], [y[i] for i in keep], alternative="greater")
        print(f"   post-hoc sensitivity (NOT the frozen test, cannot change the verdict): dropping "
              f"the {len(x) - len(keep)} constructed (0, 0) identity cells gives rho = "
              f"{s3.statistic:+.3f}, p(greater) = {s3.pvalue:.3f}, n = {len(keep)}")
        # Also post-hoc: was the NOMINAL dose a better predictor than the measured flip rate? The
        # frozen predictor is the measured one; this only says which of the two the data prefers.
        kap = [r["kappa"] for a in arms for r in a["rungs"]]
        s4 = sps.spearmanr(kap, y, alternative="greater")
        print(f"   post-hoc: against the NOMINAL dose kappa (i.e. log rho) instead of measured "
              f"disturbance, rho = {s4.statistic:+.3f}, p(greater) = {s4.pvalue:.3f}, n = {len(kap)}")
    print()

    print("=== SECONDARY (FROZEN): per-arm Jonckheere--Terpstra trend ===")
    print("    one-sided for INCREASING trend across the 4 ordered rungs x 5 seeds, alpha 0.05.")
    print("    p(dec) is descriptive only -- no frozen rule tests for a decreasing trend.\n")
    for i, a in enumerate(arms, 1):
        groups = [r["asrs"] for r in a["rungs"]]
        J, z, p_inc, p_dec, p_perm = jonckheere(groups)
        a["jt"] = dict(J=J, z=z, p_inc=p_inc, p_dec=p_dec, p_perm=p_perm)
        print(f"   arm {i} {a['d2'] + '/' + a['attack'].replace('committed_', ''):26s} "
              f"J={J:6.1f}  z={z:+.3f}  p(inc)={p_inc:.3f}  p(inc, perm)={p_perm:.3f}  "
              f"p(dec)={p_dec:.3f}")
        gated = [r for r in a["rungs"] if r["gated"]]
        if gated:
            ug = [r["asrs"] for r in a["rungs"] if not r["gated"]]
            if len(ug) >= 2:
                Jg, zg, pig, pdg, ppg = jonckheere(ug)
                print(f"          gate sensitivity ({len(gated)} rung(s) below the accuracy floor "
                      f"dropped, {len(ug)} rungs left): z={zg:+.3f} p(inc)={pig:.3f} "
                      f"p(dec)={pdg:.3f}")
    print()

    print("=== ARM VERDICTS AGAINST THE FROZEN CRITERIA ===\n")
    rise_c = [a["rungs"][-1]["rise"] for a in arms if a["cls"].startswith("c")]
    verdicts = []
    for i, a in enumerate(arms, 1):
        jt = a["jt"]
        inc = jt["p_inc"] < ALPHA
        last, first = a["rungs"][-1], a["rungs"][0]
        name = f"arm {i} {a['d2']}/{a['attack'].replace('committed_', '')}"
        print(f"-- {name}  [{a['cls']}]  predicted {a['shape']}")
        if a["shape"] == "FLAT":
            gap = abs(last["mean"] - first["mean"])
            any_hi = [r for r in a["rungs"] if r["mean"] >= 0.5]
            print(f"   confirmation: |ASR(k=2.0) - ASR(k=0)| = {gap:.3f} vs margin "
                  f"{EQUIV_MARGIN}  -> {'PASS' if gap < EQUIV_MARGIN else 'FAIL'}; "
                  f"every rung < 0.5 -> {'PASS' if not any_hi else 'FAIL'} "
                  f"(max rung {max(r['mean'] for r in a['rungs']):.3f})")
            print(f"   refutation:   JT increasing at p < 0.05 -> {'YES' if inc else 'no'}; "
                  f"any rung >= 0.5 -> {'YES' if any_hi else 'no'}")
            if inc or any_hi:
                v = "REFUTED"
            elif gap < EQUIV_MARGIN and not any_hi:
                v = "CONFIRMED"
            else:
                v = "INDETERMINATE"
        elif a["shape"] == "shallower rise":
            shallower = all(last["rise"] < r for r in rise_c)
            print(f"   confirmation: JT increasing p={jt['p_inc']:.3f} -> "
                  f"{'PASS' if inc else 'FAIL'}; k=2.0 rise {last['rise']:+.3f} smaller than both "
                  f"class-(c) arms' ({', '.join(f'{r:+.3f}' for r in rise_c)}) -> "
                  f"{'PASS' if shallower else 'FAIL'}")
            print("   refutation:   no increasing trend, or a rise steeper than both class-(c) "
                  f"arms -> {'REFUTED' if not inc else 'not triggered'}")
            v = "CONFIRMED" if (inc and shallower) else "REFUTED"
        else:
            print(f"   confirmation: JT increasing p={jt['p_inc']:.3f} -> "
                  f"{'PASS' if inc else 'FAIL'}   (k=2.0 rise {last['rise']:+.3f})")
            v = "CONFIRMED" if inc else "REFUTED"
        used = [r for r in (first, last) if r["gated"]]
        if used:
            print(f"   ACCURACY GATE: the kappa={', '.join(str(r['kappa']) for r in used)} rung(s) "
                  f"feeding this verdict have mean clean accuracy "
                  f"{', '.join(format(r['acc'], '.3f') for r in used)} < {ACC_FLOOR}, so that ASR "
                  "is a collapsed model rather than suppression and the comparison is not "
                  "interpretable as stated.")
            if v == "CONFIRMED":
                v = "CONFIRMED (on a gate-failing rung -- not interpretable)"
        print(f"   => {v}\n")
        verdicts.append((name, a["shape"], v))

    print("=== CLEAN ACCURACY vs DOSE (descriptive, no frozen rule) ===")
    print("    The ladder normalizes mean(c) = 1 to hold the aggregate's scale fixed, which fixes")
    print("    the aggregate but not any one client's relative weight. If a falling ASR were bought")
    print("    by contracting the update -- the annihilation lemma's channel rather than C2's --")
    print("    clean accuracy would fall with dose too. Whether it does is a property of the")
    print("    instrument and is reported either way; JT here is two-sided in the sense that both")
    print("    directions are printed and neither was pre-registered.\n")
    for i, a in enumerate(arms, 1):
        accs = [r["accs"] for r in a["rungs"]]
        _, z, p_inc, p_dec, _ = jonckheere(accs, n_perm=2000)
        print(f"   arm {i} {a['d2'] + '/' + a['attack'].replace('committed_', ''):26s} "
              + " ".join(f"k={r['kappa']}:{r['acc']:.3f}" for r in a["rungs"])
              + f"   z={z:+.3f}  p(inc)={p_inc:.3f}  p(dec)={p_dec:.3f}")
    print()

    print("=== ACCURACY GATE: cells excluded as uninterpretable ===")
    bad = [(a, r) for a in arms for r in a["rungs"] if r["gated"]]
    if not bad:
        print(f"  none: every cell is at or above {ACC_FLOOR} mean clean accuracy")
    for a, r in bad:
        print(f"  {a['d2'] + '/' + a['attack'].replace('committed_', ''):26s} kappa={r['kappa']:<4} "
              f"acc {r['acc']:.3f} < {ACC_FLOOR}, ASR {r['mean']:.3f} "
              "-- collapsed model, not suppression")
    print()

    if inv is not None and swap is not None:
        print("=== ECOLOGICAL VALIDITY: real upstream transforms on the same measured axis ===")
        print("    caveat 4 of the pre-registration. norm_clip and rfa were measured on three of")
        print("    these four cells; their disturbance rates come from cos_invariance_check.json")
        print("    and their ASRs from metric_swap/ (frozen n=3) and metric_swap_topup/ (n=5).")
        print("    A large residual against the synthetic rung at matched disturbance is a finding")
        print("    about the instrument, and is reported as one.\n")
        for i, a in enumerate(arms, 1):
            anc = anchors(a, inv, swap, topup)
            label = f"arm {i} {a['d2']}/{a['attack'].replace('committed_', '')}"
            if not anc:
                print(f"  {label}: NO real-transform anchor -- {a['d2']} was not in the "
                      "metric-swap menu, so this arm's ecological validity is untested.")
                continue
            print(f"  {label}   synthetic curve: " + ", ".join(
                f"d={r['disturb']:.3f}->{r['mean']:.3f}" for r in a["rungs"]))
            for r in anc:
                near = min(a["rungs"], key=lambda q: abs(q["disturb"] - r["disturb"]))
                print(f"    {r['d1']:9s} disturb {r['nd']}/{r['tot']} = {r['disturb']:.3f}  "
                      f"ASR {r['asr3']:.3f} (n={r['n3']}) / {r['asr5']:.3f} (n={r['n5']})  "
                      f"acc {r['acc']:.2f}  realized rho mean {r['rho_mean']:.2f} "
                      f"max {r['rho_max']:.2f}")
                print(f"              nearest synthetic rung kappa={near['kappa']} "
                      f"(disturb {near['disturb']:.3f}, ASR {near['mean']:.3f})  "
                      f"residual {r['asr5'] - near['mean']:+.3f}")
        print()

    if published_per_seed is not None:
        print("=== kappa=0 CROSS-CHECK vs PUBLISHED STANDALONE (per seed) ===")
        print("    the ladder's kappa=0 rung is measured fresh and is the baseline used everywhere")
        print("    above; this only asks whether the published value is the same quantity.\n")
        worst = 0.0
        for i, a in enumerate(arms, 1):
            pub = published_per_seed(a["d2"], a["attack"])
            r0 = a["rungs"][0]
            shared = [s for s in r0["seeds"] if s in pub]
            src = STANDALONE[(a["d2"], a["attack"])][3]
            if not shared:
                print(f"  arm {i} {a['d2']:13s}: no shared seed with {src} -- cannot check")
                continue
            ds = []
            for s in shared:
                mine = dict(zip(r0["seeds"], zip(r0["accs"], r0["asrs"])))[s]
                ds.append((s, mine[0] - pub[s][0], mine[1] - pub[s][1]))
            worst = max([worst] + [max(abs(d[1]), abs(d[2])) for d in ds])
            print(f"  arm {i} {a['d2']:13s} vs {src}")
            print("            " + "  ".join(
                f"s{s}: dACC={da:+.4f} dASR={dr:+.4f}" for s, da, dr in ds))
        print(f"\n  largest absolute per-seed deviation: {worst:.2e}")
        print("  " + ("kappa=0 is the exact identity on every shared seed, so the published "
                      "baselines are the same quantity." if worst <= 1e-6 else
                      "NOT identical -- the published baseline is not the same quantity and must "
                      "not be cited as though it were."))
        print()
    else:
        print("=== kappa=0 CROSS-CHECK SKIPPED ===")
        print(f"  could not import experiments.run_dose_response ({_IMPORT_ERR})\n")

    print("=== SCORECARD ===")
    print(f"  PRIMARY (pooled Spearman, 16 cells): "
          f"{'CONFIRMED' if primary else 'REFUTED' if primary is not None else 'not computed'}")
    for name, shape, v in verdicts:
        print(f"  {name:34s} predicted {shape:16s} {v}")
    n_conf = sum(1 for _, _, v in verdicts if v.startswith("CONFIRMED"))
    n_ref = sum(1 for _, _, v in verdicts if v == "REFUTED")
    n_ind = len(verdicts) - n_conf - n_ref
    print(f"\n  {n_conf} confirmed, {n_ref} refuted, {n_ind} indeterminate of {len(verdicts)} arms.")
    print("  Per non-negotiable 1, no rule above was revised after the outcome; a refutation is")
    print("  reported as a refutation and an arm meeting neither criterion is reported as")
    print("  indeterminate rather than scored in the framework's favour.")

    if "--tex" in sys.argv:
        print("\n=== LaTeX rows (mean ASR with 95% CI per rung; recomputed above) ===")
        for a in arms:
            cells = " & ".join(
                f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
                + (r"$^{\dagger}$" if r["gated"] else "") for r in a["rungs"])
            print(f"\\texttt{{{a['d2'].replace('_', chr(92) + '_')}}} / "
                  f"{a['attack'].replace('committed_', '')} & {a['cls'][:1]} & "
                  f"{a['shape']} & {cells} & "
                  f"{next(v for n, _, v in verdicts if n.endswith(a['d2'] + '/' + a['attack'].replace('committed_', '')))} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
