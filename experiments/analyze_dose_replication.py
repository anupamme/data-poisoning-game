"""
Score the rank-discordant replication arm: coord_median / committed_pixel under Mode S.

WHY THIS EXISTS. The flagship negative -- a downstream statistic can be disturbed heavily without
losing suppression -- rests on one downstream defense (krum, a selector, Prop-1 class (c)). The
positive half of the claim -- that what matters is the adversarial mass admitted -- was never ordered
prospectively, because the two existing Mode-S arms move together on both predictors. This arm is
rank-discordant: it has the SMALLEST decision change (0.482) and the LARGEST admission change (0.033)
of any interpretable Mode-S arm, so the decision reading and the admission reading predict opposite
rank positions for its rise. The rules were frozen in
experiments/pre_registration_dose_replication.md at commit 06d9bce, before results/dose_replication/
existed.

WHAT IS SCORED, all thresholds frozen before the run:
  primary    Delta = mean ASR(kappa=2) - mean ASR(kappa=0)
             Delta > +0.178          -> admission ordering CONFIRMED  (beats reputation's rise)
             Delta < +0.150          -> admission ordering REFUTED, and the flagship negative
                                        REPLICATES on a structurally different defense
             +0.150 <= Delta <= +0.178 -> INDETERMINATE, not scored in our favour
  secondary  Jonckheere-Terpstra across kappa in {0, 0.5, 1, 2}; both readings predict a rise here,
             so a flat or falling curve refutes both
  tertiary   the pooled two-channel model, re-fit with this arm's rungs added to the existing 22 cells
  gate       any cell with mean clean accuracy < 0.35 is uninterpretable and flagged

NOTHING IS RE-DERIVED HERE. The rung builder, the interval, the trend test, the Welch test and the
pooled design matrix are imported from the existing analysis scripts so this arm is scored by the same
code as the arms it is compared against. The two comparison rises (-0.026 krum, +0.178 reputation) are
RECOMPUTED per seed from results/targeted_dose/summary.json, not transcribed from the paper.

The confounded companion ladder on this exact cell is recomputed from results/dose_response/ and
printed alongside: it FALLS (0.443 -> 0.170), which is the contrast Mode S exists to explain.

Run: python3 experiments/analyze_dose_replication.py
"""
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except Exception:
    sps = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)
# single-sourced: same rung builder, same interval, same trend test, same Welch as the frozen suite
from experiments.analyze_targeted_dose import (  # noqa: E402
    ACC_FLOOR, ALPHA, KAPPAS, NUS, rungs_of, show,
)
from experiments.analyze_dose_response import jonckheere  # noqa: E402

REPL = os.path.join(base, "results", "dose_replication", "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
LADDER1 = os.path.join(base, "results", "dose_response", "summary.json")
ADM = os.path.join(base, "results", "admission_measurement.json")

D2 = "coord_median"
ATTACK = "committed_pixel"
OUT = os.path.join(base, "results", "dose_replication", "scored.json")


def companion_ladder():
    """The confounded Round-11 ladder on this exact cell, recomputed per seed."""
    if not os.path.exists(LADDER1):
        return None
    cells = json.load(open(LADDER1))["cells"]
    out = []
    for k in KAPPAS:
        c = next((c for c in cells.values() if c["d2"] == D2 and c["attack"] == ATTACK
                  and abs(c["kappa"] - k) < 1e-12), None)
        if c is None:
            return None
        asrs = [r["asr"] for r in c["per_seed"]]
        accs = [r["accuracy"] for r in c["per_seed"]]
        out.append(dict(rung=k, mean=float(np.mean(asrs)), acc=float(np.mean(accs)), asrs=asrs))
    return out


def comparison_rises(cells):
    """Recompute the two published Mode-S rises per seed. Not transcribed."""
    out = {}
    for d2 in ("krum", "reputation"):
        r = rungs_of(cells, "S", d2, KAPPAS)
        if r is None:
            continue
        out[d2] = dict(rise=float(np.mean(r[-1]["asrs"])) - float(np.mean(r[0]["asrs"])),
                       identity=float(np.mean(r[0]["asrs"])),
                       top=float(np.mean(r[-1]["asrs"])),
                       rise_seeds=[a - float(np.mean(r[0]["asrs"])) for a in r[-1]["asrs"]])
    return out


def pooled_fit(pts):
    """The frozen two-channel model: rise ~ admission change + change in adversarial share."""
    X = np.column_stack([np.ones(len(pts)), [p["admission"] for p in pts],
                         [p["dshare"] for p in pts]])
    y = np.array([p["rise"] for p in pts])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(pts) - X.shape[1], 1)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * float(resid @ resid) / dof)
    tvals = beta / np.maximum(se, 1e-12)
    p_adm = float(sps.t.sf(tvals[1], dof)) if sps is not None else float("nan")
    r2 = 1 - float(resid @ resid) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
    return dict(n=len(pts), beta=[float(b) for b in beta], t_admission=float(tvals[1]),
                p_admission=p_adm, r2=float(r2),
                confirmed=bool(beta[1] > 0 and beta[2] > 0 and p_adm < ALPHA))


def pooled_points(cells_t, cells_r, adm, share):
    """Existing 22 cells, then this arm's rungs, built by the same rule as analyze_targeted_dose."""
    pts = []
    spec = [("S", ("krum", "reputation", "cos_krum"), KAPPAS, "doseS", cells_t),
            ("A", ("reputation", "coord_median"), NUS, "doseA", cells_t)]
    if cells_r is not None:
        spec.append(("S", (D2,), KAPPAS, "doseS", cells_r))
    n_existing = None
    for mode, arms, vals, fam, cells in spec:
        for d2 in arms:
            r = rungs_of(cells, mode, d2, vals)
            if r is None:
                continue
            zero = next(x for x in r if x["rung"] == 0.0)
            for x in r:
                pts.append(dict(rise=x["mean"] - zero["mean"],
                                admission=adm["summary"][f"{fam}|{d2}|{x['rung']}|admission"],
                                dshare=share[f"{fam}|{x['rung']}"] - share[f"{fam}|0.0"]))
        if mode == "A":
            n_existing = len(pts)
    return pts, n_existing


def main():
    for p in (REPL, TARGETED, ADM):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run experiments/run_dose_replication.py first")
    rep = json.load(open(REPL))
    cells_r = rep["cells"]
    tg = json.load(open(TARGETED))
    cells_t = tg["cells"]
    adm = json.load(open(ADM))
    share = adm["adv_coeff_share"]
    cfg = rep["config"]

    print("=== PROVENANCE ===")
    print(f"  rules frozen at commit {rep['prereg_commit']} "
          "(experiments/pre_registration_dose_replication.md)")
    n_new = sum(len([r for r in c["per_seed"] if "source" not in r]) for c in cells_r.values())
    n_imp = sum(len([r for r in c["per_seed"] if "source" in r]) for c in cells_r.values())
    print(f"  {n_new} runs in this arm, {n_imp} identity runs imported from the Round-11 ladder "
          "(same computation, verified bit-identical by --harness-check)")
    sS = [share[f"doseS|{k}"] for k in KAPPAS]
    print("  mode S adversarial coefficient share: " + " ".join(f"{x:.6f}" for x in sS)
          + f"   spread {max(sS) - min(sS):.2e}   <- attenuation closed by construction")
    print(f"  frozen thresholds: CONFIRM if Delta > {cfg['confirm_above']:+.3f}, "
          f"REFUTE if Delta < {cfg['refute_below']:+.3f}, else indeterminate\n")

    r = rungs_of(cells_r, "S", D2, KAPPAS)
    if r is None:
        have = sorted({c["rung"] for c in cells_r.values()})
        sys.exit(f"INCOMPLETE: this arm has rungs {have}, needs {KAPPAS}. "
                 "Re-run when run_dose_replication.py has finished.")
    # The runner writes rung-major, so a rung can exist with only some of its seeds. Scoring a
    # partial rung would silently change the frozen seed count, so refuse.
    want = len(cfg["seeds"])
    short = [(x["rung"], len(x["asrs"])) for x in r if len(x["asrs"]) != want]
    if short:
        sys.exit(f"INCOMPLETE: {want} seeds frozen, but rungs {short} (rung, n) are partial. "
                 "Re-run when run_dose_replication.py has finished.")

    print(f"=== MODE S REPLICATION: {D2} / {ATTACK}, class {rep['arm']['prop1_class']} ===")
    print(f"    '*' = below the {ACC_FLOOR} accuracy gate: uninterpretable, not suppression.\n")
    show(f"{D2} (n={len(r[0]['asrs'])})", r)
    print(f"  {'':24s} measured decision  "
          + " ".join(f"{adm['summary'][f'doseS|{D2}|{k}|decision']:.3f}" for k in KAPPAS))
    print(f"  {'':24s} measured admission "
          + " ".join(f"{adm['summary'][f'doseS|{D2}|{k}|admission']:.3f}" for k in KAPPAS))

    delta = float(np.mean(r[-1]["asrs"])) - float(np.mean(r[0]["asrs"]))
    headroom = 1.0 - float(np.mean(r[0]["asrs"]))
    cmp_rises = comparison_rises(cells_t)
    print(f"\n  --- PRIMARY (rank discordance) ---")
    print(f"  Delta = mean ASR(k=2) - mean ASR(k=0) = {np.mean(r[-1]['asrs']):.3f} - "
          f"{np.mean(r[0]['asrs']):.3f} = {delta:+.3f}")
    print(f"  headroom 1 - ASR(k=0) = {headroom:.3f}; headroom-normalized rise = "
          f"{delta / headroom:+.3f}   (disclosure 3/4: cross-arm magnitudes, different attacks)")
    for d2, c in sorted(cmp_rises.items()):
        print(f"  comparison arm {d2:11s} rise {c['rise']:+.3f}  "
              f"(headroom-normalized {c['rise'] / (1 - c['identity']):+.3f})  "
              "-- recomputed per seed, not transcribed")

    if delta > cfg["confirm_above"]:
        verdict = ("ADMISSION ORDERING CONFIRMED -- the arm with the largest admission change and "
                   "the smallest decision change produced the largest rise")
    elif delta < cfg["refute_below"]:
        verdict = ("ADMISSION ORDERING REFUTED -- rise within the flat margin, so admission does "
                   "not order the outcomes; the positive claim is scoped to selector-type defenses. "
                   "This outcome REPLICATES THE FLAGSHIP NEGATIVE on a structurally different "
                   "defense: decision change 0.482 with suppression preserved")
    else:
        verdict = "INDETERMINATE (neither frozen criterion met) -- not scored in our favour"
    print(f"  ==> {verdict}")

    _, z, p_inc, p_dec, p_perm = jonckheere([x["asrs"] for x in r])
    print(f"\n  --- SECONDARY (within-arm shape) ---")
    print(f"  Jonckheere-Terpstra z={z:+.3f} p_inc={p_inc:.4g} p_dec={p_dec:.4g} perm={p_perm:.4g}")
    shape = ("RISING (p_inc < alpha)" if p_inc < ALPHA else
             "FALLING (p_dec < alpha) -- refutes BOTH readings, attenuation already closed"
             if p_dec < ALPHA else "NO MONOTONE TREND at alpha=0.05 -- both readings predicted a rise")
    print(f"  ==> {shape}")

    # POST-HOC, NOT PRE-REGISTERED. The frozen tests are unpaired, and between-seed variance on this
    # cell is large (identity spread 0.287-0.677), which is exactly what the JT test cannot see
    # through. The seed-matched difference is reported because it is the honest description of a small
    # consistent effect, and it is labelled post-hoc wherever it appears.
    paired = None
    if r[0]["seeds"] == r[-1]["seeds"]:
        diffs = [a - b for a, b in zip(r[-1]["asrs"], r[0]["asrs"])]
        t_p, p_p = (float("nan"), float("nan"))
        if sps is not None:
            tt = sps.ttest_rel(r[-1]["asrs"], r[0]["asrs"], alternative="greater")
            t_p, p_p = float(tt.statistic), float(tt.pvalue)
        paired = dict(diffs=diffs, mean=float(np.mean(diffs)), n_positive=sum(d > 0 for d in diffs),
                      n=len(diffs), t=t_p, p_one_sided_greater=p_p,
                      note="POST-HOC, not pre-registered. Frozen tests are unpaired.")
        print(f"\n  --- POST-HOC (not pre-registered): seed-matched difference k=2 vs k=0 ---")
        print("  per-seed diffs " + " ".join(f"{d:+.3f}" for d in diffs)
              + f"   mean {np.mean(diffs):+.3f}   positive in {paired['n_positive']}/{paired['n']} seeds")
        print(f"  paired t = {t_p:+.3f}, one-sided p = {p_p:.4g}  <- POST-HOC. The frozen primary is "
              "the unpaired Delta above, and it is not superseded by this.")

    lad = companion_ladder()
    if lad is not None:
        _, z1, pi1, pd1, _ = jonckheere([x["asrs"] for x in lad])
        print(f"\n  --- CONTRAST: the confounded companion ladder on this same cell ---")
        print("  ASR " + "  ".join(f"{x['mean']:.3f}" for x in lad)
              + f"   rise {lad[-1]['mean'] - lad[0]['mean']:+.3f}  "
                f"(JT p_dec={pd1:.4g}) -- benign AND adversarial weights both dosed")
        print("  Mode S closes the attenuation channel that explains that fall; the two ladders are "
              "the same cell measured with and without the confound.")

    gated = [f"k={x['rung']:g}" for x in r if x["gated"]]
    print(f"\n  accuracy gate: {'binds at ' + ','.join(gated) if gated else 'does not bind'} "
          f"(min mean accuracy {min(x['acc'] for x in r):.3f})")

    pooled = None
    if sps is not None:
        pts, n_existing = pooled_points(cells_t, cells_r, adm, share)
        pooled = pooled_fit(pts)
        b = pooled["beta"]
        print(f"\n  --- TERTIARY (pooled two-channel model, re-fit) ---")
        print(f"  n={pooled['n']} cells ({n_existing} existing + {pooled['n'] - n_existing} from "
              "this arm, its identity rung being a structural zero by the same convention)")
        print(f"  rise = {b[0]:+.3f} {b[1]:+.3f}*admission {b[2]:+.3f}*d(share)   "
              f"R^2 = {pooled['r2']:.3f}")
        print(f"  admission coefficient t = {pooled['t_admission']:+.3f}, one-sided p = "
              f"{pooled['p_admission']:.4g}")
        prev = pooled_fit([p for p in pts[:n_existing]])
        print(f"  before this arm: R^2 = {prev['r2']:.3f}, admission p = {prev['p_admission']:.4g} "
              f"({'CONFIRMED' if prev['confirmed'] else 'NOT CONFIRMED'})")
        print(f"  ==> two-channel reading {'CONFIRMED' if pooled['confirmed'] else 'NOT CONFIRMED'} "
              "(needs both coefficients positive and the admission coefficient significant)")
        pooled["previous"] = prev

    out = dict(
        description="Scored replication arm. Thresholds frozen at " + rep["prereg_commit"] +
                    " before results/dose_replication/ existed; nothing rescored here.",
        prereg_commit=rep["prereg_commit"],
        arm=dict(d2=D2, attack=ATTACK, mode="S", prop1_class=rep["arm"]["prop1_class"]),
        thresholds=dict(confirm_above=cfg["confirm_above"], refute_below=cfg["refute_below"],
                        acc_floor=ACC_FLOOR, alpha=ALPHA),
        rungs=[dict(rung=x["rung"], mean_asr=x["mean"], lo=x["lo"], hi=x["hi"], acc=x["acc"],
                    gated=x["gated"], imported=x["imported"], asrs=x["asrs"], seeds=x["seeds"],
                    decision=adm["summary"][f"doseS|{D2}|{x['rung']}|decision"],
                    admission=adm["summary"][f"doseS|{D2}|{x['rung']}|admission"]) for x in r],
        primary=dict(delta=delta, headroom=headroom, normalized_delta=delta / headroom,
                     comparison_arms=cmp_rises, verdict=verdict),
        secondary=dict(jt_z=float(z), p_increasing=float(p_inc), p_decreasing=float(p_dec),
                       p_permutation=float(p_perm), verdict=shape),
        post_hoc_paired=paired,
        companion_confounded_ladder=(
            dict(means=[x["mean"] for x in lad], rise=lad[-1]["mean"] - lad[0]["mean"],
                 note="benign and adversarial weights both dosed: attenuation not closed")
            if lad is not None else None),
        pooled=pooled,
    )
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nSaved to {os.path.relpath(OUT, base)}")
    print("Round 11 and Round 12 are not rescored: their refuted arms, the indeterminate cos_krum")
    print("arm and the unconfirmed pooled model stand exactly as reported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
