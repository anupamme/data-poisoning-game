"""
Seed top-up of the coord_median / committed_scaling cell: n=5 -> n=20, both designs.

WHY THIS EXISTS. The paper's headline pair -- outcome-gated -0.273 against within-defense +0.125 -- is
coord_median under committed_PIXEL at n=20. The same aggregator under the OTHER committed attack,
committed_scaling, is the cell that says the design disagreement is not specific to the pixel backdoor
(main.tex:853, :1970). That cell is at n=5 on both legs. This suite brings it to n=20.

WHAT MAKES THIS CELL WORTH THE COMPUTE, and it is not novelty:

  1. It is HELD OUT (`training: false` in results/comparability_six_cells.json), so it is not one of the
     four cells that trained the since-withdrawn DeltaLambda_a rule.
  2. Both margin legs are GENUINE TESTS on it. The identity rung's mean ASR is 0.5186, so the largest
     available fall is 0.5186 > 0.15. On the flagship cell the identity mean is 0.0439 and the lower leg
     is satisfied by arithmetic for any possible outcome; here it is not.
     results/margin_reachability.json records this cell as the only member of
     `distinct_arms_keeping_the_two_sided_reading`, and main.tex:3105 already singles it out as such.
  3. coord_median is a coordinate-wise robust STATISTIC, not a selector, so it is structurally different
     from Krum. At n=20 on both committed attacks it becomes the paper's best-powered aggregator.

THIS CELL IS A DISAGREEMENT, NOT A SIGN REVERSAL, and nothing here may call it one. At n=5 the
outcome-gated interval CONTAINS zero (+0.0019 [-0.0775, +0.0813]) while the Mode-S interval excludes it
(+0.1961 [+0.1120, +0.2802]). Both means are positive. main.tex:1970 already says so in those words.

THREE ESTIMANDS, all paired at the same 20 seeds:

  Delta_conf = mean_s[ ASR_s(confounded, k=2) - ASR_s(shared, k=0) ]
  Delta_ctrl = mean_s[ ASR_s(controlled, k=2) - ASR_s(shared, k=0) ]
  D          = mean_s[ Delta_ctrl,s - Delta_conf,s ]   <- formed PER SEED, then averaged

Because the identity rung is shared, D's per-seed value reduces exactly to
ASR_s(controlled, k=2) - ASR_s(confounded, k=2): the shared minuend cancels. That is asserted in-file, so
no later analysis can "improve" D by re-deriving it from two separately computed identity legs, which
would mix n between a minuend and a subtrahend.

THREE VERDICTS PER ESTIMAND, emitted independently and never substituted for one another: sign (does the
paired 95% interval exclude zero), margin (is |mean| < 0.15), one-sided (is the 95% upper bound below
+0.15). An interval can sit inside the margin and still exclude zero. No verdict literal names a
mechanism; the measured direction is printed beside it.

THE INTERVAL CONVENTION, fixed because the repository contains two. analyze_comparability.py:293 uses
`t_crit(n) * sd/sqrt(n)`, and analyze_headline_cis.t_crit(n) returns t.ppf(0.975, n-1) -- it takes n and
computes df = n-1 INTERNALLY. Calling t_crit(n-1) yields df = n-2 and a wider interval: at n=5 it returns
3.182 instead of 2.776 and turns [-0.0775, +0.0813] into [-0.0891, +0.0929]. Every published literal this
file reproduces does so under t_crit(n). t_crit is imported, never restated as a table lookup, because
analyze_headline_cis.T95 stops at df=9 and n=20 would raise.

45 RUNS, NOT 60. At kappa=0 the transform returns the update list unwrapped, so dose_kappa0.0 and
doseS_kappa0.0 are the same computation -- and for THIS cell that is already true BIT-FOR-BIT in
results/comparability_cells/summary.json, identically in ASR and clean accuracy at all five published
seeds under both family keys. So the identity rung is computed once per new seed and shared:
15 seeds x 3 runs = 45.

--harness-check ESTABLISHES that on this attack rather than inheriting it from the pixel cell, in four
runs at seed 42 where published values already exist (a new seed would make the check vacuous):

  1. outcome-gated kappa=0 == published dose_kappa0.0
  2. Mode-S       kappa=0 == the SAME published pair            <- what licenses sharing on THIS attack
  3. outcome-gated kappa=2 == published dose_kappa2.0
  4. Mode-S       kappa=2 == published doseS_kappa2.0

(1) and (2) tie each provenance of the shared rung to its own published value, so all four quantities are
equal by transitivity. (3) and (4) prove the harness is the same loop the published cell was run with,
not merely the same at the identity, on BOTH designs.

The check is BLOCKING: its verdict dict is PERSISTED into the artifact, and the scoring run refuses to
start unless the artifact records a pass whose published reference values still match the frozen source.
A check whose verdict is discarded is witnessed only by a terminal and costs hours to repeat.

If (2) fails, the identity is NOT shared on this attack: the suite becomes 60 runs with both identity legs
computed separately, reported as a difference between the two attacks rather than reconciled after the
fact. If (1), (3) or (4) fails, the harness is not the published one and the top-up does not run at all.

NO EXISTING ARTIFACT IS WRITTEN. results/comparability_cells/, results/comparability_six_cells.json,
results/margin_sensitivity.json and results/margin_reachability.json are read-only here. No existing
runner or analyzer is edited; run_one, cell_key, KAPPAS, SEEDS, FL_CONFIG and ADV_FRACTION are imported.

DO NOT RUN until experiments/pre_registration_coordmedian_scaling_n20.md is git-committed and
PREREG_COMMIT below is set to that hash. The suite refuses otherwise, and also refuses if that file has
uncommitted edits, because `git log -1` cannot see those.

Output: results/coordmedian_scaling_topup/summary.json (resumable; rewritten after every run).

  python3 experiments/run_coordmedian_scaling_topup.py --harness-check
  python3 experiments/run_coordmedian_scaling_topup.py
  python3 experiments/run_coordmedian_scaling_topup.py --analyze-only
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Single-sourced from the frozen comparability suite: the FL loop, the two-token design contrast, the
# cell-key format, the rung grid and the published seed list are imported rather than restated, so this
# top-up cannot drift from the cell it extends in any of them.
from experiments.run_comparability_cells import (run_one, cell_key,        # noqa: E402
                                                 KAPPAS, SEEDS,
                                                 FL_CONFIG, ADV_FRACTION)
# NOT T95[n-1]: that literal table stops at df=9, and n=20 would raise. t_crit takes n and computes
# df = n-1 internally, which is the whole convention this file is pinned to.
from experiments.analyze_headline_cis import t_crit                        # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "coordmedian_scaling_topup")
PREREG = "experiments/pre_registration_coordmedian_scaling_n20.md"

# The commit that froze PREREG, made before results/coordmedian_scaling_topup/ existed. `git log` can
# demonstrate that ordering, which is what makes this top-up prospective (non-negotiable 7). No
# amendment is made to that document once any number exists, so this hash is not expected to move.
PREREG_COMMIT = "74044f1"

DATASET, MODEL, D2, ATTACK = "cifar10", "cifar_cnn", "coord_median", "committed_scaling"
LO, HI = KAPPAS[0], KAPPAS[-1]           # 0.0 and 2.0: the frozen primary contrast, endpoints only
NEW_SEEDS = list(range(47, 62))          # 47-61, frozen in the pre-registration
PUBLISHED_SEEDS = SEEDS                  # 42-46, in the frozen comparability artifact, not re-run

# The one frozen artifact this top-up merges with at analysis time. Read-only here. Unlike the pixel
# cell, whose two designs live in two directories, both designs of this cell live in one file under
# different family keys.
SRC = os.path.join(BASE, "results", "comparability_cells", "summary.json")

# Inherited verbatim from pre_registration_comparability.md:75 and :149 -- floor on a rung's MEAN, not
# per seed, and "seeds below it are flagged, never excluded". A rung whose mean falls below it is VOID
# (uninterpretable), never "suppressed". analyze_comparability.py reads no accuracy at all, so the
# flagging is this runner's job and is written into the artifact.
ACC_FLOOR = 0.35
MARGIN = 0.15                            # the frozen practical-equivalence margin

# Published n=5, recomputed from the frozen artifact by _published_stats() rather than transcribed, and
# asserted against these literals so a silent change to the source cannot pass unnoticed. The literals
# are what the paper prints at main.tex:1962 and :1970.
PUBLISHED_N5_EXPECTED = {
    "Delta_conf": (+0.001867, [-0.077531, +0.081265]),
    "Delta_ctrl": (+0.196111, [+0.111974, +0.280248]),
}
PUBLISHED_VERDICT_N5 = (
    "DISAGREE: the outcome-gated interval CONTAINS zero (+0.0019 [-0.0775, +0.0813]) while the Mode-S "
    "interval EXCLUDES it (+0.1961 [+0.1120, +0.2802]). Both means are positive. This is a "
    "disagreement, NOT a sign reversal, and it is not called one at any n (main.tex:1970)."
)

# The published rows the harness check compares against, by (family, kappa) -> mean ASR at n=5. Recorded
# so the persisted verdict can be re-validated on a later run without recomputing anything.
HARNESS_ROWS = (("confounded", "LO"), ("controlled", "LO"), ("confounded", "HI"), ("controlled", "HI"))


def _src_cells():
    if not os.path.exists(SRC):
        return {}
    return json.load(open(SRC)).get("cells", {})


def published_rows(family, kappa):
    """{seed: row} for the published rung, or {} if absent. Read-only, never written."""
    cell = _src_cells().get(cell_key(family, D2, ATTACK, kappa))
    return {int(r["seed"]): r for r in cell["per_seed"]} if cell else {}


def _ci(d):
    """Paired 95% t interval, exactly analyze_comparability.py:293's convention."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    hw = t_crit(n) * se if n > 1 else float("nan")
    # Every field cast to a builtin float: numpy scalars leak out of sd/sqrt(n) and json.dump raises on
    # them, which would surface as a crash AFTER the first 13-minute run rather than before it.
    return {"n": n, "mean": m, "sd": sd, "se": float(se),
            "t_crit": float(t_crit(n)) if n > 1 else None,
            "half_width": float(hw) if n > 1 else None,
            "ci95": [float(m - hw), float(m + hw)] if n > 1 else None}


def _verdicts(block):
    """The three labels, read separately. None may be reported as another."""
    if block["ci95"] is None:
        return {"verdict_sign": "undefined_at_n1", "verdict_margin": "undefined_at_n1",
                "verdict_upper": "undefined_at_n1"}
    lo, hi = block["ci95"]
    m = block["mean"]
    return {
        "verdict_sign": "excludes_zero" if lo * hi > 0 else "contains_zero",
        "verdict_sign_reading": ("the interval EXCLUDES zero" if lo * hi > 0
                                 else "the interval CONTAINS zero"),
        "measured_direction": "positive" if m > 0 else ("negative" if m < 0 else "zero"),
        "verdict_margin": "inside_margin" if abs(m) < MARGIN else "outside_margin",
        "verdict_margin_reading": (f"|mean| < the frozen equivalence margin {MARGIN}"
                                   if abs(m) < MARGIN
                                   else f"|mean| >= the frozen equivalence margin {MARGIN}"),
        "verdict_upper": "upper_bounded" if hi < MARGIN else "upper_not_bounded",
        "verdict_upper_reading": (f"the 95% upper bound {hi:+.4f} is below the margin {MARGIN}"
                                  if hi < MARGIN
                                  else f"the 95% upper bound {hi:+.4f} is NOT below the margin "
                                       f"{MARGIN}"),
        "margin": MARGIN,
        "reading_rule": "verdict_sign, verdict_margin and verdict_upper are INDEPENDENT labels and none "
                        "may be reported as another: an interval can sit inside the margin and still "
                        "exclude zero, and a one-sided bound can hold while a two-sided reading fails.",
    }


def _published_stats():
    """The n=5 readings, recomputed from the frozen artifact, asserted against PUBLISHED_N5_EXPECTED."""
    lo_c, lo_s = published_rows("confounded", LO), published_rows("controlled", LO)
    hi_c, hi_s = published_rows("confounded", HI), published_rows("controlled", HI)
    if not (lo_c and lo_s and hi_c and hi_s):
        return None, ["one or more published rungs absent from the frozen source"]
    seeds = sorted(set(lo_c) & set(lo_s) & set(hi_c) & set(hi_s))
    dconf = [hi_c[s]["asr"] - lo_c[s]["asr"] for s in seeds]
    dctrl = [hi_s[s]["asr"] - lo_s[s]["asr"] for s in seeds]
    out = {"seeds": seeds,
           "Delta_conf": _ci(dconf), "Delta_ctrl": _ci(dctrl),
           "D": _ci([a - b for a, b in zip(dctrl, dconf)]),
           "identity_shared_bit_identical": all(lo_c[s]["asr"] == lo_s[s]["asr"]
                                                and lo_c[s]["accuracy"] == lo_s[s]["accuracy"]
                                                for s in seeds),
           "verdict": PUBLISHED_VERDICT_N5}
    bad = []
    for name, (m_exp, ci_exp) in PUBLISHED_N5_EXPECTED.items():
        got = out[name]
        if abs(got["mean"] - m_exp) > 5e-6 or any(
                abs(a - b) > 5e-6 for a, b in zip(got["ci95"], ci_exp)):
            bad.append(f"{name}: recomputed mean={got['mean']:+.6f} ci={got['ci95']} "
                       f"but the paper prints {m_exp:+.6f} {ci_exp}")
    return out, bad


def check_frozen():
    """Refuse to run unless the pre-registration is committed at the recorded hash AND clean.

    The hash check alone is necessary and not sufficient: `git log -1` reports the last commit that
    touched the file, which is unchanged by uncommitted edits to it. Both halves are here from the
    start rather than added after a near miss.
    """
    if not os.path.exists(os.path.join(BASE, PREREG)):
        print(f"REFUSING TO RUN: {PREREG} does not exist.")
        return False
    if PREREG_COMMIT in (None, "PENDING"):
        print("REFUSING TO RUN: the refuting branches are not frozen.\n"
              f"  1. git add {PREREG} && git commit\n"
              "  2. set PREREG_COMMIT here to that hash\n"
              "  3. rerun.\n"
              "Adding seeds is licensed ONLY by a committed pre-registered commitment to publish the "
              "refuting branch. Without the commit there is no commitment.")
        return False
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20)
    actual = out.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO RUN: {PREREG} last touched at {actual or 'UNTRACKED'}, "
              f"but PREREG_COMMIT is {PREREG_COMMIT}.")
        return False
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        print(f"REFUSING TO RUN: {PREREG} has uncommitted changes "
              f"({dirty.stdout.strip().split()[0]}), so it is not frozen at {actual} whatever "
              "`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
        return False
    if not os.path.exists(SRC):
        print(f"REFUSING TO RUN: {os.path.relpath(SRC, BASE)} does not exist. The five published "
              "seeds this top-up extends live there.")
        return False
    for family in ("confounded", "controlled"):
        for kappa in (LO, HI):
            if not published_rows(family, kappa):
                print(f"REFUSING TO RUN: no published {family} {D2}/{ATTACK} rung at kappa={kappa}. "
                      "The top-up extends an existing pair of ladders; it does not create them.")
                return False
    pub, bad = _published_stats()
    if bad:
        print("REFUSING TO RUN: the frozen source no longer reproduces the published readings:")
        for b in bad:
            print(f"    {b}")
        return False
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    print(f"[OK] published n=5 reproduces: Delta_conf {pub['Delta_conf']['mean']:+.6f} "
          f"{[round(x, 6) for x in pub['Delta_conf']['ci95']]}, "
          f"Delta_ctrl {pub['Delta_ctrl']['mean']:+.6f} "
          f"{[round(x, 6) for x in pub['Delta_ctrl']['ci95']]}")
    return True


def _one_check(n, total, label, family, kappa):
    """Recompute one published row in-suite and demand agreement to 1e-9."""
    rows = published_rows(family, kappa)
    seed = PUBLISHED_SEEDS[0]
    ref = rows.get(seed)
    if ref is None:
        print(f"  [{n}/{total}] {label}: SKIPPED, seed {seed} not published")
        return {"check": label, "family": family, "kappa": kappa, "status": "skipped",
                "why": f"seed {seed} not published"}
    print(f"  [{n}/{total}] {label}: published acc={ref['accuracy']!r} ASR={ref['asr']!r}", flush=True)
    t0 = time.time()
    acc, asr = run_one(seed, family, D2, ATTACK, kappa, DATASET, MODEL)
    dacc, dasr = acc - ref["accuracy"], asr - ref["asr"]
    ok = abs(dacc) < 1e-9 and abs(dasr) < 1e-9
    print(f"        recomputed acc={acc!r} ASR={asr!r}")
    print(f"        |d| = ({abs(dacc):.3e}, {abs(dasr):.3e})  "
          f"{'OK' if ok else '** MISMATCH'}  ({time.time() - t0:.0f}s)\n", flush=True)
    return {"check": label, "family": family, "kappa": kappa, "seed": seed,
            "status": "pass" if ok else "FAIL",
            "published": {"accuracy": ref["accuracy"], "asr": ref["asr"]},
            "recomputed": {"accuracy": float(acc), "asr": float(asr)},
            "abs_delta": {"accuracy": abs(float(dacc)), "asr": abs(float(dasr))},
            "tolerance": 1e-9, "seconds": round(time.time() - t0, 1)}


def harness_check():
    """Four runs at seed 42. Establishes the shared identity rung ON THIS ATTACK instead of assuming it.

    Returns (ok, verdict_dict). The verdict is PERSISTED by the caller: a check whose result is thrown
    away is witnessed only by a terminal and costs hours to repeat.
    """
    print(f"=== HARNESS CHECK: {D2}/{ATTACK.replace('committed_', '')} on {DATASET}, "
          f"seed {PUBLISHED_SEEDS[0]} ===")
    print("    (1) and (2) tie each provenance of the SHARED kappa=0 rung to its own published value,")
    print("    so all four quantities are equal by transitivity and the rung may be computed once.")
    print("    (3) and (4) prove this is the same loop the published cell was run with, on BOTH")
    print("    designs, not merely the same at the identity.")
    print("    This is established on committed_scaling rather than inherited from the pixel cell.\n",
          flush=True)

    checks = [
        _one_check(1, 4, f"outcome-gated kappa={LO} vs published dose_kappa{LO}", "confounded", LO),
        _one_check(2, 4, f"Mode-S kappa={LO} vs the SAME published pair", "controlled", LO),
        _one_check(3, 4, f"outcome-gated kappa={HI} vs published dose_kappa{HI}", "confounded", HI),
        _one_check(4, 4, f"Mode-S kappa={HI} vs published doseS_kappa{HI}", "controlled", HI),
    ]
    ran = [c for c in checks if c["status"] != "skipped"]
    failed = [c for c in ran if c["status"] == "FAIL"]
    shared = [c for c in checks if c["kappa"] == LO]
    shared_ok = all(c["status"] == "pass" for c in shared)

    verdict = {
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "seed": PUBLISHED_SEEDS[0], "tolerance": 1e-9,
        "source": os.path.relpath(SRC, BASE),
        "checks": checks,
        "n_run": len(ran), "n_failed": len(failed),
        "shared_identity_established": bool(shared_ok),
        "runs_licensed": None, "status": None, "reading": None,
    }

    if not ran:
        verdict.update(status="FAIL", runs_licensed=None,
                       reading="ALL FOUR SKIPPED: nothing was verified. Treated as a failure, not a "
                               "pass.")
        print("  ALL FOUR SKIPPED: nothing was verified. Treat as a failure, not a pass.")
        return False, verdict
    if failed and not shared_ok and len(failed) == len([c for c in shared if c["status"] == "FAIL"]):
        # Only a kappa=0 leg disagrees: the identity is not shared on this attack. That is the named
        # fallback, not a pass -- it changes the run from 45 to 60 and must be decided deliberately.
        verdict.update(status="FAIL_SHARED_IDENTITY", runs_licensed=60,
                       reading="The kappa=0 rung does NOT reproduce identically across both designs on "
                               "this attack, so the identity is not shared here as it is on the pixel "
                               "cell. The named fallback applies: 60 runs with both identity legs "
                               "computed separately, reported as a difference between the two attacks "
                               "and NOT reconciled after the fact. This suite does not proceed at 45.")
        print("  ** REFUSING TO CONTINUE: the shared kappa=0 rung is not one rung on this attack.")
        print("     Named fallback: 60 runs with both identity legs computed separately.")
        return False, verdict
    if failed:
        verdict.update(status="FAIL", runs_licensed=None,
                       reading="A published row did not reproduce, so this harness is not the loop the "
                               "published cell was run with. The top-up does not run at all.")
        print("  ** REFUSING TO CONTINUE. A mismatch here means this harness is not the published one.")
        return False, verdict
    if not shared_ok:
        verdict.update(status="FAIL", runs_licensed=None,
                       reading="The shared-rung check itself did not run, so sharing kappa=0 between "
                               "the two legs is unverified. Run 60 runs, not 45, or fix the skip.")
        print("  ** REFUSING TO CONTINUE: the shared-rung check did not run.")
        return False, verdict

    verdict.update(status="pass", runs_licensed=45,
                   reading=f"All {len(ran)} published rows reproduce to 1e-9 on both designs, and the "
                           "kappa=0 rung is bit-identical across them on THIS attack, so it may be "
                           "computed once per seed and shared: 15 x 3 = 45 runs, not 60.")
    print(f"  OK, all {len(ran)} checks agree to 1e-9. kappa=0 may be shared between the two legs.\n")
    return True, verdict


def load_artifact():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _harness_gate(art):
    """The harness verdict is BLOCKING, and it is re-validated rather than trusted on sight.

    Persisting a pass is not enough: the published values it compared against live in a file this suite
    does not own, so the recorded `published` numbers are checked against the frozen source again. A
    source that has moved since the check invalidates it.
    """
    hv = art.get("harness_check")
    if not hv:
        print("REFUSING TO RUN: no harness-check verdict in the artifact.\n"
              f"  python3 {os.path.relpath(__file__, BASE)} --harness-check\n"
              "The check establishes that the kappa=0 rung is shared ON THIS ATTACK. Without it, "
              "45 runs assume what 60 runs would measure.")
        return False
    if hv.get("status") != "pass":
        print(f"REFUSING TO RUN: the recorded harness verdict is {hv.get('status')!r}, not 'pass'.")
        print(f"  {hv.get('reading')}")
        return False
    if hv.get("runs_licensed") != len(NEW_SEEDS) * 3:
        print(f"REFUSING TO RUN: the harness licensed {hv.get('runs_licensed')} runs but this suite "
              f"plans {len(NEW_SEEDS) * 3}.")
        return False
    for c in hv.get("checks", []):
        if c["status"] == "skipped":
            continue
        ref = published_rows(c["family"], c["kappa"]).get(int(c["seed"]))
        if ref is None:
            print(f"REFUSING TO RUN: the harness compared against a published {c['family']} "
                  f"kappa={c['kappa']} row at seed {c['seed']} that is no longer in the source.")
            return False
        if (abs(ref["asr"] - c["published"]["asr"]) > 0
                or abs(ref["accuracy"] - c["published"]["accuracy"]) > 0):
            print(f"REFUSING TO RUN: the published {c['family']} kappa={c['kappa']} row has CHANGED "
                  f"since the harness check, so that check no longer establishes anything.")
            return False
    print(f"[OK] harness check recorded as pass, {hv['n_run']} rows re-validated against the frozen "
          f"source; {hv['runs_licensed']} runs licensed")
    return True


def save(art):
    os.makedirs(OUT, exist_ok=True)
    json.dump(art, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def _shell(cells, harness=None, extra=None):
    """The artifact's standing fields. Rebuilt on every write so a resume cannot leave them stale."""
    pub, _ = _published_stats()
    art = {
        "description": "Seed top-up of the coord_median / committed_scaling cell on CIFAR-10/cifar_cnn, "
                       "seeds 47-61, bringing BOTH designs from n=5 to n=20 at the endpoint rungs "
                       "kappa in {0, 2}. Purpose is INTERVAL WIDTH on the separation between the two "
                       "designs on a HELD-OUT cell where both margin legs are genuine tests, not a "
                       "second test of either leg's direction. Rules, estimands, verdict labels and "
                       f"both refuting branches frozen at {PREREG_COMMIT} ({PREREG}). "
                       "results/comparability_cells/ is never written here; the merge happens at "
                       "analysis time.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "dataset": DATASET, "model": MODEL,
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "kappas_run": [LO, HI], "kappas_frozen_grid": KAPPAS,
                   "new_seeds": NEW_SEEDS, "published_seeds": PUBLISHED_SEEDS,
                   "acc_floor": ACC_FLOOR, "margin": MARGIN},
        "arm": {"d2": D2, "attack": ATTACK, "families": ["confounded", "controlled"],
                "training_cell": False,
                "held_out": "training: false in results/comparability_six_cells.json -- not one of the "
                            "four cells that trained the since-withdrawn DeltaLambda_a rule",
                "label_is_not_a_reversal": "This cell is a DISAGREEMENT, never a sign reversal, at any "
                                           "n and under any outcome (main.tex:1970).",
                "d_admission_imported": 0.0,
                "d_influence_imported": 0.008619231983179303,
                "imported_not_remeasured": True},
        "reachability": {
            "identity_mean_asr_n5": 0.5185777777777778,
            "margin": MARGIN,
            "lower_leg_is_a_test": True,
            "fall_available": 0.5185777777777778,
            "why_this_cell_was_chosen": "identity mean ASR 0.5186 > 0.15, so the largest available fall "
                                        "exceeds the margin and BOTH legs are genuine tests. On the "
                                        "flagship cell the identity mean is 0.0439 and the lower leg is "
                                        "satisfied by arithmetic for any possible outcome.",
            "source": "results/margin_reachability.json -> "
                      "distinct_arms_keeping_the_two_sided_reading (this cell is its only member)",
            "recheck_at_n20": "0.5186 is an n=5 mean, and the reachability premise is a statement about "
                              "the identity rung's LEVEL, not about a paired difference. At n=20 the "
                              "level moves, so analyze() recomputes it and re-reads the premise. If the "
                              "20-seed identity mean fell below 0.15 the lower leg would become "
                              "arithmetic and this cell would lose the exact property it was chosen "
                              "for; that is branch REACHABILITY_LOST. The premise is not assumed to "
                              "survive its own top-up."},
        "identity_mean_is_a_collision": {
            "value_at_n5": 0.5185777777777778, "printed_as": "0.519 / 0.5186",
            "warning": "The identity rung's mean ASR at n=20 is a DIFFERENT quantity from the 0.519 the "
                       "paper prints, and the two round to the same three digits. Every published site "
                       "is an n=5 reading of coord_median's standalone committed-scaling ASR on seeds "
                       "42-46; none is superseded by this top-up and none may be overwritten with the "
                       "20-seed value.",
            "published_n5_sites": [
                "main.tex:1297 ('the strongest single defense reaches only $0.519$', C0 vacuity on the "
                "7-defense menu)",
                "main.tex:2225 (tab row 'fedavg -> coord_median & DEGEN & LOW & 0.519')",
                "main.tex:2241 ('whose standalone model-scaling ASR is $0.519$'; that paragraph states "
                "'every C1 input here is the n=5 single-defense baseline on seeds 42--46')",
                "main.tex:2248 (TWICE: 'the minimum single-defense max-committed ASR is CoordMedian's "
                "$0.519$' and 'whose own baseline is $0.519$')",
                "main.tex:3105 ('identity mean $0.519$', with $n{=}5$ disclosed on :3106)",
                "main.tex:3449 (tab row 'pure\\_coord\\_median ... $0.5186 \\pm 0.1413$')",
                "main.tex:3473 ('pure\\_coord\\_median returns $0.5186$')"],
            "hard_constraint": "main.tex:2248 is a SUPERLATIVE across the 7-defense menu -- 'the MINIMUM "
                               "single-defense max-committed ASR is CoordMedian's 0.519' -- where every "
                               "other defense is at n=5. Substituting the 20-seed value there would mix "
                               "n inside a comparison, which is the same defect as mixing n between a "
                               "minuend and a subtrahend. The 20-seed identity mean does not enter that "
                               "sentence, or any cross-defense comparison, under any branch."},
        "interval_convention": "95% two-sided paired Student-t, t_crit(n) * sd/sqrt(n), exactly "
                               "analyze_comparability.py:293. t_crit TAKES n and computes df = n-1 "
                               "internally; calling t_crit(n-1) would give df = n-2 and a wider "
                               "interval. This is NOT the 90% TOST convention of "
                               "supplementary.tex:441 and is never compared against a number that is.",
        "endpoint_only": "kappa=0.5 and kappa=1.0 are NOT run for the new seeds. This cell's four-rung "
                         "trend statistics stay at n=5, stay pre_registered: false and stay post hoc; "
                         "no Jonckheere-Terpstra statistic is computed for the new seeds; and no "
                         "four-rung display may print this ladder without a per-rung n.",
        "no_admission_quantity": "run_one returns (accuracy, asr) and nothing else. This arm measures "
                                 "ASR ONLY. d_admission and d_influence are imported from the frozen "
                                 "six-cell measurement, summary[...|admission] is 0.0 at every "
                                 "identity rung BY CONSTRUCTION and is never read as a level, and no "
                                 "statement that a statistic disturbance costs or buys suppression on "
                                 "this cell is licensed in either direction.",
        "oracle_scope": "The controlled leg is MASKED (adv_mask reads adversary identity), so this arm "
                        "bears nothing on the oracle objection. The sentence 'the negative reproduces "
                        "oracle-free' may not be written, here or anywhere.",
        "identity_rung_provenance": "kappa=0 is COMPUTED ONCE per new seed and shared by both legs. "
                                    "dose_kappa0.0 and doseS_kappa0.0 are the same computation -- "
                                    "bit-identical in ASR and accuracy at all five published seeds in "
                                    "results/comparability_cells/summary.json -- and --harness-check "
                                    "re-establishes that ON THIS ATTACK at seed 42 against both "
                                    "published family keys before any new run. The controlled copy of "
                                    "each shared row is labelled in `source`.",
        "published_n5": pub,
        "harness_check": harness,
        "cells": cells,
    }
    if extra:
        art.update(extra)
    return art


def _record(cells, family, kappa, seed, acc, asr, source):
    key = cell_key(family, D2, ATTACK, kappa)
    cell = cells.setdefault(key, {"d2": D2, "attack": ATTACK, "kappa": kappa, "family": family,
                                  "dataset": DATASET, "model": MODEL, "per_seed": []})
    row = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr), "source": source,
           "below_acc_floor": bool(acc < ACC_FLOOR)}
    # Replace, never append blind. The shared kappa=0 rung is written to two keys from one run, so a
    # crash between the two writes leaves the confounded row present and the controlled one absent; the
    # resume then recomputes and would append a duplicate seed to a rung that already had it, silently
    # inflating n on one leg only.
    cell["per_seed"] = [r for r in cell["per_seed"] if int(r["seed"]) != int(seed)] + [row]
    cell["per_seed"].sort(key=lambda r: r["seed"])
    cell["mean_asr"] = float(np.mean([r["asr"] for r in cell["per_seed"]]))
    cell["mean_accuracy"] = float(np.mean([r["accuracy"] for r in cell["per_seed"]]))
    cell["n_below_acc_floor"] = sum(1 for r in cell["per_seed"] if r["below_acc_floor"])
    return key


def _has(cells, family, kappa, seed):
    cell = cells.get(cell_key(family, D2, ATTACK, kappa), {})
    return any(int(r["seed"]) == seed for r in cell.get("per_seed", []))


def merged(cells, family, kappa):
    """{seed: row} over published + new seeds. Published rows are read, never rewritten."""
    out = dict(published_rows(family, kappa))
    for r in cells.get(cell_key(family, D2, ATTACK, kappa), {}).get("per_seed", []):
        out[int(r["seed"])] = r
    return out


def analyze(cells):
    """The three estimands at whatever n has actually been reached, with three verdicts each."""
    legs = {(f, k): merged(cells, f, k) for f in ("confounded", "controlled") for k in (LO, HI)}
    seeds = sorted(set.intersection(*(set(v) for v in legs.values())))
    if len(seeds) < 2:
        return {"n_paired_seeds": len(seeds), "why": "fewer than two fully paired seeds"}

    lo_c, lo_s = legs[("confounded", LO)], legs[("controlled", LO)]
    hi_c, hi_s = legs[("confounded", HI)], legs[("controlled", HI)]
    dconf = [hi_c[s]["asr"] - lo_c[s]["asr"] for s in seeds]
    dctrl = [hi_s[s]["asr"] - lo_s[s]["asr"] for s in seeds]
    dpair = [a - b for a, b in zip(dctrl, dconf)]

    # The identity rung is shared, so D per seed must reduce EXACTLY to the difference of the two
    # kappa=2 rungs. Asserted rather than assumed: if the identity legs ever diverge, the reduction is
    # false and D would be silently mixing two identity computations.
    reduction = [hi_s[s]["asr"] - hi_c[s]["asr"] for s in seeds]
    identity_shared = all(lo_c[s]["asr"] == lo_s[s]["asr"] for s in seeds)
    reduction_holds = all(abs(a - b) < 1e-12 for a, b in zip(dpair, reduction))

    rungs = {}
    for (family, kappa), rows in legs.items():
        accs = [rows[s]["accuracy"] for s in seeds]
        mean_acc = float(np.mean(accs))
        rungs[cell_key(family, D2, ATTACK, kappa)] = {
            "n": len(seeds), "mean_asr": float(np.mean([rows[s]["asr"] for s in seeds])),
            "mean_accuracy": mean_acc,
            "seeds_below_floor": [s for s in seeds if rows[s]["accuracy"] < ACC_FLOOR],
            "rung_void": bool(mean_acc < ACC_FLOOR),
            "acc_gate": ("VOID: the rung MEAN is below the floor, so the rung is uninterpretable, "
                         "not suppressed" if mean_acc < ACC_FLOOR else
                         "PASS: the rung mean is above the floor; any individual seed below it is "
                         "flagged, never excluded"),
        }

    out = {"n_paired_seeds": len(seeds), "paired_seeds": seeds, "acc_floor": ACC_FLOOR,
           "rungs": rungs,
           "identity_shared_across_designs": bool(identity_shared),
           "D_reduction_holds": bool(reduction_holds),
           "D_reduction_note": "Because the identity rung is shared, D per seed reduces exactly to "
                               "ASR(controlled, k=2) - ASR(confounded, k=2). Asserted here so no later "
                               "analysis re-derives D from two separately computed identity legs, "
                               "which would mix n between a minuend and a subtrahend.",
           "any_rung_void": any(r["rung_void"] for r in rungs.values()),
           "estimands": {}}

    # The reachability premise is about the identity rung's LEVEL, so it moves with n and is re-read
    # rather than inherited from the freeze's n=5 statement. This is the property the cell was chosen
    # for, and a top-up is exactly the thing that can take it away.
    id_mean = rungs[cell_key("confounded", D2, ATTACK, LO)]["mean_asr"]
    out["reachability_at_this_n"] = {
        "identity_mean_asr": id_mean, "identity_mean_asr_n5": 0.5185777777777778,
        "margin": MARGIN, "fall_available": id_mean,
        "lower_leg_is_a_test": bool(id_mean > MARGIN),
        "reading": (f"identity mean ASR {id_mean:.4f} > {MARGIN}, so the largest available fall still "
                    "exceeds the margin and BOTH legs remain genuine tests on this cell"
                    if id_mean > MARGIN else
                    f"identity mean ASR {id_mean:.4f} <= {MARGIN}: the lower leg is now ARITHMETIC and "
                    "this cell no longer keeps a genuinely two-sided reading"),
        "collision_warning": "This value is NOT the 0.519 the paper prints at main.tex:1297, :2225, "
                             ":2241, :2248 (twice), :3105, :3449 and :3473. Those are n=5 readings of "
                             "the same estimand and they round to the same three digits. Nothing here "
                             "supersedes them, and main.tex:2248's cross-defense minimum stays at n=5.",
    }

    for name, d, what in (("Delta_conf", dconf, "outcome-gated endpoint contrast, kappa=2 minus "
                                                "identity, paired"),
                          ("Delta_ctrl", dctrl, "Mode-S endpoint contrast, kappa=2 minus identity, "
                                                "paired"),
                          ("D", dpair, "Delta_ctrl minus Delta_conf, formed PER SEED then averaged")):
        block = _ci(d)
        block["estimand"] = name
        block["what"] = what
        block["per_seed"] = {str(s): float(v) for s, v in zip(seeds, d)}
        block["verdicts"] = _verdicts(block)
        out["estimands"][name] = block

    # The branches, read off the frozen rules rather than chosen after the fact. All three are read
    # independently and more than one can hold.
    dblock, cblock = out["estimands"]["D"], out["estimands"]["Delta_conf"]
    sblock = out["estimands"]["Delta_ctrl"]
    branches = []
    if dblock["verdicts"]["verdict_sign"] == "contains_zero":
        branches.append({
            "branch": "REFUTING_1_DESIGNS_AGREE",
            "reading": "D's interval CONTAINS zero: the two designs are not separated on this cell at "
                       f"n={len(seeds)}. The published n=5 disagreement was an artifact of five seeds. "
                       "This cell's `observed` label goes DISAGREE -> AGREE and the paper's seven-cell "
                       "count of disagreements falls from five to four. Reported as a contradiction of "
                       "our own published count and NOT converted into a scope condition.",
            "sites": ["main.tex:72 (abstract, 'disagree on five of seven cells')",
                      "main.tex:853 (section 5, 'They disagree on five' and 'coord_median under model "
                      "scaling disagrees too')",
                      "Figure 1(b) panel title ('5 of 7 cells, and its SIGN on 2') and this cell's row",
                      "main.tex:1962 (tab:sixcell: both means, both intervals, the observed column)",
                      "main.tex:1970 ('one confirms, one refutes, one was never predicted')",
                      "main.tex:1976 ('across seven cells the two designs disagree on five')"],
            "not_affected": "The two sign reversals are the coord_median/pixel cells. This cell was "
                            "never counted as one and does not become one under any branch."})
    else:
        branches.append({
            "branch": "CONFIRMING_DISAGREEMENT_NARROWED",
            "reading": "D's interval EXCLUDES zero: the two designs remain separated on this cell at "
                       f"n={len(seeds)}. Reported as the published result, narrowed, with the label "
                       "DISAGREE unchanged. Still a disagreement, still not a sign reversal."})
    if cblock["verdicts"]["verdict_sign"] == "excludes_zero":
        branches.append({
            "branch": "REFUTING_2_CONFOUNDED_LEG_RELABELLED",
            "reading": "Delta_conf, which CONTAINS zero at n=5, EXCLUDES it at "
                       f"n={len(seeds)}: the outcome-gated design does detect an effect on this cell "
                       "and the cell's description changes. Reported as measured.",
            "sites": ["main.tex:1962 (tab:sixcell, this cell's row: both means and both intervals)",
                      "main.tex:3105-3106 (one wrapped sentence: ':3105 names the arm and its identity "
                      "mean, :3106 carries the interval [-0.078,+0.081] and the n=5 disclosure, so an "
                      "edit landing on :3105 alone leaves the stale interval in place)"]})
    if not out["reachability_at_this_n"]["lower_leg_is_a_test"]:
        branches.append({
            "branch": "REACHABILITY_LOST",
            "reading": "The identity rung's mean ASR has fallen to or below the margin at "
                       f"n={len(seeds)} (0.5186 at n=5 -> "
                       f"{out['reachability_at_this_n']['identity_mean_asr']:.4f}), so the lower "
                       "equivalence leg on this cell is now satisfied by arithmetic and the cell no "
                       "longer keeps a genuinely two-sided reading. This is the property the cell was "
                       "chosen for, and the top-up removed it. Reported as measured, in the direction "
                       "that costs us the claim.",
            "sites": ["main.tex:3105-3106 (this cell is 'The fifth ... keeps a genuinely two-sided "
                      "reading and is reported as one', and the surrounding count '4 of the 5 have an "
                      "arithmetically satisfied lower leg' becomes 5 of 5)",
                      "results/margin_reachability.json -> "
                      "distinct_arms_keeping_the_two_sided_reading would empty; that file is READ-ONLY "
                      "here and is not rewritten -- the contradiction is reported, not patched"]})
    for nm, blk in (("Delta_conf", cblock), ("Delta_ctrl", sblock)):
        pub_mean = PUBLISHED_N5_EXPECTED[nm][0]
        if blk["mean"] * pub_mean < 0:
            branches.append({
                "branch": f"DIRECTION_CHANGE_{nm}",
                "reading": f"{nm}'s mean changed sign between n=5 ({pub_mean:+.6f}) and "
                           f"n={len(seeds)} ({blk['mean']:+.6f}). Reported as its own result, and the "
                           "relevant refuting branch fires regardless of the intervals."})
    out["branches"] = branches
    out["branch_labels"] = [b["branch"] for b in branches]
    return out


def _report(a):
    print(f"\n=== {D2} / {ATTACK.replace('committed_', '')}: three estimands at "
          f"n={a.get('n_paired_seeds')} ===")
    if "estimands" not in a:
        print(f"  {a.get('why')}")
        return
    for name, b in a["estimands"].items():
        v = b["verdicts"]
        ci = b["ci95"]
        print(f"  {name:11s} mean={b['mean']:+.6f} sd={b['sd']:.6f} "
              f"CI95=[{ci[0]:+.6f}, {ci[1]:+.6f}] n={b['n']}")
        print(f"              sign={v['verdict_sign']:15s} margin={v['verdict_margin']:15s} "
              f"upper={v['verdict_upper']}  (direction {v['measured_direction']})")
    print(f"  identity shared across designs: {a['identity_shared_across_designs']}, "
          f"D reduction holds: {a['D_reduction_holds']}")
    r = a["reachability_at_this_n"]
    print(f"  reachability: identity mean ASR {r['identity_mean_asr']:.4f} "
          f"(0.5186 at n=5), both legs are tests: {r['lower_leg_is_a_test']}")
    print("    NB this is NOT the 0.519 printed at main.tex:1297/:2225/:2241/:2248/:3105/:3449/:3473 "
          "-- those are n=5")
    if a["any_rung_void"]:
        for k, r in a["rungs"].items():
            if r["rung_void"]:
                print(f"  ** VOID RUNG {k}: mean accuracy {r['mean_accuracy']:.4f} < {ACC_FLOOR}")
    for b in a["branches"]:
        print(f"\n  BRANCH {b['branch']}")
        print(f"    {b['reading']}")
        for s in b.get("sites", []):
            print(f"      - {s}")


def main():
    if not check_frozen():
        return 1

    art = load_artifact()

    if "--harness-check" in sys.argv:
        ok, verdict = harness_check()
        # Persisted whether it passed or failed: a recorded failure is evidence too, and the scoring
        # run reads this field rather than a terminal.
        save(_shell(art.get("cells", {}), harness=verdict))
        print(f"Harness verdict persisted to {os.path.join(OUT, 'summary.json')} "
              f"(status {verdict['status']})")
        return 0 if ok else 1

    if "--analyze-only" in sys.argv:
        cells = art.get("cells", {})
        a = analyze(cells)
        save(_shell(cells, harness=art.get("harness_check"), extra={"analysis": a}))
        _report(a)
        return 0

    if not _harness_gate(art):
        return 1

    cells = art.get("cells", {})
    print(f"=== {D2} / {ATTACK.replace('committed_', '')} seed top-up on {DATASET} ===")
    print(f"    endpoint rungs kappa in {{{LO}, {HI}}}, seeds {NEW_SEEDS[0]}-{NEW_SEEDS[-1]}, "
          "both designs")
    print(f"    n=5 -> n=20 on BOTH legs. Rules frozen at {PREREG_COMMIT}.")
    print(f"    kappa={LO} is computed ONCE per seed and shared, so {len(NEW_SEEDS)} x 3 = "
          f"{len(NEW_SEEDS) * 3} runs, not {len(NEW_SEEDS) * 4}.")
    print("    Refuting branch 1: D's interval containing zero drops this cell's disagreement and "
          "takes\n    the paper's count from five of seven to four, in the abstract and in Figure 1.\n",
          flush=True)

    # Seed-major so that an interruption leaves COMPLETE seeds on both legs -- equal n, the only shape
    # the paired contrast can read -- rather than a long confounded leg with nothing to contrast against.
    # run_one() re-seeds torch and numpy from `seed` on entry, so no run depends on what ran before it.
    todo = [(seed, family, kappa)
            for seed in NEW_SEEDS
            for (family, kappa) in (("confounded", LO), ("confounded", HI), ("controlled", HI))]
    done = sum(len(c.get("per_seed", [])) for c in cells.values())
    print(f"  resuming: {done} rows already recorded ({len(todo)} runs planned, shared rows counted "
          "once)\n", flush=True)

    t0 = time.time()
    for i, (seed, family, kappa) in enumerate(todo, 1):
        shared = (kappa == LO)
        if _has(cells, family, kappa, seed) and (not shared or _has(cells, "controlled", LO, seed)):
            continue
        t = time.time()
        acc, asr = run_one(seed, family, D2, ATTACK, kappa, DATASET, MODEL)
        _record(cells, family, kappa, seed, acc, asr, "<computed here>")
        if shared:
            # The same computation, recorded under the controlled key so the analyzer needs no special
            # case. Labelled, not silently duplicated: `source` says which leg actually ran it.
            _record(cells, "controlled", kappa, seed, acc, asr,
                    "<shared identity rung: computed once as confounded kappa=0; "
                    "dose_kappa0.0 == doseS_kappa0.0, verified by --harness-check on this attack>")
        save(_shell(cells, harness=art.get("harness_check"), extra={"analysis": analyze(cells)}))
        print(f"  [{i}/{len(todo)}] s{seed} {family:11s} k={kappa:<4} acc={acc:.4f} ASR={asr:.4f}"
              + ("  * below acc floor" if acc < ACC_FLOOR else "")
              + ("  (shared with controlled)" if shared else "")
              + f" ({time.time() - t:.0f}s)", flush=True)

    a = analyze(cells)
    save(_shell(cells, harness=art.get("harness_check"), extra={"analysis": a}))
    _report(a)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h")
    print(f"Saved to {os.path.join(OUT, 'summary.json')}")
    print("Count `per_seed` entries in the artifact to judge progress; the [i/N] index above counts "
          "resumed-and-skipped runs and can go backwards across restarts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
