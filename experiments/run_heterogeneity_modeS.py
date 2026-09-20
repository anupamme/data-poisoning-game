"""
Does the Mode-S within-defense rise survive non-IID heterogeneity?

Pre-registration: experiments/pre_registration_heterogeneity_modeS.md, committed at the hash in
PREREG_COMMIT below, before results/heterogeneity_modeS/ existed. Read that file first. Everything
this script decides is decided there; this script is orchestration and contains no physics.

The object under test is the paper's INSTRUMENT LEG on the sign-reversal cell (d2 = coord_median,
attack = committed pixel, kappa 0 -> 2): with the adversarial coefficient pinned at c=1.0 and only
benign clients spread over rho = e^{2 kappa}, ASR RISES. Published dASR +0.098067 at n=5 (seeds
42--46) and +0.125 at n=20.

Mode S has only ever run at ONE Dirichlet concentration. run_targeted_dose.py:57 states it
verbatim: "Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn." This arm turns that one dial and holds everything else at its frozen value.

  alpha = 0.1    the leg at risk: measured accuracy 0.364 against ACC_FLOOR 0.35
  alpha = 1.0
  alpha = 10.0
  alpha = 0.5    the published anchor. NOT RE-RUN, printed as frozen beside whatever this returns.

run_one is IMPORTED from run_targeted_dose, never copied -- a copy is how two suites drift apart in
what they compute. It already takes `alpha` as a defaulted argument whose default is the frozen
0.5, so this arm adds no physics. --harness-check proves that default path unchanged BY VALUE
against the frozen artifact, and PERSISTS its verdict dict (Round 69 found the existing harness
check returns a dict its caller discards, leaving the claim witnessed only by stdout).

THE ESTIMAND IS THE WITHIN-DEFENSE RISE, per alpha, paired by seed:

  Delta_S(alpha) = ASR(doseS, kappa=2.0, alpha) - ASR(doseS, kappa=0.0, alpha)

TWO VERDICTS, FROZEN SEPARATELY, AND NEITHER MAY BE REPORTED AS THE OTHER. Round 68's regime arm
cleared its threshold in both regimes while reproducing the SIGN in only one, and the summary "the
dissociation reproduces" was therefore misleading. verdict_sign is about direction; verdict_magnitude
is an interval-overlap statement against the frozen anchor. Any mechanism reading goes in
mechanism_observed, a separate field, because Round 72's freeze fused a pass/fail label with a
mechanism guess and the guess was backwards in sign while the label was right.

THE IDENTITY RUNG IS RE-RUN AT EVERY ALPHA AND IS NEVER IMPORTED. run_targeted_dose.py:50--:55
imports doseS_kappa0.0 from results/dose_response/ because "run_one's participant RNG stream does
not depend on d1's name, so those runs are the same computation." That argument holds across d1,
NOT across alpha: alpha sets the Dirichlet partition. results/dose_response/ is not read here.

  python3 -m experiments.run_heterogeneity_modeS --harness-check   # BEFORE anything else
  python3 -m experiments.run_heterogeneity_modeS --alpha 1.0
  python3 -m experiments.run_heterogeneity_modeS --all
  python3 -m experiments.run_heterogeneity_modeS --analyze-only

Resumable: every cell is written to results/heterogeneity_modeS/summary.json as it completes, and a
completed cell is skipped. Count progress from `per_seed` entries in that file, never from the
printed [i/N], which is a todo POSITION that counts resumed-and-skipped runs and can go backwards.

Output: results/heterogeneity_modeS/summary.json. This script writes nothing else, and creates its
output directory in __main__ and never at import time -- importing it must not touch results/, or
the provenance gate ("prereg committed before the first write to the output directory") becomes
ambiguous.
"""
import argparse, json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.run_targeted_dose import (run_one as run_inst, d1_name as d1_inst, dial,
                                           ACC_FLOOR, TOL)
from experiments.analyze_headline_cis import t_crit          # noqa: E402  NOT a literal t table

# experiments/pre_registration_heterogeneity_modeS.md, committed before results/heterogeneity_modeS/
# existed. The script refuses to start otherwise: an unfrozen run would make the outcomes of that
# file's section 2 renegotiable, which is the entire point of writing them down first.
PREREG_COMMIT = "e7b7c83"              # commits PREREG alone, 195 insertions, no other file

# --- The cell, frozen. This IS the sign-reversal cell of tab:comparability. ---
D2 = "coord_median"
ATTACK = "committed_pixel"
KAPPAS = [0.0, 2.0]                    # endpoint rungs only; rho = 1.00 and 54.60
SEEDS = [42, 43, 44, 45, 46]           # n = 5, fixed now, no optional stopping (prereg section 3)
ALPHAS = [0.1, 1.0, 10.0]              # 0.5 is the published anchor and is NOT re-run
ANCHOR_ALPHA = 0.5

# The frozen anchor, READ ONLY. Not a leg of this arm and never substituted for one.
#
# WHICH ARTIFACT HOLDS THE alpha=0.5 ROWS FOR THIS CELL, checked rather than assumed.
# results/targeted_dose/ has doseS rows only into cos_krum, krum and reputation -- it has NO
# doseS_*_then_coord_median cell at all -- and results/dose_response/ carries the untargeted `dose`
# family. The published n=5 seeds 42--46 rows for THIS cell are in results/dose_replication/, whose
# own config records alpha 0.5, 50 rounds, acc_floor 0.35; the n=20 top-up's seeds 47--61 are in
# results/reversal_seed_topup/. Pointing the harness check at targeted_dose would have returned
# INDETERMINATE and looked like a missing artifact rather than a wrong path.
FROZEN_REPLICATION = os.path.join(base, "results", "dose_replication", "summary.json")
FROZEN_REVERSAL = os.path.join(base, "results", "reversal_seed_topup", "summary.json")

# NOTE: created in __main__, not at import time. See the module docstring.
out_dir = os.path.join(base, "results", "heterogeneity_modeS")
out_path = os.path.join(out_dir, "summary.json")


def cell_key(alpha, kappa):
    return f"alpha{alpha:g}|{d1_inst('S', kappa)}_then_{D2}|{ATTACK}"


def run_leg(alpha, kappa, seed):
    """One 50-round run. The ONLY place run_one is called for a leg of this arm."""
    return run_inst(seed, "S", D2, ATTACK, kappa, alpha=alpha)


# --------------------------------------------------------------------------- checkpoint store
# The load_or_init / has_run / save_one idiom is reused in shape from
# run_regime_dissociation.py:131-205: re-read the file, mutate, write. Slower than holding state in
# memory and that is the point -- an interrupted run loses at most the cell in flight.

def load_or_init():
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {
        "description": "Does the Mode-S within-defense rise on coord_median x committed pixel "
                       "survive non-IID heterogeneity? Estimand Delta_S(alpha) = ASR(kappa=2) - "
                       "ASR(kappa=0), paired by seed, per alpha. Two verdicts frozen separately: "
                       "sign and magnitude.",
        "prereg": "experiments/pre_registration_heterogeneity_modeS.md",
        "prereg_commit": PREREG_COMMIT,
        "cell": {"d2": D2, "attack": ATTACK, "kappas": KAPPAS, "seeds": SEEDS,
                 "alphas": ALPHAS, "anchor_alpha": ANCHOR_ALPHA,
                 "dials_instrument": {str(k): dial("S", k) for k in KAPPAS},
                 "acc_floor": ACC_FLOOR,
                 "config": "N=10, K=5, f=0.2, 50 rounds, cifar_cnn -- frozen; only alpha moves"},
        "endpoint_only": "kappa in {0, 2} only. These two rungs must never be displayed or "
                         "described as a four-rung ladder and no trend statistic is computed here.",
        "identity_rung_provenance": "kappa=0 is COMPUTED FRESH AT EVERY ALPHA and is never "
                                    "imported. run_targeted_dose.py imports it from "
                                    "results/dose_response/ on the ground that run_one's "
                                    "participant RNG stream does not depend on d1's name; that "
                                    "argument holds across d1 and NOT across alpha, which sets the "
                                    "Dirichlet partition. results/dose_response/ is not read here.",
        "equivalence_claim": "NONE. EQUIV_MARGIN is deliberately not imported. The magnitude "
                            "verdict is an interval-overlap statement against the frozen anchor, "
                            "not the phrase defined at supplementary.tex:441.",
        "harness_check": {},
        "frozen_anchor": {},
        "cells": {},
        "analysis": {},
    }


def has_run(key, seed):
    c = load_or_init()["cells"].get(key)
    return bool(c) and any(r["seed"] == seed for r in c["per_seed"])


def save_one(key, alpha, kappa, seed, acc, asr):
    d = load_or_init()
    c = d["cells"].setdefault(key, {"d2": D2, "attack": ATTACK, "alpha": alpha, "kappa": kappa,
                                    "rho": dial("S", kappa), "per_seed": []})
    c["per_seed"] = [r for r in c["per_seed"] if r["seed"] != seed]
    c["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr,
                          "below_acc_floor": bool(acc < ACC_FLOOR)})
    c["per_seed"].sort(key=lambda r: r["seed"])
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)


def save_top(field, value):
    d = load_or_init()
    d[field] = value
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)


def rows(key):
    """{seed: (accuracy, asr)} for one stored cell, or {}."""
    c = load_or_init()["cells"].get(key)
    return {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]} if c else {}


# --------------------------------------------------------------------------- frozen artifact reads

def frozen_row(path, d1, seed):
    """(accuracy, asr) for one seed of `d1`_then_coord_median|committed_pixel. Read, never written.

    Returns None if absent rather than inventing a value.
    """
    if not os.path.exists(path):
        return None
    cells = json.load(open(path)).get("cells", {})
    c = cells.get(f"{d1}_then_{D2}|{ATTACK}")
    if not c:
        return None
    for r in c["per_seed"]:
        if r["seed"] == seed:
            return (r["accuracy"], r["asr"])
    return None


def frozen_anchor():
    """The published alpha=0.5 n=5 instrument Delta. READ ONLY, never a leg of this arm.

    RECOMPUTED from the per-seed rows of results/dose_replication/, not transcribed from a summary
    key, and then CROSS-CHECKED against results/reversal_seed_topup/'s published_n5_verdict. Two
    sources that must agree is the only way to notice a summary key that has drifted from the rows
    beneath it; if they disagree, both are returned and the disagreement is the finding.
    """
    per = {}
    if os.path.exists(FROZEN_REPLICATION):
        cells = json.load(open(FROZEN_REPLICATION)).get("cells", {})
        for k in KAPPAS:
            c = cells.get(f"{d1_inst('S', k)}_then_{D2}|{ATTACK}")
            if c:
                per[k] = {r["seed"]: r["asr"] for r in c["per_seed"]}
    out = {"alpha": ANCHOR_ALPHA, "source": "results/dose_replication/summary.json, recomputed "
           "from per_seed rows"}
    if len(per) == len(KAPPAS):
        shared = sorted(set(per[0.0]) & set(per[2.0]) & set(SEEDS))
        d = np.array([per[2.0][s] - per[0.0][s] for s in shared], dtype=float)
        if len(d) >= 2:
            hw = float(t_crit(len(d)) * d.std(ddof=1) / np.sqrt(len(d)))
            out.update({"seeds": shared, "n": len(shared),
                        "per_seed_delta": {str(s): float(x) for s, x in zip(shared, d)},
                        "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                        "ci95": [float(d.mean() - hw), float(d.mean() + hw)]})

    if os.path.exists(FROZEN_REVERSAL):
        v = json.load(open(FROZEN_REVERSAL)).get("published_n5_verdict", {}).get("controlled")
        if v:
            out["published_summary_key"] = dict(
                v, source="results/reversal_seed_topup/summary.json:published_n5_verdict.controlled")
            if "mean" in out:
                # Agreement is judged on (mean, sd), the two quantities both sources measure. The
                # CIs differ in the 5th decimal because the published one used the rounded t=2.776
                # the pre-registration quotes while t_crit() calls scipy for 2.7764451..., so a CI
                # comparison here would report a drift that is a t-precision artifact.
                agree = (abs(out["mean"] - v["mean"]) <= 1e-5
                         and abs(out["sd"] - v["sd"]) <= 1e-4)
                out["sources_agree"] = bool(agree)
                out["ci95_note"] = ("Recomputed with scipy's t (2.7764451); the published key used "
                                    "the rounded 2.776, so the two CIs differ in the 5th decimal. "
                                    "Agreement is judged on mean and sd.")
                if not agree:
                    out["DISAGREEMENT"] = ("The recomputed Delta and the published summary key do "
                                          "not match. Neither is used until that is resolved.")
    return out if "mean" in out or "published_summary_key" in out else None


# --------------------------------------------------------------------------- gates

def check_frozen():
    """Refuse to run unless the prereg is committed AT PREREG_COMMIT and its working tree is clean.

    The `is None` test alone would make PREREG_COMMIT a LABEL rather than a check once it is set: this
    runner stamps the constant into every artifact, so the literal here and a different document on
    disk would agree with each other and disagree with the freeze. Two further checks are needed --
    the hash, because the constant could name a commit that never touched this file, and the working
    tree, because `git log -1` reports the last commit that TOUCHED the file and is wholly unchanged
    by uncommitted edits to it. Same standard as measure_admission_normclip_cifar100.py:145.
    """
    rel = "experiments/pre_registration_heterogeneity_modeS.md"
    prereg = os.path.join(base, rel)
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {rel} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the outcomes are not frozen.\n"
                 f"  1. git commit {rel}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the outcomes renegotiable, which is the entire point.")
    log = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel],
                         cwd=base, capture_output=True, text=True, timeout=20)
    actual = log.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        sys.exit(f"REFUSING TO RUN: {rel} was last touched at {actual or 'UNTRACKED'}, but "
                 f"PREREG_COMMIT is {PREREG_COMMIT}. The outcomes must be frozen before they exist.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel],
                           cwd=base, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        sys.exit(f"REFUSING TO RUN: {rel} has uncommitted changes "
                 f"({dirty.stdout.strip().split()[0]}), so it is NOT frozen at {actual} whatever "
                 "`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
    print(f"[OK] {rel} frozen at {actual}, working tree clean")


def assert_identity_dial():
    """kappa=0 must be the EXACT identity, tested with ==, not a tolerance (prereg section 4.3).

    This is the guard for the failure mode in which a manipulation hook is never called and the run
    passes silently while an unchanged number is reported as a finding.
    """
    d0 = dial("S", 0.0)
    if d0 != 1.0:
        sys.exit(f"REFUSING TO RUN: dial('S', 0.0) = {d0!r}, not exactly 1.0. The kappa=0 rung is "
                 "not the identity, so no leg of this arm measures what it claims to.")
    return {"dial_S_kappa0": d0, "exactly_one": True,
            "note": "tested with ==, not a tolerance. Per-client coefficient and channel "
                    "displacement identity at every alpha is asserted by "
                    "experiments/measure_admission_heterogeneity.py."}


def harness_check():
    """Prove BY VALUE that passing alpha explicitly changed nothing on the frozen path.

    One run at the FROZEN alpha (0.5), seed 42, kappa=0, compared to its stored row in
    results/dose_replication/summary.json -- NOT results/targeted_dose/, which carries no
    doseS_*_then_coord_median cell at all and would return INDETERMINATE, reading as a missing
    artifact rather than as a wrong path. If the values differ, this runner's call path into run_one
    is not the frozen one and no new alpha may be run.

    The verdict dict is PERSISTED. Round 69 established that the existing --harness-check returns a
    dict its caller discards, so the paper sentence asserting it was witnessed only by stdout.
    """
    print("=" * 78)
    print("HARNESS CHECK -- by-value proof of the frozen path (prereg section 4.5)")
    print("=" * 78)
    ident = assert_identity_dial()
    print(f"  dial('S', 0.0) == 1.0 exactly: {ident['exactly_one']}")

    d1 = d1_inst("S", 0.0)
    ref = frozen_row(FROZEN_REPLICATION, d1, 42)
    if ref is None:
        v = {"verdict": "INDETERMINATE", "reason": f"no frozen row for {d1} seed 42 in "
             f"{os.path.relpath(FROZEN_REPLICATION, base)}", "identity_dial": ident}
        print(f"  {v['verdict']}: {v['reason']}")
        return v

    t0 = time.time()
    acc, asr = run_leg(ANCHOR_ALPHA, 0.0, 42)
    dt = time.time() - t0
    dacc, dasr = abs(acc - ref[0]), abs(asr - ref[1])
    ok = dacc <= TOL and dasr <= TOL
    v = {"verdict": "PASS" if ok else "FAIL",
         "what": "doseS_kappa0.0 @ seed 42, alpha=0.5 (the frozen default), run through this "
                 "script's run_leg and compared BY VALUE to the frozen artifact",
         "reference": os.path.relpath(FROZEN_REPLICATION, base),
         "recomputed": {"accuracy": acc, "asr": asr},
         "frozen": {"accuracy": ref[0], "asr": ref[1]},
         "abs_diff": {"accuracy": dacc, "asr": dasr}, "tol": TOL,
         "seconds": round(dt, 1), "identity_dial": ident}
    print(f"  recomputed acc={acc:.4f} asr={asr:.4f}")
    print(f"  frozen     acc={ref[0]:.4f} asr={ref[1]:.4f}")
    print(f"  |diff|     acc={dacc:.2e} asr={dasr:.2e}   tol={TOL:.0e}")
    print(f"  {v['verdict']} in {dt:.0f}s")
    if not ok:
        print("  The alpha argument is not inert on its default. DO NOT RUN THIS ARM.")
    return v


def acc_gate(alpha):
    """Prereg section 4.1. Mean clean accuracy >= ACC_FLOOR at EVERY rung, or the alpha is VOID."""
    per_rung = {}
    for k in KAPPAS:
        r = rows(cell_key(alpha, k))
        if not r:
            return {"verdict": "INCOMPLETE", "per_rung": per_rung}
        accs = [a for a, _ in r.values()]
        per_rung[str(k)] = {"mean_accuracy": float(np.mean(accs)), "n": len(accs),
                            "min_accuracy": float(min(accs)),
                            "passes": bool(np.mean(accs) >= ACC_FLOOR)}
    ok = all(v["passes"] for v in per_rung.values())
    return {"verdict": "PASS" if ok else "VOID", "acc_floor": ACC_FLOOR, "per_rung": per_rung,
            "note": "" if ok else "A rung is below the accuracy floor, so this alpha is "
                    "uninterpretable and NO verdict stands for it. A low ASR at collapsed accuracy "
                    "is not suppression. Reported as VOID beside the alphas that were not."}


# --------------------------------------------------------------------------- the run loop

def run_alpha(alpha):
    check_frozen()
    assert_identity_dial()
    todo = [(k, s) for k in KAPPAS for s in SEEDS if not has_run(cell_key(alpha, k), s)]
    print(f"\n=== alpha={alpha:g}: {len(todo)} run(s) to do "
          f"({len(KAPPAS) * len(SEEDS)} in the grid) ===")
    print("    [i/N] below is a todo POSITION, not progress. Count per_seed in the artifact.")
    for i, (k, s) in enumerate(todo, 1):
        key = cell_key(alpha, k)
        t0 = time.time()
        acc, asr = run_leg(alpha, k, s)
        save_one(key, alpha, k, s, acc, asr)
        flag = "  BELOW ACC FLOOR" if acc < ACC_FLOOR else ""
        print(f"  [{i}/{len(todo)}] alpha={alpha:g} kappa={k} seed={s}: "
              f"acc={acc:.4f} asr={asr:.4f} ({time.time() - t0:.0f}s){flag}")
    save_top("frozen_anchor", frozen_anchor())


# --------------------------------------------------------------------------- analysis

ADMISSION_ARTIFACT = "results/heterogeneity_modeS_admission.json"


def premise_verdict(alpha):
    """The per-alpha Mode-S premise, READ from the admission artifact rather than restated.

    This field used to say "until measure_admission_heterogeneity.py reads CONSTANT at this
    alpha, this row carries no Mode-S reading" -- a condition the reader had to go and resolve
    by hand, in a file that by then had already resolved it. The hazard is one this repository
    has hit before: a sentence that was true when written and false once the sibling artifact
    landed, with nothing on disk recording the change. So read it, and record where it was read
    from and what it said verbatim.

    Returns a dict, never a bare string, and never PASS by default: a missing or unparseable
    artifact reads UNMEASURED, which is what a reader should see when nothing measured it.
    """
    rec = {"source": ADMISSION_ARTIFACT,
           "scope": ("A premise, not a result. It says the adversary is pinned and the identity "
                     "rung displaces nothing at this alpha; it is not evidence for or against "
                     "Delta_S(alpha), and it asserts no mechanism for the rise.")}
    if not os.path.exists(ADMISSION_ARTIFACT):
        rec["verdict"] = "UNMEASURED (admission artifact absent)"
        return rec
    try:
        with open(ADMISSION_ARTIFACT) as fh:
            adm = json.load(fh)
    except (ValueError, OSError) as exc:
        rec["verdict"] = f"UNMEASURED (admission artifact unreadable: {exc})"
        return rec
    if adm.get("prereg_commit") != PREREG_COMMIT:
        rec["verdict"] = ("UNMEASURED (admission artifact stamped %r, this runner is frozen at %r)"
                          % (adm.get("prereg_commit"), PREREG_COMMIT))
        return rec
    cell = (adm.get("per_alpha") or {}).get(f"alpha{alpha:g}")
    if not cell or "verdict" not in cell:
        rec["verdict"] = "UNMEASURED (no row for this alpha in the admission artifact)"
        return rec
    rec["verdict"] = cell["verdict"]
    rec["identity_gate"] = (cell.get("identity_gate") or {}).get("verdict")
    rec["share_gate"] = (cell.get("share_gate") or {}).get("verdict")
    rec["holds"] = cell["verdict"].startswith("MODE-S PREMISE HOLDS")
    return rec


def analyze():
    """Both verdicts, per alpha, recomputed from per-seed rows. Nothing is transcribed."""
    anchor = frozen_anchor()
    # The magnitude verdict compares intervals, so it needs an interval. A partially-readable anchor
    # (recompute failed, or the two sources disagree) yields None here rather than a number, and every
    # magnitude verdict below then reads INDETERMINATE instead of being scored against half an anchor.
    a_ci = None
    if anchor and "ci95" in anchor and anchor.get("DISAGREEMENT") is None:
        a_ci = anchor["ci95"]
    out = {"anchor": anchor, "anchor_ci95_used": a_ci, "per_alpha": {}}
    print("\n" + "=" * 78)
    print("Delta_S(alpha) = ASR(kappa=2) - ASR(kappa=0), paired by seed")
    print("=" * 78)
    if anchor and "mean" in anchor:
        print(f"  FROZEN anchor alpha={ANCHOR_ALPHA}: mean {anchor['mean']:+.6f} "
              f"sd {anchor['sd']:.6f} ci95 [{anchor['ci95'][0]:+.6f}, {anchor['ci95'][1]:+.6f}] "
              f"n={anchor['n']}")
        print(f"  recomputed from {anchor['source']}; two-source agreement: "
              f"{anchor.get('sources_agree')}")
        print("  (printed as frozen, not re-run, never pooled with the rows below)")
    else:
        print(f"  FROZEN anchor alpha={ANCHOR_ALPHA}: NOT READABLE. Every magnitude verdict below "
              "will read INDETERMINATE.")

    for alpha in ALPHAS:
        r0, r2 = rows(cell_key(alpha, 0.0)), rows(cell_key(alpha, 2.0))
        shared = sorted(set(r0) & set(r2))
        gate = acc_gate(alpha)
        rec = {"n": len(shared), "seeds": shared, "acc_gate": gate}

        if len(shared) < 2:
            rec["verdict_sign"] = rec["verdict_magnitude"] = "INCOMPLETE"
            rec["note"] = (f"{len(shared)} paired seed(s) present of {len(SEEDS)}; "
                           "both legs of every Delta must come from the same seed.")
        else:
            d = np.array([r2[s][1] - r0[s][1] for s in shared], dtype=float)
            mean, sd = float(d.mean()), float(d.std(ddof=1))
            hw = float(t_crit(len(d)) * sd / np.sqrt(len(d)))
            lo, hi = mean - hw, mean + hw
            rec.update({"per_seed_delta": {str(s): float(x) for s, x in zip(shared, d)},
                        "mean": mean, "sd": sd, "ci95": [lo, hi],
                        "half_width": hw, "t_crit": float(t_crit(len(d)))})

            if gate["verdict"] == "VOID":
                rec["verdict_sign"] = rec["verdict_magnitude"] = "VOID (accuracy gate)"
            else:
                if lo > 0:
                    rec["verdict_sign"] = "THE WITHIN-DEFENSE RISE REPRODUCES AT THIS ALPHA"
                elif hi < 0:
                    rec["verdict_sign"] = "REVERSED AT THIS ALPHA"
                else:
                    rec["verdict_sign"] = "NOT RESOLVABLE AT THIS ALPHA AND n=5"
                if a_ci is not None:
                    a_lo, a_hi = a_ci
                    overlap = (lo <= a_hi) and (a_lo <= hi)
                    rec["verdict_magnitude"] = ("CONSISTENT IN MAGNITUDE WITH alpha=0.5" if overlap
                                                else "DIFFERS IN MAGNITUDE FROM alpha=0.5")
                    rec["anchor_ci95"] = [a_lo, a_hi]
                    rec["magnitude_is_not_equivalence"] = (
                        "An interval-overlap statement. NOT the phrase defined at "
                        "supplementary.tex:441, and EQUIV_MARGIN is not imported by this arm.")
                else:
                    rec["verdict_magnitude"] = "INDETERMINATE (anchor unreadable)"

            rec["premise"] = premise_verdict(alpha)

        out["per_alpha"][f"alpha{alpha:g}"] = rec
        print(f"\n  alpha={alpha:g}  n={rec['n']}  acc gate {gate['verdict']}")
        if "mean" in rec:
            print(f"    Delta_S = {rec['mean']:+.6f}  sd {rec['sd']:.6f}  "
                  f"ci95 [{rec['ci95'][0]:+.6f}, {rec['ci95'][1]:+.6f}]")
        print(f"    SIGN      : {rec['verdict_sign']}")
        print(f"    MAGNITUDE : {rec['verdict_magnitude']}")
        if "premise" in rec:
            print(f"    PREMISE   : {rec['premise']['verdict'].split(':')[0]}"
                  f"   (read from {rec['premise']['source']})")

    print("\n  Neither verdict may be reported as the other, and no sentence may claim "
          "heterogeneity-robustness\n  from a subset of the grid (prereg section 5).")
    save_top("analysis", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-check", action="store_true")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    # ORDER IS LOAD-BEARING: the freeze guard runs BEFORE the output directory is created. The
    # pre-registration asserts that results/heterogeneity_modeS/ does not exist at the time it is
    # committed, and a makedirs above this line falsifies that assertion on the first --help-adjacent
    # invocation, silently, with no run attached to it.
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)

    if a.harness_check:
        save_top("harness_check", harness_check())
    elif a.analyze_only:
        analyze()
    elif a.alpha is not None:
        if a.alpha not in ALPHAS:
            sys.exit(f"REFUSING TO RUN: alpha={a.alpha} is not in the frozen grid {ALPHAS}. "
                     f"alpha={ANCHOR_ALPHA} is the published anchor and is not re-run.")
        run_alpha(a.alpha)
        analyze()
    elif a.all:
        for alpha in ALPHAS:
            run_alpha(alpha)
        analyze()
    else:
        ap.print_help()
