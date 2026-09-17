"""
Does the DESIGN DISAGREEMENT survive an adaptive adversary, and survive scale?

Pre-registration: experiments/pre_registration_regime_dissociation.md, committed at the hash in
PREREG_COMMIT below, before results/regime_dissociation/ existed. Read that file first. Everything
this script decides is decided there; this script is orchestration and contains no physics.

The object under test is the paper's flagship dissociation, which is a DISAGREEMENT BETWEEN TWO
EVALUATION DESIGNS on one cell (d2 = coord_median, attack = committed pixel, kappa 0 -> 2):

  confounded ladder  dose_kappa   (experiments/run_dose_response.py)   published dASR  -0.273
  instrument         doseS_kappa  (experiments/run_targeted_dose.py)   published dASR  +0.125

so reproducing it means running BOTH ladders' kappa=2 legs, not one ladder. Both run_one functions
are IMPORTED, never copied -- a copy is how two suites drift apart in what they compute -- and they
were given three defaulted arguments (fl_config, alpha, ca) whose defaults are the frozen
configuration. --harness-check proves that default path unchanged BY VALUE against the frozen
artifacts; an md5 is not accepted as that proof.

  Regime A   N=10, K=5, alpha=0.5  + criterion-aware adversary (the frozen regime, new adversary)
  Regime B   N=100, K=20, alpha=0.5                            (the frozen adversary, new scale)

THE PRIMARY ESTIMAND IS THE DISAGREEMENT ITSELF, paired by seed:

  D_s = dASR_confounded(s) - dASR_instrument(s) = ASR_conf(kappa2, s) - ASR_inst(kappa2, s)

The kappa=0 leg cancels exactly because the two designs SHARE that rung: at kappa=0 both d1
transforms return the update list unwrapped, so both rungs are coord_median alone. That is measured
here (the identity gate), not assumed, exactly as results/reversal_seed_topup/summary.json's
identity_rung_provenance does it. Neither dASR alone is powered to exclude zero at n=3 and the
pre-registration says so with the measured variances; only D is tested. No equivalence claim is made
anywhere in this arm, and EQUIV_MARGIN is deliberately not imported.

  python3 -m experiments.run_regime_dissociation --harness-check   # BEFORE anything else
  python3 -m experiments.run_regime_dissociation --regime A
  python3 -m experiments.run_regime_dissociation --regime B
  python3 -m experiments.run_regime_dissociation --analyze-only

Resumable: every cell is written to results/regime_dissociation/summary.json as it completes, and a
completed cell is skipped. Count progress from `per_seed` entries in that file, never from the
printed [i/N], which is a todo POSITION that counts resumed-and-skipped runs.

Output: results/regime_dissociation/summary.json. This script writes nothing else, and creates its
output directory in __main__ and never at import time -- importing it must not touch results/, or
the provenance gate ("prereg committed before the first write to the output directory") becomes
ambiguous.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from config import FLConfig
from experiments.run_dose_response import run_one as run_conf, d1_name as d1_conf, rho
from experiments.run_targeted_dose import (run_one as run_inst, d1_name as d1_inst, dial,
                                           ACC_FLOOR, TOL)
from experiments.analyze_headline_cis import t_crit          # noqa: E402  NOT a literal t table

# experiments/pre_registration_regime_dissociation.md, committed before results/regime_dissociation/
# existed. The script refuses to start otherwise: an unfrozen run would make the outcomes of that
# file's section 2 renegotiable, which is the entire point of writing them down first.
#
# PREREG_COMMIT is the state of that document that GOVERNS this run, which is the amended one --
# Amendment 1 corrected section 6.4's hook assertion, before any result existed. FREEZE_COMMIT is
# the original freeze, recorded so the pre-amendment text is recoverable and the amendment is
# visible as an edit rather than folded in silently. Both go in the artifact. This follows
# results/reversal_seed_topup/summary.json, which records its amendment's hash and not its first
# freeze's.
PREREG_COMMIT = "f4c3fec"
FREEZE_COMMIT = "b317af0"

# --- The cell, frozen. Identical in both regimes; this IS tab:comparability's cell. ---
D2 = "coord_median"
ATTACK = "committed_pixel"
KAPPAS = [0.0, 2.0]                    # endpoint rungs only; rho = 1.00 and 54.60
SEEDS = [42, 43, 44]                   # n = 3, fixed now, no optional stopping (prereg section 5)

# --- The two regimes. Nothing varies inside a regime except kappa and the ladder. ---
REGIMES = {
    "A_adaptive": {"fl_config": FLConfig(num_clients=10, clients_per_round=5, num_rounds=50),
                   "alpha": 0.5, "adaptive": True,
                   "what_changed": "adversary (criterion-aware, calibrated to coord_median)"},
    "B_scale":    {"fl_config": FLConfig(num_clients=100, clients_per_round=20, num_rounds=50),
                   "alpha": 0.5, "adaptive": False,
                   "what_changed": "N and K (100 clients, 20 per round = 20% participation)"},
}

# Regime A's calibration grid (prereg section 3): eps AND decorrelate, at kappa=0, SEED 42 ONLY,
# argmax mean ASR, then FROZEN for every rung and seed. decorrelate is swept rather than inherited
# because it is an ANTI-FOOLSGOLD device (run_criterion_aware_adversary.py:64) and the downstream
# defense here is coord_median; importing decorrelate=True because it was the argmax against
# FoolsGold->RFA would import a knob tuned against a different defense.
CALIB = [(eps, dec) for eps in (1.0, 2.0, 4.0) for dec in (True, False)]
CALIB_SEED = 42

# Frozen artifacts, READ ONLY, used for the by-value proof of the default path and for the
# calibration gate's reference. Never written, never substituted for a leg of this arm.
FROZEN_LADDER1 = os.path.join(base, "results", "dose_response", "summary.json")
FROZEN_REVERSAL = os.path.join(base, "results", "reversal_seed_topup", "summary.json")

# NOTE: created in __main__, not at import time. See the module docstring.
out_dir = os.path.join(base, "results", "regime_dissociation")
out_path = os.path.join(out_dir, "summary.json")


def calib_label(eps, dec):
    return f"ca_eps{eps:g}_{'decorr' if dec else 'nodecorr'}"


def cell_key(regime, ladder, kappa):
    """`ladder` is "conf" or "inst"; the d1 string comes from the runner that owns it."""
    d1 = d1_conf(kappa) if ladder == "conf" else d1_inst("S", kappa)
    return f"{regime}|{d1}_then_{D2}|{ATTACK}"


def run_leg(regime, ladder, kappa, seed, ca):
    """One 50-round run. The ONLY place either run_one is called for a leg of this arm."""
    r = REGIMES[regime]
    if ladder == "conf":
        return run_conf(seed, D2, ATTACK, kappa,
                        fl_config=r["fl_config"], alpha=r["alpha"], ca=ca)
    return run_inst(seed, "S", D2, ATTACK, kappa,
                    fl_config=r["fl_config"], alpha=r["alpha"], ca=ca)


# --------------------------------------------------------------------------- checkpoint store
# The load_or_init / has_run / save_one idiom is reused verbatim in shape from
# run_criterion_aware_adversary.py:79-112: re-read the file, mutate, write. Slower than holding
# state in memory and that is the point -- an interrupted run loses at most the cell in flight.

def load_or_init():
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {
        "description": "Does the design disagreement on coord_median x committed pixel survive an "
                       "adaptive adversary (regime A) and survive scale (regime B)? Primary "
                       "estimand D = dASR(confounded) - dASR(instrument), paired by seed.",
        "prereg": "experiments/pre_registration_regime_dissociation.md",
        "prereg_commit": PREREG_COMMIT,
        "prereg_freeze_commit": FREEZE_COMMIT,
        "prereg_amendments": "Amendment 1 (at prereg_commit) corrected section 6.4's hook "
                             "assertion to the attack-conditional form, before any result existed; "
                             "the pre-amendment text is at prereg_freeze_commit.",
        "cell": {"d2": D2, "attack": ATTACK, "kappas": KAPPAS, "seeds": SEEDS,
                 "rhos": {str(k): rho(k) for k in KAPPAS},
                 "dials_instrument": {str(k): dial("S", k) for k in KAPPAS},
                 "acc_floor": ACC_FLOOR},
        "regimes": {k: {"num_clients": v["fl_config"].num_clients,
                        "clients_per_round": v["fl_config"].clients_per_round,
                        "num_rounds": v["fl_config"].num_rounds,
                        "alpha": v["alpha"], "adaptive": v["adaptive"],
                        "what_changed": v["what_changed"]} for k, v in REGIMES.items()},
        "endpoint_only": "kappa in {0, 2} only. These two rungs must never be displayed or "
                         "described as a four-rung ladder and no trend statistic is computed here "
                         "(prereg section 5).",
        "identity_rung_provenance": "kappa=0 is COMPUTED ONCE per regime per seed, under the "
                                    "confounded ladder's d1 string, and SHARED by both legs -- at "
                                    "kappa=0 both d1 transforms return the update list unwrapped, "
                                    "so both rungs are coord_median alone. The instrument's "
                                    "kappa=0 rung is run at one seed per regime and asserted equal "
                                    "to TOL as a gate (prereg section 4.3); it is proof, not "
                                    "assumption.",
        "calibration": {},
        "calibration_winner": {},
        "identity_gate": {},
        "harness_check": {},
        "cells": {},
    }


def has_run(key, seed):
    s = load_or_init()
    return any(r["seed"] == seed for r in s["cells"].get(key, {}).get("per_seed", []))


def save_one(key, regime, ladder, kappa, seed, acc, asr, provenance=None):
    s = load_or_init()
    if key not in s["cells"]:
        s["cells"][key] = {"regime": regime, "ladder": ladder, "d2": D2, "attack": ATTACK,
                           "kappa": kappa,
                           "shared_by_both_ladders": bool(kappa == 0.0 and ladder == "conf"),
                           "per_seed": []}
    ps = [r for r in s["cells"][key]["per_seed"] if r["seed"] != seed]
    row = {"seed": seed, "accuracy": float(acc), "asr": float(asr)}
    if provenance:
        row["provenance"] = provenance
    ps.append(row)
    ps.sort(key=lambda r: r["seed"])
    c = s["cells"][key]
    c["per_seed"] = ps
    c["mean_asr"] = float(np.mean([r["asr"] for r in ps]))
    c["mean_accuracy"] = float(np.mean([r["accuracy"] for r in ps]))
    c["n_below_acc_floor"] = int(sum(r["accuracy"] < ACC_FLOOR for r in ps))
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)


def save_top(field, value):
    s = load_or_init()
    s[field] = value
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)


def rows(key):
    """{seed: (accuracy, asr)} for one stored cell, or {}."""
    c = load_or_init()["cells"].get(key)
    return {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]} if c else {}


# --------------------------------------------------------------------------- frozen artifact reads

def frozen_row(path, d1, seed):
    """(accuracy, asr) for one seed of `d1`_then_coord_median|committed_pixel in a frozen artifact.

    Read, never written. Used for the by-value proof of the default path (prereg section 6.3) and
    for the calibration gate's plain-backdoor reference (section 4.1). Returns None if absent
    rather than inventing a value.
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


# --------------------------------------------------------------------------- gates

def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_regime_dissociation.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the outcomes are not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the outcomes renegotiable, which is the entire point.")


def harness_check():
    """Prove BY VALUE that the three new arguments changed nothing, and that kappa=0 is shared.

    Three runs, all with DEFAULT arguments, each compared to its stored row in a frozen artifact:

      1. dose_kappa0.0  @ seed 42 vs results/dose_response/summary.json
         -- also the plain-backdoor reference the calibration gate (4.1) scores against.
      2. dose_kappa0.0  @ seed 47 vs results/reversal_seed_topup/summary.json
      3. doseS_kappa0.0 @ seed 47 vs results/reversal_seed_topup/summary.json

    Runs 2 and 3 are the same seed in the same regime under the two ladders, so together they also
    re-prove the frozen-regime identity-rung equality that this arm's own identity gate then proves
    in each new regime.

    This is a VALUE comparison, and it is not decoration. A composition arm in this repo once came
    out bit-identical to another because a manipulation hook was silently never called; an md5 of a
    results/ file would have passed that. TOL is imported, not restated.
    """
    print("=== HARNESS CHECK: the three new arguments are bit-identical no-ops ===")
    print(f"    all runs use DEFAULT arguments; tolerance {TOL:g}\n", flush=True)
    checks = [("conf", 42, FROZEN_LADDER1, "results/dose_response"),
              ("conf", 47, FROZEN_REVERSAL, "results/reversal_seed_topup"),
              ("inst", 47, FROZEN_REVERSAL, "results/reversal_seed_topup")]
    worst, bad, got, rec = 0.0, [], {}, {}
    for ladder, seed, path, short in checks:
        d1 = d1_conf(0.0) if ladder == "conf" else d1_inst("S", 0.0)
        pub = frozen_row(path, d1, seed)
        t = time.time()
        # Default arguments on purpose: fl_config=None, alpha=0.5, ca=None.
        acc, asr = (run_conf(seed, D2, ATTACK, 0.0) if ladder == "conf"
                    else run_inst(seed, "S", D2, ATTACK, 0.0))
        got[(ladder, seed)] = (acc, asr)
        if pub is None:
            print(f"  {d1:15s} s{seed}: acc={acc:.4f} ASR={asr:.4f}   NOT IN {short} "
                  f"-- cannot check", flush=True)
            bad.append(f"{d1} s{seed}: no stored row in {short}")
            continue
        d_acc, d_asr = acc - pub[0], asr - pub[1]
        ok = max(abs(d_acc), abs(d_asr)) <= TOL
        worst = max(worst, abs(d_acc), abs(d_asr))
        rec[f"{d1}|s{seed}|{short}"] = {"acc": float(acc), "asr": float(asr),
                                        "stored_acc": pub[0], "stored_asr": pub[1],
                                        "d_acc": float(d_acc), "d_asr": float(d_asr),
                                        "ok": bool(ok)}
        if not ok:
            bad.append(f"{d1} s{seed}: dACC={d_acc:+.6f} dASR={d_asr:+.6f} vs {short}")
        print(f"  {d1:15s} s{seed}: acc={acc:.4f} ASR={asr:.4f}   {short} stored: "
              f"acc={pub[0]:.4f} ASR={pub[1]:.4f}   d=({d_acc:+.6f}, {d_asr:+.6f})  "
              f"{'OK' if ok else 'MISMATCH'}  ({time.time()-t:.0f}s)", flush=True)

    if ("conf", 47) in got and ("inst", 47) in got:
        a, b = got[("conf", 47)], got[("inst", 47)]
        d_acc, d_asr = b[0] - a[0], b[1] - a[1]
        ok = max(abs(d_acc), abs(d_asr)) <= TOL
        worst = max(worst, abs(d_acc), abs(d_asr))
        rec["frozen_regime_identity_s47"] = {"d_acc": float(d_acc), "d_asr": float(d_asr),
                                             "ok": bool(ok)}
        if not ok:
            bad.append(f"frozen-regime identity s47: dACC={d_acc:+.6f} dASR={d_asr:+.6f}")
        print(f"\n  identity rung, frozen regime, s47: doseS - dose = ({d_acc:+.6f}, {d_asr:+.6f})"
              f"  {'OK' if ok else 'MISMATCH'}")

    print(f"\n  largest absolute deviation: {worst:.2e}  (tolerance {TOL:g})")
    if bad:
        print("  FAILED -- the new arguments are NOT a no-op, or the ladders do not share kappa=0:")
        for b in bad:
            print(f"    {b}")
        print("  Do not run this arm until this passes.")
    else:
        print("  PASS -- default path unchanged by value, kappa=0 shared in the frozen regime.")
    if os.path.exists(out_dir):
        save_top("harness_check", {"tolerance": TOL, "worst_abs_deviation": float(worst),
                                   "pass": not bad, "checks": rec})
    return not bad


def identity_gate(regime, ca):
    """Prereg 4.3: dose_kappa0.0 and doseS_kappa0.0 must agree to TOL at one seed in THIS regime.

    Run under the regime's own configuration AND its own ca, because the adaptive path is a distinct
    code path and an equality proved without it would not be a proof about regime A. The
    instrument's row is stored (one seed) so the artifact shows what was checked.
    """
    s = load_or_init()
    if s["identity_gate"].get(regime, {}).get("pass"):
        print(f"  [skip] identity gate already passed for {regime}")
        return True
    seed = SEEDS[0]
    conf = rows(cell_key(regime, "conf", 0.0)).get(seed)
    if conf is None:
        t = time.time()
        acc, asr = run_leg(regime, "conf", 0.0, seed, ca)
        save_one(cell_key(regime, "conf", 0.0), regime, "conf", 0.0, seed, acc, asr)
        conf = (acc, asr)
        print(f"  kappa=0 conf s{seed}: acc={acc:.4f} ASR={asr:.4f} "
              f"({(time.time()-t)/60:.1f}min)", flush=True)
    key_i = cell_key(regime, "inst", 0.0)
    inst = rows(key_i).get(seed)
    if inst is None:
        t = time.time()
        acc, asr = run_leg(regime, "inst", 0.0, seed, ca)
        save_one(key_i, regime, "inst", 0.0, seed, acc, asr,
                 provenance="identity gate only (prereg 4.3); NOT a leg of any Delta -- the "
                            "kappa=0 leg used by both Deltas is the conf row at this seed")
        inst = (acc, asr)
        print(f"  kappa=0 inst s{seed}: acc={acc:.4f} ASR={asr:.4f} "
              f"({(time.time()-t)/60:.1f}min)", flush=True)
    d_acc, d_asr = inst[0] - conf[0], inst[1] - conf[1]
    ok = max(abs(d_acc), abs(d_asr)) <= TOL
    save_top("identity_gate", {**load_or_init()["identity_gate"],
                               regime: {"seed": seed, "d_acc": float(d_acc), "d_asr": float(d_asr),
                                        "tolerance": TOL, "pass": bool(ok)}})
    print(f"  GATE 4.3 identity rung, {regime}, s{seed}: doseS - dose = "
          f"({d_acc:+.6f}, {d_asr:+.6f})  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  FAIL -- the two designs do not share an anchor in this regime. The arm stops "
              "here; the kappa=0 rung may not be shared and D's cancellation does not hold.")
    return ok


def calibration_gate():
    """Prereg 3 and 4.1, regime A only: calibrate the adversary, then score whether it is stronger.

    The grid is eps x decorrelate at kappa=0, seed 42 only, argmax mean ASR, then FROZEN. The
    winning run IS the kappa=0/seed-42 leg of regime A and is not recomputed -- it is copied into
    that cell with its provenance recorded.

    The gate: the argmax must reach ASR STRICTLY ABOVE the plain committed_pixel kappa=0 ASR at the
    same seed in the same regime, which is the frozen dose_response row (proved reproducible by
    --harness-check run 1). If none does, the criterion-aware construction does not strengthen the
    adversary against coord_median, regime A is VOID rather than negative, kappa=2 is not run, and
    the regime licenses NO SENTENCE ABOUT ADAPTIVITY.
    """
    ref = frozen_row(FROZEN_LADDER1, d1_conf(0.0), CALIB_SEED)
    if ref is None:
        sys.exit(f"REFUSING TO RUN regime A: no plain committed_pixel kappa=0 seed {CALIB_SEED} row "
                 f"in {FROZEN_LADDER1}, so gate 4.1 has no reference to score against.")
    print(f"\n--- Regime A calibration: eps x decorrelate at kappa=0, seed {CALIB_SEED} only ---")
    print(f"    reference (plain committed_pixel, frozen dose_response s{CALIB_SEED}): "
          f"acc={ref[0]:.4f} ASR={ref[1]:.4f}", flush=True)
    # decorrelate restricts each adversary to a disjoint coordinate block only when more than one
    # adversary is sampled in a round (run_criterion_aware_adversary.py, `if decorrelate and
    # n_adv > 1`). At N=10/K=5 with f=0.2 that is 2 of 10 clients, so both are sampled in about 22%
    # of rounds; the knob is therefore weaker here than its name suggests. This is inherited from
    # the frozen suite's own configuration, not introduced by this arm, and it is why decorrelate is
    # swept rather than assumed.
    s = load_or_init()
    for eps, dec in CALIB:
        label = calib_label(eps, dec)
        if label in s["calibration"]:
            print(f"  [skip] {label}", flush=True); continue
        t = time.time()
        acc, asr = run_leg("A_adaptive", "conf", 0.0, CALIB_SEED, (eps, dec))
        s = load_or_init()
        s["calibration"][label] = {"eps": eps, "decorrelate": dec, "seed": CALIB_SEED,
                                   "accuracy": float(acc), "asr": float(asr)}
        with open(out_path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"  {label:18s}: acc={acc:.4f} ASR={asr:.4f} ({(time.time()-t)/60:.1f}min)",
              flush=True)

    cal = load_or_init()["calibration"]
    # argmax over the grid in its declared order, so ties resolve deterministically to the first.
    best = max((calib_label(e, d) for e, d in CALIB if calib_label(e, d) in cal),
               key=lambda L: cal[L]["asr"], default=None)
    if best is None:
        sys.exit("REFUSING TO CONTINUE regime A: no calibration run completed.")
    b = cal[best]
    passed = b["asr"] > ref[1]
    save_top("calibration_winner",
             {"label": best, "eps": b["eps"], "decorrelate": b["decorrelate"],
              "seed": CALIB_SEED, "asr": b["asr"], "accuracy": b["accuracy"],
              "reference_plain_asr": ref[1], "reference_source":
                  f"results/dose_response/summary.json {d1_conf(0.0)}_then_{D2}|{ATTACK} s{CALIB_SEED}",
              "gate_4_1_pass": bool(passed),
              "verdict": ("adversary is stronger than the plain backdoor against coord_median"
                          if passed else
                          "REGIME A IS VOID: the criterion-aware construction does not strengthen "
                          "the adversary against coord_median. kappa=2 is not run and the regime "
                          "licenses no sentence about adaptivity.")})
    print(f"\n  GATE 4.1 argmax {best}: ASR={b['asr']:.4f} vs plain {ref[1]:.4f}  "
          f"{'PASS' if passed else 'VOID'}")
    if not passed:
        print("  Regime A is VOID, not negative. Reported as void, in the paper, next to the "
              "regimes that were not void.")
        return None
    print(f"  frozen for every rung and seed of regime A: eps={b['eps']:g}, "
          f"decorrelate={b['decorrelate']}")
    # The winner IS the kappa=0/seed-42 leg. Copied, not recomputed, with provenance.
    if not has_run(cell_key("A_adaptive", "conf", 0.0), CALIB_SEED):
        save_one(cell_key("A_adaptive", "conf", 0.0), "A_adaptive", "conf", 0.0, CALIB_SEED,
                 b["accuracy"], b["asr"],
                 provenance=f"the winning calibration run ({best}); not recomputed (prereg 3)")
    return (b["eps"], b["decorrelate"])


# --------------------------------------------------------------------------- the arm

def run_regime(regime):
    r = REGIMES[regime]
    cfg = r["fl_config"]
    print("\n" + "=" * 78)
    print(f"  REGIME {regime}: N={cfg.num_clients}, K={cfg.clients_per_round}, "
          f"alpha={r['alpha']}, f=0.2, rounds={cfg.num_rounds}, {D2} x {ATTACK}")
    print(f"  what changed vs the frozen regime: {r['what_changed']}")
    print("=" * 78, flush=True)

    ca = None
    if r["adaptive"]:
        ca = calibration_gate()
        if ca is None:
            return                                   # VOID: kappa=2 is not run
    if not identity_gate(regime, ca):
        return                                       # gate 4.3 failed: the arm stops

    # kappa=0 is computed ONCE per seed, under the confounded ladder, and shared by both Deltas.
    todo = [("conf", 0.0, s) for s in SEEDS]
    todo += [(lad, 2.0, s) for lad in ("conf", "inst") for s in SEEDS]
    t0 = time.time()
    for i, (lad, k, seed) in enumerate(todo, 1):
        key = cell_key(regime, lad, k)
        if has_run(key, seed):
            print(f"  [skip] {lad} kappa={k} s{seed}  ({i}/{len(todo)} position, not progress)",
                  flush=True)
            continue
        t = time.time()
        acc, asr = run_leg(regime, lad, k, seed, ca)
        save_one(key, regime, lad, k, seed, acc, asr)
        print(f"  {lad} kappa={k} s{seed}: acc={acc:.4f} ASR={asr:.4f} "
              f"({(time.time()-t)/60:.1f}min, {(time.time()-t0)/3600:.2f}h elapsed)", flush=True)


# --------------------------------------------------------------------------- analysis

def analyze():
    """Gates first, then the primary D, then the descriptive Deltas, then the 4.4 diagnostic.

    Every number is recomputed from per_seed rows here, and both legs of every Delta in the same
    call, so a subtrahend can never come from a different seed count than its minuend. Nothing from
    the pre-registration is transcribed into this function.
    """
    s = load_or_init()
    print("\n" + "=" * 78)
    print("  REGIME DISSOCIATION -- RESULTS")
    print(f"  prereg {PREREG_COMMIT}   cell: {D2} x {ATTACK}   kappa in {KAPPAS}   n<={len(SEEDS)}")
    print("=" * 78)

    hc = s.get("harness_check", {})
    print(f"\n  harness check (default path unchanged by value): "
          f"{'PASS' if hc.get('pass') else 'NOT RUN / FAIL'}"
          + (f", worst |d| = {hc['worst_abs_deviation']:.2e}" if hc else ""))

    out = {}
    for regime in REGIMES:
        print("\n" + "-" * 78)
        print(f"  {regime}  ({s['regimes'][regime]['what_changed']})")
        print("-" * 78)
        rep = {"regime": regime}

        if REGIMES[regime]["adaptive"]:
            w = s.get("calibration_winner", {})
            if not w:
                print("  GATE 4.1 calibration: NOT RUN"); out[regime] = rep; continue
            print(f"  GATE 4.1 calibration: argmax {w['label']} ASR={w['asr']:.4f} vs plain "
                  f"{w['reference_plain_asr']:.4f}  {'PASS' if w['gate_4_1_pass'] else 'VOID'}")
            rep["gate_4_1"] = w
            if not w["gate_4_1_pass"]:
                print(f"  {w['verdict']}"); rep["verdict"] = "VOID"; out[regime] = rep; continue

        g = s.get("identity_gate", {}).get(regime)
        if not g:
            print("  GATE 4.3 identity rung: NOT RUN"); out[regime] = rep; continue
        print(f"  GATE 4.3 identity rung (s{g['seed']}): d=({g['d_acc']:+.6f}, {g['d_asr']:+.6f}) "
              f" {'PASS' if g['pass'] else 'FAIL'}")
        rep["gate_4_3"] = g
        if not g["pass"]:
            rep["verdict"] = "STOPPED: kappa=0 not shared"; out[regime] = rep; continue

        z = rows(cell_key(regime, "conf", 0.0))
        c2 = rows(cell_key(regime, "conf", 2.0))
        i2 = rows(cell_key(regime, "inst", 2.0))
        seeds = sorted(set(z) & set(c2) & set(i2))
        print(f"\n  rung means over the {len(seeds)} seed(s) complete in all three legs "
              f"({seeds}):")
        rep["seeds"] = seeds
        if not seeds:
            print("    incomplete -- no seed has all three legs"); out[regime] = rep; continue

        # GATE 4.2, accuracy, per rung, scored BEFORE the primary. A low ASR at collapsed accuracy
        # is not suppression, so a failure here makes the regime uninterpretable and no verdict
        # stands -- including a verdict that would otherwise have been favourable.
        acc_ok, accs = True, {}
        for nm, rr in (("kappa=0 (shared)", z), ("kappa=2 confounded", c2),
                       ("kappa=2 instrument", i2)):
            a = float(np.mean([rr[x][0] for x in seeds]))
            v = float(np.mean([rr[x][1] for x in seeds]))
            accs[nm] = {"mean_accuracy": a, "mean_asr": v}
            ok = a >= ACC_FLOOR
            acc_ok &= ok
            print(f"    {nm:22s} acc={a:.4f} {'OK ' if ok else 'BELOW FLOOR'}  ASR={v:.4f}")
        rep["gate_4_2"] = {"acc_floor": ACC_FLOOR, "pass": bool(acc_ok), "rungs": accs}
        print(f"  GATE 4.2 accuracy floor {ACC_FLOOR}: {'PASS' if acc_ok else 'FAIL'}")
        if not acc_ok:
            print("  FAIL -- this regime is uninterpretable and NO VERDICT STANDS. A low ASR at "
                  "collapsed accuracy is not suppression.")
            rep["verdict"] = "UNINTERPRETABLE: accuracy floor"
            out[regime] = rep; continue

        # PRIMARY. D_s = ASR_conf(k2, s) - ASR_inst(k2, s): the kappa=0 leg cancels exactly because
        # the two designs share that rung, which gate 4.3 just proved in this regime.
        d = np.array([c2[x][1] - i2[x][1] for x in seeds], dtype=float)
        n = len(d)
        mean = float(d.mean())
        if n >= 2:
            se = float(d.std(ddof=1) / np.sqrt(n)); hw = t_crit(n) * se
        else:
            se = hw = float("nan")
        lo, hi = mean - hw, mean + hw
        excludes = n >= 2 and (lo > 0 or hi < 0)
        if not excludes:
            verdict = ("NOT REPRODUCED AT n=%d: the interval contains zero. Reported as a failure "
                       "at this seed count, NOT as evidence of absence, and NOT topped up." % n)
        elif mean < 0:
            verdict = ("THE DESIGN DISAGREEMENT REPRODUCES: the confounded design reads lower than "
                       "the instrument on the same cell and the same seeds.")
        else:
            verdict = ("REVERSED: the confounded design reads HIGHER. This contradicts the paper's "
                       "mechanism and goes in the body as a contradiction.")
        print(f"\n  PRIMARY  D = dASR(confounded) - dASR(instrument), paired by seed, n={n}")
        print(f"    per seed: " + "  ".join(f"s{x}:{v:+.4f}" for x, v in zip(seeds, d)))
        print(f"    mean {mean:+.4f}   sd {float(d.std(ddof=1)) if n >= 2 else float('nan'):.4f}"
              f"   95% CI [{lo:+.4f}, {hi:+.4f}]   excludes zero: {excludes}")
        print(f"    VERDICT: {verdict}")
        rep["primary_D"] = {"per_seed": {str(x): float(v) for x, v in zip(seeds, d)},
                            "mean": mean, "sd": float(d.std(ddof=1)) if n >= 2 else None,
                            "se": se, "ci95": [lo, hi], "n": n,
                            "excludes_zero": bool(excludes), "verdict": verdict}

        # SECONDARY, descriptive, explicitly NOT inferential: neither Delta is powered to exclude
        # zero at n=3 (prereg 2, measured), so each is printed as a point estimate with its
        # half-width beside it and no significance is claimed for either.
        print("\n  SECONDARY (descriptive only -- neither Delta is powered to exclude zero at "
              f"n={n}):")
        rep["secondary_deltas"] = {}
        for nm, rr in (("confounded", c2), ("instrument", i2)):
            dd = np.array([rr[x][1] - z[x][1] for x in seeds], dtype=float)
            m = float(dd.mean())
            h = (t_crit(n) * float(dd.std(ddof=1) / np.sqrt(n))) if n >= 2 else float("nan")
            print(f"    dASR {nm:11s} {m:+.4f}  (half-width {h:.4f}; no significance claimed)")
            rep["secondary_deltas"][nm] = {"mean": m, "half_width": h,
                                           "inferential": False}

        # 4.4 DIAGNOSTIC, deliberately NOT a gate. The frozen kappa=0 rung of this very cell is
        # 0.490 and the N=100 prospective figure 0.500, so a SUPPRESS_ASR=0.5 gate would void the
        # published flagship too. Printed so the honest reading is unavoidable.
        b = float(np.mean([z[x][1] for x in seeds]))
        print(f"\n  DIAGNOSTIC (4.4, not a gate): base rung ASR = {b:.4f}. coord_median alone does "
              f"not strongly\n    suppress this backdoor, which the frozen data already shows. No "
              f"sentence from this arm\n    may say the base rung suppresses the attack.")
        rep["base_rung_diagnostic"] = {"mean_asr": b, "is_a_gate": False}
        out[regime] = rep

    save_top("analysis", out)
    print("\n  " + "-" * 76)
    print("  Limitations restated from the pre-registration, because they bind the writing:")
    print("    n=3 fixed, no optional stopping. Two rungs, so no trend test and this is not a")
    print("    four-rung ladder. Only D is tested. One cell, one attack, cifar_cnn on CIFAR-10.")
    print("    Regime A's adversary does not anticipate d_1, which is an instrument, not a")
    print("    defense. Regime B confounds N with K. No equivalence claim is made here.")
    print(f"\n  Output: {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-check", action="store_true",
                    help="prove the three new arguments are bit-identical no-ops; run this first")
    ap.add_argument("--regime", choices=["A", "B", "both"], default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    check_frozen()
    os.makedirs(out_dir, exist_ok=True)

    if a.harness_check:
        sys.exit(0 if harness_check() else 1)
    if a.analyze_only:
        analyze(); sys.exit(0)
    if a.regime is None:
        ap.error("choose --harness-check, --regime {A,B,both}, or --analyze-only")

    if not load_or_init().get("harness_check", {}).get("pass"):
        sys.exit("REFUSING TO RUN: --harness-check has not passed in this artifact.\n"
                 "  python3 -m experiments.run_regime_dissociation --harness-check\n"
                 "Without it the three new arguments are not proved to be no-ops, and a silently\n"
                 "uncalled hook is exactly how a composition arm in this repo once came out\n"
                 "bit-identical to another.")

    for reg in (["A_adaptive"] if a.regime == "A" else
                ["B_scale"] if a.regime == "B" else ["A_adaptive", "B_scale"]):
        run_regime(reg)
    analyze()
