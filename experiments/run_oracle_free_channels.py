"""
The two channels, separated WITHOUT adversary identity, at n = 20 on a non-floor cell.

WHY THIS EXISTS. A review scored the paper 5/10 and named exactly one thing that would move it:

    "one stronger, non-floor-effect causal intervention with substantially more seeds and less
     dependence on oracle adversary identity. If that experiment reproduces the central dissociation,
     I could realistically move the review to 6-7/10."

Every published instance of either leg of that dissociation is a doseS/doseA/doseM rung, and
apply_d1_transform RAISES for all three without adv_mask. So every published instance reads adversary
identity, and the objection is correct as a statement about the evidence. This arm changes the
transform and keeps everything else:

  * the UPSTREAM STAGE is rfa -- a real defense, whose per-client coefficients are the final Weiszfeld
    weights of the geometric-median iteration, computed from the update stack alone. No mask. The
    runner passes adv_mask=None, so oracle-freeness is ENFORCED by the callee: any dose family reached
    this way raises instead of running.
  * the CHANNEL SEPARATION is the paper's own pre-registered 2x2 factorial (score_only /
    emit_only, generic_compose), not a new method. Both controls act on every client at once, so
    neither needs to know who the adversary is. score_only closes the magnitude channel DOWNSTREAM of
    the coefficient, which is why the impossibility at main.tex:2051 -- no positive rescaling can be
    both Delta_c-neutral and informative -- does not bite: nothing here is claimed to be
    Delta_c-neutral.
  * the CELL is krum under committed_pixel, and that answers the review's other objection. Baseline
    admitted adversarial mass here is 0.3333, nonzero in 4 of 12 adversary rounds, against 0.0000 in
    0 of 12 for the same Krum under committed_scaling. Admission has somewhere to fall, so an
    unchanged admission is a measurement rather than a ceiling artifact.
  * n = 20 seeds, 42-61, fixed now. No optional stopping and no top-up clause.

WHAT THE PROSPECTIVE CHANNEL MEASUREMENT ESTABLISHED (results/oracle_free_admission.json, no ASR):

  krum decision change   0.000 -> 0.533     rfa moves Krum's pick in over half of all rounds
  krum ADMISSION change  0.000 -> 0.000     the admitted adversarial mass does not move at all
  aggregate displacement 0.000 -> 0.868

which is the premise the flagship arm rests on, now with a transform that reads no adversary identity.
The three-way eligibility rule that assigned this arm type GENERALIZATION was written into the
pre-registration before those numbers were read, its two constants are imported from
measure_admission_mask.py rather than restated, and it was applied to the PUBLISHED doseS/Krum arm
first (it types that arm GENERALIZATION too, so it is not a rule that condemns only what is new).

THE THREE ARMS. All three share the same seeds and are paired by seed:

  identity   d1_override="fedavg"                 Krum standalone; apply_d1_transform returns the
                                                  update list unwrapped, so this is the untransformed
                                                  aggregator by construction, not by approximation
  statistic  d1_override="rfa", score_only=True    LEG 1: Krum SCORES the rfa-scaled stack and EMITS
                                                  the selected client's ORIGINAL update
  magnitude  d1_override="rfa", emit_only=True     LEG 2: Krum SCORES the RAW stack -- so its decision
                                                  is pinned bit-identically to the identity arm's --
                                                  and EMITS the rfa-scaled selected update

FROZEN PREDICTIONS (pre_registration_oracle_free_channels.md section 2, arm type GENERALIZATION):
leg 1 |Delta_stat| INSIDE the frozen EQUIV_MARGIN = 0.15; leg 2 Delta_mag nonzero with its paired 95%
interval EXCLUDING zero.

TWO VERDICTS PER LEG, AND NEITHER MAY BE REPORTED AS THE OTHER. An interval can sit inside the margin
and still exclude zero. So verdict_sign (does the paired interval exclude zero) and verdict_margin (is
|mean| < 0.15) are computed and reported separately for both legs, and neither verdict literal carries
a mechanism clause -- Round 72's frozen literal fused a label with a mechanism guess and the guess was
backwards in sign, so the untested half inherited the tested half's authority. Mechanism goes in its
own field, labelled as the measurement.

THE BRANCH THAT REFUTES US, frozen before any run: if |Delta_stat| >= 0.15 with an interval excluding
zero, then in this cell the statistic channel DOES carry suppression without an oracle. That
contradicts the paper's headline reading and is reported in the body as a contradiction, in the place a
confirmation would have gone. The sentence "the negative reproduces oracle-free" may not be written
under any circumstances.

THE ACCURACY GATE IS LIVE, NOT A FORMALITY. The identity cell's measured per-seed accuracy at seeds
42/43/44 is 0.3962 / 0.5922 / 0.4670 against ACC_FLOOR = 0.35. Krum selects one client per round for
50 rounds, so a seed whose picks are unrepresentative can land near the floor. The gate is scored on
the arm mean, as everywhere else in the paper, with the per-seed minimum printed beside it.

WHAT THIS ARM CANNOT CONCLUDE. One aggregator, one attack, one dataset, one upstream defense. The
sentence "the negative holds without an oracle" may not be written; "in this cell, without adversary
identity" is what it licenses. And the controls modify the aggregator's internals, so this is still an
INSTRUMENT, not a deployable defense: it answers the identity objection and not the deployability one,
which stays open and stays disclosed.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar10/cifar_cnn.
New runs: 3 arms x 20 seeds = 60. None is importable -- krum|committed_pixel exists on disk only at
n=3, in results/prospective_pilot/summary.json, produced by a different runner, and that row is used as
--harness-check's cross-suite target instead of as data.

Output: results/oracle_free_channels/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_oracle_free_channels.md is git-committed and
PREREG_COMMIT below is that hash. The script refuses to start otherwise, and it also refuses unless the
prospective channel measurement exists and reports an eligible arm type.

  PYTHONPATH=. python3 -m experiments.run_oracle_free_channels --harness-check
  PYTHONPATH=. python3 -m experiments.run_oracle_free_channels
"""
import json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen suite, so this arm and the published dose arms cannot drift apart in
# the runner, the accuracy floor, the equivalence margin or the attack implementation. run_one is
# reused rather than copied; the only change it needed is the additive d1_override keyword, and
# harness check 1 below is the proof that the keyword changed nothing for every existing call site.
from experiments.run_targeted_dose import (run_one, ACC_FLOOR, EQUIV_MARGIN,  # noqa: E402
                                           ATTACK_MAP, FL_CONFIG)
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
# t_crit IS IMPORTED, NEVER T95[n-1]: that literal table stops at df = 9 and this arm's n is 20.
from experiments.analyze_headline_cis import t_crit as _t_crit  # noqa: E402

# experiments/pre_registration_oracle_free_channels.md, committed as its own commit at a point when
# both of this arm's output paths were verified not to exist.
PREREG_COMMIT = "4e7090b"

D2, ATTACK = "krum", "committed_pixel"
D1 = "rfa"
SEEDS20 = list(range(42, 62))                  # 42-61, frozen in the pre-registration
T_FROZEN = 2.093                               # the n=20 two-sided 95% t the freeze states, df=19

# (label, d1_override, score_only, emit_only, what it holds fixed)
ARMS = [
    ("identity",  "fedavg", False, False, "nothing transformed; Krum standalone by construction"),
    ("statistic", D1,       True,  False, "emits the selected client's ORIGINAL update (leg 1)"),
    ("magnitude", D1,       False, True,  "scores the RAW stack, decision pinned (leg 2)"),
]
LEGS = [("Delta_stat", "statistic", "inside the margin"),
        ("Delta_mag",  "magnitude", "nonzero, interval excluding zero")]

TOL = 1e-9                                     # harness checks 1 and 2, as frozen in section 4.4

out_dir = os.path.join(base, "results", "oracle_free_channels")
out_path = os.path.join(out_dir, "summary.json")
CHANNELS = os.path.join(base, "results", "oracle_free_admission.json")
PILOT = os.path.join(base, "results", "prospective_pilot", "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")
PREREG = os.path.join(base, "experiments", "pre_registration_oracle_free_channels.md")

# Harness check 1's target: a cell of the frozen suite that was COMPUTED there rather than imported
# into it, so recomputing it exercises the d1_name path the override must leave untouched.
FROZEN_CELL = ("doseS_kappa2.0_then_krum|committed_scaling", "S", "krum", "committed_scaling", 2.0)


def t_crit(n):
    """Two-sided 95% t. Falls back to the frozen literal at this arm's own n, never to a wrong df."""
    try:
        return _t_crit(n)
    except KeyError:
        if n == len(SEEDS20):
            return T_FROZEN
        raise


def key_of(arm):
    """Cell key. The identity arm's key is the composition suite's own name for Krum standalone."""
    label, d1, so, eo, _ = arm
    return f"{d1}_then_{D2}|{ATTACK}" + ("|score_only" if so else "|emit_only" if eo else "")


def premise():
    """This arm's eligibility, read from the prospective measurement. Read-only, never recomputed.

    Returns a dict, never a bare string, so the source is always carried with the verdict, and every
    failure branch reads UNMEASURED rather than defaulting to something that looks like a pass.
    """
    if not os.path.exists(CHANNELS):
        return {"arm_type": "UNMEASURED", "reason": f"{CHANNELS} does not exist"}
    try:
        d = json.load(open(CHANNELS))
    except Exception as e:
        return {"arm_type": "UNMEASURED", "reason": f"{CHANNELS} is unreadable ({e})"}
    if d.get("prereg_commit") != PREREG_COMMIT:
        return {"arm_type": "UNMEASURED",
                "reason": f"{CHANNELS} is stamped at prereg_commit {d.get('prereg_commit')!r}, not "
                          f"{PREREG_COMMIT!r}, so it was not produced under this arm's frozen rule"}
    s, e = d.get("summary", {}), d.get("eligibility", {})
    return {
        "source": "results/oracle_free_admission.json",
        "arm_type": e.get("arm_type"),
        # The verdict string is recorded VERBATIM from the sibling artifact rather than paraphrased,
        # so this runner cannot soften or sharpen a verdict it did not compute.
        "eligibility_verdict_verbatim": e.get("verdict"),
        "decision_change": {"identity": s.get(f"oracleFree|{D2}|0.0|decision"),
                            D1: s.get(f"oracleFree|{D2}|1.0|decision")},
        "admission_change": {"identity": s.get(f"oracleFree|{D2}|0.0|admission"),
                             D1: s.get(f"oracleFree|{D2}|1.0|admission")},
        "agg_displacement": {"identity": s.get(f"oracleFree|{D2}|0.0|agg_disp"),
                             D1: s.get(f"oracleFree|{D2}|1.0|agg_disp")},
        "baseline_admission_not_a_floor": d.get("baseline_admission_not_a_floor"),
        "identity_rung_exactness": d.get("identity_rung_exactness"),
        "oracle_free_check": d.get("oracle_free_check", {}).get(
            "mask_independent_and_non_inert"),
        "published_dose_raises_without_adv_mask": d.get("oracle_free_check", {}).get(
            "published_dose_raises_without_adv_mask"),
        "controls_bit_exact": {
            "emit_only_decision_pinned": d.get("control_check", {}).get(
                "emit_only_decision_pinned_bit_exact"),
            "score_only_emits_raw": d.get("control_check", {}).get(
                "score_only_emits_raw_selected_update_bit_exact")},
        "published_arm_cross_check": d.get("published_arm_cross_check"),
        "structural_blockers": e.get("structural_blockers"),
    }


def published_identity():
    """{seed: (accuracy, asr)} for krum|committed_pixel, read from its source of record.

    NOT used as data: this arm computes all 20 identity runs itself. Used only by harness check 2,
    which asserts that d1_override="fedavg" reproduces it at a shared seed, so that a protocol
    difference between the pilot suite and this one surfaces before 16 hours are spent rather than
    being explained afterwards. This is the same cross-suite assertion run_dose_response.py makes for
    ("krum", "committed_scaling") against the same file.
    """
    if not os.path.exists(PILOT):
        return {}
    cells = json.load(open(PILOT)).get("cells", {})
    k = f"{D2}|{ATTACK}" if f"{D2}|{ATTACK}" in cells else f"fedavg_then_{D2}|{ATTACK}"
    if k not in cells:
        return {}
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"]))
            for r in cells[k].get("per_seed", [])}


def published_frozen_cell():
    """{seed: (accuracy, asr)} for harness check 1's frozen targeted_dose cell."""
    if not os.path.exists(TARGETED):
        return {}
    c = json.load(open(TARGETED)).get("cells", {}).get(FROZEN_CELL[0])
    if not c:
        return {}
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c.get("per_seed", [])}


def check_frozen():
    """Refuse to start unless the predictions are frozen and the premise was measured prospectively."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the two frozen predictions and the refuting branch are not "
                 f"committed.\n  1. git commit {PREREG} alone\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen prediction is unfalsifiable, which is the entire point of the freeze.")
    # `PREREG_COMMIT is None` is only half a guard: once the constant is set, the only test that ever
    # fires stops firing. So the hash is checked against the log and the tree against porcelain.
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze ({e}).")
    if not log.startswith(PREREG_COMMIT[:7]) and not PREREG_COMMIT.startswith(log[:7]):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last committed at {log!r}, not {PREREG_COMMIT!r}.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted modifications ({dirty!r}).")
    p = premise()
    if p.get("arm_type") == "UNMEASURED":
        sys.exit(f"REFUSING TO RUN: {p.get('reason')}. This arm's type and its predictions were "
                 "assigned by the prospective channel measurement; without it they have no basis. "
                 "Run experiments/measure_admission_oracle_free.py first.")
    if p["arm_type"] not in ("GENERALIZATION", "ADMISSION"):
        sys.exit(f"REFUSING TO RUN: the channel measurement reports arm_type={p['arm_type']!r}. "
                 "Under the three-way rule frozen in section 4.1, only GENERALIZATION and ADMISSION "
                 "run at all; INELIGIBLE means rfa barely disturbs Krum's decision, so a flat ASR "
                 "curve would carry no information, and INDETERMINATE means a structural check "
                 "failed. Re-read the rule before running anything.\n"
                 f"  verdict: {p.get('eligibility_verdict_verbatim')}")
    for name, ok in (("the transform is mask-independent and non-inert", p["oracle_free_check"]),
                     ("the published dose transform raises without adv_mask",
                      p["published_dose_raises_without_adv_mask"]),
                     ("emit_only's decision is pinned bit-exactly",
                      p["controls_bit_exact"]["emit_only_decision_pinned"]),
                     ("score_only emits the raw selected update bit-exactly",
                      p["controls_bit_exact"]["score_only_emits_raw"])):
        if not ok:
            sys.exit(f"REFUSING TO RUN: {CHANNELS} reports that {name} is FALSE. Section 4.3 makes "
                     "each of these blocking: an inert or oracle-reading transform cannot produce a "
                     "null result that means anything.")
    if not published_identity():
        sys.exit("REFUSING TO RUN: no published krum|committed_pixel row to check the identity arm "
                 f"against in {PILOT}. Harness check 2 would be vacuous.")


def harness_check():
    """The three checks of section 4.4, in that order, with their verdicts RETURNED for persistence.

    Round 69 established that this repository's existing harness check returns a verdict dict its
    caller discards, so the paper's sentence about it is witnessed only by stdout while re-running
    costs hours. This one returns the dict AND main() writes it into the artifact.
    """
    v = {"tol": TOL, "checks": [], "all_passed": None}
    print("=== HARNESS CHECK 1/3: d1_override=None is bit-neutral on a frozen cell ===")
    key, mode, d2, atk, val = FROZEN_CELL
    pub = published_frozen_cell()
    if not pub:
        print(f"  CANNOT CHECK: {key} is not in {TARGETED}.")
        v["checks"].append({"check": "d1_override_default_is_bit_neutral", "passed": False,
                            "reason": f"{key} absent from results/targeted_dose/summary.json"})
    else:
        seed = sorted(pub)[0]
        p_acc, p_asr = pub[seed]
        print(f"    {key} at seed {seed}: published acc={p_acc:.6f} ASR={p_asr:.6f}")
        print("    Recomputed through the PATCHED run_one with d1_override unset. This is the check "
              "that\n    the one additive keyword this arm adds to shared code changed nothing.",
              flush=True)
        t = time.time()
        acc, asr = run_one(seed, mode, d2, atk, val)
        d = (acc - p_acc, asr - p_asr)
        ok = max(abs(d[0]), abs(d[1])) <= TOL
        print(f"    recomputed: acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})   "
              f"{'BIT-NEUTRAL' if ok else 'DRIFTED'}  ({time.time() - t:.0f}s)", flush=True)
        v["checks"].append({"check": "d1_override_default_is_bit_neutral", "passed": bool(ok),
                            "cell": key, "seed": seed, "published": [p_acc, p_asr],
                            "recomputed": [float(acc), float(asr)],
                            "delta": [float(d[0]), float(d[1])]})

    print("\n=== HARNESS CHECK 2/3: the identity arm IS Krum standalone, across suites ===")
    pub = published_identity()
    seed = sorted(pub)[0]
    p_acc, p_asr = pub[seed]
    print(f"    results/prospective_pilot {D2}|{ATTACK} at seed {seed}: "
          f"acc={p_acc:.6f} ASR={p_asr:.6f}")
    print("    Recomputed as d1_override=\"fedavg\", which returns the update list unwrapped. "
          "BLOCKING:\n    if the two suites do not share a code path on the identity, this arm's 20 "
          "identity runs are\n    not comparable to any published baseline.", flush=True)
    t = time.time()
    acc, asr = run_one(seed, None, D2, ATTACK, None, d1_override="fedavg")
    d = (acc - p_acc, asr - p_asr)
    ok2 = max(abs(d[0]), abs(d[1])) <= TOL
    print(f"    recomputed: acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})   "
          f"{'IDENTICAL' if ok2 else 'MISMATCH'}  ({time.time() - t:.0f}s)", flush=True)
    v["checks"].append({"check": "identity_arm_is_krum_standalone_across_suites", "passed": bool(ok2),
                        "source": "results/prospective_pilot/summary.json", "seed": seed,
                        "published": [p_acc, p_asr], "recomputed": [float(acc), float(asr)],
                        "delta": [float(d[0]), float(d[1])]})

    print("\n=== HARNESS CHECK 3/3: the rfa transform was ENTERED and is not inert ===")
    print("    The standing failure mode here is a hook that is never called while the run passes\n"
          "    silently, so an inert transform must not be able to pass as a null result.",
          flush=True)
    torch.manual_seed(SEEDS20[0]); np.random.seed(SEEDS20[0])
    ups = [{"w": torch.randn(64, 32), "b": torch.randn(64)} for _ in range(5)]
    t3 = apply_d1_transform([{k: x.clone() for k, x in u.items()} for u in ups], D1,
                            tau=5.0, dose_key=None, adv_mask=None)
    distinct = all(t3[i][k] is not ups[i][k] for i in range(len(ups)) for k in ups[i])
    ratios = [float(torch.cat([t3[i][k].flatten().double() for k in t3[i]]).norm()
                    / torch.cat([ups[i][k].flatten().double() for k in ups[i]]).norm())
              for i in range(len(ups))]
    nontrivial = any(abs(r - 1.0) > 1e-9 for r in ratios)
    ok3 = distinct and nontrivial
    print(f"    returned objects distinct from the inputs: {distinct}")
    print(f"    per-client norm ratios: {[round(r, 6) for r in ratios]}")
    print(f"    at least one ratio != 1: {nontrivial}   -> {'APPLIED' if ok3 else 'INERT'}")
    v["checks"].append({"check": "rfa_branch_entered_and_not_inert", "passed": bool(ok3),
                        "objects_distinct": bool(distinct),
                        "per_client_norm_ratio_float64": ratios,
                        "at_least_one_ratio_differs_from_one": bool(nontrivial)})

    v["all_passed"] = all(c["passed"] for c in v["checks"])
    print(f"\n  {'ALL THREE CHECKS PASSED. The ladder may run.' if v['all_passed'] else 'A CHECK FAILED. The ladder must NOT run; the failure is the result.'}")
    return v


def load():
    if not os.path.exists(out_path):
        return {}, None
    try:
        d = json.load(open(out_path))
        return d.get("cells", {}), d.get("harness_check")
    except Exception:
        return {}, None


def paired(cells, arm_label):
    """{seed: asr} for one arm, and {seed: accuracy}, keyed by seed so no Delta can mix seed sets."""
    arm = next(a for a in ARMS if a[0] == arm_label)
    rows = cells.get(key_of(arm), {}).get("per_seed", [])
    return ({int(r["seed"]): float(r["asr"]) for r in rows},
            {int(r["seed"]): float(r["accuracy"]) for r in rows})


def leg(cells, arm_label):
    """One Delta, with BOTH verdicts, recomputed from per-seed rows in a single call.

    Both legs of the difference are read here, in this call, so a subtrahend can never come from a
    different seed count -- or a different arm -- than its minuend. Nothing is transcribed.
    """
    a_asr, a_acc = paired(cells, arm_label)
    b_asr, b_acc = paired(cells, "identity")
    seeds = sorted(set(a_asr) & set(b_asr))
    r = {"arm": arm_label, "n": len(seeds), "seeds": seeds,
         "arm_mean_asr": float(np.mean([a_asr[s] for s in seeds])) if seeds else None,
         "identity_mean_asr": float(np.mean([b_asr[s] for s in seeds])) if seeds else None,
         "arm_mean_acc": float(np.mean([a_acc[s] for s in seeds])) if seeds else None,
         "arm_min_acc": float(min(a_acc[s] for s in seeds)) if seeds else None,
         "identity_mean_acc": float(np.mean([b_acc[s] for s in seeds])) if seeds else None,
         "identity_min_acc": float(min(b_acc[s] for s in seeds)) if seeds else None,
         "per_seed_delta": {str(s): a_asr[s] - b_asr[s] for s in seeds}}
    if len(seeds) < 2:
        r.update({"mean": None, "sd": None, "ci95": None, "verdict_sign": "NOT COMPUTED",
                  "verdict_margin": "NOT COMPUTED", "acc_gate": "NOT COMPUTED"})
        return r
    d = np.array([a_asr[s] - b_asr[s] for s in seeds], dtype=float)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = t_crit(len(d)) * sd / np.sqrt(len(d))
    lo, hi = m - hw, m + hw
    # Two verdicts, computed independently. An interval can sit inside the margin AND exclude zero,
    # so neither of these may ever be reported as the other, and neither literal names a mechanism.
    r.update({"mean": m, "sd": sd, "se": float(sd / np.sqrt(len(d))), "t_crit": t_crit(len(d)),
              "ci95": [lo, hi], "half_width": float(hw),
              "verdict_sign": ("ASR MOVES on this leg (interval excludes zero), sign "
                               + ("positive" if m > 0 else "negative"))
                              if lo * hi > 0 else
                              "NOT RESOLVED at this n (interval contains zero); not evidence of "
                              "absence",
              "verdict_margin": (f"within the frozen equivalence margin (|mean| < {EQUIV_MARGIN})"
                                 if abs(m) < EQUIV_MARGIN else
                                 f"outside the frozen margin (|mean| >= {EQUIV_MARGIN})"),
              "margin": EQUIV_MARGIN,
              "acc_gate": ("PASS" if (r["arm_mean_acc"] >= ACC_FLOOR
                                      and r["identity_mean_acc"] >= ACC_FLOOR) else
                           "FAIL -- this arm is VOID, not negative: a low ASR at collapsed accuracy "
                           "is not suppression"),
              "acc_floor": ACC_FLOOR})
    return r


def save(cells, hc):
    verdicts = {name: leg(cells, arm) for name, arm, _ in LEGS}
    json.dump({
        "description":
            "The statistic and magnitude channels separated WITHOUT adversary identity, at n=20 on a "
            "non-floor cell. Upstream stage is rfa, whose Weiszfeld weights are computed from the "
            "update stack alone; adv_mask is passed as None, so any oracle rung would RAISE rather "
            "than run. Channel separation is the paper's own pre-registered score_only/emit_only "
            "factorial. Host cell krum under committed_pixel, whose baseline admitted adversarial "
            "mass is 0.3333 in 4 of 12 adversary rounds rather than a floor. Predictions frozen at "
            f"{PREREG_COMMIT} (experiments/pre_registration_oracle_free_channels.md).",
        "prereg_commit": PREREG_COMMIT,
        "prereg": "experiments/pre_registration_oracle_free_channels.md",
        "dataset": "cifar10", "model": "cifar_cnn",
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round, "f": 0.2,
                   "alpha": 0.5, "rounds": FL_CONFIG.num_rounds, "seeds": SEEDS20,
                   "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                   "t_crit_n20": t_crit(len(SEEDS20))},
        "arm": {"d2": D2, "attack": ATTACK, "attack_impl": ATTACK_MAP[ATTACK], "d1": D1,
                "arms": [{"label": a[0], "d1_override": a[1], "score_only": a[2],
                          "emit_only": a[3], "holds": a[4], "key": key_of(a)} for a in ARMS],
                "frozen_prediction": {
                    "Delta_stat": "|mean| INSIDE the frozen EQUIV_MARGIN = 0.15",
                    "Delta_mag": "nonzero, paired 95% interval EXCLUDING zero"}},
        "oracle_free": "No arm reads adversary identity. run_targeted_dose.run_one passes "
                       "adv_mask=None whenever d1_override is set, and apply_d1_transform's dose "
                       "families RAISE without it, so oracle-freeness is enforced by the callee "
                       "rather than intended by the caller. rfa's coefficients are the final "
                       "Weiszfeld weights of the geometric-median iteration over the update stack.",
        "not_a_defense": "score_only and emit_only modify the aggregator's internals, so this arm is "
                         "an instrument for causal identification and not a deployable defense. It "
                         "answers the adversary-identity objection and not the deployability one.",
        "scope": "One aggregator, one attack, one dataset, one upstream defense. This supports no "
                 "statement of the form 'the negative holds without an oracle'; what it licenses is "
                 "'in this cell, without adversary identity'.",
        "theory_scope": "main.tex:2051's impossibility -- no positive rescaling is both "
                        "Delta_c-neutral and informative -- is not contradicted here: nothing in "
                        "this arm is claimed to be Delta_c-neutral. score_only closes the magnitude "
                        "channel downstream of the coefficient, for every client at once.",
        "frozen_premise_channels": premise(),
        "harness_check": hc,
        "verdicts": verdicts,
        "verdict_reading_rule": "verdict_sign and verdict_margin are independent labels and neither "
                                "may be reported as the other: an interval can sit inside the margin "
                                "and still exclude zero. Neither literal names a mechanism. The "
                                "phrase 'no evidence of a practically meaningful change' is defined "
                                "at supplementary.tex:441 against the stricter of the paper's two "
                                "interval conventions and may not be applied to any leg here.",
        "cells": cells}, open(out_path, "w"), indent=2)


def report(cells):
    print("\n=== THE THREE ARMS (paired by seed) ===")
    for a in ARMS:
        c = cells.get(key_of(a))
        n = len(c["per_seed"]) if c else 0
        if not n:
            print(f"  {a[0]:10s} (no runs yet)")
            continue
        asr = [r["asr"] for r in c["per_seed"]]; acc = [r["accuracy"] for r in c["per_seed"]]
        print(f"  {a[0]:10s} n={n:2d}  mean ASR={np.mean(asr):.4f}  mean acc={np.mean(acc):.4f}  "
              f"min acc={min(acc):.4f}"
              + ("   * ARM MEAN BELOW ACC FLOOR: VOID, not negative"
                 if np.mean(acc) < ACC_FLOOR else ""))
    print("\n=== THE TWO LEGS, BOTH VERDICTS, RECOMPUTED FROM PER-SEED ROWS ===")
    for name, arm, pred in LEGS:
        r = leg(cells, arm)
        if r["mean"] is None:
            print(f"  {name} ({arm}): n={r['n']}, not computable yet")
            continue
        print(f"  {name} ({arm}), frozen prediction: {pred}")
        print(f"    n={r['n']}  mean={r['mean']:+.4f}  sd={r['sd']:.4f}  "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
        print(f"    SIGN  : {r['verdict_sign']}")
        print(f"    MARGIN: {r['verdict_margin']}")
        print(f"    ACC   : {r['acc_gate']}  (arm mean {r['arm_mean_acc']:.4f}, "
              f"per-seed min {r['arm_min_acc']:.4f}, floor {ACC_FLOOR})")


def main():
    check_frozen()
    cells, hc = load()
    if hc is None or not hc.get("all_passed"):
        sys.exit("REFUSING TO RUN: no passing --harness-check verdict is recorded in "
                 f"{out_path}.\n  Run: PYTHONPATH=. python3 -m experiments.run_oracle_free_channels "
                 "--harness-check\n  Section 4.4 orders the three checks before any new run, and "
                 "Round 69's lesson is that a verdict which is not persisted is witnessed only by a "
                 "terminal that has since closed.")
    p = premise()
    total = len(ARMS) * len(SEEDS20)
    print("=== ORACLE-FREE channel separation: rfa -> krum under pixel, n=20 ===")
    print(f"    {len(ARMS)} arms x {len(SEEDS20)} seeds = {total} runs, seeds "
          f"{SEEDS20[0]}-{SEEDS20[-1]}, none importable")
    print(f"    arm type {p.get('arm_type')}, rules frozen at {PREREG_COMMIT}")
    print(f"    premise: krum decision change {p['decision_change'][D1]:.3f} at admission change "
          f"{p['admission_change'][D1]:.3f}")
    print(f"    baseline admitted adversarial mass "
          f"{p['baseline_admission_not_a_floor']['mean_base_krum_admits_adv']:.4f} in "
          f"{p['baseline_admission_not_a_floor']['n_nonzero']} of "
          f"{p['baseline_admission_not_a_floor']['n_adversary_rounds']} adversary rounds "
          "(not a floor)")
    print(f"    frozen: |Delta_stat| inside {EQUIV_MARGIN}; Delta_mag nonzero with its interval "
          "excluding zero")
    print("    A |Delta_stat| outside the margin with an interval excluding zero REFUTES the "
          "paper's\n    headline reading and is reported as a contradiction, not as a scope "
          "condition.\n", flush=True)
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} rows already present "
              "(progress is counted from these rows, not from a log index)\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); done = 0
    for a in ARMS:
        label, d1o, so, eo, _ = a
        key = key_of(a)
        existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
        for seed in SEEDS20:
            if seed in existing:
                done += 1
                continue
            t = time.time()
            acc, asr = run_one(seed, None, D2, ATTACK, None,
                               score_only=so, emit_only=eo, d1_override=d1o)
            existing[seed] = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                              "source": "<computed here>"}
            done += 1
            cells[key] = {"d1": d1o, "d2": D2, "attack": ATTACK, "arm": label,
                          "score_only": bool(so), "emit_only": bool(eo),
                          "per_seed": [existing[s] for s in sorted(existing)]}
            save(cells, hc)
            print(f"  [{done}/{total}] {label:10s} s{seed}: acc={acc:.4f} ASR={asr:.4f} "
                  f"({time.time() - t:.0f}s)"
                  + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)
    save(cells, hc)
    report(cells)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        cells, _ = load()
        v = harness_check()
        os.makedirs(out_dir, exist_ok=True)
        save(cells, v)          # persisted, not printed and discarded
        print(f"Harness-check verdict written to {out_path}")
        sys.exit(0 if v["all_passed"] else 1)
    sys.exit(main())
