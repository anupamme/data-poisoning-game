"""The decomposition arm: Krum's decision channel moved, the attenuation channel held to 1e-6.

WHY THIS ARM EXISTS. App. D.7 ran the review-named oracle-free intervention, refuted its own
pre-registered prediction (Delta_stat = +0.2969, paired 95% CI [+0.0983, +0.4955], n=20), and then
disclosed POST HOC that the arm is not the clean probe its freeze took it for: the adversarial share of
coefficient mass moves in 12 of 12 adversary rounds by up to 0.0859, against the frozen SHARE_TOL of
1e-6 that Mode S holds to 3.5e-07. That freeze recorded the tolerance, its runner recorded the quantity
every round, and the gate was still not written. This runner writes that gate and runs the contrast the
gate admits, so that the paper's one oracle-free arm stops moving two channels at once.

THE INSTRUMENT WAS SELECTED BY MEASUREMENT, NOT BY PREFERENCE.
experiments/screen_oracle_free_transforms.py screens every deployed oracle-free d1 family at this cell,
with every threshold imported from an already-frozen constant, and returns NO_ELIGIBLE_FAMILY: the one
share-neutral family (norm_clip, at tau=5.0 where the maximum client norm is 3.0437, so every
coefficient is exactly 1) is the one that is inert, and reputation, foolsgold and rfa all move Krum's
decision AND move the share in 12 of 12 adversary rounds. That dichotomy is App. D.6's impossibility
in data. So the instrument is the constructed boundary blend of experiments/boundary_blend.py.

WHAT EACH ARM APPLIES, AND WHY IT IS NOT WHAT bed6562 FROZE. Per round, within its own run, a Krum
selection crossing on the path c(t) = 1 + t (c_rfa - 1) is located by bisecting the SHIPPED float32
krum_selection, and

    arm A applies c(t_lo)        arm B applies c(t_hi)

with the selections differing under the shipped statistic as an INVARIANT of the bisection, and the
coefficients agreeing to a gap driven down until the mask-free supremum share gap is at or below
SHARE_TOL/10. On a round where no crossing is confirmed by the shipped statistic both arms apply
c(1) = c_rfa exactly and the round carries no dose; n_flip_rounds is recorded per run, not assumed.

bed6562 froze instead the fixed 1001-point float64 Gram grid, with arm A at c(t* - h) and arm B at
c(t*) for h = 1e-3. That instrument FAILS both of its own gates, measured before any ASR existed
(results/oracle_free_decomposition_premise.json, md5 fd3627ad...): the realized share gap exceeds
SHARE_TOL in 11 of 12 dosed adversary rounds and the mask-free supremum exceeds it in 12 of 12, while
the shipped selections are identical across the pair in 2 of 12. Shrinking h around the float64
crossing is not the repair -- it drives the supremum to 3.331e-16 and the decision change to 0 of 12,
because the float64 and float32 statistics cross at different t. The freeze is left byte-untouched and
is superseded by experiments/pre_registration_oracle_free_decomposition_refined.md, which records the
failure with its measurements and freezes the bisection this runner implements.

THE SHARE GATE. Per round, in both arms, the counterfactual pair is gated on the SUPREMUM over every
nonempty proper adversary subset:

    share_gap_sup = max over S of | sum_{i in S} c_B / sum c_B  -  sum_{i in S} c_A / sum c_A |

which is computable from (c_A, c_B) alone, needs no adversary mask, and upper-bounds the realized gap
for whatever the true adversary set happens to be. It is a supremum rather than one realized subset, it
applies to every round of BOTH runs rather than only to round 0 where the two arms' stacks still
coincide, and it is ASSERTED -- this runner raises -- rather than recorded and read afterwards. The
superseding freeze specifies it in exactly this form, so it is the gate rather than a strengthening of
one; no verdict, threshold, seed list or predicted direction is touched anywhere.

ORACLE-FREENESS IS STRUCTURAL. The hook is called with (ups, seed, rnd) and never with the adversary
set; run_one passes adv_mask=None whenever d1_override is set, so any dose family reached this way
raises rather than runs; c_rfa comes from the update stack alone; the bisection compares selections,
never labels; and the share gate above is mask-free by construction.

    freeze guard + harness:  PYTHONPATH=. python3 -m experiments.run_oracle_free_decomposition --harness-check
    the ladder:              PYTHONPATH=. python3 -m experiments.run_oracle_free_decomposition
"""

import os, sys, json, time, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen suite so this arm cannot drift from the published dose arms in the
# runner, the accuracy floor, the equivalence margin or the attack implementation. run_one is reused
# rather than copied; the only change it needed is the additive stack_hook keyword, and harness check 1
# is the proof that the keyword changed nothing for every existing call site.
from experiments.run_targeted_dose import (run_one, ACC_FLOOR, EQUIV_MARGIN,  # noqa: E402
                                           ATTACK_MAP, FL_CONFIG)
from experiments.boundary_blend import (refined_boundary_pair, apply_coefficients,  # noqa: E402
                                        share_gap_sup, GRID_POINTS, H,
                                        SHARE_GAP_TARGET_DIVISOR, SHIPPED_BISECT_MAX_STEPS,
                                        self_check)
# SHARE_TOL is IMPORTED from the module that defines Mode S's own tolerance, never restated here. A
# tolerance restated in the script that has to pass it is a tolerance chosen to be passed.
from experiments.measure_admission import SHARE_TOL  # noqa: E402
# t_crit IS IMPORTED, NEVER a literal table: the small tables in this repository stop at df = 9.
from experiments.analyze_headline_cis import t_crit as _t_crit  # noqa: E402

# The SUPERSEDING freeze, committed as its own commit at a point when this arm's output path was
# verified not to exist. bed6562 is the superseded one: it stays on the tree byte-untouched, its premise
# failure is recorded in the superseding document and measured in
# results/oracle_free_decomposition_premise.json, and this runner implements the corrected instrument.
PREREG_COMMIT = "947c000"
SUPERSEDED_PREREG_COMMIT = "bed6562"

D2, ATTACK = "krum", "committed_pixel"
D1 = "fedavg"                                   # pass-through: the blend IS the upstream stage
SEEDS20 = list(range(42, 62))                   # 42-61, frozen in the pre-registration
T_FROZEN = 2.093                                # the n=20 two-sided 95% t, df=19

# (label, side, what it holds)
ARMS = [
    ("A", "low",  "identity-side selection: c(t_lo), the low side of the bisected crossing"),
    ("B", "high", "flipped selection: c(t_hi), with the mask-free supremum share gap from A at or "
                  "below SHARE_TOL/10"),
]
LEGS = [("Delta_decision", "B", "sign: interval EXCLUDES zero")]

TOL = 1e-9                                      # harness checks 1 and 2

out_dir = os.path.join(base, "results", "oracle_free_decomposition")
out_path = os.path.join(out_dir, "summary.json")
PREREG = os.path.join(base, "experiments",
                      "pre_registration_oracle_free_decomposition_refined.md")
SUPERSEDED_PREREG = os.path.join(base, "experiments",
                                 "pre_registration_oracle_free_decomposition.md")
PREMISE = os.path.join(base, "results", "oracle_free_decomposition_premise.json")
SCREEN = os.path.join(base, "results", "oracle_free_screen.json")
CHANNELS = os.path.join(base, "results", "oracle_free_channels", "summary.json")
TARGETED = os.path.join(base, "results", "targeted_dose", "summary.json")

# Harness check 1's target: a cell of the frozen suite COMPUTED there rather than imported into it, so
# recomputing it exercises the path the new keyword must leave untouched.
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
    return f"blend{arm[0]}_then_{D2}|{ATTACK}"


def make_hook(side, receipts):
    """The per-round transform for one arm, and the receipts that type each round.

    The hook receives (ups, seed, rnd) and NOTHING else. The adversary set is not passed, so the
    construction cannot read adversary identity even by accident, and the share gate it records is the
    mask-free supremum rather than a realized share.
    """
    def hook(ups, seed, rnd):
        pair = refined_boundary_pair(ups)
        c = pair["c_A"] if side == "low" else pair["c_B"]
        sup, arg = share_gap_sup(pair["c_A"], pair["c_B"])
        receipts.append({
            "seed": int(seed), "round": int(rnd), "side": side,
            "flip": bool(pair["flip"]), "t_star": pair["t_star"],
            "sel_A": pair["sel_A"], "sel_B": pair["sel_B"],
            "coeff_gap_linf": float(pair["coeff_gap_linf"]),
            "share_gap_sup_over_all_adversary_subsets": sup,
            "argmax_subset": arg,
            "share_gap_within_tol": bool(sup <= SHARE_TOL) if sup == sup else True,
            "rho_rfa": float(max(pair["c_rfa"]) / max(min(pair["c_rfa"]), 1e-12)),
            "n_bisect_steps": pair["n_bisect_steps"],
            "n_shipped_evaluations": pair["n_shipped_evaluations"],
            "search": pair["search"],
        })
        # BLOCKING, every round of every run. A neutrality claim that is recorded and not asserted is
        # exactly the defect App. D.7 disclosed about itself, and the instrument bed6562 froze fails
        # this assertion on 12 of 12 dosed rounds, which is why it was superseded rather than run.
        if sup == sup and sup > SHARE_TOL:
            raise AssertionError(
                f"share gate FAILED at seed {seed} round {rnd}: supremum share gap {sup:.3e} > "
                f"SHARE_TOL {SHARE_TOL:.0e}. The arm would be confounded in the attenuation channel. "
                f"The tolerance is not to be widened. The bisection exhausted its "
                f"{SHIPPED_BISECT_MAX_STEPS}-step cap without reaching SHARE_TOL/"
                f"{SHARE_GAP_TARGET_DIVISOR:g}; raise the cap, never the tolerance.")
        # A flip round must actually differ in selection under the SHIPPED statistic, or the dose is
        # imaginary. Under the bisection this is an invariant of the search rather than a hope, so a
        # failure here means the search itself is broken.
        if pair["flip"] and pair["sel_A"] == pair["sel_B"]:
            raise AssertionError(
                f"flip gate FAILED at seed {seed} round {rnd}: the pair was typed as a flip and its "
                "shipped selections agree, which the bisection's invariant forbids.")
        return apply_coefficients(ups, c)
    return hook


def identity_hook(receipts):
    """Harness check 2's hook: multiply every client by exactly 1.0.

    Exact in float32, so this must reproduce d1_override='fedavg' with no hook at all, hence App.
    D.7's identity arm. This is the check that the hook PATH is faithful rather than merely present:
    the standing failure mode in this repository is a transform that is never applied while the run
    completes silently.
    """
    def hook(ups, seed, rnd):
        receipts.append({"seed": int(seed), "round": int(rnd), "side": "t0"})
        return apply_coefficients(ups, [1.0] * len(ups))
    return hook


def check_frozen():
    """Refuse to start unless the predictions are committed, unmodified, and the screen exists."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the frozen predictions and the refuting branch are not committed.\n"
                 f"  1. git commit {PREREG} alone\n  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen prediction is unfalsifiable, which is the entire point of the freeze.")
    # `PREREG_COMMIT is None` is only half a guard: the moment the constant is set, the only test that
    # ever fires stops firing. So the hash is checked against the log and the tree against porcelain.
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze ({e}).")
    # An EMPTY log must fail loudly. `PREREG_COMMIT.startswith(log[:7])` is vacuously true when log is
    # "", so a path git knows nothing about would otherwise sail through the hash comparison, and the
    # only thing standing between that and a run would be the porcelain check below.
    if not log:
        sys.exit(f"REFUSING TO RUN: git has no commit touching {PREREG}. A freeze that git cannot see "
                 "is not a freeze.")
    if not log.startswith(PREREG_COMMIT[:7]) and not PREREG_COMMIT.startswith(log[:7]):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last committed at {log!r}, not {PREREG_COMMIT!r}.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted modifications ({dirty!r}). The freeze is "
                 "whatever is committed, not whatever is on disk.")
    if not os.path.exists(SCREEN):
        sys.exit(f"REFUSING TO RUN: {SCREEN} does not exist. The instrument is a CONSTRUCTED transform "
                 "only because the screen found no deployed family eligible; without the screen this "
                 "arm has no justification for preferring a construction.\n  Run: PYTHONPATH=. "
                 "python3 -m experiments.screen_oracle_free_transforms")
    scr = json.load(open(SCREEN))
    if scr.get("verdict") != "NO_ELIGIBLE_FAMILY":
        sys.exit(f"REFUSING TO RUN: the screen reports verdict {scr.get('verdict')!r}, not "
                 "NO_ELIGIBLE_FAMILY. §2 of bed6562, carried forward by §4 of the superseding freeze, "
                 "says a deployed family is PREFERRED when one is eligible, because a deployed defense "
                 "is a better instrument than a constructed one. Run that family instead.\n"
                 f"  eligible: {scr.get('eligible_families')}")
    # The superseding freeze's central promise is that nothing committed was rewritten. That promise is
    # checked here rather than trusted: if the superseded document has moved off bed6562, gone missing
    # or gone dirty, then this arm's provenance story is false and it must not run.
    if not os.path.exists(SUPERSEDED_PREREG):
        sys.exit(f"REFUSING TO RUN: {SUPERSEDED_PREREG} is missing. The superseding freeze asserts it "
                 "stays on the tree byte-untouched; a superseded freeze that has been deleted cannot "
                 "witness what it was superseded from.")
    try:
        slog = subprocess.run(["git", "log", "-1", "--format=%h", "--", SUPERSEDED_PREREG], cwd=base,
                              capture_output=True, text=True, timeout=30).stdout.strip()
        sdirty = subprocess.run(["git", "status", "--porcelain", "--", SUPERSEDED_PREREG], cwd=base,
                                capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the superseded freeze ({e}).")
    if not slog:
        sys.exit(f"REFUSING TO RUN: git has no commit touching {SUPERSEDED_PREREG}.")
    if not (slog.startswith(SUPERSEDED_PREREG_COMMIT[:7])
            or SUPERSEDED_PREREG_COMMIT.startswith(slog[:7])):
        sys.exit(f"REFUSING TO RUN: {SUPERSEDED_PREREG} was last committed at {slog!r}, not "
                 f"{SUPERSEDED_PREREG_COMMIT!r}. The superseding freeze asserts that file is "
                 "byte-untouched; append-only means the superseded document is never rewritten.")
    if sdirty:
        sys.exit(f"REFUSING TO RUN: {SUPERSEDED_PREREG} has uncommitted modifications ({sdirty!r}). "
                 "A superseded freeze is amended by nobody, including us.")
    if not os.path.exists(PREMISE):
        sys.exit(f"REFUSING TO RUN: {PREMISE} does not exist. The superseding freeze's §1-§3 rest on "
                 "that premise measurement, and the corrected instrument has no recorded justification "
                 "without it.\n  Run: PYTHONPATH=. python3 -m experiments.measure_boundary_premise")
    prem = json.load(open(PREMISE)).get("summary", {}).get("verdict", {})
    if not prem.get("refined_instrument_meets_both_gates"):
        sys.exit("REFUSING TO RUN: the premise artifact does not record the corrected instrument "
                 f"meeting both gates (verdict: {prem.get('label')!r}). The ladder is 12.5 h and the "
                 "gates are asserted per round, so it would fail partway through.")


def prereg_md5():
    """The freeze's content hash, written into the artifact so the artifact witnesses WHICH text
    governed the run rather than only which commit was claimed."""
    import hashlib
    return hashlib.md5(open(PREREG, "rb").read()).hexdigest()


def published_frozen_cell():
    if not os.path.exists(TARGETED):
        return {}
    c = json.load(open(TARGETED)).get("cells", {}).get(FROZEN_CELL[0])
    if not c:
        return {}
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c.get("per_seed", [])}


def published_identity():
    """App. D.7's identity arm: fedavg -> krum under pixel at n=20, the same seeds as this arm."""
    if not os.path.exists(CHANNELS):
        return {}
    c = json.load(open(CHANNELS)).get("cells", {}).get(f"fedavg_then_{D2}|{ATTACK}")
    if not c:
        return {}
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c.get("per_seed", [])}


def harness_check():
    """The four checks of §6, in order, with their verdicts RETURNED so main() can persist them.

    A check whose verdict is discarded leaves the claim it supports witnessed only by a terminal, and
    re-running this one costs hours.
    """
    v = {"tol": TOL, "share_tol": SHARE_TOL, "grid_points": GRID_POINTS, "h": H,
         "checks": [], "all_passed": None}

    print("=== HARNESS CHECK 1/4: stack_hook=None is bit-neutral on a frozen cell ===")
    key, mode, d2, atk, val = FROZEN_CELL
    pub = published_frozen_cell()
    if not pub:
        print(f"  CANNOT CHECK: {key} is not in {TARGETED}.")
        v["checks"].append({"check": "stack_hook_default_is_bit_neutral", "passed": False,
                            "reason": f"{key} absent from results/targeted_dose/summary.json"})
    else:
        seed = sorted(pub)[0]
        p_acc, p_asr = pub[seed]
        print(f"    {key} at seed {seed}: published acc={p_acc:.6f} ASR={p_asr:.6f}")
        print("    Recomputed through the PATCHED run_one with stack_hook unset. This is the check "
              "that\n    the one additive keyword this arm adds to shared code changed nothing.",
              flush=True)
        t = time.time()
        acc, asr = run_one(seed, mode, d2, atk, val)
        d = (acc - p_acc, asr - p_asr)
        ok = max(abs(d[0]), abs(d[1])) <= TOL
        print(f"    recomputed: acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})   "
              f"{'BIT-NEUTRAL' if ok else 'DRIFTED'}  ({time.time() - t:.0f}s)", flush=True)
        v["checks"].append({"check": "stack_hook_default_is_bit_neutral", "passed": bool(ok),
                            "cell": key, "seed": seed, "published": [p_acc, p_asr],
                            "recomputed": [float(acc), float(asr)],
                            "delta": [float(d[0]), float(d[1])]})

    print("\n=== HARNESS CHECK 2/4: the hook at t=0 reproduces App. D.7's identity arm ===")
    pub = published_identity()
    if not pub:
        print(f"  CANNOT CHECK: no identity row in {CHANNELS}.")
        v["checks"].append({"check": "hook_at_t0_reproduces_published_identity", "passed": False,
                            "reason": "results/oracle_free_channels/summary.json has no identity cell"})
    else:
        seed = sorted(pub)[0]
        p_acc, p_asr = pub[seed]
        print(f"    oracle_free_channels fedavg_then_{D2}|{ATTACK} at seed {seed}: "
              f"acc={p_acc:.6f} ASR={p_asr:.6f}")
        print("    Recomputed with a hook that multiplies every client by exactly 1.0, which is exact "
              "in\n    float32. BLOCKING: if the hook path is entered and the result still matches, "
              "the path is\n    faithful; if the hook is never called the run completes silently and "
              "arm B equals arm A.", flush=True)
        rec = []
        t = time.time()
        acc, asr = run_one(seed, None, D2, ATTACK, None, d1_override=D1,
                           stack_hook=identity_hook(rec))
        d = (acc - p_acc, asr - p_asr)
        entered = len(rec) == FL_CONFIG.num_rounds
        ok2 = max(abs(d[0]), abs(d[1])) <= TOL and entered
        print(f"    recomputed: acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})")
        print(f"    hook invoked {len(rec)} times for {FL_CONFIG.num_rounds} rounds -> "
              f"{'ENTERED' if entered else 'NEVER CALLED'}   "
              f"{'IDENTICAL' if ok2 else 'MISMATCH'}  ({time.time() - t:.0f}s)", flush=True)
        v["checks"].append({"check": "hook_at_t0_reproduces_published_identity", "passed": bool(ok2),
                            "source": "results/oracle_free_channels/summary.json", "seed": seed,
                            "published": [p_acc, p_asr], "recomputed": [float(acc), float(asr)],
                            "delta": [float(d[0]), float(d[1])],
                            "hook_invocations": len(rec), "rounds": FL_CONFIG.num_rounds,
                            "hook_entered_every_round": bool(entered)})

    print("\n=== HARNESS CHECK 3/4: the Gram proposal is validated against the shipped statistic ===")
    print("    At t=0 and t=1 on live stacks. Under the corrected instrument the Gram path only "
          "PROPOSES\n    brackets and every reported pair is confirmed by the shipped "
          "krum_selection, so this check\n    licenses the proposal and nothing more. Note: "
          "self_check's flip leg exercises the SUPERSEDED\n    pair (boundary_pair at h=1e-3), whose "
          "shipped selections agree on 2 of 12 dosed rounds; a pass\n    here is not evidence about "
          "the ladder's pair, which check 4 measures.", flush=True)
    sc = self_check(seeds=(SEEDS20[0],), rounds=3, verbose=True)
    endpoints_ok = all(r["t0_gram_equals_shipped_raw"] and r["t1_gram_equals_shipped_rfa"]
                       and r["identity_is_exact_ones"] for r in sc["per_round"])
    ok3 = bool(endpoints_ok)
    print(f"    -> {'AGREES AT BOTH ENDPOINTS' if ok3 else 'DISAGREES'}  "
          f"({sc['n_flip_rounds']}/{sc['n_rows']} rounds carried a flip on the superseded grid)")
    v["checks"].append({"check": "gram_proposal_equals_shipped_krum_selection_at_endpoints",
                        "passed": ok3,
                        "scope": "endpoint equality and exact-ones identity only; the flip leg in "
                                 "self_check concerns the superseded instrument",
                        "self_check": sc})

    print("\n=== HARNESS CHECK 4/4: the ladder's own pair passes both gates ===")
    print("    Measured on the instrument the ladder uses, refined_boundary_pair, which bisects the\n"
          "    SHIPPED float32 krum_selection. Two gates, both on the pair: the mask-free supremum\n"
          "    over all 30 nonempty proper adversary subsets must be within SHARE_TOL, and the "
          "shipped\n    selections must differ. The supremum needs no mask, so the construction "
          "stays oracle-free.", flush=True)
    prem = json.load(open(PREMISE))
    ps = prem["summary"]
    r1, r2, r3 = ps["route1_frozen"], ps["route2_wrong_predicate"], ps["route3_refined"]
    print(f"    superseded instrument, {r1['n_dosed_adversary_rounds']} dosed rounds: supremum over "
          f"tolerance in {r1['n_sup_exceeding_tol']}, shipped selections identical in "
          f"{r1['n_selections_identical_under_shipped']}  -> FAILS BOTH GATES")
    print(f"    shrinking h on the float64 predicate: supremum max {r2['max_sup_gap']:.3e} but "
          f"selections differ in {r2['n_selections_differ_under_shipped']} of "
          f"{r2['n_dosed_adversary_rounds']}  -> NEUTRAL AND UNINFORMATIVE")
    print(f"    the ladder's instrument: both gates pass in "
          f"{r3['n_dosed_adversary_rounds']} of {r3['n_dosed_adversary_rounds']}, supremum max "
          f"{r3['max_sup_gap']:.3e} vs tol {SHARE_TOL:.0e}, {r3['bisect_steps_min_max']} steps")
    # The artifact is the population evidence. A passing verdict is never inherited from a file alone,
    # so the ladder's OWN pair function is also run live here, in this process, on fresh stacks.
    print("    live, in this process, on refined_boundary_pair:", flush=True)
    live = self_check(seeds=(SEEDS20[0],), rounds=3, verbose=True,
                      pair_fn=refined_boundary_pair)
    live_ok = bool(live["all_pass"]
                   and live["all_flip_rounds_within_tol_on_supremum"] is not False
                   and live["all_flip_rounds_differ_under_shipped"] is not False)
    print(f"    live: pair_fn={live['pair_fn']}, {live['n_flip_rounds']}/{live['n_rows']} flip "
          f"rounds, max supremum {live['max_share_gap_sup_on_flip_rounds']:.3e} vs tol "
          f"{SHARE_TOL:.0e} -> {'PASSES' if live_ok else 'FAILS'}")
    ok4 = bool(r3["share_gate_passes_every_dosed_round"]
               and r3["flip_gate_passes_every_dosed_round"]
               and prem["summary"]["verdict"]["refined_instrument_meets_both_gates"]
               and r3["max_sup_gap"] <= SHARE_TOL
               and live_ok)
    print(f"    -> {'BOTH GATES PASS' if ok4 else 'A GATE FAILED'}")
    v["checks"].append({"check": "ladder_pair_passes_share_and_flip_gates", "passed": bool(ok4),
                        "share_tol": SHARE_TOL,
                        "share_gap_target": SHARE_TOL / float(SHARE_GAP_TARGET_DIVISOR),
                        "premise_artifact": "results/oracle_free_decomposition_premise.json",
                        "route1_superseded": r1, "route2_wrong_predicate": r2,
                        "route3_ladder_instrument": r3,
                        "live_in_process": live,
                        "note": "The ladder additionally asserts the mask-free supremum and the "
                                "shipped selection difference every round of both runs, and raises "
                                "rather than recording a violation."})

    v["all_passed"] = all(c["passed"] for c in v["checks"])
    print("\n  " + ("ALL FOUR CHECKS PASSED. The ladder may run."
                    if v["all_passed"] else
                    "A CHECK FAILED. The ladder must NOT run; the failure is the result."))
    return v


def load():
    if not os.path.exists(out_path):
        return {}, None
    try:
        d = json.load(open(out_path))
        return d.get("cells", {}), d.get("harness_check")
    except Exception:
        return {}, None


def paired(cells, label):
    arm = next(a for a in ARMS if a[0] == label)
    rows = cells.get(key_of(arm), {}).get("per_seed", [])
    return ({int(r["seed"]): float(r["asr"]) for r in rows},
            {int(r["seed"]): float(r["accuracy"]) for r in rows})


def leg(cells, label):
    """The paired B - A difference, with BOTH verdicts, recomputed from per-seed rows in one call.

    Both legs of the difference are read here, in this call, so a subtrahend can never come from a
    different seed count -- or a different arm -- than its minuend. Nothing is transcribed.
    """
    b_asr, b_acc = paired(cells, label)
    a_asr, a_acc = paired(cells, "A")
    seeds = sorted(set(a_asr) & set(b_asr))
    r = {"arm": label, "minuend": label, "subtrahend": "A", "n": len(seeds), "seeds": seeds,
         "arm_mean_asr": float(np.mean([b_asr[s] for s in seeds])) if seeds else None,
         "reference_mean_asr": float(np.mean([a_asr[s] for s in seeds])) if seeds else None,
         "arm_mean_acc": float(np.mean([b_acc[s] for s in seeds])) if seeds else None,
         "arm_min_acc": float(min(b_acc[s] for s in seeds)) if seeds else None,
         "reference_mean_acc": float(np.mean([a_acc[s] for s in seeds])) if seeds else None,
         "reference_min_acc": float(min(a_acc[s] for s in seeds)) if seeds else None,
         "per_seed_delta": {str(s): b_asr[s] - a_asr[s] for s in seeds}}
    if len(seeds) < 2:
        r.update({"mean": None, "sd": None, "ci95": None, "verdict_sign": "NOT COMPUTED",
                  "verdict_margin": "NOT COMPUTED", "acc_gate": "NOT COMPUTED"})
        return r
    d = np.array([b_asr[s] - a_asr[s] for s in seeds], dtype=float)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = t_crit(len(d)) * sd / np.sqrt(len(d))
    lo, hi = m - hw, m + hw
    # Two verdicts, computed independently. An interval can sit inside the margin AND exclude zero, so
    # neither may ever be reported as the other, and neither literal names a mechanism.
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
                                      and r["reference_mean_acc"] >= ACC_FLOOR) else
                           "FAIL -- this arm is VOID, not negative: a low ASR at collapsed accuracy "
                           "is not suppression"),
              "acc_floor": ACC_FLOOR})
    return r


def dose_summary(cells):
    """What the instrument actually delivered, counted from receipts rather than assumed.

    A round with no flip on the path is bit-identical across arms and carries no dose. The
    pre-registration says so in advance; this is the measurement of how often it happened.
    """
    out = {}
    for a in ARMS:
        rows = cells.get(key_of(a), {}).get("per_seed", [])
        flips = [r.get("n_flip_rounds") for r in rows if r.get("n_flip_rounds") is not None]
        gaps = [r.get("max_share_gap_sup") for r in rows if r.get("max_share_gap_sup") is not None]
        cg = [r.get("max_coeff_gap_linf") for r in rows if r.get("max_coeff_gap_linf") is not None]
        out[a[0]] = {
            "n_runs": len(rows),
            "mean_flip_rounds_per_run": float(np.mean(flips)) if flips else None,
            "min_flip_rounds_per_run": int(min(flips)) if flips else None,
            "max_flip_rounds_per_run": int(max(flips)) if flips else None,
            "rounds_per_run": FL_CONFIG.num_rounds,
            "max_share_gap_sup_over_all_runs": float(max(gaps)) if gaps else None,
            "max_coeff_gap_linf_over_all_runs": float(max(cg)) if cg else None,
            "share_tol": SHARE_TOL,
        }
    return out


def save(cells, hc):
    verdicts = {name: leg(cells, arm) for name, arm, _ in LEGS}
    json.dump({
        "description":
            "Krum's decision channel moved WITHOUT adversary identity and with the attenuation "
            "channel held to Mode S's own tolerance. Arms straddle a Krum selection crossing on the "
            "blend c(t) = 1 + t (c_rfa - 1), located by bisecting the SHIPPED float32 krum_selection: "
            "arm A at c(t_lo), arm B at c(t_hi). App. D.7's oracle-free arm moved both channels at "
            "once (share moved in 12 of 12 adversary rounds, up to 0.0859, against SHARE_TOL 1e-6); "
            "this arm separates them. Predictions frozen at "
            f"{PREREG_COMMIT} "
            "(experiments/pre_registration_oracle_free_decomposition_refined.md), which supersedes "
            f"{SUPERSEDED_PREREG_COMMIT} without amending it.",
        "prereg_commit": PREREG_COMMIT,
        "prereg": "experiments/pre_registration_oracle_free_decomposition_refined.md",
        "prereg_md5": prereg_md5(),
        "superseded_prereg": {
            "path": "experiments/pre_registration_oracle_free_decomposition.md",
            "commit": SUPERSEDED_PREREG_COMMIT,
            "left_byte_untouched": True,
            "why": "Its instrument -- the fixed 1001-point float64 Gram grid at h=1e-3 -- fails both "
                   "of its own gates, measured before any ASR existed: the realized share gap exceeds "
                   "SHARE_TOL in 11 of 12 dosed adversary rounds (max 8.586e-05), the mask-free "
                   "supremum exceeds it in 12 of 12 dosed rounds and 13 of 13 flip rounds (max "
                   "1.0586e-04), and the shipped selections are IDENTICAL across the pair in 2 of 12. "
                   "The 5.109e-07 its §3 quoted as the existence proof is the minimum over those 12 "
                   "rounds, the only one that passes. Its §§1,2,4,5,7,8,9 are carried forward "
                   "unchanged; only its §3 instrument and §6 gate wording are superseded.",
            "premise_artifact": "results/oracle_free_decomposition_premise.json",
            "premise_artifact_md5": "fd3627adcafdb1aad96c76727e8486d5"},
        "dataset": "cifar10", "model": "cifar_cnn",
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round, "f": 0.2,
                   "alpha": 0.5, "rounds": FL_CONFIG.num_rounds, "seeds": SEEDS20,
                   "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                   "share_tol": SHARE_TOL,
                   "share_gap_target_divisor": SHARE_GAP_TARGET_DIVISOR,
                   "share_gap_target": SHARE_TOL / float(SHARE_GAP_TARGET_DIVISOR),
                   "bisect_max_steps": SHIPPED_BISECT_MAX_STEPS,
                   "coarse_proposal_grid_points": GRID_POINTS, "coarse_proposal_h": H,
                   "t_crit_n20": t_crit(len(SEEDS20))},
        "arm": {"d2": D2, "attack": ATTACK, "attack_impl": ATTACK_MAP[ATTACK], "d1": D1,
                "arms": [{"label": a[0], "side": a[1], "holds": a[2], "key": key_of(a)}
                         for a in ARMS],
                "frozen_prediction": {
                    "Delta_decision": "the decision channel alone, with the attenuation channel "
                                      "closed to 1e-6, DOES carry suppression: interval excludes "
                                      "zero"},
                "refuting_branch": "|Delta| > 0.15 with the interval excluding zero in the direction "
                                   "of INCREASED ASR contradicts this paper's headline reading with "
                                   "App. D.7's confound removed, and is reported as a contradiction "
                                   "rather than converted into a scope condition.",
                "unresolved_branch": "The flip direction is uncontrolled and non-flip rounds are "
                                     "bit-identical across arms, so an interval inside the margin "
                                     "containing zero is UNRESOLVED, not 'no effect'. It is not "
                                     "evidence of absence, and the equivalence phrase defined at "
                                     "supplementary.tex:441 may not be applied to it."},
        "instrument_selected_by_measurement": {
            "screen": "results/oracle_free_screen.json",
            "verdict": "NO_ELIGIBLE_FAMILY",
            "why_a_construction": "Every deployed oracle-free d1 family at this cell fails: norm_clip "
                                  "is share-neutral in 0 of 12 rounds and inert (tau=5.0, max client "
                                  "norm 3.0437, every coefficient exactly 1), while reputation, "
                                  "foolsgold and rfa each move Krum's decision AND move the share in "
                                  "12 of 12. A deployed family is PREFERRED when one is eligible; "
                                  "none is."},
        "oracle_free": "The hook is called with (ups, seed, rnd) and never with the adversary set. "
                       "c_rfa comes from the update stack alone, the bisection compares selections "
                       "rather than labels, run_one passes adv_mask=None whenever d1_override is set "
                       "so any dose family reached this way RAISES, and the share gate is the "
                       "mask-free supremum over every adversary subset rather than a realized share.",
        "share_gate": {
            "implemented_as": "within-run counterfactual pair, supremum over every nonempty proper "
                              "adversary subset, asserted every round of both runs",
            "why_not_across_arms": "A decision flip changes the aggregate, so the two arms' "
                                   "trajectories diverge by construction and their round-r stacks "
                                   "are different objects after the first flip. Across arms the "
                                   "quantity is well defined only at round 0.",
            "relation_to_the_freeze": "§3 of the superseding freeze specifies the gate in exactly "
                                      "this form, so this is the gate rather than a strengthening of "
                                      "one. Against bed6562's §6 sentence it is stricter in three "
                                      "ways -- a supremum rather than one realized subset, every "
                                      "round of both runs rather than the rounds where the stacks "
                                      "coincide, and asserted rather than recorded -- and that "
                                      "precisification was written before any ASR existed. No "
                                      "verdict, threshold, seed list or predicted direction is "
                                      "touched by it.",
            "stopping_rule": "The bisection stops when the supremum share gap is at or below "
                             "SHARE_TOL / SHARE_GAP_TARGET_DIVISOR. The step cap is a COST bound: a "
                             "round that exhausts it without reaching the target FAILS the gate and "
                             "raises. The tolerance is never widened to admit a round.",
            "share_tol": SHARE_TOL},
        "not_a_defense": "No deployed Krum interpolates its input stack toward a geometric-median "
                         "reweighting and stops one grid step short of a selection flip. This arm is "
                         "an instrument: it answers the adversary-identity objection and not the "
                         "deployability one.",
        "scope": "One aggregator, one attack, one dataset, one upstream transform, one architecture. "
                 "This supports no statement of the form 'the negative fails without an oracle' or "
                 "its converse; what it licenses is 'in this cell, without adversary identity, with "
                 "the coefficient share held to 1e-6'. The contrast is LOCAL, at the decision "
                 "boundary: both arms are nearly-rfa stacks at adjacent points on the blend path, so "
                 "it is not 'identity against a statistic disturbance' and must not be reported as "
                 "one. No composed-pair ASR result on a second dataset exists anywhere in this paper "
                 "and this arm does not change that.",
        "theory_scope": "App. D.6's impossibility -- no oracle-free positive per-client rescaling is "
                        "both EXACTLY share-neutral and informative -- is not contradicted. The "
                        "blend's gap is positive, so it is consistent with the theorem. What this arm "
                        "adds is the tolerance-parameterized statement: APPROXIMATE uniform "
                        "neutrality is achievable at any tolerance, and the price is paid in "
                        "informativeness, which is confined to rounds carrying a flip.",
        "may_not_be_written": "The sentence that the negative reproduces oracle-free may not be "
                              "written anywhere under any outcome. App. D.7 refuted its own "
                              "prediction and that refutation stands unamended.",
        "harness_check": hc,
        "dose_delivered": dose_summary(cells),
        "verdicts": verdicts,
        "verdict_reading_rule": "verdict_sign and verdict_margin are independent labels and neither "
                                "may be reported as the other: an interval can sit inside the margin "
                                "and still exclude zero. Neither literal names a mechanism; the "
                                "measured direction is printed beside it, never inside it.",
        "cells": cells}, open(out_path, "w"), indent=2)


def report(cells):
    print("\n=== THE TWO ARMS (paired by seed) ===")
    for a in ARMS:
        c = cells.get(key_of(a))
        rows = c["per_seed"] if c else []
        if not rows:
            print(f"  arm {a[0]} (no runs yet)")
            continue
        asr = [r["asr"] for r in rows]; acc = [r["accuracy"] for r in rows]
        print(f"  arm {a[0]}  n={len(rows):2d}  mean ASR={np.mean(asr):.4f}  "
              f"mean acc={np.mean(acc):.4f}  min acc={min(acc):.4f}"
              + ("   * ARM MEAN BELOW ACC FLOOR: VOID, not negative"
                 if np.mean(acc) < ACC_FLOOR else ""))
    d = dose_summary(cells)
    print("\n=== THE DOSE THE INSTRUMENT DELIVERED (counted from receipts) ===")
    for k, s in d.items():
        if not s["n_runs"]:
            continue
        print(f"  arm {k}: flip rounds per run mean {s['mean_flip_rounds_per_run']:.1f} "
              f"[{s['min_flip_rounds_per_run']}, {s['max_flip_rounds_per_run']}] of "
              f"{s['rounds_per_run']}; max supremum share gap "
              f"{s['max_share_gap_sup_over_all_runs']:.3e} vs tol {s['share_tol']:.0e}")
    print("\n=== THE LEG, BOTH VERDICTS, RECOMPUTED FROM PER-SEED ROWS ===")
    for name, arm, pred in LEGS:
        r = leg(cells, arm)
        if r["mean"] is None:
            print(f"  {name} (arm {arm} - arm A): n={r['n']}, not computable yet")
            continue
        print(f"  {name} (arm {arm} - arm A), frozen prediction: {pred}")
        print(f"    n={r['n']}  mean={r['mean']:+.4f}  sd={r['sd']:.4f}  "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
        print(f"    SIGN  : {r['verdict_sign']}")
        print(f"    MARGIN: {r['verdict_margin']}")
        print(f"    ACC   : {r['acc_gate']}  (arm mean {r['arm_mean_acc']:.4f}, "
              f"per-seed min {r['arm_min_acc']:.4f}, floor {ACC_FLOOR})")
        if r["verdict_sign"].startswith("NOT RESOLVED"):
            print("    UNRESOLVED IS NOT A ZERO: the flip direction is uncontrolled and non-flip "
                  "rounds carry\n    no dose, so this is not evidence of absence.")


def main():
    check_frozen()
    cells, hc = load()
    if hc is None or not hc.get("all_passed"):
        sys.exit("REFUSING TO RUN: no passing --harness-check verdict is recorded in "
                 f"{out_path}.\n  Run: PYTHONPATH=. python3 -m "
                 "experiments.run_oracle_free_decomposition --harness-check\n  §6 orders the four "
                 "checks before any new run, and a verdict that is not persisted is witnessed only "
                 "by a terminal that has since closed.")
    total = len(ARMS) * len(SEEDS20)
    print("=== DECOMPOSITION arm: decision moved, attenuation held to 1e-6, oracle-free ===")
    print(f"    {len(ARMS)} arms x {len(SEEDS20)} seeds = {total} runs, seeds "
          f"{SEEDS20[0]}-{SEEDS20[-1]}, both arms NEW (no imported leg, no mixed n)")
    print(f"    grid {GRID_POINTS} points, h={H:g}; SHARE_TOL={SHARE_TOL:.0e} imported from "
          "measure_admission.py")
    print(f"    rules frozen at {PREREG_COMMIT}, md5 {prereg_md5()}")
    print("    instrument selected by results/oracle_free_screen.json -> NO_ELIGIBLE_FAMILY")
    print("    frozen: the decision channel alone DOES carry suppression (interval excludes zero)")
    print("    A |Delta| > 0.15 excluding zero toward INCREASED ASR refutes this paper's headline\n"
          "    reading and is reported as a contradiction, not as a scope condition. An interval\n"
          "    inside the margin containing zero is UNRESOLVED, not evidence of absence.\n",
          flush=True)
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} rows already present "
              "(progress is counted from these rows, not from a log index)\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); done = 0
    for a in ARMS:
        label, side, _ = a
        key = key_of(a)
        existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
        for seed in SEEDS20:
            if seed in existing:
                done += 1
                continue
            t = time.time()
            rec = []
            acc, asr = run_one(seed, None, D2, ATTACK, None, d1_override=D1,
                              stack_hook=make_hook(side, rec))
            if len(rec) != FL_CONFIG.num_rounds:
                sys.exit(f"REFUSING TO CONTINUE: the hook was invoked {len(rec)} times for "
                         f"{FL_CONFIG.num_rounds} rounds at seed {seed}. A transform that is not "
                         "applied every round makes arm B equal to arm A and the run still "
                         "completes; that is the failure mode this check exists for.")
            gaps = [r["share_gap_sup_over_all_adversary_subsets"] for r in rec
                    if r["share_gap_sup_over_all_adversary_subsets"]
                    == r["share_gap_sup_over_all_adversary_subsets"]]
            existing[seed] = {
                "seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                "n_flip_rounds": int(sum(1 for r in rec if r["flip"])),
                "max_share_gap_sup": float(max(gaps)) if gaps else None,
                "max_coeff_gap_linf": float(max(r["coeff_gap_linf"] for r in rec)),
                "mean_t_star_on_flip_rounds":
                    float(np.mean([r["t_star"] for r in rec if r["flip"]]))
                    if any(r["flip"] for r in rec) else None,
                "per_round": rec,
                "source": "<computed here>"}
            done += 1
            cells[key] = {"arm": label, "side": side, "d1": D1, "d2": D2, "attack": ATTACK,
                          "per_seed": [existing[s] for s in sorted(existing)]}
            save(cells, hc)
            print(f"  [{done}/{total}] arm {label} s{seed}: acc={acc:.4f} ASR={asr:.4f}  "
                  f"flips={existing[seed]['n_flip_rounds']}/{FL_CONFIG.num_rounds}  "
                  f"max share gap={existing[seed]['max_share_gap_sup']:.2e}  "
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
