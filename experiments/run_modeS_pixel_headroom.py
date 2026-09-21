"""Mode S on krum / committed_pixel: the first cell where BOTH margin legs bind.

WHY THIS ARM EXISTS, AND THE DEFECT IS OURS. The paper's flagship dissociation is Mode S on
krum / committed_scaling, and that cell cannot move in either of the two channels it is asked about.

  (i) Admission has nowhere to fall. Krum admits 0.000 adversarial mass at every rung of that cell,
      in 0 of 12 adversary rounds. The paper says so and declines to lean on it.
  (ii) The margin's lower leg has nowhere to fall either, and this the paper did not say. The arm's
      practical-equivalence reading is a two-sided TOST against a frozen +/-0.15. ASR is bounded
      below by zero, so for ANY dosed outcome whatsoever

          mean_i (asr_dosed_i - asr_identity_i)  >=  -mean_i asr_identity_i

      and on that cell the controlled identity mean is 0.0439. The largest fall arithmetically
      available is 0.0439, so the leg Delta > -0.15 is satisfied before a single run happens. It is
      not a test. experiments/measure_margin_reachability.py measures this over every equivalence
      reading in the paper (results/margin_reachability.json): 3 of the 4 readings the comparability
      ladder's own conjunct admits have an arithmetically satisfied lower leg.

THIS ARM MOVES THE CELL AND HOLDS EVERYTHING ELSE. Same aggregator, dataset, architecture, N, K, f,
alpha, round count and seed grid; only the attack changes, and it changes both headrooms at once:
baseline admission 0.3333, nonzero in 4 of 12 adversary rounds (results/oracle_free_admission.json),
and identity mean ASR 0.3232 > 0.15, so the lower leg is a genuine test here. It is also App. D.8's
own host cell, where the oracle-free boundary arm returned +0.3218 (95% CI [+0.1680, +0.4757], n=20)
AGAINST this paper's headline reading, so a masked Mode S arm here discriminates between two live
explanations of that contradiction -- the cell, or the boundary construction. The freeze registers no
prediction for Delta_full's sign for exactly that reason: two of our own measurements on this cell
point opposite ways, and picking one now would be choosing which of our own results to believe.

NO SHARED CODE CHANGES. run_one already reaches this cell through (mode="S", d2="krum",
attack_name="committed_pixel", val, score_only); ARMS_S drives only run_targeted_dose.py's own loop
and is not consulted. Neither previously-granted keyword exception (d1_override, stack_hook) is used,
and no existing runner, pre-registration or artifact is touched. This runner imports run_one.

THE STAGING RULE IS IN THE FREEZE, NOT HERE. Leg B runs first; leg D runs only if Delta_full does not
fire the refuting branch, because the decomposition of a refutation is a separate question that this
pre-registration does not cover. The rule cannot suppress a result that goes against us: the refuting
branch is the branch that STOPS further work, so staging can only ever withhold a control that would
have followed a favourable result. If leg D is not run the artifact records leg_D_run false with its
reason, and no sentence anywhere may describe the score-only control on this cell as having been run.

WHAT THIS CANNOT ESTABLISH. No admission quantity -- run_one returns (accuracy, asr) and nothing
else, the 0.3333 / 4-of-12 premise is IMPORTED and not re-measured, and summary[...|admission] is 0.0
at every identity rung by construction and is never read as a level. The arm is MASKED, like every
Mode S rung in the paper, so it answers nothing about the oracle-free objection and does not weaken
App. D.8's contradiction. The sentence that the negative reproduces oracle-free may not be written
anywhere, under any outcome.

    freeze guard + harness:  PYTHONPATH=. python3 -m experiments.run_modeS_pixel_headroom --harness-check
    the arm:                 PYTHONPATH=. python3 -m experiments.run_modeS_pixel_headroom
"""

import os, sys, json, time, subprocess, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the frozen suite so this arm cannot drift from the published dose arms in the
# runner, the accuracy floor, the equivalence margin, the FL config or the attack implementation.
# run_one is reused rather than copied, and it is reused UNCHANGED: this arm needs no keyword.
from experiments.run_targeted_dose import (run_one, cell_key, dial, ACC_FLOOR,  # noqa: E402
                                           EQUIV_MARGIN, ADV_FRACTION, ATTACK_MAP, FL_CONFIG)
# The coefficient constructor Mode S's premise is a statement about. Imported so that harness check 3
# tests the shipped function rather than a restatement of it.
from experiments.run_all_compositions import dose_coefficients_statistic_only  # noqa: E402
# t_crit IS IMPORTED, NEVER a literal table: the small tables in this repository stop at df = 9.
from experiments.analyze_headline_cis import t_crit  # noqa: E402

# Committed by itself, at a point when both of this arm's output paths were verified not to exist.
# `PREREG_COMMIT is None` is only half a guard -- the moment the constant is set, the only test that
# ever fires stops firing -- so check_frozen() also reads git log and git status for the path.
PREREG_COMMIT = "c89e963"

D2, ATTACK = "krum", "committed_pixel"
SEEDS20 = list(range(42, 62))                    # 42-61, frozen in the pre-registration
LO, HI = 0.0, 2.0                                # the frozen endpoint convention; kappa, not rho
TOL = 1e-9                                       # harness checks 1 and 2: bit-equality, not "close"

# (label, kappa, score_only, imported, what it is)
ARMS = [
    ("A", LO, False, True,  "identity rung: mode S at kappa=0 returns the stack unwrapped"),
    ("B", HI, False, False, "full dose: mode S at kappa=2, rho=54.598150033144236"),
    ("C", LO, True,  True,  "identity rung, score-only: no coefficient to split at the identity"),
    ("D", HI, True,  False, "score-only dose: the statistic sees the transform, the payload does not"),
]
# (name, minuend, subtrahend, what it is). Both legs of each difference are read in ONE call so a
# subtrahend can never come from a different seed count -- or a different arm -- than its minuend.
LEGS = [
    ("Delta_full", "B", "A", "mode S endpoint contrast, full dose, paired at 20 seeds"),
    ("Delta_score", "D", "C", "the magnitude-closed control: the same contrast with the emitted "
                              "update left at its original scale"),
]

out_dir = os.path.join(base, "results", "modeS_pixel_headroom")
out_path = os.path.join(out_dir, "summary.json")
PREREG = os.path.join(base, "experiments", "pre_registration_modeS_pixel_headroom.md")
# The imported identity rows, and the two premises this arm's choice of cell rests on. All three are
# opened for reading only; nothing under results/ that already exists is written.
CHANNELS = os.path.join(base, "results", "oracle_free_channels", "summary.json")
ADMISSION = os.path.join(base, "results", "oracle_free_admission.json")
REACHABILITY = os.path.join(base, "results", "margin_reachability.json")
IDENTITY_CELL = f"fedavg_then_{D2}|{ATTACK}"


def key_of(arm):
    """doseS_kappa<K>_then_krum|committed_pixel, plus the |score_only suffix convention already in
    results/emit_only_topup/. cell_key is imported so the rung naming cannot drift from the frozen
    ladders; the suffix is appended rather than reimplemented."""
    k = cell_key("S", D2, ATTACK, arm[1])
    return k + "|score_only" if arm[2] else k


def check_frozen():
    """Refuse to start unless the predictions are committed, unmodified, and the premises exist."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the frozen branches are not committed.\n"
                 f"  1. git commit {PREREG} alone\n  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen prediction is unfalsifiable, which is the entire point of the freeze.")
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze ({e}).")
    # An EMPTY log must fail loudly. PREREG_COMMIT.startswith(log[:7]) is vacuously true when log is
    # "", so a path git knows nothing about would otherwise sail through the hash comparison.
    if not log:
        sys.exit(f"REFUSING TO RUN: git has no commit touching {PREREG}. A freeze git cannot see is "
                 "not a freeze.")
    if not (log.startswith(PREREG_COMMIT[:7]) or PREREG_COMMIT.startswith(log[:7])):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last committed at {log!r}, not {PREREG_COMMIT!r}.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted modifications ({dirty!r}). The freeze "
                 "is whatever is committed, not whatever is on disk.")
    # The three premises the freeze cites. Each is a MEASUREMENT this arm imports rather than repeats,
    # so its absence means this arm's justification for choosing this cell is not on disk.
    for path, why in (
            (CHANNELS, "App. D.7's identity rows are this arm's imported legs A and C (§4 of the "
                       "freeze), and without them there is nothing to import and nothing for harness "
                       "check 1 to compare against"),
            (ADMISSION, "the 0.3333 / 4-of-12 baseline admission is the first half of why this cell "
                        "was chosen (§2), and this arm cannot re-measure it: run_one returns "
                        "(accuracy, asr) and no admission quantity at all"),
            (REACHABILITY, "the identity mean 0.3232 > 0.15 is the second half (§1-§2): it is what "
                           "makes the lower TOST leg a genuine test here rather than arithmetic")):
        if not os.path.exists(path):
            sys.exit(f"REFUSING TO RUN: {os.path.relpath(path, base)} does not exist. {why}.")
    if not json.load(open(CHANNELS)).get("cells", {}).get(IDENTITY_CELL, {}).get("per_seed"):
        sys.exit(f"REFUSING TO RUN: {os.path.relpath(CHANNELS, base)} has no per-seed rows for "
                 f"{IDENTITY_CELL}. Legs A and C are imported from exactly that cell.")


def prereg_md5():
    """The freeze's content hash, written into the artifact so the artifact witnesses WHICH text
    governed the run rather than only which commit was claimed."""
    import hashlib
    return hashlib.md5(open(PREREG, "rb").read()).hexdigest()


def published_identity():
    """{seed: (accuracy, asr)} for App. D.7's fedavg -> krum rows under pixel, seeds 42-61.

    Mode S at kappa = 0 returns the update list UNWRAPPED (run_all_compositions.py, mode-S branch)
    before the permutation is drawn, and the permutation when drawn uses a local
    np.random.default_rng([seed, round]) rather than the global stream, so doseS_kappa0.0_then_krum is
    the same computation as a fedavg pass-through at the same seed. That argument is why these rows
    can be imported; harness check 1 is why they may be.
    """
    c = json.load(open(CHANNELS)).get("cells", {}).get(IDENTITY_CELL, {})
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"])) for r in c.get("per_seed", [])}


def imported_rows():
    """Legs A and C, materialized from the published rows with their provenance on every row.

    A and C are the SAME rows: at the identity there is no coefficient to split between the score
    channel and the emitted update. That is asserted by harness check 2, not assumed here.
    """
    pub = published_identity()
    return {a[0]: [{"seed": int(s), "accuracy": pub[s][0], "asr": pub[s][1],
                    "source": f"imported from results/oracle_free_channels/summary.json "
                              f"cells[{IDENTITY_CELL}]",
                    "licensed_by": "harness check 1 (full) / 2 (score-only): bit-for-bit at seed "
                                   f"{min(pub)}, tolerance {TOL:g} on accuracy and ASR"}
                   for s in sorted(pub) if s in SEEDS20]
            for a in ARMS if a[3]}


def harness_check():
    """The three blocking checks of §4, with their verdicts RETURNED so main() can persist them.

    A check whose verdict is discarded leaves the claim it supports witnessed only by a terminal that
    has since closed, and checks 1 and 2 cost about eleven minutes each to repeat.
    """
    v = {"tol": TOL, "checks": [], "all_passed": None}
    pub = published_identity()
    seed = sorted(pub)[0]
    p_acc, p_asr = pub[seed]

    # --- 1 and 2: the imported identity rung, full and score-only ---------------------------------
    # The mechanism is argued in published_identity(). The one asymmetry is what makes these blocking
    # rather than decorative: the imported rows ran with adv_mask=None (via d1_override="fedavg") and
    # this path passes the REAL mask, because d1_override is unset. If either fails, the import is
    # void, legs A and C become 40 further runs, and no reconciliation is attempted.
    for n, (label, score_only, why) in enumerate((
            ("A", False, "mode S at kappa=0 returns the stack unwrapped, so the masked call must "
                         "reproduce the mask-free published row bit-for-bit"),
            ("C", True, "at the identity there is no coefficient to split between the score channel "
                        "and the emitted update, which is exactly why it is asserted")), start=1):
        print(f"\n=== HARNESS CHECK {n}/3: imported leg {label} reproduces the published identity "
              f"row{' (score-only)' if score_only else ''} ===")
        print(f"    {IDENTITY_CELL} at seed {seed}: published acc={p_acc:.6f} ASR={p_asr:.6f}")
        print(f"    {why}.")
        print(f"    BLOCKING, tolerance {TOL:g} on both: not 'close', bit-for-bit.", flush=True)
        t = time.time()
        acc, asr = run_one(seed, "S", D2, ATTACK, LO, score_only=score_only)
        d = (acc - p_acc, asr - p_asr)
        ok = max(abs(d[0]), abs(d[1])) <= TOL
        print(f"    recomputed: acc={acc:.6f} ASR={asr:.6f}   d=({d[0]:+.2e}, {d[1]:+.2e})   "
              f"{'IDENTICAL' if ok else 'DRIFTED -- THE IMPORT IS VOID'}  ({time.time() - t:.0f}s)",
              flush=True)
        v["checks"].append({
            "check": f"imported_leg_{label}_is_bit_identical_to_published_identity",
            "passed": bool(ok), "score_only": bool(score_only),
            "source": "results/oracle_free_channels/summary.json", "cell": IDENTITY_CELL,
            "seed": int(seed), "published": [p_acc, p_asr],
            "recomputed": [float(acc), float(asr)], "delta": [float(d[0]), float(d[1])],
            "asymmetry_this_check_exists_for": "the published row ran with adv_mask=None via "
                                               "d1_override='fedavg'; this call passes the real mask",
            "on_failure": "the import is void: legs A and C are re-run as 40 further runs and no "
                          "reconciliation is attempted"})

    # --- 3: Mode S pins the adversary, every round --------------------------------------------------
    # This is the premise the entire Mode S family rests on, and this cell has never carried a Mode S
    # arm before. It is a property of the shipped coefficient constructor, which is a pure function of
    # (adv_mask, kappa, dose_key), so it is checked over EVERY mask the sampler could draw at every
    # one of the 50 rounds of every one of the 20 seeds rather than over the realized sequence. That
    # is strictly stronger than per-round: it establishes the property for whatever mask each round
    # realizes, without a hook to observe the realized mask -- and §8 of the freeze uses no hook.
    print(f"\n=== HARNESS CHECK 3/3: mode S pins the adversary on every round ===")
    K = FL_CONFIG.clients_per_round
    n_adv_total = int(FL_CONFIG.num_clients * ADV_FRACTION)
    masks = [m for m in itertools.product([False, True], repeat=K)]
    realizable = [m for m in masks if sum(m) <= n_adv_total]
    print(f"    every adversarial coefficient exactly 1.0, and the adversarial share of coefficient")
    print(f"    mass equal to n_adv/K to float read-back, at kappa={HI} (rho={dial('S', HI)!r}).")
    print(f"    Checked over all {len(masks)} masks of K={K} participants x {FL_CONFIG.num_rounds} "
          f"rounds x {len(SEEDS20)} seeds")
    print(f"    = {len(masks) * FL_CONFIG.num_rounds * len(SEEDS20)} coefficient vectors; "
          f"{len(realizable)} of the masks are realizable at f={ADV_FRACTION} "
          f"(n_adv <= {n_adv_total}).", flush=True)
    worst_coeff, worst_share, degenerate, n_checked = 0.0, 0.0, 0, 0
    worst_at = None
    for s in SEEDS20:
        for rnd in range(FL_CONFIG.num_rounds):
            for m in masks:
                c = np.asarray(dose_coefficients_statistic_only(list(m), HI, (s, rnd)), dtype=float)
                n_checked += 1
                na = sum(m)
                # Exactly 1.0, not approximately: c is initialized to ones and adversarial slots are
                # never written, so any deviation means the constructor has changed shape.
                ca = max((abs(c[i] - 1.0) for i, a in enumerate(m) if a), default=0.0)
                share = float(c[[i for i, a in enumerate(m) if a]].sum() / c.sum()) if na else 0.0
                sg = abs(share - na / float(K))
                if ca > worst_coeff or sg > worst_share:
                    worst_at = {"seed": int(s), "round": int(rnd), "mask": list(map(bool, m))}
                worst_coeff, worst_share = max(worst_coeff, ca), max(worst_share, sg)
                # Fewer than two benign participants admits no dispersion and returns the identity.
                # That is a degenerate round carrying no dose; it cannot happen at f=0.2 with K=5, and
                # counting it is how we would find out if that ever stopped being true.
                if HI != 0.0 and (K - na) < 2 and m in realizable:
                    degenerate += 1
    ok3 = worst_coeff == 0.0 and worst_share <= 1e-12 and degenerate == 0
    print(f"    max |c_adv - 1| over {n_checked} vectors : {worst_coeff:.3e}  (must be exactly 0)")
    print(f"    max |share - n_adv/K|                    : {worst_share:.3e}")
    print(f"    realizable masks with fewer than 2 benign: {degenerate}  (a degenerate round carries "
          f"no dose)")
    print(f"    -> {'ADVERSARY PINNED' if ok3 else 'MODE S PREMISE FAILS ON THIS CELL'}", flush=True)
    v["checks"].append({
        "check": "mode_S_pins_the_adversary_every_round", "passed": bool(ok3),
        "kappa": HI, "rho": dial("S", HI), "K": K, "rounds": FL_CONFIG.num_rounds,
        "seeds": SEEDS20, "n_coefficient_vectors_checked": int(n_checked),
        "max_abs_adversarial_coefficient_minus_one": float(worst_coeff),
        "max_abs_share_minus_n_adv_over_K": float(worst_share),
        "share_tolerance": 1e-12,
        "realizable_masks_with_fewer_than_two_benign": int(degenerate),
        "worst_case_at": worst_at,
        "masks": f"exhaustive over all {len(masks)} subsets of K={K} participants, of which "
                 f"{len(realizable)} are realizable at f={ADV_FRACTION}",
        "why_exhaustive_rather_than_realized": "the realized per-round mask is not observable without "
                                              "a stack_hook, and §8 of the freeze uses none. The "
                                              "coefficient constructor is a pure function of "
                                              "(adv_mask, kappa, dose_key), so checking every mask at "
                                              "every round establishes the property for whatever mask "
                                              "each round realizes. This is a STRENGTHENING of the "
                                              "frozen check, not a substitute for it.",
        "trains_nothing": True})

    v["all_passed"] = all(c["passed"] for c in v["checks"])
    print("\n  " + ("ALL THREE CHECKS PASSED. The arm may run." if v["all_passed"] else
                    "AT LEAST ONE CHECK FAILED. The arm may not run."))
    return v


def load():
    if not os.path.exists(out_path):
        return {}, None
    try:
        d = json.load(open(out_path))
        return d.get("cells", {}), d.get("harness_check")
    except Exception:
        return {}, None


def rows_of(cells, label):
    arm = next(a for a in ARMS if a[0] == label)
    return cells.get(key_of(arm), {}).get("per_seed", [])


def paired(cells, label):
    rows = rows_of(cells, label)
    return ({int(r["seed"]): float(r["asr"]) for r in rows},
            {int(r["seed"]): float(r["accuracy"]) for r in rows})


def _ci(d):
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = float(t_crit(len(d)) * sd / np.sqrt(len(d)))
    return {"n": int(len(d)), "mean": m, "sd": sd, "se": float(sd / np.sqrt(len(d))),
            "t_crit": float(t_crit(len(d))), "half_width": hw, "ci95": [m - hw, m + hw]}


def _verdicts(ci):
    """The three verdicts, computed independently. NONE may be reported as another.

    They are genuinely independent: an interval can sit inside the margin and still exclude zero, and
    a one-sided upper bound can hold while a two-sided reading fails. No literal names a mechanism --
    a frozen literal that fused a label with a mechanism guess was backwards in sign once already in
    this project -- so the measured direction is printed beside the label and never inside it.
    """
    lo, hi, m = ci["ci95"][0], ci["ci95"][1], ci["mean"]
    return {
        "verdict_sign": ("excludes_zero" if lo * hi > 0 else "contains_zero"),
        "verdict_sign_reading": ("the interval EXCLUDES zero" if lo * hi > 0 else
                                 "NOT RESOLVED at this n (the interval contains zero); this is not "
                                 "evidence of absence"),
        "measured_direction": ("positive" if m > 0 else "negative" if m < 0 else "zero"),
        "verdict_margin": ("inside_margin" if abs(m) < EQUIV_MARGIN else "outside_margin"),
        "verdict_margin_reading": (f"|mean| {'<' if abs(m) < EQUIV_MARGIN else '>='} the frozen "
                                   f"equivalence margin {EQUIV_MARGIN}"),
        "verdict_upper": ("non_increase_bounded" if hi < EQUIV_MARGIN else "not_bounded"),
        "verdict_upper_reading": (f"the 95% upper bound {hi:+.4f} is "
                                  f"{'below' if hi < EQUIV_MARGIN else 'at or above'} the margin "
                                  f"{EQUIV_MARGIN}"),
        "margin": EQUIV_MARGIN,
    }


def _branch(ci, lower_leg_is_a_test):
    """Which of the freeze's four named branches fired, decided by the runner and not afterwards."""
    lo, hi, m = ci["ci95"][0], ci["ci95"][1], ci["mean"]
    excl, inside = lo * hi > 0, abs(m) < EQUIV_MARGIN
    if excl and not inside and m > 0:
        return ("REFUTING", "The interval excludes zero with |Delta| outside the margin toward "
                            "INCREASED ASR. The flagship dissociation fails on the one cell where "
                            "admission is live and both margin legs bind, so the published negative "
                            "is specific to a floor cell. Reported as a contradiction in the body "
                            "and NOT converted into a scope condition afterwards.")
    if excl and inside:
        return ("SMALL_CONSISTENT_RISE", "The interval excludes zero while the bound stays inside "
                                        "the margin: non-increase FAILS and practical non-increase "
                                        "holds, and the report must say both. This is not an "
                                        "equivalence result, and the equivalence phrase defined at "
                                        "supplementary.tex:441 against the stricter of our two "
                                        "interval conventions is not applied to it.")
    if excl and not inside:
        return ("FALL_OUTSIDE_MARGIN", "The interval excludes zero toward DECREASED ASR with "
                                       "|Delta| outside the margin. Not a branch the freeze names "
                                       "as confirming: it is a large suppression by the dose, "
                                       "reported as measured.")
    if inside:
        return ("CONFIRMING", "The interval contains zero and |Delta| is inside the margin"
                              + (", with the lower leg a genuine test on this cell, so both verdicts "
                                 "mean here what they only appeared to mean on the flagship cell."
                                 if lower_leg_is_a_test else
                                 " -- but the lower leg is NOT a test on this seed block, so the "
                                 "two-sided reading is not earned and only the upper bound is."))
    return ("UNRESOLVED", "The interval contains zero and is wider than the margin. That is "
                          "unresolved, not no effect, and it is not evidence of absence.")


def leg(cells, name, minuend, subtrahend, what):
    """One estimand, with BOTH its seed blocks, recomputed from per-seed rows in a single call.

    Nothing is transcribed and both legs are read here, so a subtrahend can never come from a
    different seed count -- or a different arm -- than its minuend. Two blocks are reported because
    §5 of the freeze makes a sub-floor rung VOID rather than negative: the floor-filtered Delta is
    primary and the retained-value Delta is reported beside it, and when no seed is void they are the
    same number, which is the usual case and is stated rather than left to inference.
    """
    b_asr, b_acc = paired(cells, minuend)
    a_asr, a_acc = paired(cells, subtrahend)
    seeds = sorted(set(a_asr) & set(b_asr))
    r = {"estimand": name, "minuend": minuend, "subtrahend": subtrahend, "what": what,
         "n_paired_seeds": len(seeds), "paired_seeds": seeds,
         "minuend_mean_asr": float(np.mean([b_asr[s] for s in seeds])) if seeds else None,
         "subtrahend_mean_asr": float(np.mean([a_asr[s] for s in seeds])) if seeds else None,
         "minuend_mean_acc": float(np.mean([b_acc[s] for s in seeds])) if seeds else None,
         "minuend_min_acc": float(min(b_acc[s] for s in seeds)) if seeds else None,
         "subtrahend_mean_acc": float(np.mean([a_acc[s] for s in seeds])) if seeds else None,
         "subtrahend_min_acc": float(min(a_acc[s] for s in seeds)) if seeds else None,
         "per_seed_delta": {str(s): b_asr[s] - a_asr[s] for s in seeds}}
    # The reachability arithmetic, on THIS seed block rather than quoted from the freeze. ASR >= 0, so
    # the identity mean is the largest fall the paired mean could possibly take; if it does not exceed
    # the margin, the lower TOST leg is satisfied for every conceivable outcome and is not a test.
    a_mean = r["subtrahend_mean_asr"]
    r["reachability"] = {
        "identity_mean_asr": a_mean, "fall_available": a_mean, "margin": EQUIV_MARGIN,
        "lower_leg_is_a_test": bool(a_mean > EQUIV_MARGIN) if a_mean is not None else None,
        "lower_leg_slack": (a_mean - EQUIV_MARGIN) if a_mean is not None else None,
        "bound": "mean_i(asr_dosed_i - asr_identity_i) >= -mean_i(asr_identity_i), because ASR >= 0",
        "why_this_cell_was_chosen": "on the flagship cell the identity mean is 0.0439 < 0.15 and the "
                                    "lower leg is satisfied by arithmetic for any possible outcome"}
    # The frozen conjunction the paper's margin reading is stated under, kept attached rather than
    # dropped: the margin CONJOINED with every rung mean below 0.5.
    r["rung_means_below_half"] = (
        {"lo": r["subtrahend_mean_asr"], "hi": r["minuend_mean_asr"],
         "all_below_0.5": bool(max(r["subtrahend_mean_asr"], r["minuend_mean_asr"]) < 0.5),
         "note": "the identity rung is 0.3232, so a rise past +0.1768 breaks the conjunction "
                 "independently of the margin"} if seeds else None)
    if len(seeds) < 2:
        r.update({"floor_filtered": None, "all_seeds": None, "verdicts": None,
                  "branch": "NOT COMPUTED", "acc_gate": "NOT COMPUTED"})
        return r
    void = sorted(s for s in seeds if min(b_acc[s], a_acc[s]) < ACC_FLOOR)
    kept = [s for s in seeds if s not in void]
    r["void_seeds"] = void
    r["acc_floor"] = ACC_FLOOR
    r["acc_gate"] = ("PASS: no rung of any retained seed falls below the accuracy floor" if not void
                     else f"{len(void)} seed(s) VOID, not negative: a low ASR at collapsed accuracy "
                          f"is not suppression. Dropped from the primary Delta; the retained-value "
                          f"Delta over all {len(seeds)} seeds is reported beside it.")
    r["all_seeds"] = _ci(np.array([b_asr[s] - a_asr[s] for s in seeds], dtype=float))
    r["floor_filtered"] = (_ci(np.array([b_asr[s] - a_asr[s] for s in kept], dtype=float))
                           if len(kept) >= 2 else None)
    r["primary"] = "floor_filtered" if void else "all_seeds"
    r["identical_blocks"] = not void
    ci = r["floor_filtered"] if (void and r["floor_filtered"]) else r["all_seeds"]
    r["verdicts"] = _verdicts(ci)
    r["verdicts"]["computed_on"] = r["primary"]
    r["branch"], r["branch_reading"] = _branch(ci, r["reachability"]["lower_leg_is_a_test"])
    return r


def refutes(cells):
    """Did Delta_full fire the refuting branch? The staging rule of §7 reads this and nothing else.

    Delta_full's spec is taken from LEGS rather than restated, so the estimand the staging rule reads
    cannot drift from the estimand the artifact reports.
    """
    spec = next(l for l in LEGS if l[0] == "Delta_full")
    r = leg(cells, *spec)
    return (r["branch"] == "REFUTING"), r


def save(cells, hc, leg_D_run, leg_D_reason):
    verdicts = {name: leg(cells, name, mi, su, what) for name, mi, su, what in LEGS}
    adm = json.load(open(ADMISSION)).get("baseline_admission_not_a_floor", {})
    json.dump({
        "description":
            "Mode S on krum / committed_pixel at n=20: the paper's first dissociation cell where "
            "BOTH readings of preservation have somewhere to fall. The flagship cell "
            "(krum / committed_scaling) admits 0.000 adversarial mass at every rung AND has an "
            "identity mean ASR of 0.0439, so with ASR bounded below by zero the lower TOST leg is "
            "satisfied by arithmetic for any possible outcome and is not a test. Here baseline "
            "admission is 0.3333 (nonzero in 4 of 12 adversary rounds) and the identity mean is "
            "0.3232 > 0.15, so the lower leg is a genuine test. Branches frozen at "
            f"{PREREG_COMMIT} (experiments/pre_registration_modeS_pixel_headroom.md), which "
            "registers NO prediction for Delta_full's sign.",
        "prereg_commit": PREREG_COMMIT,
        "prereg": "experiments/pre_registration_modeS_pixel_headroom.md",
        "prereg_md5": prereg_md5(),
        "dataset": "cifar10", "model": "cifar_cnn",
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "tau": 5.0, "seeds": SEEDS20, "kappa_lo": LO, "kappa_hi": HI,
                   "rho_lo": dial("S", LO), "rho_hi": dial("S", HI),
                   "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN,
                   "t_crit_n20": t_crit(len(SEEDS20)), "bit_equality_tol": TOL},
        "arm": {"d2": D2, "attack": ATTACK, "attack_impl": ATTACK_MAP[ATTACK],
                "arms": [{"label": a[0], "kappa": a[1], "score_only": a[2], "imported": a[3],
                          "is": a[4], "key": key_of(a)} for a in ARMS],
                "endpoints_only": "kappa = 0 and kappa = 2. The interior rungs (0.5, 1.0) are NOT "
                                  "run and no Jonckheere-Terpstra trend statistic is computed or "
                                  "reported for this arm: a two-rung cell is not a four-rung ladder "
                                  "and will not be described as one.",
                "no_registered_sign": "The freeze registers no prediction for Delta_full's sign. Two "
                                      "of our own measurements on this cell point opposite ways -- "
                                      "the flagship's -0.010 under a masked dose on the scaling "
                                      "attack, and App. D.8's +0.3218 [+0.1680, +0.4757] under an "
                                      "oracle-free construction on this very cell -- and registering "
                                      "a direction would be choosing which of our own results to "
                                      "believe. What is registered is the branch structure.",
                "refuting_branch": "Delta_full's interval excluding zero with |Delta| > 0.15 toward "
                                   "INCREASED ASR. Reported as a contradiction in the body and not "
                                   "converted into a scope condition afterwards."},
        "why_this_cell": {
            "baseline_admission": {"mean_base_krum_admits_adv": adm.get("mean_base_krum_admits_adv"),
                                   "n_nonzero": adm.get("n_nonzero"),
                                   "n_adversary_rounds": adm.get("n_adversary_rounds"),
                                   "source": "results/oracle_free_admission.json -> "
                                             "baseline_admission_not_a_floor",
                                   "imported_not_remeasured": True},
            "flagship_admission": "0.000, nonzero in 0 of 12 adversary rounds: the zero is a floor "
                                  "there and the paper already declines to lean on it",
            "reachability": "results/margin_reachability.json: of the 4 equivalence readings the "
                            "comparability ladder's own conjunct admits, 3 have an arithmetically "
                            "satisfied lower leg -- the flagship, its EMNIST replication, and the "
                            "flagship's outcome-gated twin. One (coord_median / scaling, "
                            "outcome-gated) is genuinely two-sided and keeps its reading.",
            "also_app_D8_host_cell": "The oracle-free boundary arm returned +0.3218 (95% CI "
                                     "[+0.1680, +0.4757], n=20) against this paper's headline "
                                     "reading on this cell, so a masked arm here discriminates "
                                     "between the CELL and the BOUNDARY CONSTRUCTION as explanations "
                                     "of that contradiction."},
        "imported_legs": {
            "source": "results/oracle_free_channels/summary.json",
            "cell": IDENTITY_CELL,
            "why_the_same_computation": "mode S at kappa=0 returns the update list unwrapped before "
                                        "the permutation is drawn, and the permutation when drawn "
                                        "uses a local np.random.default_rng([seed, round]) rather "
                                        "than the global stream",
            "asymmetry": "the published rows ran with adv_mask=None via d1_override='fedavg' and this "
                         "arm passes the real mask, which is why harness checks 1 and 2 are blocking "
                         "rather than decorative",
            "legs": [a[0] for a in ARMS if a[3]],
            "A_and_C_are_the_same_rows": True},
        "staging": {
            "rule": "Leg B runs first. Leg D runs only if Delta_full does not fire the refuting "
                    "branch, because the decomposition of a refutation is a separate question that "
                    "this pre-registration does not cover.",
            "cannot_suppress_an_unfavourable_result": "The refuting branch is the branch that STOPS "
                                                      "further work, so the staging can only ever "
                                                      "withhold a control that would have followed a "
                                                      "favourable result.",
            "leg_D_run": bool(leg_D_run),
            "leg_D_reason": leg_D_reason,
            "if_not_run": "no sentence anywhere may describe the score-only control on this cell as "
                          "having been run"},
        "no_shared_code_change": "run_one already reaches this cell through (mode='S', d2='krum', "
                                 "attack_name='committed_pixel', val, score_only). ARMS_S drives "
                                 "only run_targeted_dose.py's own loop and is not consulted. Neither "
                                 "previously-granted keyword exception (d1_override, stack_hook) is "
                                 "used, and no existing runner, pre-registration or artifact is "
                                 "touched.",
        "no_admission_quantity": "run_one returns (accuracy, asr) and nothing else. Admission is "
                                 "measured by the measure_admission* family, which this arm does not "
                                 "run, so the 0.3333 / 4-of-12 premise above is IMPORTED and is not "
                                 "an admission result of this arm. summary[...|admission] is 0.0 at "
                                 "every identity rung BY CONSTRUCTION and is never read as a level.",
        "masked": "This arm reads adversary identity by construction, like every Mode S rung in the "
                  "paper. It answers nothing about the oracle-free objection and does not weaken App. "
                  "D.8's contradiction, which stands as measured.",
        "may_not_be_written": "The sentence that the negative reproduces oracle-free may not be "
                              "written anywhere, under any outcome.",
        "scope": "One aggregator, one attack, one dataset, one architecture, two rungs. No claim of "
                 "the form 'the dissociation holds generally' is licensed; what is licensed is 'in "
                 "this cell, at n=20, with both margin legs binding and admission off the floor'. No "
                 "composed-pair ASR result on a second dataset exists anywhere in this paper and "
                 "this arm does not change that. The attack is non-adaptive by construction, so "
                 "nothing here concerns adaptive robustness.",
        "changes_no_published_row": "The flagship's -0.010 [-0.032, +0.012] stands exactly as "
                                    "published, with its lower leg relabelled as arithmetic rather "
                                    "than retracted. No frozen threshold, seed list, interval "
                                    "convention or published verdict is revised.",
        "verdict_reading_rule": "verdict_sign, verdict_margin and verdict_upper are independent "
                                "labels and none may be reported as another: an interval can sit "
                                "inside the margin and still exclude zero, and a one-sided bound can "
                                "hold while a two-sided reading fails. No literal names a mechanism; "
                                "the measured direction is printed beside it, never inside it.",
        "harness_check": hc,
        "verdicts": verdicts,
        "cells": cells}, open(out_path, "w"), indent=2)


def report(cells):
    print("\n=== THE FOUR LEGS (paired by seed) ===")
    for a in ARMS:
        rows = rows_of(cells, a[0])
        if not rows:
            print(f"  leg {a[0]} {key_of(a)}: no rows"
                  + ("   (conditional on the staging rule)" if a[0] == "D" else ""))
            continue
        asr = [r["asr"] for r in rows]; acc = [r["accuracy"] for r in rows]
        print(f"  leg {a[0]}  n={len(rows):2d}  mean ASR={np.mean(asr):.4f}  "
              f"mean acc={np.mean(acc):.4f}  min acc={min(acc):.4f}  "
              f"{'imported' if a[3] else 'computed here'}"
              + ("   * LEG MEAN BELOW ACC FLOOR" if np.mean(acc) < ACC_FLOOR else ""))
    print("\n=== THE ESTIMANDS: THREE VERDICTS EACH, READ SEPARATELY ===")
    for name, mi, su, what in LEGS:
        r = leg(cells, name, mi, su, what)
        if not r.get("verdicts"):
            print(f"  {name} (leg {mi} - leg {su}): n={r['n_paired_seeds']}, not computable yet")
            continue
        ci = r[r["primary"]]
        print(f"  {name} (leg {mi} - leg {su}), on {r['primary']}"
              + ("" if r["identical_blocks"] else f"; {len(r['void_seeds'])} void seed(s)"))
        print(f"    n={ci['n']}  mean={ci['mean']:+.4f}  sd={ci['sd']:.4f}  "
              f"95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]")
        v = r["verdicts"]
        print(f"    SIGN  : {v['verdict_sign']:14s} {v['verdict_sign_reading']}")
        print(f"    MARGIN: {v['verdict_margin']:14s} {v['verdict_margin_reading']}")
        print(f"    UPPER : {v['verdict_upper']:14s} {v['verdict_upper_reading']}")
        rc = r["reachability"]
        print(f"    LOWER LEG: identity mean {rc['identity_mean_asr']:.4f}, so the largest possible "
              f"fall is {rc['fall_available']:.4f} -> "
              f"{'A GENUINE TEST' if rc['lower_leg_is_a_test'] else 'SATISFIED BY ARITHMETIC'}")
        print(f"    CONJUNCTION: every rung mean below 0.5: {r['rung_means_below_half']['all_below_0.5']}")
        print(f"    ACC   : {r['acc_gate']}")
        print(f"    BRANCH: {r['branch']} -- {r['branch_reading']}")


def run_leg(cells, hc, label, total_new, done_offset, leg_D_run, leg_D_reason):
    """One leg's 20 runs, resumable, with the artifact rewritten after every single run.

    Progress is counted from the rows already in the artifact, never from a log index: a [i/N] index
    counts resumed-and-skipped runs and can go backwards across restarts.
    """
    arm = next(a for a in ARMS if a[0] == label)
    _, kappa, score_only, _, what = arm
    key = key_of(arm)
    existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
    done = done_offset
    for seed in SEEDS20:
        if seed in existing:
            done += 1
            continue
        t = time.time()
        acc, asr = run_one(seed, "S", D2, ATTACK, kappa, score_only=score_only)
        existing[seed] = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                          "source": "<computed here>"}
        done += 1
        cells[key] = {"leg": label, "kappa": kappa, "rho": dial("S", kappa),
                      "score_only": bool(score_only), "d2": D2, "attack": ATTACK, "is": what,
                      "per_seed": [existing[s] for s in sorted(existing)]}
        save(cells, hc, leg_D_run, leg_D_reason)
        print(f"  [{done}/{total_new} new runs complete] leg {label} s{seed}: acc={acc:.4f} "
              f"ASR={asr:.4f}  ({time.time() - t:.0f}s)"
              + ("  * below acc floor: VOID, not negative" if acc < ACC_FLOOR else ""), flush=True)
    return done


def main():
    check_frozen()
    cells, hc = load()
    if hc is None or not hc.get("all_passed"):
        sys.exit("REFUSING TO RUN: no passing --harness-check verdict is recorded in "
                 f"{out_path}.\n  Run: PYTHONPATH=. python3 -m "
                 "experiments.run_modeS_pixel_headroom --harness-check\n  §4 orders the three "
                 "blocking checks before any new run, and a verdict that is not persisted is "
                 "witnessed only by a terminal that has since closed.")
    os.makedirs(out_dir, exist_ok=True)
    # Legs A and C are the imported identity rows, licensed by harness checks 1 and 2 and marked with
    # their provenance on every row. They are materialized into the artifact so that every number
    # behind an estimand is visible in one file rather than resolved by a reader across two.
    for label, imported in imported_rows().items():
        arm = next(a for a in ARMS if a[0] == label)
        cells[key_of(arm)] = {"leg": label, "kappa": arm[1], "rho": dial("S", arm[1]),
                              "score_only": bool(arm[2]), "d2": D2, "attack": ATTACK, "is": arm[4],
                              "imported_from": "results/oracle_free_channels/summary.json",
                              "imported_cell": IDENTITY_CELL, "per_seed": imported}

    print("=== MODE S ON KRUM / COMMITTED_PIXEL: BOTH MARGIN LEGS BIND, ADMISSION OFF THE FLOOR ===")
    print(f"    legs A and C imported (n={len(rows_of(cells, 'A'))}); legs B and D are "
          f"{len(SEEDS20)} new runs each, seeds {SEEDS20[0]}-{SEEDS20[-1]}, no mixed n anywhere")
    print(f"    endpoints only: kappa {LO} (rho {dial('S', LO):g}) and {HI} "
          f"(rho {dial('S', HI):.12f}); no interior rungs, no trend statistic")
    print(f"    branches frozen at {PREREG_COMMIT}, md5 {prereg_md5()}")
    print("    NO prediction is registered for Delta_full's sign: the flagship measured -0.010 under "
          "a masked\n    dose on the scaling attack and App. D.8 measured +0.3218 under an "
          "oracle-free construction on\n    THIS cell, and choosing between them now would be "
          "choosing which of our own results to believe.")
    print("    Refuting branch, named in advance: interval excludes zero with |Delta| > "
          f"{EQUIV_MARGIN} toward\n    INCREASED ASR. That is reported as a contradiction and is NOT "
          "converted into a scope condition.")
    print("    Leg D is CONDITIONAL: it runs only if Delta_full does not fire the refuting branch.\n",
          flush=True)
    n_present = sum(len(rows_of(cells, a[0])) for a in ARMS if not a[3])
    if n_present:
        print(f"  resuming: {n_present} new-run rows already present (progress is counted from these "
              "rows, not from a log index)\n", flush=True)

    t0 = time.time()
    leg_D_run, leg_D_reason = False, "leg B has not finished; the staging rule has not been evaluated"
    save(cells, hc, leg_D_run, leg_D_reason)
    done = run_leg(cells, hc, "B", 2 * len(SEEDS20), 0, leg_D_run, leg_D_reason)

    refuted, full = refutes(cells)
    print("\n=== THE STAGING RULE, EVALUATED ON Delta_full AND NOTHING ELSE ===")
    # A staging rule read off an uncomputed estimand would silently choose the permissive branch, so
    # it is an error state rather than a default: leg B just finished, so this cannot be uncomputable.
    if full["branch"] == "NOT COMPUTED":
        save(cells, hc, False, "NOT RUN: Delta_full was not computable after leg B, so the staging "
                               "rule could not be evaluated. This is an error state, not a decision.")
        sys.exit(f"REFUSING TO CONTINUE: Delta_full is not computable at "
                 f"n={full['n_paired_seeds']} paired seed(s) after leg B finished. The staging rule "
                 "must not default to the permissive branch when its input is missing.")
    ci = full[full["primary"]]
    print(f"    Delta_full = {ci['mean']:+.4f}, 95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}], "
          f"n={ci['n']}  ->  branch {full['branch']}")
    if refuted:
        leg_D_reason = ("NOT RUN. Delta_full fired the REFUTING branch, and the decomposition of a "
                        "refutation is a separate question that this pre-registration does not "
                        "cover. The honest report is the refutation with the decomposition named as "
                        "unrun; no sentence may describe the score-only control on this cell as "
                        "having been run.")
        print(f"    Leg D is NOT RUN. {leg_D_reason}", flush=True)
    else:
        leg_D_run, leg_D_reason = True, (f"RUN. Delta_full fired branch {full['branch']}, not the "
                                         "refuting branch, so the decomposition question stands.")
        print(f"    {leg_D_reason} Running leg D ({len(SEEDS20)} runs).\n", flush=True)
        done = run_leg(cells, hc, "D", 2 * len(SEEDS20), done, leg_D_run, leg_D_reason)

    save(cells, hc, leg_D_run, leg_D_reason)
    report(cells)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        cells, _ = load()
        v = harness_check()
        os.makedirs(out_dir, exist_ok=True)
        save(cells, v, False, "the harness check has run; no leg has run yet")
        print(f"Harness-check verdict written to {out_path}")
        sys.exit(0 if v["all_passed"] else 1)
    sys.exit(main())
