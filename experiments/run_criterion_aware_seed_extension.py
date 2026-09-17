"""
Seed top-up of the adaptive FoolsGold->RFA arm, to n=30 (Round 69, ninth review).

Rules frozen at 580854a (experiments/pre_registration_criterion_aware_topup.md). This file is
ORCHESTRATION ONLY: `run_one` is imported from experiments/run_criterion_aware_adversary.py so the
physics cannot drift between the frozen seeds 42-46 and the new seeds 47-71.

WHAT IS RUN AND WHAT IS REUSED
  Run here  : the NUMERATOR only -- condition ca_eps1_decorr (eps=1.0, decorrelate=True) at seeds
              47-71, 25 runs, ~0.73 h/run, ~18 h. Written to results/criterion_aware_topup/ ONLY.
  Reused    : the DENOMINATOR -- results/fg_rfa_flagship/summary.json's
              base_composition/committed_pixel, which ALREADY spans seeds 42-71 (n=30). Recomputing a
              leg that exists at the same 30 seeds would buy nothing.

The reuse is a claim about two runners agreeing, so it is PROVED BY VALUE before it is used
(`--verify-denominator`, also run automatically before the analysis): the two runners' committed_pixel
rows are compared per seed on the five seeds where both exist, and the arm refuses to report a pooled
number unless they agree to < 1e-9. They currently agree at 0.000000000 in both ASR and accuracy.

BOTH LEGS ARE RECOMPUTED AT THE 30 SHARED SEEDS. A top-up that moves only the minuend hides a mixed-n
comparison inside the subtrahend, which is exactly how the published 3.7x came to be a 5-seed ratio
with an n=30 denominator sitting on disk beside it.

Pre-registered arithmetic, from the freeze: the denominator alone rises 0.0452 -> 0.0644 from n=5 to
n=30, so the ratio falls to 2.588x EVEN IF the adaptive mean is unchanged at 0.1668. main.tex:489 and
main.tex:2222 both quote 3.7x and both are expected to move down.

Usage (from the repository root):
    PYTHONPATH=. python3 -m experiments.run_criterion_aware_seed_extension --harness-check
    PYTHONPATH=. python3 -m experiments.run_criterion_aware_seed_extension
    PYTHONPATH=. python3 -m experiments.run_criterion_aware_seed_extension --analyze
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore")

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

# run_one is IMPORTED, never copied: the frozen seeds and the new seeds must be the same computation
# by construction rather than by inspection. FL_CONFIG/ADV_FRACTION/DEFENSE/CONDITIONS come with it so
# the configuration this arm claims can be asserted against the one the frozen suite actually used.
from experiments.run_criterion_aware_adversary import (          # noqa: E402
    run_one, CONDITIONS, FL_CONFIG, ADV_FRACTION, DEFENSE, SEEDS as FROZEN_SEEDS,
)
from experiments.analyze_headline_cis import t_crit              # noqa: E402  NOT a literal T95 table

PREREG = "experiments/pre_registration_criterion_aware_topup.md"
PREREG_COMMIT = "580854a"

LABEL = "ca_eps1_decorr"
EPS = 1.0
DECORR = True
SEEDS_NEW = list(range(47, 72))          # 47-71 inclusive, 25 seeds, fixed by the freeze
HARNESS_SEED = 42                        # a PUBLISHED seed: at a new seed the check would be vacuous
TOL = 1e-9

# The frozen ladder this arm draws one condition from. Asserted, not assumed.
assert (LABEL, EPS, DECORR) in CONDITIONS, (
    f"{LABEL} (eps={EPS}, decorrelate={DECORR}) is not a condition of the frozen suite: {CONDITIONS}")
assert ("committed_pixel", None, False) in CONDITIONS
assert FROZEN_SEEDS == [42, 43, 44, 45, 46], FROZEN_SEEDS
assert not (set(SEEDS_NEW) & set(FROZEN_SEEDS)), "the top-up must not re-run a frozen seed"

FROZEN_ARM = os.path.join(base, "results", "criterion_aware_adversary", "summary.json")
FLAGSHIP = os.path.join(base, "results", "fg_rfa_flagship", "summary.json")
out_dir = os.path.join(base, "results", "criterion_aware_topup")
out_path = os.path.join(out_dir, "summary.json")


# --------------------------------------------------------------------------------------------------
# The freeze
# --------------------------------------------------------------------------------------------------
def check_frozen():
    """A hash that nobody checks is a claim, not a freeze.

    Three things must hold, each with its own failure mode: (a) the commit resolves, or the hash is a
    typo pointing at nothing; (b) the pre-registration exists AT that commit, or the freeze names a
    commit that does not contain the rules; (c) the committed blob equals the working copy byte for
    byte, or the file was edited after the freeze and the run would be scored against rules that are
    not the ones on record. rev-parse and hash-object write nothing.
    """
    path = os.path.join(base, PREREG)
    if not os.path.exists(path):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the rules are not frozen.\n"
                 f"  1. git add {PREREG} && git commit\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes the prediction unfalsifiable, which is the entire point.")
    try:
        head = subprocess.run(["git", "-C", base, "rev-parse", "--verify", f"{PREREG_COMMIT}^{{commit}}"],
                              capture_output=True, text=True)
        if head.returncode != 0:
            sys.exit(f"REFUSING TO RUN: PREREG_COMMIT {PREREG_COMMIT} does not resolve to a commit in "
                     f"{base}. The freeze names nothing.")
        committed = subprocess.run(["git", "-C", base, "rev-parse", f"{PREREG_COMMIT}:{PREREG}"],
                                   capture_output=True, text=True)
        if committed.returncode != 0:
            sys.exit(f"REFUSING TO RUN: {PREREG} does not exist at commit {PREREG_COMMIT}. The hash "
                     "names a commit that does not contain the pre-registration.")
        working = subprocess.run(["git", "-C", base, "hash-object", PREREG],
                                 capture_output=True, text=True)
        if working.returncode != 0:
            sys.exit(f"REFUSING TO RUN: cannot hash {PREREG} to compare it against the freeze.")
        if committed.stdout.strip() != working.stdout.strip():
            sys.exit(f"REFUSING TO RUN: {PREREG} has been EDITED since it was frozen at "
                     f"{PREREG_COMMIT}.\n"
                     f"  committed blob: {committed.stdout.strip()[:12]}\n"
                     f"  working blob:   {working.stdout.strip()[:12]}\n"
                     "The rules on disk are not the rules on record. Revert the file, or commit the\n"
                     "amendment AS an amendment and set PREREG_COMMIT to the new hash -- but a post-hoc\n"
                     "edit to a decision rule is not an amendment, it is the thing pre-registration "
                     "forbids.")
    except FileNotFoundError:
        sys.exit("REFUSING TO RUN: git is not available, so the freeze cannot be verified. An "
                 "unverifiable freeze is not a freeze.")
    print(f"  freeze verified: {PREREG} @ {PREREG_COMMIT}, blob equal to the working copy")


# --------------------------------------------------------------------------------------------------
# Checkpointing. Writes results/criterion_aware_topup/ and nothing else.
# --------------------------------------------------------------------------------------------------
def load_or_init():
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {
        "description": ("Seed top-up of the adaptive FoolsGold->RFA arm to n=30: condition "
                        f"{LABEL} at seeds {SEEDS_NEW[0]}-{SEEDS_NEW[-1]}. The numerator only; the "
                        "committed_pixel denominator is REUSED from results/fg_rfa_flagship/, which "
                        "already spans seeds 42-71, after a per-seed value comparison against "
                        "results/criterion_aware_adversary/ on the five overlapping seeds."),
        "prereg_commit": PREREG_COMMIT,
        "defense": "foolsgold_then_rfa",
        "config": {"num_clients": FL_CONFIG.num_clients, "clients_per_round": FL_CONFIG.clients_per_round,
                   "adversarial_fraction": ADV_FRACTION, "num_rounds": FL_CONFIG.num_rounds,
                   "model": "cifar_cnn", "defense": list(DEFENSE),
                   "condition": LABEL, "eps": EPS, "decorrelate": DECORR,
                   "seeds_new": SEEDS_NEW, "seeds_frozen": FROZEN_SEEDS},
        "conditions": {},
    }


def rows(state, label):
    return state["conditions"].get(label, {}).get("per_seed", [])


def has_run(label, seed):
    return any(r["seed"] == seed for r in rows(load_or_init(), label))


def save_one(label, seed, acc, asr, wall_s):
    s = load_or_init()
    c = s["conditions"].setdefault(label, {"eps": EPS, "decorrelate": DECORR, "per_seed": []})
    ps = [r for r in c["per_seed"] if r["seed"] != seed]
    ps.append({"seed": seed, "accuracy": float(acc), "asr": float(asr), "wall_time_s": float(wall_s)})
    c["per_seed"] = sorted(ps, key=lambda r: r["seed"])
    a = [r["asr"] for r in c["per_seed"]]
    c["mean_asr"] = float(np.mean(a))
    c["std_asr"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    c["max_asr"] = float(np.max(a))
    c["mean_acc"] = float(np.mean([r["accuracy"] for r in c["per_seed"]]))
    c["n"] = len(c["per_seed"])
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)


# --------------------------------------------------------------------------------------------------
# The two pre-registered pre-run checks, both BY VALUE against stored per-seed rows
# --------------------------------------------------------------------------------------------------
def _frozen_rows(label):
    with open(FROZEN_ARM) as f:
        d = json.load(f)
    return {r["seed"]: r for r in d["conditions"][label]["per_seed"]}


def _flagship_rows():
    with open(FLAGSHIP) as f:
        d = json.load(f)
    return {r["seed"]: r for r in d["base_composition"]["committed_pixel"]["per_seed"]}


def verify_denominator(verbose=True):
    """Prove the reused denominator is the same computation, per seed, before pooling anything.

    An md5 of results/fg_rfa_flagship/summary.json would prove nothing about this: two arms in this
    repository were once bit-identical because a manipulation hook was never called and failed
    silently. So the check is on values, on the seeds where both runners have rows.
    """
    ca = _frozen_rows("committed_pixel")
    fg = _flagship_rows()
    shared = sorted(set(ca) & set(fg))
    if not shared:
        sys.exit("REFUSING: the two runners share no committed_pixel seed, so the reuse of the "
                 "flagship denominator cannot be proved. Run the denominator here instead.")
    d_asr = max(abs(ca[s]["asr"] - fg[s]["asr"]) for s in shared)
    d_acc = max(abs(ca[s]["accuracy"] - fg[s]["accuracy"]) for s in shared)
    if verbose:
        print(f"  denominator reuse check on seeds {shared[0]}-{shared[-1]} (n={len(shared)}): "
              f"worst |dASR| {d_asr:.9e}, worst |dacc| {d_acc:.9e}")
    if d_asr > TOL or d_acc > TOL:
        sys.exit(f"REFUSING: results/fg_rfa_flagship/ and results/criterion_aware_adversary/ disagree "
                 f"on committed_pixel (worst |dASR| {d_asr:.3e}, |dacc| {d_acc:.3e} > {TOL:.0e}). The "
                 "denominator is NOT reusable and this arm must run its own base leg.")
    return {"shared_seeds": shared, "worst_abs_dev_asr": d_asr, "worst_abs_dev_acc": d_acc,
            "conclusion": "the two runners compute the same committed_pixel cell; the flagship "
                          "denominator at seeds 42-71 is reusable"}


def harness_check():
    """Re-run ONE PUBLISHED seed through the imported run_one and compare against its stored row.

    A published seed is used deliberately. At a new seed there is nothing to compare against and the
    check would be vacuous. One run settles it for all 25 new seeds, because the code path does not
    depend on the seed. If this fails, the top-up does not run.
    """
    stored = _frozen_rows(LABEL).get(HARNESS_SEED)
    if stored is None:
        sys.exit(f"REFUSING: no stored {LABEL} row at seed {HARNESS_SEED} to check the harness against.")
    print(f"  harness check: re-running seed {HARNESS_SEED} under {LABEL} through the IMPORTED run_one")
    print(f"    stored: acc={stored['accuracy']:.6f} ASR={stored['asr']:.6f}")
    t = time.time()
    acc, asr = run_one(HARNESS_SEED, LABEL, EPS, DECORR)
    print(f"    fresh : acc={acc:.6f} ASR={asr:.6f}   ({(time.time()-t)/60:.1f} min)")
    d_acc, d_asr = abs(acc - stored["accuracy"]), abs(asr - stored["asr"])
    print(f"    |dacc| {d_acc:.9e}   |dASR| {d_asr:.9e}   tol {TOL:.0e}")
    if d_acc > TOL or d_asr > TOL:
        sys.exit("HARNESS CHECK FAILED. The imported run_one no longer reproduces the frozen arm, so "
                 "seeds 42-46 and 47-71 would not be one condition. The top-up does not run.")
    print("  HARNESS CHECK PASSED: the imported run_one reproduces the frozen row bit-for-bit.")
    return {"seed": HARNESS_SEED, "label": LABEL, "abs_dev_acc": d_acc, "abs_dev_asr": d_asr,
            "tol": TOL, "passed": True}


# --------------------------------------------------------------------------------------------------
# Analysis. Every number is recomputed from per-seed rows; nothing is transcribed.
# --------------------------------------------------------------------------------------------------
def interval(d):
    n = len(d)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    hw = float(t_crit(n) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "hw95": hw, "lo": m - hw, "hi": m + hw}


def analyze(save=True):
    den_check = verify_denominator()

    num = dict(_frozen_rows(LABEL))                       # frozen seeds 42-46
    for r in rows(load_or_init(), LABEL):                 # plus whatever this arm has reached
        num[r["seed"]] = r
    den = _flagship_rows()                                # reused denominator, seeds 42-71

    shared = sorted(set(num) & set(den))
    n = len(shared)
    if n == 0:
        sys.exit("no shared seeds between the adaptive leg and the denominator.")

    # BOTH LEGS AT THE SAME SEEDS. A Delta whose two legs come from different seed sets is not a
    # within-arm contrast, so the seed set is intersected first and every figure below is computed on
    # it. The n reached is reported as the n reached; there is no resume-until-a-threshold.
    a = np.array([num[s]["asr"] for s in shared])
    b = np.array([den[s]["asr"] for s in shared])
    paired = interval(a - b)

    sp = np.sqrt(((n - 1) * np.var(a, ddof=1) + (n - 1) * np.var(b, ddof=1)) / (2 * n - 2))
    hw_un = float(t_crit(2 * n - 1) * sp * np.sqrt(2.0 / n))     # df = 2n-2 == (2n-1)-1
    unpaired = {"n_per_leg": n, "mean_diff": float(a.mean() - b.mean()), "pooled_sd": float(sp),
                "hw95": hw_un, "lo": float(a.mean() - b.mean()) - hw_un,
                "hi": float(a.mean() - b.mean()) + hw_un}

    five = sorted(set(FROZEN_SEEDS) & set(num) & set(den))
    a5 = np.array([num[s]["asr"] for s in five])
    b5 = np.array([den[s]["asr"] for s in five])

    acc = np.array([num[s]["accuracy"] for s in shared])
    ACC_FLOOR = 0.35
    below = [int(s) for s in shared if num[s]["accuracy"] < ACC_FLOOR]

    res = {
        "prereg_commit": PREREG_COMMIT,
        "denominator_reuse_check": den_check,
        "seeds_scored": shared,
        "n": n,
        "numerator": {"label": LABEL, "mean_asr": float(a.mean()),
                      "sd_asr": float(np.std(a, ddof=1)) if n > 1 else None,
                      "median_asr": float(np.median(a)), "max_asr": float(a.max()),
                      "mean_acc": float(acc.mean()), "min_acc": float(acc.min())},
        "denominator": {"label": "committed_pixel (results/fg_rfa_flagship, REUSED)",
                        "mean_asr": float(b.mean()),
                        "sd_asr": float(np.std(b, ddof=1)) if n > 1 else None,
                        "median_asr": float(np.median(b)), "max_asr": float(b.max())},
        "primary_paired_difference": paired,
        "secondary_unpaired_difference": unpaired,
        "ratio_descriptive": {
            "at_n": n, "ratio": float(a.mean() / b.mean()) if b.mean() else None,
            "published_n5_ratio": float(a5.mean() / b5.mean()) if len(five) and b5.mean() else None,
            "published_n5_seeds": five,
            "note": ("A ratio of means gets no confidence statement at these sample sizes. It is "
                     "reported as a point figure with its n attached, per the freeze."),
        },
        "accuracy_gate": {"floor": ACC_FLOOR, "mean_acc": float(acc.mean()),
                          "seeds_below_floor": below,
                          "ok": bool(acc.mean() >= ACC_FLOOR)},
        "verdict": ("ADAPTIVE GAIN CONFIRMED at n=%d" % n) if paired["lo"] > 0 else
                   ("ADAPTIVE GAIN REFUTED at n=%d: the paired 95%% interval contains zero" % n),
        "no_reranking_claim": ("Only %s is measured at n>5. The other five adaptive conditions stay "
                              "at n=5 and no statement of the form 'X is the strongest adaptive "
                              "attack' is licensed at this n." % LABEL),
    }

    print("\n" + "=" * 78)
    print("  ADAPTIVE FoolsGold->RFA ARM, BOTH LEGS AT THE SHARED SEEDS")
    print("=" * 78)
    print(f"  seeds scored: {shared[0]}-{shared[-1]} (n={n})"
          f"{'  [PARTIAL -- the freeze targets n=30]' if n < 30 else ''}")
    print(f"  numerator   {LABEL:<20s} mean {a.mean():.4f}  median {np.median(a):.4f}  "
          f"max {a.max():.4f}")
    print(f"  denominator committed_pixel      mean {b.mean():.4f}  median {np.median(b):.4f}  "
          f"max {b.max():.4f}   (REUSED)")
    print(f"  PRIMARY paired difference: mean {paired['mean']:+.4f} sd {paired['sd']:.4f}  "
          f"95% CI [{paired['lo']:+.4f}, {paired['hi']:+.4f}]")
    print(f"  secondary unpaired       : mean {unpaired['mean_diff']:+.4f}  "
          f"95% CI [{unpaired['lo']:+.4f}, {unpaired['hi']:+.4f}]")
    if len(five) and b5.mean():
        print(f"  ratio at n={n}: {a.mean()/b.mean():.3f}x     "
              f"published at n={len(five)}: {a5.mean()/b5.mean():.3f}x  "
              f"({a5.mean():.4f} / {b5.mean():.4f})")
        print("  BOTH are reported. The two differ because seeds 42-46 sit in the left tail of the "
              "denominator's\n  right-skewed distribution (main.tex:2189), not because of "
              "adaptive-attack variance.")
    print(f"  accuracy gate: mean {acc.mean():.4f} vs floor {ACC_FLOOR} -> "
          f"{'OK' if acc.mean() >= ACC_FLOOR else 'UNINTERPRETABLE FOR ASR'}"
          f"{'   seeds below floor: %s' % below if below else ''}")
    print(f"  VERDICT: {res['verdict']}")
    print("  " + res["no_reranking_claim"])

    if save:
        s = load_or_init()
        s["analysis"] = res
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\n  written: {out_path}")
    return res


# --------------------------------------------------------------------------------------------------
def main():
    argv = sys.argv[1:]
    print("=" * 78)
    print("  SEED TOP-UP OF THE ADAPTIVE FoolsGold->RFA ARM  (Round 69)")
    print("=" * 78)
    print(f"  condition {LABEL} (eps={EPS}, decorrelate={DECORR}), new seeds "
          f"{SEEDS_NEW[0]}-{SEEDS_NEW[-1]} ({len(SEEDS_NEW)} runs)")
    print(f"  config N={FL_CONFIG.num_clients} K={FL_CONFIG.clients_per_round} f={ADV_FRACTION} "
          f"rounds={FL_CONFIG.num_rounds} defense={DEFENSE[0]}->{DEFENSE[1]}")
    print("  ONE condition only. The other five stay at n=5 and no re-ranking claim is licensed.")
    check_frozen()

    if "--verify-denominator" in argv:
        verify_denominator(); return
    if "--harness-check" in argv:
        harness_check(); return
    if "--analyze" in argv:
        analyze(); return

    verify_denominator()

    todo = [s for s in SEEDS_NEW if not has_run(LABEL, s)]
    print(f"  {len(SEEDS_NEW) - len(todo)} of {len(SEEDS_NEW)} already in the artifact; "
          f"{len(todo)} to run")
    t0, done = time.time(), 0
    for seed in todo:
        tr = time.time()
        acc, asr = run_one(seed, LABEL, EPS, DECORR)
        save_one(LABEL, seed, acc, asr, time.time() - tr)
        done += 1
        # Progress is the count of per_seed rows in the artifact, NOT a [i/N] todo position: that
        # position counts resumed-and-skipped runs and can move backwards across restarts.
        have = len(rows(load_or_init(), LABEL))
        eta = (time.time() - t0) / done * (len(todo) - done) / 3600
        print(f"  seed {seed}: acc={acc:.4f} ASR={asr:.4f}  "
              f"({(time.time()-tr)/60:.1f} min; artifact holds {have}/{len(SEEDS_NEW)} new seeds; "
              f"ETA {eta:.1f} h)", flush=True)

    analyze()


if __name__ == "__main__":
    main()
