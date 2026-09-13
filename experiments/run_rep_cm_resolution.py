"""
Seed resolution for the one certified pair whose interval is not resolved below threshold.

The criterion certifies two pairs in distribution: fg->cm (0.107) and rep->cm (0.423). The paper
already discloses (main.tex:806) that rep->cm's 95% interval [0.231, 0.616] crosses the 0.5 threshold
and that one of its five seeds reaches 0.665, so "precision 2/2" rests on one clean case and one whose
uncertainty admits the HIGH side. That is a power problem in a single cell.

This adds seeds 47--71 to reputation->coord_median under committed_pixel only, bringing that cell to
n=30 -- the same seed count as results/fg_rfa_flagship, so the pair the criterion certifies is measured
at the same power as the pair it rejects. The committed_scaling arm sits at 0.0198 over the same five
seeds and cannot set the max-committed label, so it is not extended.

Pre-registered in experiments/pre_registration_rep_cm_resolution.md, which fixes the seed list, the
unchanged 0.5 label rule, and the three outcomes (RESOLVED LOW / STILL UNRESOLVED / LABEL FLIPS)
before any run. The seed count is NOT extended afterwards under any outcome: that would be optional
stopping.

Existing seeds are not re-run for the record and existing values are never modified. Seed 42 is
re-run once as an identity guard and its value is compared, not stored.

Config identical to results/all_compositions: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
Output: results/rep_cm_resolution/summary.json
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

from experiments.run_all_compositions import run_one                          # noqa: E402

# --- Frozen by the pre-registration; changing any of these invalidates it --------------------
PREREG = "experiments/pre_registration_rep_cm_resolution.md"
PREREG_COMMIT = "de77a7f"          # the prereg's commit; the runner aborts if it moved
D1, D2 = "reputation", "coord_median"
ATTACK = "committed_pixel"
NEW_SEEDS = list(range(47, 72))    # 25 seeds -> n=30 with the frozen 42--46
FROZEN_SEEDS = [42, 43, 44, 45, 46]
THRESH = 0.5                       # analyze_condition_ablation.py:50, unchanged
ACC_FLOOR = 0.35

# Guard 1: the frozen cell must reproduce exactly, or the environment moved and the top-up is
# not comparable to the five seeds it extends.
GUARD_SEED = 42
GUARD_ASR = 0.43177777777777776

# Frozen artifacts this run must not touch.
FROZEN_FILES = {
    "results/all_compositions/summary.json": None,
    "results/headline_seed_topup/summary.json": None,
}

out_dir = os.path.join(base, "results", "rep_cm_resolution")
out_path = os.path.join(out_dir, "summary.json")


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def check_prereg():
    """The pre-registration must exist, be committed, and be named by its own hash here."""
    path = os.path.join(base, PREREG)
    if not os.path.exists(path):
        sys.exit(f"ABORT: {PREREG} does not exist. Pre-register before running.")
    if PREREG_COMMIT == "PENDING":
        sys.exit("ABORT: PREREG_COMMIT is PENDING. Commit the pre-registration and set the hash.")
    try:
        tracked = subprocess.run(["git", "-C", base, "log", "-1", "--format=%H", "--", PREREG],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        sys.exit(f"ABORT: cannot read git history for {PREREG}: {exc}")
    if not tracked:
        sys.exit(f"ABORT: {PREREG} is not committed. Commit it before running.")
    if not tracked.startswith(PREREG_COMMIT):
        sys.exit(f"ABORT: PREREG_COMMIT={PREREG_COMMIT} but {PREREG} was last committed at {tracked}.")
    dirty = subprocess.run(["git", "-C", base, "status", "--porcelain", "--", PREREG],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"ABORT: {PREREG} has uncommitted changes. The frozen rules must be the committed ones.")
    print(f"  [OK] pre-registration {PREREG} committed at {tracked[:7]}, clean")


def check_runner_shared():
    """Guard 3: run_one is the suite's function, never a copy.

    Keyed on the resolved function's own module, not on a source grep: a grep for the definition
    keyword matches this file's own guard text and fails every clean run.
    """
    mod = getattr(run_one, "__module__", None)
    if mod != "experiments.run_all_compositions":
        sys.exit(f"ABORT: run_one resolves to module {mod!r}, not experiments.run_all_compositions. "
                 "It must be the suite's function, not a fork.")
    print(f"  [OK] run_one is {mod}.run_one, not forked")


def check_guard_cell():
    """Guard 1: re-run the frozen seed and require the frozen value bit-for-bit."""
    print(f"  running identity guard: seed {GUARD_SEED} must return ASR {GUARD_ASR!r}", flush=True)
    t = time.time()
    acc, asr = run_one(GUARD_SEED, D1, D2, ATTACK)
    print(f"  guard returned acc={acc:.4f} ASR={asr!r} ({time.time() - t:.0f}s)", flush=True)
    if float(asr) != GUARD_ASR:
        sys.exit(f"ABORT: identity guard failed. Expected {GUARD_ASR!r}, got {float(asr)!r}. "
                 "The code path or environment moved; the new seeds would not be comparable "
                 "to the frozen five.")
    print("  [OK] frozen cell reproduces exactly", flush=True)


def frozen_seed_values():
    """The five frozen ASRs, read from the two existing artifacts and never rewritten."""
    dev = json.load(open(os.path.join(base, "results/all_compositions/summary.json")))
    top = json.load(open(os.path.join(base, "results/headline_seed_topup/summary.json")))
    rows = {}
    for r in dev["pairs"][f"{D1}_then_{D2}"][ATTACK]["per_seed"]:
        rows[r["seed"]] = (r["asr"], r["accuracy"])
    for r in top["cells"][f"{D1}_then_{D2}|{ATTACK}"]["per_seed"]:
        rows.setdefault(r["seed"], (r["asr"], r["accuracy"]))
    missing = [s for s in FROZEN_SEEDS if s not in rows]
    if missing:
        sys.exit(f"ABORT: frozen seeds {missing} absent from the artifacts.")
    return rows


def main():
    print(f"=== Rep->CM committed-pixel seed resolution: n=5 -> n={len(FROZEN_SEEDS) + len(NEW_SEEDS)} "
          f"({len(NEW_SEEDS)} new runs) ===")
    print(f"    seeds {NEW_SEEDS[0]}--{NEW_SEEDS[-1]}, pre-registered at {PREREG_COMMIT[:7]}")
    print("    resolves the one certified pair whose 95% interval crosses the 0.5 threshold\n")

    print("Pre-flight:")
    check_prereg()
    check_runner_shared()
    for rel in FROZEN_FILES:
        FROZEN_FILES[rel] = md5(os.path.join(base, rel))
        print(f"  [md5] {rel} {FROZEN_FILES[rel]}")
    frozen = frozen_seed_values()
    fa = np.array([frozen[s][0] for s in FROZEN_SEEDS])
    print(f"  frozen n=5: mean {fa.mean():.6f}  sd {fa.std(ddof=1):.6f}  "
          f"per-seed {' '.join(f's{s}={frozen[s][0]:.4f}' for s in FROZEN_SEEDS)}")

    os.makedirs(out_dir, exist_ok=True)
    per_seed = {}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            per_seed = {r["seed"]: r for r in prev.get("per_seed", [])}
            print(f"  resuming: {len(per_seed)} of {len(NEW_SEEDS)} new seeds already on disk")
        except Exception:
            per_seed = {}

    if len(per_seed) < len(NEW_SEEDS):
        check_guard_cell()
    else:
        print("  all new seeds present; skipping the identity guard (nothing left to run)")
    print()

    t0 = time.time()
    for i, seed in enumerate(NEW_SEEDS, 1):
        if seed in per_seed:
            print(f"  [{i}/{len(NEW_SEEDS)}] s{seed}: cached", flush=True)
            continue
        t = time.time()
        acc, asr = run_one(seed, D1, D2, ATTACK)
        per_seed[seed] = {"seed": seed, "accuracy": float(acc), "asr": float(asr)}
        print(f"  [{i}/{len(NEW_SEEDS)}] s{seed}: acc={acc:.3f} ASR={asr:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
        json.dump({
            "description": (f"Seeds {NEW_SEEDS[0]}--{NEW_SEEDS[-1]} for {D1}->{D2} under {ATTACK}. "
                            "Merge with results/all_compositions (42-44) and "
                            "results/headline_seed_topup (45/46) for n=30. "
                            "Pre-registered in " + PREREG),
            "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
            "pair": f"{D1}_then_{D2}", "attack": ATTACK,
            "new_seeds": NEW_SEEDS, "frozen_seeds": FROZEN_SEEDS,
            "threshold": THRESH, "acc_floor": ACC_FLOOR,
            "frozen_md5": FROZEN_FILES,
            "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                       "model": "cifar_cnn", "dataset": "cifar10"},
            "per_seed": [per_seed[s] for s in sorted(per_seed)],
        }, open(out_path, "w"), indent=2)

    # ---- verdict, by the pre-registered rule and no other -----------------------------------
    allv = {s: frozen[s][0] for s in FROZEN_SEEDS}
    alla = {s: frozen[s][1] for s in FROZEN_SEEDS}
    for s, r in per_seed.items():
        allv[s], alla[s] = r["asr"], r["accuracy"]
    a = np.array([allv[s] for s in sorted(allv)], dtype=float)
    acc = np.array([alla[s] for s in sorted(alla)], dtype=float)
    n = len(a)
    try:
        from scipy import stats as sps
        tc = float(sps.t.ppf(0.975, n - 1))
    except Exception:
        tc = 1.96
    half = tc * a.std(ddof=1) / np.sqrt(n)
    lo, hi = max(a.mean() - half, 0.0), min(a.mean() + half, 1.0)

    print(f"\n--- n={n} ---")
    print(f"  mean {a.mean():.6f}  sd {a.std(ddof=1):.6f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  frozen n=5 beside it: mean {fa.mean():.6f}  (never replaced)")
    print(f"  mean clean accuracy {acc.mean():.4f} (floor {ACC_FLOOR})")
    if acc.mean() < ACC_FLOOR:
        print("  VERDICT: UNINTERPRETABLE, accuracy below floor. No verdict stands.")
    elif a.mean() >= THRESH:
        print(f"  VERDICT: THE LABEL FLIPS. mean {a.mean():.4f} >= {THRESH}. The certified set becomes "
              "one pair and the criterion's precision is reported as 1/2.")
    elif hi < THRESH:
        print(f"  VERDICT: RESOLVED LOW. Upper bound {hi:.4f} < {THRESH}. Both certified pairs are "
              "resolved below threshold.")
    else:
        print(f"  VERDICT: STILL UNRESOLVED at n={n}. Upper bound {hi:.4f} >= {THRESH}. Reported as "
              "unresolved; no claim is upgraded and the seed count is NOT extended.")

    print("\nFrozen artifacts, after:")
    ok = True
    for rel, before in FROZEN_FILES.items():
        after = md5(os.path.join(base, rel))
        state = "unchanged" if after == before else "CHANGED -- INVALID"
        ok &= after == before
        print(f"  [md5] {rel} {after} {state}")
    print(f"\nWall time: {(time.time() - t0) / 3600:.2f} h\nSaved to {out_path}")
    print("Run experiments/analyze_headline_cis.py to merge and report the interval.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
