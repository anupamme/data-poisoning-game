"""Re-run the FoolsGold->RFA consensus-shift arm with the attack actually applied.

Frozen by experiments/pre_registration_consensus_shift_fix.md @ 784a201.

WHY THIS EXISTS. run_fg_rfa_flagship.py:343 builds the attack as get_attack("backdoor_pixel").
BackdoorPixelAttack does not override manipulate_update, so it inherits the base class's identity
at attack_strategies.py:61-63. The runner's Phase-1 shift at :374 therefore evaluates to

    update[k] + shift_rate * (update[k] - update[k])  ==  update[k]

for EVERY shift_rate, and the Phase-2 injection at :378 is likewise the identity. The published
arm is bit-identically the base committed-pixel composition: results/fg_rfa_flagship/summary.json's
adaptive_consensus_shift/rate_0.01 and rate_0.05 agree with base_composition/committed_pixel on
all five seeds, in ASR and in clean accuracy, to twelve decimal places. tab:fg_rfa_flagship's
"0.045 +/- 0.012" is the base pixel arm re-labelled as an attack result.

ONE FACTOR MOVES against that void arm: the attack object. The composition, the config, the seeds
and the rates are imported or copied verbatim from the arm being replaced, so nothing else can drift.

TWO SEMANTICS ARE FROZEN RATHER THAN CORRECTED, so this arm stays comparable to the published
rep+tm arm at main.tex:2499 that it is meant to sit beside:
  (1) one ConsensusShiftAttack instance is shared across adversaries per run, exactly as
      run_consensus_shift_vs_composition.py:114 does, so self._round advances per adversary-update
      EVENT and shift_rounds=25 is 25 such events rather than 25 rounds;
  (2) scale_factor=1.0, so _phase2_inject returns the update unchanged
      (consensus_shift_attack.py:73-77) and Phase 2 is ordinary committed-pixel behaviour --
      which main.tex:2499 already discloses. The attack's whole distinguishing mechanism is
      Phase-1 drift, so a small effect is expected and was predicted in the freeze.

HARNESS CHECK. --harness-check runs seed 42 through THIS file's loop with the attack object
overridden to get_attack("backdoor_pixel") -- the void arm's configuration -- and asserts
bit-equality against the stored base_composition/committed_pixel seed-42 row. Reproducing the BASE
row is deliberate: the stored consensus-shift rows are void, so agreeing with them would prove
nothing. This isolates the attack object as the single moving factor.

Writes results/fg_rfa_consensus_shift_fixed/ and nothing else. Every existing results/ directory,
including results/fg_rfa_flagship/, is read-only here; the buggy run_consensus_shift is left in
place as the record of the defect.

Usage (from code/data_poisoning_game):
    PYTHONPATH=. python3 -m experiments.run_fg_rfa_consensus_shift_fixed --harness-check
    PYTHONPATH=. python3 -m experiments.run_fg_rfa_consensus_shift_fixed
"""
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

from fl_core import (get_federated_dataset, get_model, FederatedServer,   # noqa: E402
                     FederatedClient)
from attacks import get_attack                                            # noqa: E402
from attacks.consensus_shift_attack import ConsensusShiftAttack           # noqa: E402
from experiments.run_payoff_matrix import evaluate_backdoor               # noqa: E402
from experiments.run_all_compositions import generic_compose             # noqa: E402
from experiments.analyze_headline_cis import t_crit  # noqa: E402  NOT a literal T95 table

# The config and the adversary fraction come from the arm being replaced, not from new literals,
# so this arm cannot silently differ from the void one in anything but the attack object.
from experiments.run_fg_rfa_flagship import FL_CONFIG, ADV_FRACTION       # noqa: E402

PREREG = "experiments/pre_registration_consensus_shift_fix.md"
PREREG_COMMIT = "784a201"

SHIFT_RATES = [0.01, 0.05]          # exactly what tab:fg_rfa_flagship's artifact covers
SEEDS = [42, 43, 44, 45, 46]        # frozen; not extended, not filtered
SHIFT_ROUNDS = 25
SCALE_FACTOR = 1.0
DATASET = "cifar10"
MODEL = "cifar_cnn"
DIRICHLET = 0.5
D1, D2, TAU = "foolsgold", "rfa", 5.0
ACC_FLOOR = 0.35
TOL = 1e-9

PARENT = os.path.join(base, "results", "fg_rfa_flagship", "summary.json")
out_dir = os.path.join(base, "results", "fg_rfa_consensus_shift_fixed")
out_path = os.path.join(out_dir, "summary.json")


def banner():
    print("=" * 94)
    print("  FG->RFA CONSENSUS-SHIFT ARM, ATTACK ACTUALLY APPLIED  (Round 69, ninth review)")
    print("=" * 94)
    print(f"    {DATASET}/{MODEL}, {D1}->{D2} (tau={TAU}), N={FL_CONFIG.num_clients}, "
          f"K={FL_CONFIG.clients_per_round}, f={ADV_FRACTION}, {FL_CONFIG.num_rounds} rounds")
    print(f"    rates {SHIFT_RATES}, seeds {SEEDS} -> {len(SHIFT_RATES) * len(SEEDS)} runs")
    print("    ONE factor moves against the void arm: the attack object. Config, composition, seeds")
    print("    and rates are imported or copied verbatim from run_fg_rfa_flagship.py.")
    print(f"    shift_rounds={SHIFT_ROUNDS} counts ADVERSARY-UPDATE EVENTS, not rounds; "
          f"scale_factor={SCALE_FACTOR}")
    print("    so Phase 2 is the identity and the attack's whole mechanism is Phase-1 drift.")
    print("    THE VOID ROW IS WITHDRAWN REGARDLESS OF OUTCOME; the defect is disclosed in the paper.")


def check_frozen():
    """A hash that nobody checks is a claim, not a freeze.

    (a) the commit resolves, or the hash is a typo pointing at nothing; (b) the pre-registration
    exists AT that commit, or the freeze names a commit that does not contain the rules; (c) the
    committed blob equals the working copy byte for byte, or the run would be scored against rules
    that are not the ones on record. rev-parse and hash-object write nothing.
    """
    path = os.path.join(base, PREREG)
    if not os.path.exists(path):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the rules are not frozen.\n"
                 f"  1. git add {PREREG} && git commit\n"
                 "  2. set PREREG_COMMIT here to that hash.")
    try:
        head = subprocess.run(["git", "-C", base, "rev-parse", "--verify",
                               f"{PREREG_COMMIT}^{{commit}}"], capture_output=True, text=True)
        if head.returncode != 0:
            sys.exit(f"REFUSING TO RUN: PREREG_COMMIT {PREREG_COMMIT} does not resolve to a commit "
                     f"in {base}. The freeze names nothing.")
        committed = subprocess.run(["git", "-C", base, "rev-parse", f"{PREREG_COMMIT}:{PREREG}"],
                                   capture_output=True, text=True)
        if committed.returncode != 0:
            sys.exit(f"REFUSING TO RUN: {PREREG} does not exist at commit {PREREG_COMMIT}.")
        working = subprocess.run(["git", "-C", base, "hash-object", PREREG],
                                 capture_output=True, text=True)
        if working.returncode != 0:
            sys.exit(f"REFUSING TO RUN: cannot hash {PREREG} to compare it against the freeze.")
        if committed.stdout.strip() != working.stdout.strip():
            sys.exit(f"REFUSING TO RUN: {PREREG} has been EDITED since it was frozen at "
                     f"{PREREG_COMMIT}.\n"
                     f"  committed blob: {committed.stdout.strip()[:12]}\n"
                     f"  working blob:   {working.stdout.strip()[:12]}\n"
                     "A post-hoc edit to a decision rule is not an amendment, it is the thing "
                     "pre-registration forbids.")
    except FileNotFoundError:
        sys.exit("REFUSING TO RUN: git is not available, so the freeze cannot be verified.")
    print(f"  freeze verified: {PREREG} @ {PREREG_COMMIT}, blob equal to the working copy")


# --------------------------------------------------------------------------------------------------
# The run. Mirrors run_consensus_shift_vs_composition.run_one (:99-151) exactly, with one
# substitution: generic_compose(foolsgold, rfa) in place of compose_reputation_trimmed_mean.
# --------------------------------------------------------------------------------------------------
def run_one(seed, shift_rate, attack_override=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        DATASET, FL_CONFIG.num_clients, DIRICHLET, seed
    )
    model = get_model(MODEL, num_classes)
    server = FederatedServer(model, device)

    num_adv = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adv_ids = set(range(num_adv))

    if attack_override is not None:
        attack = attack_override          # harness check: the void arm's configuration
    else:
        attack = ConsensusShiftAttack(
            shift_rate=shift_rate,
            shift_rounds=SHIFT_ROUNDS,
            scale_factor=SCALE_FACTOR,
        )

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adv_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adv_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = generic_compose(server, updates, D1, D2, tau=TAU)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --------------------------------------------------------------------------------------------------
# The reused base leg. results/fg_rfa_flagship/ is READ ONLY here.
# --------------------------------------------------------------------------------------------------
def base_rows():
    if not os.path.exists(PARENT):
        sys.exit(f"REFUSING TO RUN: the base leg {PARENT} is missing; there is nothing to pair against.")
    with open(PARENT) as f:
        parent = json.load(f)
    try:
        rows = parent["base_composition"]["committed_pixel"]["per_seed"]
    except KeyError:
        sys.exit("REFUSING TO RUN: results/fg_rfa_flagship/summary.json has no "
                 "base_composition/committed_pixel cell.")
    by_seed = {r["seed"]: r for r in rows}
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        sys.exit(f"REFUSING TO RUN: the base leg is missing seeds {missing}; the pairing would be "
                 "mixed-n, which is the defect this repository has already shipped once.")
    return {s: by_seed[s] for s in SEEDS}


def harness_check():
    """Reproduce the BASE row through this file's loop, not the void consensus-shift rows.

    A value comparison against stored per-seed figures, never an md5 of a results file: this
    repository has shipped an arm that was bit-identical to another for the wrong reason, which is
    the very defect being repaired here.
    """
    stored = base_rows()[42]
    print(f"  harness check: seed 42 through THIS loop with the void arm's attack object")
    print(f"    stored (base_composition/committed_pixel): "
          f"acc={stored['accuracy']:.6f} ASR={stored['asr']:.12f}")
    t0 = time.time()
    acc, asr = run_one(42, SHIFT_RATES[-1], attack_override=get_attack("backdoor_pixel"))
    print(f"    fresh : acc={acc:.6f} ASR={asr:.12f}   ({(time.time() - t0) / 3600:.2f} h)")
    d_acc, d_asr = abs(acc - stored["accuracy"]), abs(asr - stored["asr"])
    print(f"    |dacc| {d_acc:.9e}   |dASR| {d_asr:.9e}   tol {TOL:g}")
    if d_acc > TOL or d_asr > TOL:
        sys.exit("HARNESS CHECK FAILED. This loop does not reproduce the base composition, so a "
                 "difference against the void arm could not be attributed to the attack object. "
                 "Nothing is written.")
    print("  HARNESS CHECK PASSED: the loop, the aggregator, the partition, the metric and the RNG")
    print("  consumption are unchanged, so the attack object is the single moving factor.")


# --------------------------------------------------------------------------------------------------
# Checkpointing. Writes results/fg_rfa_consensus_shift_fixed/ and nothing else.
# --------------------------------------------------------------------------------------------------
def load_or_init():
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {
        "description": ("The fg->rfa consensus-shift arm re-run with ConsensusShiftAttack actually "
                        "applied. Supersedes results/fg_rfa_flagship/summary.json's "
                        "adaptive_consensus_shift cells, which are VOID: that runner passed a "
                        "backdoor_pixel attack whose inherited manipulate_update is the identity, so "
                        "its Phase-1 shift was arithmetically zero at every rate and its rows are "
                        "bit-identical to base_composition/committed_pixel."),
        "prereg_commit": PREREG_COMMIT,
        "supersedes": "results/fg_rfa_flagship/summary.json :: adaptive_consensus_shift/*",
        "defense": f"{D1}_then_{D2}",
        "config": {"num_clients": FL_CONFIG.num_clients,
                   "clients_per_round": FL_CONFIG.clients_per_round,
                   "adversarial_fraction": ADV_FRACTION, "num_rounds": FL_CONFIG.num_rounds,
                   "dataset": DATASET, "model": MODEL, "dirichlet": DIRICHLET,
                   "d1": D1, "d2": D2, "tau": TAU,
                   "shift_rates": SHIFT_RATES, "seeds": SEEDS,
                   "shift_rounds": SHIFT_ROUNDS, "scale_factor": SCALE_FACTOR,
                   "shift_rounds_counts": "adversary-update events, not rounds"},
        "rates": {},
    }


def rate_key(rate):
    return f"rate_{rate}"


def rows_for(state, rate):
    return state["rates"].get(rate_key(rate), {}).get("per_seed", [])


def has_run(rate, seed):
    return any(r["seed"] == seed for r in rows_for(load_or_init(), rate))


def save_one(rate, seed, acc, asr, wall_s):
    os.makedirs(out_dir, exist_ok=True)
    s = load_or_init()
    k = rate_key(rate)
    s["rates"].setdefault(k, {"shift_rate": rate, "per_seed": []})
    keep = [r for r in s["rates"][k]["per_seed"] if r["seed"] != seed]
    keep.append({"seed": seed, "accuracy": float(acc), "asr": float(asr),
                 "wall_time_s": float(wall_s)})
    s["rates"][k]["per_seed"] = sorted(keep, key=lambda r: r["seed"])
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)


# --------------------------------------------------------------------------------------------------
# Analysis. The frozen primary is the paired difference against the reused base leg, per rate.
# --------------------------------------------------------------------------------------------------
def paired(d):
    a = np.asarray(d, dtype=float)
    n = int(a.size)
    m = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    hw = float(t_crit(n) * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "hw95": hw, "lo": m - hw, "hi": m + hw}


def analyze():
    s = load_or_init()
    bl = base_rows()
    print()
    print("=== ANALYSIS (frozen primary: the paired difference against the reused base pixel leg) ===")
    print(f"    base leg REUSED from results/fg_rfa_flagship/ base_composition/committed_pixel, "
          f"seeds {SEEDS[0]}-{SEEDS[-1]}")
    b_asr = np.array([bl[x]["asr"] for x in SEEDS], dtype=float)
    print(f"    base: mean {b_asr.mean():.4f}  sd(ddof=0) {b_asr.std(ddof=0):.4f}  "
          f"acc mean {np.mean([bl[x]['accuracy'] for x in SEEDS]):.4f}")

    out = {"prereg_commit": PREREG_COMMIT,
           "base_leg": {"label": "committed_pixel (results/fg_rfa_flagship, REUSED)",
                        "seeds": SEEDS,
                        "mean_asr": float(b_asr.mean()),
                        "sd_asr_ddof0": float(b_asr.std(ddof=0))},
           "rates": {}, "verdicts": {}}

    any_effective = False
    for rate in SHIFT_RATES:
        rows = {r["seed"]: r for r in rows_for(s, rate)}
        seeds = [x for x in SEEDS if x in rows]
        if not seeds:
            print(f"\n  rate {rate}: no runs on disk yet.")
            continue
        a_asr = np.array([rows[x]["asr"] for x in seeds], dtype=float)
        a_acc = np.array([rows[x]["accuracy"] for x in seeds], dtype=float)
        d = [rows[x]["asr"] - bl[x]["asr"] for x in seeds]
        p = paired(d)
        below = [x for x in seeds if rows[x]["accuracy"] < ACC_FLOOR]

        print(f"\n  rate {rate}  (n={len(seeds)}, seeds {seeds})")
        print(f"    ASR   mean {a_asr.mean():.4f}  sd(ddof=0) {a_asr.std(ddof=0):.4f}  "
              f"median {np.median(a_asr):.4f}  max {a_asr.max():.4f}")
        print(f"    acc   mean {a_acc.mean():.4f}  min {a_acc.min():.4f}")
        print(f"    PRIMARY paired difference vs base: mean {p['mean']:+.4f} sd {p['sd']:.4f} "
              f"95% CI [{p['lo']:+.4f}, {p['hi']:+.4f}]")
        print(f"    per seed: " + "  ".join(f"{x}:{rows[x]['asr'] - bl[x]['asr']:+.4f}" for x in seeds))

        # The accuracy gate is scored before any effect is named, per the freeze.
        if a_acc.mean() < ACC_FLOOR:
            verdict = "UNINTERPRETABLE FOR ASR: mean clean accuracy below the 0.35 floor"
        elif len(seeds) < 2:
            verdict = "INDETERMINATE: fewer than two seeds, no interval"
        elif p["lo"] > 0:
            verdict = "ATTACK EFFECTIVE: the paired 95% CI excludes zero and lies above it"
            any_effective = True
        elif p["hi"] < 0:
            verdict = "ATTACK HARMFUL TO THE ADVERSARY: the paired 95% CI lies below zero"
        else:
            verdict = "ATTACK INEFFECTIVE: the paired 95% CI contains zero (the predicted outcome)"
        print(f"    accuracy gate: mean {a_acc.mean():.4f} vs floor {ACC_FLOOR} -> "
              f"{'OK' if a_acc.mean() >= ACC_FLOOR else 'FAILED'}"
              + (f"; seeds below floor {below}" if below else ""))
        print(f"    VERDICT: {verdict}")

        out["rates"][rate_key(rate)] = {
            "shift_rate": rate, "n": len(seeds), "seeds": seeds,
            "mean_asr": float(a_asr.mean()), "sd_asr_ddof0": float(a_asr.std(ddof=0)),
            "median_asr": float(np.median(a_asr)), "max_asr": float(a_asr.max()),
            "mean_acc": float(a_acc.mean()), "min_acc": float(a_acc.min()),
            "seeds_below_acc_floor": below,
            "primary_paired_difference": p,
            "per_seed_difference": {str(x): rows[x]["asr"] - bl[x]["asr"] for x in seeds},
        }
        out["verdicts"][rate_key(rate)] = verdict

    complete = all(len(rows_for(s, r)) == len(SEEDS) for r in SHIFT_RATES)
    out["complete"] = complete
    out["void_row_withdrawn"] = (
        "tab:fg_rfa_flagship's Consensus-shift row (0.045 +/- 0.012) is superseded regardless of "
        "outcome, and the defect is disclosed in the paper, not only in the response letter.")
    out["count_consequence"] = (
        "main.tex:2242's count moves to 'four of the six' or higher"
        if any_effective else
        "main.tex:2242's count stays 'three of the six'; consensus-shift remains among the arms "
        "level with the base, now for a measured reason. A 5-seed null on a right-skewed "
        "denominator is weak evidence of a null and is reported as such.")
    out["no_claim_at_n30"] = ("This arm is n=5. No statement about consensus-shift at n=30 is made, "
                              "and rates 0.1 and 0.2 remain unmeasured against this composition.")

    print()
    print(f"  {out['count_consequence']}")
    if not complete:
        done = {rate_key(r): len(rows_for(s, r)) for r in SHIFT_RATES}
        print(f"  INCOMPLETE: runs on disk per rate {done} of {len(SEEDS)} each. The figures above "
              "are at the n actually reached.")

    s["analysis"] = out
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)


if __name__ == "__main__":
    banner()
    check_frozen()

    if "--harness-check" in sys.argv:
        harness_check()
        sys.exit(0)

    if "--analyze-only" in sys.argv:
        analyze()
        sys.exit(0)

    todo = [(r, x) for r in SHIFT_RATES for x in SEEDS if not has_run(r, x)]
    print(f"\n  {len(todo)} of {len(SHIFT_RATES) * len(SEEDS)} runs remain "
          "(counted from per_seed in the artifact, not from a log position).")
    t_all = time.time()
    for i, (rate, seed) in enumerate(todo, 1):
        t0 = time.time()
        print(f"  [{i}/{len(todo)}] rate {rate}, seed {seed} ...", flush=True)
        acc, asr = run_one(seed, rate)
        w = time.time() - t0
        save_one(rate, seed, acc, asr, w)
        print(f"      acc {acc:.4f}  ASR {asr:.4f}   ({w / 60:.1f} min, "
              f"{(time.time() - t_all) / 3600:.2f} h elapsed)", flush=True)

    analyze()
