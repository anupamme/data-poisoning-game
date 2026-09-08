"""Standalone single-defense baselines on FEMNIST/simple_cnn under MODEL SCALING -- C1 inputs.

WHY THIS EXISTS. Round 49 asks whether a SEVENTH comparability cell is admissible: a
downstream defense outside {krum, cos_krum, reputation, coord_median} that satisfies the
paper's own powered condition (standalone ASR < 0.5 at accuracy >= 0.35), so that a
Mode-S contrast on it could say anything at all.

THIS SCRIPT WAS WRONG ONCE, AND THE GUARD BELOW IS WHY IT CANNOT BE WRONG THE SAME WAY.
Its first run (quarantined at results/femnist_c1_inputs_scaling_INVALID/) measured the
PIXEL arm while reporting model scaling, because run_cross_distribution_compositions.run_one
never called attack.manipulate_update -- and that call is the only thing separating the two
attacks, which share poison_dataset outright. The runner is fixed; every value here is now
gated on VERIFY_CELL below, which re-measures a cell whose true scaling value is already
frozen and aborts unless it reproduces it. krum is that cell because its two arms are far
apart (scaling 0.027, pixel 0.660), so a single seed distinguishes a working hook from a
skipped one; trimmed_mean/rfa/norm_clip would not (their arms differ by ~0.01).

TARGETS ARE ONLY THE GENUINELY UNMEASURED DEFENSES. results/femnist_c1_inputs/summary.json's
borrowed_for_context already carries FEMNIST scaling for fedavg, krum, multi_krum,
trimmed_mean, coord_median, norm_clip and rfa -- and every one except krum fails the powered
condition (trimmed_mean 0.9999, rfa 0.9990, norm_clip 0.9986, coord_median 0.6567,
multi_krum 0.3333 at 0.041 accuracy). The first run wasted 12 runs re-measuring three of
those. What is missing is reputation, foolsgold, cos_krum and cos_reputation.
reputation is FIRST because the paper's own reading is that reputation holds model scaling
and not the pixel backdoor, making it the most likely powered candidate on this axis.
fltrust is excluded by argument rather than measurement: main.tex records it as bit-identical
per seed under positive per-client rescaling, which is exactly the Mode-S dose, so a ladder
on it would observe an algebraic identity.

WHY THIS IS NOT A PRE-REGISTRATION VIOLATION. Identical reasoning to
measure_femnist_c1_inputs.py, whose runner and protocol this reuses: C1 is defined on
standalone suppression, those standalone numbers are INPUTS to the criterion rather than
predictions of it, and the convention (experiments/pre_registration_prospective.md:33) is
that they are measured BEFORE any per-pair prediction is frozen. Nothing here evaluates a
composition, and no prediction of any kind is being tested. The pre-registration for a
seventh comparability cell is written only if this screen returns a powered candidate.

`fedavg` as d1 emits no per-client transform (run_all_compositions.py:360-362), so
fedavg->X IS X alone under the identical protocol.

FROZEN ARTIFACTS ARE NOT TOUCHED. Output goes to its own directory;
results/femnist_c1_inputs/summary.json is read for context and never written.

Resumable: every measured cell is checkpointed and never recomputed.

Output: results/femnist_c1_inputs_scaling/summary.json
Run:    python3 experiments/measure_femnist_c1_inputs_scaling.py
"""
import json
import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from experiments.run_cross_distribution_compositions import (  # noqa: E402
    FL_CONFIG, ADV_FRACTION, run_one)

DATASET = "femnist"
MODEL = "simple_cnn"
SEEDS = [42, 43, 44]          # the suite's frozen seeds, unchanged
THRESH = 0.5                  # C1's suppression threshold, unchanged
ACC_FLOOR = 0.35              # the powered condition's accuracy gate, unchanged

# The cell that proves the scaling hook fires. Its frozen true value is far from its
# pixel value, so one seed is enough to tell a working runner from a skipped hook.
VERIFY_CELL = ("krum", "committed_scaling")
VERIFY_SEED = 42
VERIFY_EXPECT_SCALING = 0.0270     # frozen, results/femnist/payoff_results.json
VERIFY_PIXEL_VALUE    = 0.6601     # what a skipped manipulate_update would return
VERIFY_MAX_ASR        = 0.35       # abort above this: the hook did not fire

# ORDER IS "COULD THIS BE A SEVENTH CELL?", NOT "IS IT LIKELY TO SUPPRESS?".
# The comparability table already uses krum, cos_krum, reputation and coord_median, so a
# powered reading on one of THOSE is a second-dataset replication of a cell that exists --
# it cannot be a seventh cell. Only a defense outside that set can. An earlier ordering put
# reputation and cos_krum first because they looked likeliest to suppress; that answered the
# wrong question and cost 5 of 11 remaining runs. reputation is in fact not powered here
# (seed 42: ASR 0.9947 at 0.446 accuracy), which is why its remaining seeds are last.
TARGETS = [
    # -- could yield a SEVENTH cell: outside the table's four defenses --
    ("foolsgold",      "committed_scaling"),
    ("cos_reputation", "committed_scaling"),
    # -- replication only: already in the table, cannot be a seventh cell --
    ("reputation",     "committed_scaling"),
    ("cos_krum",       "committed_scaling"),
]
# Everything from NEW_CELL_CANDIDATES onward decides the seventh-cell question; the rest is
# optional replication. Stop after these two if compute is the binding constraint.
NEW_CELL_CANDIDATES = {"foolsgold", "cos_reputation"}

OUT_DIR = os.path.join(base_dir, "results", "femnist_c1_inputs_scaling")
OUT = os.path.join(OUT_DIR, "summary.json")


def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"description": "Standalone single-defense baselines on FEMNIST/simple_cnn under "
                           "committed model scaling, measured as fedavg->X under the identical "
                           "composition protocol as results/femnist_c1_inputs/. These are C1's "
                           "inputs, not predictions. Screen for a seventh comparability cell.",
            "dataset": DATASET, "model": MODEL,
            "protocol": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                         "rounds": FL_CONFIG.num_rounds, "f": ADV_FRACTION,
                         "dirichlet_alpha": 0.5, "seeds": SEEDS},
            "threshold": THRESH, "accuracy_floor": ACC_FLOOR,
            "cells": {}}


def save(d):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(d, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    data = load()
    cells = data["cells"]

    print("=== FEMNIST STANDALONE BASELINES UNDER MODEL SCALING (C1 inputs) ===")
    print(f"  {DATASET}/{MODEL}  N={FL_CONFIG.num_clients} K={FL_CONFIG.clients_per_round} "
          f"rounds={FL_CONFIG.num_rounds} f={ADV_FRACTION} seeds={SEEDS}")
    print(f"  powered condition: ASR < {THRESH} at accuracy >= {ACC_FLOOR}")
    print(f"  targets, in decisiveness order: {[t[0] for t in TARGETS]}")
    print()

    # ---- static guard: the runner must still apply the update-level hook ----
    runner_src = open(os.path.join(base_dir, "experiments",
                                   "run_cross_distribution_compositions.py")).read()
    if "manipulate_update(" not in runner_src:
        sys.exit("ABORT: run_cross_distribution_compositions no longer calls "
                 "manipulate_update, so committed_scaling would silently equal "
                 "committed_pixel. See results/femnist_c1_inputs_scaling_INVALID/.")
    print("  [guard] runner applies attack.manipulate_update: OK")

    # ---- functional guard: reproduce a cell whose true scaling value is frozen ----
    t_start = time.time()
    n_new = n_cached = 0
    vd, va = VERIFY_CELL
    vkey = f"{vd}|{va}"
    vcell = cells.setdefault(vkey, {"defense": vd, "attack": va, "per_seed": {},
                                    "role": "verification, not a screen target"})
    if str(VERIFY_SEED) in vcell["per_seed"]:
        vasr = vcell["per_seed"][str(VERIFY_SEED)]["asr"]
        print(f"  [verify] {vkey} seed {VERIFY_SEED}: ASR={vasr:.4f} (cached)")
    else:
        print(f"  [verify] {vkey} seed {VERIFY_SEED}: running "
              f"(expect ~{VERIFY_EXPECT_SCALING:.4f}; a skipped hook gives "
              f"~{VERIFY_PIXEL_VALUE:.4f})", flush=True)
        t0 = time.time()
        vacc, vasr = run_one(VERIFY_SEED, "fedavg", vd, va, DATASET, MODEL)
        vcell["per_seed"][str(VERIFY_SEED)] = {"accuracy": vacc, "asr": vasr,
                                               "seconds": time.time() - t0}
        n_new += 1
        save(data)
        print(f"  [verify] ASR={vasr:.4f} acc={vacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if vasr > VERIFY_MAX_ASR:
        data["verification"] = {"cell": vkey, "seed": VERIFY_SEED, "asr": vasr,
                                "passed": False}
        save(data)
        sys.exit(f"ABORT: verification cell {vkey} returned ASR={vasr:.4f} > "
                 f"{VERIFY_MAX_ASR}. Frozen model scaling is {VERIFY_EXPECT_SCALING}; "
                 f"the pixel arm is {VERIFY_PIXEL_VALUE}. The attack under test is not "
                 f"model scaling. No target was run.")
    data["verification"] = {"cell": vkey, "seed": VERIFY_SEED, "asr": vasr,
                            "expected_scaling": VERIFY_EXPECT_SCALING,
                            "pixel_value_if_hook_skipped": VERIFY_PIXEL_VALUE,
                            "passed": True}
    save(data)
    print("  [verify] scaling hook confirmed live\n", flush=True)

    for defense, attack in TARGETS:
        key = f"{defense}|{attack}"
        cell = cells.setdefault(key, {"defense": defense, "attack": attack, "per_seed": {}})
        print(f"[{defense} alone / {attack}]", flush=True)
        for seed in SEEDS:
            s = str(seed)
            if s in cell["per_seed"]:
                n_cached += 1
                r = cell["per_seed"][s]
                print(f"    seed={seed}: acc={r['accuracy']:.3f} ASR={r['asr']:.3f} (cached)",
                      flush=True)
                continue
            t0 = time.time()
            acc, asr = run_one(seed, "fedavg", defense, attack, DATASET, MODEL)
            el = time.time() - t0
            cell["per_seed"][s] = {"accuracy": acc, "asr": asr, "seconds": el}
            n_new += 1
            print(f"    seed={seed}: acc={acc:.3f} ASR={asr:.3f} ({el:.0f}s)", flush=True)
            save(data)   # checkpoint after every run

        asrs = [cell["per_seed"][str(s)]["asr"] for s in SEEDS if str(s) in cell["per_seed"]]
        accs = [cell["per_seed"][str(s)]["accuracy"] for s in SEEDS if str(s) in cell["per_seed"]]
        if len(asrs) == len(SEEDS):
            cell["mean_asr"] = sum(asrs) / len(asrs)
            cell["mean_accuracy"] = sum(accs) / len(accs)
            cell["suppresses"] = cell["mean_asr"] < THRESH
            cell["powered"] = bool(cell["suppresses"] and cell["mean_accuracy"] >= ACC_FLOOR)
            print(f"  -> mean ASR {cell['mean_asr']:.4f} at accuracy "
                  f"{cell['mean_accuracy']:.4f}: "
                  f"{'POWERED' if cell['powered'] else 'NOT POWERED'} "
                  f"(needs ASR < {THRESH} at acc >= {ACC_FLOOR})", flush=True)
            save(data)
        print(flush=True)

    print("=== IS A SEVENTH COMPARABILITY CELL ADMISSIBLE? ===")
    screened = {d for d, _ in TARGETS}
    powered = [c["defense"] for c in cells.values()
               if c.get("powered") and c["defense"] in screened]
    data["powered_candidates"] = powered
    for k, c in sorted(cells.items()):
        if "mean_asr" in c and c["defense"] in screened:
            print(f"    {c['defense']:14s} ASR={c['mean_asr']:.4f} acc={c['mean_accuracy']:.4f}"
                  f"{'   <-- POWERED' if c.get('powered') else ''}")
    print()
    if powered:
        print(f"  ADMISSIBLE: {powered}. A Mode-S contrast on one of these has suppression to")
        print("  lose, so the seventh cell can be pre-registered and run.")
    else:
        print(f"  (seventh-cell candidates screened: {sorted(NEW_CELL_CANDIDATES)};")
        print("   any powered reading on reputation/cos_krum would be replication, not a new cell)")
        print("  NOT ADMISSIBLE: no unmeasured defense satisfies C1's own precondition on")
        print("  this axis. Together with the seven FEMNIST scaling cells already frozen")
        print("  (krum the only powered one, and it is cell 6) and the CIFAR-10")
        print("  measurements, the powered set is exhausted by the cells already in the")
        print("  table, and the exhaustion is measured rather than assumed.")
    save(data)

    print(f"\n  {n_new} new runs, {n_cached} cached, {(time.time() - t_start)/3600:.2f} h")
    print(f"Saved to {os.path.relpath(OUT, base_dir)}")
