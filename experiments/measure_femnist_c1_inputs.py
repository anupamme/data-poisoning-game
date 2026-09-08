"""Standalone single-defense baselines on FEMNIST/simple_cnn -- the INPUTS to C1.

WHY THIS IS NOT A PRE-REGISTRATION VIOLATION. C1 is defined on standalone suppression:
for every committed attack a, min over {d1, d2} of ASR(d, a) < 0.5. Those standalone
numbers are INPUTS to the criterion, not predictions of it, and the established
convention (experiments/pre_registration_prospective.md:33) is that they are measured
BEFORE any per-pair prediction is frozen. Nothing here evaluates a composition.

WHY IT IS RUN AT ALL. Wave B (the FEMNIST breadth arm) is sized against a measured
42.4 min/run on this box, so the full 8-pair design costs ~55 h. Before spending that,
this script settles the one quantity that decides whether the wave can have a positive
arm: whether ANY FEMNIST defense suppresses the pixel backdoor standalone. On CIFAR the
certified pairs clear C1 because the two constituents cover DIFFERENT attacks --
foolsgold kills scaling (0.200) while coord_median kills pixel (0.443). On FEMNIST the
pixel side looks absent: foolsgold measured 0.9959 under this exact protocol, and
results/femnist/payoff_results.json puts norm_clip at 0.996, rfa at 0.998, fedavg at
0.997 and coord_median at 0.609 -- but that artifact's protocol comparability is
unverified, so coord_median, the only candidate anywhere near the threshold, is
re-measured here under the identical runner. If it lands at or above 0.5, no pair can
satisfy C1 on FEMNIST and the honest breadth result is a scope limit, not a wave.

`fedavg` as d1 emits no per-client transform (run_all_compositions.py:360-362), so
fedavg->X IS X alone under the identical protocol. The runner is
run_cross_distribution_compositions.run_one, already parameterized by dataset/model and
already used for the CIFAR-100 and ResNet18 arms -- reused rather than forked, and its
protocol constants (N=10, K=5, 50 rounds, f=0.2, Dirichlet 0.5) are verified equal to
run_wave3_emergent.py's and run_all_compositions.py's.

Resumable: every measured cell is cached in the output artifact and never recomputed.

Output: results/femnist_c1_inputs/summary.json
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

# What to measure, in decisiveness order. coord_median under the pixel backdoor is the
# cell that decides whether C1 is satisfiable on FEMNIST at all, so it comes first.
TARGETS = [
    ("coord_median", "committed_pixel"),
]

OUT_DIR = os.path.join(base_dir, "results", "femnist_c1_inputs")
OUT = os.path.join(OUT_DIR, "summary.json")


def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {"description": "Standalone single-defense baselines on FEMNIST/simple_cnn, "
                           "measured as fedavg->X under the identical composition "
                           "protocol. These are C1's inputs, not predictions.",
            "dataset": DATASET, "model": MODEL,
            "protocol": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                         "rounds": FL_CONFIG.num_rounds, "f": ADV_FRACTION,
                         "dirichlet_alpha": 0.5, "seeds": SEEDS},
            "threshold": THRESH,
            "cells": {}}


def save(d):
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(d, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    data = load()
    cells = data["cells"]

    print("=== FEMNIST STANDALONE BASELINES (C1 inputs) ===")
    print(f"  {DATASET}/{MODEL}  N={FL_CONFIG.num_clients} K={FL_CONFIG.clients_per_round} "
          f"rounds={FL_CONFIG.num_rounds} f={ADV_FRACTION} seeds={SEEDS}")
    print(f"  measuring: {TARGETS}")
    print()

    t_start = time.time()
    n_new = n_cached = 0
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
            print(f"  -> mean ASR {cell['mean_asr']:.4f} at accuracy "
                  f"{cell['mean_accuracy']:.4f}: "
                  f"{'SUPPRESSES' if cell['suppresses'] else 'DOES NOT SUPPRESS'} "
                  f"(threshold {THRESH})")
            save(data)
        print()

    # ---- what this means for C1 on FEMNIST -------------------------------------------
    # Read the already-available standalone ASRs rather than transcribing them, so the
    # verdict below is derived from artifacts.
    payoff_path = os.path.join(base_dir, "results", "femnist", "payoff_results.json")
    ATTACK_INTERNAL = {"committed_pixel": "backdoor_pixel", "committed_scaling": "model_scaling"}
    borrowed = {}
    if os.path.exists(payoff_path):
        payoff = json.load(open(payoff_path))
        for atk, internal in ATTACK_INTERNAL.items():
            for k, v in payoff.items():
                if isinstance(v, dict) and v.get("attack") == internal and "defense" in v:
                    borrowed[f"{v['defense']}|{atk}"] = {
                        "asr": v["attack_success_rate"], "accuracy": v["accuracy"],
                        "source": "results/femnist/payoff_results.json (protocol comparability "
                                  "unverified; indicative only)"}
    data["borrowed_for_context"] = borrowed

    print("=== DOES ANY FEMNIST DEFENSE SUPPRESS THE PIXEL BACKDOOR? ===")
    print("  (C1 needs min over the pair's constituents < 0.5 for EVERY committed attack)")
    rows = []
    for key, cell in sorted(cells.items()):
        if cell["attack"] == "committed_pixel" and "mean_asr" in cell:
            rows.append((cell["defense"], cell["mean_asr"], "measured here, identical protocol"))
    for key, v in sorted(borrowed.items()):
        d, atk = key.split("|")
        if atk == "committed_pixel" and f"{d}|{atk}" not in cells:
            rows.append((d, v["asr"], "payoff matrix, indicative"))
    for d, asr, src in sorted(rows, key=lambda r: r[1]):
        mark = "  <-- SUPPRESSES" if asr < THRESH else ""
        print(f"    {d:14s} ASR={asr:.4f}  [{src}]{mark}")
    any_supp = any(asr < THRESH for _, asr, _ in rows)
    data["pixel_suppressor_exists"] = any_supp
    print()
    if any_supp:
        print("  A pixel suppressor EXISTS, so C1 is satisfiable on FEMNIST and the wave")
        print("  has a positive arm. Next: measure the remaining standalone baselines,")
        print("  then freeze per-pair predictions.")
    else:
        print("  NO defense measured on FEMNIST suppresses the pixel backdoor, so for every")
        print("  candidate pair min over its constituents is >= 0.5 on that attack and C1")
        print("  FAILS. Strategy B certifies nothing on FEMNIST, every pair is predicted")
        print("  HIGH, and the strategy-ordering comparison the wave was designed to make")
        print("  cannot be made on this dataset under this protocol. That is a SCOPE LIMIT")
        print("  on the criterion and it is reportable as one; it is not a replication")
        print("  failure, because the criterion's own precondition is unmet.")
    save(data)

    print(f"\n  {n_new} new runs, {n_cached} cached, {(time.time() - t_start)/3600:.2f} h")
    print(f"Saved to {os.path.relpath(OUT, base_dir)}")
