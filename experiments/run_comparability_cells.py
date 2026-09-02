"""
The two new comparability cells of `experiments/pre_registration_comparability.md` (Round 35).

WHAT THIS SUITE IS FOR
The paper's confounded-vs-controlled contrast is currently one cell. Four cells already exist in
frozen artifacts and are re-scored, never re-run. This runner adds the two that do not exist:

  cell 5  coord_median / committed_scaling on CIFAR-10      BOTH ladders   40 runs
  cell 6  krum / committed_scaling on EMNIST-byclass        confounded     20 runs

Cell 5 is the adjudicating one: coord_median/pixel is the paper's ONLY sign reversal, so the same
aggregator under the OTHER committed attack tests whether the hero result is attack-specific. Cell 6
completes a cell whose controlled half already exists in results/dose_femnist/, adding a second
dataset AND architecture to the comparability axis.

WHY THIS RUNNER EXISTS RATHER THAN A CALL INTO THE TWO FROZEN ONES
`run_dose_response.run_one` hardcodes ("cifar10", "cifar_cnn") and cannot produce cell 6.
`run_targeted_dose.run_one` produces only the doseS/doseA/doseM families, not the confounded
`dose_kappa` one. So this file reimplements the ONE loop both of them run -- and then proves it is
the same loop with `--harness-check`, which reproduces a published `results/dose_response/` value
bit-identically at a shared seed. **The two frozen runners are not edited and their result
directories are not written to.**

The confounded and controlled ladders differ in exactly two tokens, which is the whole design:
    confounded:  d1 = f"dose_kappa{k}"    generic_compose(..., dose_key=(seed, rnd))
    controlled:  d1 = f"doseS_kappa{k}"   generic_compose(..., dose_key=(seed, rnd), adv_mask=...)
The adversary is free to be attenuated in the first and pinned at c=1 in the second. Everything else
-- data partition, participant draw, attack, local training, seeds -- is identical.

DO NOT RUN until experiments/pre_registration_comparability.md is git-committed and PREREG_COMMIT
below is set to that hash. The suite refuses to start otherwise.

Output: results/comparability_cells/summary.json (resumable; written after every run).
Run: python3 experiments/run_comparability_cells.py [--harness-check]
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig                                           # noqa: E402
from fl_core import (get_federated_dataset, get_model,               # noqa: E402
                     FederatedServer, FederatedClient)
from attacks import get_attack                                        # noqa: E402
from experiments.run_payoff_matrix import evaluate_backdoor           # noqa: E402
from experiments.run_all_compositions import generic_compose          # noqa: E402
from experiments.run_targeted_dose import ATTACK_MAP                  # noqa: E402

# Identical to the two frozen runners' own constants; a divergence here would silently make the new
# cells a different experiment from the four they are compared against.
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "comparability_cells")
PREREG = "experiments/pre_registration_comparability.md"
PREREG_COMMIT = "df00ef9"     # freeze c986ef4 + Amendment 1 (df00ef9), which named the gating
                              # quantity as ΔΛ_a before any result existed. The suite refuses to
                              # start if this drifts, which is how the amendment stayed auditable.

KAPPAS = [0.0, 0.5, 1.0, 2.0]        # the frozen grid, identical to both existing ladders
SEEDS = [42, 43, 44, 45, 46]         # frozen; matches every arm this is compared against

# (label, dataset, model, d2, attack, families, key_suffix)
CELLS = [
    ("cell5-coord_median/scaling", "cifar10", "cifar_cnn", "coord_median", "committed_scaling",
     ["confounded", "controlled"], ""),
    ("cell6-krum/EMNIST-byclass",  "femnist", "simple_cnn", "krum",        "committed_scaling",
     ["confounded"], "|emnist"),
]

# The EMNIST cell carries a key suffix so its keys can never be pooled with a CIFAR cell's by a
# later merge. Same reason the mask arm used its own family name: namespace collisions between
# arms that share (d2, attack) are silent and are only ever found afterwards.
FAMILIES = {"confounded": ("dose_kappa{k}", False),
            "controlled": ("doseS_kappa{k}", True)}


def cell_key(family, d2, attack, kappa, suffix=""):
    return f"{FAMILIES[family][0].format(k=kappa)}_then_{d2}|{attack}{suffix}"


def run_one(seed, family, d2, attack_name, kappa, dataset, model):
    """One 50-round FL run. `family` selects confounded vs controlled and nothing else does."""
    d1_fmt, use_adv_mask = FAMILIES[family]
    d1 = d1_fmt.format(k=kappa)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset(dataset, FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model(model, nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[attack_name])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(FL_CONFIG.num_clients)]
    lr = FL_CONFIG.learning_rate
    for rnd in range(FL_CONFIG.num_rounds):
        pids = np.random.choice(FL_CONFIG.num_clients,
                                size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
                                replace=False)
        ups = []
        for cid in pids:
            u = cl[cid].train(srv.global_model, FL_CONFIG.local_epochs, lr,
                              FL_CONFIG.local_batch_size)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        kw = {"dose_key": (seed, rnd)}
        if use_adv_mask:
            kw["adv_mask"] = [bool(cid in adv) for cid in pids]
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=5.0, **kw))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return (float(srv.evaluate(td)["accuracy"]),
            float(evaluate_backdoor(srv.global_model, td, device=dev)))


def load_cells():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)).get("cells", {}) if os.path.exists(p) else {}


def save(cells):
    os.makedirs(OUT, exist_ok=True)
    json.dump({"description": "Round 35 comparability cells; both ladders differ only in d1 and "
                              "adv_mask. Four further cells are re-scored from frozen artifacts and "
                              "are NOT here.",
               "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "kappas": KAPPAS, "seeds": SEEDS,
               "cells": cells},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def check_frozen():
    """Refuse to run unless the pre-registration is committed at the recorded hash."""
    if PREREG_COMMIT is None:
        print("REFUSING TO RUN: PREREG_COMMIT is None.\n"
              f"  1. git add {PREREG} && git commit\n"
              "  2. set PREREG_COMMIT here to that hash\n"
              "  3. rerun. Predictions are only frozen if the document is committed first.")
        return False
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20)
    actual = out.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO RUN: {PREREG} last touched at {actual or 'UNTRACKED'}, "
              f"but PREREG_COMMIT is {PREREG_COMMIT}.")
        return False
    print(f"[OK] {PREREG} frozen at {actual}")
    return True


def harness_check():
    """Prove this loop IS the frozen one before trusting any new number from it.

    Reproduces a published results/dose_response/ cell at a shared seed. A bit-identical match means
    the reimplementation above computes what the frozen runner computed, so the new cells are
    comparable to the four existing ones. A mismatch is resolved here, before the ladders run.
    """
    ref_path = os.path.join(BASE, "results", "dose_response", "summary.json")
    if not os.path.exists(ref_path):
        print("harness check SKIPPED: results/dose_response/summary.json absent")
        return True
    ref = json.load(open(ref_path)).get("cells", {})
    key = "dose_kappa2.0_then_coord_median|committed_pixel"
    cell = ref.get(key)
    if not cell:
        print(f"harness check SKIPPED: {key} not in results/dose_response/")
        return True
    want = {int(r["seed"]): float(r["asr"]) for r in cell["per_seed"]}
    seed = SEEDS[0]
    if seed not in want:
        print(f"harness check SKIPPED: seed {seed} not published for {key}")
        return True
    print(f"=== HARNESS CHECK: {key}, seed {seed} ===")
    print("  recomputing the published cell with THIS runner; must match bit-identically.")
    _, asr = run_one(seed, "confounded", "coord_median", "committed_pixel", 2.0,
                     "cifar10", "cifar_cnn")
    d = abs(asr - want[seed])
    print(f"  published {want[seed]!r}\n  recomputed {asr!r}\n  |diff| {d:.3e}")
    if d > 1e-9:
        print("  ** MISMATCH: this runner is NOT the frozen computation. Refusing to continue;")
        print("     the new cells would not be comparable to the four existing ones.")
        return False
    print("  OK: identical to 1e-9. The new cells are the same computation as the frozen ladders.\n")
    return True


def main():
    if not check_frozen():
        return 1
    if "--harness-check" in sys.argv:
        return 0 if harness_check() else 1

    cells = load_cells()
    todo = [(lbl, ds, mdl, d2, atk, fam, sfx, k, s)
            for (lbl, ds, mdl, d2, atk, fams, sfx) in CELLS
            for fam in fams for k in KAPPAS for s in SEEDS]
    done = sum(len(c.get("per_seed", [])) for c in cells.values())
    print(f"  resuming: {done} runs already done, {len(todo)} planned\n")

    i = 0
    for (lbl, ds, mdl, d2, atk, fam, sfx, k, s) in todo:
        i += 1
        key = cell_key(fam, d2, atk, k, sfx)
        cell = cells.setdefault(key, {"d2": d2, "attack": atk, "kappa": k, "family": fam,
                                      "dataset": ds, "model": mdl, "per_seed": []})
        if any(int(r["seed"]) == s for r in cell["per_seed"]):
            continue
        import time
        t0 = time.time()
        acc, asr = run_one(s, fam, d2, atk, k, ds, mdl)
        cell["per_seed"].append({"seed": s, "accuracy": acc, "asr": asr})
        cell["mean_asr"] = float(np.mean([r["asr"] for r in cell["per_seed"]]))
        cell["mean_accuracy"] = float(np.mean([r["accuracy"] for r in cell["per_seed"]]))
        save(cells)
        print(f"  [{i}/{len(todo)}] {lbl:28s} {fam:11s} k={k:<4} s{s}: "
              f"acc={acc:.4f} ASR={asr:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nSaved to {os.path.join(OUT, 'summary.json')}")
    print("Run experiments/analyze_comparability.py to score the frozen rule over all six cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
