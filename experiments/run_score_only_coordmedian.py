"""
The magnitude control on the sign-reversal cell: a score-only COORDINATE MEDIAN.

WHAT THIS ARM IS FOR
`experiments/pre_registration_score_only_coordmedian.md`. The paper's headline is one cell estimated
two ways -- coord_median / committed_pixel on CIFAR-10, where the outcome-gated design moves ASR
-0.273 and the within-defense (Mode S) design moves +0.125, both at n=20 -- and under Mode S the
upstream dose still displaces the aggregate the defense emits (Delta agg. = 0.670). Every existing
magnitude control in this paper is defined for SELECTORS only: run_all_compositions.generic_compose
raises at :578--:581 for any d2 outside ("krum", "cos_krum"), because "there is no single selected
update whose magnitude could be held fixed or rescaled". So the one cell the abstract headlines is
the one cell whose magnitude channel has never been closed. This closes it, or fails to.

THE INSTRUMENT
Per coordinate j: take the argmedian over clients of the TRANSFORMED stack, emit the UNTRANSFORMED
value of that client at that coordinate.

    idx[j] = argmedian_i  T(u)_i[j]     which client the transformed stack puts at the median
    out[j] = u_{idx[j]}[j]              that client's OWN, untransformed coordinate

fl_core/federated.py:180--:187 computes coord_median as stack.median(dim=0).values, and torch's
median returns matched .values/.indices, so this is that computation with .values replaced by a
gather at .indices into the untransformed stack. K=5 is odd every round, so the index is
single-valued. At kappa=0 apply_d1_transform executes `return updates` -- the caller's list OBJECT,
not a multiply by 1.0 -- so the two stacks are the same object and the gather returns exactly
.values. The identity is therefore EXACT, and --harness-check asserts it at 0.00e+00 against the
published kappa=0 rung before any new number is trusted. Not a tolerance: a nonzero difference voids
the kappa=0 import and stops the arm.

WHY THIS FILE EXISTS RATHER THAN A CALL INTO generic_compose
That function raises for this aggregator, by design and correctly. So this reimplements the ONE
50-round loop the frozen runners run and imports every constant that defines the cell rather than
restating it -- a divergence in any of them would silently make this a different experiment from the
arm it is compared against. The frozen runners are not edited and their result directories are not
written to. The one constant NOT imported is the seed list: run_comparability_cells.SEEDS is frozen
at [42..46] and this arm needs the published 42--61.

DO NOT RUN until the pre-registration is git-committed and PREREG_COMMIT below is that hash. The
suite refuses to start otherwise, and refuses on a dirty working tree too, because `git log -1`
reports the last commit that touched the file and is unchanged by uncommitted edits to it.

Output: results/score_only_coordmedian/summary.json (resumable; written after every run).
Run: PYTHONPATH=. python3 experiments/run_score_only_coordmedian.py [--harness-check]
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_core import (get_federated_dataset, get_model,                 # noqa: E402
                     FederatedServer, FederatedClient)
from attacks import get_attack                                         # noqa: E402
from experiments.run_payoff_matrix import evaluate_backdoor            # noqa: E402
from experiments.run_all_compositions import apply_d1_transform        # noqa: E402
# Imported, never retyped: a divergence here would make this a different experiment from the arm
# this one is compared against. ATTACK_MAP arrives through run_comparability_cells, which is where
# the cells this arm mirrors get it from as well.
from experiments.run_comparability_cells import (FL_CONFIG, ADV_FRACTION,   # noqa: E402
                                                 ATTACK_MAP, cell_key)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "score_only_coordmedian")
PREREG = "experiments/pre_registration_score_only_coordmedian.md"
PREREG_COMMIT = "16a17b2"     # the freeze, committed alone before this file was written

# The cell, fixed by the pre-registration and not a command-line option.
DATASET, MODEL, D2, ATTACK = "cifar10", "cifar_cnn", "coord_median", "committed_pixel"
KAPPA = 2.0                   # the frozen primary contrast rung; 0.5 and 1.0 are NOT run
SEEDS = list(range(42, 62))   # 42--61, the full published seed set (NOT run_comparability_cells.SEEDS,
                              # which is frozen at five and would silently give n=5 here)
ACC_FLOOR = 0.35              # on the rung MEAN, as in every arm on this cell; no rung substitution

# The published legs this arm is scored against. Read, never written.
PUBLISHED = [
    ("results/dose_replication/summary.json", [42, 43, 44, 45, 46]),
    ("results/reversal_seed_topup/summary.json", list(range(47, 62))),
]


def score_only_coord_median(transformed, updates):
    """Coordinate median of the TRANSFORMED stack, emitting UNTRANSFORMED values.

    Mirrors fl_core/federated.py:_coordinate_median line for line -- same per-parameter loop, same
    flatten, same .cpu().float() cast, same reshape and device restore -- with one change: where it
    takes `.values`, this gathers the untransformed stack at `.indices`. At kappa=0 the two stacks
    are the same object, so the gather returns exactly `.values` and this IS coord_median.
    """
    result = {}
    for name in updates[0]:
        shape = updates[0][name].shape
        device = updates[0][name].device
        scored = torch.stack([u[name].flatten().cpu().float() for u in transformed])
        emitted = torch.stack([u[name].flatten().cpu().float() for u in updates])
        idx = scored.median(dim=0).indices                 # which client sits at the median
        picked = emitted.gather(0, idx.unsqueeze(0)).squeeze(0)
        result[name] = picked.reshape(shape).to(device)
    return result


def run_one(seed, kappa, control=True):
    """One 50-round FL run on the frozen cell. `control` selects the instrument and nothing else.

    control=True  -> score-only coordinate median (this arm)
    control=False -> plain coord_median, i.e. the published Mode S arm, used only by the harness
    """
    d1 = f"doseS_kappa{kappa}"
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset(DATASET, FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model(MODEL, nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[ATTACK])
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
        # Mode S: the adversarial coefficient is pinned at 1 and only benign clients are dosed,
        # which is what adv_mask buys and why this arm still reads adversary identity.
        transformed = apply_d1_transform(ups, d1, tau=5.0, server=srv, dose_key=(seed, rnd),
                                         adv_mask=[bool(cid in adv) for cid in pids])
        srv.apply_update(score_only_coord_median(transformed, ups) if control
                         else score_only_coord_median(transformed, transformed))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return (float(srv.evaluate(td)["accuracy"]),
            float(evaluate_backdoor(srv.global_model, td, device=dev)))


def published():
    """The frozen Mode S legs at both rungs, per seed. Read-only."""
    out = {}
    for rel, seeds in PUBLISHED:
        p = os.path.join(BASE, rel)
        if not os.path.exists(p):
            continue
        cells = json.load(open(p)).get("cells", {})
        for kappa in (0.0, KAPPA):
            key = cell_key("controlled", D2, ATTACK, kappa)
            for r in cells.get(key, {}).get("per_seed", []):
                if int(r["seed"]) in seeds:
                    out.setdefault(kappa, {})[int(r["seed"])] = (float(r["accuracy"]),
                                                                 float(r["asr"]))
    return out


def load_cells():
    p = os.path.join(OUT, "summary.json")
    return json.load(open(p)).get("cells", {}) if os.path.exists(p) else {}


def save(cells):
    os.makedirs(OUT, exist_ok=True)
    json.dump({"description": "Score-only coordinate median: the magnitude control on the sign-"
                              "reversal cell. Per coordinate, argmedian of the TRANSFORMED stack, "
                              "emitting the UNTRANSFORMED value. kappa=0 is IMPORTED from the "
                              "published Mode S leg under an exact identity verified at 0.00e+00 "
                              "by --harness-check; only kappa=2 runs here.",
               "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "dataset": DATASET, "model": MODEL, "d2": D2, "attack": ATTACK,
               "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                          "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                          "kappa": KAPPA, "seeds": SEEDS, "acc_floor": ACC_FLOOR},
               "rung_provenance": "kappa=2.0 is COMPUTED here for seeds 42-61. kappa=0 is NOT here: "
                                  "at the identity the transform returns the stack unwrapped, so "
                                  "this control IS coord_median and the published Mode S kappa=0 "
                                  "rung is its value. Asserted bit-identical by --harness-check.",
               "estimands": {"primary": "paired mean of ASR(kappa=2, score-only) - ASR(kappa=0), "
                                        "two-sided 95% t interval, t_19 = 2.093",
                             "secondary": "paired mean of ASR(kappa=2, uncontrolled Mode S) - "
                                          "ASR(kappa=2, score-only): the magnitude channel's "
                                          "contribution, the kappa=0 term cancelling exactly"},
               "published_comparison": {"controlled_n20_delta": 0.125061,
                                        "controlled_n20_ci95": [0.0954, 0.1547],
                                        "confounded_n20_delta": -0.273278,
                                        "confounded_n20_ci95": [-0.3337, -0.2128]},
               "cells": cells},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def check_frozen():
    """Refuse to run unless the pre-registration is committed at the recorded hash AND clean."""
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
    # The hash check is necessary and NOT sufficient: `git log -1` reports the last commit that
    # touched the file, which uncommitted edits to it do not change. A freeze means the working tree
    # matches the commit, so check that too.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        print(f"REFUSING TO RUN: {PREREG} has uncommitted changes "
              f"({dirty.stdout.strip().split()[0]}), so it is not frozen at {actual} whatever "
              f"`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
        return False
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    return True


def harness_check():
    """Prove the kappa=0 import is exact before spending 20 runs that depend on it.

    At kappa=0 apply_d1_transform returns the caller's list object, so the scored and emitted stacks
    are identical and the gather returns exactly .values: this control IS coord_median, and the
    published Mode S kappa=0 rung is its value at every seed. The pre-registration makes the
    assertion a PRECONDITION at exactly 0.00e+00, on a seed where a published value exists, because
    a new seed would make the check vacuous. A nonzero difference voids the import and stops the arm.
    """
    pub = published()
    if 0.0 not in pub or 42 not in pub[0.0]:
        print("REFUSING TO RUN: no published Mode S kappa=0 value at seed 42 to assert against. "
              "The import cannot be verified, so it is not made.")
        return False
    want_acc, want_asr = pub[0.0][42]
    print(f"=== HARNESS CHECK: score-only coord_median at kappa=0 must BE the imported rung ===")
    print(f"    {DATASET}/{MODEL}, {D2}/{ATTACK} at seed 42")
    print(f"    imported (published Mode S kappa=0): acc={want_acc:.6f} ASR={want_asr:.6f}")
    acc, asr = run_one(42, 0.0, control=True)
    d_acc, d_asr = abs(acc - want_acc), abs(asr - want_asr)
    print(f"\n  score_only  acc={acc:.6f} ASR={asr:.6f}   "
          f"d=(+{d_acc:.2e}, +{d_asr:.2e})")
    if d_acc != 0.0 or d_asr != 0.0:
        print("\n  ** NOT BIT-IDENTICAL. The pre-registration makes 0.00e+00 a precondition, so the")
        print("     kappa=0 import is VOID and this arm does not run. The tolerance is not loosened")
        print("     and no approximate-identity variant is substituted. Report the failure.")
        return False
    print("\n  BIT-IDENTICAL. The kappa=0 import is valid and the arm may run.")
    return True


def main():
    if not check_frozen():
        return 1
    if "--harness-check" in sys.argv:
        return 0 if harness_check() else 1
    if not harness_check():
        return 1

    key = cell_key("controlled", D2, ATTACK, KAPPA, suffix="|score_only")
    cells = load_cells()
    cell = cells.setdefault(key, {"d2": D2, "attack": ATTACK, "rung": KAPPA,
                                  "control": "score_only_coord_median", "per_seed": []})
    done = {int(r["seed"]) for r in cell["per_seed"]}
    todo = [s for s in SEEDS if s not in done]
    print(f"\n{len(done)} of {len(SEEDS)} scored; {len(todo)} to run at kappa={KAPPA}")

    for i, seed in enumerate(todo, 1):
        import time
        t0 = time.time()
        acc, asr = run_one(seed, KAPPA, control=True)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        cell["mean_acc"] = sum(r["accuracy"] for r in cell["per_seed"]) / len(cell["per_seed"])
        cell["mean_asr"] = sum(r["asr"] for r in cell["per_seed"]) / len(cell["per_seed"])
        save(cells)
        print(f"  [{i}/{len(todo)}] score_only coord_median  s{seed}: "
              f"acc={acc:.4f} ASR={asr:.4f} ({time.time() - t0:.0f}s)")

    n = len(cell["per_seed"])
    print(f"\n{n} seeds scored, mean acc={cell.get('mean_acc', float('nan')):.4f} "
          f"mean ASR={cell.get('mean_asr', float('nan')):.4f}")
    if n == len(SEEDS) and cell["mean_acc"] < ACC_FLOOR:
        print(f"  ** rung-mean accuracy {cell['mean_acc']:.4f} is below ACC_FLOOR {ACC_FLOOR}: the")
        print("     contrast is uninterpretable, NO VERDICT STANDS, and no rung is substituted.")
    print("Score with experiments/analyze_score_only_coordmedian.py; this runner states no verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
