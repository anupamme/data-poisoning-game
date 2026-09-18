"""
Stage 2 of the CIFAR-100 composition-level replication: the ASR ladder, gated on stage 1's premise.

THE ARM. norm_clip -> coord_median on CIFAR-100 / cifar_cnn under committed_pixel, two rungs:

    tau = infinity   the identity. min(1.0, inf/||u||) is exactly 1.0, and u[k] * 1.0 is bit-identical
                     for a float32 tensor, so this leg IS coord_median standalone rather than merely
                     close to it. CifarCNN uses GroupNorm (fl_core/models.py:26), so no update entry is
                     an integer buffer that type promotion could disturb.
    tau = tau*       the median client update norm measured in stage 1 at this loop's own epoch count,
                     so roughly half the clients clip. Read from stage 1's artifact, never re-derived
                     here and never hand-entered.

WHY THIS ARM AND NOT ANOTHER. The twelfth review's single top ask is a composition-level replication of
the causal dissociation on a second dataset with NONZERO baseline adversarial admission. Every existing
arm in the paper fails at least one of those three: every nonzero-admission arm is a single-defense arm
on CIFAR-10, every second-dataset arm is at the admission floor, and no composition-level arm anywhere
in the paper has its baseline admission measured at all. This arm is all three at once, and its dose is
a REAL, deployed, ORACLE-FREE defense's own hyperparameter -- a member of the invariance theorem's
positive per-client rescaling class -- rather than a synthetic coefficient vector needing an adversary
mask.

IT IS GATED, AND IT CAN FAIL. This runner refuses to start unless
results/normclip_cifar100_admission.json exists and carries stage 1's ELIGIBLE verdict, which requires
all three premise conditions to have passed. The three failure literals are stage 1's to report; this
file adds no way around them.

--harness-check, TWO CHECKS, BOTH AGAINST PUBLISHED VALUES. A new-seed check would be vacuous, so both
run at seed 42 where a published number already exists:

  1. THE CIFAR-10 PAIR (registered). The same loop, dataset=cifar10 and tau=5.0, must reproduce
     results/all_compositions/summary.json's norm_clip_then_coord_median / committed_pixel row at seed
     42 to < 1e-9. That proves this is the loop the published pair was run with.
  2. THE CIFAR-100 IDENTITY RUNG (Amendment 1). Cell 7 of the comparability suite already publishes
     coord_median standalone on this exact dataset, model, attack and seed family, under
     dose_kappa0.0_then_coord_median|committed_pixel|cifar100. The tau = infinity leg IS that rung, so
     it must reproduce seed 42's published row to < 1e-9. That proves the identity leg is exact on the
     dataset this arm runs on, not merely on the one the theorem was illustrated on.

Check 2 also settles a question check 1 cannot: the two published anchors were produced by loops that
differ in whether FederatedServer was given a clean holdout, which fl_core/federated.py:225 reads only
for FLTrust and which is therefore inert for coord_median. This runner carries ONE loop and, if both
checks pass, that inference is established rather than assumed. Either verdict dict is written into the
artifact, not merely printed: Round 69 found one check in this repository whose verdict its caller
discarded, leaving a paper sentence witnessed only by a run's stdout.

AFTER THE LADDER, the five identity-leg seeds are compared against cell 7's five published values and
the comparison is written into the artifact. A mismatch is reported, not absorbed.

WHAT A NULL DOES NOT LICENSE. A flat ASR at unchanged admission is a dissociation on ONE pair, ONE
dataset, ONE attack, TWO rungs, n = 5. It is not a general law, it is not evidence that norm_clip is
safe, and no sentence in the paper may say that it is.

Rules frozen in experiments/pre_registration_normclip_cifar100.md (freeze 9cbd009, Amendment 1
2eeff40) before either runner existed. This suite writes only results/normclip_cifar100/. No existing
results directory is written and no existing runner, analyzer or pre-registration is edited.

Output: results/normclip_cifar100/summary.json (resumable; written after every run)

    PYTHONPATH=. python3 -m experiments.run_normclip_cifar100 --harness-check
    PYTHONPATH=. python3 -m experiments.run_normclip_cifar100
"""
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack                                                          # noqa: E402
from experiments.run_payoff_matrix import evaluate_backdoor                             # noqa: E402
# The composition step, the cell-key format and the FL configuration are imported rather than
# restated, so this ladder cannot drift from the suites it is checked against in any of them.
from experiments.run_all_compositions import generic_compose, pair_key                 # noqa: E402
from experiments.run_all_compositions import FL_CONFIG as COMPOSITIONS_CONFIG          # noqa: E402
from experiments.run_comparability_cells import (cell_key, FL_CONFIG,                   # noqa: E402
                                                 ADV_FRACTION)
from experiments.run_targeted_dose import ATTACK_MAP                                    # noqa: E402
# One definition of the write-boundary sanitizer, in the stage this runner is already gated on, so the
# two artifacts of this arm cannot end up with two different notions of what a non-finite float
# serializes to. Importing it runs stage 1's module body, which is constants and imports only.
from experiments.measure_admission_normclip_cifar100 import jsonable                    # noqa: E402

OUT_DIR = os.path.join(BASE, "results", "normclip_cifar100")
OUT = os.path.join(OUT_DIR, "summary.json")
STAGE1 = os.path.join(BASE, "results", "normclip_cifar100_admission.json")
ANCHOR_C10 = os.path.join(BASE, "results", "all_compositions", "summary.json")
ANCHOR_C100 = os.path.join(BASE, "results", "comparability_cells", "summary.json")

PREREG = "experiments/pre_registration_normclip_cifar100.md"
PREREG_COMMIT = "2eeff40"       # freeze 9cbd009 + Amendment 1 (2eeff40, append-only: 68 insertions,
                                # 0 deletions). 9cbd009 froze every threshold, seed list, tau rule and
                                # verdict literal this runner reads; the bump exists so the guard
                                # checks the document the paper will cite, which is what a resume must
                                # also read.

DATASET, MODEL = "cifar100", "cifar_cnn"
D1, D2, ATTACK = "norm_clip", "coord_median", "committed_pixel"
SEEDS = [42, 43, 44, 45, 46]            # frozen; the published pair's and cell 7's own seed family
INF = float("inf")
RUNGS = ("identity", "tau_star")        # endpoints only, frozen. No interior rung is run or implied.

# The primary interval, frozen: a paired 95% Student-t interval on 4 degrees of freedom. The constant
# is spelled out rather than computed so that no library version can move a published number.
T_CRIT_4DF = 2.776

# Inherited from pre_registration_comparability.md, where it is a floor on a rung's MEAN and where
# "seeds below it are flagged, never excluded". Cell 7's published identity accuracy on this dataset is
# 0.393, so the floor is informative here rather than vacuous.
ACC_FLOOR = 0.35

# The CIFAR-10 anchor this arm is NOT differenced against: n = 3 against this ladder's n = 5, and the
# two counts are never merged. It exists to show the pair is not new to the paper.
ANCHOR_C10_PAIR = (D1, D2)
ANCHOR_C10_MEAN_N3 = 0.37666666666666665

# Stage 2's verdict. The reversal literal is frozen verbatim in the pre-registration's demotion clause;
# the null wording is constrained there rather than fixed, and is stated here before any ASR exists so
# that this runner reports one of two strings rather than composing prose after seeing the numbers.
V_REVERSES = "DISSOCIATION REVERSES ON CIFAR-100 AT THE COMPOSITION LEVEL."
V_REPLICATES = ("DISSOCIATION REPLICATES AT THE COMPOSITION LEVEL ON CIFAR-100: one pair, one "
                "dataset, one attack, two rungs, n = 5. Not a general law, and not evidence that "
                "norm_clip is safe.")


# --------------------------------------------------------------------------------------------------
# The loop. One function, used for the ladder and for both harness checks.
# --------------------------------------------------------------------------------------------------
def run_one(seed, tau, dataset=DATASET, model=MODEL, attack_name=ATTACK, d1=D1, d2=D2):
    """One 50-round FL run of d1 -> d2 at a fixed tau. Returns (clean accuracy, ASR).

    This is run_all_compositions.run_one's loop with the dataset, model and tau lifted into arguments,
    which is why it can reproduce both published anchors. It is a separate function rather than an edit
    to that file: no existing runner is touched, and a runner whose dataset is hardcoded cannot express
    this arm.

    FederatedServer is constructed WITHOUT a clean holdout, as run_all_compositions.run_one does.
    run_comparability_cells.run_one passes one, but fl_core/federated.py:225 reads it only for FLTrust,
    so it is inert for coord_median: it consumes no RNG and touches no update. --harness-check
    establishes that by reproducing both anchors with this one loop.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset(dataset, FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model(model, nc), dev)
    adv = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[attack_name])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(FL_CONFIG.num_clients)]
    lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
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
        # No dose_key and no adv_mask: norm_clip is a real defense and needs neither. That is the
        # property that makes this arm oracle-free, and it is visible right here in the call.
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=tau))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return (float(srv.evaluate(td)["accuracy"]),
            float(evaluate_backdoor(srv.global_model, td, device=dev)))


# --------------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------------
def stage1_doc():
    return json.load(open(STAGE1)) if os.path.exists(STAGE1) else None


def check_config_agreement():
    """The CIFAR-10 check depends on this loop's FL configuration equalling the published pair's.

    run_all_compositions and run_comparability_cells each instantiate their own FLConfig. They agree
    field by field today; if a later edit moves either, check 1 would fail with no indication of why,
    so the equality is asserted here where it can be read.
    """
    fields = ("num_clients", "clients_per_round", "num_rounds", "local_epochs", "local_batch_size",
              "learning_rate", "lr_decay")
    diff = {f: (getattr(FL_CONFIG, f), getattr(COMPOSITIONS_CONFIG, f)) for f in fields
            if getattr(FL_CONFIG, f) != getattr(COMPOSITIONS_CONFIG, f)}
    if diff:
        print("REFUSING TO RUN: this loop's FL configuration does not equal the published CIFAR-10 "
              f"pair's.\n  differing fields (comparability, compositions): {diff}")
        return False
    return True


def check_frozen():
    """Refuse to run unless the pre-registration is committed at the recorded hash AND clean, and
    unless stage 1 has certified the premise on this dataset.

    The hash check alone is necessary and not sufficient: `git log -1` reports the last commit that
    touched the file, which is unchanged by uncommitted edits to it. That defect let a Round-57
    amendment pass its own gate, so the working-tree check is part of this guard from the start.
    """
    if not os.path.exists(os.path.join(BASE, PREREG)):
        print(f"REFUSING TO RUN: {PREREG} does not exist.")
        return False
    if PREREG_COMMIT in (None, "PENDING"):
        print("REFUSING TO RUN: the demotion clause is not frozen.\n"
              f"  1. git add {PREREG} && git commit\n"
              "  2. set PREREG_COMMIT here to that hash\n"
              "  3. rerun.\n"
              "An arm whose three verdicts are not committed in advance cannot report one against "
              "itself, which is the only thing that licenses running it.")
        return False
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20)
    actual = out.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO RUN: {PREREG} last touched at {actual or 'UNTRACKED'}, "
              f"but PREREG_COMMIT is {PREREG_COMMIT}.")
        return False
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        print(f"REFUSING TO RUN: {PREREG} has uncommitted changes "
              f"({dirty.stdout.strip().split()[0]}), so it is not frozen at {actual} whatever "
              "`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
        return False
    if not check_config_agreement():
        return False

    doc = stage1_doc()
    if doc is None:
        print(f"REFUSING TO RUN: {os.path.relpath(STAGE1, BASE)} does not exist. Stage 1 measures "
              "the premise -- nonzero baseline admission on CIFAR-100, an exact identity rung, and a "
              "coefficient share that does not move -- and stage 2 is gated on it.\n"
              "  PYTHONPATH=. python3 -m experiments.measure_admission_normclip_cifar100\n"
              "This gate covers --harness-check too, deliberately: the frozen order is stage 1 first, "
              "and a\nstage-2 artifact written before the premise exists would record a null premise "
              "that a later\nstage-1 run turns stale. The cost is ordering, not correctness.")
        return False
    if doc.get("prereg_commit") != PREREG_COMMIT:
        print(f"REFUSING TO RUN: stage 1 was run against prereg {doc.get('prereg_commit')}, "
              f"but this runner is frozen at {PREREG_COMMIT}. The two stages must read the same rules.")
        return False
    premise = doc.get("premise", {})
    if not premise.get("eligible"):
        print("REFUSING TO RUN: stage 1's verdict is not ELIGIBLE.\n"
              f"  {premise.get('verdict', '(no verdict recorded)')}\n"
              "The paper reports the attempt and this verdict in the abstract and in section 5, not "
              "in a footnote. Stage 2 does not run and no ASR for this arm is computed.")
        return False
    tau = doc.get("tau_star")
    if not (isinstance(tau, float) and np.isfinite(tau) and tau > 0.0):
        print(f"REFUSING TO RUN: stage 1 recorded tau_star = {tau!r}, which is not a usable "
              "threshold.")
        return False
    for path, what in ((ANCHOR_C10, "the published CIFAR-10 pair"),
                       (ANCHOR_C100, "cell 7's published CIFAR-100 identity rung")):
        if not os.path.exists(path):
            print(f"REFUSING TO RUN: {os.path.relpath(path, BASE)} does not exist, so {what} cannot "
                  "be reproduced and --harness-check would be vacuous.")
            return False
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    print(f"[OK] stage 1 ELIGIBLE at prereg {doc['prereg_commit']}, tau* = {tau:.6f}")
    return True


# --------------------------------------------------------------------------------------------------
# The two published anchors, read-only
# --------------------------------------------------------------------------------------------------
def anchor_c10_rows():
    """{seed: row} for the published CIFAR-10 norm_clip -> coord_median pixel pair. Never written."""
    if not os.path.exists(ANCHOR_C10):
        return {}
    pairs = json.load(open(ANCHOR_C10)).get("pairs", {})
    cell = pairs.get(pair_key(*ANCHOR_C10_PAIR), {}).get(ATTACK, {})
    return {int(r["seed"]): r for r in cell.get("per_seed", [])}


def anchor_c100_rows():
    """{seed: row} for cell 7's published CIFAR-100 coord_median identity rung. Never written."""
    if not os.path.exists(ANCHOR_C100):
        return {}
    key = cell_key("confounded", D2, ATTACK, 0.0, "|cifar100")
    cell = json.load(open(ANCHOR_C100)).get("cells", {}).get(key, {})
    return {int(r["seed"]): r for r in cell.get("per_seed", [])}


def _one_check(n, label, rows, seed, **run_kw):
    """Recompute one published row with this loop and demand equality to 1e-9."""
    ref = rows.get(seed)
    if ref is None:
        print(f"  [{n}/2] {label}: SKIPPED, seed {seed} not published")
        return {"check": label, "seed": seed, "status": "skipped", "ok": None}
    print(f"  [{n}/2] {label}\n        published  acc={ref['accuracy']!r} ASR={ref['asr']!r}",
          flush=True)
    t0 = time.time()
    acc, asr = run_one(seed, **run_kw)
    dacc, dasr = acc - ref["accuracy"], asr - ref["asr"]
    ok = bool(abs(dacc) < 1e-9 and abs(dasr) < 1e-9)
    print(f"        recomputed acc={acc!r} ASR={asr!r}")
    print(f"        |d| = ({abs(dacc):.3e}, {abs(dasr):.3e})  {'OK' if ok else '** MISMATCH'}  "
          f"({time.time() - t0:.0f}s)\n", flush=True)
    return {"check": label, "seed": seed, "status": "run", "ok": ok,
            "published": {"accuracy": ref["accuracy"], "asr": ref["asr"]},
            "recomputed": {"accuracy": acc, "asr": asr},
            "abs_delta": {"accuracy": abs(dacc), "asr": abs(dasr)}, "tolerance": 1e-9}


def harness_check():
    """Two runs at seed 42, both against published values, verdict written into the artifact."""
    seed = SEEDS[0]
    print(f"=== HARNESS CHECK: two published anchors, seed {seed} ===")
    print("    (1) the CIFAR-10 pair, so this is the loop the published pair was run with.")
    print("    (2) cell 7's CIFAR-100 identity rung, so the tau = infinity leg is exact on the")
    print("        dataset this arm runs on. Passing both also establishes that the clean holdout")
    print("        distinguishing the two published loops is inert for coord_median.\n", flush=True)

    a = _one_check(1, "CIFAR-10 norm_clip -> coord_median / pixel vs results/all_compositions/",
                   anchor_c10_rows(), seed, tau=5.0, dataset="cifar10", model=MODEL)
    b = _one_check(2, "CIFAR-100 coord_median identity vs results/comparability_cells/ (cell 7)",
                   anchor_c100_rows(), seed, tau=INF)

    ran = [x for x in (a, b) if x["status"] == "run"]
    passed = bool(ran) and all(x["ok"] for x in ran) and len(ran) == 2
    if not ran:
        print("  ALL CHECKS SKIPPED: nothing was verified. Treat as a failure, not a pass.")
    elif len(ran) < 2:
        print("  ** ONE CHECK DID NOT RUN. A half-checked harness is not a checked harness.")
    elif not passed:
        print("  ** REFUSING TO CONTINUE. A mismatch means either that this is not the loop the")
        print("     published values were produced with, or that the tau = infinity leg is not the")
        print("     identity on this dataset. Either way the ladder's two rungs would not be")
        print("     comparable to anything, including each other.")
    else:
        print("  OK, both checks equal to 1e-9. The loop is the published one and the identity leg")
        print("  is exact on CIFAR-100.")

    verdict = {"passed": passed, "n_run": len(ran), "n_expected": 2, "checks": [a, b],
               "note": "Written into the artifact rather than only printed. Round 69 found one check "
                       "in this repository whose verdict dict its caller discarded, leaving a paper "
                       "sentence witnessed only by a run's stdout."}
    save(load_cells(), harness=verdict)
    print(f"\n  verdict written to {os.path.relpath(OUT, BASE)} under 'harness_check'")
    return passed


# --------------------------------------------------------------------------------------------------
# Artifact
# --------------------------------------------------------------------------------------------------
def rung_key(rung):
    """A namespace of this arm's own. Cell 7's keys share (d2, attack, dataset) with this ladder, and a
    collision between two arms that share those is silent and is only ever found afterwards."""
    return f"norm_clip_{rung}_then_{D2}|{ATTACK}|cifar100"


def load_doc():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def load_cells():
    return load_doc().get("cells", {})


def save(cells, harness=None, verdict=None):
    """Write the artifact. `harness` and `verdict` are preserved when not supplied, so a ladder run
    cannot erase the harness verdict that licensed it."""
    prev = load_doc()
    doc = stage1_doc() or {}
    premise = doc.get("premise", {})
    json.dump(jsonable({
        "description": "Stage 2 of the CIFAR-100 composition-level replication: the ASR ladder for "
                       f"{D1} -> {D2} on {DATASET}/{MODEL} under {ATTACK}, two rungs "
                       "(tau = infinity, tau = tau*), seeds 42-46. Gated on stage 1's premise "
                       "verdict in results/normclip_cifar100_admission.json. No existing results "
                       "directory is written.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "dataset": DATASET, "model": MODEL, "d1": D1, "d2": D2, "attack": ATTACK,
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "local_epochs": FL_CONFIG.local_epochs, "seeds": SEEDS, "rungs": list(RUNGS),
                   "tau_star": doc.get("tau_star"), "acc_floor": ACC_FLOOR,
                   "t_crit_4df": T_CRIT_4DF},
        "stage1": {"artifact": os.path.relpath(STAGE1, BASE),
                   "prereg_commit": doc.get("prereg_commit"),
                   "verdict": premise.get("verdict"), "eligible": premise.get("eligible"),
                   "tau_rule": doc.get("tau_rule")},
        "endpoint_only": "Endpoint rungs only. No interior tau is run, so no trend statistic over tau "
                         "exists for this arm and none may be reported.",
        "n_discipline": f"The published CIFAR-10 anchor for this pair is n = 3 (mean ASR "
                        f"{ANCHOR_C10_MEAN_N3}) and this ladder is n = 5. The two counts are never "
                        "merged and the anchor is never differenced against CIFAR-100; it exists to "
                        "show the pair is not new to the paper.",
        "harness_check": prev.get("harness_check") if harness is None else harness,
        "verdict": prev.get("verdict") if verdict is None else verdict,
        "cells": cells,
    }), open(OUT, "w"), indent=2)


def _record(cells, rung, seed, tau, acc, asr):
    cell = cells.setdefault(rung_key(rung), {"d1": D1, "d2": D2, "attack": ATTACK, "rung": rung,
                                             "tau": (None if tau == INF else tau),
                                             "tau_is_infinity": bool(tau == INF),
                                             "dataset": DATASET, "model": MODEL, "per_seed": []})
    row = {"seed": int(seed), "accuracy": float(acc), "asr": float(asr),
           "below_acc_floor": bool(acc < ACC_FLOOR)}
    # Replace, never append blind: a resume that appended would silently inflate n on one leg.
    cell["per_seed"] = [r for r in cell["per_seed"] if int(r["seed"]) != int(seed)] + [row]
    cell["per_seed"].sort(key=lambda r: r["seed"])
    cell["mean_asr"] = float(np.mean([r["asr"] for r in cell["per_seed"]]))
    cell["mean_accuracy"] = float(np.mean([r["accuracy"] for r in cell["per_seed"]]))
    cell["n"] = len(cell["per_seed"])
    cell["n_below_acc_floor"] = sum(1 for r in cell["per_seed"] if r["below_acc_floor"])


def _has(cells, rung, seed):
    return any(int(r["seed"]) == seed
               for r in cells.get(rung_key(rung), {}).get("per_seed", []))


# --------------------------------------------------------------------------------------------------
# The frozen primary rule
# --------------------------------------------------------------------------------------------------
def paired(cells, field):
    """Per-seed differences tau* minus identity, over the seeds present on BOTH legs."""
    lo = {int(r["seed"]): r[field] for r in cells.get(rung_key("identity"), {}).get("per_seed", [])}
    hi = {int(r["seed"]): r[field] for r in cells.get(rung_key("tau_star"), {}).get("per_seed", [])}
    seeds = sorted(set(lo) & set(hi))
    return seeds, [float(hi[s] - lo[s]) for s in seeds]


def interval(diffs):
    """The frozen paired 95% Student-t interval. t_4 = 2.776 is a constant, not a library call."""
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": (float(diffs[0]) if n else float("nan")), "sd": float("nan"),
                "ci95": [float("nan"), float("nan")], "excludes_zero": None}
    m = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    half = T_CRIT_4DF * sd / float(np.sqrt(n))
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - half, m + half],
            "half_width": half, "excludes_zero": bool((m - half) * (m + half) > 0.0),
            "t_crit": T_CRIT_4DF,
            "note": ("t_4 = 2.776 is frozen for n = 5. At any other n this interval is not the "
                     "pre-registered one and is labelled as such.") if n != 5 else None}


def identity_anchor_comparison(cells):
    """Every identity seed against cell 7's published value. Reported, never absorbed."""
    pub = anchor_c100_rows()
    rows = []
    for r in cells.get(rung_key("identity"), {}).get("per_seed", []):
        p = pub.get(int(r["seed"]))
        if p is None:
            rows.append({"seed": r["seed"], "status": "not_published"})
            continue
        rows.append({"seed": r["seed"], "status": "compared",
                     "published_asr": p["asr"], "asr": r["asr"],
                     "abs_delta_asr": abs(r["asr"] - p["asr"]),
                     "published_accuracy": p["accuracy"], "accuracy": r["accuracy"],
                     "abs_delta_accuracy": abs(r["accuracy"] - p["accuracy"]),
                     "bit_equal": bool(abs(r["asr"] - p["asr"]) < 1e-9
                                       and abs(r["accuracy"] - p["accuracy"]) < 1e-9)})
    compared = [r for r in rows if r["status"] == "compared"]
    return {"per_seed": rows, "n_compared": len(compared),
            "all_bit_equal": bool(compared) and all(r["bit_equal"] for r in compared),
            "published_key": cell_key("confounded", D2, ATTACK, 0.0, "|cifar100"),
            "meaning": "The tau = infinity leg IS cell 7's published coord_median standalone rung, "
                       "recomputed. Equality is a five-seed bit-exact replication of a published row; "
                       "a mismatch is news and is reported as such."}


def stage1_admission_at_tau_star(doc):
    """coord_median's measured admission displacement at tau*, from stage 1, at the ladder's own epoch
    count. This is the premise the suppression reading rests on and it is quoted, not assumed."""
    ep = doc.get("config", {}).get("local_epochs_ladder")
    rows = [r for r in doc.get("per_round_rungs", [])
            if r.get("rung") == "tau_star" and r.get("local_epochs") == ep
            and r.get("n_adv_in_round", 0) > 0]
    if not rows:
        return None
    d = [abs(float(r["coord_median_adv_frac_delta"])) for r in rows]
    return {"local_epochs": ep, "n_adv_rounds": len(d), "mean_abs_displacement": float(np.mean(d)),
            "max_abs_displacement": float(max(d)),
            "support_unchanged_rounds": int(sum(1 for x in d if x == 0.0))}


def report(cells):
    """The frozen primary rule, evaluated on whatever rows exist, plus the two frozen literals."""
    doc = stage1_doc() or {}
    seeds, dasr = paired(cells, "asr")
    _, dacc = paired(cells, "accuracy")
    asr_ci, acc_ci = interval(dasr), interval(dacc)
    adm = stage1_admission_at_tau_star(doc)

    print(f"\n{'=' * 78}")
    print("PRIMARY ESTIMAND: paired per-seed ASR(tau*) - ASR(infinity)")
    print(f"{'=' * 78}")
    for rung in RUNGS:
        cell = cells.get(rung_key(rung), {})
        rows = cell.get("per_seed", [])
        if not rows:
            print(f"  {rung:9s} (no rows yet)")
            continue
        print(f"  {rung:9s} n={len(rows)}  mean ASR {cell['mean_asr']:.6f}  "
              f"mean acc {cell['mean_accuracy']:.4f}"
              + (f"  ** {cell['n_below_acc_floor']} seed(s) below the {ACC_FLOOR} accuracy floor "
                 "(flagged, never excluded)" if cell["n_below_acc_floor"] else ""))
    if seeds:
        print(f"\n  paired over seeds {seeds}:")
        print("    " + "  ".join(f"s{s}:{d:+.4f}" for s, d in zip(seeds, dasr)))
        print(f"    ASR      mean {asr_ci['mean']:+.6f}  sd {asr_ci['sd']:.6f}  "
              f"95% CI [{asr_ci['ci95'][0]:+.6f}, {asr_ci['ci95'][1]:+.6f}]  "
              f"excludes zero: {asr_ci['excludes_zero']}")
        print(f"    accuracy mean {acc_ci['mean']:+.6f}  sd {acc_ci['sd']:.6f}  "
              f"95% CI [{acc_ci['ci95'][0]:+.6f}, {acc_ci['ci95'][1]:+.6f}]  "
              f"excludes zero: {acc_ci['excludes_zero']}")
    print(f"\n  the CIFAR-10 anchor for this pair, NOT differenced against the above: "
          f"mean ASR {ANCHOR_C10_MEAN_N3:.6f} at n = 3.")

    anchor = identity_anchor_comparison(cells)
    print(f"\n  identity leg against cell 7's published rung: {anchor['n_compared']} seed(s) "
          f"compared, all bit-equal: {anchor['all_bit_equal']}")
    for r in anchor["per_seed"]:
        if r["status"] == "compared" and not r["bit_equal"]:
            print(f"    ** seed {r['seed']}: |dASR| {r['abs_delta_asr']:.3e}, "
                  f"|dacc| {r['abs_delta_accuracy']:.3e}")

    if adm:
        print(f"\n  stage 1's measured admission displacement at tau* ({adm['local_epochs']} local "
              f"epochs): mean |d| {adm['mean_abs_displacement']:.6f}, max {adm['max_abs_displacement']:.6f}, "
              f"exactly unchanged in {adm['support_unchanged_rounds']}/{adm['n_adv_rounds']} rounds")

    verdict = None
    if len(seeds) < len(SEEDS):
        print(f"\n  VERDICT WITHHELD: {len(seeds)}/{len(SEEDS)} seeds are paired. The frozen rule is "
              "an n = 5 interval and is not evaluated at a smaller n.")
    elif asr_ci["excludes_zero"]:
        verdict = V_REVERSES
        print(f"\n  {verdict}")
        print("  Suppression moved between the two rungs while stage 1 certified that the "
              "adversarial\n  coefficient share did not. This is the round's headline, reported "
              "against us, in the\n  abstract and in Figure 1, at the cost of the paper's "
              "dissociation claim.")
    else:
        verdict = V_REPLICATES
        print(f"\n  {verdict}")
        print("  A flat ASR at unchanged admission is a dissociation on this one cell. It is not a "
              "general\n  law, it is not evidence that norm_clip is safe, and no sentence in the "
              "paper may say so.")

    return {"verdict": verdict, "paired_seeds": seeds,
            "asr_diff": dict(asr_ci, per_seed=dict(zip(map(str, seeds), dasr))),
            "accuracy_diff": acc_ci,
            "cifar10_anchor_n3_mean_asr": ANCHOR_C10_MEAN_N3,
            "identity_anchor_check": anchor,
            "stage1_admission_at_tau_star": adm}


# --------------------------------------------------------------------------------------------------
def main():
    if not check_frozen():
        return 1
    if "--harness-check" in sys.argv:
        return 0 if harness_check() else 1

    doc = stage1_doc()
    tau_star = float(doc["tau_star"])
    cells = load_cells()
    prev = load_doc()
    if not (prev.get("harness_check") or {}).get("passed"):
        print("REFUSING TO RUN: --harness-check has not passed on this tree.\n"
              "  PYTHONPATH=. python3 -m experiments.run_normclip_cifar100 --harness-check\n"
              "Both rungs are compared against published values before either leg is trusted.")
        return 1

    taus = {"identity": INF, "tau_star": tau_star}
    print(f"=== STAGE 2: {D1} -> {D2} on {DATASET}/{MODEL}, {ATTACK.replace('committed_', '')} ===")
    print(f"    rungs tau in {{infinity, {tau_star:.6f}}}, seeds {SEEDS[0]}-{SEEDS[-1]}, "
          f"{len(SEEDS) * len(RUNGS)} runs")
    print(f"    rules frozen at {PREREG_COMMIT}; stage 1's verdict: {doc['premise']['verdict']}")
    print("    seed-major, so an interruption leaves COMPLETE pairs -- equal n on both legs, which "
          "is the\n    only shape the paired contrast can read.\n", flush=True)

    todo = [(seed, rung) for seed in SEEDS for rung in RUNGS]
    t0 = time.time()
    for i, (seed, rung) in enumerate(todo, 1):
        if _has(cells, rung, seed):
            continue
        t = time.time()
        acc, asr = run_one(seed, taus[rung])
        _record(cells, rung, seed, taus[rung], acc, asr)
        save(cells)
        print(f"  [{i}/{len(todo)}] s{seed} {rung:9s} acc={acc:.4f} ASR={asr:.4f}"
              + ("  * below acc floor" if acc < ACC_FLOOR else "")
              + f" ({time.time() - t:.0f}s)", flush=True)

    verdict = report(cells)
    save(cells, verdict=verdict)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h")
    print(f"Saved to {OUT}")
    print("Count `per_seed` entries in the artifact to judge progress; the [i/N] index above counts "
          "resumed-and-skipped runs and can go backwards across restarts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
