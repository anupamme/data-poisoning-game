"""Arm D: the full 42-pair composition suite on CIFAR-100.

WHY THIS ARM EXISTS. A review named a "full composition replication on a second dataset" as one of two
especially important requirements, and objected that the existing CIFAR-100 evidence is an
intervention ladder rather than a suite. experiments/pre_registration_cifar100_composition_suite.md
(committed alone at f559b85) is the freeze and this is its runner.

WHAT IS IMPORTED UNCHANGED. Both pair lists, so the 42 are the published 42 and not a re-listing:
run_all_compositions.PAIRS (18, the development wave) and run_wave2_held_out.PAIRS (24, held out).
generic_compose and apply_d1_transform come from run_all_compositions, ATTACKS and FL_CONFIG and
ADV_FRACTION from the same place, evaluate_backdoor from run_payoff_matrix, ACC_FLOOR and ATTACK_MAP
from run_targeted_dose, SUPPRESS_ASR from run_dose_resnet18, and the screen itself -- predict_pair,
predict_composition and SUPPRESSION_THRESHOLD -- from analyze_composability, the module that computes
the paper's published screen figures. Nothing about the screen is re-derived here. run_one hardcodes
"cifar10" at run_all_compositions.py:602 and is not editable, so this arm supplies its own loop and
calls the shared primitives; nothing is copied out of it.

TWO THRESHOLDS THAT ARE NOT THE SAME THRESHOLD, AND MUST NOT BE SUBSTITUTED FOR EACH OTHER. The
screen's C1 asks whether a defense suppresses at SUPPRESSION_THRESHOLD = 0.3
(analyze_composability.py); the power rule that decides whether a CELL CAN BE INFORMATIVE asks
whether ASR is below SUPPRESS_ASR = 0.5 at accuracy at or above ACC_FLOOR (run_dose_resnet18.py:133,
paper/supplementary.tex:247). The freeze writes gate 0's decision rule at 0.5, and the freeze's rule
is what decides the run. Both numbers are imported from their own homes, both are reported, and
neither stands in for the other anywhere below.

GATE 0 IS ALSO A MISSING-BASELINE FIX, WHICH IS MORE THAN THE FREEZE CLAIMED FOR IT. The screen is not
a function of the pair alone: predict_composition reads STANDALONE per-dataset ASR for both members,
so on CIFAR-100 it needs seven standalone aggregators and results/cifar100/per_seed_results.json has
five. reputation and foolsgold are absent, and one or both is a member of 22 of the 42 pairs -- 20
pairs with exactly one missing and the 2 reciprocal pairs with both. So gate 0's 20-run probe is not
only the feasibility test the freeze describes; it supplies the two baselines without which the
screen's label on those 22 does not rest on evidence. That is recorded here rather than discovered
when the confusion matrix is scored.

AND THE IMPORTED PREDICTOR HAS TWO MISSING-BASELINE FAILURE MODES THIS ARM HANDLES AT THE CALL SITE.
Neither is a defect on the dataset the screen was built for, where all seven baselines exist.

  1. predict_composition returns UNKNOWN only when BOTH members are missing, and predict_pair then
     returns HIGH-if-any-HIGH else LOW -- so UNKNOWN on both attacks comes back LOW. Measured before
     the probe: exactly the 2 reciprocal pairs reputation<->foolsgold, both labelled LOW, which is
     the scarce class the screen is scored on.
  2. With ONE member missing, nothing is signalled at all: that member's `suppresses` is False and
     the pair comes back HIGH on a baseline that does not exist, since C1 needs only one suppressor.
     Measured before the probe: 20 of the 42 pairs, all HIGH. This mode produces no UNKNOWN anywhere
     and is invisible unless the baseline table is checked directly.

analyze_composability.py is not editable and is not edited. screen_pair below applies a stricter rule
than the predictor's own -- a pair is UNPREDICTABLE if EITHER member lacks a baseline for EITHER
attack, whatever label came back -- and such pairs are excluded from the confusion matrix with their
count stated. With gate 0's two aggregators in the table all 42 become predictable on evidence, which
is why the probe is a precondition of scoring and not only of running.

WHAT THIS ARM DOES NOT DO. It does not weaken gate 2's floor to keep cells. CIFAR-100 at cifar_cnn
reaches about 0.40 clean accuracy against an imported floor of 0.35, so a substantial fraction of the
42 pairs is expected to land below it; those cells are reported uninterpretable and COUNTED, never
dropped, and never scored HIGH because a collapsed model has a high ASR. Every fraction this arm
prints names its exclusion count in the same sentence, and every accuracy figure is printed beside
the base rate and the constant-HIGH baseline.

    the CIFAR-10 cross-check:  PYTHONPATH=. python3 -m experiments.run_cifar100_composition_suite --harness-check
    gate 0's 20-run probe:     PYTHONPATH=. python3 -m experiments.run_cifar100_composition_suite --probe
    the suite (gate 0 first):  PYTHONPATH=. python3 -m experiments.run_cifar100_composition_suite
    capped to wave 1 (180 runs): ... run_cifar100_composition_suite --wave1-only

--wave1-only caps the ATTEMPT to wave 1's 18 pairs. It changes no threshold, no estimand and no
denominator: the menu stays 42 everywhere, and the cap is recorded in the artifact's run_scope field so
a deliberately unrun held-out wave can never be confused with a crash. See RUN_WAVES below.
"""

import os, sys, json, time, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from fl_core import (get_federated_dataset, get_model, FederatedServer,  # noqa: E402
                     FederatedClient)
from attacks import get_attack  # noqa: E402
# The two pair lists, IMPORTED so the 42 are the published 42. Both modules call
# os.makedirs(..., exist_ok=True) at module level on a directory that already exists, which is a
# no-op and not a write.
from experiments.run_all_compositions import (PAIRS as PAIRS_WAVE1, ATTACKS,  # noqa: E402
                                              FL_CONFIG, ADV_FRACTION, pair_key,
                                              generic_compose, apply_d1_transform)
from experiments.run_wave2_held_out import PAIRS as PAIRS_WAVE2  # noqa: E402
from experiments.run_payoff_matrix import evaluate_backdoor  # noqa: E402
# The gate the freeze mandates, imported rather than rewritten. It pins the expected behaviour PER
# ATTACK -- the update must change iff the attack overrides manipulate_update -- which is why a naive
# "assert the tensor changed" would have aborted all 210 committed_pixel runs.
from experiments.adversary_hook import apply_adversary, defines_own_manipulate  # noqa: E402
from experiments.run_targeted_dose import ACC_FLOOR, ATTACK_MAP  # noqa: E402
# The power rule's threshold, from the module where it already lives.
from experiments.run_dose_resnet18 import SUPPRESS_ASR  # noqa: E402
# The screen itself. Not re-derived: this is the module that computes the paper's published figures.
from experiments.analyze_composability import (predict_pair, predict_composition,  # noqa: E402
                                               SUPPRESSION_THRESHOLD, COMMITTED_ATTACKS,
                                               DEFENSE_TYPES)

PREREG_COMMIT = "f559b85"

DATASET, MODEL = "cifar100", "cifar_cnn"
SEEDS = list(range(42, 47))                     # 42-46, frozen
# Wave 1 first, because all 5 CIFAR-10 LOW pairs are in wave 1, so wave 1 is where the LOW class can
# exist at all and where the replication question is decidable. Order frozen so any partial
# completion is a pre-specified subset rather than an arbitrary truncation.
PAIRS = [tuple(p) for p in PAIRS_WAVE1] + [tuple(p) for p in PAIRS_WAVE2]
WAVE_OF = {tuple(p): 1 for p in PAIRS_WAVE1}
WAVE_OF.update({tuple(p): 2 for p in PAIRS_WAVE2})

# --wave1-only CAPS the run to wave 1. This is a coverage decision, NOT a rule change: PAIRS stays 42,
# check_frozen still asserts 42, every denominator in score() stays 42, and the cap is written into the
# artifact as a DELIBERATE stop so it can never later be read as a crash or a silent truncation. The
# freeze pre-authorizes this exact shape -- the pair order was frozen so that "any partial completion is
# a pre-specified subset rather than an arbitrary truncation" (prereg :171-172) -- and it also fixes the
# reporting: the completed pair count, the completed wave, and the fact that the remainder is unrun,
# "as an incompleteness, not as a suite" (prereg :184-186).
RUN_WAVES = (1, 2)


def run_pairs():
    """The pairs this invocation will attempt, in the frozen order, filtered by the cap."""
    return [p for p in PAIRS if WAVE_OF[p] in RUN_WAVES]


def scope_note():
    """The cap, stated as a disclosure, or None when the whole menu is being attempted."""
    if tuple(RUN_WAVES) == (1, 2):
        return None
    todo = run_pairs()
    waves = "+".join(str(w) for w in RUN_WAVES)
    return (
        f"CAPPED TO WAVE {waves}: {len(todo)} of the {len(PAIRS)} menu pairs x {len(ATTACKS)} attacks "
        f"x {len(SEEDS)} seeds = {len(todo) * len(ATTACKS) * len(SEEDS)} of "
        f"{len(PAIRS) * len(ATTACKS) * len(SEEDS)} runs were ATTEMPTED. The other "
        f"{len(PAIRS) - len(todo)} pairs are DELIBERATELY UNRUN -- a compute decision taken before the "
        "loop started, not a failure and not a loss -- which is why the freeze fixed the pair order in "
        "advance. What the cap keeps: wave 1 is where the replication question is decidable, since all "
        "5 CIFAR-10 LOW pairs are in wave 1. What the cap gives up, stated rather than glossed: the "
        "held-out wave is unrun, so no held-out generalization claim rests on this arm, and this is NOT "
        "the full 42-pair menu and is never described as one. n_incomplete counts these deliberately "
        "unrun pairs together with any that merely did not finish; THIS field is what separates them.")


def scope_field():
    """Sticky. A recorded cap is never erased by a later uncapped --harness-check or --report, which
    would delete the disclosure while leaving the capped numbers in place."""
    new = scope_note()
    if new is not None:
        return new
    if os.path.exists(out_path):
        try:
            return json.load(open(out_path)).get("run_scope")
        except Exception:
            return None
    return None

# Gate 0's probe: the two aggregators absent from the CIFAR-100 artifact, standalone. d1 is fedavg,
# which apply_d1_transform passes through, so "fedavg then X" IS X standalone -- the same DEGENERATE
# convention the screen itself uses. The probe therefore shares one code path with the suite instead
# of being a second implementation of "standalone".
PROBE_D1 = "fedavg"
PROBE_D2 = ["reputation", "foolsgold"]

# The CIFAR-10 cross-check cell. A real d1 transform and a real d2 weighting, so the check exercises
# apply_d1_transform's live branch rather than a pass-through, on both attacks: one that overrides
# manipulate_update and one that does not.
XCHECK_PAIR, XCHECK_SEED = ("norm_clip", "reputation"), 42
# Accuracy is stored to FOUR DECIMALS in both published wave artifacts (verified across all 356 rows),
# so accuracy cannot be compared bit-exactly against them and this runner does not claim to. ASR is
# full precision there and IS compared exactly. Stated rather than glossed, because "bit-identical"
# about a rounded number is a false claim.
XCHECK_ACC_DECIMALS = 4

out_dir = os.path.join(base, "results", "cifar100_composition_suite")
out_path = os.path.join(out_dir, "summary.json")
probe_path = os.path.join(out_dir, "probe.json")
PREREG = os.path.join(base, "experiments", "pre_registration_cifar100_composition_suite.md")
# Read, never written: the CIFAR-100 standalone baselines the screen needs, and the two published
# CIFAR-10 waves the cross-check reads.
CIFAR100_STANDALONE = os.path.join(base, "results", "cifar100", "per_seed_results.json")
WAVE1_PUBLISHED = os.path.join(base, "results", "all_compositions", "summary.json")
WAVE2_PUBLISHED = os.path.join(base, "results", "wave2_held_out", "summary.json")


def prereg_md5():
    import hashlib
    return hashlib.md5(open(PREREG, "rb").read()).hexdigest()


def check_frozen():
    """Refuse to start unless the freeze is committed and clean, and the 42 really are 42."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the frozen predictions and the outcome branches are not "
                 f"committed.\n  1. git commit {PREREG} alone\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen prediction is unfalsifiable, which is the entire point of the freeze.")
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify the freeze ({e}).")
    # An empty log must fail LOUDLY. PREREG_COMMIT.startswith(log[:7]) is vacuously true when log is
    # "", so a file git knows nothing about would otherwise pass the hash comparison.
    if not log:
        sys.exit(f"REFUSING TO RUN: git has no commit touching {PREREG}. A freeze that git cannot "
                 "see is not a freeze.")
    if not log.startswith(PREREG_COMMIT[:7]) and not PREREG_COMMIT.startswith(log[:7]):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last committed at {log!r}, not "
                 f"{PREREG_COMMIT!r}.")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted modifications ({dirty!r}). The freeze "
                 "is whatever is committed, not whatever is on disk.")
    # The freeze's own arithmetic, re-checked rather than trusted: the union of the two imported
    # waves must be 42 with no overlap, because 42 is the denominator of the paper's published
    # screen figure and a silently overlapping union would quietly change it.
    u = set(PAIRS)
    if len(u) != 42 or len(PAIRS) != 42:
        sys.exit(f"REFUSING TO RUN: the imported waves give {len(PAIRS)} pairs, {len(u)} distinct, "
                 "not 42. The suite's denominator is the paper's published denominator; it is not "
                 "adjusted here.")
    if not os.path.exists(CIFAR100_STANDALONE):
        sys.exit(f"REFUSING TO RUN: {CIFAR100_STANDALONE} does not exist. The screen's CIFAR-100 "
                 "predictions are a function of standalone per-dataset baselines, so without it "
                 "there is nothing to predict with.")


# ---------------------------------------------------------------------------------------------
# One run. The loop MIRRORS run_all_compositions.run_one exactly, with two differences, both stated.
# ---------------------------------------------------------------------------------------------

def run_one(seed, d1, d2, attack_name, dataset=DATASET, model=MODEL, verify=True):
    """One composition run, the wave-1 loop with dataset="cifar100" and the adversary step hoisted.

    Two differences from run_all_compositions.run_one, and nothing else:

      1. dataset is a parameter. That file hardcodes "cifar10" at :602 and is not editable.
      2. The adversary step is hoisted out of the participant loop into
         adversary_hook.apply_adversary(..., verify=True), which is the freeze's gate 1.

    The hoist is numerically neutral for both committed attacks, by the argument run_targeted_dose
    already records for its own hoist: backdoor_pixel inherits the identity manipulate_update and
    model_scaling is an elementwise multiply, so neither consumes RNG nor reads another client's
    update, and server.global_model is not mutated mid-round. The CIFAR-10 cross-check
    (--harness-check) tests that claim against published rows rather than resting on it.

    Everything else is the wave-1 loop: same seeding order, same device choice, same alpha of 0.5,
    same adversary set, same poison_dataset, same client construction order, same participant draw,
    same generic_compose(server, updates, d1, d2, tau=5.0) with no adv_mask, same
    lr_decay, same server.evaluate, same evaluate_backdoor at its default trigger_size=4 and
    target_class=0 -- which are dataset-agnostic, so they are valid on a 100-class problem.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        dataset, FL_CONFIG.num_clients, 0.5, seed)
    server = FederatedServer(get_model(model, num_classes), device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))
    attack = get_attack(ATTACK_MAP.get(attack_name, attack_name))

    clients = []
    for i in range(FL_CONFIG.num_clients):
        ds = client_datasets[i]
        if i in adversarial_ids:
            ds = attack.poison_dataset(ds)
        clients.append(FederatedClient(i, ds, device))

    current_lr = FL_CONFIG.learning_rate
    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients), replace=False)
        updates = [clients[cid].train(server.global_model, FL_CONFIG.local_epochs,
                                      current_lr, FL_CONFIG.local_batch_size)
                   for cid in participant_ids]
        # Gate 1, on EVERY round of EVERY run. The cost is one tensor clone on one probe client, and
        # the defect it catches has already shipped once in this codebase: a committed_scaling arm
        # that was bit-identical to its committed_pixel arm because manipulate_update was never
        # called, failing silently. An AssertionError here aborts the run and is never downgraded.
        updates = apply_adversary(updates, participant_ids, adversarial_ids, attack,
                                  server.global_model, ca=None, verify=verify)
        server.apply_update(generic_compose(server, updates, d1, d2, tau=5.0))
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# ---------------------------------------------------------------------------------------------
# The power rule and the screen, each at its own threshold, from its own home.
# ---------------------------------------------------------------------------------------------

def admitted(acc, asr):
    """The power rule: informative only where there is suppression for a composition to preserve.

    paper/supplementary.tex:247, recorded there as fixed before that freeze. SUPPRESS_ASR = 0.5 is
    imported from run_dose_resnet18.py:133 and ACC_FLOOR from run_targeted_dose.py. This is NOT the
    screen's C1 threshold, which is SUPPRESSION_THRESHOLD = 0.3 and answers a different question.
    """
    return bool(asr < SUPPRESS_ASR and acc >= ACC_FLOOR)


def cifar100_standalone_baselines():
    """The five CIFAR-100 standalone aggregators the artifact already has, as the screen wants them.

    Returns {defense: {committed_attack: mean_asr}} plus a parallel accuracy dict. The artifact keys
    are <internal_attack>_<defense> and its per-seed ASR key is "attack_success_rate" -- a THIRD
    naming convention in this repository, after dose's "asr" and the payoff matrix's own
    "attack_success_rate". Every read is subscripted rather than defaulted, so a renamed key raises
    instead of silently returning a suppressing zero.
    """
    d = json.load(open(CIFAR100_STANDALONE))
    asr, acc = {}, {}
    for att in COMMITTED_ATTACKS:
        internal = ATTACK_MAP[att]
        for k, rows in d.items():
            if not k.startswith(internal + "_"):
                continue
            defense = k[len(internal) + 1:]
            # Membership in the screen's OWN taxonomy, not `get_defense_type(...) is None`, which can
            # never be true: it returns the string "unknown" for an unknown name, so that test would
            # be dead code and krum/multi_krum -- measured on CIFAR-100 but absent from the 42-pair
            # menu -- would silently enter the screen's baseline table.
            if defense not in DEFENSE_TYPES:
                continue
            asr.setdefault(defense, {})[att] = float(np.mean([r["attack_success_rate"]
                                                              for r in rows]))
            acc.setdefault(defense, {})[att] = float(np.mean([r["accuracy"] for r in rows]))
    return asr, acc


def screen_pair(d1, d2, pure_asr):
    """The screen's prediction for one pair, with TWO missing-baseline failure modes handled here.

    Neither is a defect in the screen on the dataset it was built for, where all seven baselines
    exist. Both bite on CIFAR-100, where two do not, and analyze_composability.py is not editable and
    is not edited -- so both are handled at this call site.

      1. predict_composition returns UNKNOWN only when BOTH members' baselines are missing, and
         predict_pair then returns HIGH-if-any-HIGH else LOW -- so UNKNOWN on both attacks comes back
         LOW. On this dataset LOW is the scarce class the screen is scored on, so that would
         manufacture the very positives the suite exists to count. Measured: exactly the 2
         reciprocal pairs reputation<->foolsgold are in this state before the probe, and
         predict_pair labels both LOW.
      2. With ONE member's baseline missing the predictor does not signal anything. It sets that
         member's `suppresses` to False and carries on, so a missing baseline is silently read as
         NOT SUPPRESSING -- and since C1 needs only one suppressor, the pair comes back HIGH on
         evidence that does not exist. Measured: 20 of the 42 pairs are in this state before the
         probe, all labelled HIGH. This one produces no UNKNOWN anywhere and is invisible unless the
         baseline table is checked directly.

    So the rule here is stricter than the predictor's own: a pair is UNPREDICTABLE if EITHER member
    lacks a baseline for EITHER attack, whatever label came back. That is what makes gate 0's probe a
    precondition of scoring rather than only a feasibility test -- with the probe's two aggregators in
    the table, all 42 pairs become predictable on evidence.
    """
    overall, per_attack = predict_pair(d1, d2, pure_asr)
    unknown = [a for a, v in per_attack.items() if v["prediction"] == "UNKNOWN"]
    missing = sorted({f"{d}/{a}" for d in (d1, d2) for a in COMMITTED_ATTACKS
                      if pure_asr.get(d, {}).get(a) is None})
    return {"prediction": "UNPREDICTABLE" if missing else overall,
            "predict_pair_overall": overall, "unknown_attacks": unknown,
            "missing_baselines": missing,
            "per_attack": per_attack,
            "c1_threshold": SUPPRESSION_THRESHOLD,
            "note": ("" if not missing else
                     f"baseline missing for {', '.join(missing)}, so the imported predictor's label "
                     f"({overall}) is NOT used: it rests either on an UNKNOWN collapsed to LOW or on "
                     "an absent baseline read as non-suppression.")}


# ---------------------------------------------------------------------------------------------
# Gate 0: the 20-run probe.
# ---------------------------------------------------------------------------------------------

def load_probe():
    if not os.path.exists(probe_path):
        return {}
    try:
        return json.load(open(probe_path)).get("cells", {})
    except Exception:
        return {}


def save_probe(cells):
    verdict = probe_verdict(cells)
    json.dump({
        "description":
            "Gate 0 of experiments/pre_registration_cifar100_composition_suite.md: reputation and "
            "foolsgold standalone on CIFAR-100, both committed attacks, seeds 42-46, 20 runs. These "
            "two aggregators are ABSENT from results/cifar100/per_seed_results.json, they are d1 in "
            "12 of the 42 pairs and d2 in 12 more, and foolsgold_then_rfa is the lowest CIFAR-10 LOW "
            "pair at 0.0454 with rfa as d2, so the suppression there comes from foolsgold itself. "
            "The question whether any CIFAR-100 pair can be LOW turns on exactly these two.",
        "prereg_commit": PREREG_COMMIT, "prereg_md5": prereg_md5(),
        "prereg": "experiments/pre_registration_cifar100_composition_suite.md",
        "standalone_convention": f"d1={PROBE_D1} is a pass-through in apply_d1_transform, so "
                                 f"'{PROBE_D1} then X' IS X standalone. The probe and the suite "
                                 "share one code path rather than having two implementations of "
                                 "'standalone'.",
        "dataset": DATASET, "model": MODEL, "seeds": SEEDS, "attacks": list(ATTACKS),
        "decision_rule_frozen":
            "If EITHER aggregator is admitted by the power rule on EITHER attack -- mean ASR < "
            f"{SUPPRESS_ASR} at mean accuracy >= {ACC_FLOOR} -- a LOW composed pair is reachable on "
            "CIFAR-100 and the full 420-run suite runs. If neither is admitted, the suite does not "
            "run and the probe IS the finding: on CIFAR-100 at this architecture and round budget no "
            "defense in the menu suppresses either committed attack, the screen's LOW class is "
            "empty, and the screen cannot be evaluated on this dataset at all. Either way the 20 "
            "runs are reported with both aggregators' accuracy and ASR and the admitted count as a "
            "fraction. A null probe is not silently dropped.",
        "two_thresholds_not_interchangeable":
            f"The power rule's SUPPRESS_ASR is {SUPPRESS_ASR} and decides whether a cell can be "
            f"informative; the screen's C1 SUPPRESSION_THRESHOLD is {SUPPRESSION_THRESHOLD} and "
            "decides what the screen predicts. Both are imported from their own homes, both are "
            "reported below, and neither substitutes for the other.",
        "also_supplies_missing_baselines":
            "Beyond feasibility, these 20 runs are the two standalone baselines the screen needs on "
            "CIFAR-100. One or both is a member of 22 of the 42 pairs, and without them the "
            "screen's label on those 22 does not rest on evidence: the 2 reciprocal "
            "reputation<->foolsgold pairs come back LOW from a collapsed UNKNOWN, and 20 come back "
            "HIGH from an absent baseline read as non-suppression.",
        "verdict": verdict, "cells": cells}, open(probe_path, "w"), indent=2)


def probe_verdict(cells):
    rows = []
    for d2 in PROBE_D2:
        for att in ATTACKS:
            k = f"{PROBE_D1}_then_{d2}|{att}"
            per = cells.get(k, {}).get("per_seed", [])
            if not per:
                continue
            a = float(np.mean([r["asr"] for r in per]))
            c = float(np.mean([r["accuracy"] for r in per]))
            rows.append({"aggregator": d2, "attack": att, "n": len(per),
                         "mean_asr": a, "mean_accuracy": c,
                         "min_accuracy": float(min(r["accuracy"] for r in per)),
                         "sd_asr": float(np.std([r["asr"] for r in per], ddof=1))
                         if len(per) > 1 else None,
                         "admitted_by_power_rule": admitted(c, a),
                         "suppresses_at_screen_c1_threshold": bool(a < SUPPRESSION_THRESHOLD),
                         "accuracy_clears_floor": bool(c >= ACC_FLOOR)})
    n_adm = sum(1 for r in rows if r["admitted_by_power_rule"])
    complete = len(rows) == len(PROBE_D2) * len(ATTACKS)
    return {"n_cells": len(rows), "n_cells_expected": len(PROBE_D2) * len(ATTACKS),
            "complete": bool(complete),
            "n_admitted": n_adm, "admitted_fraction": f"{n_adm}/{len(rows)}" if rows else "0/0",
            "suppress_asr": SUPPRESS_ASR, "acc_floor": ACC_FLOOR,
            "screen_c1_threshold": SUPPRESSION_THRESHOLD,
            "go": bool(complete and n_adm > 0) if complete else None,
            "decision": (None if not complete else
                         ("GO: a LOW composed pair is reachable on CIFAR-100, so the 420-run suite "
                          f"runs ({n_adm} of {len(rows)} probe cells admitted)."
                          if n_adm > 0 else
                          "NO-GO: neither aggregator is admitted, so no composed pair in the menu "
                          "has a reachable LOW. The suite does NOT run and this probe is reported as "
                          "the external-validity finding the freeze specifies. It cost 20 runs "
                          "instead of 420.")),
            "cells": rows}


def probe(verbose=True):
    cells = load_probe()
    todo = [(d2, att, s) for d2 in PROBE_D2 for att in ATTACKS for s in SEEDS]
    print("=== GATE 0: the 20-run feasibility probe, which runs FIRST ===")
    print(f"    {PROBE_D1} (pass-through) then {{{', '.join(PROBE_D2)}}} on {DATASET}/{MODEL}, "
          f"{len(ATTACKS)} attacks x {len(SEEDS)} seeds = {len(todo)} runs")
    print(f"    Admitted iff mean ASR < {SUPPRESS_ASR} at mean accuracy >= {ACC_FLOOR}, both "
          "imported. The screen's own C1\n    threshold is a DIFFERENT number "
          f"({SUPPRESSION_THRESHOLD}) and is reported beside it, never instead of it.")
    print("    0 of the 14 measured CIFAR-100 standalone cells are admitted, which is why this gate")
    print("    exists; reputation and foolsgold are the two the artifact has never measured.\n",
          flush=True)
    done = 0
    for d2, att, seed in todo:
        k = f"{PROBE_D1}_then_{d2}|{att}"
        existing = {r["seed"]: r for r in cells.get(k, {}).get("per_seed", [])}
        done += 1
        if seed in existing:
            continue
        t = time.time()
        acc, asr = run_one(seed, PROBE_D1, d2, att)
        existing[seed] = {"seed": int(seed), "accuracy": acc, "asr": asr}
        cells[k] = {"d1": PROBE_D1, "d2": d2, "attack": att, "dataset": DATASET, "model": MODEL,
                    "is_standalone_d2": True,
                    "per_seed": [existing[s] for s in sorted(existing)]}
        os.makedirs(out_dir, exist_ok=True)
        save_probe(cells)
        if verbose:
            print(f"  [{done}/{len(todo)}] {d2:11s} {att:18s} s{seed}: acc={acc:.4f} asr={asr:.4f}"
                  + ("  * below acc floor" if acc < ACC_FLOOR else "")
                  + f"  ({time.time() - t:.0f}s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    save_probe(cells)
    v = probe_verdict(cells)
    print("\n  probe cells (progress counted from per_seed, never from the [n/20] index above):")
    for r in v["cells"]:
        print(f"    {r['aggregator']:11s} {r['attack']:18s} n={r['n']}  ASR={r['mean_asr']:.4f}  "
              f"acc={r['mean_accuracy']:.4f} (min {r['min_accuracy']:.4f})  -> "
              f"{'ADMITTED' if r['admitted_by_power_rule'] else 'not admitted'}"
              f"  [screen C1 at {SUPPRESSION_THRESHOLD}: "
              f"{'suppresses' if r['suppresses_at_screen_c1_threshold'] else 'does not suppress'}]")
    print(f"\n  admitted: {v['admitted_fraction']}")
    print(f"  {v['decision']}")
    print(f"\n  Written to {probe_path}")
    return v


# ---------------------------------------------------------------------------------------------
# The CIFAR-10 cross-check. A STRENGTHENING, and not one of the freeze's three gates.
# ---------------------------------------------------------------------------------------------

def published_cifar10_cell(d1, d2, attack, seed):
    for p in (WAVE1_PUBLISHED, WAVE2_PUBLISHED):
        if not os.path.exists(p):
            continue
        pairs = json.load(open(p)).get("pairs", {})
        cell = pairs.get(pair_key(d1, d2))
        if not cell or attack not in cell:
            continue
        for r in cell[attack]["per_seed"]:
            if r["seed"] == seed:
                return float(r["accuracy"]), float(r["asr"]), os.path.relpath(p, base)
    return None


def harness_check():
    """Does this arm's loop reproduce the published CIFAR-10 rows it was mirrored from?

    THIS IS NOT ONE OF THE FREEZE'S GATES. The freeze names three -- gate 0's probe, gate 1's
    adversary hook, gate 2's accuracy floor -- and this is none of them. It is a strengthening added
    while writing the runner, it is labelled one everywhere it is recorded, and a failure here is
    reported as a discrepancy in a check the freeze did not require rather than as a pre-registered
    gate firing. It is run because the loop is a mirror of code that is not editable, and the cheapest
    way to test a mirror is against the original's published output.

    Two runs on one wave-1 pair at seed 42: a real d1 transform and a real d2 weighting, on
    committed_scaling (which OVERRIDES manipulate_update) and committed_pixel (which does NOT), so the
    hoist into apply_adversary is tested on both sides of gate 1's per-attack expectation.

    ASR is compared EXACTLY. Accuracy is compared to four decimals, because that is all the published
    artifacts store -- verified across all 356 rows of both waves. This runner does not claim
    bit-identity on a rounded number.
    """
    d1, d2 = XCHECK_PAIR
    v = {"label": "STRENGTHENING, not one of the freeze's three gates",
         "freezes_gates": ["gate 0: the 20-run probe", "gate 1: adversary_hook.apply_adversary",
                           "gate 2: the ACC_FLOOR floor"],
         "pair": [d1, d2], "seed": XCHECK_SEED, "dataset": "cifar10", "model": "cifar_cnn",
         "accuracy_comparison": f"to {XCHECK_ACC_DECIMALS} decimals; the published artifacts store "
                                "accuracy rounded to 4 dp, so bit-identity on accuracy is not "
                                "available and is not claimed",
         "asr_comparison": "exact", "checks": [], "all_passed": None}
    print("=== CIFAR-10 CROSS-CHECK (a strengthening, NOT one of the freeze's three gates) ===")
    print(f"    {d1} -> {d2} at seed {XCHECK_SEED}, both attacks, against the published wave rows.")
    print(f"    ASR exactly; accuracy to {XCHECK_ACC_DECIMALS} dp, which is all those artifacts "
          "store.\n", flush=True)
    for att in ATTACKS:
        pub = published_cifar10_cell(d1, d2, att, XCHECK_SEED)
        if pub is None:
            print(f"  {att}: NO PUBLISHED ROW at seed {XCHECK_SEED}; cannot cross-check.")
            v["checks"].append({"attack": att, "passed": False,
                                "reason": "no published row for this pair/attack/seed"})
            continue
        p_acc, p_asr, src = pub
        overrides = defines_own_manipulate(get_attack(ATTACK_MAP[att]))
        t = time.time()
        acc, asr = run_one(XCHECK_SEED, d1, d2, att, dataset="cifar10", model="cifar_cnn")
        asr_exact = bool(asr == p_asr)
        acc_ok = bool(round(acc, XCHECK_ACC_DECIMALS) == round(p_acc, XCHECK_ACC_DECIMALS))
        ok = bool(asr_exact and acc_ok)
        v["checks"].append({"attack": att, "passed": ok, "source": src,
                            "attack_overrides_manipulate_update": bool(overrides),
                            "published": [p_acc, p_asr], "recomputed": [acc, asr],
                            "asr_exact": asr_exact,
                            "asr_delta": float(asr - p_asr),
                            "accuracy_equal_to_4dp": acc_ok,
                            "accuracy_delta": float(acc - p_acc)})
        print(f"  {att:18s} (overrides manipulate_update: {overrides})")
        print(f"    published  acc={p_acc:.4f}        asr={p_asr:.16f}   [{src}]")
        print(f"    recomputed acc={acc:.16f}  asr={asr:.16f}")
        print(f"    -> acc equal to {XCHECK_ACC_DECIMALS} dp: {acc_ok} (d={acc - p_acc:+.2e})   "
              f"asr exact: {asr_exact} (d={asr - p_asr:+.2e})   ({time.time() - t:.0f}s)",
              flush=True)
    v["all_passed"] = all(c["passed"] for c in v["checks"]) and len(v["checks"]) == len(ATTACKS)
    print("\n  " + ("The mirrored loop reproduces both published rows, so the hoist into "
                    "apply_adversary is\n  numerically neutral on both attacks as claimed."
                    if v["all_passed"] else
                    "DISCREPANCY against the published rows. This is a strengthening rather than a "
                    "frozen\n  gate, so it is reported as a discrepancy and not as a gate firing -- "
                    "and the suite does not\n  proceed on a loop that cannot reproduce the code it "
                    "mirrors."))
    return v


# ---------------------------------------------------------------------------------------------
# Scoring: the screen's agreement, with every denominator carrying its exclusion count.
# ---------------------------------------------------------------------------------------------

def load():
    if not os.path.exists(out_path):
        return {}, None
    try:
        d = json.load(open(out_path))
        return d.get("pairs", {}), d.get("cross_check")
    except Exception:
        return {}, None


def cell_stats(cell, att):
    per = cell.get(att, {}).get("per_seed", [])
    if not per:
        return None
    return {"n": len(per),
            "mean_asr": float(np.mean([r["asr"] for r in per])),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in per])),
            "min_accuracy": float(min(r["accuracy"] for r in per)),
            "sd_asr": float(np.std([r["asr"] for r in per], ddof=1)) if len(per) > 1 else None}


def score(pairs, probe_cells):
    """The screen's CIFAR-100 confusion matrix, or the reason there cannot be one.

    Denominators, all three of them stated rather than one of them chosen:
      42                      -- the menu
      n_complete              -- pairs with all seeds on both attacks
      n_scored                -- complete AND clearing the floor AND predictable

    Observed LOW uses the freeze's < 0.5 on max_committed_asr, which is SUPPRESS_ASR from its own
    home. Predicted LOW comes from the imported screen at its own C1 threshold of 0.3. The base rate
    and the constant-HIGH baseline are computed over n_scored and are reported in the same breath as
    any accuracy figure, which is this paper's standing rule for the ~90% figure.
    """
    pure_asr, _ = cifar100_standalone_baselines()
    # The probe's two aggregators join the five the artifact already had. Without them 22 of the 42
    # pairs are UNPREDICTABLE by screen_pair's rule; with them, none are. Verified by construction:
    # every menu defense appears in the table below before any pair is scored.
    probe_v = probe_verdict(probe_cells)
    for r in probe_v["cells"]:
        pure_asr.setdefault(r["aggregator"], {})[r["attack"]] = r["mean_asr"]
    rows, incomplete = [], 0
    for (d1, d2) in PAIRS:
        cell = pairs.get(pair_key(d1, d2))
        if not cell:
            incomplete += 1
            continue
        stats = {att: cell_stats(cell, att) for att in ATTACKS}
        if any(s is None or s["n"] < len(SEEDS) for s in stats.values()):
            incomplete += 1
            continue
        max_asr = max(s["mean_asr"] for s in stats.values())
        min_acc = min(s["mean_accuracy"] for s in stats.values())
        pred = screen_pair(d1, d2, pure_asr)
        rows.append({"d1": d1, "d2": d2, "pair": pair_key(d1, d2), "wave": WAVE_OF[(d1, d2)],
                     "max_committed_asr": max_asr,
                     "min_mean_accuracy_over_attacks": min_acc,
                     "clears_floor": bool(min_acc >= ACC_FLOOR),
                     "observed": "LOW" if max_asr < SUPPRESS_ASR else "HIGH",
                     "observed_threshold": SUPPRESS_ASR,
                     "predicted": pred["prediction"], "screen": pred,
                     "per_attack": stats})
    floored = [r for r in rows if not r["clears_floor"]]
    unpred = [r for r in rows if r["clears_floor"] and r["predicted"] == "UNPREDICTABLE"]
    scored = [r for r in rows if r["clears_floor"] and r["predicted"] in ("LOW", "HIGH")]
    n_low = sum(1 for r in scored if r["observed"] == "LOW")
    n_high = len(scored) - n_low
    agree = sum(1 for r in scored if r["observed"] == r["predicted"])
    tp = sum(1 for r in scored if r["predicted"] == "LOW" and r["observed"] == "LOW")
    fp = sum(1 for r in scored if r["predicted"] == "LOW" and r["observed"] == "HIGH")
    fn = sum(1 for r in scored if r["predicted"] == "HIGH" and r["observed"] == "LOW")
    degenerate = (n_low == 0)
    return {
        "n_menu": len(PAIRS), "n_incomplete": incomplete, "n_complete": len(rows),
        "n_below_floor": len(floored), "n_unpredictable": len(unpred), "n_scored": len(scored),
        "acc_floor": ACC_FLOOR, "observed_low_threshold": SUPPRESS_ASR,
        "screen_c1_threshold": SUPPRESSION_THRESHOLD,
        "below_floor_pairs": [{"pair": r["pair"], "min_mean_accuracy": r[
            "min_mean_accuracy_over_attacks"]} for r in floored],
        "unpredictable_pairs": [{"pair": r["pair"],
                                 "missing_baselines": r["screen"]["missing_baselines"],
                                 "unused_predictor_label": r["screen"]["predict_pair_overall"]}
                                for r in unpred],
        "n_observed_low": n_low, "n_observed_high": n_high,
        "base_rate_high": (n_high / len(scored)) if scored else None,
        "constant_high_baseline": (f"{n_high}/{len(scored)}" if scored else None),
        "agreement": (f"{agree}/{len(scored)}" if scored else None),
        "agreement_fraction": (agree / len(scored)) if scored else None,
        "precision_low": (tp / (tp + fp)) if (tp + fp) else None,
        "recall_low": (tp / (tp + fn)) if (tp + fn) else None,
        "low_class_empty": bool(degenerate),
        "reporting_rule":
            "Any accuracy figure from this arm is reported in the same sentence as the base rate and "
            "the constant-HIGH baseline, in every position including captions and section titles, "
            "and any fraction names its exclusion counts: "
            f"{len(floored)} below the floor and {len(unpred)} unpredictable of {len(rows)} "
            f"complete, out of a {len(PAIRS)}-pair menu.",
        "degeneracy_clause":
            (None if not degenerate else
             "The LOW class is EMPTY among the scored cells, so recall is undefined, the "
             "constant-HIGH predictor scores every cell, and the screen CANNOT BE EVALUATED on this "
             "dataset. This is not a replication of the screen and may not be reported as answering "
             "the review's ask. It is reported as the external-validity finding the freeze "
             "specifies."),
        "rows": rows}


def save(pairs, xcheck, probe_cells):
    s = score(pairs, probe_cells)
    json.dump({
        "description":
            f"The full 42-pair composition menu on {DATASET}/{MODEL}: the union of "
            "run_all_compositions.PAIRS (18, development) and run_wave2_held_out.PAIRS (24, held "
            "out), both IMPORTED and not re-listed, against both committed attacks at seeds "
            f"{SEEDS[0]}-{SEEDS[-1]}. Wave 1 runs first because all 5 CIFAR-10 LOW pairs are in "
            "wave 1.",
        "prereg_commit": PREREG_COMMIT, "prereg_md5": prereg_md5(),
        "prereg": "experiments/pre_registration_cifar100_composition_suite.md",
        "run_scope": scope_field(),
        "dataset": DATASET, "model": MODEL,
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": 0.5, "rounds": FL_CONFIG.num_rounds,
                   "local_epochs": FL_CONFIG.local_epochs, "lr": FL_CONFIG.learning_rate,
                   "lr_decay": getattr(FL_CONFIG, "lr_decay", 1.0),
                   "seeds": SEEDS, "attacks": list(ATTACKS),
                   "n_pairs": len(PAIRS), "n_runs_planned": len(PAIRS) * len(ATTACKS) * len(SEEDS),
                   "acc_floor": ACC_FLOOR, "suppress_asr": SUPPRESS_ASR,
                   "screen_c1_threshold": SUPPRESSION_THRESHOLD},
        "loop_provenance":
            "run_one here MIRRORS run_all_compositions.run_one with exactly two differences: the "
            "dataset is a parameter (that file hardcodes \"cifar10\" at :602 and is not editable), "
            "and the adversary step is hoisted out of the participant loop into "
            "adversary_hook.apply_adversary(..., verify=True), the freeze's gate 1. The hoist is "
            "numerically neutral for both committed attacks -- backdoor_pixel inherits the identity "
            "manipulate_update and model_scaling is an elementwise multiply, so neither consumes RNG "
            "nor reads another client's update -- and the CIFAR-10 cross-check tests that against "
            "published rows rather than resting on the argument.",
        "gates": {
            "gate_0_probe": "results/cifar100_composition_suite/probe.json. 20 runs, decided by the "
                            "frozen power rule BEFORE the suite starts.",
            "gate_1_adversary_hook": "adversary_hook.apply_adversary(..., verify=True) on every "
                                     "round of every run, including the probe and the cross-check. "
                                     "It pins the expected behaviour PER ATTACK -- the update must "
                                     "change iff the attack overrides manipulate_update -- so it "
                                     "catches both a dead hook and a mis-typed attack object. Never "
                                     "caught and downgraded to a warning.",
            "gate_2_accuracy_floor": f"ACC_FLOOR = {ACC_FLOOR}, imported. Cells below it are "
                                     "reported uninterpretable and COUNTED with their accuracy "
                                     "printed; they are not dropped and not scored HIGH because a "
                                     "collapsed model has a high ASR."},
        "cross_check": xcheck,
        "two_thresholds_not_interchangeable":
            f"SUPPRESS_ASR = {SUPPRESS_ASR} (run_dose_resnet18.py:133) is the power rule and decides "
            f"whether a cell can be informative and what counts as an observed LOW. "
            f"SUPPRESSION_THRESHOLD = {SUPPRESSION_THRESHOLD} (analyze_composability.py) is the "
            "screen's own C1 and decides what the screen PREDICTS. Two numbers, two jobs, both "
            "imported, neither substituted for the other.",
        "imported_predictor_failure_modes":
            "Two, neither a defect on the dataset the screen was built for. (1) predict_composition "
            "returns UNKNOWN only when BOTH members' baselines are missing, and predict_pair returns "
            "HIGH-if-any-HIGH else LOW, so UNKNOWN on both attacks comes back LOW -- measured before "
            "the probe: exactly the 2 reciprocal pairs reputation<->foolsgold, both LOW, which is the "
            "scarce class the screen is scored on. (2) "
            "With ONE member missing nothing is signalled: that member's `suppresses` is False and "
            "the pair comes back HIGH on a baseline that does not exist, since C1 needs only one "
            "suppressor -- measured before the probe: 20 of 42 pairs, all HIGH, with no UNKNOWN "
            "anywhere. analyze_composability.py is not edited; screen_pair applies the stricter rule "
            "that a pair is UNPREDICTABLE if EITHER member lacks a baseline for EITHER attack, "
            "whatever label came back, and such pairs are excluded with their count stated.",
        "screen_baselines":
            "The screen is a function of STANDALONE per-dataset ASR, so its CIFAR-100 predictions "
            "need seven aggregators and results/cifar100/per_seed_results.json has five. reputation "
            "and foolsgold come from gate 0's probe, which is therefore a baseline fix as well as a "
            "feasibility test. Its per-seed ASR key there is \"attack_success_rate\", a third naming "
            "convention after dose's \"asr\" and the payoff matrix's own; every read is subscripted, "
            "never defaulted.",
        "scope": "One dataset, one architecture, one round budget, 42 pairs, n=5. No sentence "
                 "produced from this arm generalizes past that.",
        "screen": s, "pairs": pairs}, open(out_path, "w"), indent=2)


def report(pairs, probe_cells):
    s = score(pairs, probe_cells)
    print("\n=== COVERAGE, counted from per_seed rows in the artifact ===")
    sc = scope_field()
    if sc:
        print(f"  {sc}")
    print(f"  menu {s['n_menu']}  complete {s['n_complete']}  incomplete/unrun {s['n_incomplete']}")
    print(f"  of the complete: below the ACC_FLOOR of {ACC_FLOOR}: {s['n_below_floor']}   "
          f"unpredictable (missing baseline): {s['n_unpredictable']}   scored: {s['n_scored']}")
    if s["n_incomplete"]:
        w1 = sum(1 for r in s["rows"] if r["wave"] == 1)
        print(f"  INCOMPLETE: {s['n_incomplete']} of {s['n_menu']} pairs are unrun "
              f"({w1} of {len(PAIRS_WAVE1)} wave-1 pairs complete). This is an incompleteness, not a "
              "suite, and\n  a partial suite is never described as the full 42-pair menu.")
    if s["n_below_floor"]:
        print("\n=== BELOW THE FLOOR: uninterpretable and COUNTED, not dropped ===")
        for r in s["below_floor_pairs"]:
            print(f"    {r['pair']:34s} min mean accuracy {r['min_mean_accuracy']:.4f}")
    if s["n_unpredictable"]:
        print("\n=== UNPREDICTABLE: the screen has no standalone baseline for a member ===")
        for r in s["unpredictable_pairs"]:
            print(f"    {r['pair']:34s} missing {', '.join(r['missing_baselines'])}"
                  f"   (unused predictor label: {r['unused_predictor_label']})")
    if not s["n_scored"]:
        print("\n  Nothing scored yet.")
        return
    print(f"\n=== THE SCREEN ON {DATASET.upper()} ===")
    print(f"  observed LOW is max_committed_asr < {s['observed_low_threshold']}; the screen's own C1 "
          f"threshold is {s['screen_c1_threshold']}, a different number")
    print(f"  observed: {s['n_observed_low']} LOW, {s['n_observed_high']} HIGH of {s['n_scored']} "
          "scored")
    print(f"  agreement {s['agreement']} = "
          f"{s['agreement_fraction'] * 100:.1f}%, against a constant-HIGH baseline of "
          f"{s['constant_high_baseline']} = {s['base_rate_high'] * 100:.1f}% "
          f"({s['n_below_floor']} of {s['n_complete']} complete pairs excluded below the floor, "
          f"{s['n_unpredictable']} unpredictable)")
    print(f"  precision on LOW: {s['precision_low']}   recall on LOW: {s['recall_low']}")
    if s["low_class_empty"]:
        print("\n  " + s["degeneracy_clause"])


def main():
    check_frozen()
    pairs, xcheck = load()
    todo = run_pairs()
    print("=== ARM D: the 42-pair composition menu on CIFAR-100 ===")
    print(f"    {DATASET}/{MODEL}, menu {len(PAIRS)} pairs (18 wave-1 + 24 wave-2, imported, union "
          f"verified = 42) x {len(ATTACKS)} attacks x {len(SEEDS)} seeds = "
          f"{len(PAIRS) * len(ATTACKS) * len(SEEDS)} runs")
    print(f"    rules frozen at {PREREG_COMMIT}, md5 {prereg_md5()}")
    print("    Wave 1 first: all 5 CIFAR-10 LOW pairs are in wave 1, so that is where the LOW class")
    print("    can exist at all and where the replication question is decidable.")
    note = scope_note()
    if note is None:
        print("    ATTEMPTING THE WHOLE MENU: all 42 pairs.\n", flush=True)
    else:
        print(f"\n    {note}\n", flush=True)

    if xcheck is None or not xcheck.get("all_passed"):
        sys.exit("REFUSING TO RUN: no passing CIFAR-10 cross-check is recorded in "
                 f"{out_path}.\n  Run: PYTHONPATH=. python3 -m "
                 "experiments.run_cifar100_composition_suite --harness-check\n  This loop is a "
                 "mirror of code that is not editable, and the cheapest test of a mirror is against "
                 "the\n  original's published output. (The cross-check is a STRENGTHENING, not one "
                 "of the freeze's\n  three gates -- but a loop that cannot reproduce what it mirrors "
                 "does not get 420 runs.)")

    # Gate 0 first, and its frozen decision binds.
    pv = probe_verdict(load_probe())
    if not pv["complete"]:
        print("  Gate 0 is incomplete; running the probe before anything else.\n", flush=True)
        pv = probe()
    if not pv["go"]:
        os.makedirs(out_dir, exist_ok=True)
        save(pairs, xcheck, load_probe())
        print(f"\n{pv['decision']}")
        print("\nThe frozen decision rule stops the arm here. The probe's 20 runs ARE the reported "
              "finding,\nnot a failed attempt: on CIFAR-100 at this architecture and round budget no "
              "defense in the menu\nsuppresses either committed attack, so the screen's LOW class is "
              "empty and the screen cannot be\nevaluated on this dataset. The abstract's CIFAR-100 "
              "clause is then scoped down to the ladder it\nactually rests on, exactly as the "
              "no-reproduction branch requires. No threshold is adjusted to\nturn this into a go.")
        print(f"\nWritten to {out_path} and {probe_path}")
        return 0
    print(f"  Gate 0: {pv['decision']}\n", flush=True)

    probe_cells = load_probe()
    total = len(todo) * len(ATTACKS) * len(SEEDS)
    if pairs:
        have = sum(len(c[a]["per_seed"]) for c in pairs.values() for a in ATTACKS if a in c)
        print(f"  resuming: {have} rows already present, counted from per_seed and never from the "
              f"[n/{total}] index below\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); idx = 0
    for (d1, d2) in todo:
        k = pair_key(d1, d2)
        cell = pairs.get(k, {"d1": d1, "d2": d2, "wave": WAVE_OF[(d1, d2)],
                             "dataset": DATASET, "model": MODEL})
        for att in ATTACKS:
            existing = {r["seed"]: r for r in cell.get(att, {}).get("per_seed", [])}
            for seed in SEEDS:
                idx += 1
                if seed in existing:
                    continue
                t = time.time()
                acc, asr = run_one(seed, d1, d2, att)
                existing[seed] = {"seed": int(seed), "accuracy": acc, "asr": asr}
                cell[att] = {"per_seed": [existing[s] for s in sorted(existing)]}
                cell[att]["mean_asr"] = float(np.mean([r["asr"] for r in cell[att]["per_seed"]]))
                pairs[k] = cell
                save(pairs, xcheck, probe_cells)
                print(f"  [{idx}/{total}] w{WAVE_OF[(d1, d2)]} {k:34s} {att:18s} s{seed}: "
                      f"acc={acc:.4f} asr={asr:.4f}"
                      + ("  * below acc floor" if acc < ACC_FLOOR else "")
                      + f"  ({time.time() - t:.0f}s)", flush=True)
        pairs[k] = cell
        cell["max_committed_asr"] = max(
            (cell[a]["mean_asr"] for a in ATTACKS if a in cell and "mean_asr" in cell[a]),
            default=None)
        save(pairs, xcheck, probe_cells)
    save(pairs, xcheck, probe_cells)
    report(pairs, probe_cells)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    # Parsed before any branch so the cap is in force wherever save() runs, and so a capped invocation
    # cannot be started for the suite and then have its disclosure overwritten by an uncapped --report.
    if "--wave1-only" in sys.argv:
        RUN_WAVES = (1,)
    if "--harness-check" in sys.argv:
        check_frozen()
        pairs, _ = load()
        v = harness_check()
        os.makedirs(out_dir, exist_ok=True)
        save(pairs, v, load_probe())
        print(f"\nCross-check verdict written to {out_path}")
        sys.exit(0 if v["all_passed"] else 1)
    if "--probe" in sys.argv:
        check_frozen()
        v = probe()
        sys.exit(0)
    if "--report" in sys.argv:
        check_frozen()
        pairs, _ = load()
        report(pairs, load_probe())
        sys.exit(0)
    sys.exit(main())
