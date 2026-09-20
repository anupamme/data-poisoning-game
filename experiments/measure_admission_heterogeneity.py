"""
Stage 1 of the heterogeneity arm: the Mode-S PREMISE, measured at each alpha before any ASR exists.

WHY THIS RUNS FIRST, AND SEPARATELY. run_heterogeneity_modeS.py will spend 6--12 h scoring
Delta_S(alpha) = ASR(kappa=2, alpha) - ASR(kappa=0, alpha) on coord_median x committed pixel. That
number is a Mode-S reading only if two things hold AT EACH ALPHA, and neither has ever been checked
off alpha=0.5. Each is a way the arm can fail, each is a gate frozen in
experiments/pre_registration_heterogeneity_modeS.md (sections 4.3 and 4.4) before this file existed,
and each is checked here, in ~1 h, rather than discovered afterwards:

  1. THE IDENTITY RUNG DISPLACES NOTHING, EXACTLY (prereg 4.3). At kappa=0 every per-client
     coefficient must be 1.0 and every aggregate displacement 0.0, tested with `==` and not a
     tolerance. This is the guard against the failure mode this repository has already hit, where a
     manipulation hook was never called and the run passed silently looking like a null.
  2. THE ADVERSARY MUST STAY PINNED (prereg 4.4). Mode S's entire claim is that c_adv == 1.0 at every
     rung, so that ASR movement is attributable to the benign reweighting the downstream statistic
     reads. That is a property of the PARTITION as well as of the transform: the share is
     sum(c_adv)/sum(c), and how many adversaries land in a round of K=5 is a sampling fact. If the
     share moves at some alpha, that alpha's leg measures the attenuation channel as well as the
     statistic channel and it CARRIES NO MODE-S READING.

WHAT IS NOT A GATE HERE. Baseline adversarial admission is REPORTED per alpha, beside the frozen
CIFAR-10 alpha=0.5 level, as a diagnostic. It is not scored against a floor, because this arm's
pre-registration froze no admission floor and inventing one after the fact is exactly what a freeze
exists to prevent. The floor thresholds in pre_registration_normclip_cifar100.md belong to that arm.

NOTHING ABOUT THE CHANNELS IS REIMPLEMENTED. flatten, krum_selection, rep_weights, argmedian,
admission, aggregates, rel_disp, DECISION_KEY, ADMISSION_KEY, SHARE_TOL, ATTACK_MAP and N/K/F_ADV are
all imported from measure_admission.py, so CIFAR-10 at alpha=0.5 and these three new alphas cannot end
up measured by two different definitions of "admission" -- which is the whole point of the comparison
this script prints. measure_admission.py is NOT edited.

WHY THIS FILE EXISTS AT ALL, GIVEN THAT measure() TAKES A `rungs` ARGUMENT. It does, and it takes
dataset, model, seeds and rounds too -- but its Dirichlet concentration is the literal 0.5 at
measure_admission.py:126 (`get_federated_dataset(dataset, N, 0.5, seed)`) and there is no alpha
parameter to pass. So the loop is reproduced here with alpha varied and everything else held, and
--harness-check PROVES the reproduction BY VALUE against the frozen artifact at alpha=0.5 before any
new alpha is measured. If that check fails, this loop is not measure()'s loop and none of its numbers
mean what they claim.

ONE LOCAL EPOCH, WHICH IS CORRECT HERE AND WOULD NOT BE FOR A NORM. measure_admission.py:132 trains
one local epoch while every ASR ladder trains FLConfig.local_epochs = 2. Round 72's Amendment 1 was
about that mismatch, and it mattered there because norm_clip's dose is a threshold WITH UNITS. Mode S's
dial is a scale-free dispersion ratio and its coefficients are built from a mean-1 ladder over the
benign participants, never read from the update norms, so both gates above are epoch-independent: c_adv
is exactly 1.0 by construction and the share depends on the ladder and on how many adversaries the
round sampled. The one-epoch pass is also the configuration in which every existing admission number in
the paper was measured, so the harness check has something to compare to.

Output: results/heterogeneity_modeS_admission.json    (~1 h; no ASR, no 50-round run)
Run:    PYTHONPATH=. python3 -m experiments.measure_admission_heterogeneity --harness-check
        PYTHONPATH=. python3 -m experiments.measure_admission_heterogeneity
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base)

from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient  # noqa: E402
from attacks import get_attack                                                          # noqa: E402
from experiments.run_all_compositions import apply_d1_transform                         # noqa: E402
from experiments.measure_admission import (                                             # noqa: E402
    flatten, krum_selection, rep_weights, argmedian,
    admission, aggregates, rel_disp,
    DECISION_KEY, ADMISSION_KEY, SHARE_TOL, ATTACK_MAP,
    N, K, F_ADV,
)

OUT = os.path.join(base, "results", "heterogeneity_modeS_admission.json")
CIFAR10_FROZEN = os.path.join(base, "results", "admission_measurement.json")

PREREG = "experiments/pre_registration_heterogeneity_modeS.md"
PREREG_COMMIT = "e7b7c83"   # commits PREREG alone, 195 insertions, no other file. Before this hash
                            # existed the script refused to start: an unfrozen run would make the two
                            # gates above renegotiable.

DATASET, MODEL = "cifar10", "cifar_cnn"
D2, ATTACK = "coord_median", "committed_pixel"
ALPHAS = [0.1, 1.0, 10.0]
ANCHOR_ALPHA = 0.5          # the published Mode-S configuration; measured here ONLY as the harness
                            # check's reference, never as a new leg
SEEDS3 = [42, 43, 44]       # the seed grid of results/heterogeneity_sweep/, so each row is comparable
                            # to an existing criterion row at the same alpha
ROUNDS = 6
HARNESS_ROUNDS = 3          # results/admission_measurement.json's own ROUNDS; rounds 0--2 of a longer
                            # traversal are the same computation, the stream being sequential
LR, BATCH, EPOCHS = 0.01, 64, 1     # measure_admission.py:132's own literals

# The two rungs the ASR ladder uses. Endpoints only, and never to be described as a four-rung ladder.
RUNGS = [("doseS", 0.0, "doseS_kappa0.0"), ("doseS", 2.0, "doseS_kappa2.0")]

# Read back from the transformed norms, so a float32 readback tolerance is needed for the SHARE
# comparison. The EXACTNESS assertions below use `==` and no tolerance, as prereg 4.3 requires.
READBACK_TOL = 1e-6

# The frozen verdict literals. Constants, so this script REPORTS one rather than composing prose after
# seeing the numbers, and carrying no mechanism clause: Round 72's freeze fused a pass/fail label with
# a mechanism guess and the guess was backwards in sign while the label was right.
V_IDENTITY_FAILED = ("IDENTITY RUNG IS NOT THE EXACT IDENTITY AT THIS ALPHA: NO LEG AT THIS ALPHA "
                     "MEASURES WHAT IT CLAIMS TO.")
V_CONFOUNDED = ("CONFOUNDED AT THIS ALPHA: THE ADVERSARIAL COEFFICIENT SHARE MOVES ACROSS RUNGS, SO "
                "THIS ALPHA'S LEG MEASURES THE ATTENUATION CHANNEL AS WELL AS THE STATISTIC CHANNEL "
                "AND CARRIES NO MODE-S READING.")
V_PREMISE_HOLDS = ("MODE-S PREMISE HOLDS AT THIS ALPHA: the identity rung displaces nothing exactly "
                   "and the adversarial coefficient share is constant across rungs to SHARE_TOL.")
V_INDETERMINATE = ("PREMISE NOT ESTABLISHED AT THIS ALPHA: too few non-degenerate rounds to compare "
                   "the adversarial coefficient share across rungs. This is absence of evidence, NOT "
                   "evidence that the share moved, and it blocks the arm at this alpha either way.")


def jsonable(o):
    """Replace every non-finite float with None, recursively, at the write boundary.

    json.dump emits float("nan") as the bare token `NaN`, which Python's own json.loads accepts and
    every strict reader in another language rejects -- for the WHOLE file, not the offending field.
    A round with no adversary has no adversarial share, so NaN is reachable here.
    """
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def check_frozen():
    """Refuse to write results/ unless PREREG is committed AT PREREG_COMMIT and its tree is clean.

    The `is None` test alone would make PREREG_COMMIT a LABEL rather than a check. This script stamps
    the constant into the artifact, so once it is set, a string literal here and a different document
    on disk would agree with each other and disagree with the freeze. Two further checks are needed:

      - the hash, because the constant could name a commit that never touched this file;
      - the working tree, because `git log -1` reports the last commit that touched the file and is
        wholly unchanged by uncommitted edits to it. That is the defect that let a Round-57 amendment
        pass its own gate.
    """
    p = os.path.join(base, PREREG)
    if not os.path.exists(p):
        sys.exit(f"REFUSING TO RUN: {p} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the gates are not frozen.\n"
                 f"  1. git commit {PREREG}\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen run makes prereg 4.3 and 4.4 renegotiable after the numbers exist.")
    log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=base, capture_output=True, text=True, timeout=20)
    actual = log.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        sys.exit(f"REFUSING TO RUN: {PREREG} was last touched at {actual or 'UNTRACKED'}, but "
                 f"PREREG_COMMIT is {PREREG_COMMIT}. The gates must be frozen before they are "
                 "measured against.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=base, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        sys.exit(f"REFUSING TO RUN: {PREREG} has uncommitted changes "
                 f"({dirty.stdout.strip().split()[0]}), so it is NOT frozen at {actual} whatever "
                 "`git log` says. Commit it and set PREREG_COMMIT to the new hash.")
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")


# --------------------------------------------------------------------------- the traversal
# measure_admission.py:106-193's loop, with ONE literal changed: the Dirichlet concentration passed to
# get_federated_dataset. Every statistic, mirror, displacement and field name is imported. Kept in the
# same order and with the same calls so that --harness-check can hold it to the frozen artifact by
# value; a reordering that consumed the global RNG differently would show up there as a mismatch.

def measure_alpha(alpha, seeds, rounds):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        cd, _, nc = get_federated_dataset(DATASET, N, alpha, seed)
        srv = FederatedServer(get_model(MODEL, nc), dev)
        atk = get_attack(ATTACK_MAP[ATTACK])
        adv = set(range(int(N * F_ADV)))
        cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
              for i in range(N)]
        for rnd in range(rounds):
            pids = np.random.choice(N, K, replace=False)
            ups = []
            for cid in pids:
                u = cl[cid].train(srv.global_model, EPOCHS, LR, BATCH)
                if cid in adv:
                    u = atk.manipulate_update(u, srv.global_model)
                ups.append(u)
            raw = flatten(ups)
            adv_rows = [i for i, cid in enumerate(pids) if cid in adv]
            adv_mask = [bool(cid in adv) for cid in pids]

            sel_krum_0, _ = krum_selection(raw, False)
            sel_cos_0, _ = krum_selection(raw, True)
            w_rep_0 = rep_weights(raw, False)
            am_0 = argmedian(raw)
            adm_0 = admission(raw, adv_rows, sel_krum_0, sel_cos_0, w_rep_0, am_0)
            agg_0 = aggregates(raw, sel_krum_0, sel_cos_0, w_rep_0)

            for family, val, d1 in RUNGS:
                t = apply_d1_transform(ups, d1, tau=5.0, dose_key=(seed, rnd), adv_mask=adv_mask)
                st = flatten(t)
                c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
                sel_krum, _ = krum_selection(st, False)
                sel_cos, _ = krum_selection(st, True)
                w_rep = rep_weights(st, False)
                am = argmedian(st)
                adm = admission(st, adv_rows, sel_krum, sel_cos, w_rep, am)
                agg = aggregates(st, sel_krum, sel_cos, w_rep)
                c_adv = [float(c[a].item()) for a in adv_rows]
                disp = {a: rel_disp(agg[a], agg_0[a]) for a in agg}
                row = {
                    "alpha": float(alpha),
                    "family": family, "rung": float(val), "d1": d1, "attack": ATTACK,
                    "seed": int(seed), "round": int(rnd),
                    "n_adv_in_round": len(adv_rows), "n_benign_in_round": K - len(adv_rows),
                    "degenerate": bool(len(adv_rows) == 0 or K - len(adv_rows) < 2),
                    "rho_realized": float((c.max() / c.min().clamp(min=1e-12)).item()),
                    "c_mean": float(c.mean().item()),
                    "c_min": float(c.min().item()), "c_max": float(c.max().item()),
                    "c_adv_min": min(c_adv) if c_adv else float("nan"),
                    "c_adv_max": max(c_adv) if c_adv else float("nan"),
                    "c_adv_max_dev_from_one": (max(abs(x - 1.0) for x in c_adv) if c_adv
                                               else float("nan")),
                    "adv_coeff_share": (float(sum(c_adv) / float(c.sum().item()))
                                        if c_adv else float("nan")),
                    "krum_selection_changed": bool(sel_krum != sel_krum_0),
                    "cos_krum_selection_changed": bool(sel_cos != sel_cos_0),
                    "reputation_order_changed": bool(
                        not torch.equal(torch.argsort(w_rep_0), torch.argsort(w_rep))),
                    "coord_median_frac_argmedian_changed": float((am != am_0).float().mean().item()),
                    "krum_admission_changed": bool(
                        adm["krum_admits_adv"] != adm_0["krum_admits_adv"]),
                    "cos_krum_admission_changed": bool(
                        adm["cos_krum_admits_adv"] != adm_0["cos_krum_admits_adv"]),
                    "reputation_adv_share_delta": (adm["reputation_adv_weight_share"]
                                                   - adm_0["reputation_adv_weight_share"]),
                    "coord_median_adv_frac_delta": (adm["coord_median_adv_argmedian_frac"]
                                                    - adm_0["coord_median_adv_argmedian_frac"]),
                }
                row.update({f"base_{k}": v for k, v in adm_0.items()})
                row.update({f"post_{k}": v for k, v in adm.items()})
                row.update({f"agg_disp_{a}": v for a, v in disp.items()})
                rows.append(row)
            srv.apply_update(srv.aggregate(ups))
        print(f"    alpha={alpha:g} seed={seed}: {rounds} round(s), "
              f"{rounds * len(RUNGS)} row(s)")
    return rows


# --------------------------------------------------------------------------- the two frozen gates

def identity_gate(rows):
    """Prereg 4.3. At kappa=0, every c exactly 1.0 and every aggregate displacement exactly 0.0.

    Tested with `==`. A tolerance here would pass the failure mode the gate exists to catch: a
    manipulation hook that is never called, whose run then reports an unchanged number as a finding.
    """
    ident = [r for r in rows if r["rung"] == 0.0]
    bad_c = [r for r in ident if not (r["c_min"] == 1.0 and r["c_max"] == 1.0)]
    disp_keys = [k for k in ident[0] if k.startswith("agg_disp_")] if ident else []
    bad_d = [r for r in ident if any(r[k] != 0.0 for k in disp_keys)]
    # An empty identity-rung set must NOT read PASS. A gate that certifies exactness by finding no
    # counterexample among zero rows is the same defect as a grep whose glob matched nothing.
    if not ident or not disp_keys:
        verdict = "INDETERMINATE"
    else:
        verdict = "PASS" if not (bad_c or bad_d) else "FAIL"
    return {"verdict": verdict,
            "n_identity_rows": len(ident),
            "tested_with": "== , not a tolerance (prereg 4.3)",
            "coefficient_violations": len(bad_c),
            "displacement_violations": len(bad_d),
            "displacement_fields": sorted(disp_keys),
            "offenders": [{k: r[k] for k in ("seed", "round", "c_min", "c_max")}
                          for r in (bad_c + bad_d)[:6]]}


def mean_share_spread(rows):
    """The spread of PER-RUNG MEAN adversarial coefficient shares -- the statistic SHARE_TOL is for.

    This is the quantity every frozen user of SHARE_TOL compares: measure_admission.py:250-260 and
    measure_admission_normclip_cifar100.py:378-393 both average adv_coeff_share WITHIN a rung over
    the rounds that contain an adversary, then take max-minus-min across rungs, with a STRICT `<`.
    Returned separately from the gate so the gate cannot silently change which quantity it tests.
    """
    per_rung = {}
    for r in rows:
        if r["n_adv_in_round"] > 0 and not r["degenerate"]:
            per_rung.setdefault(r["d1"], []).append(float(r["adv_coeff_share"]))
    means = {k: float(np.mean(v)) for k, v in sorted(per_rung.items())}
    seq = list(means.values())
    return means, (float(max(seq) - min(seq)) if len(seq) > 1 else float("nan")), len(seq)


def share_gate(rows, calibration=None):
    """Prereg 4.4. c_adv pinned at 1.0, and the adversarial coefficient share CONSTANT across rungs.

    THE GATING STATISTIC IS THE PER-RUNG MEAN SPREAD, NOT A PER-CELL WORST CASE. This mattered:
    the first version of this function compared adv_coeff_share within each (seed, round) cell and
    took the worst over cells, which is a strictly larger statistic than the one SHARE_TOL is
    calibrated for. measure_admission.py:60-67 states the calibration in its own comment -- "the
    measured spread is 3.48e-07 on CIFAR-10 and 3.93e-07 on FEMNIST" -- and no per-cell maximum is
    that small; those figures are reachable only as per-rung MEAN spreads. Applied to the per-cell
    worst case, a 1e-6 absolute tolerance is failed by the PAPER'S OWN PUBLISHED alpha=0.5 leg at
    3.967e-06, which is larger than any of the three new alphas. So the old form was not a test of
    heterogeneity at all: it condemned every alpha including the published one, and the CONFOUNDED
    literal it emitted asserts a MECHANISM ("the share moves ... measures the attenuation channel")
    that the measurement does not support. SHARE_TOL itself is unchanged at 1e-6; what changed is
    that the gate now tests the quantity that constant is defined for.

    The per-cell worst case is still computed and REPORTED as a diagnostic, so the stricter reading
    stays on the record and this fix hides nothing.

    `calibration` is the same statistic measured on the frozen published configuration. If that
    fails, the gate is mis-specified rather than the data confounded, and this returns INDETERMINATE
    instead of condemning a new alpha with an instrument that condemns the published one.
    """
    live = [r for r in rows if not r["degenerate"]]
    devs = [r["c_adv_max_dev_from_one"] for r in live if np.isfinite(r["c_adv_max_dev_from_one"])]
    exact = (max(devs) == 0.0) if devs else False

    means, spread, n_rungs = mean_share_spread(rows)

    # Diagnostic only: the per-cell reading, kept so the stricter statistic remains visible.
    by_cell = {}
    for r in live:
        by_cell.setdefault((r["seed"], r["round"]), []).append(r["adv_coeff_share"])
    cells = {f"{s}|{d}": float(max(v) - min(v)) for (s, d), v in by_cell.items() if len(v) > 1}
    worst_cell = max(cells.values()) if cells else 0.0

    # THREE OUTCOMES, NOT TWO. With no live row there is nothing to compare, and calling that FAIL
    # would emit the CONFOUNDED literal -- a claim that the share MOVED -- on the strength of no
    # measurement at all. Absence of evidence gets its own label; it still blocks the arm.
    if not live or n_rungs < 2 or not np.isfinite(spread):
        verdict = "INDETERMINATE"
    elif calibration is not None and not calibration["published_passes"]:
        verdict = "INDETERMINATE"
    elif spread < SHARE_TOL and exact:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict,
            "share_tol": SHARE_TOL,
            "gating_statistic": "spread of per-rung MEAN adv_coeff_share, strict <, over rounds with "
                                "n_adv_in_round > 0; matches measure_admission.py:250-260",
            "per_rung_mean_share": means,
            "mean_share_spread_across_rungs": spread,
            "n_rungs_compared": n_rungs,
            "c_adv_exactly_one": bool(exact),
            "c_adv_max_dev_from_one": (float(max(devs)) if devs else None),
            "n_live_rows": len(live), "n_degenerate_rows": len(rows) - len(live),
            "diagnostic_per_cell_worst_spread": worst_cell,
            "diagnostic_per_cell_n_paired": len(cells),
            "diagnostic_per_cell_worst": sorted(cells.items(), key=lambda kv: -kv[1])[:5],
            "diagnostic_note": "The per-cell spread is NOT the gate. It is a strictly larger "
                               "statistic that the published alpha=0.5 leg also fails (3.967e-06), "
                               "so it does not discriminate alpha. Reported for completeness only.",
            "calibration": calibration,
            "note": "c_adv is asserted EXACTLY 1.0; the share is a float32 readback from the "
                    "transformed norms and so is held to SHARE_TOL, per measure_admission.py:60-67."}


def share_gate_calibration():
    """Measure the gating statistic on the PUBLISHED alpha=0.5 configuration before judging any alpha.

    A gate that fails the configuration the paper already published is mis-specified, and this is the
    check that would have caught the per-cell defect above automatically instead of after three alphas
    had been labelled CONFOUNDED. Read-only: results/admission_measurement.json is never written.
    """
    if not os.path.exists(CIFAR10_FROZEN):
        return {"published_passes": False, "reason": f"{os.path.relpath(CIFAR10_FROZEN, base)} absent"}
    froz = json.load(open(CIFAR10_FROZEN))
    want = {name for _, _, name in RUNGS}
    rows = [r for r in froz["per_round"] if r["d1"] in want]
    if not rows:
        return {"published_passes": False,
                "reason": f"no {sorted(want)} rows in {os.path.relpath(CIFAR10_FROZEN, base)}"}
    means, spread, n = mean_share_spread(rows)
    ok = bool(n > 1 and np.isfinite(spread) and spread < SHARE_TOL)
    return {"published_passes": ok,
            "source": os.path.relpath(CIFAR10_FROZEN, base),
            "alpha": ANCHOR_ALPHA, "rungs": sorted(want), "n_rows": len(rows),
            "per_rung_mean_share": means, "mean_share_spread_across_rungs": spread,
            "share_tol": SHARE_TOL,
            "why": "If the published configuration fails this gate, the gate is mis-specified and no "
                   "new alpha may be labelled CONFOUNDED by it."}


def baseline_admission(rows):
    """Diagnostic, NOT a gate. Baseline adversarial admission at this alpha.

    Read from the per-row `base_*` fields. The admission-CHANGE fields are displacements and are 0.0
    at an identity rung BY CONSTRUCTION, so reading one of those as the level makes every baseline
    look like zero.
    """
    live = [r for r in rows if not r["degenerate"]]
    out = {}
    for arm, field in (("coord_median", "base_coord_median_adv_argmedian_frac"),
                       ("reputation", "base_reputation_adv_weight_share"),
                       ("krum", "base_krum_admits_adv"),
                       ("cos_krum", "base_cos_krum_admits_adv")):
        v = [float(r[field]) for r in live if np.isfinite(float(r[field]))]
        out[arm] = {"field": field, "mean_level": (float(np.mean(v)) if v else None),
                    "nonzero_rounds": sum(1 for x in v if x > 0.0), "n_rounds": len(v)}
    out["is_a_gate"] = False
    out["why_not"] = ("This arm's pre-registration froze no admission floor, so no pass/fail is "
                      "scored against one here. Reported beside the frozen alpha=0.5 level for "
                      "comparison only.")
    return out


def frozen_reference():
    """The frozen alpha=0.5 doseS rows, READ ONLY, from results/admission_measurement.json."""
    if not os.path.exists(CIFAR10_FROZEN):
        return None
    d = json.load(open(CIFAR10_FROZEN))
    rows = d.get("per_round") or d.get("rows") or []
    keep = [r for r in rows
            if r.get("family") == "doseS" and r.get("attack") == ATTACK
            and float(r.get("rung", -1)) in (0.0, 2.0)
            and int(r.get("seed", -1)) in SEEDS3 and int(r.get("round", -1)) < HARNESS_ROUNDS]
    return keep


def harness_check():
    """Prove BY VALUE that this file's loop IS measure_admission.py's loop at alpha=0.5.

    Prereg 4.5. Without this, the only evidence that the alpha=0.1 numbers come from the paper's own
    admission definition is that the code looks similar. The verdict dict is PERSISTED; Round 69
    established that the existing harness check returns a dict its caller discards, leaving the claim
    witnessed only by stdout.
    """
    print("=" * 78)
    print(f"HARNESS CHECK -- this loop vs the frozen artifact at alpha={ANCHOR_ALPHA} (prereg 4.5)")
    print("=" * 78)
    ref = frozen_reference()
    if not ref:
        v = {"verdict": "INDETERMINATE",
             "reason": f"no doseS/{ATTACK} rows at rung 0/2, seeds {SEEDS3}, round < "
                       f"{HARNESS_ROUNDS} in {os.path.relpath(CIFAR10_FROZEN, base)}"}
        print(f"  {v['verdict']}: {v['reason']}")
        return v

    mine = measure_alpha(ANCHOR_ALPHA, SEEDS3, HARNESS_ROUNDS)
    idx = {(int(r["seed"]), int(r["round"]), float(r["rung"])): r for r in mine}
    fields = ["adv_coeff_share", "rho_realized", "c_mean",
              "base_coord_median_adv_argmedian_frac", "coord_median_frac_argmedian_changed",
              "base_reputation_adv_weight_share", "reputation_adv_share_delta"]
    diffs, missing = [], []
    for r in ref:
        key = (int(r["seed"]), int(r["round"]), float(r["rung"]))
        m = idx.get(key)
        if m is None:
            missing.append(key)
            continue
        for f in fields:
            if f not in r or f not in m:
                continue
            a, b = float(r[f]), float(m[f])
            if not (np.isfinite(a) and np.isfinite(b)):
                if np.isfinite(a) != np.isfinite(b):
                    diffs.append({"cell": str(key), "field": f, "frozen": a, "recomputed": b})
                continue
            if abs(a - b) > READBACK_TOL:
                diffs.append({"cell": str(key), "field": f, "frozen": a, "recomputed": b,
                              "abs_diff": abs(a - b)})
    ok = not diffs and not missing
    v = {"verdict": "PASS" if ok else "FAIL",
         "what": f"this file's measure_alpha() at alpha={ANCHOR_ALPHA}, seeds {SEEDS3}, "
                 f"{HARNESS_ROUNDS} rounds, doseS rungs 0/2, compared BY VALUE to the frozen "
                 "artifact",
         "reference": os.path.relpath(CIFAR10_FROZEN, base),
         "n_reference_rows": len(ref), "fields_compared": fields, "tol": READBACK_TOL,
         "n_mismatches": len(diffs), "missing_cells": [str(k) for k in missing],
         "mismatches": diffs[:8]}
    print(f"  compared {len(ref)} frozen row(s) on {len(fields)} field(s), tol {READBACK_TOL:.0e}")
    print(f"  {v['verdict']}: {len(diffs)} mismatch(es), {len(missing)} missing cell(s)")
    for d in diffs[:8]:
        print(f"    {d['cell']} {d['field']}: frozen {d['frozen']!r} vs {d['recomputed']!r}")
    if not ok:
        print("  This loop is NOT measure_admission.py's loop. DO NOT MEASURE A NEW ALPHA.")
    return v


# --------------------------------------------------------------------------- driver

def load_or_init():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {
        "description": "Mode-S premise per Dirichlet alpha, for the heterogeneity arm on "
                       "coord_median x committed pixel. Two frozen gates: the identity rung is the "
                       "exact identity (prereg 4.3), and the adversarial coefficient share is "
                       "constant across rungs (prereg 4.4).",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "config": {"dataset": DATASET, "model": MODEL, "d2": D2, "attack": ATTACK,
                   "N": N, "K": K, "f_adv": F_ADV, "alphas": ALPHAS,
                   "anchor_alpha": ANCHOR_ALPHA, "seeds": SEEDS3, "rounds": ROUNDS,
                   "rungs": [r[2] for r in RUNGS], "local_epochs": EPOCHS,
                   "lr": LR, "batch": BATCH, "share_tol": SHARE_TOL},
        "endpoint_only": "kappa in {0, 2} only. Never to be displayed or described as a four-rung "
                         "ladder, and no trend statistic is computed from two rungs.",
        "epoch_note": "One local epoch, as every existing admission number in the paper. Both gates "
                      "are epoch-independent: Mode S's coefficients come from a mean-1 benign ladder "
                      "and are never read from the update norms, unlike a norm threshold, which is "
                      "what made the epoch count load-bearing for the norm_clip arm.",
        "harness_check": {}, "per_alpha": {}, "per_round": [],
    }


GATE_CORRECTION = (
    "SHARE-GATE STATISTIC CORRECTED AFTER THE FIRST MEASUREMENT, THRESHOLD UNCHANGED. The first run "
    "of this script compared adv_coeff_share within each (seed, round) cell and took the worst spread "
    "over cells, then held that to SHARE_TOL = 1e-6. On that form all three alphas read FAIL at "
    "2.56e-06, 3.81e-06 and 2.88e-06 -- but so does the paper's OWN PUBLISHED alpha=0.5 leg, at "
    "3.967e-06, which is larger than any of them. A gate the published configuration fails cannot "
    "discriminate alpha, and the literal it emitted asserts a mechanism (the share moves, so the leg "
    "measures the attenuation channel) that the measurement does not support. SHARE_TOL is calibrated "
    "in its own defining comment at measure_admission.py:60-67 for a different quantity: 'the "
    "measured spread is 3.48e-07 on CIFAR-10 and 3.93e-07 on FEMNIST', figures reachable only as "
    "spreads of PER-RUNG MEAN shares, which is what measure_admission.py:250-260 and "
    "measure_admission_normclip_cifar100.py:378-393 both compare. The gate now tests that statistic "
    "with the same strict <. SHARE_TOL is still 1e-6 and was NOT widened. On the corrected statistic "
    "the published alpha=0.5 reads 9.22e-08 and the three alphas read 1.09e-07, 1.12e-07 and "
    "6.75e-07, all PASS. The per-cell worst case is still computed and reported per alpha under "
    "diagnostic_per_cell_*, so the stricter reading is not hidden. share_gate_calibration() now "
    "measures the published configuration BEFORE any alpha is judged and forces INDETERMINATE if the "
    "gate fails it, which is the check that would have caught this automatically.")


def save(d):
    d["gate_correction"] = GATE_CORRECTION
    with open(OUT, "w") as f:
        json.dump(jsonable(d), f, indent=2)


def report(alpha, rows, frozen_level, calibration=None):
    ig = identity_gate(rows)
    sg = share_gate(rows, calibration if calibration is not None else share_gate_calibration())
    ba = baseline_admission(rows)
    if ig["verdict"] == "INDETERMINATE":
        verdict = V_INDETERMINATE
    elif ig["verdict"] == "FAIL":
        verdict = V_IDENTITY_FAILED
    elif sg["verdict"] == "FAIL":
        verdict = V_CONFOUNDED
    elif sg["verdict"] == "INDETERMINATE":
        verdict = V_INDETERMINATE
    else:
        verdict = V_PREMISE_HOLDS
    rec = {"alpha": float(alpha), "n_rows": len(rows), "identity_gate": ig, "share_gate": sg,
           "baseline_admission": ba, "frozen_alpha0.5_coord_median_level": frozen_level,
           "verdict": verdict,
           "scope": "This verdict is about the PREMISE at this alpha only. It says nothing about "
                    "Delta_S(alpha), which run_heterogeneity_modeS.py measures, and it is not "
                    "evidence for or against the rise reproducing."}
    print(f"\n  --- alpha={alpha:g} ---")
    print(f"  identity rung (==)        : {ig['verdict']}  "
          f"({ig['coefficient_violations']} coeff, {ig['displacement_violations']} disp violations "
          f"in {ig['n_identity_rows']} rows)")
    print(f"  c_adv exactly 1.0         : {sg['c_adv_exactly_one']}  "
          f"(max dev {sg['c_adv_max_dev_from_one']!r})")
    print(f"  per-rung MEAN share spread: {sg['mean_share_spread_across_rungs']:.3e}  "
          f"vs SHARE_TOL {SHARE_TOL:.0e} (strict <)  -> {sg['verdict']}")
    cal = sg.get("calibration") or {}
    print(f"    gate calibration        : published alpha={ANCHOR_ALPHA:g} spread "
          f"{cal.get('mean_share_spread_across_rungs', float('nan')):.3e}, "
          f"passes={cal.get('published_passes')}")
    print(f"    per-cell worst spread   : {sg['diagnostic_per_cell_worst_spread']:.3e}   "
          f"[DIAGNOSTIC, not the gate; published alpha=0.5 fails this form too]")
    cm = ba["coord_median"]
    print(f"  baseline coord_median adm : {cm['mean_level']!r} over {cm['n_rounds']} live round(s), "
          f"nonzero in {cm['nonzero_rounds']}   [DIAGNOSTIC, not a gate]")
    print(f"    frozen alpha=0.5 level  : {frozen_level!r}")
    print(f"  {verdict}")
    return rec


def frozen_coord_median_level():
    ref = frozen_reference()
    if not ref:
        return None
    v = [float(r["base_coord_median_adv_argmedian_frac"]) for r in ref
         if not r.get("degenerate") and np.isfinite(
             float(r.get("base_coord_median_adv_argmedian_frac", float("nan"))))]
    return float(np.mean(v)) if v else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-check", action="store_true")
    ap.add_argument("--alpha", type=float, default=None)
    # Re-derive every verdict from the per_round rows ALREADY on disk, running no new training. Prereg
    # section 2 requires every number to be recomputed from the stored rows rather than transcribed,
    # so this is that recomputation and not a second measurement: the rows are the measurement.
    ap.add_argument("--reanalyze", action="store_true")
    a = ap.parse_args()
    check_frozen()

    d = load_or_init()
    if a.reanalyze:
        cal = share_gate_calibration()
        print(f"gate calibration on the published alpha={ANCHOR_ALPHA:g} configuration: "
              f"spread {cal.get('mean_share_spread_across_rungs')!r}, "
              f"passes={cal.get('published_passes')}")
        if not d.get("per_round"):
            sys.exit("REFUSING TO REANALYZE: no per_round rows on disk. There is nothing to "
                     "recompute, and an empty recomputation must not read as a verdict.")
        lvl = frozen_coord_median_level()
        for alpha in sorted({float(r["alpha"]) for r in d["per_round"]}):
            rows = [r for r in d["per_round"] if float(r["alpha"]) == alpha]
            d["per_alpha"][f"alpha{alpha:g}"] = report(alpha, rows, lvl, cal)
        save(d)
        print(f"\nRecomputed {os.path.relpath(OUT, base)} from stored rows; no training was run.")
        sys.exit(0)

    if a.harness_check:
        d["harness_check"] = harness_check()
        save(d)
        sys.exit(0)

    if not d.get("harness_check", {}).get("verdict") == "PASS":
        sys.exit("REFUSING TO RUN: --harness-check has not been run to PASS.\n"
                 "  PYTHONPATH=. python3 -m experiments.measure_admission_heterogeneity "
                 "--harness-check\n"
                 "Prereg 4.5 orders it first: until this loop is held to the frozen artifact by "
                 "value, a new alpha's numbers are not known to come from the paper's own "
                 "definition of admission.")

    todo = ALPHAS if a.alpha is None else [a.alpha]
    for alpha in todo:
        if alpha not in ALPHAS:
            sys.exit(f"REFUSING TO RUN: alpha={alpha} is not in the frozen grid {ALPHAS}. "
                     f"alpha={ANCHOR_ALPHA} is the published anchor and is measured only by "
                     "--harness-check.")
        print(f"\n=== measuring alpha={alpha:g} "
              f"({len(SEEDS3)} seeds x {ROUNDS} rounds x {len(RUNGS)} rungs) ===")
        rows = measure_alpha(alpha, SEEDS3, ROUNDS)
        d = load_or_init()
        d["per_round"] = [r for r in d["per_round"] if float(r["alpha"]) != float(alpha)] + rows
        d["per_alpha"][f"alpha{alpha:g}"] = report(alpha, rows, frozen_coord_median_level(),
                                                  share_gate_calibration())
        save(d)

    print(f"\nWrote {os.path.relpath(OUT, base)}")
    print("A premise verdict is not an ASR result. Delta_S(alpha) is measured by "
          "experiments/run_heterogeneity_modeS.py.")
