"""
Stage 1 of the CIFAR-100 composition-level replication: the PREMISE, measured before any ASR exists.

WHY THIS RUNS FIRST, AND SEPARATELY. The arm stage 2 will run is norm_clip -> coord_median on
CIFAR-100 / cifar_cnn under committed_pixel, and it is only informative if three things are true on
this dataset. Each is a way the arm can fail, and each is checked here, with its own artifact, before
a single ASR is computed:

  1. NONZERO BASELINE ADMISSION. coord_median must actually admit adversarial mass at baseline on
     CIFAR-100. If it does not, the arm inherits the exact defect the twelfth review identified in the
     paper's flagship Krum cell -- a zero that is a floor rather than a finding -- and it cannot
     answer the question it was registered to answer.
  2. THE IDENTITY RUNG DISPLACES NOTHING, EXACTLY. At tau = infinity the clip must be the identity to
     the last bit. This is the guard against the failure mode this repository has already hit once,
     where a manipulation hook was never called and the run passed silently looking like a null.
  3. THE CLIP MUST NOT ATTENUATE ADVERSARIES. If the adversarial share of coefficient mass moves
     across rungs, then Delta_c != 0, the arm measures the attenuation channel rather than the
     statistic channel, and it cannot speak to the dissociation. This is the honest risk of using a
     REAL defense instead of an oracle, and it is the condition most likely to fail.

Rules frozen in experiments/pre_registration_normclip_cifar100.md (freeze 9cbd009, Amendment 1
2eeff40) BEFORE this file existed. This script writes only results/normclip_cifar100_admission.json.
No existing results directory is read for writing, and no existing runner is edited.

WHY norm_clip AT ALL. run_all_compositions.py:426-434 implements it as
scale = min(1.0, tau / max(norm, 1e-8)) applied to every client, i.e. T(u_i) = c_i u_i with c_i > 0:
a member of the invariance theorem's positive per-client rescaling class AND a standard, deployed,
ORACLE-FREE defense. It needs no adversary mask, unlike every Mode-S, Mode-A and Mode-M rung in the
paper, so the dose here is a real defense's own hyperparameter rather than a synthetic coefficient
vector.

TWO EPOCH COUNTS, WHICH IS AMENDMENT 1'S SUBJECT. measure_admission.py:132 trains ONE local epoch;
every ASR ladder in the paper trains FLConfig.local_epochs = 2. For the dose families that mismatch is
harmless because kappa is a scale-free dispersion dial. norm_clip's dose is a NORM THRESHOLD -- it has
units -- so a tau measured on 1-epoch updates does not transfer to a 2-epoch ladder. So:

  * the same seeds and rounds are traversed at BOTH epoch counts;
  * PREMISE 1 is evaluated on the 1-epoch pass, which is the configuration in which the 0.05 floor was
    calibrated and in which every existing baseline in the paper was measured, and the 2-epoch number
    is printed beside it. Disagreement on either half of condition 1 means NOT ELIGIBLE;
  * TAU* is the median client update norm of the 2-epoch pass, because the clip must be a threshold on
    the updates it will actually face in stage 2;
  * PREMISES 2 AND 3 must pass at BOTH epoch counts.

FOUR TRAVERSALS, NOT TWO. tau* cannot be known until the norms are measured, so each epoch count is
traversed once to collect baselines and norms and once again to score the rungs at
tau in {infinity, tau*}. The second traversal re-seeds from the same seed and repeats the same
operations, so it must reproduce the first bit-for-bit; that is ASSERTED per round against a recorded
fingerprint rather than assumed, because a divergence would silently mean the rungs were scored on a
different update stack than the baselines were.

Output: results/normclip_cifar100_admission.json    (~15 min; no ASR, no 50-round run)
Run:    PYTHONPATH=. python3 -m experiments.measure_admission_normclip_cifar100
"""
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
from experiments.run_all_compositions import apply_d1_transform                          # noqa: E402
# Single-sourced from the frozen CIFAR-10 measurement: the same admission definition, the same
# aggregate mirrors, the same displacement, the same decision/admission field names, the same
# coefficient-share tolerance and the same N/K/f. Nothing about the channels is reimplemented here, so
# CIFAR-10 and CIFAR-100 cannot end up measured by two different definitions of admission -- which is
# the whole point of the premise comparison this script prints.
from experiments.measure_admission import (                                              # noqa: E402
    flatten, krum_selection, rep_weights, argmedian,
    admission, aggregates, rel_disp,
    DECISION_KEY, ADMISSION_KEY, SHARE_TOL, ATTACK_MAP,
    N, K, F_ADV,
)

OUT = os.path.join(base, "results", "normclip_cifar100_admission.json")
CIFAR10_FROZEN = os.path.join(base, "results", "admission_measurement.json")
FEMNIST_FROZEN = os.path.join(base, "results", "femnist_admission.json")

PREREG = "experiments/pre_registration_normclip_cifar100.md"
PREREG_COMMIT = "2eeff40"       # freeze 9cbd009 + Amendment 1 (2eeff40, append-only: 68 insertions,
                                # 0 deletions). 9cbd009 is the hash that froze every threshold, seed
                                # list, tau rule and verdict literal this file reads; the bump exists
                                # so the guard checks the document the paper will cite.

DATASET, MODEL = "cifar100", "cifar_cnn"
D2, ATTACK = "coord_median", "committed_pixel"
SEEDS3 = [42, 43, 44]
ROUNDS_ELIG = 6
# measure_admission.py:132's own literals, which are also FLConfig's defaults (config.py:10-11), so the
# premise measurement and the ASR ladder agree on learning rate and batch size and differ only in the
# epoch count Amendment 1 is about.
LR, BATCH = 0.01, 64
PREMISE_EPOCHS = 1      # the pass premise 1 is evaluated on: every existing baseline in the paper
LADDER_EPOCHS = 2       # FLConfig.local_epochs, i.e. the configuration stage 2 runs; tau* comes here
EPOCHS = (PREMISE_EPOCHS, LADDER_EPOCHS)

# Premise 1's two halves, both frozen in the pre-registration. BOTH are load-bearing: cos_krum's
# CIFAR-10 level is 0.0833, which clears a 0.05 floor, but it is nonzero in only 1 of 12 rounds, so a
# level test alone would certify an arm this paper already classifies as near the floor.
ADM_FLOOR = 0.05
NONZERO_FRAC_MIN = 0.5

# The three frozen verdict literals, byte-exact from the pre-registration's demotion clause. They are
# constants so that this runner REPORTS one rather than composing prose after seeing the numbers.
V_PREMISE_FAILED = ("PREMISE FAILED: THE COMPOSITION-LEVEL REPLICATION IS NOT ESTABLISHED ON "
                    "CIFAR-100.")
V_CONFOUNDED = "ARM CONFOUNDED ON THIS DATASET: THE CLIP ATTENUATES ADVERSARIES, SO Δ_c ≠ 0."
V_ELIGIBLE = ("ELIGIBLE: coord_median admits adversarial mass at baseline on CIFAR-100, the identity "
              "rung displaces nothing exactly, and the clip does not move the adversarial coefficient "
              "share. Stage 2 may run.")

INF = float("inf")


def jsonable(o):
    """Replace every non-finite float with None, recursively, at the write boundary.

    json.dump emits float("nan") and float("inf") as the bare tokens `NaN` and `Infinity`. Python's
    own json.loads accepts them, which is why such a file looks fine from inside this repository and
    fails for everyone else: they are not valid JSON, so jq and every strict reader in another
    language reject the WHOLE file, not just the offending field. An artifact a paper cites has to
    parse for a reader who is not using Python's tolerant loader.

    This runs after all logic, so it cannot disturb a comparison or a format string. Several fields
    here fall back to NaN on an empty collection -- the branch a degenerate future run would take --
    and sanitizing at the boundary covers those too rather than one at a time.
    """
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def check_frozen():
    """Refuse to write results/ unless the pre-registration is committed at PREREG_COMMIT and clean.

    Without this the constant above would be a LABEL rather than a check: this script stamps it into the
    artifact, and stage 2's gate compares that stamp to its own module constant, so two string literals
    would agree while the document on disk said something else. The pre-registration's non-negotiable is
    that nothing is written to results/ before it is committed, and that is enforced here.

    The hash check alone is necessary and not sufficient: `git log -1` reports the last commit that
    touched the file and is unchanged by uncommitted edits to it, which is the defect that let a
    Round-57 amendment pass its own gate. Hence the working-tree check.
    """
    if not os.path.exists(os.path.join(base, PREREG)):
        return f"{PREREG} does not exist."
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=base, capture_output=True, text=True, timeout=20)
    actual = out.stdout.strip()
    if not actual or not actual.startswith(PREREG_COMMIT[:7]):
        return (f"{PREREG} was last touched at {actual or 'UNTRACKED'}, but PREREG_COMMIT is "
                f"{PREREG_COMMIT}. The premise thresholds must be frozen before they are measured "
                "against.")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=base, capture_output=True, text=True, timeout=20)
    if dirty.stdout.strip():
        return (f"{PREREG} has uncommitted changes ({dirty.stdout.strip().split()[0]}), so it is not "
                f"frozen at {actual} whatever `git log` says. Commit it and set PREREG_COMMIT to the "
                "new hash.")
    print(f"[OK] {PREREG} frozen at {actual}, working tree clean")
    return None


def dev_name():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def traverse(local_epochs, taus, expect=None):
    """One traversal of (seeds x rounds) at a fixed local epoch count.

    Returns (rung_rows, base_rows, fingerprints). `taus` may be empty, in which case only the
    baselines and the client update norms are measured and no transform is applied at all.

    The server is advanced with the DEFAULT aggregation, plain fedavg, exactly as
    measure_admission.measure() does: the point of this loop is to hand every rung the same raw update
    stack, not to run any one defense, and the frozen CIFAR-10 baselines this premise is compared
    against were measured on a fedavg-advanced trajectory. Under committed_pixel the adversary does not
    scale its update, so the FEMNIST overflow that forced measure_admission_femnist.py to drop
    non-finite rounds does not arise here; rounds are still checked for finiteness rather than assumed
    to be finite.

    `expect` is the fingerprint list from an earlier traversal at the same epoch count. When given,
    every round is asserted to reproduce it bit-for-bit, which is what licenses scoring the rungs in a
    second traversal instead of caching ~90 update stacks in memory.
    """
    dev = dev_name()
    rung_rows, base_rows, fps = [], [], []
    for seed in SEEDS3:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cd, _, nc = get_federated_dataset(DATASET, N, 0.5, seed)
        srv = FederatedServer(get_model(MODEL, nc), dev)
        atk = get_attack(ATTACK_MAP[ATTACK])
        adv = set(range(int(N * F_ADV)))
        cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
              for i in range(N)]
        for rnd in range(ROUNDS_ELIG):
            pids = np.random.choice(N, K, replace=False)
            ups = []
            for cid in pids:
                u = cl[cid].train(srv.global_model, local_epochs, LR, BATCH)
                if cid in adv:
                    u = atk.manipulate_update(u, srv.global_model)
                ups.append(u)
            raw = flatten(ups)
            adv_rows = [i for i, cid in enumerate(pids) if cid in adv]

            sel_krum_0, _ = krum_selection(raw, False)
            sel_cos_0, _ = krum_selection(raw, True)
            w_rep_0 = rep_weights(raw, False)
            am_0 = argmedian(raw)
            adm_0 = admission(raw, adv_rows, sel_krum_0, sel_cos_0, w_rep_0, am_0)
            agg_0 = aggregates(raw, sel_krum_0, sel_cos_0, w_rep_0)
            client_norms = [float(x) for x in raw.norm(dim=1).tolist()]

            # The fingerprint is the pair of things a divergence would show up in: who participated
            # and what they produced. Compared with ==, not with a tolerance: a re-seeded repeat of a
            # deterministic loop is either the same computation or it is not one.
            fp = {"seed": int(seed), "round": int(rnd), "pids": [int(p) for p in pids],
                  "client_norms": client_norms}
            if expect is not None:
                ref = expect[len(fps)]
                if (ref["seed"], ref["round"]) != (fp["seed"], fp["round"]):
                    sys.exit("REFUSING TO CONTINUE: traversal order diverged at "
                             f"(seed {seed}, round {rnd}).")
                if ref["pids"] != fp["pids"] or ref["client_norms"] != fp["client_norms"]:
                    sys.exit(f"REFUSING TO CONTINUE: the second traversal at {local_epochs} local "
                             f"epoch(s) did not reproduce the first at (seed {seed}, round {rnd}). "
                             "The rungs would be scored on a different update stack than the "
                             "baselines were, so nothing here would be comparable.")
            fps.append(fp)

            row0 = {"seed": int(seed), "round": int(rnd), "local_epochs": int(local_epochs),
                    "n_adv_in_round": len(adv_rows), "n_benign_in_round": K - len(adv_rows),
                    "degenerate": bool(len(adv_rows) == 0 or K - len(adv_rows) < 2),
                    "client_norms": client_norms,
                    "raw_finite": bool(np.isfinite(client_norms).all())}
            row0.update({f"base_{k}": v for k, v in adm_0.items()})
            base_rows.append(row0)

            for tau in taus:
                t = apply_d1_transform(ups, "norm_clip", tau=tau)
                st = flatten(t)
                # Coefficients read back from the transformed NORMS rather than taken from the
                # builder, exactly as measure_admission.py:152 does, so this also verifies that the
                # shipped code path applied what the docstring says it applies.
                c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
                sel_krum, _ = krum_selection(st, False)
                sel_cos, _ = krum_selection(st, True)
                w_rep = rep_weights(st, False)
                am = argmedian(st)
                adm = admission(st, adv_rows, sel_krum, sel_cos, w_rep, am)
                agg = aggregates(st, sel_krum, sel_cos, w_rep)
                c_adv = [float(c[a].item()) for a in adv_rows]
                row = {
                    "seed": int(seed), "round": int(rnd), "local_epochs": int(local_epochs),
                    # tau is null rather than Infinity at the identity rung, and the flag carries
                    # what the null means. json.dump writes float("inf") as the bare token
                    # `Infinity`, which is NOT valid JSON: json.loads accepts it, jq and every
                    # strict reader in another language reject the whole file. An artifact the paper
                    # cites must parse for a reader who does not happen to use Python's tolerant
                    # loader. Same convention as run_normclip_cifar100.py's own cells.
                    "tau": (None if tau == INF else float(tau)),
                    "tau_is_infinity": bool(tau == INF),
                    "rung": ("identity" if tau == INF else "tau_star"),
                    "n_adv_in_round": len(adv_rows),
                    "degenerate": bool(len(adv_rows) == 0 or K - len(adv_rows) < 2),
                    "c_min": float(c.min().item()), "c_max": float(c.max().item()),
                    "c_mean": float(c.mean().item()),
                    "max_dev_from_1": float((c - 1.0).abs().max().item()),
                    "n_clipped": int((c < 1.0).sum().item()),
                    # null, not NaN, in the rounds where no adversary was sampled. Same reason as
                    # tau above, plus a second one: adv_rounds() filters these rows out of every
                    # premise, but a consumer that forgets to filter gets NaN, which propagates
                    # silently through a mean and prints as a number-shaped nothing. A null raises.
                    "c_adv_min": (min(c_adv) if c_adv else None),
                    "c_adv_max": (max(c_adv) if c_adv else None),
                    "adv_coeff_share": (float(sum(c_adv) / float(c.sum().item()))
                                        if c_adv else None),
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
                row.update({f"agg_disp_{a}": rel_disp(agg[a], agg_0[a]) for a in agg})
                rung_rows.append(row)
            srv.apply_update(srv.aggregate(ups))
        print(f"    seed {seed} done ({ROUNDS_ELIG} rounds, {local_epochs} local epoch(s), "
              f"{len(taus)} rung(s))", flush=True)
    return rung_rows, base_rows, fps


def adv_rounds(rows):
    """The rounds in which an adversary was actually sampled. A baseline averaged over rounds with no
    adversary present is an average over rounds in which admitting adversarial mass was impossible."""
    return [r for r in rows if r["n_adv_in_round"] > 0]


def level_and_support(rows, field):
    """(mean level, nonzero rounds, adversary rounds) for a base_* field.

    The level alone is an average over rounds and one admitting round can carry it, which is exactly
    how cos_krum reads 0.0833 on a single admitting round in twelve. So the support is returned with
    it and both halves of premise 1 are evaluated.
    """
    sub = adv_rounds(rows)
    vals = [float(r[field]) for r in sub]
    if not vals:
        return float("nan"), 0, 0
    return float(np.mean(vals)), int(sum(1 for v in vals if v > 0.0)), len(vals)


def frozen_level(path, field, attack):
    """The same statistic recomputed from a frozen artifact, read-only, never written.

    Reads per_round rather than summary. summary["doseS|<arm>|<rung>|admission"] is a DISPLACEMENT from
    the identity rung and is 0.0 at every rung == 0 cell BY CONSTRUCTION, so reading it as a level
    makes every baseline look like a floor. The level lives in per_round's base_* fields. One row per
    (seed, round) is taken by fixing family=doseS and rung=0, since base_* does not depend on the rung.
    """
    if not os.path.exists(path):
        return None
    rows = [r for r in json.load(open(path))["per_round"]
            if r.get("family") == "doseS" and abs(float(r["rung"])) < 1e-12
            and r.get("attack", attack) == attack and r["n_adv_in_round"] > 0]
    if not rows:
        return None
    lvl, nz, n = level_and_support(rows, field)
    return {"level": lvl, "nonzero_rounds": nz, "adv_rounds": n, "attack": attack,
            "source": os.path.relpath(path, base)}


def identity_exact(rows):
    """Premise 2: at tau = infinity every coefficient is exactly 1 and every channel displacement is
    exactly 0. Tested with ==, not with a tolerance: the clip returns u[k] * 1.0 there, and
    multiplying a float32 tensor by exactly 1.0 is bit-identical, so anything but zero is a defect in
    the code path rather than rounding."""
    checks = {}
    dev = [r["max_dev_from_1"] for r in rows]
    checks["c_exactly_1"] = bool(rows and max(dev) == 0.0)
    checks["max_dev_from_1"] = float(max(dev)) if dev else float("nan")
    disp = [abs(float(r[f"agg_disp_{a}"]))
            for r in rows for a in ("krum", "cos_krum", "reputation", "coord_median")]
    checks["agg_disp_exactly_0"] = bool(disp and max(disp) == 0.0)
    checks["max_agg_disp"] = float(max(disp)) if disp else float("nan")
    dec = [r[k] for r in rows for k in DECISION_KEY.values()]
    checks["no_decision_change"] = bool(dec and all(float(x) == 0.0 for x in dec))
    adm = [abs(float(r[k])) for r in rows for k in ADMISSION_KEY.values()]
    checks["no_admission_change"] = bool(adm and max(adm) == 0.0)
    checks["ok"] = bool(checks["c_exactly_1"] and checks["agg_disp_exactly_0"]
                        and checks["no_decision_change"] and checks["no_admission_change"])
    return checks


def share_constant(rows):
    """Premise 3: the adversarial share of coefficient mass must not move across the two rungs.

    Compared against SHARE_TOL, imported from measure_admission rather than restated, because the
    share is recomputed from float32 norms and so cannot be bit-exact even when the construction is.
    If it moves, the clip is attenuating adversaries, Delta_c != 0, and this arm measures the
    attenuation channel rather than the statistic channel.
    """
    per_rung = {}
    for r in adv_rounds(rows):
        per_rung.setdefault(r["rung"], []).append(float(r["adv_coeff_share"]))
    means = {k: float(np.mean(v)) for k, v in per_rung.items()}
    seq = list(means.values())
    spread = float(max(seq) - min(seq)) if len(seq) > 1 else float("nan")
    return {"per_rung_mean_share": means, "spread": spread, "share_tol": SHARE_TOL,
            "ok": bool(len(seq) > 1 and spread < SHARE_TOL)}


def main():
    why = check_frozen()
    if why is not None:
        print(f"REFUSING TO RUN: {why}")
        return 1
    print("=== STAGE 1: the premise of the CIFAR-100 composition-level replication ===")
    print(f"    arm under consideration: norm_clip -> {D2} on {DATASET}/{MODEL}, "
          f"{ATTACK.replace('committed_', '')}")
    print(f"    {len(SEEDS3)} seeds x {ROUNDS_ELIG} rounds, no ASR, no 50-round run")
    print(f"    rules frozen at {PREREG_COMMIT} ({PREREG}); this script computes no ASR and freezes "
          "nothing")
    print("    PROSPECTIVE: stage 2 does not exist as an artifact and cannot run unless all three "
          "premise conditions pass.\n", flush=True)

    # --- Phase 1: baselines and client norms, at both epoch counts. No transform is applied. ---
    print("=== PHASE 1: baselines and client update norms (no transform applied) ===", flush=True)
    base_rows = {}
    fps = {}
    for ep in EPOCHS:
        print(f"  -- {ep} local epoch(s) --", flush=True)
        _, base_rows[ep], fps[ep] = traverse(ep, ())

    norms = {ep: [x for r in base_rows[ep] for x in r["client_norms"]] for ep in EPOCHS}
    medians = {ep: float(np.median(norms[ep])) for ep in EPOCHS}
    tau_star = medians[LADDER_EPOCHS]
    print("\n  client update norm, median over every (seed, round, client):")
    for ep in EPOCHS:
        print(f"    {ep} local epoch(s): median {medians[ep]:.6f}   "
              f"range [{min(norms[ep]):.6f}, {max(norms[ep]):.6f}]   n={len(norms[ep])}")
    print(f"\n  tau* := {tau_star:.6f}, the median at {LADDER_EPOCHS} local epochs, which is the")
    print("  configuration stage 2 runs. The rule was frozen before this number existed and is not")
    print("  amended by whatever it turned out to be: if tau* clips nobody or everybody, that is the")
    print(f"  measured outcome of the frozen rule. The {PREMISE_EPOCHS}-epoch median "
          f"({medians[PREMISE_EPOCHS]:.6f}) is recorded and NOT used, because a clip that will face")
    print("  2-epoch updates must be a threshold on 2-epoch norms (Amendment 1).", flush=True)

    # --- PREMISE 1, on the pass the floor was calibrated in, with the other printed beside it ---
    print(f"\n=== PREMISE 1: nonzero baseline admission for {D2} on {DATASET} ===")
    print(f"    condition: level >= {ADM_FLOOR} AND nonzero in >= {NONZERO_FRAC_MIN:.0%} of the "
          "rounds an adversary was sampled")
    print("    both halves are load-bearing: cos_krum's CIFAR-10 level is 0.0833, which clears the")
    print("    floor, on ONE admitting round in twelve.\n")
    field = "base_coord_median_adv_argmedian_frac"
    p1 = {}
    for ep in EPOCHS:
        lvl, nz, n = level_and_support(base_rows[ep], field)
        frac = (nz / n) if n else float("nan")
        ok = bool(lvl >= ADM_FLOOR and n > 0 and frac >= NONZERO_FRAC_MIN)
        p1[ep] = {"level": lvl, "nonzero_rounds": nz, "adv_rounds": n, "nonzero_frac": frac,
                  "ok": ok}
        tag = "PRIMARY" if ep == PREMISE_EPOCHS else "comparison"
        print(f"  {ep} local epoch(s) [{tag:10s}]: level {lvl:.4f}  nonzero {nz}/{n}  "
              f"{'PASS' if ok else 'FAIL'}")

    agree = bool(p1[PREMISE_EPOCHS]["ok"] == p1[LADDER_EPOCHS]["ok"])
    p1_ok = bool(p1[PREMISE_EPOCHS]["ok"] and agree)
    if not agree:
        print("\n  ** THE TWO PASSES DISAGREE. Amendment 1 fixes the conservative direction in")
        print("     advance: a disagreement is NOT ELIGIBLE, because the premise would then hold in")
        print("     the configuration the floor was calibrated in but not in the one stage 2 runs, or")
        print("     the reverse, and neither reading licenses the arm.")

    print("\n  the same statistic on the two frozen artifacts, recomputed rather than transcribed:")
    frozen = {
        "cifar10_coord_median_pixel": frozen_level(CIFAR10_FROZEN, field, "committed_pixel"),
        "cifar10_coord_median_scaling": frozen_level(CIFAR10_FROZEN, field, "committed_scaling"),
        "femnist_coord_median_scaling": frozen_level(FEMNIST_FROZEN, field, "committed_scaling"),
    }
    for name, f in frozen.items():
        if f is None:
            print(f"    {name:34s} ABSENT")
        else:
            print(f"    {name:34s} level {f['level']:.4f}  nonzero "
                  f"{f['nonzero_rounds']}/{f['adv_rounds']}  ({f['source']})")
    print("    FEMNIST's nonzero level is positive evidence that this premise transfers across")
    print("    datasets for this aggregator; it was recorded in the pre-registration before this ran.")

    # --- Phase 2: the two rungs, at both epoch counts, re-traversing the same stacks ---
    print(f"\n=== PHASE 2: the two rungs, tau in {{inf, {tau_star:.6f}}}, both epoch counts ===")
    print("    each traversal is asserted to reproduce phase 1's stacks bit-for-bit, per round.\n",
          flush=True)
    rungs = {}
    for ep in EPOCHS:
        print(f"  -- {ep} local epoch(s) --", flush=True)
        rungs[ep], _, _ = traverse(ep, (INF, tau_star), expect=fps[ep])

    ident = {ep: identity_exact([r for r in rungs[ep] if r["rung"] == "identity"]) for ep in EPOCHS}
    share = {ep: share_constant(rungs[ep]) for ep in EPOCHS}
    clipped = {}
    for ep in EPOCHS:
        sub = [r for r in rungs[ep] if r["rung"] == "tau_star"]
        clipped[ep] = {"n_clipped": int(sum(r["n_clipped"] for r in sub)),
                       "n_client_rounds": int(len(sub) * K),
                       "c_min": float(min(r["c_min"] for r in sub)) if sub else float("nan")}

    print("\n=== PREMISE 2: the identity rung displaces nothing, exactly (== 0, not < tol) ===")
    for ep in EPOCHS:
        c = ident[ep]
        print(f"  {ep} local epoch(s): c exactly 1 {c['c_exactly_1']}  "
              f"agg disp exactly 0 {c['agg_disp_exactly_0']}  "
              f"no decision change {c['no_decision_change']}  "
              f"no admission change {c['no_admission_change']}  "
              f"-> {'PASS' if c['ok'] else 'FAIL'}")
    p2_ok = bool(all(ident[ep]["ok"] for ep in EPOCHS))

    print(f"\n=== PREMISE 3: the adversarial coefficient share does not move (tol {SHARE_TOL:.0e}) ===")
    for ep in EPOCHS:
        s = share[ep]
        cells = "  ".join(f"{k}:{v:.6f}" for k, v in sorted(s["per_rung_mean_share"].items()))
        print(f"  {ep} local epoch(s): {cells}   spread {s['spread']:.2e}  "
              f"-> {'PASS' if s['ok'] else 'FAIL'}")
        print(f"       clipped {clipped[ep]['n_clipped']}/{clipped[ep]['n_client_rounds']} "
              f"client-rounds at tau*, smallest coefficient {clipped[ep]['c_min']:.6f}")
    p3_ok = bool(all(share[ep]["ok"] for ep in EPOCHS))

    # --- The verdict, one of three literals frozen before any of these numbers existed ---
    if not (p1_ok and p2_ok):
        verdict, eligible = V_PREMISE_FAILED, False
    elif not p3_ok:
        verdict, eligible = V_CONFOUNDED, False
    else:
        verdict, eligible = V_ELIGIBLE, True

    print("\n=== STAGE 1 VERDICT (one of three literals frozen in the pre-registration) ===")
    print(f"  premise 1 (nonzero baseline admission): {'PASS' if p1_ok else 'FAIL'}")
    print(f"  premise 2 (exact identity rung):        {'PASS' if p2_ok else 'FAIL'}")
    print(f"  premise 3 (share does not move):        {'PASS' if p3_ok else 'FAIL'}")
    print(f"\n  {verdict}")
    if not eligible:
        print("\n  Stage 2 does not run. experiments/run_normclip_cifar100.py reads this verdict from")
        print("  the artifact and refuses to start, and the paper reports the attempt and its failure")
        print("  in the abstract and in section 5, not in a footnote.")

    premise = {
        "verdict": verdict, "eligible": eligible,
        "condition_1_nonzero_baseline_admission": {
            "field": field, "floor": ADM_FLOOR, "nonzero_frac_min": NONZERO_FRAC_MIN,
            "primary_epochs": PREMISE_EPOCHS, "per_epochs": {str(k): v for k, v in p1.items()},
            "passes_agree": agree, "ok": p1_ok,
            "note": "Evaluated on the 1-epoch pass, the configuration in which the 0.05 floor was "
                    "calibrated and in which every existing baseline in the paper was measured. A "
                    "disagreement between the two passes is NOT ELIGIBLE (Amendment 1)."},
        "condition_2_exact_identity": {str(k): v for k, v in ident.items()},
        "condition_3_share_constant": {str(k): v for k, v in share.items()},
        "clipping_at_tau_star": {str(k): v for k, v in clipped.items()},
    }

    json.dump(jsonable({
        "description": "Stage 1 of the CIFAR-100 composition-level replication: the PREMISE of the "
                       f"norm_clip -> {D2} arm on {DATASET}/{MODEL} under {ATTACK}, measured before "
                       "any ASR for this arm exists. No ASR here. Freezes nothing. "
                       "results/admission_measurement.json and results/femnist_admission.json are "
                       "read for comparison and never written.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "dataset": DATASET, "model": MODEL, "d1": "norm_clip", "d2": D2, "attack": ATTACK,
        "config": {"N": N, "K": K, "f": F_ADV, "alpha": 0.5, "seeds": SEEDS3,
                   "rounds_per_seed": ROUNDS_ELIG, "lr": LR, "batch": BATCH,
                   "local_epochs_measured": list(EPOCHS),
                   "local_epochs_premise_primary": PREMISE_EPOCHS,
                   "local_epochs_ladder": LADDER_EPOCHS,
                   "server_advance": "fedavg (measure_admission.measure()'s default), so every rung "
                                     "sees the same raw stack and the frozen CIFAR-10 baselines are "
                                     "comparable"},
        "tau_rule": "the median client update norm measured in stage 1, so that roughly half the "
                    "clients clip; instantiated on the 2-epoch pass because the clip must be a "
                    "threshold on the updates stage 2 produces (Amendment 1). Frozen before the "
                    "number existed and not amended by it.",
        "tau_star": tau_star,
        "client_norm_medians": {str(k): v for k, v in medians.items()},
        "client_norm_n": {str(k): len(norms[k]) for k in EPOCHS},
        "frozen_comparison": frozen,
        "premise": premise,
        "per_round_baselines": [r for ep in EPOCHS for r in base_rows[ep]],
        "per_round_rungs": [r for ep in EPOCHS for r in rungs[ep]],
        "fingerprints": {str(k): v for k, v in fps.items()},
    }), open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    print("The verdict is in the artifact, not only on stdout: Round 69 found one check in this "
          "repository whose\nverdict dict its caller discarded, leaving a paper sentence witnessed "
          "only by a run's stdout.")
    print("results/admission_measurement.json and results/femnist_admission.json were not modified.")
    return 0 if eligible else 2


if __name__ == "__main__":
    sys.exit(main())
