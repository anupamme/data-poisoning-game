"""
Dose-response test of C2.

The two previous pre-registered suites tried to validate C2 by CHOOSING a different downstream
defense, and both failed for the same structural reason: C0 & C1 force ASR(d2, a*) < 0.5, so
mechanism and standalone effectiveness co-vary across any cross-defense contrast
(mechanism--effectiveness confounding; see the testability proposition).

That proposition constrains cross-defense designs only. Here d2 AND the attack are held fixed and
only the DOSE of upstream disturbance varies, so standalone effectiveness -- the C1 input -- is
numerically the same quantity at every rung and the confound is impossible by construction rather
than matched away. The proposition therefore stops being the reason C2 cannot be validated and
becomes the derivation of the only design that can validate it.

  d1 = dose_kappa<K>:  c_j  proportional to  exp(K * (2j/(K_participants-1) - 1)),  mean(c) = 1,
  assigned by a (seed, round)-keyed permutation, NOT by client id. See dose_coefficients() in
  experiments/run_all_compositions.py. The realized weight ratio is exactly exp(2K) = rho, the
  theorem's own quantity, so kappa is a dial on the theory. kappa=0 is the exact identity.

  dose_kappa is an INSTRUMENT, NOT A DEFENSE. cos_krum is a mechanism-isolating ablation, not a
  proposed defense; its exact invariance was verified in code before it was claimed
  (experiments/verify_cos_invariance.py / results/cos_invariance_check.json).

PRIMARY test (frozen): pooled across all 16 (arm, kappa) cells, the rise in ASR over that arm's own
kappa=0 rung is rank-correlated with the MEASURED disturbance of that arm's statistic (Spearman,
one-sided, alpha 0.05). The predictor is a measured property of the statistic -- not the identity of
the defense and not its standalone effectiveness, both held fixed within every arm.

  arm  d2            attack    Prop 1 class            measured    predicted shape in kappa
                                                       disturbance
                                                       at kappa=2
  1    krum          scaling   (c) not invariant          0.867     monotone rise
  2    reputation    scaling   (c) not invariant          0.933     monotone rise
  3    cos_krum      pixel     (a) EXACTLY INVARIANT      0.000     FLAT  <-- negative control
  4    coord_median  pixel     (b) conditionally inv.     0.487     rise, shallower than arms 1-2

Arms were selected by the paper's own power rule -- standalone d2 must genuinely suppress the
attack (ASR < 0.5 at clean accuracy >= ACC_FLOOR) -- because a baseline already above threshold has
no suppression left to lose and would give a flat curve by ceiling effect. Applying that rule to
the measured baselines leaves exactly these four cells.

Arm 3 is falsifiable, unlike the ->FLTrust pairs of the previous round: cos_krum's SELECTION is
bit-identical at every kappa -- measured, 0 changes in 120 cells at rho up to 54.6 -- but the
aggregate is c_j*u_j for the selected j, so the 50-round trajectory is not, and suppression can
still be lost through the magnitude channel (as rfa->cos_krum already did, Delta = -0.027).
Symmetrically, a flat arm 1 or 2 refutes the framework's central fragility claim. Whichever happens
is what gets written.

The disturbance column above is measured, before the freeze and without computing any ASR, by
experiments/measure_dose_disturbance.py -> results/dose_disturbance.json. It establishes that there
IS a dose (0 at the identity rung, rising monotonically in every non-invariant arm) and that the
class-(a) invariance claim holds in the shipped code. Run it, and read it, before this script.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
Seeds 42--46 (n=5). 4 arms x 4 rungs x 5 seeds = 80 runs. The kappa=0 rung is run FRESH at all
five seeds rather than reused, so every rung of every curve comes from one code path at one seed
set; the published standalone baselines then serve as an independent cross-check of that rung
(--harness-check) instead of as data in it.

Output: results/dose_response/summary.json (resumable; written after every cell).

DO NOT RUN until experiments/pre_registration_dose_response.md is git-committed and PREREG_COMMIT
below is set to that hash. The script refuses to start otherwise: an unfrozen run would make the
predicted shapes unfalsifiable, which is the entire point.

  python3 experiments/run_dose_response.py --harness-check   # kappa=0 vs published standalone
  python3 experiments/run_dose_response.py                   # the ladder
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from torch.utils.data import Subset
from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import generic_compose
from experiments.adversary_hook import apply_adversary

# experiments/pre_registration_dose_response.md, committed before results/dose_response/ existed.
PREREG_COMMIT = "e711a95"

# (d2, attack, prop1_class, predicted_shape) -- copied VERBATIM from the frozen pre-registration
# table, in the same order. The prediction is a SHAPE in kappa, not a per-cell label: that is what
# a dose-response design licenses and what the previous two suites could not deliver.
ARMS = [
    ("krum",         "committed_scaling", "c) not invariant",      "monotone rise"),
    ("reputation",   "committed_scaling", "c) not invariant",      "monotone rise"),
    ("cos_krum",     "committed_pixel",   "a) exactly invariant",  "FLAT"),
    ("coord_median", "committed_pixel",   "b) conditionally inv.", "shallower rise"),
]
# Measured BEFORE the freeze, from results/dose_disturbance.json: the fraction of live rounds (or,
# for coord_median, of coordinates) on which each rung actually changes that arm's decision. This is
# the ABSCISSA of the dose-response curves and the predictor of the primary pooled test, so it is
# carried here verbatim and the analysis reads it from the JSON rather than from this table.
MEASURED_DISTURBANCE = {
    ("krum", "committed_scaling"):         {0.0: 0.000, 0.5: 0.600, 1.0: 0.733, 2.0: 0.867},
    ("reputation", "committed_scaling"):   {0.0: 0.000, 0.5: 0.867, 1.0: 0.867, 2.0: 0.933},
    ("cos_krum", "committed_pixel"):       {0.0: 0.000, 0.5: 0.000, 1.0: 0.000, 2.0: 0.000},
    ("coord_median", "committed_pixel"):   {0.0: 0.000, 0.5: 0.251, 1.0: 0.378, 2.0: 0.487},
}
KAPPAS = [0.0, 0.5, 1.0, 2.0]          # frozen grid; rho = exp(2*kappa) = 1.00, 2.72, 7.39, 54.60
SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}
ACC_FLOOR = 0.35          # below this a low ASR is a collapsed model, not suppression

# Published standalone baselines: (ASR, accuracy, n, source). NOT used as data -- the kappa=0 rung
# is measured fresh. Used only by --harness-check, which asserts that kappa=0 (the exact identity)
# reproduces them, so that a protocol difference between suites surfaces BEFORE the ladder runs
# rather than being explained afterwards. Every value is re-read PER SEED from the source JSON by
# published_per_seed() below, never transcribed; the means here are for the printed header only.
STANDALONE = {
    ("krum", "committed_scaling"):
        (0.061, 0.577, 3, "results/prospective_pilot/summary.json"),
    ("reputation", "committed_scaling"):
        (0.017, 0.777, 5, "results/pure_defense_baselines/summary.json"),
    ("cos_krum", "committed_pixel"):
        (0.300, 0.598, 3, "results/metric_swap_baselines/summary.json"),
    ("coord_median", "committed_pixel"):
        (0.443, 0.767, 5, "results/pure_defense_baselines/summary.json"),
}


def published_per_seed(d2, attack):
    """{seed: (accuracy, asr)} for d2's standalone baseline, read from its source of record.

    The four arms' baselines live in three different suites with three different JSON layouts, so
    the lookup is explicit per arm rather than guessed. Returns {} if the source is missing.
    """
    p = os.path.join(base, STANDALONE[(d2, attack)][3])
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    if "cells" in d:                                    # prospective_pilot / metric_swap_baselines
        key = f"{d2}|{attack}" if f"{d2}|{attack}" in d["cells"] else f"fedavg_then_{d2}|{attack}"
        rows = d["cells"][key]["per_seed"]
    else:                                               # pure_defense_baselines
        rows = d["defenses"][d2]["adversary_policies"][attack]["per_seed"]
    return {r["seed"]: (r["accuracy"], r["asr"]) for r in rows}

# NOTE: the directory is created in __main__, not at import time. Importing this module must not
# touch results/dose_response/, or the provenance gate ("prereg committed before the first write to
# the output directory") becomes ambiguous.
out_dir = os.path.join(base, "results", "dose_response")
out_path = os.path.join(out_dir, "summary.json")


def d1_name(kappa):
    """The synthetic upstream stage for a rung. Formatted so parse_dose() round-trips it."""
    return f"dose_kappa{kappa}"


def rho(kappa):
    """Realized upstream weight ratio max(c)/min(c) -- exactly the theorem's rho."""
    return float(np.exp(2.0 * kappa))


def run_one(seed, d2, attack_name, kappa, fl_config=None, alpha=0.5, ca=None):
    """One 50-round FL run of dose_kappa<kappa> -> d2 under attack_name.

    fl_config/alpha/ca are the regime knobs of pre_registration_regime_dissociation.md, added as
    defaulted arguments so that every existing call site is bit-identical: fl_config=None means the
    frozen FL_CONFIG, alpha=0.5 is the frozen Dirichlet concentration (it was a hardcoded literal
    before and is now this argument's default), and ca=None means the frozen adversary.
    --harness-check proves the default path unchanged rather than assuming it. ca=(eps,
    decorrelate) SUBSTITUTES the criterion-aware construction for manipulate_update; the shared
    implementation is experiments/adversary_hook.py, so this runner and run_targeted_dose.py cannot
    drift in the adversary they apply.
    """
    cfg = fl_config or FL_CONFIG
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset("cifar10", cfg.num_clients, alpha, seed)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(cfg.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[attack_name])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(cfg.num_clients)]
    d1 = d1_name(kappa)
    lr = cfg.learning_rate
    for rnd in range(cfg.num_rounds):
        pids = np.random.choice(cfg.num_clients,
                                size=min(cfg.clients_per_round, cfg.num_clients),
                                replace=False)
        ups = []
        for cid in pids:
            ups.append(cl[cid].train(srv.global_model, cfg.local_epochs, lr, cfg.local_batch_size))
        # The adversarial step is hoisted out of the training loop so that this runner and
        # run_targeted_dose.py share ONE implementation (experiments/adversary_hook.py) -- the
        # design comparison between the two ladders is void if they apply different adversaries.
        # Bit-neutral for this suite's two attacks: backdoor_pixel inherits the identity
        # manipulate_update and model_scaling is an elementwise multiply, so neither consumes RNG
        # nor reads another client's update. --harness-check is the proof, not this comment.
        ups = apply_adversary(ups, pids, adv, atk, srv.global_model, ca=ca, verify=(rnd == 0))
        # dose_key = (seed, round): every arm at a given (seed, round) receives the IDENTICAL
        # coefficient vector, so the arms differ in d2 and nothing else. Keying on the round rather
        # than fixing one vector per seed keeps the dose uncorrelated with client identity, and so
        # with adversary status, across the trajectory.
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=5.0, dose_key=(seed, rnd)))
        lr *= getattr(cfg, "lr_decay", 1.0)
    return float(srv.evaluate(td)["accuracy"]), float(evaluate_backdoor(srv.global_model, td, device=dev))


def cell_key(d2, attack, kappa):
    return f"{d1_name(kappa)}_then_{d2}|{attack}"


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    json.dump({"description": "Dose-response test of C2; shapes frozen at "
                              f"{PREREG_COMMIT} (experiments/pre_registration_dose_response.md)",
               "prereg_commit": PREREG_COMMIT,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "seeds": SEEDS, "kappas": KAPPAS,
                          "rhos": {str(k): rho(k) for k in KAPPAS},
                          "acc_floor": ACC_FLOOR},
               "arms": [{"d2": d2, "attack": a, "prop1_class": c, "predicted_shape": s,
                         "standalone_published": STANDALONE[(d2, a)][:2]}
                        for d2, a, c, s in ARMS],
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_dose_response.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None or any(a[3] is None for a in ARMS):
        sys.exit("REFUSING TO RUN: predicted shapes are not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash, and copy the predicted shapes into\n"
                 "     ARMS verbatim from that file.\n"
                 "An unfrozen run makes the shapes unfalsifiable, which is the entire point.")


TOL = 1e-6           # float32 training is deterministic here; anything above this is a real drift


def harness_check():
    """kappa=0 is the exact identity (apply_d1_transform returns the update list unwrapped), and the
    participant RNG stream does not depend on d1, so kappa=0 must reproduce d2's published
    standalone value at the SAME seed, to numerical noise. This compares per seed, not against a
    multi-seed mean, so it is a genuine identity check rather than a plausibility check.

    A mismatch is a harness bug or a protocol difference between the suite that produced the
    baseline and this one. Either way it is resolved here, before the ladder runs, not explained
    afterwards -- and note the ladder's own kappa=0 rung is measured fresh at all five seeds, so a
    mismatch does not contaminate the curves; it only tells us the published baseline is not
    comparable and must not be cited as the same quantity."""
    seed = SEEDS[0]
    print(f"=== HARNESS CHECK: kappa=0 (exact identity) vs published standalone, seed {seed} ===")
    print(f"    per-seed comparison, tolerance {TOL:g}\n", flush=True)
    worst, bad = 0.0, []
    for d2, atk, _, _ in ARMS:
        _, _, n, src = STANDALONE[(d2, atk)]
        pub = published_per_seed(d2, atk)
        t = time.time()
        acc, asr = run_one(seed, d2, atk, 0.0)
        if seed not in pub:
            print(f"  {d2:13s} {atk.replace('committed_',''):8s} kappa=0: acc={acc:.3f} "
                  f"ASR={asr:.3f}   NO PUBLISHED SEED {seed} in {src} -- cannot check", flush=True)
            bad.append(f"{d2}/{atk}: no published seed {seed}")
            continue
        p_acc, p_asr = pub[seed]
        d_asr, d_acc = asr - p_asr, acc - p_acc
        ok = max(abs(d_asr), abs(d_acc)) <= TOL
        worst = max(worst, abs(d_asr), abs(d_acc))
        if not ok:
            bad.append(f"{d2}/{atk}: dASR={d_asr:+.4f} dACC={d_acc:+.4f} vs {src}")
        print(f"  {d2:13s} {atk.replace('committed_',''):8s} kappa=0: acc={acc:.4f} ASR={asr:.4f}"
              f"   published s{seed}: acc={p_acc:.4f} ASR={p_asr:.4f}"
              f"   d=({d_acc:+.4f}, {d_asr:+.4f})  {'OK' if ok else 'MISMATCH'}"
              f"  ({time.time()-t:.0f}s)", flush=True)
    print(f"\n  largest absolute deviation: {worst:.2e}  (tolerance {TOL:g})")
    if bad:
        print("  MISMATCHES -- report these, and cite the fresh kappa=0 rung rather than the")
        print("  published baseline for any arm listed here:")
        for b in bad:
            print("   !", b)
    else:
        print("  All four arms reproduce their published standalone value exactly at this seed.")


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        harness_check()
        return 0

    total = len(ARMS) * len(KAPPAS) * len(SEEDS)
    print(f"=== DOSE-RESPONSE LADDER: {len(ARMS)} arms x {len(KAPPAS)} rungs x {len(SEEDS)} "
          f"seeds = {total} runs ===")
    print(f"    shapes frozen in pre_registration_dose_response.md @ {PREREG_COMMIT}")
    print(f"    kappa {KAPPAS}  ->  rho {[round(rho(k), 2) for k in KAPPAS]}\n", flush=True)

    cells = load_cells()
    done_prior = sum(len(c["per_seed"]) for c in cells.values())
    if done_prior:
        print(f"  resuming: {len(cells)} cells, {done_prior} runs already recorded\n", flush=True)
    t0 = time.time(); done = 0
    for d2, atk, cls, shape in ARMS:
        print(f"-- arm: {d2} / {atk.replace('committed_','')}  [Prop 1 {cls}]  "
              f"predicted: {shape}", flush=True)
        for kappa in KAPPAS:
            key = cell_key(d2, atk, kappa)
            cell = cells.get(key, {"d2": d2, "attack": atk, "kappa": kappa, "rho": rho(kappa),
                                   "prop1_class": cls, "predicted_shape": shape, "per_seed": []})
            have = {r["seed"] for r in cell["per_seed"]}
            for seed in SEEDS:
                if seed in have:
                    continue
                t = time.time()
                acc, asr = run_one(seed, d2, atk, kappa)
                cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
                cell["per_seed"].sort(key=lambda r: r["seed"])
                asrs = [r["asr"] for r in cell["per_seed"]]
                accs = [r["accuracy"] for r in cell["per_seed"]]
                cell["mean_asr"] = float(np.mean(asrs)); cell["std_asr"] = float(np.std(asrs, ddof=0))
                cell["mean_acc"] = float(np.mean(accs))
                cells[key] = cell
                save(cells)
                done += 1
                print(f"  [{done}/{total - done_prior}] kappa={kappa} (rho={rho(kappa):.2f}) "
                      f"s{seed}: acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
        print(flush=True)

    print("=== CURVES (per-attack mean ASR, n=5) ===")
    print("    shapes are scored by experiments/analyze_dose_response.py against the frozen")
    print("    decision rules; this table is the raw ladder.\n")
    print(f"  {'arm':26s} {'predicted':14s} " + " ".join(f"{'k='+str(k):>13s}" for k in KAPPAS))
    for d2, atk, cls, shape in ARMS:
        row = []
        for kappa in KAPPAS:
            c = cells.get(cell_key(d2, atk, kappa))
            row.append("na" if c is None else
                       f"{c['mean_asr']:.3f}@{c['mean_acc']:.2f}"
                       + ("!" if c["mean_acc"] < ACC_FLOOR else " "))
        print(f"  {d2 + '/' + atk.replace('committed_',''):26s} {shape:14s} "
              + " ".join(f"{v:>13s}" for v in row))
    print(f"\n  '!' = mean clean accuracy < {ACC_FLOOR}: uninterpretable, not suppression.")
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
