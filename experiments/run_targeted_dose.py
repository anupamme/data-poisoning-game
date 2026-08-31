"""
Targeted-dose test of C2: the intervention Round 11's ladder could not perform.

WHAT WENT WRONG LAST ROUND, IN ONE PARAGRAPH. The Round-11 ladder varied the DISPERSION of a
positive per-client rescaling and assigned the coefficients by a (seed, round)-keyed permutation
uncorrelated with adversary status -- the choice that stopped the instrument from manufacturing a
success. Mean-1 normalization held the aggregate's scale fixed, but it does not hold any individual
client's RELATIVE weight fixed, so a 55x spread mostly dilutes whichever client happens to carry the
poison. Three of the four arms therefore measured Theorem 1's attenuation regime (an adversary
attenuated until it carries no payload) rather than the C2 disturbance the design was meant to
isolate, the pooled prediction was refuted, and we reported it as refuted. This suite separates the
two channels the Round-11 dose confounds.

  MODE S -- statistic-only.  d1 = doseS_kappa<K>. Every adversary pinned at c = 1.0 EXACTLY; the
  benign participants spread over rho = exp(2K), normalized to mean 1 among the benign so the mean
  over all participants is still exactly 1. The adversary's coefficient -- the quantity Lemma 1's
  attenuation channel runs through -- is therefore IDENTICAL at every rung, and a rise in ASR across
  this ladder cannot be attenuation. What varies is the relative weighting of the benign updates,
  which is what the statistics d2 reads are computed from. This is the controlled intervention on
  C2, with the confound closed by construction rather than matched away.

  MODE A -- payload-only.  d1 = doseA_nu<V>. Benign uniform; adversary-to-benign coefficient ratio
  exactly gamma = exp(V); whole vector scaled to mean 1. Sweeping gamma through 1 sweeps the
  adversary through BOTH mechanism-preserving regimes of Theorem 1 at once, which yields a
  prediction this paper has never made before and which no arm of Round 11 could have tested.

BOTH MODES READ ADVERSARY IDENTITY. They are INSTRUMENTS FOR CAUSAL IDENTIFICATION, NOT DEFENSES:
no deployable defense knows which clients are adversarial. Nothing here is proposed for deployment,
and cos_krum remains a mechanism-isolating ablation rather than a proposed defense.

FROZEN PREDICTIONS (copied verbatim from the pre-registration; see the table in ARMS_S / ARMS_A).

  Mode S, class (c) arms  (krum, reputation)   MONOTONE RISE in kappa.
      Refuted by a flat or falling curve -- and a flat curve here is a STRONGER negative than
      Round 11's, because attenuation has been excluded as an explanation.
  Mode S, class (a) arm   (cos_krum)           FLAT (equivalence margin 0.15, every rung < 0.5).
      This is the preservation half of C2's mechanism, which Round 11 left untested: its designated
      control came out indeterminate on a bimodal identity rung, so this arm runs at n=8.
  Mode A, both arms       (krum, coord_median) SINGLE-PEAKED in gamma, maximal near gamma = 1.
      Regime A: an adversary kept extreme enough is discarded by a rank aggregator, so ASR falls as
      gamma grows. Lemma 1 / Regime B: an adversary attenuated far enough carries no payload, so ASR
      falls as gamma shrinks. In between, the payload lands. A MONOTONE curve in either direction
      refutes the two-regime picture, which is the shape of Theorem 1 itself.

If mode A reproduces the fall on the attenuated side and mode S reproduces the rise, Round 11's
refutation is EXPLAINED rather than explained away. If it does not, that is the result and it gets
written as such. No rule below is revised after an outcome; an arm meeting neither its confirmation
nor its refutation criterion is INDETERMINATE, not scored in our favour.

THE IDENTITY RUNGS ARE NOT RE-RUN. doseS_kappa0.0 and doseA_nu0.0 both return the update list
unwrapped -- the same object the Round-11 dose_kappa0.0 rung returned -- and run_one's participant
RNG stream does not depend on d1's name, so those runs are the same computation, not merely the same
quantity. They are imported from results/dose_response/ and marked with their provenance in the
output. --harness-check re-runs them at one seed per arm and asserts bit-equality; if that fails,
the import is invalid and the identity rungs must be re-run.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds, cifar_cnn.
New runs: mode S 15 + 15 + 27 = 57, mode A 20 + 20 = 40, total 97.

Output: results/targeted_dose/summary.json (resumable; written after every run).

DO NOT RUN until experiments/pre_registration_targeted_dose.md is git-committed and PREREG_COMMIT
below is set to that hash. The script refuses to start otherwise.

  python3 experiments/run_targeted_dose.py --harness-check
  python3 experiments/run_targeted_dose.py
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

# experiments/pre_registration_targeted_dose.md, committed before results/targeted_dose/ existed.
PREREG_COMMIT = "5130cec"

KAPPAS = [0.0, 0.5, 1.0, 2.0]                 # mode S dial: rho = exp(2k) = 1.00, 2.72, 7.39, 54.60
NUS = [-2.0, -1.0, 0.0, 1.0, 2.0]             # mode A dial: gamma = exp(nu) = 0.135 .. 7.389
SEEDS5 = [42, 43, 44, 45, 46]
# cos_krum runs at n=8. Its Round-11 identity rung was bimodal across seeds (0.003-0.864, 95% CI
# [0.058, 0.922]), so the equivalence margin its FLAT prediction is tested against never had the
# power that test presumed. The extra seeds are frozen here, before any outcome exists, and apply
# to that arm only.
SEEDS8 = [42, 43, 44, 45, 46, 47, 48, 49]

# (d2, attack, prop1_class, predicted_shape, seeds) -- verbatim from the frozen pre-registration.
ARMS_S = [
    ("krum",         "committed_scaling", "c) not invariant",     "monotone rise", SEEDS5),
    ("reputation",   "committed_scaling", "c) not invariant",     "monotone rise", SEEDS5),
    ("cos_krum",     "committed_pixel",   "a) exactly invariant", "FLAT",          SEEDS8),
]
# krum/scaling was considered for mode A and DROPPED before the freeze, on the pre-freeze
# measurement: its adversarial admission is 0.0833/0.0000/0.0000/0.0000/0.0000 across nu, i.e. Krum
# discards the scaling adversary at every gamma >= 0.368, so mode A has no dose to give it. Recorded
# in the pre-registration rather than quietly omitted.
ARMS_A = [
    ("reputation",   "committed_scaling", "c) not invariant",      "single-peaked at gamma=1", SEEDS5),
    ("coord_median", "committed_pixel",   "b) conditionally inv.", "single-peaked at gamma=1", SEEDS5),
]

FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}
ACC_FLOOR = 0.35              # carried forward unchanged; below this a low ASR is a collapsed model
EQUIV_MARGIN = 0.15           # carried forward unchanged, for the class-(a) FLAT arm

out_dir = os.path.join(base, "results", "targeted_dose")
out_path = os.path.join(out_dir, "summary.json")
LADDER1 = os.path.join(base, "results", "dose_response", "summary.json")


def d1_name(mode, val):
    """The synthetic upstream stage for a rung; formatted so parse_targeted_dose round-trips it."""
    return f"doseS_kappa{val}" if mode == "S" else f"doseA_nu{val}"


def dial(mode, val):
    """The rung's position on its own family's dial: rho for mode S, gamma for mode A. The two are
    NOT the same quantity and are never pooled into one axis."""
    return float(np.exp(2.0 * val)) if mode == "S" else float(np.exp(val))


def run_one(seed, mode, d2, attack_name, val, dataset="cifar10", model="cifar_cnn",
            score_only=False, emit_only=False):
    """One 50-round FL run of the targeted dose into d2 under attack_name.

    dataset/model default to the CIFAR-10 configuration this suite was frozen on, so every existing
    call site is the same computation it always was. The second-dataset replication arm
    (experiments/run_dose_femnist.py) passes ("femnist", "simple_cnn") and reuses this runner rather
    than copying it, because a copy is how two suites drift apart in what they compute.

    score_only=True is the score-only control of pre_registration_score_only.md and emit_only=True
    is its mirror, the fourth cell of the factorial, frozen in pre_registration_emit_only.md. Both
    are passed straight through to generic_compose (see its docstring) and both default to False, so
    no existing call site changes and --harness-check still verifies bit-equality against the frozen
    ladders.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset(dataset, FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model(model, nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(FL_CONFIG.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[attack_name])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(FL_CONFIG.num_clients)]
    d1 = d1_name(mode, val)
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
        # adv_mask is the whole point of this suite: the coefficient a client receives depends on
        # whether it is adversarial. dose_key still seeds the permutation of the BENIGN clients in
        # mode S; mode A needs no permutation, since the assignment is determined by adversary
        # status alone.
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=5.0, dose_key=(seed, rnd),
                                         adv_mask=[bool(cid in adv) for cid in pids],
                                         score_only=score_only, emit_only=emit_only))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return (float(srv.evaluate(td)["accuracy"]),
            float(evaluate_backdoor(srv.global_model, td, device=dev)))


def cell_key(mode, d2, attack, val):
    return f"{d1_name(mode, val)}_then_{d2}|{attack}"


def ladder1_identity(d2, attack):
    """{seed: (accuracy, asr)} for the Round-11 kappa=0 rung of this cell.

    Both identity rungs of this suite ARE that run: apply_d1_transform returns the update list
    unwrapped at kappa=0 / nu=0, and the participant RNG stream does not depend on d1's name, so the
    computation is the same one. Imported rather than repeated, and marked in the output so a reader
    can see which cells are new compute. --harness-check verifies the claim at one seed per arm.
    """
    if not os.path.exists(LADDER1):
        return {}
    cells = json.load(open(LADDER1)).get("cells", {})
    for c in cells.values():
        if c["d2"] == d2 and c["attack"] == attack and c["kappa"] == 0.0:
            return {r["seed"]: (r["accuracy"], r["asr"]) for r in c["per_seed"]}
    return {}


def load_cells():
    if not os.path.exists(out_path):
        return {}
    try:
        return json.load(open(out_path)).get("cells", {})
    except Exception:
        return {}


def save(cells):
    json.dump({"description": "Targeted-dose test of C2 (mode S: statistic-only, adversary pinned "
                              "at c=1; mode A: payload-only, adversary ratio gamma). Shapes frozen "
                              f"at {PREREG_COMMIT} "
                              "(experiments/pre_registration_targeted_dose.md).",
               "prereg_commit": PREREG_COMMIT,
               "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50,
                          "kappas": KAPPAS, "nus": NUS,
                          "seeds_default": SEEDS5, "seeds_cos_krum": SEEDS8,
                          "rhos": {str(k): dial("S", k) for k in KAPPAS},
                          "gammas": {str(v): dial("A", v) for v in NUS},
                          "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN},
               "arms": ([{"mode": "S", "d2": d2, "attack": a, "prop1_class": c,
                          "predicted_shape": s, "seeds": sd} for d2, a, c, s, sd in ARMS_S]
                        + [{"mode": "A", "d2": d2, "attack": a, "prop1_class": c,
                            "predicted_shape": s, "seeds": sd} for d2, a, c, s, sd in ARMS_A]),
               "cells": cells}, open(out_path, "w"), indent=2)


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_targeted_dose.md")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: predicted shapes are not frozen.\n"
                 f"  1. git commit {prereg}\n"
                 "  2. set PREREG_COMMIT here to that hash, and copy the predicted shapes into\n"
                 "     ARMS_S / ARMS_A verbatim from that file.\n"
                 "An unfrozen run makes the shapes unfalsifiable, which is the entire point.")


TOL = 1e-6


def harness_check():
    """Both identity rungs must reproduce the Round-11 kappa=0 rung EXACTLY, per seed.

    This is not a plausibility check. doseS_kappa0.0 and doseA_nu0.0 return the update list
    unwrapped, so at the same seed this is bit-for-bit the same computation as dose_kappa0.0; any
    nonzero deviation means the two suites do not share a code path and the imported identity rungs
    must be discarded and re-run rather than reconciled.
    """
    seed = SEEDS5[0]
    print("=== HARNESS CHECK: targeted identity rungs vs the Round-11 kappa=0 rung ===")
    print(f"    per-seed comparison at seed {seed}, tolerance {TOL:g}\n", flush=True)
    worst, bad = 0.0, []
    for mode, arms, zero in (("S", ARMS_S, 0.0), ("A", ARMS_A, 0.0)):
        for d2, atk, _, _, _ in arms:
            pub = ladder1_identity(d2, atk)
            t = time.time()
            acc, asr = run_one(seed, mode, d2, atk, zero)
            if seed not in pub:
                print(f"  mode {mode} {d2:13s} {atk.replace('committed_',''):8s}: acc={acc:.4f} "
                      f"ASR={asr:.4f}   NO ROUND-11 kappa=0 SEED {seed} -- cannot check", flush=True)
                bad.append(f"mode {mode} {d2}/{atk}: no Round-11 identity seed {seed}")
                continue
            p_acc, p_asr = pub[seed]
            d_asr, d_acc = asr - p_asr, acc - p_acc
            ok = max(abs(d_asr), abs(d_acc)) <= TOL
            worst = max(worst, abs(d_asr), abs(d_acc))
            if not ok:
                bad.append(f"mode {mode} {d2}/{atk}: dASR={d_asr:+.6f} dACC={d_acc:+.6f}")
            print(f"  mode {mode} {d2:13s} {atk.replace('committed_',''):8s}: acc={acc:.4f} "
                  f"ASR={asr:.4f}   Round-11 s{seed}: acc={p_acc:.4f} ASR={p_asr:.4f}"
                  f"   d=({d_acc:+.6f}, {d_asr:+.6f})  {'OK' if ok else 'MISMATCH'}"
                  f"  ({time.time()-t:.0f}s)", flush=True)
    print(f"\n  largest absolute deviation: {worst:.2e}  (tolerance {TOL:g})")
    if bad:
        print("  MISMATCHES -- the identity rungs may NOT be imported; re-run them in this suite:")
        for b in bad:
            print("   !", b)
    else:
        print("  Both identity rungs are the same computation as Round 11's. Import is valid.")


def plan():
    """Every (mode, arm, rung, seed) this suite needs, with the imported identity rungs marked."""
    todo = []
    for mode, arms, vals in (("S", ARMS_S, KAPPAS), ("A", ARMS_A, NUS)):
        for d2, atk, cls, shape, seeds in arms:
            for v in vals:
                for seed in seeds:
                    todo.append((mode, d2, atk, cls, shape, v, seed))
    return todo


def main():
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    if "--harness-check" in sys.argv:
        harness_check()
        return 0

    todo = plan()
    cells = load_cells()

    # Import the identity rungs from Round 11 before counting work, so the printed total is the
    # number of NEW 50-round runs and the reader can see exactly what is being spent.
    imported = 0
    for mode, d2, atk, cls, shape, v, seed in todo:
        if v != 0.0:
            continue
        pub = ladder1_identity(d2, atk)
        if seed not in pub:
            continue
        key = cell_key(mode, d2, atk, v)
        cell = cells.setdefault(key, {"mode": mode, "d2": d2, "attack": atk, "rung": v,
                                      "dial": dial(mode, v), "prop1_class": cls,
                                      "predicted_shape": shape, "per_seed": []})
        if any(r["seed"] == seed for r in cell["per_seed"]):
            continue
        acc, asr = pub[seed]
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr,
                                 "source": "results/dose_response kappa=0 (identical computation)"})
        imported += 1
    todo = [t for t in todo
            if not any(r["seed"] == t[6]
                       for r in cells.get(cell_key(t[0], t[1], t[2], t[5]), {}).get("per_seed", []))]

    print(f"=== TARGETED DOSE: {len(todo)} new runs "
          f"({imported} identity runs imported from Round 11) ===")
    print(f"    shapes frozen in pre_registration_targeted_dose.md @ {PREREG_COMMIT}")
    print(f"    mode S kappa {KAPPAS} -> rho {[round(dial('S', k), 2) for k in KAPPAS]}")
    print(f"    mode A nu    {NUS} -> gamma {[round(dial('A', v), 3) for v in NUS]}\n", flush=True)

    t0 = time.time(); done = 0
    for mode, d2, atk, cls, shape, v, seed in todo:
        key = cell_key(mode, d2, atk, v)
        cell = cells.setdefault(key, {"mode": mode, "d2": d2, "attack": atk, "rung": v,
                                      "dial": dial(mode, v), "prop1_class": cls,
                                      "predicted_shape": shape, "per_seed": []})
        t = time.time()
        acc, asr = run_one(seed, mode, d2, atk, v)
        cell["per_seed"].append({"seed": seed, "accuracy": acc, "asr": asr})
        cell["per_seed"].sort(key=lambda r: r["seed"])
        asrs = [r["asr"] for r in cell["per_seed"]]; accs = [r["accuracy"] for r in cell["per_seed"]]
        cell["mean_asr"] = float(np.mean(asrs)); cell["std_asr"] = float(np.std(asrs, ddof=0))
        cell["mean_acc"] = float(np.mean(accs))
        cells[key] = cell
        save(cells)
        done += 1
        lbl = f"kappa={v}" if mode == "S" else f"nu={v:+.1f}"
        print(f"  [{done}/{len(todo)}] mode {mode} {d2}/{atk.replace('committed_','')} {lbl} "
              f"s{seed}: acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)

    save(cells)
    print("\n=== CURVES (per-attack mean ASR @ mean clean accuracy) ===")
    for mode, arms, vals in (("S", ARMS_S, KAPPAS), ("A", ARMS_A, NUS)):
        d = "kappa" if mode == "S" else "nu"
        print(f"\n  -- mode {mode} --")
        print(f"  {'arm':26s} {'predicted':26s} " + " ".join(f"{d+'='+str(v):>13s}" for v in vals))
        for d2, atk, cls, shape, seeds in arms:
            row = []
            for v in vals:
                c = cells.get(cell_key(mode, d2, atk, v))
                row.append("na" if c is None else
                           f"{c['mean_asr']:.3f}@{c['mean_acc']:.2f}"
                           + ("!" if c["mean_acc"] < ACC_FLOOR else " "))
            print(f"  {d2 + '/' + atk.replace('committed_',''):26s} {shape:26s} "
                  + " ".join(f"{v:>13s}" for v in row))
    print(f"\n  '!' = mean clean accuracy < {ACC_FLOOR}: uninterpretable, not suppression.")
    print("  Shapes are scored by experiments/analyze_targeted_dose.py against the frozen rules;")
    print("  this table is the raw ladder.")
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
