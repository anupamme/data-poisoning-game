"""Arm C: the oracle-free decision contrast on a SECOND dataset and architecture.

WHY THIS ARM EXISTS. A review asked for one intervention that identifies the decision channel
WITHOUT reading adversary identity, on a second dataset, at n >= 20, independently pre-registered.
experiments/pre_registration_oracle_free_femnist.md (committed alone at 9cbe359) is that freeze and
this is its runner. The instrument is not new: it is the already-frozen boundary blend of
experiments/boundary_blend.py, run by experiments/run_oracle_free_decomposition.py under
experiments/pre_registration_oracle_free_decomposition_refined.md (947c000), applied UNCHANGED to
EMNIST-byclass (femnist, 62 classes, 1x28x28) with simple_cnn.

WHAT IS IMPORTED RATHER THAN REIMPLEMENTED, AND WHY THAT IS THE POINT. make_hook and identity_hook
are imported from run_oracle_free_decomposition itself, not copied: they carry the share gate and the
flip gate, and a copy is how two arms drift apart in what they assert. Neither closes over that
arm's cell -- both take only (side, receipts) and read SHARE_TOL, SHARE_GAP_TARGET_DIVISOR and
SHIPPED_BISECT_MAX_STEPS from their imported homes -- so the same two gates fire here, on this
dataset, with the same tolerance and the same refusal to widen it. t_crit comes from the same place
for the same reason. run_one, ACC_FLOOR, EQUIV_MARGIN, ATTACK_MAP, FL_CONFIG and ADV_FRACTION come
from experiments/run_targeted_dose.py; SHARE_TOL from experiments/measure_admission.py, the module
that defines Mode S's own tolerance; SUPPRESS_ASR from experiments/run_dose_resnet18.py:133, which is
where this paper's power-rule threshold already lives. No threshold is restated here. A tolerance
restated in the script that has to pass it is a tolerance chosen to be passed.

WHAT CHANGES, STATED SO NO LATER READING CAN CALL THIS A SINGLE-VARIABLE REPLICATION. Against the
CIFAR-10 arm, dataset, architecture AND attack all change. The attack changes because this paper's
own pre-existing power rule (supplementary.tex:247, recorded there as fixed before that freeze)
admits a cell only where standalone d2 genuinely suppresses -- ASR < SUPPRESS_ASR at clean accuracy
>= ACC_FLOOR -- and on FEMNIST that admits model_scaling_krum (ASR 0.0270) and excludes
backdoor_pixel_krum (0.6601, a ceiling cell). Harness check 3 applies that rule to
results/femnist/payoff_results.json at run time and REFUSES if the artifact does not bear the
freeze's attack choice out. Against results/dose_femnist/summary.json -- the paper's FEMNIST Mode-S
replication -- only the instrument changes, and that is the comparison the review's ask is about.

THE TWO ASR KEY CONVENTIONS, WHICH ARE A LIVE HAZARD HERE. This runner reads two artifacts that do
not agree on the key name: results/femnist/payoff_results.json stores ASR as
"attack_success_rate", while results/dose_femnist/summary.json's per-seed rows store it as "asr".
A silent .get("asr", 0.0) against the payoff matrix would read every ASR as zero and make every cell
look admissible. Each key is named at its own call site and every read is subscripted, never
defaulted, so a renamed key raises instead of quietly returning a passing number.

ORACLE-FREENESS IS STRUCTURAL, NOT INTENDED. The hook is called with (ups, seed, rnd) and never with
the adversary set; run_one passes adv_mask=None whenever d1_override is set, so any dose family
reached this way raises rather than runs; c_rfa comes from the update stack alone; the bisection
compares selections, never labels; and the share gate is the mask-free supremum over all 30 nonempty
proper subsets of the K=5 participants.

WHY self_check IS NOT THE SECOND-DATASET CHECK. experiments/boundary_blend.py's self_check hardcodes
CIFAR-10 and cifar_cnn at its :359-360, and that module is not editable, so it cannot witness the
instrument's endpoint identities on FEMNIST stacks. Harness check 4 runs it unchanged on CIFAR-10 --
which is what witnesses that the instrument is the frozen one rather than a variant -- and harness
check 5 mirrors its four legs on FEMNIST stacks built exactly the way run_one builds them. The
mirror is stated as a mirror, with its source legs named, so nobody reads check 4 as evidence about
this dataset.

TWO SENTENCES MAY NOT BE WRITTEN FROM THIS ARM, whatever it returns, anywhere: that the negative
reproduces oracle-free, and its inverse, that the negative fails without an oracle. Both generalize
beyond one cell, one aggregator, one attack, one dataset and one upstream transform, which is all
this arm has.

    freeze guard + the five checks:  PYTHONPATH=. python3 -m experiments.run_oracle_free_femnist --harness-check
    the two arms:                    PYTHONPATH=. python3 -m experiments.run_oracle_free_femnist
"""

import os, sys, json, time, inspect, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from torch.utils.data import Subset  # noqa: E402
from fl_core import (get_federated_dataset, get_model, FederatedServer,  # noqa: E402
                     FederatedClient)
from attacks import get_attack  # noqa: E402
# Single-sourced from the frozen suite so this arm cannot drift from the published dose arms in the
# runner, the accuracy floor, the equivalence margin, the adversary fraction or the attack mapping.
from experiments.run_targeted_dose import (run_one, ACC_FLOOR, EQUIV_MARGIN,  # noqa: E402
                                           ATTACK_MAP, FL_CONFIG, ADV_FRACTION)
from experiments.run_all_compositions import generic_compose, apply_d1_transform  # noqa: E402
from experiments.adversary_hook import apply_adversary  # noqa: E402
from experiments.boundary_blend import (refined_boundary_pair, apply_coefficients,  # noqa: E402
                                        share_gap_sup, rfa_coefficients, gram_matrix,
                                        gram_krum_selection, blend, adv_share,
                                        GRID_POINTS, H, SHARE_GAP_TARGET_DIVISOR,
                                        SHIPPED_BISECT_MAX_STEPS, self_check)
# The shipped statistic, imported rather than mirrored, so every selection claim below is typed by
# the same function that types every published Krum row in this paper.
from experiments.verify_cos_invariance import krum_selection, flatten  # noqa: E402
# SHARE_TOL is IMPORTED from the module that defines Mode S's own tolerance, never restated here.
from experiments.measure_admission import SHARE_TOL  # noqa: E402
# t_crit IS IMPORTED, NEVER a literal table: the small tables in this repository stop at df = 9.
# make_hook and identity_hook carry the share gate and the flip gate; they are imported from the
# arm that froze them so the two arms cannot diverge in what they assert or in what they refuse.
from experiments.run_oracle_free_decomposition import (make_hook, identity_hook,  # noqa: E402
                                                       t_crit)
# The power rule's suppression threshold already has a module home, at run_dose_resnet18.py:133,
# where its own comment records that it is "the same 0.5 the FEMNIST arm's eligibility rule was
# written against". Imported from there rather than written down again here.
from experiments.run_dose_resnet18 import SUPPRESS_ASR  # noqa: E402

# experiments/pre_registration_oracle_free_femnist.md, committed ALONE before results/
# oracle_free_femnist/ existed.
PREREG_COMMIT = "9cbe359"
# The instrument's own freeze. This arm's provenance story is "the already-frozen instrument, applied
# unchanged", and that story is false if that document has moved, so it is checked rather than
# trusted.
INSTRUMENT_PREREG_COMMIT = "947c000"

DATASET, MODEL = "femnist", "simple_cnn"
D2, ATTACK = "krum", "committed_scaling"
D1 = "fedavg"                                   # pass-through: the blend IS the upstream stage
SEEDS20 = list(range(42, 62))                   # 42-61, frozen in the pre-registration

# (label, side, what it holds) -- the same two arms the instrument's freeze defines.
ARMS = [
    ("A", "low",  "identity-side selection: c(t_lo), the low side of the bisected crossing"),
    ("B", "high", "flipped selection: c(t_hi), with the mask-free supremum share gap from A at or "
                  "below SHARE_TOL/10"),
]
LEGS = [("Delta_decision", "B", "sign: interval EXCLUDES zero")]

# Gate 1 demands bit-identity: the freeze's words are "Anything other than a difference of
# (+0.00e+00, +0.00e+00) means the two harnesses disagree on this dataset and the arm does not run."
# So the pass criterion is EXACT float equality, not a tolerance. A tolerance is recorded beside it
# as a DIAGNOSTIC only -- it is never the gate, and a run that misses exact equality does not
# proceed because it came within 1e-9.
DIAGNOSTIC_TOL = 1e-9

out_dir = os.path.join(base, "results", "oracle_free_femnist")
out_path = os.path.join(out_dir, "summary.json")
PREREG = os.path.join(base, "experiments", "pre_registration_oracle_free_femnist.md")
INSTRUMENT_PREREG = os.path.join(base, "experiments",
                                 "pre_registration_oracle_free_decomposition_refined.md")
# The two artifacts this arm reads. Both are frozen; neither is written.
PAYOFF = os.path.join(base, "results", "femnist", "payoff_results.json")
DOSE_FEMNIST = os.path.join(base, "results", "dose_femnist", "summary.json")
CIFAR10_ARM = os.path.join(base, "results", "oracle_free_decomposition", "summary.json")

# Gate 1's target cell, named here and READ at run time. The numbers are not transcribed.
FROZEN_CELL_KEY = "doseS_kappa0.0_then_krum|committed_scaling"
# The two payoff cells the power rule is applied to. Which one it admits is COMPUTED, not asserted.
ELIGIBLE_CELL, EXCLUDED_CELL = "model_scaling_krum", "backdoor_pixel_krum"

# alpha is run_one's OWN default, one of the three separate homes the freeze names; read off the
# signature rather than written down again, so the live check below cannot drift from the ladder.
ALPHA = inspect.signature(run_one).parameters["alpha"].default


def key_of(arm):
    return f"blend{arm[0]}_then_{D2}|{ATTACK}"


def prereg_md5():
    """The freeze's content hash, written into the artifact so the artifact witnesses WHICH text
    governed the run rather than only which commit was claimed."""
    import hashlib
    return hashlib.md5(open(PREREG, "rb").read()).hexdigest()


def _frozen_at(path, want, what):
    """git log + porcelain for one path. Both halves, because either alone is half a guard."""
    try:
        log = subprocess.run(["git", "log", "-1", "--format=%h", "--", path], cwd=base,
                             capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", path], cwd=base,
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        sys.exit(f"REFUSING TO RUN: cannot verify {what} ({e}).")
    # An EMPTY log must fail loudly. `want.startswith(log[:7])` is vacuously true when log is "", so
    # a path git knows nothing about would otherwise sail through the hash comparison.
    if not log:
        sys.exit(f"REFUSING TO RUN: git has no commit touching {path}. A freeze that git cannot see "
                 "is not a freeze.")
    if not log.startswith(want[:7]) and not want.startswith(log[:7]):
        sys.exit(f"REFUSING TO RUN: {path} was last committed at {log!r}, not {want!r} ({what}).")
    if dirty:
        sys.exit(f"REFUSING TO RUN: {path} has uncommitted modifications ({dirty!r}). The freeze is "
                 "whatever is committed, not whatever is on disk.")


def check_frozen():
    """Refuse to start unless this arm's rules and the instrument's rules are both committed clean."""
    if not os.path.exists(PREREG):
        sys.exit(f"REFUSING TO RUN: {PREREG} does not exist.")
    if PREREG_COMMIT is None:
        sys.exit("REFUSING TO RUN: the frozen predictions and the outcome branches are not "
                 f"committed.\n  1. git commit {PREREG} alone\n"
                 "  2. set PREREG_COMMIT here to that hash.\n"
                 "An unfrozen prediction is unfalsifiable, which is the entire point of the freeze.")
    _frozen_at(PREREG, PREREG_COMMIT, "this arm's freeze")
    # The instrument's freeze. This arm claims to apply it UNCHANGED, so if that document has moved
    # off 947c000 or gone dirty, the claim is false and this arm must not run under it.
    if not os.path.exists(INSTRUMENT_PREREG):
        sys.exit(f"REFUSING TO RUN: {INSTRUMENT_PREREG} is missing. This arm's claim is that it "
                 "applies an ALREADY-FROZEN instrument unchanged; without that document there is no "
                 "instrument to have applied unchanged.")
    _frozen_at(INSTRUMENT_PREREG, INSTRUMENT_PREREG_COMMIT, "the instrument's freeze")
    for p, why in ((PAYOFF, "the standalone FEMNIST payoff matrix the power rule is applied to"),
                   (DOSE_FEMNIST, "the FEMNIST Mode-S replication gate 1 reproduces")):
        if not os.path.exists(p):
            sys.exit(f"REFUSING TO RUN: {p} does not exist ({why}). Both figures are READ at run "
                     "time and neither is transcribed into this runner, so a missing artifact is a "
                     "missing gate rather than a missing convenience.")


def payoff_cell(name):
    """One standalone FEMNIST cell, with ASR read under the key THAT artifact uses.

    Subscripted, never .get()-with-a-default: this file stores ASR as "attack_success_rate" while
    results/dose_femnist/summary.json stores it as "asr", and a silent default of 0.0 here would
    read every ASR as zero and make every cell look admissible by the power rule.
    """
    cells = json.load(open(PAYOFF))
    if name not in cells:
        sys.exit(f"REFUSING TO RUN: {name} is absent from {PAYOFF}. The freeze's attack choice is "
                 "derived from that artifact; it cannot be derived from a cell that is not there.")
    c = cells[name]
    return float(c["accuracy"]), float(c["attack_success_rate"])


def admitted(acc, asr):
    """This paper's own pre-existing power rule, at this paper's own constants, both imported.

    supplementary.tex:247, recorded there as fixed before that freeze: a cell is informative only
    where standalone d2 genuinely suppresses, because a baseline already above SUPPRESS_ASR "has no
    suppression left to lose and will show Delta approximately 0 whatever the transform did: a
    ceiling effect, not preservation".
    """
    return bool(asr < SUPPRESS_ASR and acc >= ACC_FLOOR)


def frozen_target():
    """Gate 1's target pair, read from the artifact at run time. {seed: (accuracy, asr)}."""
    cells = json.load(open(DOSE_FEMNIST)).get("cells", {})
    if FROZEN_CELL_KEY not in cells:
        sys.exit(f"REFUSING TO RUN: {FROZEN_CELL_KEY} is absent from {DOSE_FEMNIST}.")
    return {int(r["seed"]): (float(r["accuracy"]), float(r["asr"]))
            for r in cells[FROZEN_CELL_KEY]["per_seed"]}


def dose_femnist_config():
    """The config gate 1's bit-identity demand rests on.

    Bit-identity is a legitimate PRECONDITION rather than an overreach only because the two
    configurations agree exactly. That agreement is checked here against the artifact instead of
    being asserted in prose: N, K, alpha and the round count must match FL_CONFIG + ADV_FRACTION +
    run_one's own alpha default, and seed 42 must be in the artifact's seed list.
    """
    d = json.load(open(DOSE_FEMNIST))
    cfg = d.get("config", {})
    want = {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
            "f": ADV_FRACTION, "alpha": ALPHA, "rounds": FL_CONFIG.num_rounds}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    return cfg, want, bad


def femnist_pair_check(seed, rounds=3, verbose=True):
    """boundary_blend.self_check's four legs, mirrored on THIS arm's dataset.

    This is a mirror and is labelled one. self_check hardcodes CIFAR-10 and cifar_cnn at
    boundary_blend.py:359-360 and that module is not editable, so it cannot speak about FEMNIST
    stacks; harness check 4 runs it unchanged on CIFAR-10 to witness that the instrument is the
    frozen one, and this runs its legs here. The legs, each asserted against a computation this
    module does not own:

      1. gram_krum_selection at t=0 equals the shipped krum_selection on the raw stack.
      2. gram_krum_selection at t=1 equals the shipped krum_selection on the rfa stack. With 1 this
         pins both ends of the blend path to the real statistic on this dataset.
      3. On a crossing round the pair's selections differ under the SHIPPED statistic, on
         materialized scaled stacks -- gate 2 of the freeze.
      4. c(0) is exactly ones, so the identity end is the identity rather than approximately it.

    Plus the freeze's gate 3, which self_check records and this ASSERTS nowhere: the mask-free
    supremum share gap on every crossing round, compared against SHARE_TOL. It is reported here, and
    the ladder's imported hook raises on it every round of both runs.

    The stacks are built exactly as run_one builds them -- same dataset, model, clean holdout,
    adversary set, attack, alpha and learning-rate schedule -- and the trajectory is advanced through
    this arm's OWN downstream composition (D1 -> D2), not through plain fedavg, so the stacks the
    check sees are the stacks the ladder will see.
    """
    cfg = FL_CONFIG
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset(DATASET, cfg.num_clients, ALPHA, seed)
    srv = FederatedServer(get_model(MODEL, nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
    adv = set(range(int(cfg.num_clients * ADV_FRACTION)))
    atk = get_attack(ATTACK_MAP[ATTACK])
    cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
          for i in range(cfg.num_clients)]
    lr, rows, ok = cfg.learning_rate, [], True
    for rnd in range(rounds):
        pids = np.random.choice(cfg.num_clients,
                                size=min(cfg.clients_per_round, cfg.num_clients), replace=False)
        ups = [cl[cid].train(srv.global_model, cfg.local_epochs, lr, cfg.local_batch_size)
               for cid in pids]
        # verify=True every round here, not only round 0: this check is three rounds long and the
        # clone it costs is one probe client's update. model_scaling OVERRIDES manipulate_update, so
        # a dead adversary hook is caught rather than assumed away.
        ups = apply_adversary(ups, pids, adv, atk, srv.global_model, ca=None, verify=True)
        adv_rows = [i for i, cid in enumerate(pids) if cid in adv]
        c_rfa, raw = rfa_coefficients(ups)
        G = gram_matrix(raw)
        sel_raw_shipped, _ = krum_selection(raw, False)
        sel_raw_gram, _ = gram_krum_selection(G, blend(c_rfa, 0.0))
        rfa_stack = flatten(apply_d1_transform(ups, "rfa", tau=5.0, dose_key=None, adv_mask=None))
        sel_rfa_shipped, _ = krum_selection(rfa_stack, False)
        sel_rfa_gram, _ = gram_krum_selection(G, blend(c_rfa, 1.0))
        pair = refined_boundary_pair(ups)
        sel_a, _ = krum_selection(flatten(apply_coefficients(ups, pair["c_A"])), False)
        sel_b, _ = krum_selection(flatten(apply_coefficients(ups, pair["c_B"])), False)
        sup, arg = share_gap_sup(pair["c_A"], pair["c_B"])
        r = {"seed": int(seed), "round": int(rnd), "dataset": DATASET, "model": MODEL,
             "t0_gram_equals_shipped_raw": bool(sel_raw_gram == sel_raw_shipped),
             "t1_gram_equals_shipped_rfa": bool(sel_rfa_gram == sel_rfa_shipped),
             "identity_is_exact_ones": bool(np.all(blend(c_rfa, 0.0) == 1.0)),
             "flip": bool(pair["flip"]), "t_star": pair["t_star"],
             "coeff_gap_linf": float(pair["coeff_gap_linf"]),
             "sel_A_shipped": int(sel_a), "sel_B_shipped": int(sel_b),
             "pair_selections_differ_under_shipped_statistic": bool(sel_a != sel_b),
             "share_gap_sup": float(sup) if sup == sup else float("nan"),
             "argmax_subset": arg,
             "sup_within_tol": bool(sup <= SHARE_TOL) if sup == sup else True,
             "n_bisect_steps": pair["n_bisect_steps"],
             "n_shipped_evaluations": pair["n_shipped_evaluations"],
             "rho_rfa": float(np.max(c_rfa) / max(np.min(c_rfa), 1e-12)),
             "n_adv_in_round": len(adv_rows),
             # Recorded beside the supremum, never instead of it: a realized gap can pass while the
             # supremum fails, which means neutral for the mask this round happened to draw rather
             # than neutral. The construction never reads adv_rows; only this measurement does.
             "realized_share_gap": (abs(adv_share(pair["c_B"], adv_rows)
                                        - adv_share(pair["c_A"], adv_rows))
                                    if adv_rows else float("nan"))}
        legs = [r["t0_gram_equals_shipped_raw"], r["t1_gram_equals_shipped_rfa"],
                r["identity_is_exact_ones"]]
        if r["flip"]:
            legs += [r["pair_selections_differ_under_shipped_statistic"], r["sup_within_tol"]]
        r["all_legs_pass"] = bool(all(legs))
        ok = ok and r["all_legs_pass"]
        rows.append(r)
        if verbose:
            print(f"    seed {seed} rnd {rnd}: flip={r['flip']} t*={r['t_star']} "
                  f"gram==shipped t0/t1: {r['t0_gram_equals_shipped_raw']}/"
                  f"{r['t1_gram_equals_shipped_rfa']}  sel A/B={r['sel_A_shipped']}/"
                  f"{r['sel_B_shipped']}  sup share gap={r['share_gap_sup']:.3e} vs tol "
                  f"{SHARE_TOL:.0e}  {r['n_bisect_steps']} bisect steps", flush=True)
        srv.apply_update(generic_compose(srv, ups, D1, D2, tau=5.0, dose_key=(seed, rnd),
                                         adv_mask=None))
        lr *= getattr(cfg, "lr_decay", 1.0)
    flips = [r for r in rows if r["flip"]]
    sups = [r["share_gap_sup"] for r in flips if r["share_gap_sup"] == r["share_gap_sup"]]
    return {"all_pass": bool(ok), "n_rows": len(rows), "n_flip_rounds": len(flips),
            "dataset": DATASET, "model": MODEL, "attack": ATTACK, "pair_fn": "refined_boundary_pair",
            "mirrors": "experiments/boundary_blend.py::self_check, whose dataset is hardcoded to "
                       "cifar10/cifar_cnn at :359-360 and which therefore cannot speak about this "
                       "dataset",
            "max_share_gap_sup_on_flip_rounds": float(max(sups)) if sups else float("nan"),
            "all_flip_rounds_within_tol_on_supremum": (bool(all(s <= SHARE_TOL for s in sups))
                                                       if sups else None),
            "all_flip_rounds_differ_under_shipped": (
                bool(all(r["pair_selections_differ_under_shipped_statistic"] for r in flips))
                if flips else None),
            "share_tol": SHARE_TOL, "per_round": rows}


def harness_check():
    """The freeze's gates that can be scored BEFORE any ASR exists, with verdicts RETURNED.

    Five checks. Gate 1 is checks 1 and 2, gate 2 and gate 3 are check 5 (and are additionally
    asserted every round of both runs by the imported hook), and check 3 witnesses the freeze's
    attack choice against the artifact it was derived from. A check whose verdict is discarded
    leaves the claim it supports witnessed only by a terminal, and re-running this one costs about
    an hour of FEMNIST compute.
    """
    v = {"diagnostic_tol": DIAGNOSTIC_TOL, "share_tol": SHARE_TOL, "grid_points": GRID_POINTS,
         "h": H, "dataset": DATASET, "model": MODEL, "checks": [], "all_passed": None}
    tgt = frozen_target()
    seed = sorted(tgt)[0]
    p_acc, p_asr = tgt[seed]

    print("=== HARNESS CHECK 1/5: gate 1, bit-identity against the published FEMNIST kappa=0 cell ===")
    cfg, want, bad = dose_femnist_config()
    print(f"    config agreement, checked not asserted: artifact {cfg.get('N')}/{cfg.get('K')}/"
          f"{cfg.get('f')}/{cfg.get('alpha')}/{cfg.get('rounds')} vs this suite "
          f"{want['N']}/{want['K']}/{want['f']}/{want['alpha']}/{want['rounds']}"
          + ("  AGREE" if not bad else f"  DISAGREE: {bad}"))
    seed_ok = seed in tgt
    print(f"    {FROZEN_CELL_KEY} at seed {seed}: published acc={p_acc:.16f} ASR={p_asr:.16f}")
    print("    Recomputed through run_one at this dataset and model with NO hook. The freeze demands "
          "a\n    difference of (+0.00e+00, +0.00e+00): the pass criterion is exact equality, and a "
          "near miss\n    does not proceed.", flush=True)
    t = time.time()
    acc, asr = run_one(seed, "S", D2, ATTACK, 0.0, dataset=DATASET, model=MODEL)
    d = (acc - p_acc, asr - p_asr)
    exact = bool(acc == p_acc and asr == p_asr)
    ok1 = bool(exact and not bad and seed_ok)
    print(f"    recomputed: acc={acc:.16f} ASR={asr:.16f}")
    print(f"    d=({d[0]:+.2e}, {d[1]:+.2e})  -> {'BIT-IDENTICAL' if exact else 'NOT IDENTICAL'}"
          f"   (diagnostic only, not the gate: within {DIAGNOSTIC_TOL:g}? "
          f"{max(abs(d[0]), abs(d[1])) <= DIAGNOSTIC_TOL})  ({time.time() - t:.0f}s)", flush=True)
    v["checks"].append({"check": "gate1_bit_identity_against_published_femnist_identity_rung",
                        "passed": ok1, "cell": FROZEN_CELL_KEY, "seed": seed,
                        "source": "results/dose_femnist/summary.json",
                        "published": [p_acc, p_asr], "recomputed": [float(acc), float(asr)],
                        "delta": [float(d[0]), float(d[1])], "bit_identical": exact,
                        "within_diagnostic_tol": bool(max(abs(d[0]), abs(d[1]))
                                                      <= DIAGNOSTIC_TOL),
                        "config_agreement": {"artifact": cfg, "this_suite": want,
                                             "disagreements": bad},
                        "seed_present_in_artifact": bool(seed_ok),
                        "criterion": "exact float equality; the freeze's words are a difference of "
                                     "(+0.00e+00, +0.00e+00). The diagnostic tolerance is recorded "
                                     "beside the verdict and is never the verdict."})

    print("\n=== HARNESS CHECK 2/5: the hook at c = 1 reproduces the same published pair ===")
    print("    Same target, reached the OTHER way: d1_override='fedavg' with a hook that multiplies "
          "every\n    client by exactly 1.0, which is exact in float32. If the hook path is entered "
          "and the result\n    still matches, the path is faithful on this dataset; if the hook is "
          "never called the run\n    completes silently and arm B equals arm A. That silent "
          "collapse is the standing failure mode\n    in this repository.", flush=True)
    rec = []
    t = time.time()
    acc2, asr2 = run_one(seed, None, D2, ATTACK, None, dataset=DATASET, model=MODEL,
                         d1_override=D1, stack_hook=identity_hook(rec))
    d2_ = (acc2 - p_acc, asr2 - p_asr)
    entered = len(rec) == FL_CONFIG.num_rounds
    exact2 = bool(acc2 == p_acc and asr2 == p_asr)
    ok2 = bool(exact2 and entered)
    print(f"    recomputed: acc={acc2:.16f} ASR={asr2:.16f}   d=({d2_[0]:+.2e}, {d2_[1]:+.2e})")
    print(f"    hook invoked {len(rec)} times for {FL_CONFIG.num_rounds} rounds -> "
          f"{'ENTERED' if entered else 'NEVER CALLED'}   "
          f"{'BIT-IDENTICAL' if exact2 else 'NOT IDENTICAL'}  ({time.time() - t:.0f}s)", flush=True)
    v["checks"].append({"check": "identity_hook_reproduces_published_femnist_identity_rung",
                        "passed": ok2, "seed": seed, "published": [p_acc, p_asr],
                        "recomputed": [float(acc2), float(asr2)],
                        "delta": [float(d2_[0]), float(d2_[1])], "bit_identical": exact2,
                        "within_diagnostic_tol": bool(max(abs(d2_[0]), abs(d2_[1]))
                                                      <= DIAGNOSTIC_TOL),
                        "hook_invocations": len(rec), "rounds": FL_CONFIG.num_rounds,
                        "hook_entered_every_round": bool(entered),
                        "why": "Mode S at kappa=0 and d1_override='fedavg' with c = 1 are the same "
                               "computation -- apply_d1_transform returns the stack unwrapped at "
                               "kappa=0, fedavg is a pass-through, and the participant RNG stream "
                               "does not depend on d1's name -- so both must land on the published "
                               "pair. Check 1 tests the ladder's call shape; this tests the hook "
                               "path."})

    print("\n=== HARNESS CHECK 3/5: the freeze's ATTACK CHOICE, recomputed from the payoff matrix ===")
    print("    This paper's own power rule, at this paper's own imported constants: a cell is "
          "admitted only\n    where standalone d2 suppresses, ASR < SUPPRESS_ASR at accuracy >= "
          "ACC_FLOOR. The freeze says\n    the rule admits exactly one FEMNIST Krum cell and that "
          "this is why the attack differs from the\n    CIFAR-10 arm's. Here the rule is applied "
          "rather than quoted.", flush=True)
    rows = []
    for name in (ELIGIBLE_CELL, EXCLUDED_CELL):
        a, s = payoff_cell(name)
        rows.append({"cell": name, "accuracy": a, "attack_success_rate": s,
                     "admitted": admitted(a, s)})
        print(f"    {name:24s} acc={a:.4f} ASR={s:.4f}  -> "
              f"{'ADMITTED' if admitted(a, s) else 'EXCLUDED'}")
    ok3 = bool(rows[0]["admitted"] and not rows[1]["admitted"])
    print(f"    -> {'the rule bears the freeze out' if ok3 else 'THE RULE DOES NOT BEAR THE FREEZE OUT'}"
          f"   (suppress<{SUPPRESS_ASR}, acc>={ACC_FLOOR}, both imported)")
    v["checks"].append({"check": "power_rule_admits_the_frozen_attack_and_excludes_the_other",
                        "passed": ok3, "source": "results/femnist/payoff_results.json",
                        "asr_key_in_this_artifact": "attack_success_rate",
                        "suppress_asr": SUPPRESS_ASR, "acc_floor": ACC_FLOOR, "cells": rows,
                        "rule": "supplementary.tex:247, recorded there as fixed before that "
                                "freeze; SUPPRESS_ASR imported from run_dose_resnet18.py:133 and "
                                "ACC_FLOOR from run_targeted_dose.py, neither restated here."})

    print("\n=== HARNESS CHECK 4/5: the instrument is the frozen one, witnessed on CIFAR-10 ===")
    print("    boundary_blend.self_check, imported and run UNCHANGED, on refined_boundary_pair. Its "
          "dataset\n    is hardcoded to cifar10/cifar_cnn at :359-360, so this is evidence that the "
          "instrument is\n    unmodified and is NOT evidence about FEMNIST. Check 5 is the "
          "second-dataset evidence.", flush=True)
    sc = self_check(seeds=(SEEDS20[0],), rounds=3, verbose=True, pair_fn=refined_boundary_pair)
    ok4 = bool(sc["all_pass"]
               and sc["all_flip_rounds_within_tol_on_supremum"] is not False
               and sc["all_flip_rounds_differ_under_shipped"] is not False)
    print(f"    -> {'PASSES on cifar10' if ok4 else 'FAILS on cifar10'}  "
          f"({sc['n_flip_rounds']}/{sc['n_rows']} crossing rounds, max supremum "
          f"{sc['max_share_gap_sup_on_flip_rounds']:.3e} vs tol {SHARE_TOL:.0e})")
    v["checks"].append({"check": "instrument_unchanged_self_check_on_cifar10", "passed": ok4,
                        "scope": "cifar10/cifar_cnn only -- self_check's dataset is hardcoded at "
                                 "boundary_blend.py:359-360. This check witnesses that the "
                                 "instrument is the frozen one; it says nothing about femnist.",
                        "self_check": sc})

    print("\n=== HARNESS CHECK 5/5: gates 2 and 3 on FEMNIST stacks, before 40 runs are spent ===")
    print("    self_check's four legs mirrored on this arm's dataset, plus the freeze's share gate. "
          "The\n    ladder's hook asserts both every round of both runs; this measures them first, "
          "because a\n    per-round assertion that fires at run 7 of 40 has already spent the "
          "compute.", flush=True)
    live = femnist_pair_check(SEEDS20[0], rounds=3, verbose=True)
    ok5 = bool(live["all_pass"]
               and live["all_flip_rounds_within_tol_on_supremum"] is not False
               and live["all_flip_rounds_differ_under_shipped"] is not False)
    print(f"    -> {'PASSES on femnist' if ok5 else 'FAILS on femnist'}  "
          f"({live['n_flip_rounds']}/{live['n_rows']} crossing rounds, max supremum "
          f"{live['max_share_gap_sup_on_flip_rounds']:.3e} vs tol {SHARE_TOL:.0e})")
    if live["n_flip_rounds"] == 0:
        print("    NOTE: zero crossing rounds in this sample is not a pass and not a failure of the "
              "gates --\n    it is an absence of dose. n_flip_rounds is recorded per run and "
              "reported as measured.")
    v["checks"].append({"check": "gates_2_and_3_live_on_femnist", "passed": ok5,
                        "scope": "femnist/simple_cnn, this arm's own dataset and attack",
                        "live": live})

    v["all_passed"] = all(c["passed"] for c in v["checks"])
    print("\n  " + ("ALL FIVE CHECKS PASSED. The two arms may run."
                    if v["all_passed"] else
                    "A CHECK FAILED. The arms must NOT run; the failure is the result, and no "
                    "threshold is adjusted to clear it."))
    return v


def load():
    if not os.path.exists(out_path):
        return {}, None
    try:
        d = json.load(open(out_path))
        return d.get("cells", {}), d.get("harness_check")
    except Exception:
        return {}, None


def paired(cells, label):
    arm = next(a for a in ARMS if a[0] == label)
    rows = cells.get(key_of(arm), {}).get("per_seed", [])
    return ({int(r["seed"]): float(r["asr"]) for r in rows},
            {int(r["seed"]): float(r["accuracy"]) for r in rows})


def leg(cells, label):
    """The paired B - A difference, with all THREE verdicts, from per-seed rows in one call.

    Both legs of the difference are read here, in this call, so a subtrahend can never come from a
    different seed count -- or a different arm -- than its minuend. Nothing is transcribed.

    Three verdicts, computed independently, and none may be reported as another: an interval can sit
    inside the margin AND exclude zero, and both arms can sit against the same boundary while the
    interval excludes zero on a difference that had nowhere to move.
    """
    b_asr, b_acc = paired(cells, label)
    a_asr, a_acc = paired(cells, "A")
    seeds = sorted(set(a_asr) & set(b_asr))
    r = {"arm": label, "minuend": label, "subtrahend": "A", "n": len(seeds), "seeds": seeds,
         "arm_mean_asr": float(np.mean([b_asr[s] for s in seeds])) if seeds else None,
         "reference_mean_asr": float(np.mean([a_asr[s] for s in seeds])) if seeds else None,
         "arm_mean_acc": float(np.mean([b_acc[s] for s in seeds])) if seeds else None,
         "arm_min_acc": float(min(b_acc[s] for s in seeds)) if seeds else None,
         "reference_mean_acc": float(np.mean([a_acc[s] for s in seeds])) if seeds else None,
         "reference_min_acc": float(min(a_acc[s] for s in seeds)) if seeds else None,
         "per_seed_delta": {str(s): b_asr[s] - a_asr[s] for s in seeds}}
    if len(seeds) < 2:
        r.update({"mean": None, "sd": None, "ci95": None, "verdict_sign": "NOT COMPUTED",
                  "verdict_margin": "NOT COMPUTED", "verdict_room": "NOT COMPUTED",
                  "acc_gate": "NOT COMPUTED"})
        return r
    d = np.array([b_asr[s] - a_asr[s] for s in seeds], dtype=float)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    hw = t_crit(len(d)) * sd / np.sqrt(len(d))
    lo, hi = m - hw, m + hw
    # Gate 6 of the freeze, on the EXISTING EQUIV_MARGIN and with no new constant: if both arms land
    # within the margin of the SAME boundary, the contrast had no room to move and the result is a
    # ceiling or floor artifact rather than evidence in either direction.
    both_low = r["arm_mean_asr"] <= EQUIV_MARGIN and r["reference_mean_asr"] <= EQUIV_MARGIN
    both_high = (r["arm_mean_asr"] >= 1.0 - EQUIV_MARGIN
                 and r["reference_mean_asr"] >= 1.0 - EQUIV_MARGIN)
    r.update({"mean": m, "sd": sd, "se": float(sd / np.sqrt(len(d))), "t_crit": t_crit(len(d)),
              "ci95": [lo, hi], "half_width": float(hw),
              "verdict_sign": ("ASR MOVES on this leg (interval excludes zero), sign "
                               + ("positive" if m > 0 else "negative"))
                              if lo * hi > 0 else
                              "NOT RESOLVED at this n (interval contains zero); not evidence of "
                              "absence",
              "verdict_margin": (f"within the frozen equivalence margin (|mean| < {EQUIV_MARGIN})"
                                 if abs(m) < EQUIV_MARGIN else
                                 f"outside the frozen margin (|mean| >= {EQUIV_MARGIN})"),
              "margin": EQUIV_MARGIN,
              "verdict_room": ("FLOOR ARTIFACT: both arms within EQUIV_MARGIN of ASR 0, so the "
                               "contrast had no room to move; NOT evidence in either direction"
                               if both_low else
                               "CEILING ARTIFACT: both arms within EQUIV_MARGIN of ASR 1, so the "
                               "contrast had no room to move; NOT evidence in either direction"
                               if both_high else
                               "room to move: the arms are not both against the same boundary"),
              "acc_gate": ("PASS" if (r["arm_mean_acc"] >= ACC_FLOOR
                                      and r["reference_mean_acc"] >= ACC_FLOOR) else
                           "FAIL -- this arm is VOID, not negative: a low ASR at collapsed accuracy "
                           "is not suppression"),
              "acc_floor": ACC_FLOOR})
    return r


def dose_summary(cells):
    """What the instrument actually delivered, counted from receipts rather than assumed.

    A round with no crossing is bit-identical across arms and carries no dose. The freeze's gate 4
    says n_flip_rounds is recorded per run and not assumed; this is that record. The CIFAR-10 arm
    delivered means of 28.6 (A) and 27.9 (B) of 50, and that is a fact about CIFAR-10, not a
    prediction for here.
    """
    out = {}
    for a in ARMS:
        rows = cells.get(key_of(a), {}).get("per_seed", [])
        flips = [r.get("n_flip_rounds") for r in rows if r.get("n_flip_rounds") is not None]
        gaps = [r.get("max_share_gap_sup") for r in rows if r.get("max_share_gap_sup") is not None]
        cg = [r.get("max_coeff_gap_linf") for r in rows if r.get("max_coeff_gap_linf") is not None]
        out[a[0]] = {
            "n_runs": len(rows),
            "mean_flip_rounds_per_run": float(np.mean(flips)) if flips else None,
            "min_flip_rounds_per_run": int(min(flips)) if flips else None,
            "max_flip_rounds_per_run": int(max(flips)) if flips else None,
            "rounds_per_run": FL_CONFIG.num_rounds,
            "max_share_gap_sup_over_all_runs": float(max(gaps)) if gaps else None,
            "max_coeff_gap_linf_over_all_runs": float(max(cg)) if cg else None,
            "share_tol": SHARE_TOL,
        }
    return out


def cifar10_prior():
    """The published CIFAR-10 reading of this same estimand, READ rather than transcribed.

    Quoted by the freeze for reference and explicitly NOT as a prediction. One dataset is not a prior
    for another, and this arm's attack differs as well, so the pair is printed beside this arm's
    numbers and never subtracted from them.
    """
    if not os.path.exists(CIFAR10_ARM):
        return None
    v = json.load(open(CIFAR10_ARM)).get("verdicts", {}).get("Delta_decision")
    if not v:
        return None
    return {"source": "results/oracle_free_decomposition/summary.json",
            "dataset": "cifar10", "model": "cifar_cnn", "attack": "committed_pixel",
            "n": v.get("n"), "mean": v.get("mean"), "ci95": v.get("ci95"),
            "verdict_sign": v.get("verdict_sign"),
            "not_a_prediction": "The freeze quotes this for reference only. Dataset, architecture "
                                "and attack all differ from this arm, so it bounds nothing here."}


def save(cells, hc):
    verdicts = {name: leg(cells, arm) for name, arm, _ in LEGS}
    json.dump({
        "description":
            "Krum's decision channel moved WITHOUT adversary identity, with the attenuation channel "
            "held to Mode S's own tolerance, on a SECOND dataset and architecture: femnist "
            "(EMNIST-byclass, 62 classes) with simple_cnn. Arms straddle a Krum selection crossing "
            "on the blend c(t) = 1 + t (c_rfa - 1), located by bisecting the SHIPPED float32 "
            "krum_selection: arm A at c(t_lo), arm B at c(t_hi). The instrument is the one frozen at "
            f"{INSTRUMENT_PREREG_COMMIT}, applied unchanged; the rules for THIS arm are frozen at "
            f"{PREREG_COMMIT} (experiments/pre_registration_oracle_free_femnist.md).",
        "prereg_commit": PREREG_COMMIT,
        "prereg": "experiments/pre_registration_oracle_free_femnist.md",
        "prereg_md5": prereg_md5(),
        "instrument": {
            "module": "experiments/boundary_blend.py",
            "runner_it_was_frozen_for": "experiments/run_oracle_free_decomposition.py",
            "prereg": "experiments/pre_registration_oracle_free_decomposition_refined.md",
            "commit": INSTRUMENT_PREREG_COMMIT,
            "applied_unchanged": True,
            "hooks_imported_not_copied": "make_hook and identity_hook are imported from "
                                         "experiments/run_oracle_free_decomposition.py, so the "
                                         "share gate and the flip gate are the same code with the "
                                         "same tolerance. A copy is how two arms drift apart in "
                                         "what they assert."},
        "dataset": DATASET, "model": MODEL,
        "what_changed_relative_to_the_cifar10_arm":
            "dataset, architecture AND attack. The attack changes because this paper's own "
            "pre-existing power rule (supplementary.tex:247) admits only one FEMNIST Krum cell: "
            "model_scaling_krum at ASR 0.0270, while backdoor_pixel_krum at 0.6601 is a ceiling "
            "cell. Harness check 3 applies that rule to results/femnist/payoff_results.json rather "
            "than quoting it.",
        "what_changed_relative_to_the_femnist_mode_s_arm":
            "only the instrument. Same dataset, architecture, aggregator and attack as "
            "results/dose_femnist/summary.json; Mode S's oracle-dependent pinning is replaced by "
            "the oracle-free blend. That is the comparison this arm is for.",
        "config": {"N": FL_CONFIG.num_clients, "K": FL_CONFIG.clients_per_round,
                   "f": ADV_FRACTION, "alpha": ALPHA, "rounds": FL_CONFIG.num_rounds,
                   "seeds": SEEDS20,
                   "acc_floor": ACC_FLOOR, "equiv_margin": EQUIV_MARGIN, "share_tol": SHARE_TOL,
                   "suppress_asr": SUPPRESS_ASR,
                   "share_gap_target_divisor": SHARE_GAP_TARGET_DIVISOR,
                   "share_gap_target": SHARE_TOL / float(SHARE_GAP_TARGET_DIVISOR),
                   "bisect_max_steps": SHIPPED_BISECT_MAX_STEPS,
                   "coarse_proposal_grid_points": GRID_POINTS, "coarse_proposal_h": H,
                   "t_crit_n20": t_crit(len(SEEDS20)),
                   "constant_homes": "FL_CONFIG carries N, K and rounds only; ADV_FRACTION is a "
                                     "separate module constant; alpha is run_one's own default and "
                                     "is read off its signature here rather than written down. "
                                     "Three homes, none of them restated."},
        "arm": {"d2": D2, "attack": ATTACK, "attack_impl": ATTACK_MAP[ATTACK], "d1": D1,
                "arms": [{"label": a[0], "side": a[1], "holds": a[2], "key": key_of(a)}
                         for a in ARMS],
                "estimand": "Delta_decision = mean ASR(arm B) - mean ASR(arm A), paired per seed "
                            "over seeds 42-61, n = 20, two-sided 95% t interval at t_crit(20).",
                "outcome_branches": {
                    "excludes_zero_positive": "Krum's decision channel moves ASR on a second "
                                              "dataset and architecture without adversary identity "
                                              "and with the coefficient share held to SHARE_TOL. "
                                              "The licensed statement is the arm's own scope.",
                    "contains_zero": "On this cell the oracle-free decision contrast does not move "
                                     "ASR. The CIFAR-10 +0.3218 is then scoped to CIFAR-10 in the "
                                     "body, not in a footnote, and tab:evidence's oracle-free row "
                                     "is restated at the scope the evidence licenses.",
                    "excludes_zero_negative": "Reported as measured, with the direction stated, and "
                                              "NOT scored in our favour.",
                    "any_gate_fires": "No ASR verdict. The arm is reported as attempted and "
                                      "uninterpretable, naming the gate that fired and its measured "
                                      "value. A gate failure is not converted into a finding, and "
                                      "no threshold is adjusted to clear it."}},
        "cifar10_reference": cifar10_prior(),
        "oracle_free": "The hook is called with (ups, seed, rnd) and never with the adversary set. "
                       "c_rfa comes from the update stack alone, the bisection compares selections "
                       "rather than labels, run_one passes adv_mask=None whenever d1_override is "
                       "set so any dose family reached this way RAISES, and the share gate is the "
                       "mask-free supremum over every adversary subset rather than a realized "
                       "share. Oracle-freeness is enforced by the callee, not intended by the "
                       "caller.",
        "share_gate": {
            "implemented_as": "within-run counterfactual pair, supremum over every nonempty proper "
                              "adversary subset, asserted every round of both runs by the imported "
                              "hook, which raises rather than recording a violation",
            "stopping_rule": "The bisection stops when the supremum share gap is at or below "
                             "SHARE_TOL / SHARE_GAP_TARGET_DIVISOR. The step cap is a COST bound: a "
                             "round that exhausts it without reaching the target FAILS the gate and "
                             "raises. The tolerance is never widened to admit a round.",
            "share_tol": SHARE_TOL},
        "theory_scope": "App. D.6's impossibility -- no oracle-free positive per-client rescaling is "
                        "both EXACTLY share-neutral and informative -- is not contradicted. This "
                        "instrument is the APPROXIMATE boundary blend and its gap is positive, "
                        "which is why the share gate is a premise rather than a diagnostic: were "
                        "the instrument exact, this arm would be impossible by the paper's own "
                        "theorem.",
        "not_a_defense": "No deployed Krum interpolates its input stack toward a geometric-median "
                         "reweighting and stops one bisection step short of a selection crossing. "
                         "This arm is an instrument: it answers the adversary-identity objection "
                         "and not the deployability one.",
        "scope": "One aggregator, one attack, one dataset, one architecture, one upstream "
                 "transform. The contrast is LOCAL, between two adjacent points on a blend path "
                 "either side of a selection crossing, and must not be reported as identity "
                 "against a statistic disturbance.",
        "may_not_be_written": "Neither the sentence that the negative reproduces oracle-free nor "
                              "its inverse, that the negative fails without an oracle, may be "
                              "written anywhere under any outcome: in the paper, in the supplement, "
                              "in any artifact or in any response letter.",
        "one_existing_statement_stays_true":
            "results/oracle_free_decomposition/summary.json records that no composed-pair ASR "
            "result on a second dataset exists anywhere in this paper and that that arm does not "
            "change it. This arm does not change it either: its d1 is a synthetic coefficient "
            "vector, not a second real defense, so it is not a composed pair in that sentence's "
            "sense. That artifact is frozen and is not amended.",
        "harness_check": hc,
        "dose_delivered": dose_summary(cells),
        "verdicts": verdicts,
        "verdict_reading_rule": "verdict_sign, verdict_margin and verdict_room are independent "
                                "labels and none may be reported as another: an interval can sit "
                                "inside the margin and still exclude zero, and both arms can sit "
                                "against the same boundary while the difference excludes zero. No "
                                "literal names a mechanism; the measured direction is printed "
                                "beside it, never inside it.",
        "cells": cells}, open(out_path, "w"), indent=2)


def report(cells):
    print("\n=== THE TWO ARMS (paired by seed) ===")
    for a in ARMS:
        c = cells.get(key_of(a))
        rows = c["per_seed"] if c else []
        if not rows:
            print(f"  arm {a[0]} (no runs yet)")
            continue
        asr = [r["asr"] for r in rows]; acc = [r["accuracy"] for r in rows]
        print(f"  arm {a[0]}  n={len(rows):2d}  mean ASR={np.mean(asr):.4f}  "
              f"mean acc={np.mean(acc):.4f}  min acc={min(acc):.4f}"
              + ("   * ARM MEAN BELOW ACC FLOOR: VOID, not negative"
                 if np.mean(acc) < ACC_FLOOR else ""))
    print("\n=== THE DOSE THE INSTRUMENT DELIVERED (counted from receipts) ===")
    for k, s in dose_summary(cells).items():
        if not s["n_runs"]:
            continue
        print(f"  arm {k}: crossing rounds per run mean {s['mean_flip_rounds_per_run']:.1f} "
              f"[{s['min_flip_rounds_per_run']}, {s['max_flip_rounds_per_run']}] of "
              f"{s['rounds_per_run']}; max supremum share gap "
              f"{s['max_share_gap_sup_over_all_runs']:.3e} vs tol {s['share_tol']:.0e}")
    print("\n=== THE LEG, ALL THREE VERDICTS, RECOMPUTED FROM PER-SEED ROWS ===")
    for name, arm, pred in LEGS:
        r = leg(cells, arm)
        if r["mean"] is None:
            print(f"  {name} (arm {arm} - arm A): n={r['n']}, not computable yet")
            continue
        print(f"  {name} (arm {arm} - arm A), frozen prediction: {pred}")
        print(f"    n={r['n']}  mean={r['mean']:+.4f}  sd={r['sd']:.4f}  "
              f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
        print(f"    arms: A mean ASR={r['reference_mean_asr']:.4f}  "
              f"B mean ASR={r['arm_mean_asr']:.4f}")
        print(f"    SIGN  : {r['verdict_sign']}")
        print(f"    MARGIN: {r['verdict_margin']}")
        print(f"    ROOM  : {r['verdict_room']}")
        print(f"    ACC   : {r['acc_gate']}  (arm mean {r['arm_mean_acc']:.4f}, "
              f"per-seed min {r['arm_min_acc']:.4f}, floor {ACC_FLOOR})")
        if r["verdict_sign"].startswith("NOT RESOLVED"):
            print("    UNRESOLVED IS NOT A ZERO: the crossing direction is uncontrolled and "
                  "non-crossing rounds\n    carry no dose, so this is not evidence of absence.")
    prior = cifar10_prior()
    if prior:
        print("\n=== THE CIFAR-10 READING OF THE SAME ESTIMAND, FOR REFERENCE AND NOT AS A PRIOR ===")
        print(f"  {prior['dataset']}/{prior['model']}/{prior['attack']}: n={prior['n']} "
              f"mean={prior['mean']:+.4f} CI [{prior['ci95'][0]:+.4f}, {prior['ci95'][1]:+.4f}]")
        print("  Dataset, architecture and attack all differ from this arm. It bounds nothing here.")


def main():
    check_frozen()
    cells, hc = load()
    if hc is None or not hc.get("all_passed"):
        sys.exit("REFUSING TO RUN: no passing --harness-check verdict is recorded in "
                 f"{out_path}.\n  Run: PYTHONPATH=. python3 -m "
                 "experiments.run_oracle_free_femnist --harness-check\n  The freeze orders its "
                 "gates before any ASR is scored, and a verdict that is not persisted is witnessed "
                 "only by a terminal that has since closed.")
    total = len(ARMS) * len(SEEDS20)
    print("=== ARM C: the oracle-free decision contrast on a SECOND dataset ===")
    print(f"    {DATASET}/{MODEL}, d1={D1} (pass-through) + blend, d2={D2}, {ATTACK}")
    print(f"    {len(ARMS)} arms x {len(SEEDS20)} seeds = {total} runs, seeds "
          f"{SEEDS20[0]}-{SEEDS20[-1]}, both arms NEW (no imported leg, no mixed n)")
    print(f"    SHARE_TOL={SHARE_TOL:.0e} imported from measure_admission.py; instrument frozen at "
          f"{INSTRUMENT_PREREG_COMMIT}")
    print(f"    rules frozen at {PREREG_COMMIT}, md5 {prereg_md5()}")
    print("    The CIFAR-10 arm read +0.3218, CI [+0.168, +0.476]. That is quoted by the freeze for")
    print("    reference and is NOT a prediction: dataset, architecture and attack all differ.")
    print("    Whatever this returns, neither 'the negative reproduces oracle-free' nor 'the "
          "negative\n    fails without an oracle' may be written anywhere.\n", flush=True)
    if cells:
        print(f"  resuming: {sum(len(c['per_seed']) for c in cells.values())} rows already present "
              "(progress is counted from these rows, not from a log index)\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); done = 0
    for a in ARMS:
        label, side, _ = a
        key = key_of(a)
        existing = {r["seed"]: r for r in cells.get(key, {}).get("per_seed", [])}
        for seed in SEEDS20:
            if seed in existing:
                done += 1
                continue
            t = time.time()
            rec = []
            acc, asr = run_one(seed, None, D2, ATTACK, None, dataset=DATASET, model=MODEL,
                               d1_override=D1, stack_hook=make_hook(side, rec))
            if len(rec) != FL_CONFIG.num_rounds:
                sys.exit(f"REFUSING TO CONTINUE: the hook was invoked {len(rec)} times for "
                         f"{FL_CONFIG.num_rounds} rounds at seed {seed}. A transform that is not "
                         "applied every round makes arm B equal to arm A and the run still "
                         "completes; that is the failure mode this check exists for.")
            gaps = [r["share_gap_sup_over_all_adversary_subsets"] for r in rec
                    if r["share_gap_sup_over_all_adversary_subsets"]
                    == r["share_gap_sup_over_all_adversary_subsets"]]
            existing[seed] = {
                "seed": int(seed), "accuracy": float(acc), "asr": float(asr),
                "n_flip_rounds": int(sum(1 for r in rec if r["flip"])),
                "max_share_gap_sup": float(max(gaps)) if gaps else None,
                "max_coeff_gap_linf": float(max(r["coeff_gap_linf"] for r in rec)),
                "mean_t_star_on_flip_rounds":
                    float(np.mean([r["t_star"] for r in rec if r["flip"]]))
                    if any(r["flip"] for r in rec) else None,
                "per_round": rec,
                "source": "<computed here>"}
            done += 1
            cells[key] = {"arm": label, "side": side, "d1": D1, "d2": D2, "attack": ATTACK,
                          "dataset": DATASET, "model": MODEL,
                          "per_seed": [existing[s] for s in sorted(existing)]}
            save(cells, hc)
            gap = existing[seed]["max_share_gap_sup"]
            print(f"  [{done}/{total}] arm {label} s{seed}: acc={acc:.4f} ASR={asr:.4f}  "
                  f"crossings={existing[seed]['n_flip_rounds']}/{FL_CONFIG.num_rounds}  "
                  f"max share gap={'n/a' if gap is None else f'{gap:.2e}'}  "
                  f"({time.time() - t:.0f}s)"
                  + ("  * below acc floor" if acc < ACC_FLOOR else ""), flush=True)
    save(cells, hc)
    report(cells)
    print(f"\nWall time: {(time.time() - t0) / 3600:.1f} h\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    if "--harness-check" in sys.argv:
        check_frozen()
        cells, _ = load()
        v = harness_check()
        os.makedirs(out_dir, exist_ok=True)
        save(cells, v)          # persisted, not printed and discarded
        print(f"Harness-check verdict written to {out_path}")
        sys.exit(0 if v["all_passed"] else 1)
    sys.exit(main())
