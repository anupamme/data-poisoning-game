"""
PHASE 2 of the metric-swap ablation.

The design breaks mechanism--effectiveness confounding by holding the suppression principle
fixed and varying ONLY the invariance class of d_2's statistic. Each test arm is a
minimal-change ablation of the control immediately above it -- same soft-trim or same
selection rule, cosine in place of L2 -- so standalone effectiveness is matched by
construction and measured in Phase 0 before any label is frozen.

  d_1 in {norm_clip, rfa}: heterogeneous positive rescalings, and neither suppresses
  model-scaling alone (0.935, 0.885), so C0 holds and the composition cannot inherit
  suppression from d_1.

  arm      d_2               statistic                       C2      predicted (scaling)
  control  reputation        consensus distance (L2)         fail    HIGH
  test     cos_reputation    direction-consensus distance    hold    (see prereg)
  control  krum              pairwise distance (L2)          fail    HIGH
  test     cos_krum          pairwise cosine                 hold    (see prereg)

cos_krum and cos_reputation are MECHANISM-ISOLATING ABLATIONS, not proposed defenses.
Their invariance is verified in code before it is claimed: see
experiments/verify_cos_invariance.py / results/cos_invariance_check.json.

PRE-REGISTERED CONTINGENCY (recorded before running, not after): both variants are
magnitude-blind, and the Euclidean originals may be catching model-scaling *via* magnitude.
If a variant fails to suppress an attack standalone in Phase 0 (mean ASR >= 0.5), then C1
fails for that (variant, attack), the prediction is HIGH, and THAT ARM YIELDS NO POSITIVE
TEST. It is recorded as such -- no switching of attack, threshold, or seed count. That
outcome is itself a finding: magnitude information would be load-bearing for the defense
family, i.e. invariance and standalone effectiveness inherently coupled here, which
strengthens the testability proposition rather than rescuing the framework.

Both attacks are run. On backdoor_pixel, C1 fails for every pair (no d_2 in this arm
suppresses pixel alone), so all pixel cells are predicted HIGH -- a negative control the
framework must get right. The sanity gate holds: FedAvg reaches 0.667 (scaling) and 0.768
(pixel), both >= 0.5, so predicted-HIGH cells are informative.

Config identical to the rest of the paper: N=10, K=5, f=0.2, alpha=0.5, 50 rounds,
cifar_cnn, seeds 42/43/44, both committed attacks. The two reputation controls were already
measured under exactly this protocol and these seeds in the development suite, so they are
REUSED from results/all_compositions/summary.json rather than re-run: 6 new pairs x 2 attacks
x 3 seeds = 36 runs. Reused cells are marked "reuse" in the output and in the printed tables.
Output: results/metric_swap/summary.json

DO NOT RUN until experiments/pre_registration_metric_swap.md is git-committed and
PREREG_COMMIT below is filled in with that commit hash. The script refuses to start
otherwise -- an unfrozen run would make the labels unfalsifiable, which is the entire
point of the experiment.
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

# experiments/pre_registration_metric_swap.md, committed before results/metric_swap/ existed.
PREREG_COMMIT = "356897b"

# (d1, d2, predicted_label, category) -- copied VERBATIM from the frozen pre-registration
# table, in the same order. C1 is the binding condition for all eight pairs, so every label is
# HIGH and the suite contains no positive test. That follows from the Phase 0 inputs and was
# recorded before any composition ran.
PAIRS = [
    ("norm_clip", "reputation",      "HIGH", "C1-fail(pixel)/C2-fail"),
    ("norm_clip", "cos_reputation",  "HIGH", "C1-fail(both)/C2-hold"),
    ("rfa",       "reputation",      "HIGH", "C1-fail(pixel)/C2-fail"),
    ("rfa",       "cos_reputation",  "HIGH", "C1-fail(both)/C2-hold"),
    ("norm_clip", "krum",            "HIGH", "C1-fail(pixel)/C2-fail"),
    ("norm_clip", "cos_krum",        "HIGH", "C1-fail(acc-gate)/C2-hold"),
    ("rfa",       "krum",            "HIGH", "C1-fail(pixel)/C2-fail"),
    ("rfa",       "cos_krum",        "HIGH", "C1-fail(acc-gate)/C2-hold"),
]
# The one two-sided prediction. cos_krum's C1 verdict on model-scaling turns on whether the
# accuracy gate is applied (0.078 ASR at 0.146 clean accuracy). The pre-registration freezes
# BOTH readings and a resolution rule: primary = accuracy-gated C1 fails -> HIGH (above);
# secondary = numeric C1 only -> certified -> LOW. If these pairs land LOW with mean accuracy
# >= ACC_FLOOR the gate was wrong to exclude them and the primary labels are misses; LOW below
# the floor vindicates the gate; HIGH refutes the secondary reading.
SECONDARY = {("norm_clip", "cos_krum"): "LOW", ("rfa", "cos_krum"): "LOW"}
# Already measured under the identical protocol and seeds in the development suite; reused
# rather than re-run. Any other pair is run fresh.
REUSE = {("norm_clip", "reputation"), ("rfa", "reputation")}
DEV = os.path.join(base, "results", "all_compositions", "summary.json")
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACK_MAP = {"committed_scaling": "model_scaling", "committed_pixel": "backdoor_pixel"}
ACC_FLOOR = 0.35          # below this a low ASR is a collapsed model, not suppression

# NOTE: the directory is created in __main__, not at import time. Importing this module must not
# touch results/metric_swap/, or the provenance gate ("prereg committed before the first write to
# the output directory") becomes ambiguous.
out_dir = os.path.join(base, "results", "metric_swap")
out_path = os.path.join(out_dir, "summary.json")


def run_one(seed, d1, d2, attack_name):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cd, td, nc = get_federated_dataset("cifar10", FL_CONFIG.num_clients, 0.5, seed)
    srv = FederatedServer(get_model("cifar_cnn", nc), dev,
                          clean_holdout_dataset=Subset(td, list(range(100))), holdout_batch_size=32)
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
            u = cl[cid].train(srv.global_model, FL_CONFIG.local_epochs, lr, FL_CONFIG.local_batch_size)
            if cid in adv:
                u = atk.manipulate_update(u, srv.global_model)
            ups.append(u)
        srv.apply_update(generic_compose(srv, ups, d1, d2, tau=5.0))
        lr *= getattr(FL_CONFIG, "lr_decay", 1.0)
    return float(srv.evaluate(td)["accuracy"]), float(evaluate_backdoor(srv.global_model, td, device=dev))


def reuse_cell(d1, d2, attack):
    """Pull an already-measured control cell from the development suite, seeds checked."""
    c = json.load(open(DEV))["pairs"][f"{d1}_then_{d2}"][attack]
    got = sorted(r["seed"] for r in c["per_seed"])
    if got != sorted(SEEDS):
        sys.exit(f"REFUSING TO REUSE {d1}->{d2}|{attack}: seeds {got} != {sorted(SEEDS)}")
    ps = [{"seed": r["seed"], "accuracy": r["accuracy"], "asr": r["asr"]} for r in c["per_seed"]]
    return ps


def check_frozen():
    prereg = os.path.join(base, "experiments", "pre_registration_metric_swap.md")
    if PREREG_COMMIT is None or any(p[2] is None for p in PAIRS):
        sys.exit("REFUSING TO RUN: predictions are not frozen.\n"
                 f"  1. complete Phase 0 (results/metric_swap_baselines/summary.json)\n"
                 f"  2. write {prereg} with per-pair C0/C1/C2/C3 values and labels\n"
                 "  3. git commit it, then set PREREG_COMMIT and the PAIRS labels here\n"
                 "     verbatim from that file.")
    if not os.path.exists(prereg):
        sys.exit(f"REFUSING TO RUN: {prereg} does not exist.")


if __name__ == "__main__":
    check_frozen()
    os.makedirs(out_dir, exist_ok=True)
    n_new = len([p for p in PAIRS if (p[0], p[1]) not in REUSE])
    total = n_new * len(ATTACKS) * len(SEEDS)
    print(f"=== PHASE 2: metric-swap suite ({len(PAIRS)} pairs, {len(REUSE)} reused, "
          f"{total} new runs) ===")
    print(f"    predictions frozen in pre_registration_metric_swap.md @ {PREREG_COMMIT}\n", flush=True)
    cells = {}
    if os.path.exists(out_path):
        try:
            cells = json.load(open(out_path)).get("cells", {}); print(f"  resuming: {len(cells)} cells\n", flush=True)
        except Exception:
            cells = {}
    t0 = time.time(); done = 0
    for d1, d2, pred, cat in PAIRS:
        for atk in ATTACKS:
            key = f"{d1}_then_{d2}|{atk}"
            if key in cells and len(cells[key]["per_seed"]) == len(SEEDS):
                if (d1, d2) not in REUSE:
                    done += len(SEEDS)
                continue
            if (d1, d2) in REUSE:
                ps = reuse_cell(d1, d2, atk); src = "all_compositions"
                print(f"  [reuse] {d1}->{d2:16s} {atk:18s} "
                      f"ASR={np.mean([r['asr'] for r in ps]):.3f} (development suite)", flush=True)
            else:
                ps = []; src = "metric_swap"
                for seed in SEEDS:
                    t = time.time(); acc, asr = run_one(seed, d1, d2, atk)
                    ps.append({"seed": seed, "accuracy": acc, "asr": asr}); done += 1
                    print(f"  [{done}/{total}] {d1}->{d2:16s} {atk:18s} s{seed}: "
                          f"acc={acc:.3f} ASR={asr:.3f} ({time.time()-t:.0f}s)", flush=True)
            cells[key] = {"d1": d1, "d2": d2, "attack": atk, "predicted": pred, "category": cat,
                          "source": src,
                          "per_seed": ps, "mean_asr": float(np.mean([r["asr"] for r in ps])),
                          "std_asr": float(np.std([r["asr"] for r in ps])),
                          "mean_acc": float(np.mean([r["accuracy"] for r in ps]))}
            json.dump({"description": "Phase 2: metric-swap ablation; predictions frozen "
                                      f"at {PREREG_COMMIT}",
                       "prereg_commit": PREREG_COMMIT,
                       "config": {"N": 10, "K": 5, "f": 0.2, "alpha": 0.5, "rounds": 50, "seeds": SEEDS},
                       "cells": cells}, open(out_path, "w"), indent=2)

    print("\n=== RESULTS vs FROZEN PREDICTIONS (max-committed) ===")
    print(f"{'pair':34s} {'category':18s} {'pred':>5} {'maxASR':>7} {'acc':>6} {'actual':>7}  "
          f"hit   source")
    tp = fp = tn = fn = 0
    unusable = []
    for d1, d2, pred, cat in PAIRS:
        ks = [f"{d1}_then_{d2}|{a}" for a in ATTACKS]
        if not all(k in cells for k in ks):
            continue
        worst = max((cells[k] for k in ks), key=lambda c: c["mean_asr"])
        mx, acc = worst["mean_asr"], worst["mean_acc"]
        actual = "LOW" if mx < 0.5 else "HIGH"
        hit = actual == pred
        if min(cells[k]["mean_acc"] for k in ks) < ACC_FLOOR:
            unusable.append(f"{d1}->{d2} (min mean accuracy "
                            f"{min(cells[k]['mean_acc'] for k in ks):.3f})")
        if pred == "LOW" and hit: tp += 1
        elif pred == "LOW" and not hit: fp += 1
        elif pred == "HIGH" and hit: tn += 1
        else: fn += 1
        print(f"{d1+'->'+d2:34s} {cat:18s} {pred:>5} {mx:7.3f} {acc:6.2f} {actual:>7}  "
              f"{'OK' if hit else 'MISS':5s} {worst.get('source', '?')}")
    n = tp + fp + tn + fn
    if n:
        print(f"\n  Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}  ({tp+tn}/{n} correct)")

    print("\n=== SECONDARY (numeric-C1) READING of pairs 6 and 8 ===")
    print("    pre-registered alternative label; resolution rule fixed before the run")
    for (d1, d2), sec in SECONDARY.items():
        ks = [f"{d1}_then_{d2}|{a}" for a in ATTACKS]
        if not all(k in cells for k in ks):
            continue
        worst = max((cells[k] for k in ks), key=lambda c: c["mean_asr"])
        mx = worst["mean_asr"]
        acc = min(cells[k]["mean_acc"] for k in ks)
        actual = "LOW" if mx < 0.5 else "HIGH"
        if actual == "HIGH":
            verdict = "secondary REFUTED; accuracy gate vindicated"
        elif acc >= ACC_FLOOR:
            verdict = (f"secondary CONFIRMED at usable accuracy ({acc:.3f}) -- the accuracy gate "
                       f"was wrong to exclude this pair; PRIMARY LABEL IS A MISS")
        else:
            verdict = f"LOW but accuracy {acc:.3f} < {ACC_FLOOR}: hollow, gate was right"
        print(f"  {d1}->{d2:12s} primary=HIGH secondary={sec} actual={actual} "
              f"maxASR={mx:.3f} minacc={acc:.3f}\n      -> {verdict}")

    print("\n=== PER-ATTACK CONTRAST (control vs test, d_1 fixed) -- CONFOUNDED, NOT A TEST ===")
    print("    C1 differs between the arms (see below), so no gap here is attributable to C2 alone.")
    for atk in ATTACKS:
        base = {"reputation": {"committed_scaling": 0.017, "committed_pixel": 0.842},
                "cos_reputation": {"committed_scaling": 0.982, "committed_pixel": 0.754},
                "krum": {"committed_scaling": 0.061, "committed_pixel": 0.583},
                "cos_krum": {"committed_scaling": 0.078, "committed_pixel": 0.300}}
        print(f"  -- {atk} --")
        for d1 in ("norm_clip", "rfa"):
            for ctrl, test in (("reputation", "cos_reputation"), ("krum", "cos_krum")):
                kc, kt = f"{d1}_then_{ctrl}|{atk}", f"{d1}_then_{test}|{atk}"
                if kc in cells and kt in cells:
                    print(f"    {d1:9s} {ctrl:14s} {cells[kc]['mean_asr']:.3f} "
                          f"(standalone {base[ctrl][atk]:.3f})   vs   {test:14s} "
                          f"{cells[kt]['mean_asr']:.3f} (standalone {base[test][atk]:.3f})")

    if unusable:
        print("\n--- ACCURACY GATE: report these as uninterpretable, not as successes ---")
        for u in unusable:
            print("  !", u)
    print(f"\nWall time: {(time.time()-t0)/3600:.1f} h\nSaved to {out_path}")
