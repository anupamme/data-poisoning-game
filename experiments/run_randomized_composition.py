"""
Compute-matched randomization: is randomization ITSELF costly, or only a degenerate order?

WHAT THIS REPLACES AND WHY IT IS NOT THE WITHDRAWN COMPARISON
An earlier version of the paper compared deterministic composition against a randomized menu and
reported that composition dominates. That comparison is withdrawn (paper Related Work) and no claim
rests on it. It is withdrawn for a reason this experiment is designed around: the randomized arm of
`run_compute_matched_mixing.py` samples 50/50 between FG->CM and CM->FG, and CM->FG is DEGENERATE --
CoordMedian emits an aggregate rather than per-client updates, so `apply_d1_transform` passes them
through and the round reduces to FoolsGold alone. Half the rounds therefore carry ONE defense, and
the measured 9x/19x gaps are consistent with "one defense per round is worse than two" without
saying anything about randomization per se. That is exactly the objection, and it is correct.

Here every member of every randomized set has a genuine per-client upstream (FoolsGold), so NO round
degenerates to a single defense. If randomization is intrinsically costly under a persistent committed
attack, arm 2 must be worse than the best member of its own set. If it is not, the withdrawal stands
and we report a null. Both outcomes answer the question; nothing in the paper's conclusions depends on
which.

THREE HARNESS DEFECTS IN THE PREDECESSOR, ALL FIXED HERE
(1) THE ARMS WERE NOT SEED-MATCHED. `run_compute_matched_mixing.py` reads

        if randomize_order and rng.random() < 0.5:

    and Python SHORT-CIRCUITS, so the fixed arm never draws that number while the randomized arm
    draws one per round -- from the same generator that selects participants. The two arms therefore
    saw different clients, and no paired test was licensed on that artifact. Here the policy draw
    comes from a SEPARATE Generator (`policy_rng`, seed+2000), which cannot perturb the participant
    sequence however many times it fires, so every arm sees identical participants at a given seed.

(2) THE PREDECESSOR WAS NOT ON THE CANONICAL PARTICIPANT STREAM AT ALL, which is the deeper defect
    and the one that decides this file's design. `run_all_compositions.run_one` -- the path behind
    results/all_compositions, results/wave2_held_out, results/fg_cm_survivor and every
    fedavg_then_X baseline -- draws participants from the LEGACY GLOBAL np.random.choice seeded by
    np.random.seed(seed). `run_compute_matched_mixing.py` used np.random.default_rng(seed + 1000),
    a different sequence, so it re-measured rather than reproduced every cell it shares with the
    canonical suite: its fg_cm_fixed pixel arm is 0.051/0.061/0.051/0.072/0.144 (mean 0.076) where
    the canonical path is 0.114/0.087/0.131/0.081/0.081 (mean 0.099) at the same five seeds, a
    maximum per-seed deviation of 0.079. Meanwhile fg_cm_survivor IS bit-identical to the frozen
    all_compositions on the shared seeds 42--44 (max deviation 0.00e+00), i.e. the canonical cell
    extended to n=5 rather than re-measured. This runner therefore uses the CANONICAL stream and
    calls `generic_compose`, so its fixed arms are expected to be bit-identical to the canonical
    artifacts -- and are NOT expected to match run_compute_matched_mixing.

(3) A MIXED-n TRAP. The frozen fixed compositions in results/all_compositions/summary.json are
    seeds 42--44 (n=3) while the mixing artifact is 42--46 (n=5). Scoring a new n=5 randomized arm
    against an n=3 fixed leg puts a mixed-n comparison in the SUBTRAHEND where no column shows it.
    So every arm here is re-run: all seven conditions, one artifact, one seed set, no leg reused.

ARMS (compute is matched WITHIN arms 1-2 only; 3 and 4 use half the defense computation and are
labelled as such, because the reviewer asked to see that confound rather than have it hidden)
  arm 1  fixed_fg_cm / fixed_fg_rfa / fixed_fg_tm   FG->CM, FG->RFA, FG->TM fixed     2 defenses
  arm 2  rand_comp_strong   uniform over {FG->CM, FG->RFA}   (PRIMARY, no weak member) 2 defenses
  arm 2  rand_comp_all3     uniform over all three of arm 1  (SECONDARY, heterogeneous) 2 defenses
  arm 3  temporal_mix       each round uniform over {FG, CM, RFA, TM} alone           1 defense
  arm 4  pure_coord_median  CoordMedian alone (best max-committed standalone, 0.519)   1 defense

Arm 4 doubles as a harness control: it is the same quantity as `fedavg_then_coord_median` in
results/wave2_held_out/summary.json (0.519 scaling / 0.443 pixel, n=5, seeds 42--46).

Config: N=10, K=5, f=0.2, cifar_cnn, 50 rounds, Dirichlet 0.5, seeds 42--46 (n=5)
Attacks: committed_scaling, committed_pixel.  70 runs total.
Output: results/randomized_composition/summary.json  (the predecessor artifact is left untouched)
Pre-registered at experiments/pre_registration_randomized_composition.md before any run.

Run: python3 experiments/run_randomized_composition.py [--only COND]
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor
from experiments.run_all_compositions import generic_compose

SEEDS = [42, 43, 44, 45, 46]
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2
ATTACKS = ["committed_scaling", "committed_pixel"]

# A policy is a list of (d1, d2) stages sampled uniformly per round. d1='fedavg' is the identity
# pass-through (apply_d1_transform Type C/D), so ('fedavg', X) is "X alone" -- the same object the
# single-defense baselines are defined as (fedavg_then_X). A single-element list is a fixed policy.
COMPOSED = [("foolsgold", "coord_median"),
            ("foolsgold", "rfa"),
            ("foolsgold", "trimmed_mean")]
SINGLES = [("fedavg", "foolsgold"),
           ("fedavg", "coord_median"),
           ("fedavg", "rfa"),
           ("fedavg", "trimmed_mean")]

# TWO randomized arms, because one of them would otherwise carry the predecessor's confound in a
# new form. Canonically (results/all_compositions, n=3) fg->TM's committed-scaling arm is
# 0.508 +/- 0.269 -- ABOVE the 0.5 threshold, and it is a criterion-FAIL pair (C3). A randomized set
# containing it could degrade simply because a non-suppressing member was drawn, which is the same
# "one round was undefended" objection that sank the CM->FG arm, only harder to see. So:
#   rand_comp_strong  uniform over members that EACH suppress both attacks alone -> PRIMARY
#   rand_comp_all3    uniform over the full structural set, heterogeneous       -> SECONDARY
# The primary is the confound-free test of whether randomization is intrinsically costly; the
# secondary prices a realistic menu whose members are not equally good. Set membership is fixed by
# the rule above, not by any outcome measured in this artifact.
STRONG = [COMPOSED[0], COMPOSED[1]]          # fg->cm, fg->rfa: both suppress both arms alone

CONDITIONS = [
    # name,                policy,                    defenses_per_round, arm
    ("fixed_fg_cm",        [COMPOSED[0]],             2, 1),
    ("fixed_fg_rfa",       [COMPOSED[1]],             2, 1),
    ("fixed_fg_tm",        [COMPOSED[2]],             2, 1),
    ("rand_comp_strong",   STRONG,                    2, 2),
    ("rand_comp_all3",     COMPOSED,                  2, 2),
    ("temporal_mix",       SINGLES,                   1, 3),
    ("pure_coord_median",  [("fedavg", "coord_median")], 1, 4),
]

PREREG = os.path.join(base_dir, "experiments", "pre_registration_randomized_composition.md")
output_dir = os.path.join(base_dir, "results", "randomized_composition")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


def prereg_commit():
    """The commit that added the pre-registration, or None if it is absent or dirty.

    Same guard as run_wave3_emergent.py, for the same reason: the direction of the primary
    contrast has to be frozen in git before any of this data exists, and that is enforced
    rather than asserted in prose. The hash is also written into the artifact, so a reader
    of results/randomized_composition/summary.json can check the freeze without trusting
    this docstring.
    """
    rel = os.path.relpath(PREREG, base_dir)
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel],
                             cwd=base_dir, capture_output=True, text=True,
                             check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    if not out:
        return None
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel],
                           cwd=base_dir, capture_output=True, text=True).stdout.strip()
    return None if dirty else out


def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {"description": "Compute-matched randomization: fixed vs randomized composition, "
                           "against one-defense-per-round references",
            "config": {"num_clients": FL_CONFIG.num_clients,
                       "clients_per_round": FL_CONFIG.clients_per_round,
                       "num_rounds": FL_CONFIG.num_rounds,
                       "adv_fraction": ADV_FRACTION, "dirichlet_alpha": 0.5,
                       "model": "cifar_cnn", "dataset": "cifar10", "seeds": SEEDS},
            "conditions": {}}


def has_run(cond_name, attack, seed):
    c = load_or_init()["conditions"].get(cond_name, {})
    return any(r["seed"] == seed for r in c.get(attack, {}).get("per_seed", []))


def save_one(cond_name, attack, seed, accuracy, asr, policy_log, policy, dpr, arm):
    s = load_or_init()
    cond = s["conditions"].setdefault(cond_name, {})
    cond["policy"] = [list(p) for p in policy]
    cond["defenses_per_round"] = dpr
    cond["arm"] = arm
    atk = cond.setdefault(attack, {"per_seed": []})
    per_seed = [r for r in atk["per_seed"] if r["seed"] != seed]
    # policy_log is the REALIZED per-round policy. It is stored, not summarized away: an arm whose
    # sampling hook silently never fired is indistinguishable from a fixed arm in the ASR alone,
    # and this project has shipped exactly that defect before (run_cross_distribution_compositions
    # never called manipulate_update). analyze_randomized_composition.py asserts on this field.
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr),
                     "policy_log": policy_log})
    atk["per_seed"] = sorted(per_seed, key=lambda r: r["seed"])
    asrs = [r["asr"] for r in atk["per_seed"]]
    accs = [r["accuracy"] for r in atk["per_seed"]]
    atk["mean_asr"] = float(np.mean(asrs))
    # BOTH conventions, labelled. Every published +/- in these papers is the POPULATION sd, because
    # every prior runner wrote np.std(...) with no ddof -- verified against the numbers in print:
    # compute_matched_mixing's fg_cm_fixed pixel sd is 0.0350 (ddof=0) not 0.0391 (ddof=1), and the
    # supplement's "0.679 +/- 0.159" is fg_cm_survivor's mixed pixel arm at ddof=0 (0.1593) not
    # ddof=1 (0.1781). Emitting only ddof=1 here would silently widen every interval relative to the
    # tables it sits beside; emitting only ddof=0 understates by 10% at n=5. So both are stored and
    # the analyzer states which it prints.
    atk["std_asr"] = float(np.std(asrs)) if len(asrs) > 1 else 0.0          # ddof=0, corpus convention
    atk["std_asr_ddof1"] = float(np.std(asrs, ddof=1)) if len(asrs) > 1 else 0.0
    atk["mean_accuracy"] = float(np.mean(accs))
    atk["n"] = len(asrs)
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


def run_one(seed, policy, attack_name):
    """One 50-round FL run under `policy`. Returns (accuracy, asr, realized policy per round)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        "cifar10", FL_CONFIG.num_clients, 0.5, seed
    )
    model = get_model("cifar_cnn", num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(FL_CONFIG.num_clients * ADV_FRACTION)
    adversarial_ids = set(range(num_adversarial))

    internal = {"committed_scaling": "model_scaling",
                "committed_pixel": "backdoor_pixel"}.get(attack_name, attack_name)
    attack = get_attack(internal)

    clients = []
    for i in range(FL_CONFIG.num_clients):
        dataset = client_datasets[i]
        if i in adversarial_ids:
            dataset = attack.poison_dataset(dataset)
        clients.append(FederatedClient(i, dataset, device))

    # TWO STREAMS, AND THE PARTICIPANT ONE IS THE *CANONICAL* STREAM, NOT THE PREDECESSOR'S.
    # This is the load-bearing line of the whole file. `run_all_compositions.run_one` -- the path
    # that produced results/all_compositions, results/wave2_held_out, results/fg_cm_survivor and
    # every fedavg_then_X single-defense baseline -- selects participants with the LEGACY GLOBAL
    # np.random.choice, seeded by np.random.seed(seed) above. `run_compute_matched_mixing.py`
    # instead used np.random.default_rng(seed + 1000), which is a DIFFERENT participant sequence,
    # and that is why its fg_cm_fixed pixel arm reads 0.076 where the canonical path reads 0.099 on
    # the same nominal cell at the same five seeds (per-seed 0.051/0.061/0.051/0.072/0.144 against
    # 0.114/0.087/0.131/0.081/0.081; max deviation 0.079). Copying the predecessor's Generator
    # "for fidelity" would have made these arms comparable only to a superseded artifact and would
    # have failed controls 3 and 4 of the pre-registration by construction. So we use the canonical
    # stream, and the fixed arms are expected to reproduce all_compositions / fg_cm_survivor
    # BIT-IDENTICALLY.
    #
    # `policy_rng` is a separate Generator, so a policy draw cannot perturb the participant
    # sequence no matter how many times it fires -- which is the defect being fixed: the
    # predecessor's `randomize_order and rng.random() < 0.5` short-circuited, so its randomized arm
    # drew from the participant stream and its fixed arm did not, desynchronizing the two arms it
    # was built to compare.
    policy_rng = np.random.default_rng(seed + 2000)
    current_lr = FL_CONFIG.learning_rate
    policy_log = []

    for _ in range(FL_CONFIG.num_rounds):
        participant_ids = np.random.choice(
            FL_CONFIG.num_clients,
            size=min(FL_CONFIG.clients_per_round, FL_CONFIG.num_clients),
            replace=False,
        )
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, FL_CONFIG.local_epochs,
                current_lr, FL_CONFIG.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        # Drawn for every arm, including the fixed ones, so the log means the same thing in every
        # row. For a single-element policy the draw is forced and consumes only policy_rng.
        d1, d2 = policy[int(policy_rng.integers(len(policy)))]
        policy_log.append(f"{d1}->{d2}")

        # generic_compose rather than a local apply_d1_transform + aggregate pair, so bit-identity
        # with the canonical artifacts is structural instead of maintained by hand: the canonical
        # runner threads tau=5.0 and server=server into apply_d1_transform and tau into
        # server.aggregate, and reimplementing those two calls is exactly how two paths that are
        # meant to compute the same cell drift apart.
        aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)

        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    accuracy = float(server.evaluate(test_dataset)["accuracy"])
    asr = float(evaluate_backdoor(server.global_model, test_dataset, device=device))
    return accuracy, asr, policy_log


def main():
    commit = prereg_commit()
    if commit is None:
        raise SystemExit(
            f"REFUSING TO RUN: {os.path.relpath(PREREG, base_dir)} is not committed "
            "(or has uncommitted changes).\nThe four arms, the randomized set, the paired "
            "primary contrast and its DIRECTION must be frozen in git before any of this "
            "data exists.")
    s = load_or_init()
    if s.get("prereg_commit") not in (None, commit):
        print(f"  NOTE: earlier seeds ran at prereg {s['prereg_commit']}, now {commit}")
    s["prereg_commit"] = commit
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)

    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    conds = [c for c in CONDITIONS if only is None or c[0] == only]
    print(f"=== Compute-matched randomization (prereg committed at {commit}) ===")
    print(f"  conditions: {[c[0] for c in conds]}")
    print(f"  attacks: {ATTACKS}   seeds: {SEEDS}   "
          f"runs: {len(conds) * len(ATTACKS) * len(SEEDS)}")
    print(f"  FL: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}, cifar_cnn\n")

    t0 = time.time()
    total = len(conds) * len(ATTACKS) * len(SEEDS)
    done = 0
    for cond_name, policy, dpr, arm in conds:
        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(cond_name, attack, seed):
                    done += 1
                    print(f"  [cached] {cond_name} | {attack} | seed={seed}", flush=True)
                    continue
                t1 = time.time()
                acc, asr, plog = run_one(seed, policy, attack)
                save_one(cond_name, attack, seed, acc, asr, plog, policy, dpr, arm)
                done += 1
                print(f"[{done}/{total}] arm{arm} {cond_name} | {attack} | seed={seed} "
                      f"acc={acc:.3f} asr={asr:.3f} "
                      f"({time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    s = load_or_init()
    print("\n=== RANDOMIZED COMPOSITION: RAW ARMS (analysis is in "
          "analyze_randomized_composition.py) ===")
    for cond_name, policy, dpr, arm in CONDITIONS:
        cond = s["conditions"].get(cond_name, {})
        row = f"  arm{arm} {cond_name:20s} {dpr}def/rd"
        for atk in ATTACKS:
            a = cond.get(atk, {})
            row += (f"  {atk.replace('committed_',''):8s} "
                    f"ASR={a.get('mean_asr', float('nan')):.3f} "
                    f"acc={a.get('mean_accuracy', float('nan')):.3f} n={a.get('n', 0)}")
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
