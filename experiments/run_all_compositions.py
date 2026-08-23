"""
Expanded composability criterion validation — 18 pairs covering all prediction categories.

Defenses: fedavg, norm_clip, trimmed_mean, coord_median, rfa, foolsgold, reputation
Pairs: 18 targeted ordered pairs (2 DEGEN, 3 C2-FAIL, 5 C3-FAIL, 5 PASS, 3 borderline)
Attacks: committed_scaling, committed_pixel
Seeds: 42-44 (3 seeds)
FL params: N=10, K=5, f=0.2, 50 rounds, cifar_cnn

Output: results/all_compositions/summary.json
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from config import FLConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from experiments.run_payoff_matrix import evaluate_backdoor

# --- Configuration ---
DEFENSES = ["fedavg", "norm_clip", "trimmed_mean", "coord_median", "rfa", "foolsgold", "reputation"]
# Expanded validation: 18 targeted pairs covering all criterion prediction categories.
# With C3 (boundary preservation), the previous 2 "misses" (NC->tm, FG->tm) become
# correctly predicted C3-FAIL cases. We now add more PASS cases to demonstrate
# above-chance accuracy in the practitioner-relevant direction.
#
# Categories:
#   DEGEN (d1 no per-client transform): fedavg->X (2 pairs, already done)
#   C2-FAIL (d2 destroys d1's signal): NC->rep, NC->fg, rfa->rep (already done)
#   C3-FAIL (d1 compresses boundary): NC->tm, FG->tm (already done), NC->cm, NC->rfa, rfa->tm, rfa->cm
#   C1+C2+C3 PASS (composition works): rep->cm (done), rep->rfa, rep->fg, rep->NC
PAIRS = [
    # --- Already cached (8 pairs from prior run) ---
    ("fedavg", "norm_clip"),        # DEGEN -> HIGH
    ("fedavg", "trimmed_mean"),     # DEGEN -> HIGH
    ("norm_clip", "reputation"),    # C2-FAIL -> HIGH
    ("norm_clip", "foolsgold"),     # C2-FAIL -> HIGH
    ("rfa", "reputation"),          # C2-FAIL -> HIGH
    ("foolsgold", "trimmed_mean"),  # C3-FAIL -> HIGH (FG compresses -> TM boundary fails)
    ("norm_clip", "trimmed_mean"),  # C3-FAIL -> HIGH (NC compresses -> TM boundary fails)
    ("reputation", "coord_median"), # PASS -> LOW (rep kills scaling; CM preserves signal+boundary)
    # --- New C3-FAIL pairs (predicted HIGH) ---
    ("norm_clip", "coord_median"),  # C3-FAIL: NC compresses -> CM rank boundary fails
    ("norm_clip", "rfa"),           # C3-FAIL: NC normalizes -> RFA distance boundary fails
    ("rfa", "trimmed_mean"),        # C3-FAIL: RFA reweights -> TM boundary fails
    ("rfa", "coord_median"),        # C3-FAIL: RFA reweights -> CM boundary fails
    ("foolsgold", "coord_median"),  # C3-FAIL: FG compresses -> CM boundary fails
    # --- New PASS pairs (predicted LOW, C1+C2+C3 all hold) ---
    ("reputation", "rfa"),          # PASS: rep kills scaling (0.017); RFA preserves signal+boundary
    ("reputation", "foolsgold"),    # PASS: rep kills scaling; FG cosine-sim scale-invariant
    ("reputation", "norm_clip"),    # PASS: rep kills scaling; NC on pre-weighted updates preserves
    ("reputation", "trimmed_mean"), # PASS: rep kills scaling (0.017); TM preserves — canonical case
    ("foolsgold", "rfa"),           # PASS?: FG diversity + RFA geometric — test empirically
]
ATTACKS = ["committed_scaling", "committed_pixel"]
SEEDS = [42, 43, 44]  # 3 seeds for reliable directional check
FL_CONFIG = FLConfig(num_clients=10, clients_per_round=5, num_rounds=50)
ADV_FRACTION = 0.2

output_dir = os.path.join(base_dir, "results", "all_compositions")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "summary.json")


# --- Checkpointing ---
def load_or_init():
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {
        "description": "All 21 ordered defense pairs, composition ASR",
        "defenses": DEFENSES,
        "pairs": {},
    }


def pair_key(d1, d2):
    return f"{d1}_then_{d2}"


def has_run(d1, d2, attack, seed):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        return False
    attack_key = attack.replace("committed_", "committed_")
    if attack_key not in s["pairs"][pk]:
        return False
    per_seed = s["pairs"][pk][attack_key].get("per_seed", [])
    return any(r["seed"] == seed for r in per_seed)


def save_one(d1, d2, attack, seed, accuracy, asr):
    s = load_or_init()
    pk = pair_key(d1, d2)
    if pk not in s["pairs"]:
        s["pairs"][pk] = {"d1": d1, "d2": d2}
    if attack not in s["pairs"][pk]:
        s["pairs"][pk][attack] = {"per_seed": []}
    per_seed = s["pairs"][pk][attack]["per_seed"]
    per_seed = [r for r in per_seed if r["seed"] != seed]
    per_seed.append({"seed": seed, "accuracy": float(accuracy), "asr": float(asr)})
    s["pairs"][pk][attack]["per_seed"] = per_seed
    # Recompute stats
    asrs = [r["asr"] for r in per_seed]
    s["pairs"][pk][attack]["mean_asr"] = float(np.mean(asrs))
    s["pairs"][pk][attack]["std_asr"] = float(np.std(asrs))
    # Recompute max_committed_asr across both attacks
    max_asr = 0.0
    for atk in ATTACKS:
        if atk in s["pairs"][pk] and "mean_asr" in s["pairs"][pk][atk]:
            max_asr = max(max_asr, s["pairs"][pk][atk]["mean_asr"])
    s["pairs"][pk]["max_committed_asr"] = max_asr
    with open(output_path, "w") as f:
        json.dump(s, f, indent=2)


# --- Controlled-dose upstream transform (dose-response experiment) --------------------
# Not a defense. A synthetic d1 whose only parameter is the DISPERSION of a positive
# per-client rescaling, used to vary the disturbance inflicted on d2's statistic while
# holding d2 and the attack -- and therefore C1, standalone effectiveness -- exactly fixed.
# See experiments/pre_registration_dose_response.md.
DOSE_PREFIX = "dose_kappa"

# --- Targeted dose (Round 12): the same rescaling, but varying WHO is rescaled ---------
# The Round-11 instrument varies the DISPERSION of the coefficients and assigns them by a
# permutation uncorrelated with adversary status. That holds the aggregate's scale fixed but
# not any client's relative weight, so a wide spread mostly dilutes whichever client carries
# the poison and three of four arms ended up measuring Theorem 1's attenuation regime rather
# than C2 disturbance. These two modes separate the channels the Round-11 dose confounds:
#
#   doseS_kappa<K>  statistic-only: every adversary pinned at c = 1 EXACTLY, benign spread
#                   over rho = exp(2K). Payload weight is constant across rungs by
#                   construction, so a rise in ASR cannot be attenuation.
#   doseA_nu<V>     payload-only: benign uniform, adversary-to-benign ratio exp(V), whole
#                   vector scaled to mean 1. Sweeps the adversary's weight through both of
#                   Theorem 1's mechanism-preserving regimes.
#
# Both read adversary identity and are therefore INSTRUMENTS FOR CAUSAL IDENTIFICATION, not
# defenses: no deployable defense knows which clients are adversarial. See
# experiments/pre_registration_targeted_dose.md.
DOSE_S_PREFIX = "doseS_kappa"
DOSE_A_PREFIX = "doseA_nu"


def parse_dose(d1_name):
    """kappa for a 'dose_kappa<K>' d1 name, else None."""
    if not isinstance(d1_name, str) or not d1_name.startswith(DOSE_PREFIX):
        return None
    return float(d1_name[len(DOSE_PREFIX):])


def parse_targeted_dose(d1_name):
    """(mode, value) for a targeted-dose d1 name, else None.

    Mode S carries kappa, the dispersion imposed on the benign coefficients; mode A carries nu,
    the log of the adversary-to-benign coefficient ratio. "doseS_kappa"/"doseA_nu" do not share a
    prefix with "dose_kappa", so parse_dose cannot claim them and the Round-11 ladder is untouched.
    """
    if not isinstance(d1_name, str):
        return None
    if d1_name.startswith(DOSE_S_PREFIX):
        return "S", float(d1_name[len(DOSE_S_PREFIX):])
    if d1_name.startswith(DOSE_A_PREFIX):
        return "A", float(d1_name[len(DOSE_A_PREFIX):])
    return None


def dose_coefficients(n, kappa, dose_key):
    """Positive per-client coefficients with controlled dispersion.

        c_j  proportional to  exp(kappa * (2j/(n-1) - 1)),   normalized so mean(c) = 1

    The realized weight ratio is exactly max(c)/min(c) = exp(2*kappa) -- the same rho that
    Theorem 1 and the two mechanism-preserving regimes are stated in -- so kappa is a dial
    on the theory's own quantity. Normalizing the MEAN to 1 (rather than the sum, or the
    max) holds the aggregate's scale fixed, which separates statistic disturbance (C2) from
    the magnitude-contraction channel of Lemma 1 that costs accuracy.

    j is the client's position in a (seed, round)-keyed pseudorandom permutation, NOT its
    client id: assigning by id would place the adversaries (ids 0..f*N-1) at a fixed end of
    the ladder and systematically attenuate them, which would manufacture a success. The
    permutation depends only on (seed, round), so every downstream arm at a given seed and
    round receives the IDENTICAL coefficient vector -- the arms differ only in d2.
    """
    if n == 1:
        return np.ones(1)
    ladder = np.exp(kappa * (2.0 * np.arange(n) / (n - 1) - 1.0))
    ladder = ladder / ladder.mean()
    perm = np.random.default_rng([int(dose_key[0]), int(dose_key[1])]).permutation(n)
    c = np.empty(n)
    c[perm] = ladder
    return c


def dose_coefficients_statistic_only(adv_mask, kappa, dose_key):
    """Mode S. Benign coefficients spread over rho = exp(2*kappa); every adversary pinned at 1.0.

    The benign ladder is normalized to mean 1 over the BENIGN participants, so with every adversary
    at exactly 1 the mean over all participants is 1 as well: aggregate scale is held fixed exactly
    as in the Round-11 dose, and the adversary's coefficient -- the quantity Lemma 1's attenuation
    channel runs through -- is identical at every rung. Any change in ASR across this ladder is
    therefore attributable to the relative reweighting of the benign updates, which is what the
    statistics d2 reads (pairwise distance, consensus distance, coordinate ordering) are computed
    from. That is the whole design: it closes the channel that confounded Round 11.

    rho = max(c)/min(c) is still exactly exp(2*kappa), because the mean-1 benign ladder straddles 1.

    Degenerate rounds, recorded rather than silently absorbed: fewer than two benign participants
    admits no dispersion at all and returns the identity; a round with no adversary makes this
    transform coincide with the Round-11 dose over the whole round.
    """
    n = len(adv_mask)
    c = np.ones(n)
    ben = [i for i, a in enumerate(adv_mask) if not a]
    m = len(ben)
    if kappa == 0.0 or m < 2:
        return c
    ladder = np.exp(kappa * (2.0 * np.arange(m) / (m - 1) - 1.0))
    ladder = ladder / ladder.mean()
    perm = np.random.default_rng([int(dose_key[0]), int(dose_key[1])]).permutation(m)
    vals = np.empty(m)
    vals[perm] = ladder
    for slot, i in enumerate(ben):
        c[i] = vals[slot]
    return c


def dose_coefficients_payload_only(adv_mask, nu):
    """Mode A. Benign uniform, adversary-to-benign coefficient ratio exactly gamma = exp(nu).

    c_adv = gamma*s, c_ben = s, with s = n / (n + n_a*(gamma - 1)) chosen so mean(c) = 1. s > 0 for
    every gamma > 0 and every 0 <= n_a <= n, so the coefficients stay strictly positive and the
    transform stays inside Proposition 1's hypothesis. Benign clients keep a COMMON coefficient, so
    their relative weighting -- and with it most of what d2's statistic reads among the benign
    majority -- is untouched; what moves is the adversary's share of the aggregate.

    Sweeping gamma through 1 sweeps the adversary through both mechanism-preserving regimes of
    Theorem 1: kept extreme enough for a rank aggregator to discard (gamma large), or attenuated
    until it carries no payload (gamma small). No dose_key is needed -- the assignment is determined
    by adversary status, not by a permutation.
    """
    n = len(adv_mask)
    n_a = sum(bool(a) for a in adv_mask)
    if nu == 0.0 or n_a == 0 or n_a == n:
        return np.ones(n)
    gamma = float(np.exp(nu))
    s = n / (n + n_a * (gamma - 1.0))
    return np.array([gamma * s if a else s for a in adv_mask])


# --- Generic Defense Composition ---
def apply_d1_transform(updates, d1_name, tau=5.0, server=None, dose_key=None,
                       adv_mask=None):
    """Apply d1's per-client transformation without final aggregation.

    Type A (weighting): reputation, foolsgold — compute weights, scale updates
    Type B (clipping): norm_clip — clip each update to norm bound
    Type C/D (aggregating/no-op): trimmed_mean, coord_median, rfa, fedavg — pass through
    Type E (controlled dose): dose_kappa<K> — synthetic rescaling of dispersion e^{2K};
        requires dose_key=(seed, round) and is not a defense (see dose_coefficients)
    Type F (targeted dose): doseS_kappa<K> / doseA_nu<V> — the same rescaling with the
        assignment tied to adversary status; requires adv_mask and is not a defense
        (see dose_coefficients_statistic_only / dose_coefficients_payload_only)
    """
    keys = list(updates[0].keys())
    n = len(updates)

    targeted = parse_targeted_dose(d1_name)
    if targeted is not None:
        mode, val = targeted
        if adv_mask is None:
            raise ValueError(f"{d1_name} requires adv_mask; refusing to run a targeted dose "
                             "without knowing which participants are adversarial")
        if len(adv_mask) != n:
            raise ValueError(f"adv_mask has {len(adv_mask)} entries for {n} updates")
        if mode == "S":
            if val == 0.0:
                return updates          # identity, returned unwrapped as in the Round-11 dose
            if dose_key is None:
                raise ValueError(f"{d1_name} requires dose_key=(seed, round); refusing to run "
                                 "with an unseeded permutation")
            c = dose_coefficients_statistic_only(adv_mask, val, dose_key)
        else:
            if val == 0.0:
                return updates
            c = dose_coefficients_payload_only(adv_mask, val)
        return [{k: u[k] * float(c[i]) for k in keys} for i, u in enumerate(updates)]

    kappa = parse_dose(d1_name)
    if kappa is not None:
        if kappa == 0.0:
            # c == 1 exactly, so the identity. Returned unwrapped rather than multiplied by
            # 1.0 so that the kappa=0 rung is bit-identical to d2 standalone by construction
            # and the harness check in run_dose_response.py cannot be fooled by rounding.
            return updates
        if dose_key is None:
            raise ValueError(f"{d1_name} requires dose_key=(seed, round); refusing to run "
                             "with an unseeded permutation")
        c = dose_coefficients(n, kappa, dose_key)
        return [{k: u[k] * float(c[i]) for k in keys} for i, u in enumerate(updates)]

    if d1_name == "fedavg":
        # No transformation, pass through
        return updates

    elif d1_name == "norm_clip":
        # Clip each update to L2 norm bound tau
        clipped = []
        for u in updates:
            flat = torch.cat([u[k].flatten().float() for k in keys])
            norm = flat.norm().item()
            scale = min(1.0, tau / max(norm, 1e-8))
            clipped.append({k: u[k] * scale for k in keys})
        return clipped

    elif d1_name == "reputation":
        # Reputation weighting: exp(-dist_from_median / scale)
        flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
        client_stack = torch.stack(flats)
        consensus = client_stack.median(dim=0).values
        dists = (client_stack - consensus.unsqueeze(0)).norm(dim=1)
        scale = float(dists.median().clamp(min=1e-6).item())
        rep_weights = torch.exp(-dists / scale)
        rep_weights = rep_weights / rep_weights.sum().clamp(min=1e-8)
        transformed = []
        for i, u in enumerate(updates):
            w = rep_weights[i].item() * n  # scale by n so downstream averaging works
            transformed.append({k: u[k] * w for k in keys})
        return transformed

    elif d1_name == "foolsgold":
        # FoolsGold: downweight clients with high pairwise cosine similarity
        flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
        client_stack = torch.stack(flats)
        normed = F.normalize(client_stack, dim=1)
        sim = normed @ normed.T
        sim.fill_diagonal_(0.0)
        max_sim, _ = sim.abs().max(dim=1)
        max_sim = max_sim.clamp(0.0, 1.0 - 1e-6)
        max_overall = max_sim.max().clamp(min=1e-8)
        max_sim = max_sim / max_overall
        weights = torch.log((1.0 - max_sim) / (max_sim + 1e-5) + 1e-5)
        weights = weights - weights.min()
        weights = weights.clamp(min=0.0)
        total = weights.sum()
        if total.item() < 1e-8:
            return updates  # degenerate case, pass through
        weights = weights / total
        transformed = []
        for i, u in enumerate(updates):
            w = weights[i].item() * n
            transformed.append({k: u[k] * w for k in keys})
        return transformed

    elif d1_name == "rfa":
        # RFA iterative reweighting: compute geometric median weights
        flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
        client_stack = torch.stack(flats)
        estimate = client_stack.mean(dim=0)
        for _ in range(50):
            diffs = client_stack - estimate.unsqueeze(0)
            norms = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            iter_weights = 1.0 / norms
            iter_weights = iter_weights / iter_weights.sum()
            new_estimate = (iter_weights * client_stack).sum(dim=0)
            if (new_estimate - estimate).norm() < 1e-6:
                break
            estimate = new_estimate
        # Extract final weights
        diffs = client_stack - estimate.unsqueeze(0)
        norms = diffs.norm(dim=1).clamp(min=1e-8)
        rfa_weights = 1.0 / norms
        rfa_weights = rfa_weights / rfa_weights.sum()
        transformed = []
        for i, u in enumerate(updates):
            w = rfa_weights[i].item() * n
            transformed.append({k: u[k] * w for k in keys})
        return transformed

    elif d1_name == "fltrust":
        if server is None or getattr(server, "clean_holdout_dataset", None) is None:
            raise ValueError("fltrust as d1 requires a server with a clean_holdout_dataset")
        # FLTrust as d1: per-client positive rescaling by ReLU(cosine to the server
        # reference gradient), with each update renormalized to the server norm.
        # Combined c_i = w_i * n * (||g_srv|| / ||u_i||) >= 0 -- a positive scalar
        # rescaling, so Proposition 1 applies (and ReLU zeros invoke Lemma 1).
        server_update = server._compute_server_update()
        server_flat = torch.cat([server_update[k].flatten().float() for k in keys])
        server_norm = server_flat.norm().clamp(min=1e-8)
        flats = [torch.cat([u[k].flatten().float() for k in keys]) for u in updates]
        client_stack = torch.stack(flats)
        cos_sims = F.cosine_similarity(client_stack, server_flat.unsqueeze(0), dim=1)
        weights = F.relu(cos_sims)
        total = weights.sum()
        if total.item() < 1e-8:
            return updates  # degenerate case, pass through
        weights = weights / total
        transformed = []
        for i, u in enumerate(updates):
            u_norm = flats[i].norm().clamp(min=1e-8)
            c = weights[i].item() * n * (server_norm / u_norm).item()
            transformed.append({k: u[k] * c for k in keys})
        return transformed

    elif d1_name in ("trimmed_mean", "coord_median", "krum", "multi_krum"):
        # These emit a single aggregate (rank statistic, or a selected/averaged subset
        # for krum/multi_krum) rather than per-client updates, so they cannot serve as an
        # upstream stage. Pass through unchanged (composition = effectively d2 only).
        return updates

    else:
        return updates


def generic_compose(server, updates, d1_name, d2_name, tau=5.0, dose_key=None, adv_mask=None):
    """Compose two defenses: d1's transformation then d2's aggregation."""
    # Step 1: Apply d1's per-client transformation
    transformed = apply_d1_transform(updates, d1_name, tau=tau, server=server,
                                     dose_key=dose_key, adv_mask=adv_mask)
    # Step 2: Aggregate with d2
    aggregated = server.aggregate(transformed, method=d2_name, tau=tau)
    return aggregated


# --- Training Loop ---
def run_one(seed, d1, d2, attack_name):
    """Run one composition experiment: (d1, d2) against attack_name with given seed."""
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

    # Map committed attack names to internal attack names
    if attack_name == "committed_scaling":
        internal_attack = "model_scaling"
    elif attack_name == "committed_pixel":
        internal_attack = "backdoor_pixel"
    else:
        internal_attack = attack_name

    attack = get_attack(internal_attack)

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

        # Apply composition: d1 transform then d2 aggregate
        aggregated = generic_compose(server, updates, d1, d2, tau=5.0)
        server.apply_update(aggregated)
        current_lr *= getattr(FL_CONFIG, "lr_decay", 1.0)

    eval_result = server.evaluate(test_dataset)
    asr = evaluate_backdoor(server.global_model, test_dataset, device=device)
    return float(eval_result["accuracy"]), float(asr)


# --- Main ---
if __name__ == "__main__":
    total_runs = len(PAIRS) * len(ATTACKS) * len(SEEDS)
    print("=== Expanded Composability Criterion Validation (18 pairs) ===")
    print(f"  Defenses: {DEFENSES}")
    print(f"  Pairs: {len(PAIRS)} targeted ordered pairs")
    print(f"  Attacks: {ATTACKS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs} ({8*2*2} cached from prior run)")
    print(f"  FL config: N={FL_CONFIG.num_clients}, K={FL_CONFIG.clients_per_round}, "
          f"f={ADV_FRACTION}, rounds={FL_CONFIG.num_rounds}, cifar_cnn")
    print()

    t0 = time.time()
    completed = 0
    skipped = 0

    for pair_idx, (d1, d2) in enumerate(PAIRS):
        pk = pair_key(d1, d2)
        print(f"[{pair_idx+1}/{len(PAIRS)}] {pk}", flush=True)

        for attack in ATTACKS:
            for seed in SEEDS:
                if has_run(d1, d2, attack, seed):
                    skipped += 1
                    continue
                t_run = time.time()
                acc, asr = run_one(seed, d1, d2, attack)
                save_one(d1, d2, attack, seed, acc, asr)
                dt = time.time() - t_run
                completed += 1
                done_total = completed + skipped
                print(f"    {attack} seed={seed}: acc={acc:.3f} ASR={asr:.3f} "
                      f"({dt:.0f}s) [{done_total}/{total_runs}]", flush=True)

    # --- Summary Table ---
    print(f"\n{'='*70}")
    print("COMPOSABILITY TABLE: max committed ASR per pair (18 targeted pairs)")
    print(f"{'='*70}")
    print(f"{'Pair':<35} {'scaling ASR':>12} {'pixel ASR':>12} {'max ASR':>10}")
    print(f"{'-'*70}")

    s = load_or_init()
    for d1, d2 in PAIRS:
        pk = pair_key(d1, d2)
        if pk in s["pairs"]:
            pair_data = s["pairs"][pk]
            scaling_asr = pair_data.get("committed_scaling", {}).get("mean_asr", float("nan"))
            pixel_asr = pair_data.get("committed_pixel", {}).get("mean_asr", float("nan"))
            max_asr = pair_data.get("max_committed_asr", float("nan"))
            print(f"  {pk:<33} {scaling_asr:>10.3f}   {pixel_asr:>10.3f}   {max_asr:>8.3f}")
        else:
            print(f"  {pk:<33} {'N/A':>10}   {'N/A':>10}   {'N/A':>8}")

    print(f"\nCompleted: {completed} new runs, {skipped} cached")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")
    print(f"Output: {output_path}")
