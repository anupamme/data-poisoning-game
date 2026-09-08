"""
Multi-Krum is not a distinct defense at K=5, and this is what it costs the prospective suite.

`_krum(updates, multi=True, k=5)` selects `sorted(range(n), key=score)[:k]`. Every experiment
in this project runs `clients_per_round = 5` and nothing passes `k`, so the selection is ALL
FIVE clients and the call reduces to `_fedavg`. Verified at the bit level in
`experiments/verify_wave3_invariance.py` test (D): max|multi_krum - fedavg| = 1.9e-06 at K=5
and 2.5e+01 at K=10, i.e. a genuine selector again as soon as K > k.

Two separate degeneracies then compose, and the paper had only recognized one of them:

  (A) `krum` and `multi_krum` as $d_1$ are pass-through (`apply_d1_transform`, the
      trimmed_mean/coord_median/krum/multi_krum branch), so `krum -> X` measures X alone.
      PRE-EXISTING and already labelled DEGEN in tab:prospective.
  (B) `multi_krum` as $d_2$ is FedAvg, so `X -> mkrum` measures the $d_1$ transform followed by
      plain averaging -- which for every weighting or clipping $d_1$ in this menu IS that
      defense standalone. NEW, and the affected rows are labelled C1-fail / C2-fail rather
      than DEGEN because this was not known when the labels were frozen.

The consequence the body has to state is (B)'s effect on two claims:

  1. Two of the three "matched contrasts" of sec:prospective used MKrum as the non-invariant
     $d_2$, so they contained no downstream defense to contrast at all. Substituting the
     genuine selector `krum` does NOT repair them, which is the finding: the Krum arms take
     their max-committed ASR from the pixel backdoor, which Krum fails to suppress standalone
     (0.583), so the gap measures C1 failure rather than disturbed invariance -- and
     nc -> krum is byte-identical to Krum alone on all three seeds, because no client exceeds
     tau=5 under that attack (verify_wave3_invariance.py test F).
  2. Arm (iii) of the suite, "neither defense is FLTrust", is described as the only arm that
     could have surprised us (5/7). Under (A)+(B), 5 of its 7 pairs reduce to a single
     defense, leaving two genuine two-mechanism compositions.

Nothing here relabels a frozen pre-registered category: tab:prospective's labels stay verbatim,
as its caption promises and as main.tex's scoring rule requires. What changes is the body's
interpretation of those rows, disclosed as a correction.

Standalone comparisons use only SHARED seeds, recomputed from per-seed records, so a 5-seed
suite is never compared against a 3-seed one.

Output: results/mkrum_degeneracy.json
"""
import json
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

R = os.path.join(base_dir, "results")
OUT = os.path.join(R, "mkrum_degeneracy.json")

# Byte-identity tolerance. multi_krum and fedavg differ by ~1e-6 per aggregation (test D), so
# two 50-round runs that took the same code path agree to far better than ASR's granularity
# (1/900 of the poisoned test set). Anything above this is a different computation.
IDENTICAL = 1e-9


def load(*parts):
    with open(os.path.join(R, *parts)) as fh:
        return json.load(fh)


def per_seed_map(entry):
    """{seed: (asr, acc)} from any of the per_seed record shapes in results/."""
    return {r["seed"]: (r["asr"], r.get("accuracy", r.get("acc")))
            for r in entry["per_seed"]}


def keyed_cells(path):
    """A `cells` dict keyed "<pair>|<attack>" -> {attack: {seed: (asr, acc)}} by pair."""
    out = {}
    for key, v in load(path)["cells"].items():
        name, attack = key.split("|")
        out.setdefault(name, {})[attack] = per_seed_map(v)
    return out


def pair_suites():
    """Every `<d1>_then_<d2>` cell from the suites keyed by pair name -> {attack: {seed: ...}}.

    all_compositions and wave2_held_out both use the `pairs` shape; both run the protocol of
    interest (cifar10/cifar_cnn, N=10, K=5, f=0.2, 50 rounds, Dirichlet 0.5) at 5 seeds
    42--46, so only seeds 42--44 are ever compared against the 3-seed suites.
    """
    out = {}
    for path in (("all_compositions", "summary.json"), ("wave2_held_out", "summary.json")):
        for name, entry in load(*path)["pairs"].items():
            for attack in ATTACKS:
                if attack in entry:
                    out.setdefault(name, {})[attack] = per_seed_map(entry[attack])
    return out


def standalone_cells():
    """Single-defense ASR at this protocol, per seed, from the suites that measured it.

    `fedavg_then_X` is X alone with a pass-through upstream; prospective_pilot measured the
    three new defenses directly; pure_defense_baselines measured three more.
    """
    out = {}
    for name, cells in pair_suites().items():
        if name.startswith("fedavg_then_"):
            out[name[len("fedavg_then_"):]] = cells
    for key, v in load("prospective_pilot", "summary.json")["cells"].items():
        name, attack = key.split("|")
        out.setdefault(name, {})[attack] = per_seed_map(v)
    for name, entry in load("pure_defense_baselines", "summary.json")["defenses"].items():
        for attack, v in entry["adversary_policies"].items():
            out.setdefault(name, {}).setdefault(attack, per_seed_map(v))
    return out


def shared_mean(a, b):
    """(mean_a, mean_b, max|diff|, n_shared) over seeds both cells ran."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    xa = [a[s][0] for s in seeds]
    xb = [b[s][0] for s in seeds]
    return (sum(xa) / len(xa), sum(xb) / len(xb),
            max(abs(p - q) for p, q in zip(xa, xb)), len(seeds))


def max_committed(cell):
    """(max mean ASR over attacks, the attack achieving it, its mean accuracy)."""
    best = None
    for attack, seeds in cell.items():
        asr = sum(v[0] for v in seeds.values()) / len(seeds)
        acc = sum(v[1] for v in seeds.values()) / len(seeds)
        if best is None or asr > best[0]:
            best = (asr, attack, acc)
    return best


ATTACKS = ("committed_scaling", "committed_pixel")
PASS_THROUGH_D1 = ("krum", "multi_krum", "trimmed_mean", "coord_median", "fedavg")


if __name__ == "__main__":
    prosp = keyed_cells(os.path.join("prospective_suite", "summary.json"))
    mswap = keyed_cells(os.path.join("metric_swap", "summary.json"))
    alone = standalone_cells()
    check = load("wave3_invariance_check.json")

    # --- (1) the mechanism ------------------------------------------------------------
    deg = check["multi_krum_degeneracy"]
    k5 = max(r["max_abs_diff"] for r in deg if r["clients_per_round"] == 5)
    k10 = max(r["max_abs_diff"] for r in deg if r["clients_per_round"] == 10)
    print("=== (1) Mechanism (verify_wave3_invariance.py test D) ===")
    print(f"    K=5  max|multi_krum - fedavg| = {k5:.3e}   (degenerate)")
    print(f"    K=10 max|multi_krum - fedavg| = {k10:.3e}   (a genuine selector)")

    # --- (2) degeneracy (A): krum/mkrum as d1 -> d2 alone ----------------------------
    print("\n=== (2) Degeneracy (A): pass-through d1, so the row measures d2 alone ===")
    dup_rows = []
    for d2 in sorted({p.split("_then_")[1] for p in prosp}):
        a, b = f"krum_then_{d2}", f"multi_krum_then_{d2}"
        if a in prosp and b in prosp:
            cmp_ = {atk: shared_mean(prosp[a][atk], prosp[b][atk]) for atk in ATTACKS
                    if atk in prosp[a] and atk in prosp[b]}
            worst = max(c[2] for c in cmp_.values())
            dup_rows.append({"row_a": a, "row_b": b, "max_abs_asr_diff": worst,
                             "byte_identical": worst <= IDENTICAL})
            print(f"    {a:24s} vs {b:24s} max|dASR| = {worst:.3e}"
                  f"  identical: {worst <= IDENTICAL}")

    # --- (3) degeneracy (B): mkrum as d2 -> d1 alone ---------------------------------
    # X -> mkrum applies X's per-client transform and then averages. For a weighting d1 the
    # transform is u_i -> w_i * n * u_i, so averaging returns sum_i w_i u_i, which is exactly
    # that defense; for norm_clip it is clip-then-average, which is exactly NormClip. So each
    # such row should reproduce the standalone measurement on shared seeds.
    # The exact counterpart test is `d1 -> multi_krum` against `d1 -> fedavg`: identical code
    # path, identical seeds, only the d2 name swapped. Those two cells are the SAME FUNCTION
    # if the degeneracy holds, so whatever they differ by is 50 rounds of amplified float-order
    # divergence from a 1.9e-06 per-round perturbation -- a same-function noise floor, and the
    # only honest one available, since ASR is chaotic in the training trajectory.
    print("\n=== (3) Degeneracy (B): mkrum as d2, so the row measures d1 alone ===")
    pairs_all = pair_suites()
    mkrum_d2 = []
    for pair in sorted(p for p in prosp if p.endswith("_then_multi_krum")):
        d1 = pair.split("_then_")[0]
        counterpart = f"{d1}_then_fedavg"
        rows, cf = {}, {}
        for atk in ATTACKS:
            if atk not in prosp[pair]:
                continue
            if d1 in alone and atk in alone[d1]:
                rows[atk] = shared_mean(prosp[pair][atk], alone[d1][atk])
            if counterpart in pairs_all and atk in pairs_all[counterpart]:
                cf[atk] = shared_mean(prosp[pair][atk], pairs_all[counterpart][atk])
        mc = max_committed(prosp[pair])
        mkrum_d2.append({
            "pair": pair, "d1": d1, "max_committed_asr": mc[0],
            "max_committed_attack": mc[1], "max_committed_acc": mc[2],
            "vs_standalone": {atk: {"composition_mean_asr": v[0], "standalone_mean_asr": v[1],
                                    "max_abs_seed_diff": v[2], "n_shared_seeds": v[3]}
                              for atk, v in rows.items()},
            "vs_fedavg_counterpart": {atk: {"mkrum_mean_asr": v[0], "fedavg_mean_asr": v[1],
                                            "max_abs_seed_diff": v[2], "n_shared_seeds": v[3]}
                                      for atk, v in cf.items()},
            "fedavg_counterpart": counterpart if cf else None,
        })
        for atk, v in rows.items():
            print(f"    {pair:26s} {atk:18s} comp {v[0]:.4f} vs {d1} alone {v[1]:.4f}"
                  f"   max|dASR| {v[2]:.3e}  (n={v[3]})")
        for atk, v in cf.items():
            print(f"      counterpart {counterpart:22s} {atk:18s} "
                  f"mkrum {v[0]:.4f} vs fedavg {v[1]:.4f}  max|dASR| {v[2]:.3e} (n={v[3]})")
    same_fn = [v["max_abs_seed_diff"]
               for e in mkrum_d2 for v in e["vs_fedavg_counterpart"].values()]
    if same_fn:
        print(f"    -> same-function float-order noise floor: max per-seed |dASR| = "
              f"{max(same_fn):.4f} over {len(same_fn)} cells")

    # --- (4) the three matched contrasts ---------------------------------------------
    # Held d1 fixed, varied d2 between FLTrust (exactly invariant statistic) and a Euclidean
    # rule. Substituting krum for mkrum is not a repair, and the decomposition shows why: the
    # Krum arm's max-committed ASR is compared against KRUM'S OWN standalone ASR on that same
    # attack, which is what C1 would have to hold fixed for the gap to be about invariance.
    print("\n=== (4) The three matched contrasts, with the genuine selector krum ===")
    contrasts = []
    for d1, label in (("reputation", "Rep"), ("norm_clip", "NC"), ("rfa", "RFA")):
        inv = max_committed(prosp[f"{d1}_then_fltrust"])
        src = prosp if f"{d1}_then_krum" in prosp else mswap
        suite = "prospective_suite" if src is prosp else "metric_swap"
        cell = src[f"{d1}_then_krum"]
        noninv = max_committed(cell)
        atk = noninv[1]
        krum_alone = shared_mean(cell[atk], alone["krum"][atk])
        old_key = f"{d1}_then_multi_krum"
        old = max_committed(prosp[old_key])[0] if old_key in prosp else None
        contrasts.append({
            "d1": d1, "label": label,
            "invariant_d2": "fltrust", "invariant_asr": inv[0],
            "noninvariant_d2": "krum", "noninvariant_asr": noninv[0],
            "noninvariant_attack": atk, "noninvariant_acc": noninv[2],
            "noninvariant_source": suite, "ratio": noninv[0] / inv[0],
            "krum_standalone_asr_same_attack": krum_alone[1],
            "krum_suppresses_alone_here": krum_alone[1] < 0.5,
            "excess_over_krum_alone": noninv[0] - krum_alone[1],
            "max_abs_seed_diff_vs_krum_alone": krum_alone[2],
            "byte_identical_to_krum_alone": krum_alone[2] <= IDENTICAL,
            "superseded_multi_krum_asr": old,
            "superseded_ratio": (old / inv[0] if old is not None else None),
        })
        c = contrasts[-1]
        print(f"    {label:4s}: fltrust {inv[0]:.3f}  vs krum {noninv[0]:.3f}"
              f" ({atk.replace('committed_','')}, acc {noninv[2]:.3f}) = {c['ratio']:.1f}x"
              f"  [{suite}]")
        print(f"          krum ALONE on that attack {krum_alone[1]:.4f}"
              f" -> excess {c['excess_over_krum_alone']:+.4f},"
              f" max|dASR| {krum_alone[2]:.3e}, identical: "
              f"{c['byte_identical_to_krum_alone']}")
        if old is not None:
            print(f"          (superseded: {label}->mkrum {old:.3f}, "
                  f"{c['superseded_ratio']:.1f}x -- mkrum was FedAvg)")
    ratios = [c["ratio"] for c in contrasts]
    n_genuine = sum(1 for c in contrasts if c["krum_suppresses_alone_here"])
    print(f"    -> {n_genuine}/3 contrasts have a d2 that suppresses its own max-committed "
          f"attack standalone")

    # --- (5) arm (iii) census: what each pair actually measures ----------------------
    print("\n=== (5) Arm (iii) of tab:prospective, 'neither defense is FLTrust' ===")
    arm3 = [p for p in prosp
            if "fltrust" not in p and p.split("_then_")[0] != "fltrust"]
    census = []
    for pair in sorted(arm3, key=lambda p: max_committed(prosp[p])[0]):
        d1, d2 = pair.split("_then_")
        if d1 in PASS_THROUGH_D1:
            reduces_to, why = d2, f"{d1} as d1 is pass-through"
        elif d2 == "multi_krum":
            reduces_to, why = d1, "multi_krum as d2 is FedAvg at K=5"
        else:
            reduces_to, why = None, None
        mc = max_committed(prosp[pair])
        census.append({"pair": pair, "max_committed_asr": mc[0],
                       "reduces_to_single_defense": reduces_to, "reason": why})
        tag = f"= {reduces_to} alone ({why})" if reduces_to else "GENUINE two-mechanism"
        print(f"    {pair:28s} {mc[0]:.3f}   {tag}")
    n_genuine_pairs = sum(1 for c in census if c["reduces_to_single_defense"] is None)
    print(f"    -> {n_genuine_pairs} of {len(census)} pairs compose two real mechanisms")

    # --- (6) how many of the 20 rows are degenerate ----------------------------------
    all_deg = []
    for pair in sorted(prosp):
        d1, d2 = pair.split("_then_")
        if d1 in PASS_THROUGH_D1:
            all_deg.append((pair, f"{d1} pass-through as d1"))
        elif d2 == "multi_krum":
            all_deg.append((pair, "multi_krum == fedavg as d2"))
    print(f"\n=== (6) Degenerate rows in the 20-row suite: {len(all_deg)} ===")
    for pair, why in all_deg:
        print(f"    {pair:28s} {why}")

    # --- (7) the same-function noise floor, and why it needs the accuracy gate ---------
    # Every cell below computes the same function twice: multi_krum against fedavg, at matched
    # seeds. The spread is therefore pure float-order divergence amplified over 50 rounds. It
    # is regime-dependent, and that is the point: where the model has not collapsed the floor
    # is small, and where it has (clean accuracy 0.10 under model-scaling) ASR is chaotic and
    # the same function disagrees with itself by an order more. So the accuracy gate is a
    # precondition for reading small ASR differences, not only for reading suppression claims.
    ACC_FLOOR = 0.35
    floor = []
    for pair in sorted(p for p in prosp if p.endswith("_then_multi_krum")):
        cp = f"{pair.split('_then_')[0]}_then_fedavg"
        if cp not in pairs_all:
            continue
        for atk in ATTACKS:
            if atk in prosp[pair] and atk in pairs_all[cp]:
                seeds = sorted(set(prosp[pair][atk]) & set(pairs_all[cp][atk]))
                acc = min(min(prosp[pair][atk][s][1], pairs_all[cp][atk][s][1]) for s in seeds)
                d = max(abs(prosp[pair][atk][s][0] - pairs_all[cp][atk][s][0]) for s in seeds)
                floor.append({"cells": [pair, cp], "attack": atk, "max_abs_seed_diff": d,
                              "min_acc": acc, "above_acc_floor": acc >= ACC_FLOOR})
    for atk in ATTACKS:
        a, b = alone["multi_krum"][atk], alone["fedavg"][atk]
        seeds = sorted(set(a) & set(b))
        floor.append({"cells": ["multi_krum alone", "fedavg alone"], "attack": atk,
                      "max_abs_seed_diff": max(abs(a[s][0] - b[s][0]) for s in seeds),
                      "min_acc": min(min(a[s][1], b[s][1]) for s in seeds),
                      "above_acc_floor": min(min(a[s][1], b[s][1]) for s in seeds) >= ACC_FLOOR})
    gated = [f["max_abs_seed_diff"] for f in floor if f["above_acc_floor"]]
    ungated = [f["max_abs_seed_diff"] for f in floor if not f["above_acc_floor"]]
    print(f"\n=== (7) Same-function float-order floor (multi_krum vs fedavg, matched seeds) ===")
    for f in floor:
        print(f"    {f['cells'][0]:26s} vs {f['cells'][1]:22s} {f['attack']:18s} "
              f"max|dASR| {f['max_abs_seed_diff']:.4f}  min acc {f['min_acc']:.3f}"
              f"  gated: {f['above_acc_floor']}")
    print(f"    -> acc >= {ACC_FLOOR}: floor {max(gated):.4f} over {len(gated)} cells")
    if ungated:
        print(f"    -> acc <  {ACC_FLOOR} (collapsed model): {max(ungated):.4f} over "
              f"{len(ungated)} cells, so ASR differences there are unreadable")

    results = {
        "description": "multi_krum reduces to fedavg at clients_per_round <= k (default 5): "
                       "consequences for tab:prospective's rows and matched contrasts",
        "identity_tolerance": IDENTICAL,
        "mechanism": {"k_default": 5, "clients_per_round_in_all_experiments": 5,
                      "max_abs_diff_K5": k5, "max_abs_diff_K10": k10,
                      "degenerate_at_K5": check["multi_krum_degenerate_at_K5"],
                      "distinct_at_K10": check["multi_krum_distinct_at_K10"],
                      "source": "results/wave3_invariance_check.json (test D)"},
        "degeneracy_A_passthrough_d1_duplicate_rows": dup_rows,
        "degeneracy_B_mkrum_as_d2_vs_standalone": mkrum_d2,
        "matched_contrasts": contrasts,
        "ratio_range": [min(ratios), max(ratios)],
        "superseded_ratio_range": [
            min(c["superseded_ratio"] or c["ratio"] for c in contrasts),
            max(c["superseded_ratio"] or c["ratio"] for c in contrasts)],
        "n_contrasts_with_d2_suppressing_alone": n_genuine,
        "arm_iii_census": census,
        "arm_iii_n_pairs": len(census),
        "arm_iii_n_genuine_two_mechanism": n_genuine_pairs,
        "same_function_noise_floor": {
            "acc_floor": ACC_FLOOR, "cells": floor,
            "floor_above_acc_gate": max(gated) if gated else None,
            "floor_below_acc_gate": max(ungated) if ungated else None,
        },
        "degenerate_rows_in_suite": [{"pair": p, "reason": w} for p, w in all_deg],
        "n_degenerate_rows": len(all_deg),
        "n_rows": len(prosp),
    }
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {OUT}")
