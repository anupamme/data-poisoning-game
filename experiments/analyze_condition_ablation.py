"""
Factorial ablation of C1/C2/C3 over the 42 already-run two-way compositions.

Reviewer question (Round 9): "Why do I need all three? Perhaps C1 alone is doing most
of the work." The paper's stored labels record only the FIRST failing condition, which
cannot answer that. Here each pair is scored on all three conditions INDEPENDENTLY:

  C1  from measured single-defense baselines (fedavg->X == X alone, same protocol):
      for EVERY committed attack a, min over {d1,d2} of ASR(d,a) < 0.5.
  C2  from Proposition 1's invariance classes -- determined by what statistic d2 reads
      and whether d1 is a positive per-client rescaling:
        cosine similarity (foolsgold, fltrust)      -> invariant, C2 always holds
        coordinate ordering (trimmed_mean, coord_median) -> conditional (see C3)
        consensus distance (reputation) / residual (rfa) / pairwise distance (krum)
                                                     -> not invariant under heterogeneous w
        norm magnitude (norm_clip)                   -> not invariant
      Degenerate d1 (an aggregator upstream) emits no per-client signal: C2 vacuously fails.
  C3  margin preservation, taken from the per-pair reasoning stored with each label.

THE C2 COLUMN IS NOT ALL ONE KIND OF EVIDENCE, AND THE PAPER OWES THAT DISCLOSURE.
Two of the four C2 branches are algebra; the conditional classes (`coord_median`,
`trimmed_mean`, `rfa`) fall through to `stored_cat`, i.e. to a per-pair judgement. And every
one of those fall-through rows is a DEV-SET pair -- `CRITERION_PREDICTIONS` marks them
"Dev set (18 pairs) -- from Table 1" -- so the judgement is IN-SAMPLE, made while the criterion
was being written, not pre-registered. (The prospective pre-registrations are separate files and
separate pair lists.) NOTE a wording error in a frozen file, not editable here: §3(E) of
`experiments/pre_registration_wave3_emergent.md` @ 96f8856 calls these "the pre-registered
per-pair call". They are not pre-registered; the split field below shows they are dev-set. The
prereg's substantive point -- that strategy A is not a pure statistic-preservation screen for
those three d2 classes -- is unaffected and in fact strengthened. Since the strategy
comparison labels the C2-alone variant "statistic
preservation", `c2_provenance` records per row which kind of evidence its verdict rests on, and
the census at the end scores C2 restricted to the algebraically settled region. Emitting it here
rather than in the frontier script keeps the semantics with the table that defines them.

C1'S QUANTIFIER IS NOT THE SAME AS C1'S DIRECTION, AND SUBSTITUTING ONE FOR THE OTHER IS NOT FREE
The last section prices a proposed substitution of C1 by the directional C1' ("the DOWNSTREAM d2
suppresses a"), which is often assumed equivalent on the grounds that C0 forces d2 anyway. It does
not: C0 is existential over attacks while C1 is universal over them, so Lemma 2 pins d2 only on the
witnessing attack a*. Off a*, the suppressing constituent may be d1 -- and on all four pairs where
the two definitions disagree, d2 is coord_median (model-scaling 0.519, above threshold) while the
upstream foolsgold (0.200) or reputation (0.017) is what holds that arm. Both pairs the criterion
certifies are of that kind, so C1' would empty the certified set and leave precision undefined.
Reported as a scope fact; the paper keeps C1.

No new compute. Output: results/condition_ablation/summary.json
"""
import json
import os
import sys
from itertools import product

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from experiments.analyze_continuous_metric import CRITERION_PREDICTIONS

ATTACKS = ("committed_scaling", "committed_pixel")
THRESH = 0.5

# ---- load actuals -------------------------------------------------------------
pairs = {}
# Which split a pair came from is not recorded in CRITERION_PREDICTIONS (only as a source
# comment), but it is recoverable from WHICH artifact holds the pair -- and the provenance
# census below needs it, because an in-sample per-pair judgement is weaker evidence than a
# held-out one. Read here so the claim is derived rather than asserted.
SPLIT_OF = {}
for path, split in (("results/all_compositions/summary.json", "dev"),
                    ("results/wave2_held_out/summary.json", "held_out")):
    for k, v in json.load(open(path))["pairs"].items():
        pairs[k] = {a: v[a]["mean_asr"] for a in ATTACKS if a in v}
        SPLIT_OF.setdefault(k, split)

def max_committed(k):
    return max(pairs[k].values())

# single-defense baselines: fedavg->X is X alone under the identical protocol
single = {k.replace("fedavg_then_", ""): v for k, v in pairs.items() if k.startswith("fedavg_then_")}
single["fedavg"] = pairs["fedavg_then_norm_clip"]  # placeholder, replaced below if present

# ---- condition semantics ------------------------------------------------------
# what statistic each defense reads as d2, and whether it survives heterogeneous
# positive per-client rescaling (Proposition 1)
SIGNAL_INVARIANT = {
    "foolsgold": True,      # pairwise cosine  -- exactly invariant  (Prop 1a)
    "fltrust": True,        # cosine to server reference -- invariant (Prop 1a)
    "coord_median": None,   # coordinate ordering -- conditional      (Prop 1b)
    "trimmed_mean": None,   # coordinate ordering -- conditional      (Prop 1b)
    "reputation": False,    # consensus distance -- not invariant     (Prop 1c)
    # residual magnitude is not invariant in general (Prop 1c), but Theorem 1(2) gives an
    # explicit separation condition under which RFA's suppression IS preserved -- so this is
    # a conditional class like coordinate ordering, not an automatic C2 failure.
    "rfa": None,
    "norm_clip": False,     # norm magnitude     -- not invariant
    "krum": False,          # pairwise distance  -- not invariant     (Prop 1c)
    "multi_krum": False,
    "fedavg": False,        # no discriminative signal at all
}
# d1 that emit an aggregate rather than per-client updates
DEGENERATE_D1 = {"coord_median", "trimmed_mean", "fedavg", "krum", "multi_krum"}


def c1_holds(d1, d2):
    """For EVERY committed attack, at least one constituent suppresses it."""
    for a in ATTACKS:
        best = min(single.get(d1, {}).get(a, 1.0), single.get(d2, {}).get(a, 1.0))
        if best >= THRESH:
            return False
    return True


def c2_holds(d1, d2, stored_cat):
    if d1 in DEGENERATE_D1:
        return False                      # no per-client signal reaches d2
    inv = SIGNAL_INVARIANT.get(d2)
    if inv is True:
        return True                       # Proposition 1(a)
    if inv is False:
        return False                      # Proposition 1(c)
    return stored_cat != "C2-FAIL"        # conditional class: defer to the stored per-pair call


def c2_provenance(d1, d2):
    """WHERE a row's C2 verdict comes from, which is not the same thing as the verdict.

    C2 is described throughout as read off the two defenses' definitions, and for two of the
    four branches above that is literally true. It is not true for the conditional classes:
    `coord_median`, `trimmed_mean` and `rfa` are `None` in SIGNAL_INVARIANT, so `c2_holds`
    falls through to `stored_cat != "C2-FAIL"` and the verdict is A PER-PAIR CALL, not an
    invariance theorem -- and, since every fall-through row is a dev-set pair, a call made
    IN-SAMPLE. A row can therefore carry C2=True on judgement where another carries it on
    algebra, and nothing in the C2 column distinguishes them.

    This matters because the strategy comparison labels the C2-alone column "statistic
    preservation". For the conditional classes that label overstates what was computed, so the
    provenance is emitted per row and censused below rather than left implicit.
    """
    if d1 in DEGENERATE_D1:
        return "degenerate_d1"                 # vacuous fail, no per-client signal at all
    inv = SIGNAL_INVARIANT.get(d2)
    if inv is True:
        return "algebraic_invariant"           # Prop 1(a), exactly invariant
    if inv is False:
        return "algebraic_not_invariant"       # Prop 1(c)
    return "per_pair_call"                 # Prop 1(b) / Thm 1(2) conditional class


def c3_holds(stored_cat):
    return stored_cat != "C3-FAIL"


if __name__ == "__main__":
    rows = []
    for k, (cat, _) in CRITERION_PREDICTIONS.items():
        if k not in pairs:
            continue
        d1, d2 = k.split("_then_")
        r = {"pair": k, "d1": d1, "d2": d2, "stored_category": cat,
             "C1": c1_holds(d1, d2), "C2": c2_holds(d1, d2, cat), "C3": c3_holds(cat),
             "max_committed_asr": max_committed(k), "low_asr": max_committed(k) < THRESH,
             "c2_provenance": c2_provenance(d1, d2), "split": SPLIT_OF.get(k)}
        rows.append(r)

    print(f"=== C1/C2/C3 FACTORIAL ABLATION over {len(rows)} pairs ===\n")
    print(f"{'C1':>3} {'C2':>3} {'C3':>3} | {'#pairs':>6} {'low ASR':>8} {'rate':>7}")
    print("-" * 44)
    cells = {}
    for c1, c2, c3 in product([True, False], repeat=3):
        sub = [r for r in rows if (r["C1"], r["C2"], r["C3"]) == (c1, c2, c3)]
        if not sub:
            continue
        lo = sum(r["low_asr"] for r in sub)
        cells[f"C1={c1},C2={c2},C3={c3}"] = {"n": len(sub), "low": lo, "rate": lo / len(sub)}
        print(f"{str(c1):>3} {str(c2):>3} {str(c3):>3} | {len(sub):6d} {lo:8d} {lo/len(sub):7.0%}")

    print("\n=== MARGINAL VALUE OF EACH CONDITION ===")
    marg = {}
    for name, sel in [("C1 alone", lambda r: r["C1"]),
                      ("C2 alone", lambda r: r["C2"]),
                      ("C3 alone", lambda r: r["C3"]),
                      ("C1 and C2", lambda r: r["C1"] and r["C2"]),
                      ("C1 and C3", lambda r: r["C1"] and r["C3"]),
                      ("C2 and C3", lambda r: r["C2"] and r["C3"]),
                      ("C1 and C2 and C3", lambda r: r["C1"] and r["C2"] and r["C3"])]:
        sub = [r for r in rows if sel(r)]
        lo = sum(r["low_asr"] for r in sub) if sub else 0
        marg[name] = {"n": len(sub), "low": lo, "precision": (lo / len(sub)) if sub else None}
        p = f"{lo/len(sub):.0%}" if sub else "n/a"
        print(f"  {name:20s} selects {len(sub):3d} pairs, {lo:2d} low-ASR -> precision {p}")

    total_low = sum(r["low_asr"] for r in rows)
    print(f"\n  (total low-ASR pairs in the set: {total_low}/{len(rows)}; "
          f"base rate {total_low/len(rows):.0%})")

    # ---- where the C2=True labels actually come from -------------------------------
    # The C2-alone variant is the high-recall end of the frontier and the strategy comparison
    # prints it as "statistic preservation". This census is the scope limit on that reading.
    admitted = [r for r in rows if r["C2"]]
    by_prov = {}
    for r in admitted:
        by_prov.setdefault(r["c2_provenance"], []).append(r)
    algebraic = by_prov.get("algebraic_invariant", [])
    per_pair = by_prov.get("per_pair_call", [])
    lows = [r for r in rows if r["low_asr"]]
    low_algebraic = [r for r in lows if r["c2_provenance"] == "algebraic_invariant"]

    print("\n=== PROVENANCE OF THE C2=True LABELS (scope limit on 'statistic preservation') ===")
    print(f"  C2 admits {len(admitted)} of {len(rows)} pairs. Of those:")
    print(f"    {len(algebraic):2d} on ALGEBRAIC invariance (Prop 1(a); d2 reads a cosine)")
    n_dev = sum(1 for r in per_pair if r["split"] == "dev")
    print(f"    {len(per_pair):2d} on the PER-PAIR CALL (conditional class: "
          f"{', '.join(sorted({r['d2'] for r in per_pair}))})")
    for r in sorted(per_pair, key=lambda r: r["pair"]):
        print(f"       {r['pair']:34s} stored={r['stored_category']:8s} split={r['split']:8s} "
              f"ASR {r['max_committed_asr']:.3f}{'  LOW' if r['low_asr'] else ''}")
    print(f"    -> {n_dev} of those {len(per_pair)} are DEV-SET pairs, so the call was made "
          f"IN-SAMPLE,")
    print("       while the criterion was being written. It is not a pre-registered call.")
    print(f"\n  And every one of the {len(lows)} low-ASR pairs has a conditional-class d2: "
          f"{len(low_algebraic)} of {len(lows)} rest on algebra.")
    # The restricted variant is the honest reading of "statistic preservation" as the paper's
    # own Proposition 1 licenses it: keep only pairs whose C2 is algebraically settled.
    tp_restricted = sum(1 for r in algebraic if r["low_asr"])
    prec_r = f"{tp_restricted/len(algebraic):.0%}" if algebraic else "n/a"
    rec_r = f"{tp_restricted/len(lows):.0%}" if lows else "n/a"
    print(f"  So C2 RESTRICTED to the algebraically settled region selects {len(algebraic)}, "
          f"precision {tp_restricted}/{len(algebraic)} = {prec_r}, "
          f"recall {tp_restricted}/{len(lows)} = {rec_r}.")
    print("  The 100% recall of C2-alone is therefore carried ENTIRELY by the in-sample call,")
    print("  and the algebraically settled sub-region discriminates nothing. Disclosed, not fixed:")
    print("  fixing it needs an invariance result for the conditional classes, which we do not have.")

    # ---- C1's quantifier: would a DIRECTIONAL C1 move any label? -------------------
    # A reviewer asked us to replace C1's "some constituent suppresses a" with the directional
    # "the downstream d2 suppresses a", on the grounds that C0 forces d2 anyway so the change is
    # notational. C0 does NOT force it: C0 is existential over attacks ("ASR(d1,a) >= 0.5 for at
    # least one committed a") while C1 is universal over them, so Lemma 2 pins d2 only on the
    # witnessing attack a*. Off a*, the suppressing constituent may be d1. This census prices the
    # substitution instead of arguing about it.
    def c1_directional(d2):
        """d2 alone suppresses EVERY committed attack -- the reviewer's C1'."""
        return all(single.get(d2, {}).get(a, 1.0) < THRESH for a in ATTACKS)

    def witnesses_c0(d1):
        """Attacks a with ASR(d1,a) >= 0.5; any one of them can serve as a*."""
        return [a for a in ATTACKS if single.get(d1, {}).get(a, 1.0) >= THRESH]

    flips = [r for r in rows if r["C1"] != c1_directional(r["d2"])]
    print("\n=== C1's QUANTIFIER: SYMMETRIC vs DIRECTIONAL (would C1' move a label?) ===")
    print("    C1  (as used):  min over {d1,d2} of ASR(d,a) < 0.5, for every committed a")
    print("    C1' (proposed): ASR(d2,a) < 0.5, for every committed a")
    print(f"\n  The two disagree on {len(flips)} of {len(rows)} pairs, and never in the other")
    print("  direction: C1' is strictly stronger, so every disagreement is a pair C1 admits")
    print("  and C1' rejects.\n")
    for r in sorted(flips, key=lambda r: r["pair"]):
        d1a = {a: single.get(r["d1"], {}).get(a, 1.0) for a in ATTACKS}
        d2a = {a: single.get(r["d2"], {}).get(a, 1.0) for a in ATTACKS}
        # the attack C1 satisfies via d1 rather than d2 is what C1' loses
        via_d1 = [a for a in ATTACKS if d1a[a] < THRESH <= d2a[a]]
        print(f"  {r['pair']:34s} stored={r['stored_category']:8s} "
              f"{'LOW' if r['low_asr'] else 'HIGH':4s} ASR {r['max_committed_asr']:.3f}")
        print(f"      d1={r['d1']:14s} " + "  ".join(f"{a.replace('committed_','')}={d1a[a]:.3f}"
                                                    for a in ATTACKS))
        print(f"      d2={r['d2']:14s} " + "  ".join(f"{a.replace('committed_','')}={d2a[a]:.3f}"
                                                    for a in ATTACKS))
        print(f"      C0 witnessed by {[a.replace('committed_','') for a in witnesses_c0(r['d1'])]}"
              f"; C1 met via d1 (not d2) on "
              f"{[a.replace('committed_','') for a in via_d1]}")

    clean = [r for r in rows if r["C1"] and r["C2"] and r["C3"]]
    clean_dir = [r for r in rows if c1_directional(r["d2"]) and r["C2"] and r["C3"]]
    tp, tp_dir = sum(r["low_asr"] for r in clean), sum(r["low_asr"] for r in clean_dir)
    print(f"\n  Clean C1^C2^C3 selects {len(clean)} pairs "
          f"({', '.join(sorted(r['pair'] for r in clean))}), "
          f"precision {tp}/{len(clean)}, recall {tp}/{len(lows)}.")
    print(f"  Clean C1'^C2^C3 selects {len(clean_dir)} pairs"
          f"{' (' + ', '.join(sorted(r['pair'] for r in clean_dir)) + ')' if clean_dir else ''}, "
          f"precision {tp_dir}/{len(clean_dir) if clean_dir else 0}"
          f"{' -- UNDEFINED' if not clean_dir else ''}, recall {tp_dir}/{len(lows)}.")
    if clean and not clean_dir:
        print("  => The proposed substitution EMPTIES the certified set. Both certified pairs are")
        print("     pairs whose model-scaling arm is held by the UPSTREAM defense, and the same")
        print("     precision it would delete is the statistic the same review asks us to lead")
        print("     with. We therefore keep C1 and disclose the asymmetry rather than adopt C1'.")

    c1_quantifier = {
        "definition_symmetric": "min_{d in {d1,d2}} ASR(d,a) < 0.5 for every committed a",
        "definition_directional": "ASR(d2,a) < 0.5 for every committed a",
        "n_pairs": len(rows),
        "n_disagreements": len(flips),
        "disagreements": [
            {"pair": r["pair"], "d1": r["d1"], "d2": r["d2"],
             "stored_category": r["stored_category"], "low_asr": r["low_asr"],
             "d1_standalone": {a: single.get(r["d1"], {}).get(a) for a in ATTACKS},
             "d2_standalone": {a: single.get(r["d2"], {}).get(a) for a in ATTACKS},
             "c0_witnesses": witnesses_c0(r["d1"]),
             "c1_met_via_d1_on": [a for a in ATTACKS
                                  if single.get(r["d1"], {}).get(a, 1.0) < THRESH
                                  <= single.get(r["d2"], {}).get(a, 1.0)]}
            for r in sorted(flips, key=lambda r: r["pair"])],
        "clean_symmetric": {"pairs": sorted(r["pair"] for r in clean),
                            "true_positives": tp, "n_low_asr": len(lows)},
        "clean_directional": {"pairs": sorted(r["pair"] for r in clean_dir),
                              "true_positives": tp_dir, "n_low_asr": len(lows)},
    }

    provenance = {
        "c2_admitted": len(admitted),
        "n_pairs": len(rows),
        "by_provenance": {k: len(v) for k, v in sorted(by_prov.items())},
        "conditional_class_d2": sorted({r["d2"] for r in per_pair}),
        "per_pair_call_splits": {s: sum(1 for r in per_pair if r["split"] == s)
                                 for s in sorted({r["split"] for r in per_pair})},
        "n_low_asr": len(lows),
        "n_low_asr_algebraic": len(low_algebraic),
        "c2_restricted_to_algebraic": {
            "selected": len(algebraic),
            "true_positives": tp_restricted,
            "precision": (tp_restricted / len(algebraic)) if algebraic else None,
            "recall": (tp_restricted / len(lows)) if lows else None,
        },
    }

    os.makedirs("results/condition_ablation", exist_ok=True)
    json.dump({"description": "Factorial C1/C2/C3 ablation over the 42 evaluated pairs",
               "threshold": THRESH, "cells": cells, "marginal": marg,
               "base_rate_low_asr": total_low / len(rows), "rows": rows,
               "c2_provenance": provenance, "c1_quantifier": c1_quantifier},
              open("results/condition_ablation/summary.json", "w"), indent=2)
    print("\nSaved to results/condition_ablation/summary.json")
