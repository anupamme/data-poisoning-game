"""
Score Wave 3 against its own pre-registration. Written BEFORE the data landed.

WHY THIS FILE EXISTS SEPARATELY FROM THE RUNNER
`run_wave3_emergent.py` deliberately writes ASR and accuracy and stops there: it computes no
verdict. Scoring lives here so that the rule can be read, and this file can be dated, against
`experiments/pre_registration_wave3_emergent.md` rather than against the numbers. Every constant
below is transcribed from §5 and §6 of that file, which is frozen at a commit this script checks.

WHAT IS TRANSCRIBED AND WHAT IS RE-DERIVED
Transcribed (from the frozen prereg, and nothing else): the four pairs, their predictions, their
cell roles, the confidence flags, the 0.5 outcome threshold, the 0.35 accuracy floor, the FedAvg
sanity reference, and §7's four-way resolution table. RE-DERIVED from artifacts rather than
trusted: every mean ASR (recomputed from per-seed and cross-checked against the stored mean), the
seven frozen census cells (read from their own summary.json files and cross-checked against the
values §2.1 transcribes), and the d2-alone per-seed baselines the near-identity check needs.

THE FOUR GATES OF §6, ALL APPLIED, NONE OPTIONAL
  1. outcome    max-committed mean ASR over seeds 42/43/44; LOW iff < 0.5.
  2. acc gate   mean clean accuracy < 0.35 ON THE ARM SUPPLYING the max -> DEGENERATE-ACC,
                scored HIGH. A collapsed model is not a defended one.
  3. sanity     ASR identically 0.0 or 1.0 across all three seeds, or accuracy above the FedAvg
                reference by more than seed noise -> flagged for a harness check. ADVISORY: this
                gate never changes a score, it only demands a look.
  4. near-id    per-seed comparison against d2 alone. An arm identical in 3 of 3 seeds is
                NEAR-IDENTITY and is not evidence about composition. §6.4 makes this mandatory,
                and §2.1 shows it already caught two census cells.

Reads:  results/wave3_emergent/summary.json                (the wave)
        results/metric_swap_baselines/summary.json          (d2 alone, per seed)
        results/metric_swap/summary.json                    (4 frozen census cells)
        results/all_compositions/summary.json               (2 frozen census cells)
        results/wave2_held_out/summary.json                 (1 frozen census cell)
Writes: results/wave3_emergent/scored.json

Run: python3 experiments/analyze_wave3_emergent.py
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = "experiments/pre_registration_wave3_emergent.md"
WAVE = os.path.join(BASE, "results", "wave3_emergent", "summary.json")
SCORED = os.path.join(BASE, "results", "wave3_emergent", "scored.json")

ATTACKS = ("committed_scaling", "committed_pixel")
SEEDS = (42, 43, 44)

# ---- §6.1 and §6.2: the only thresholds scored, frozen before the data existed -------------
THRESH = 0.5            # LOW iff max-committed mean ASR < 0.5
ACC_FLOOR = 0.35        # below this on the max-supplying arm: DEGENERATE-ACC, scored HIGH

# ---- §6.3: the FedAvg reference the sanity gate compares against ---------------------------
FEDAVG_ACC_REF = {"committed_scaling": 0.667, "committed_pixel": 0.768}
# "by more than seed noise" is not a number in the prereg, so the gate is advisory only and its
# margin is stated rather than tuned: 2 sigma of the cell's own across-seed spread, floored at
# 0.05 so a low-variance cell cannot trip it on nothing. NOTHING here changes a verdict.
SANITY_FLOOR_MARGIN = 0.05

# ---- §5: the four pairs, verbatim ----------------------------------------------------------
FROZEN = {
    "reputation_then_cos_krum": {
        "n": 1, "predicted": "LOW", "confidence": "low (uncertain, see prereg 5.1)",
        "cell": "certified (C1 and C2)", "role": "tests strategy B's precision",
        "c1": "holds (per attack, gated)", "c2": "holds (measured 0/9)",
    },
    "reputation_then_cos_reputation": {
        "n": 2, "predicted": "HIGH", "confidence": "medium",
        "cell": "C1-fail and C2-true", "role": "tests strategy B's recall, the discriminating cell",
        "c1": "fails (pixel: rep 0.8422, cos_rep 0.7543)", "c2": "holds (measured 0/9)",
    },
    "foolsgold_then_cos_krum": {
        "n": 3, "predicted": "HIGH", "confidence": "high",
        "cell": "measured-C2-fail control", "role": "shows the C2 probe is not inert",
        "c1": "fails (both arms gated)", "c2": "FAILS (measured 9/9)",
    },
    "foolsgold_then_cos_reputation": {
        "n": 4, "predicted": "HIGH", "confidence": "high",
        "cell": "measured-C2-fail control", "role": "shows the C2 probe is not inert",
        "c1": "fails (both arms)", "c2": "FAILS (measured 9/9)",
    },
}
# Strategy A = C2 alone; strategy B = C1 and C2. §5, line "Strategy A selects pairs 1 and 2".
SELECTED_BY = {"A_statistic_preservation": ["reputation_then_cos_krum",
                                            "reputation_then_cos_reputation"],
               "B_criterion": ["reputation_then_cos_krum"]}

# ---- §2.1: the seven frozen census cells, with the artifact each must be read from ---------
# The transcribed value is kept ONLY as a cross-check target; the script reads the artifact.
CENSUS_FROZEN = {
    ("norm_clip", "foolsgold"):       (0.8869, "results/all_compositions/summary.json"),
    ("reputation", "foolsgold"):      (0.8070, "results/all_compositions/summary.json"),
    ("rfa", "foolsgold"):             (0.9135, "results/wave2_held_out/summary.json"),
    ("norm_clip", "cos_krum"):        (0.5085, "results/metric_swap/summary.json"),
    ("norm_clip", "cos_reputation"):  (0.9346, "results/metric_swap/summary.json"),
    ("rfa", "cos_krum"):              (0.5657, "results/metric_swap/summary.json"),
    ("rfa", "cos_reputation"):        (0.8612, "results/metric_swap/summary.json"),
}
CENSUS_D1 = ("norm_clip", "rfa", "reputation")
CENSUS_D2 = ("foolsgold", "cos_krum", "cos_reputation")

# ---- §7: two-sided resolution, verbatim. Keyed (pair 1 outcome, pair 2 outcome). ----------
RESOLUTION = {
    ("LOW", "HIGH"): ("Both strategies correct.",
                      "B keeps 100% precision out of sample; A pays one extra pair-evaluation "
                      "for no additional true positive. The census closes 9/9 with exactly one "
                      "LOW, the one B selects."),
    ("LOW", "LOW"): ("A wins on recall.",
                     "Emergent suppression exists in the C1-fail and C2-true cell; B misses it, "
                     "exactly as its 40% recall predicts. Report as the first PROSPECTIVE "
                     "demonstration that B's recall gap is real and costly."),
    ("HIGH", "HIGH"): ("B's precision fails out of sample.",
                       "The one certified pair in the census is a false positive; statistic "
                       "invariance plus per-attack gated C1 was not sufficient. Report in the "
                       "body; prereg 5.1 registers this as the live risk, so it cannot be "
                       "recast as a surprise."),
    ("HIGH", "LOW"): ("B is inverted on this cell.",
                      "The pair it certified is HIGH and the pair it rejected is LOW. The worst "
                      "case for the criterion, and reported as such."),
}


# ---------------------------------------------------------------------------------------------
def prereg_gate():
    """§8: the prereg must be committed, and clean. Re-checked here, not taken on trust."""
    def git(*a):
        return subprocess.run(["git", "-C", BASE, *a], capture_output=True, text=True).stdout.strip()
    commit = git("log", "-1", "--format=%H", "--", PREREG)
    dirty = git("status", "--porcelain", "--", PREREG)
    return {"prereg": PREREG, "commit": commit[:7] if commit else None,
            "committed": bool(commit), "uncommitted_modifications": bool(dirty)}


def mean(xs):
    return sum(xs) / len(xs)


def arm_stats(cell):
    """Recompute the arm's mean ASR and accuracy from per-seed, never trusting the stored mean."""
    per = {s["seed"]: s for s in cell["per_seed"]}
    seeds = sorted(per)
    asr = [per[s]["asr"] for s in seeds]
    acc = [per[s]["accuracy"] for s in seeds]
    out = {"seeds": seeds, "asr": asr, "accuracy": acc,
           "mean_asr": mean(asr), "mean_accuracy": mean(acc)}
    stored = cell.get("mean_asr")
    out["stored_mean_asr"] = stored
    out["mean_asr_matches_stored"] = stored is None or abs(stored - out["mean_asr"]) < 1e-12
    return out


def d2_alone_per_seed(d2):
    """d2 with no upstream: the `fedavg -> d2` cells, at this exact protocol."""
    path = os.path.join(BASE, "results", "metric_swap_baselines", "summary.json")
    if not os.path.exists(path):
        return None
    cells = json.load(open(path))["cells"]
    out = {}
    for a in ATTACKS:
        k = f"fedavg_then_{d2}|{a}"
        if k in cells:
            out[a] = {s["seed"]: (s["accuracy"], s["asr"]) for s in cells[k]["per_seed"]}
    return out or None


def near_identity(pair_cells, d2):
    """§6.4, mandatory. Per-seed comparison against d2 alone, per arm."""
    base = d2_alone_per_seed(d2)
    verdicts = {}
    for a in ATTACKS:
        if a not in pair_cells:
            continue
        if not base or a not in base:
            verdicts[a] = {"checked": False, "reason": f"no fedavg_then_{d2} baseline for {a}"}
            continue
        same, worst = 0, 0.0
        n = 0
        for s in pair_cells[a]["per_seed"]:
            if s["seed"] not in base[a]:
                continue
            n += 1
            b_acc, b_asr = base[a][s["seed"]]
            d = max(abs(s["asr"] - b_asr), abs(s["accuracy"] - b_acc))
            worst = max(worst, d)
            if d == 0.0:
                same += 1
        verdicts[a] = {"checked": True, "seeds_compared": n, "identical_seeds": same,
                       "max_abs_diff": worst,
                       "verdict": "NEAR-IDENTITY" if n and same == n else "genuine composition"}
    return verdicts


def sanity(pair_cells):
    """§6.3, ADVISORY ONLY. Never changes a score; demands a harness look."""
    flags = []
    for a, cell in pair_cells.items():
        if a not in ATTACKS:
            continue
        st = arm_stats(cell)
        if all(x == 0.0 for x in st["asr"]):
            flags.append(f"{a}: ASR identically 0.0 across all {len(st['asr'])} seeds")
        if all(x == 1.0 for x in st["asr"]):
            flags.append(f"{a}: ASR identically 1.0 across all {len(st['asr'])} seeds")
        ref = FEDAVG_ACC_REF.get(a)
        if ref is not None:
            spread = max(st["accuracy"]) - min(st["accuracy"])
            margin = max(SANITY_FLOOR_MARGIN, spread)
            if st["mean_accuracy"] - ref > margin:
                flags.append(f"{a}: mean accuracy {st['mean_accuracy']:.4f} exceeds the FedAvg "
                             f"reference {ref} by more than {margin:.4f}")
        if not st["mean_asr_matches_stored"]:
            flags.append(f"{a}: recomputed mean ASR {st['mean_asr']!r} != stored "
                         f"{st['stored_mean_asr']!r}")
    return flags


def score_pair(key, pair):
    """§6.1 + §6.2. The accuracy gate is applied to the arm that SUPPLIES the maximum."""
    d1, d2 = key.split("_then_")
    arms = {a: arm_stats(pair[a]) for a in ATTACKS if a in pair}
    if not arms:
        return None
    max_arm = max(arms, key=lambda a: arms[a]["mean_asr"])
    max_asr = arms[max_arm]["mean_asr"]
    gate_acc = arms[max_arm]["mean_accuracy"]
    degenerate = gate_acc < ACC_FLOOR
    outcome = "HIGH" if (degenerate or max_asr >= THRESH) else "LOW"
    meta = FROZEN.get(key, {})
    return {
        "pair": key, "d1": d1, "d2": d2, "n": meta.get("n"),
        "arms": arms, "complete": len(arms) == len(ATTACKS),
        "max_committed_arm": max_arm, "max_committed_asr": max_asr,
        "accuracy_on_max_arm": gate_acc,
        "degenerate_acc": degenerate,
        "outcome": outcome,
        "outcome_reason": ("DEGENERATE-ACC: accuracy %.4f < %.2f on the max-supplying arm, "
                           "scored HIGH" % (gate_acc, ACC_FLOOR)) if degenerate else
                          ("max-committed %.4f %s %.1f" % (max_asr,
                           "<" if max_asr < THRESH else ">=", THRESH)),
        "predicted": meta.get("predicted"),
        "correct": (outcome == meta.get("predicted")) if meta.get("predicted") else None,
        "cell": meta.get("cell"), "role": meta.get("role"),
        "confidence": meta.get("confidence"), "c1": meta.get("c1"), "c2": meta.get("c2"),
        "near_identity": near_identity({a: pair[a] for a in arms}, d2),
        "sanity_flags": sanity({a: pair[a] for a in arms}),
    }


def read_frozen_census():
    """Read the seven frozen cells from their own artifacts and cross-check §2.1's transcription."""
    got = {}
    caches = {}
    for (d1, d2), (transcribed, src) in CENSUS_FROZEN.items():
        path = os.path.join(BASE, src)
        if path not in caches:
            caches[path] = json.load(open(path)) if os.path.exists(path) else None
        blob = caches[path]
        key = f"{d1}_then_{d2}"
        value = None
        if blob is None:
            note = f"missing artifact {src}"
        elif "cells" in blob:                       # metric_swap layout: pair|attack
            per = [blob["cells"][f"{key}|{a}"]["mean_asr"]
                   for a in ATTACKS if f"{key}|{a}" in blob["cells"]]
            value, note = (max(per), "read") if per else (None, f"{key} absent from {src}")
        elif "pairs" in blob and key in blob["pairs"]:
            v = blob["pairs"][key]
            per = [v[a]["mean_asr"] for a in ATTACKS if a in v]
            value, note = (max(per), "read") if per else (None, f"no committed arms for {key}")
        else:
            note = f"{key} absent from {src}"
        agrees = value is not None and abs(value - transcribed) < 5e-5
        got[key] = {"d1": d1, "d2": d2, "max_committed_asr": value, "source": src,
                    "transcribed_in_prereg": transcribed, "agrees_with_prereg": agrees,
                    "note": note, "outcome": None if value is None else
                    ("LOW" if value < THRESH else "HIGH")}
    return got


# ---- C1 across the WHOLE census, so both strategies are priced on the same closed set -------
# The prereg fixes C1 only for its own four pairs. Pricing strategy B over the closed 9-cell
# census needs C1 for the other five, and that costs NO new runs: C1 is defined on STANDALONE
# suppression, and every one of the six defenses in the census already has a `fedavg -> X` cell
# at this exact protocol (fedavg as d1 emits no per-client transform, so fedavg->X IS X alone).
# Read from the artifacts, never transcribed.
STANDALONE_SOURCES = (
    ("results/all_compositions/summary.json", "pairs"),
    ("results/wave2_held_out/summary.json", "pairs"),
    ("results/metric_swap_baselines/summary.json", "cells"),
)


def standalone_baselines():
    """{defense: {attack: {mean_asr, mean_accuracy, source}}} for every `fedavg -> X` available."""
    out = {}
    for src, layout in STANDALONE_SOURCES:
        path = os.path.join(BASE, src)
        if not os.path.exists(path):
            continue
        blob = json.load(open(path))
        container = blob.get("pairs" if layout == "pairs" else "cells", {})
        for k, v in container.items():
            if layout == "cells":
                if "|" not in k:
                    continue
                pair_key, attack = k.split("|")
                if not pair_key.startswith("fedavg_then_") or attack not in ATTACKS:
                    continue
                name, arms = pair_key.replace("fedavg_then_", ""), {attack: v}
            else:
                if not k.startswith("fedavg_then_"):
                    continue
                name = k.replace("fedavg_then_", "")
                arms = {a: v[a] for a in ATTACKS if a in v}
            for a, cell in arms.items():
                acc = cell.get("mean_accuracy", cell.get("mean_acc"))
                if acc is None and cell.get("per_seed"):
                    accs = [s["accuracy"] for s in cell["per_seed"] if "accuracy" in s]
                    acc = mean(accs) if accs else None
                # First source wins, so a defense measured in two artifacts is not silently
                # overwritten by whichever file happens to be read last.
                out.setdefault(name, {}).setdefault(
                    a, {"mean_asr": cell["mean_asr"], "mean_accuracy": acc, "source": src})
    return out


def suppresses(base, defense, attack):
    """Does `defense` alone suppress `attack`? Both readings returned; neither hidden.

    GATED is the paper's own convention and the one the prereg's C1 notes use ("gated"): a cell
    below the 0.35 clean-accuracy floor is uninterpretable, not suppressed, so a defense that
    only drives ASR down by destroying the model does not count. UNGATED is the bare threshold.
    Emitting both means the choice of reading is visible rather than buried in a helper.
    """
    cell = base.get(defense, {}).get(attack)
    if cell is None:
        return {"available": False, "gated": None, "ungated": None,
                "note": f"no fedavg_then_{defense} baseline for {attack}"}
    asr, acc = cell["mean_asr"], cell["mean_accuracy"]
    ungated = asr < THRESH
    degenerate = acc is not None and acc < ACC_FLOOR
    return {"available": True, "mean_asr": asr, "mean_accuracy": acc,
            "degenerate_acc": degenerate, "ungated": ungated,
            "gated": bool(ungated and not degenerate), "source": cell["source"]}


def c1_over_census(base):
    """C1 for each of the 9 census cells: for EVERY committed attack, some constituent suppresses.

    C2 is True for all 9 by the census's own construction (prereg §2): d1 is drawn from the
    strictly-positive per-client rescalings, so none is a degenerate aggregator, and d2 from the
    exactly scale-invariant statistics, so Prop 1(a) settles C2 algebraically -- this is the one
    region of the menu where the C2 column is NOT a per-pair call. Asserted, not assumed.
    """
    # Imported rather than transcribed, so the degenerate-aggregator list has one owner. That
    # module chdirs and loads artifacts at import time, so the cwd is restored afterwards.
    cwd = os.getcwd()
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    try:
        from experiments.analyze_condition_ablation import DEGENERATE_D1
    finally:
        os.chdir(cwd)
    assert not (set(CENSUS_D1) & DEGENERATE_D1), \
        f"census d1 set overlaps the degenerate aggregators: {set(CENSUS_D1) & DEGENERATE_D1}"
    out = {}
    for d1 in CENSUS_D1:
        for d2 in CENSUS_D2:
            per_attack = {}
            for a in ATTACKS:
                s1, s2 = suppresses(base, d1, a), suppresses(base, d2, a)
                per_attack[a] = {
                    d1: s1, d2: s2,
                    "any_gated": bool(s1["gated"] or s2["gated"]),
                    "any_ungated": bool(s1["ungated"] or s2["ungated"]),
                    "both_available": s1["available"] and s2["available"]}
            out[f"{d1}_then_{d2}"] = {
                "d1": d1, "d2": d2,
                "C1_gated": all(v["any_gated"] for v in per_attack.values()),
                "C1_ungated": all(v["any_ungated"] for v in per_attack.values()),
                "all_baselines_available": all(v["both_available"] for v in per_attack.values()),
                "C2": True, "C2_basis": "algebraic (Prop 1(a)): d2 exactly invariant, d1 not "
                                        "a degenerate aggregator",
                "per_attack": per_attack}
    return out


def main():
    gate = prereg_gate()
    print("WAVE 3 -- SCORED AGAINST ITS PRE-REGISTRATION")
    print(f"  prereg {gate['prereg']}")
    print(f"    committed: {gate['committed']} at {gate['commit']}, "
          f"uncommitted modifications: {gate['uncommitted_modifications']}")
    if not gate["committed"] or gate["uncommitted_modifications"]:
        print("  *** The prereg is not cleanly committed. Per prereg section 8 the wave is")
        print("  *** REPORTED AS NON-PROSPECTIVE. Scoring continues; the label does not.")

    if not os.path.exists(WAVE):
        print(f"\nMISSING: {WAVE}\n  Nothing to score yet. Re-run once the wave writes results.")
        return 1
    wave = json.load(open(WAVE))
    pairs = wave.get("pairs", {})
    print(f"\n  {len(pairs)} of {len(FROZEN)} registered pairs present in the artifact.")
    unexpected = sorted(set(pairs) - set(FROZEN))
    if unexpected:
        print(f"  *** UNREGISTERED PAIRS PRESENT: {unexpected}. Prereg section 8 forbids adding")
        print("  *** pairs after the data is seen. Reported, not scored away.")

    scored = {}
    print("\n=== PER-PAIR (prereg sections 6.1, 6.2) ===")
    hdr = f"  {'#':>1} {'pair':32s} {'pred':>5s} {'max-ASR':>8s} {'acc':>6s} {'outcome':>8s} {'':>4s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(FROZEN, key=lambda k: FROZEN[k]["n"]):
        if key not in pairs:
            print(f"  {FROZEN[key]['n']} {key:32s} {FROZEN[key]['predicted']:>5s} "
                  f"{'--':>8s} {'--':>6s} {'PENDING':>8s}")
            continue
        r = score_pair(key, pairs[key])
        scored[key] = r
        mark = "ok" if r["correct"] else "MISS"
        print(f"  {r['n']} {key:32s} {r['predicted']:>5s} {r['max_committed_asr']:8.4f} "
              f"{r['accuracy_on_max_arm']:6.3f} {r['outcome']:>8s} {mark:>4s}")
        if not r["complete"]:
            print(f"       INCOMPLETE: only {sorted(r['arms'])} present, so this max is provisional")
        print(f"       {r['cell']}; {r['role']}")
        print(f"       max supplied by {r['max_committed_arm']}; {r['outcome_reason']}")
        if r["degenerate_acc"]:
            print("       *** DEGENERATE-ACC (prereg 6.2): scored HIGH on the accuracy gate ***")
        for a, ni in r["near_identity"].items():
            if not ni.get("checked"):
                print(f"       near-identity {a}: NOT CHECKED -- {ni['reason']}")
            elif ni["verdict"] == "NEAR-IDENTITY":
                print(f"       *** NEAR-IDENTITY on {a}: identical to {r['d2']} alone in "
                      f"{ni['identical_seeds']}/{ni['seeds_compared']} seeds. Per prereg 6.4 this "
                      f"arm is NOT evidence about composition. ***")
            else:
                print(f"       near-identity {a}: genuine ({ni['identical_seeds']}/"
                      f"{ni['seeds_compared']} identical, max diff {ni['max_abs_diff']:.3e})")
        for f in r["sanity_flags"]:
            print(f"       SANITY FLAG (advisory, changes no score): {f}")

    # ---- per-seed accuracy floor, reported even where the mean clears it --------------------
    print("\n=== SEEDS BELOW THE 0.35 ACCURACY FLOOR (reported; the gate is on the mean) ===")
    any_below = False
    for key, r in scored.items():
        for a, st in r["arms"].items():
            for s, acc in zip(st["seeds"], st["accuracy"]):
                if acc < ACC_FLOOR:
                    any_below = True
                    print(f"  {key} / {a} / seed {s}: accuracy {acc:.4f} "
                          f"(arm mean {st['mean_accuracy']:.4f})")
    if not any_below:
        print("  none")

    # ---- the tally, and the discriminating comparison ---------------------------------------
    done = [r for r in scored.values() if r["complete"]]
    correct = [r for r in done if r["correct"]]
    print(f"\n=== TALLY over the {len(done)} complete pairs: {len(correct)}/{len(done)} correct ===")

    strategies = {}
    print("\n=== WHAT THE TWO STRATEGIES BOUGHT ON THIS WAVE (prereg section 5) ===")
    lows = [k for k, r in scored.items() if r["complete"] and r["outcome"] == "LOW"]
    # A rate over a partly-scored selection is not a rate, it is a preview of one. Both figures
    # are withheld until every pair the strategy selects is complete, and recall additionally
    # needs all four pairs, since an unscored pair could still be the LOW that recall divides by.
    all_complete = all(k in scored and scored[k]["complete"] for k in FROZEN)
    for name, sel in SELECTED_BY.items():
        have = [k for k in sel if k in scored and scored[k]["complete"]]
        tp = [k for k in have if scored[k]["outcome"] == "LOW"]
        sel_complete = len(have) == len(sel)
        prec = (len(tp) / len(have)) if sel_complete and have else None
        rec = (len(tp) / len(lows)) if all_complete and lows else None
        strategies[name] = {"selected": sel, "scored": have, "true_positives": tp,
                            "precision": prec, "recall": rec, "runs": 10 * len(sel),
                            "selection_complete": sel_complete, "wave_complete": all_complete}
        ps = f"{100*prec:.0f}%" if prec is not None else \
             f"WITHHELD ({len(have)}/{len(sel)} selected pairs scored)"
        rs = f"{100*rec:.0f}%" if rec is not None else \
             ("n/a (no LOW in the wave)" if all_complete else "WITHHELD (wave incomplete)")
        print(f"  {name:26s} selects {len(sel)}, scored {len(have)}, "
              f"{len(tp)} LOW -> precision {ps}, recall {rs}")

    # ---- §7 resolution, only once both discriminating pairs are in --------------------------
    p1, p2 = "reputation_then_cos_krum", "reputation_then_cos_reputation"
    resolution = None
    print("\n=== PRE-REGISTERED RESOLUTION (prereg section 7) ===")
    if p1 in scored and p2 in scored and scored[p1]["complete"] and scored[p2]["complete"]:
        k = (scored[p1]["outcome"], scored[p2]["outcome"])
        reading, writeup = RESOLUTION[k]
        resolution = {"pair1_outcome": k[0], "pair2_outcome": k[1],
                      "reading": reading, "what_goes_in_the_paper": writeup}
        print(f"  pair 1 = {k[0]}, pair 2 = {k[1]}  ->  {reading}")
        print(f"  {writeup}")
    else:
        print("  PENDING: needs both pair 1 and pair 2 complete. No cell of section 7's table is")
        print("  selected until then, and none is previewed here.")

    # ---- the 9-cell census -----------------------------------------------------------------
    print("\n=== THE 9-PAIR CENSUS (prereg section 2): strictly-positive d1 x exactly-invariant d2 ===")
    census = read_frozen_census()
    disagreements = [v for v in census.values() if not v["agrees_with_prereg"]]
    for key, r in scored.items():
        if (r["d1"], r["d2"]) in [(a, b) for a in CENSUS_D1 for b in CENSUS_D2]:
            census[key] = {"d1": r["d1"], "d2": r["d2"],
                           "max_committed_asr": r["max_committed_asr"] if r["complete"] else None,
                           "source": "results/wave3_emergent/summary.json",
                           "transcribed_in_prereg": None, "agrees_with_prereg": True,
                           "note": "this wave" if r["complete"] else "this wave, incomplete",
                           "outcome": r["outcome"] if r["complete"] else None}
    print(f"  {'d1 / d2':14s}" + "".join(f"{d2:>20s}" for d2 in CENSUS_D2))
    for d1 in CENSUS_D1:
        row = f"  {d1:14s}"
        for d2 in CENSUS_D2:
            c = census.get(f"{d1}_then_{d2}")
            row += f"{'--':>20s}" if not c or c["max_committed_asr"] is None else \
                   f"{c['max_committed_asr']:14.4f} {c['outcome']:>5s}"
        print(row)
    filled = [c for c in census.values() if c["max_committed_asr"] is not None]
    census_lows = [c for c in filled if c["outcome"] == "LOW"]
    print(f"\n  {len(filled)} of 9 cells measured; {len(census_lows)} LOW.")
    if disagreements:
        print("  *** CROSS-CHECK FAILED against the prereg's section 2.1 transcription:")
        for d in disagreements:
            print(f"      {d['d1']}->{d['d2']}: artifact {d['max_committed_asr']}, prereg "
                  f"{d['transcribed_in_prereg']} ({d['note']})")
    else:
        print("  cross-check: every frozen cell read from its artifact matches section 2.1.")
    census_strategies = None
    if len(filled) == 9:
        print(f"  THE CENSUS IS CLOSED. Strategy A (statistic preservation) selects all 9 for "
              f"{len(census_lows)} true positive(s):")
        print(f"    precision {len(census_lows)}/9 = {100*len(census_lows)/9:.0f}% in the very "
              f"class where invariance is EXACTLY true.")

        # ---- both strategies, same closed census, C1 recomputed from standalone baselines ----
        base = standalone_baselines()
        c1 = c1_over_census(base)
        print("\n=== BOTH STRATEGIES OVER THE CLOSED CENSUS ===")
        print("  C1 recomputed from the `fedavg -> X` standalone cells (no new runs); C2 is")
        print("  algebraic for all 9 by the census's construction, so this is the one region")
        print("  where the C2 column rests on Prop 1(a) rather than on a per-pair call.")
        missing = [k for k, v in c1.items() if not v["all_baselines_available"]]
        if missing:
            print(f"  *** standalone baselines missing for {missing}: C1 not evaluable there ***")
        print(f"\n  {'pair':32s} {'C1':>6s} {'C1*':>5s} {'outcome':>8s}")
        for key in sorted(c1, key=lambda k: (CENSUS_D1.index(c1[k]['d1']),
                                             CENSUS_D2.index(c1[k]['d2']))):
            c, cell = c1[key], census.get(key)
            oc = cell["outcome"] if cell and cell["outcome"] else "--"
            print(f"  {key:32s} {str(c['C1_gated']):>6s} {str(c['C1_ungated']):>5s} {oc:>8s}")
        print("  (C1 = accuracy-gated, the reading the prereg uses; C1* = bare 0.5 threshold)")

        census_strategies = {}
        for name, sel_keys in (
                ("A_statistic_preservation", list(c1)),
                ("B_criterion", [k for k, v in c1.items() if v["C1_gated"] and v["C2"]])):
            tp = [k for k in sel_keys
                  if census.get(k, {}).get("outcome") == "LOW"]
            census_strategies[name] = {
                "selected": sorted(sel_keys), "n_selected": len(sel_keys),
                "true_positives": sorted(tp),
                "precision": (len(tp) / len(sel_keys)) if sel_keys else None,
                "recall": (len(tp) / len(census_lows)) if census_lows else None,
                "runs": 10 * len(sel_keys)}
        print()
        for name, s in census_strategies.items():
            p = f"{100*s['precision']:.0f}%" if s["precision"] is not None else "n/a"
            r = f"{100*s['recall']:.0f}%" if s["recall"] is not None else "n/a"
            print(f"  {name:26s} selects {s['n_selected']}/9, {len(s['true_positives'])} LOW "
                  f"-> precision {p}, recall {r}, {s['runs']} runs")
        a, b = census_strategies["A_statistic_preservation"], census_strategies["B_criterion"]
        print(f"\n  So on a CLOSED census in the exactly-invariant class, statistic preservation")
        print(f"  costs {a['runs']} runs for precision {100*a['precision']:.0f}%, and the criterion")
        print(f"  costs {b['runs']} for {100*b['precision']:.0f}% at recall "
              f"{100*b['recall']:.0f}%. Both readings of C1 are emitted; if they disagree on any")
        print(f"  cell the disagreement is in the table above, not hidden in this summary.")
        disagree = [k for k, v in c1.items() if v["C1_gated"] != v["C1_ungated"]]
        print(f"  C1 gated vs ungated disagree on: {disagree if disagree else 'no cell'}")
        out_c1 = c1
    else:
        out_c1 = None

    out = {"description": "Wave 3 scored against experiments/pre_registration_wave3_emergent.md",
           "pre_registration": gate, "thresholds": {"low_asr": THRESH, "accuracy_floor": ACC_FLOOR,
           "fedavg_accuracy_reference": FEDAVG_ACC_REF},
           "pairs": scored, "tally": {"complete": len(done), "correct": len(correct)},
           "strategies": strategies, "resolution": resolution,
           "census": census, "census_measured": len(filled), "census_lows": len(census_lows),
           "census_c1": out_c1, "census_strategies": census_strategies,
           "unregistered_pairs_present": unexpected}
    json.dump(out, open(SCORED, "w"), indent=2)
    print(f"\nSaved to {os.path.relpath(SCORED, BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
