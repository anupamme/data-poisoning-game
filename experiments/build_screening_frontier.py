"""
The screening frontier: precision, recall and evaluation cost for every criterion variant.

WHAT THIS IS FOR
A reviewer's objection to the criterion is that it has 40% recall and excludes the paper's own best
composition, so "why would anyone use it?". That objection treats C1&C2&C3 as *the* criterion. It is
not: it is one point on a frontier the ablation already measured, and this script draws the whole
frontier so the operating point can be chosen rather than defended.

THE RESULT THAT MAKES THIS WORTH PRINTING
`C2` alone selects 14 of 42 pairs at 36% precision and **100% recall**: it finds every low-ASR pair in
the menu, including FoolsGold->RFA, which C1&C2 misses. And the reason is exactly the one
`prop:emergent` predicts -- FG->RFA has C1=False, C2=True, C3=True. **The blind spot is attributable
to one named condition, C1, which is the inherited-suppression gate the proposition proves cannot
certify emergent synergy.** Dropping C1 recovers the pair and costs precision. That is a frontier, not
a failure.

THIS SCRIPT ADDS NO RUNS AND NO NUMBERS OF ITS OWN
Everything is read from two frozen artifacts and joined:
  results/condition_ablation/summary.json  (per-pair C1/C2/C3 and low_asr, 42 pairs)
  results/screening_cost.json              (the run accounting)
Both are produced by their own generators (`analyze_condition_ablation.py`,
`analyze_screening_cost.py`). If either is missing this script says so and exits rather than
recomputing a number that has a generator elsewhere.

COST ACCOUNTING, AND THE ONE ASYMMETRY IT HAS TO DISCLOSE
Composition cells cost `2 attacks x 5 seeds = 10` runs per admitted pair. **C1 additionally requires
the standalone measurements** (60 runs) because it is a condition *on measured standalone
suppression*; C2 and C3 are read off the two defenses' definitions. So a C1-bearing variant carries a
fixed +60 that a C2-only variant does not, and the table states that in its own column rather than
burying it in a total.

Run: python3 experiments/build_screening_frontier.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABLATION = os.path.join(BASE, "results", "condition_ablation", "summary.json")
COST = os.path.join(BASE, "results", "screening_cost.json")

RUNS_PER_PAIR = 10          # 2 committed attacks x 5 seeds, the menu's own cell size
STANDALONE_RUNS = 60        # what C1 needs before it can be evaluated at all

# The frontier, as (label, predicate over a row's C1/C2/C3). `None` is the unscreened baseline.
VARIANTS = [
    ("no screen (all pairs)", None),
    ("C3 alone",              lambda r: r["C3"]),
    ("C2 alone",              lambda r: r["C2"]),
    # The same screen restricted to the region where C2 is settled by Proposition 1 rather than
    # by an in-sample per-pair call. This is what "statistic preservation" means if the phrase is
    # held to the algebra the paper proves; see `c2_provenance` in the ablation.
    ("C2 algebraic only",     lambda r: r["C2"] and r["c2_provenance"] == "algebraic_invariant"),
    ("C2 and C3",             lambda r: r["C2"] and r["C3"]),
    ("C1 alone",              lambda r: r["C1"]),
    ("C1 and C3",             lambda r: r["C1"] and r["C3"]),
    ("C1 and C2",             lambda r: r["C1"] and r["C2"]),
    ("C1 and C2 and C3",      lambda r: r["C1"] and r["C2"] and r["C3"]),
]

# The emergent witness. Named explicitly because the whole point of the frontier is which variants
# can reach it, and `prop:emergent` predicts that no C1-bearing variant can.
EMERGENT = "foolsgold_then_rfa"


def load(path, what):
    if not os.path.exists(path):
        print(f"MISSING: {path}\n  Run the generator that produces it; this script recomputes nothing.")
        return None
    return json.load(open(path))


def main():
    abl = load(ABLATION, "ablation")
    if abl is None:
        return 1
    rows = abl["rows"]
    cost = load(COST, "cost")           # optional: only used to cross-check the baseline

    total = len(rows)
    lows = [r for r in rows if r["low_asr"]]
    n_low = len(lows)
    emergent = next((r for r in rows if r["pair"] == EMERGENT), None)

    print("SCREENING FRONTIER")
    print(f"  {total} ordered pairs, {n_low} of them low-ASR (base rate "
          f"{100.0 * n_low / total:.0f}%), {RUNS_PER_PAIR} runs per admitted pair.\n")

    if emergent is not None:
        print(f"  emergent witness {EMERGENT}: C1={emergent['C1']} C2={emergent['C2']} "
              f"C3={emergent['C3']}, max-committed ASR {emergent['max_committed_asr']:.3f}, "
              f"low_asr={emergent['low_asr']}")
        print("  prop:emergent predicts NO C1-bearing variant reaches it. Checked per row below.\n")

    hdr = (f"  {'variant':22s} {'sel':>4s} {'prec':>6s} {'recall':>7s} {'runs':>6s} "
           f"{'+C1 standalone':>15s}  {'reaches FG->RFA':>16s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    table = []
    for label, pred in VARIANTS:
        sel = rows if pred is None else [r for r in rows if pred(r)]
        tp = sum(1 for r in sel if r["low_asr"])
        prec = (tp / len(sel)) if sel else float("nan")
        rec = tp / n_low if n_low else float("nan")
        uses_c1 = "C1" in label
        runs = RUNS_PER_PAIR * len(sel)
        reaches = emergent is not None and (pred is None or pred(emergent))
        table.append({"label": label, "selected": len(sel), "tp": tp, "precision": prec,
                      "recall": rec, "runs": runs, "uses_c1": uses_c1, "reaches_emergent": reaches})
        print(f"  {label:22s} {len(sel):4d} {100*prec:5.0f}% {100*rec:6.0f}% {runs:6d} "
              f"{('yes, +' + str(STANDALONE_RUNS)) if uses_c1 else 'no':>15s}  "
              f"{('YES' if reaches else 'no'):>16s}")

    # The two ends of the frontier, stated rather than left to the reader.
    best_prec = max(table[1:], key=lambda t: (t["precision"], -t["selected"]))
    best_rec = max(table[1:], key=lambda t: (t["recall"], -t["selected"]))
    print(f"\n  HIGH-PRECISION END: {best_prec['label']} -- {best_prec['tp']}/{best_prec['selected']} "
          f"at {100*best_prec['precision']:.0f}% precision, {100*best_prec['recall']:.0f}% recall,")
    print(f"    {best_prec['runs']} composition runs"
          + (f" plus {STANDALONE_RUNS} standalone" if best_prec["uses_c1"] else "")
          + f", against {RUNS_PER_PAIR * total} unscreened.")
    print(f"  HIGH-RECALL END:    {best_rec['label']} -- {best_rec['tp']}/{best_rec['selected']} "
          f"at {100*best_rec['precision']:.0f}% precision, {100*best_rec['recall']:.0f}% recall,")
    print(f"    {best_rec['runs']} composition runs"
          + (f" plus {STANDALONE_RUNS} standalone" if best_rec["uses_c1"] else "") + ".")

    if best_rec["reaches_emergent"] and not best_prec["reaches_emergent"]:
        print("\n  THE BLIND SPOT IS ATTRIBUTABLE TO ONE CONDITION, AND THE THEORY NAMED IT FIRST:")
        print(f"    the high-precision variant misses {EMERGENT}; the high-recall variant reaches it.")
        print("    They differ by C1, the inherited-suppression gate, which is exactly what")
        print("    prop:emergent proves cannot certify an emergent composition. So the exclusion is")
        print("    a predicted property of one condition, not an unexplained miss -- and it is")
        print("    tradeable, at a measured cost in precision.")

    # ---- the scope limit on calling the C2 column "statistic preservation" -------------
    # Read, not recomputed: `c2_provenance` is emitted per row by analyze_condition_ablation.py.
    prov = abl.get("c2_provenance")
    if prov is not None:
        alg = prov["by_provenance"].get("algebraic_invariant", 0)
        call = prov["by_provenance"].get("per_pair_call", 0)
        rest = prov["c2_restricted_to_algebraic"]
        print("\n  WHAT THE C2 COLUMN IS MADE OF, WHICH THE HIGH-RECALL CLAIM DEPENDS ON:")
        print(f"    of the {prov['c2_admitted']} pairs C2 admits, {alg} rest on algebraic")
        print(f"    invariance (Prop 1(a)) and {call} on a per-pair call for the")
        print(f"    conditional classes ({', '.join(prov['conditional_class_d2'])}), "
              f"{prov['per_pair_call_splits'].get('dev', 0)} of them in-sample dev pairs.")
        print(f"    {prov['n_low_asr_algebraic']} of the {prov['n_low_asr']} low-ASR pairs rest on "
              f"algebra, so restricting the screen to the")
        print(f"    settled region selects {rest['selected']} at {100*rest['precision']:.0f}% "
              f"precision and {100*rest['recall']:.0f}% recall: it discriminates nothing.")
        print("    So C2-alone's 100% recall is carried by the in-sample call, and the column is not")
        print("    a pure statistic-preservation screen for those three d2 classes. This is a scope")
        print("    disclosure, not a correction: no number above changes.")

    # Consistency check against the frozen cost artifact, which computes the C1&C2 point its own way.
    if cost is not None:
        for k in ("runs_unscreened", "unscreened_runs", "runs_screened", "screened_runs"):
            if k in cost:
                print(f"\n  cross-check, results/screening_cost.json[{k}] = {cost[k]}")

    print("\n--- LaTeX rows (variant, selected, precision, recall, composition runs) ---")
    for t in table:
        star = "$^\\dagger$" if t["uses_c1"] else ""
        reach = "\\checkmark" if t["reaches_emergent"] else "---"
        print(f"{t['label'].replace('and', '$\\wedge$')}{star} & {t['selected']} & "
              f"{100*t['precision']:.0f}\\% & {100*t['recall']:.0f}\\% & {t['runs']} & {reach} \\\\")
    print("\\multicolumn{6}{l}{\\footnotesize $\\dagger$ additionally requires "
          f"{STANDALONE_RUNS} standalone runs before the condition can be evaluated.}} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
