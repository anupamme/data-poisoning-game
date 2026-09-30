"""Check every number Arm D's appendix prose states against the FROZEN scorer, verbatim in the paper.

WHY THIS EXISTS. Arm D's coverage changes as slices land, so its subsection has to be re-substituted
more than once, against a deadline. Twelve of its figures are counts and fractions that no build, gate
or hash can see, and three of them are stated as ENGLISH WORDS ("Two of the 11 complete pairs", "the
two observed-LOW pairs", "the remaining twelve readings") -- the one form of magnitude in this paper
with no emitter behind it at all. Hand-substituting that set twice is where a transcription error
lands, and a wrong denominator here would misreport the arm's headline.

WHAT IT DOES NOT DO. It computes no estimand. Every fraction, base rate, precision and recall comes
from the frozen runner's own score() over the merged rows, called here exactly as --report calls it.
What this file adds is the display rounding, the number words, and the assertion that the resulting
string appears in the paper -- either document, since Round 83 moved the section into the supplement.
It writes nothing, and it never calls save() -- which would delete
the merge's provenance fields.

WHAT IT READS. results/cifar100_composition_suite/summary.json and the probe artifact, both through the
frozen runner's own loaders, plus paper/main.tex, paper/supplementary.tex and -- where the response letter PARAPHRASES one of
these counts instead of quoting it, so that audit_letter_cites.py cannot verify it --
paper/response_reviewer_21st_channels.md, matched whitespace-collapsed because one such paraphrase wraps
across a markdown line break.

    PYTHONPATH=. python3 -m experiments.check_cifar100_menu_prose

Exit 0 iff every derived string is present. A failure prints the frozen value and the sentence it
belongs in, so the substitution is mechanical. Run it after every merge, beside
`emit_perseed_cifar100_menu --check`, which covers the per-seed table this one does not.

ONE THING IT CANNOT SEE. It checks the numbers, not the surrounding claim. Two prose facts are coupled
to the same artifact and must still be re-read by eye: the subsection TITLE's "it predicts LOW for no
pair", and the "equal by construction" sentence. Both are asserted below as PREDICATES on score()'s
output rather than as strings, so this file fails if they go false -- but the wording is on the reader.
"""

import math
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from experiments.run_cifar100_composition_suite import (  # noqa: E402
    ACC_FLOOR, ATTACKS, COMMITTED_ATTACKS, PAIRS_WAVE1, SEEDS, SUPPRESS_ASR,
    SUPPRESSION_THRESHOLD,
    cifar100_standalone_baselines, load, load_probe, probe_verdict, score)

# The paper is two xr-linked documents, and Round 83 relocated Arm D's section from the main paper's
# appendix into the supplement. PAPER therefore reads BOTH and a string present in either satisfies its
# check: absent from both is still a hard failure, so the check is scoped to where the prose lives, not
# softened. Nothing else about the check changes -- the values still come from the frozen scorer.
DOCS = (os.path.join(BASE, "paper", "main.tex"),
        os.path.join(BASE, "paper", "supplementary.tex"))


def paper_text():
    """Both documents, joined. A claim may live in either; it may not live in neither."""
    return "\n".join(open(p).read() for p in DOCS)

LETTER = os.path.join(BASE, "paper", "response_reviewer_21st_channels.md")

WORD = {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
        8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
        14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen"}

# "twice" has no emitter either, and it is the one number form that does not degrade gracefully: at a
# count of 3 the sentence needs "three times", not a substituted numeral.
TIMES = {0: "never", 1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times",
         6: "six times", 7: "seven times", 8: "eight times", 9: "nine times", 10: "ten times"}


def baseline_table():
    """The screen's standalone readings: the artifact's five aggregators plus the probe's two."""
    pure, _ = cifar100_standalone_baselines()
    for r in probe_verdict(load_probe())["cells"]:
        pure.setdefault(r["aggregator"], {})[r["attack"]] = r["mean_asr"]
    return pure


def tt(name):
    r"""A defense name as main.tex writes it: \texttt{} with underscores escaped."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def near(name, value):
    r"""A regex for "\texttt{pair} <a few words> $value$".

    The prose names the quantity on the FIRST item of a list and elides it on the rest
    ("\texttt{a} at minimum mean accuracy $0.3$ and \texttt{b} at $0.4$"), so a plain substring is
    right for one form and wrong for the other. The gap is bounded so this cannot match across a
    sentence boundary and silently pass on an unrelated number.
    """
    return re.compile(re.escape(tt(name)) + r"[^.$]{0,45}" + re.escape(f"${value}$"))


def checks(s, pure):
    """[(item, label, required substring or compiled regex)], all from the frozen score()."""
    w1 = sum(1 for r in s["rows"] if r["wave"] == 1)
    n_runs = len(ATTACKS) * len(SEEDS)
    out = []

    out.append((1, "coverage, wave-1 complete vs not",
                f"Of wave~1's ${len(PAIRS_WAVE1)}$ pairs, ${w1}$ completed all ${n_runs}$ of their "
                f"runs and ${len(PAIRS_WAVE1) - w1}$ did not."))

    nb, nc = s["n_below_floor"], s["n_complete"]
    out.append((2, "below-floor count, WORD-stated, and the complete denominator",
                f"{WORD[nb]} of the ${nc}$ complete pairs fall below the frozen ${ACC_FLOOR}$ "
                "clean-accuracy floor"))
    for r in s["below_floor_pairs"]:
        out.append((3, f"below-floor pair {r['pair']} and its accuracy",
                    near(r["pair"], f"{r['min_mean_accuracy']:.4f}")))

    scored = [r for r in s["rows"] if r["clears_floor"] and r["predicted"] in ("LOW", "HIGH")]
    if scored:
        acc = [r["min_mean_accuracy_over_attacks"] for r in scored]
        out.append((4, "the scored pairs' accuracy band",
                    f"lies between ${min(acc):.4f}$ and ${max(acc):.4f}$"))
        # A WORD-STATED MAGNITUDE with no emitter: the prose rounds the largest headroom UP to the next
        # hundredth and says no cell clears the floor by as much as that. It is coverage-dependent
        # through max(acc), so one new scored pair with more headroom falsifies it silently.
        head = math.ceil((max(acc) - ACC_FLOOR) * 100) / 100
        out.append((4, f"the floor-headroom ceiling (largest is {max(acc) - ACC_FLOOR:.4f})",
                    f"no cell clears the floor by as much as ${head:.2f}$"))

    agree = int(s["agreement"].split("/")[0])
    nhigh, ns = s["n_observed_high"], s["n_scored"]
    out.append((5, "agreement, its fraction, the constant-HIGH baseline and both exclusion counts",
                f"on ${agree}$ of the ${ns}$ scored pairs, ${s['agreement_fraction'] * 100:.1f}\\%$, "
                f"against a constant-HIGH predictor's ${nhigh}$ of ${ns}$ on the same pairs, "
                f"${s['base_rate_high'] * 100:.1f}\\%$, which is also the base rate, with ${nb}$ of "
                f"the ${nc}$ complete pairs excluded below the floor and ${s['n_unpredictable']}$ "
                "excluded for a missing baseline."))
    out.append((8, "the screen returns HIGH on every scored pair",
                f"the screen returns HIGH on all ${ns}$"))

    lows = [r for r in scored if r["observed"] == "LOW"]
    out.append((9, "the observed-LOW count, WORD-stated",
                f"the {WORD[len(lows)].lower()} observed-LOW pairs"))
    for r in sorted(lows, key=lambda r: r["max_committed_asr"]):
        out.append((9, f"missed LOW pair {r['pair']} and its max committed ASR",
                    near(r["pair"], f"{r['max_committed_asr']:.4f}")))

    # The standalone baselines: fixed by the published artifact plus gate 0's probe, so these do NOT
    # move as later slices land. Checked anyway, because "does not move" is worth measuring once --
    # and because reading them over the FIVE artifact aggregators instead of all SEVEN is the defect
    # this file was written to catch. It reported the lowest standalone scaling ASR as fedavg's 0.658
    # when reputation's is 0.005, which inverted the paragraph's mechanism: C1 does have an input on
    # this menu, on committed scaling, and the pair label is HIGH because the PIXEL backdoor has no
    # standalone suppressor and a pair needs C1 met for every committed attack.
    reads = sorted((v[a], d, a) for d, v in pure.items() for a in COMMITTED_ATTACKS
                   if v.get(a) is not None)
    if len(reads) > 2:
        below = [r for r in reads if r[0] < SUPPRESSION_THRESHOLD]
        out.append((10, f"how many of the {len(reads)} readings clear C1's threshold",
                    f"give {WORD[len(reads)].lower()} standalone readings, and \\textbf{{exactly "
                    f"{WORD[len(below)].lower()} falls below the screen's own C1 threshold of "
                    f"${SUPPRESSION_THRESHOLD}$}}"))
        for val, d, a in below:
            out.append((10, f"the one standalone suppressor: {d} / {a}",
                        f"{tt(d)} against {a.replace('committed_', 'committed ')}, at ${val:.3f}$"))
        rest = [r for r in reads if r not in below]
        out.append((10, "the non-suppressing readings' range, count WORD-stated",
                    f"The other {WORD[len(rest)].lower()} run from {tt(min(rest)[1])}'s "
                    f"${min(rest)[0]:.3f}$ to {tt(max(rest)[1])}'s ${max(rest)[0]:.3f}$"))
        pix = [r for r in reads if r[2] == "committed_pixel"]
        out.append((10, "the pixel backdoor's own range, the attack with no suppressor",
                    f"its {WORD[len(pix)].lower()} readings run from {tt(min(pix)[1])}'s "
                    f"${min(pix)[0]:.3f}$ to ${max(pix)[0]:.3f}$"))
        # The SAME reading is stated a second time in the lead paragraph, in different words. A
        # substitution that fixes one site and not its twin is the ordinary way these drift apart.
        for val, d, a in below:
            out.append((10, f"the lead paragraph's restatement of the {d} suppressor",
                        f"{tt(d)} does suppress {a.replace('committed_', 'committed ')} alone, at "
                        f"ASR ${val:.3f}$"))

    # The count of per-attack LOW predictions, stated TWICE in the lead paragraph and once as an
    # adverb ("twice"). This is the number that made the paragraph's corrected mechanism reportable
    # rather than a bare concession, and it is coverage-dependent: a newly completed pair containing
    # reputation can raise it, at which point both phrasings go false together.
    n_low_att = sum(1 for r in scored
                    for v in r["screen"]["per_attack"].values() if v["prediction"] == "LOW")
    if scored:
        out.append((16, f"per-attack LOW predictions, as a pair count ({n_low_att})",
                    f"on {WORD[n_low_att].lower()} scored pairs C1 and C2 both hold for that attack"))
        out.append((16, f"the same count as an adverb ({TIMES.get(n_low_att)})",
                    f"reaches a LOW per-attack prediction {TIMES.get(n_low_att, '??')}"))

    # A self-counting claim about ANOTHER table, which no build or gate can see: the lead paragraph
    # says four of tab:composability's five C1 rows read "C1 (pixel)". Counted from the table's own
    # rows rather than trusted, because a self-count of a neighbouring float is exactly the kind of
    # sentence that goes false without any edit to it.
    n_c1, n_pix = composability_c1_rows()
    if n_c1:
        out.append((15, f"the lead paragraph's self-count of Table tab:composability's C1 rows "
                        f"({n_pix} of {n_c1} read 'C1 (pixel)')",
                    f"{WORD[n_pix].lower()} of its {WORD[n_c1].lower()} C1 rows reading "
                    "\\emph{C1 (pixel)}"))
    return out


def composability_c1_rows():
    """(C1 rows, of which labelled 'C1 (pixel)') in tab:composability, read out of main.tex."""
    lines = paper_text().split("\n")
    try:
        lab = next(i for i, L in enumerate(lines) if "\\label{tab:composability}" in L)
    except StopIteration:
        return 0, 0
    end = next((i for i in range(lab, len(lines)) if "\\end{table}" in lines[i]), lab)
    cols = [L.split("&")[1].strip() for L in lines[lab:end]
            if "&" in L and "rule" not in L and not L.strip().startswith("%")]
    c1 = [c for c in cols if c.startswith("C1")]
    return len(c1), sum(1 for c in c1 if c == "C1 (pixel)")


def letter_checks(s):
    """The response letter's OWN derived counts, which no auditor can see.

    audit_letter_cites.py verifies a QUOTE against its cited line. Where the letter paraphrases instead
    of quoting, it is unguarded prose with a bare cite, and check (A) only tests that the cited line is
    non-blank. Two of those paraphrases restate coverage-dependent counts from this arm, so they go
    false at a re-substitution with nothing to catch them.

    Matched against whitespace-COLLAPSED text, because one of them wraps across a markdown line break
    and a single-line substring search cannot see it.
    """
    nb = s["n_below_floor"]
    scored = [r for r in s["rows"] if r["clears_floor"] and r["predicted"] in ("LOW", "HIGH")]
    n_low_att = sum(1 for r in scored
                    for v in r["screen"]["per_attack"].values() if v["prediction"] == "LOW")
    return [
        ("the below-floor count, paraphrased and WRAPPED across a line break",
         f"{WORD[nb]} complete pairs sit below the frozen clean-accuracy floor"),
        ("the per-attack LOW count, paraphrased rather than quoted",
         f"on {WORD[n_low_att].lower()} scored pairs C1 and C2 both hold for that attack"),
    ]


def predicates(s, pure):
    """The coupled CLAIMS, asserted on score() rather than matched as strings."""
    scored = [r for r in s["rows"] if r["clears_floor"] and r["predicted"] in ("LOW", "HIGH")]
    sup = {a: [d for d, v in pure.items()
               if v.get(a) is not None and v[a] < SUPPRESSION_THRESHOLD]
           for a in COMMITTED_ATTACKS}
    return [
        ("the corrected mechanism: the committed PIXEL backdoor has no standalone suppressor, which "
         "is what denies C1 at the pair level",
         not sup["committed_pixel"],
         f"{sup['committed_pixel']} suppress committed_pixel alone, so the pixel backdoor is no "
         "longer the attack that denies C1 and the whole paragraph's mechanism must be restated"),
        ("and committed SCALING has exactly one, which the prose names rather than denying",
         len(sup["committed_scaling"]) == 1,
         f"committed_scaling has {len(sup['committed_scaling'])} standalone suppressor(s) "
         f"{sup['committed_scaling']}, not the one the prose names"),
        ("the subsection TITLE and the 'equal by construction' sentence: the screen must predict LOW "
         "for NO scored pair",
         bool(scored) and all(r["predicted"] == "HIGH" for r in scored),
         f"{sum(1 for r in scored if r['predicted'] == 'LOW')} scored pair(s) are predicted LOW"),
        ("the TITLE's 'recall on the LOW class is $0$': recall must be 0.0 and not undefined",
         s["recall_low"] == 0.0,
         f"recall_low is {s['recall_low']!r}; if it is None the LOW class is empty among the scored "
         "cells and the title, the lead paragraph and the 'equal by construction' sentence all go "
         "false together"),
        ("precision on LOW must be undefined, as the prose says",
         s["precision_low"] is None,
         f"precision_low is {s['precision_low']!r}, so the screen predicted LOW somewhere"),
        # Corollary cor:zero_recall's hypothesis, asserted as MEMBERSHIP rather than as a number, and
        # tested at the LOOSER 0.5 mark so that it also holds at C1's 0.3. This is the claim most
        # exposed by new coverage: reputation suppresses committed scaling alone, so the moment an
        # observed-LOW pair contains reputation the sentence "the suppression that is observed being
        # emergent rather than inherited from a constituent" is false, and with it the paper's reading
        # of the corollary on this menu.
        ("cor:zero_recall's hypothesis: no observed-LOW pair contains a constituent that suppresses "
         "either committed attack alone (tested at the 0.5 mark, so it holds at C1's 0.3 too)",
         not _inherited(scored, pure, SUPPRESS_ASR),
         f"these observed-LOW pairs DO inherit suppression from a constituent: "
         f"{_inherited(scored, pure, SUPPRESS_ASR)}; the emergent-suppression reading of "
         "Corollary cor:zero_recall must be withdrawn from this menu"),
    ]


def _inherited(scored, pure, thr):
    """Observed-LOW pairs having a constituent defense that suppresses a committed attack alone."""
    return [(r["pair"], d, a, round(pure[d][a], 4)) for r in scored if r["observed"] == "LOW"
            for d in (r["d1"], r["d2"]) for a in COMMITTED_ATTACKS
            if pure.get(d, {}).get(a) is not None and pure[d][a] < thr]


def main():
    pairs, _ = load()
    s = score(pairs, load_probe())
    if not s["n_scored"]:
        sys.exit("REFUSING TO CHECK: nothing is scored yet, so the prose has no referent.")
    text = paper_text()
    pure = baseline_table()
    items, claims = checks(s, pure), predicates(s, pure)

    print("=== Arm D prose vs the frozen scorer, over "
          f"{s['n_complete']} complete / {s['n_scored']} scored pairs ===")
    bad = []
    for item, label, need in items:
        ok = bool(need.search(text)) if hasattr(need, "search") else (need in text)
        print(f"  [{'OK' if ok else 'FAIL'}] item {item:2d}  {label}")
        if not ok:
            print(f"           MISSING: {need.pattern if hasattr(need, 'pattern') else need}")
            bad.append((item, label, need))

    if os.path.exists(LETTER):
        flat = re.sub(r"\s+", " ", open(LETTER).read())
        print("\n=== the response letter's own paraphrased counts (whitespace-collapsed) ===")
        for label, need in letter_checks(s):
            ok = re.sub(r"\s+", " ", need) in flat
            print(f"  [{'OK' if ok else 'FAIL'}] {label}")
            if not ok:
                print(f"           MISSING: {need}")
                bad.append(("letter", label, need))

    print("\n=== the coupled claims, as predicates on score() ===")
    for label, ok, why in claims:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"           {why}")
            bad.append(("claim", label, why))

    if bad:
        sys.exit(f"\nREFUSING TO PASS: {len(bad)} of {len(items) + len(claims)} checks failed. "
                 "Substitute the values printed above into the paper; do not adjust this file "
                 "to match the paper.")
    print(f"\n  [OK] every figure Arm D's prose and the letter state matches the frozen "
          f"scorer at "
          f"{s['n_complete']} complete pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
