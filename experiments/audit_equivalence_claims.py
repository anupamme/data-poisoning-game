"""Does any sentence still claim two-sided equivalence on an arm whose lower leg is arithmetic?

Round 76, the audit that gates Phase A's prose. Companion to measure_margin_reachability.py: that script
decides WHICH arms have an unreachable lower leg, and this one decides which SENTENCES are still written
as though they did not.

THE BURDEN IS ON THE CLAIM, NOT ON THE AUDITOR. The first version of this script tried to identify which
arm each sentence was about and only demanded a qualifier when it recognised an unreachable one. That is
the wrong way round, and it failed on the very first sentence it existed for: the body's flagship margin
claim calls the arm `Krum`, not `Krum / scaling`, so the arm pattern missed it while flagging five
appendix navigation tables instead. So the rule here is the opposite. EVERY unit that makes an
equivalence claim must either

  (a) carry a one-sided qualifier -- it has said out loud which leg is a test; or
  (b) appear in CLASSIFIED below with a reason and a class.

Anything else is a defect. A new claim sentence therefore cannot enter either document silently: it
arrives as a defect until someone classifies it. CLASSIFIED is the census Phase A5 asks for, kept as
executable code rather than as a list in a letter, because a list goes stale the moment a line moves.

  (ii) REACHABLE  the arm's identity-rung mean ASR exceeds the margin, both legs are tests, and the
                  two-sided reading is earned. These must NOT be relabelled; downgrading them would
                  give away a real result.
  (iii) METHODS   prose about the test, the rule, or the appendix's own structure. Asserts no arm's
                  verdict, so there is no leg to qualify.
  (iv) OTHER-SENSE the word `equivalence` in an unrelated sense -- statistic equivalence between two
                  defenses (P1-P3), not equivalence of an outcome to zero.

THE UNIT IS A PARAGRAPH, NOT A LINE. A LaTeX source line break falls wherever the text wrapped, so a
claim and its subject routinely sit on different lines. Units are runs of consecutive non-blank,
non-comment lines; a full-comment line SPLITS a unit even though LaTeX would weld the paragraphs, which
is the conservative direction here because it stops a qualifier in one paragraph from excusing an
unqualified claim in the next.

TWO DEFECT KINDS, because the inverse error is just as wrong:
  MISSING_QUALIFIER  an equivalence claim with neither a one-sided qualifier nor a classification.
  WRONGLY_QUALIFIED  a one-sided qualifier attached to a unit whose only named arm is a reachable one.

A SECOND, UNGATED POPULATION IS REPORTED IN FULL. The frozen point-estimate rule |Delta| < 0.15 has the
same arithmetic lower side as the TOST lower leg: `Delta > -0.15` is satisfied for free whenever the
identity mean is below the margin. Those units are counted and listed rather than gated, because that
rule is a pre-registered decision procedure and not an inference claim -- but the count is printed so the
disclosure cannot be described as narrower than it is.

Run: PYTHONPATH=. python3 -m experiments.audit_equivalence_claims
Exit: 0 iff there are no defects and no stale classifications.
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REACH = os.path.join(BASE, "results", "margin_reachability.json")
DOCS = [os.path.join(BASE, "paper", "main.tex"),
        os.path.join(BASE, "paper", "supplementary.tex")]

# An equivalence claim: the word itself, the test's name, or the supplement's defined phrase for it.
CLAIM = re.compile(r"equivalen|TOST|practically meaningful change")
# The one-sided reading, in any form Phase A is allowed to write it in.
QUALIFIER = re.compile(
    r"one-sided|non-increase|does not rise|cannot (fall|rise)|largest possible fall|fall available"
    r"|arithmetic|satisfied before|upper bound|upper leg|lower leg|reachab|unreachab"
    r"|bounded below by \$?0|ref\{par:reach")
# The frozen point-estimate rule, whose lower side is arithmetic for the same reason. Reported, not
# gated. Excludes units already in the CLAIM population so the two counts do not overlap.
POINT_RULE = re.compile(r"\\pm ?0\.15|\\pm0\.15|< ?0\.15|<0\.15|inside [^.]{0,40}margin")
# Arms whose lower leg IS a test, named as the paper names them. Used only for the inverse check, where
# the names are specific enough to be safe.
REACHABLE_TEX = r"cos\\?_?krum|coord\\?_?median|CIFAR-100"


def units(path):
    """Paragraph units of (first_line, last_line, joined_text), comment lines dropped and splitting."""
    out, cur, start = [], [], None
    for i, line in enumerate(open(path).read().split("\n"), 1):
        s = line.strip()
        if not s or s.startswith("%"):
            if cur:
                out.append((start, i - 1, " ".join(cur)))
            cur, start = [], None
            continue
        if start is None:
            start = i
        cur.append(s)
    if cur:
        out.append((start, start + len(cur) - 1, " ".join(cur)))
    return out


# THE CENSUS. Keyed on a distinguishing substring of the joined unit, so ordinary re-wrapping does not
# invalidate an entry. A key that stops matching is reported as stale: a dead classification silently
# narrows the audit's population, which is the same defect as a dead exemption in any other instrument
# here.
CLASSIFIED = {
    # ---- (iii) METHODS: prose about the test, the rule, or the appendix's own structure ----
    "A point estimate inside a margin is consistent":
        ("iii", "App. F's own methods paragraph: defines what TOST adds to the point rule, asserts no "
                "arm's verdict"),
    "Equivalence requires the $90\\%$ interval inside":
        ("iii", "tab:tost's caption defines the test and its dagger marker; the per-arm verdicts are "
                "in the rows"),
    "equivalence testing (App.~\\ref{sec:limitations}":
        ("iii", "the appendix roadmap's list of section topics, in document order"),
    "equivalence testing at the existing seed counts, that is, where the frozen":
        ("iii", "the per-section index line for App. F, naming its topic"),
    "the ASR practical-equivalence results at $n{=}5$, one of which was topped up":
        ("iii", "tab:tiers' evidence-tier cell, which locates the results rather than scoring them"),
    "A reading of no evidence of a practically meaningful change requires the paired":
        ("iii", "supplementary.tex's DEFINITION of the phrase by the stricter interval convention; "
                "standing rule forbids redefining it, and it is not a claim about an arm"),
    "no null result is read as flatness unless it clears a pre-registered practical-equivalence":
        ("iii", "the criterion menu's regime paragraph states the same discipline, not a verdict"),
    "no null is read as flatness} without a pre-registered practical-equivalence margin":
        ("iii", "tab:can_cannot states the discipline the paper holds itself to, not any arm's "
                "verdict"),
    "against the suite's existing $\\pm 0.15$ practical-equivalence margin, on \\texttt{krum}":
        ("iii", "recites the score-only arm's FROZEN primary rule as frozen; a freeze is quoted, "
                "never amended, and the reading built on it is relabelled at its own site instead"),
    "no equivalence claim is made anywhere in this arm":
        ("iii", "an explicit denial: the modeS-causal disagreement arm makes no equivalence claim, "
                "so it has no lower leg to qualify"),
    "which is an interval-overlap statement and not the equivalence phrase the supplement defines":
        ("iii", "an explicit denial; EQUIV_MARGIN is not imported by that arm"),
    "the equivalence phrase the supplement defines against the stricter of our two interval":
        ("iii", "an explicit denial, and it already says unresolved is not a zero"),
    "Leg~1 was frozen as $|\\Delta\\mathrm{ASR}|$ \\emph{inside} the $0.15$ equivalence margin":
        ("iii", "a REPORTED FAILURE of the frozen rule at +0.2969 with the interval excluding zero "
                "and outside the margin, in the direction of increase; nothing here rests on the "
                "lower side"),
    # ---- (iv) OTHER SENSE: equivalence between two things, not of an outcome to zero ----
    "equal values induce the same ordering":
        ("iv", "P1-P3: equivalence of a preservation statistic across defenses, an unrelated sense"),
    "Within the positive per-client rescaling class there exist transform families":
        ("iv", "Prop. nonidentifiability: observational equivalence of two transform families"),
    "can identify attack-suppression equivalence, because (a) exhibits statistic disturbance":
        ("iv", "the corollary's identification statement: equivalence of suppression BETWEEN two "
               "transform families, not of one arm's effect to zero"),
    "the analytical formulation is equivalent without requiring a differentiable relaxation":
        ("iv", "equivalence of two optimizer formulations at convergence"),
    "so quoted spans are rendered with a LaTeX equivalent":
        ("iv", "typographic substitution in the literature audit's quotations"),
}


def main():
    if not os.path.exists(REACH):
        sys.exit(f"MISSING {os.path.relpath(REACH, BASE)}: run measure_margin_reachability first.")
    art = json.load(open(REACH))
    rows = art["rows"]
    margin = art["frozen_equivalence_margin"]
    unreach = sorted({r["arm"] for r in rows if not r["lower_leg_is_a_test"]})
    reach = sorted({r["arm"] for r in rows if r["lower_leg_is_a_test"]})

    print("=== ARE ANY TWO-SIDED EQUIVALENCE CLAIMS STILL WRITTEN ON AN ARITHMETIC LOWER LEG? ===")
    print(f"    measured population, {os.path.relpath(REACH, BASE)}, frozen margin {margin}:")
    print(f"      lower leg ARITHMETIC, {len(unreach)} arm-rows: {'; '.join(unreach)}")
    print(f"      lower leg A TEST,     {len(reach)} arm-rows: {'; '.join(reach)}")
    print("    Every unit making an equivalence claim must carry a one-sided qualifier or be "
          "classified.")

    re_re = re.compile(REACHABLE_TEX)
    defects, classified, used, point_only = [], [], set(), []
    n_units = n_claims = 0
    for path in DOCS:
        rel = os.path.relpath(path, BASE)
        for lo, hi, text in units(path):
            n_units += 1
            if not CLAIM.search(text):
                if POINT_RULE.search(text):
                    point_only.append((rel, lo, hi))
                continue
            n_claims += 1
            key = next((k for k in CLASSIFIED if k in text), None)
            if key is not None:
                used.add(key)
                cls, why = CLASSIFIED[key]
                classified.append((rel, lo, hi, cls, why))
                continue
            if not QUALIFIER.search(text):
                defects.append(("MISSING_QUALIFIER", rel, lo, hi, text))
            elif re_re.search(text) and not re.search(r"flagship|Krum|krum|EMNIST|esnet18", text):
                defects.append(("WRONGLY_QUALIFIED", rel, lo, hi, text))

    print(f"\n    {n_units} paragraph units across both documents.")
    print(f"    {n_claims} make an equivalence claim: {len(classified)} classified, "
          f"{n_claims - len(classified) - len(defects)} carry a one-sided qualifier, "
          f"{len(defects)} defect(s).")
    print(f"    {len(point_only)} further units invoke the frozen point rule without the word: "
          "reported, not gated.")

    stale = [k for k in CLASSIFIED if k not in used]
    if stale:
        print("\n    ** STALE CLASSIFICATIONS: these keys match nothing, so they classify nothing and")
        print("       narrow the audit's population. Delete them or fix the key. **")
        for k in stale:
            print(f"      {k[:80]!r}")

    if classified:
        print("\n  CLASSIFIED, the census, with the class and the reason:")
        for rel, lo, hi, cls, why in sorted(classified, key=lambda r: (r[0], r[1])):
            print(f"    ({cls}) {rel}:{lo}-{hi}\n          {why}")

    if point_only:
        print("\n  THE POINT RULE'S OWN LOWER SIDE, same arithmetic, reported and not gated:")
        print("    " + ", ".join(f"{r}:{a}-{b}" for r, a, b in point_only))

    if defects:
        print("\n  DEFECTS:")
        for kind, rel, lo, hi, text in defects:
            print(f"\n    [{kind}] {rel}:{lo}-{hi}\n      {text[:260]}")
        print(f"\n    {len(defects)} defect(s). A MISSING_QUALIFIER unit must say which leg is a test "
              "or be\n    classified; a WRONGLY_QUALIFIED unit is about an arm that earned the "
              "two-sided reading.")
    else:
        print("\n    No equivalence claim is unqualified and unclassified, and no one-sided qualifier")
        print("    is attached to a unit whose only named arm has a testable lower leg.")

    return 1 if (defects or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
