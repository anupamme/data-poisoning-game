#!/usr/bin/env python3
"""Check that the body stands alone and that the appendix index can be navigated.

Written in answer to a review whose complaint is *"74 pages of appendix against 9 pages of body"*. The
previous round measured the appendix and found nothing to delete: 81% of it is prose, and of 14
near-duplicate sentence pairs exactly one has the same numbers in both members
(`measure_appendix_redundancy`). So the complaint is not about duplication, and the answer to it is not
a deletion. What a reader is actually reacting to is 66 pointers out of a 9-page body into a 74-page
appendix, roughly ten per page. This file measures the property that makes that acceptable -- every
pointer is *provenance for a claim the body already states*, not a deferral of the claim itself -- and
the property that makes the appendix usable, which is that its index is complete and in document order.

    python3 -m experiments.measure_self_containment          # report, exit 1 on any defect
    python3 -m experiments.measure_self_containment --quiet   # verdict lines only

Four checks, each of which has caught something real:

1. **Every body pointer resolves.** A `\\ref` with no `\\label` is a LaTeX warning, but a body pointer
   into a *section that was renamed* resolves to the wrong place silently. This asserts every target
   label exists and reports which appendix section each body pointer lands in.

2. **No body sentence defers its content.** The defect this was written for: `:530` read "The screening
   conditions the levels induce, and their names, are in App. C" -- so C0--C3, the paper's first
   contribution, were never *named* in the body at all. The deny-list below is phrasal and small on
   purpose; it fires on a clause whose whole content is a pointer. A pointer in parentheses after a
   stated claim is the good case and must not fire, which is why the patterns all require the pointer
   to be the clause's grammatical object.

3. **The index covers every top-level appendix section exactly once.** The index is hand-maintained and
   the appendix grows every round, so a new `\\section` silently drops out of it. No build, gate or hash
   can see this: the index is prose about the document's own shape.

4. **The index is in document order.** It says it is. It was not: entries 4 and 5 printed as "App. E"
   then "App. D", so a reader using it as an index walked backwards at item 5. Order is checked on the
   *section start line* each label resolves to, not on the label's own line, because a label sitting a
   line below its `\\section` is the normal case and a label sitting several lines below a `\\section`
   inside a preceding subsection is the failure this catches.

What it does not do: it cannot tell whether a body claim is *adequately* stated, only whether the
sentence hands its content away. Check 2 is a tripwire on phrasing, not a judgement about content.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"
MAIN = PAPER / "main.tex"

# The body window, as `measure_body_chars` and `measure_clarity_load` define it: the prose between
# the first \section of the paper and the appendix. Resolved from the source, never hardcoded, because
# every round moves these lines.
BODY_START_RE = re.compile(r"^\\section\{")
APPENDIX_RE = re.compile(r"^\\appendix\b")

REF_RE = re.compile(r"\\(?:ref|autoref|Cref|cref)\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
SECTION_RE = re.compile(r"^\\section\*?\{(.+?)\}\s*$")
ITEM_REF_RE = re.compile(r"^\\item\s+\\textbf\{App\.~\\ref\{([^}]+)\}\}")

# A clause defers when the pointer is what the sentence is *about*. Each pattern is anchored on a verb
# whose object is the reference, so "X holds (App. C)" cannot match and "X is in App. C" must.
DEFERRAL_PATTERNS = (
    (r"\b(?:are|is)\s+in\s+(?:App|Appendix|§)", "'... is in App. X' -- state the thing, then cite it"),
    (r"\bsee\s+(?:App|Appendix|§)\.?~?\\ref", "'see App. X' -- a bare redirection"),
    (r"\b(?:are|is)\s+(?:given|listed|stated|defined|deferred)\s+(?:in|to)\s+(?:App|Appendix|§)",
     "'... is stated in App. X' -- the body never states it"),
    (r"\b(?:we\s+)?(?:defer|relegate)\b.{0,40}?(?:App|Appendix|§)", "an explicit deferral"),
    (r"\bfor\s+(?:details|the\s+details|more)\b.{0,30}?(?:App|Appendix|§)", "'for details see ...'"),
)

# Clauses that match a deferral pattern and are not deferrals, because what they hand to the appendix
# is *provenance metadata* rather than a claim: an artifact path, a round count, an emitter. A caption
# cannot carry an emitter path, and no reader's evaluation of the table depends on having it inline.
# Each entry must match at least one flagged clause, or it is itself reported as a defect -- the same
# discipline as measure_negation_density's PROTECTED, and for the same reason: an exception that has
# gone stale is an exception that is silently covering something else. Adding an entry here is a
# judgement on record, not a way to make the check pass; the object deferred must be metadata, and the
# clause must be the tail of a caption whose claims are all stated in it.
ALLOWED = (
    ("channel-table/provenance-in-appendix",
     r"provenance, round counts and the emitter are in App",
     "Table 3's caption states every claim it makes; what it points out to is the artifact's "
     "provenance, its per-arm round counts and the emitter script."),
)


def load():
    lines = MAIN.read_text().split("\n")
    body_start = next(i for i, l in enumerate(lines) if BODY_START_RE.match(l))
    appendix = next(i for i, l in enumerate(lines) if APPENDIX_RE.match(l))
    return lines, body_start, appendix


def is_comment(line: str) -> bool:
    return line.lstrip().startswith("%")


def labels(lines):
    """label -> 1-indexed defining line, non-comment lines only."""
    out = {}
    for i, l in enumerate(lines, 1):
        if is_comment(l):
            continue
        for m in LABEL_RE.finditer(l):
            out.setdefault(m.group(1), i)
    return out


def sections(lines, appendix):
    """[(1-indexed start line, title)] for every top-level appendix \\section, in document order."""
    out = []
    for i in range(appendix, len(lines)):
        if is_comment(lines[i]):
            continue
        m = SECTION_RE.match(lines[i].strip())
        if m:
            out.append((i + 1, m.group(1)))
    return out


def owning_section(secs, line):
    best = None
    for start, title in secs:
        if start <= line:
            best = (start, title)
    return best


def clauses(text):
    """Split a source line into clauses. A deferral lives in one clause, so splitting on ; and : as
    well as . keeps a parenthetical citation in a different clause from the claim it supports."""
    return re.split(r"(?<=[.:;])\s+", re.sub(r"\s+", " ", text).strip())


def main(argv):
    quiet = "--quiet" in argv
    lines, body_start, appendix = load()
    lab = labels(lines)
    secs = sections(lines, appendix)
    defects = []

    # ---- 1. every body pointer into the appendix resolves -------------------------------------
    # An xr pointer into supplementary.tex resolves in the twin document, so a target is missing only
    # if it is absent from BOTH. Reading the twin is what keeps this check from firing on every \ref
    # the body makes into the supplement, which is the reason it was dead when first written: the
    # earlier version tested membership in a dict it had already filtered on, so it could never fire
    # while the paper asserted that it does.
    supp_lab = labels((PAPER / "supplementary.tex").read_text().split("\n"))
    pointers, unresolved = [], []
    for i in range(body_start, appendix):
        ln, l = i + 1, lines[i]
        if is_comment(l):
            continue
        for m in REF_RE.finditer(l):
            t = m.group(1)
            d = lab.get(t)
            if d is None:
                if t not in supp_lab:
                    unresolved.append((ln, t))
                continue
            if d > appendix:
                pointers.append((ln, t, d))
    for ln, t in unresolved:
        defects.append(f"[1] :{ln} pointer \\ref{{{t}}} has no \\label in main.tex or supplementary.tex")

    if not quiet:
        print(f"body window        : source lines {body_start + 1}--{appendix}")
        print(f"pointers into appx : {len(pointers)} "
              f"({len({t for _, t, _ in pointers})} distinct targets)")
        landed = {}
        for ln, t, d in pointers:
            o = owning_section(secs, d)
            landed.setdefault(o[1] if o else "?", []).append(t)
        for title, ts in sorted(landed.items(), key=lambda kv: -len(kv[1])):
            print(f"    {len(ts):3d}  {title[:66]}")

    # ---- 2. no body sentence defers its content ----------------------------------------------
    deferrals = []
    for i in range(body_start, appendix):
        ln, l = i + 1, lines[i]
        if is_comment(l):
            continue
        if not REF_RE.search(l):
            continue
        for c in clauses(l):
            if not REF_RE.search(c):
                continue
            for pat, why in DEFERRAL_PATTERNS:
                if re.search(pat, c, re.I):
                    deferrals.append((ln, why, c))
                    break
    allowed_hits = {name: 0 for name, _, _ in ALLOWED}
    kept = []
    for ln, why, c in deferrals:
        exc = next((n for n, pat, _ in ALLOWED if re.search(pat, c, re.I)), None)
        if exc:
            allowed_hits[exc] += 1
        else:
            kept.append((ln, why, c))
    for ln, why, c in kept:
        defects.append(f"[2] :{ln} defers its content to the appendix ({why})\n        {c[:160]}")
    for name, _, _ in ALLOWED:
        if allowed_hits[name] == 0:
            defects.append(f"[2] documented exception '{name}' no longer matches any clause; the "
                           f"prose it excused has changed, so the exception must be re-judged or removed")

    # ---- 3. the index covers every top-level appendix section exactly once --------------------
    index = []
    for i in range(appendix, len(lines)):
        if is_comment(lines[i]):
            continue
        m = ITEM_REF_RE.match(lines[i].strip())
        if m:
            index.append((i + 1, m.group(1)))
    # The guide's own section is not indexed by itself.
    content_secs = [(s, t) for s, t in secs if t != "How to read this appendix"]
    covered = {}
    for ln, t in index:
        d = lab.get(t)
        if d is None:
            defects.append(f"[3] :{ln} index entry \\ref{{{t}}} has no \\label")
            continue
        o = owning_section(secs, d)
        covered.setdefault(o[0], []).append(t)
    for s, t in content_secs:
        if s not in covered:
            defects.append(f"[3] appendix section :{s} \"{t[:60]}\" has no index entry")
    for s, ts in covered.items():
        if len(ts) > 1:
            defects.append(f"[3] appendix section :{s} indexed {len(ts)} times: {ts}")

    # ---- 4. the index is in document order ---------------------------------------------------
    prev_start, prev_label = -1, None
    for ln, t in index:
        d = lab.get(t)
        if d is None:
            continue
        o = owning_section(secs, d)
        if o is None:
            continue
        if o[0] < prev_start:
            defects.append(
                f"[4] :{ln} index entry \\ref{{{t}}} (section :{o[0]}) is listed after "
                f"\\ref{{{prev_label}}} (section :{prev_start}); the index claims document order")
        prev_start, prev_label = o[0], t

    if not quiet:
        print(f"index entries      : {len(index)} for {len(content_secs)} content sections")
        print(f"deferral tripwire  : {len(deferrals)} clause(s) flagged, "
              f"{sum(allowed_hits.values())} covered by a documented exception, {len(kept)} defect(s)")

    print()
    if defects:
        print(f"SELF-CONTAINMENT: FAIL, {len(defects)} defect(s)")
        for d in defects:
            print(f"  {d}")
        return 1
    print("SELF-CONTAINMENT: OK")
    print("  every body pointer into the appendix resolves and attaches to a stated claim;")
    print("  the index covers every top-level appendix section exactly once, in document order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
