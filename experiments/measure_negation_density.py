"""Measure the fraction of body-prose sentences that carry a grammatical negation.

Round 49 measured this at 50%, Round 51 at 63%, and the fifteenth review's §10
("the paper's many disclaimers sometimes obscure the positive contribution") is
what 63% reads like from the outside. The number drifted 13 points across two
rounds with nobody watching it, so the measurement lives in the repo rather than
in a session transcript, and the marker list is FROZEN here: a density is only
comparable against itself.

This counts GRAMMAR, not content. A sentence that states a scope condition, a
limitation or a withdrawal is doing necessary work; the target is the same
content written in the positive ("is not sufficient to establish X" ->
"establishes X only when Y"), never the deletion of a caveat. So a rising count
is a warning and a falling count is not automatically an improvement: read the
diff.

Body window and prose filtering follow measure_body_chars.py, so the two agree
on what "the body" is.

SECOND MEASURE: THE DISCLAIMER FLOOR. "Read the diff" does not scale to a round
that rewrites a dozen sentences, and the failure mode is specific: a positive
rewrite and a deleted caveat both make the density fall. So PROTECTED below is a
frozen inventory of the disclosures the paper may not lose, and its count is a
FLOOR -- it may never drop, while the density above it may fall freely. A round
that softens a caveat fails here; a round that only changes voice passes.

Each pattern matches the disclosure's REFERENT, never its negation. Anchoring on
"not" would make the floor fire on exactly the rewrites it is meant to permit,
which inverts the whole measure. Two counts are reported because the standing
rule allows a disclosure to MOVE (consolidation relocates a home, it never
removes one): presence anywhere in the file is the floor, and the in-body subset
is informational -- a drop there means something moved to the appendix and wants
a human's eye. Presence is scanned over the RAW body window, not the prose
sentences, because several of these live in boxes, captions and list items that
prose_lines() deliberately drops.

Usage:
    python3 -m experiments.measure_negation_density
    python3 -m experiments.measure_negation_density --tex workshop_paper/main.tex
    python3 -m experiments.measure_negation_density --rev HEAD    # compare to a commit
    python3 -m experiments.measure_negation_density --list        # print the negatives
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BODY_START = "\\section{Introduction}"
BODY_END = ("\\section*{Ethics statement}", "\\appendix")

# Floats and displays are not prose: their captions are measured elsewhere and
# their cells are not sentences.
SKIP_ENVS = ("table", "table*", "figure", "figure*", "tabular", "center",
             "itemize", "enumerate", "align", "equation", "algorithm")

# Structural lines carry no prose.
SKIP_PREFIX = ("\\section", "\\subsection", "\\subsubsection", "\\label",
               "\\begin", "\\end", "\\caption", "\\includegraphics", "\\toprule",
               "\\midrule", "\\bottomrule", "\\vspace", "\\hspace", "\\centering",
               "\\bibliography", "\\input", "\\fbox", "%")

# FROZEN marker list. Grammatical negation only: valence words ("fails",
# "refutes", "limitation") are deliberately absent, because rewriting those is a
# content change and this measure must not reward it.
NEG_TOKENS = ("not", "no", "none", "neither", "nor", "never", "nothing",
              "nobody", "cannot", "without", "nowhere", "nonetheless")
NEG_RE = re.compile(r"\b(?:" + "|".join(NEG_TOKENS) + r")\b|n't", re.I)

# FROZEN disclosure inventory. (name, referent pattern). See the module docstring:
# every pattern names WHAT is disclosed, so a positive-voice rewrite of the same
# content still matches and only a deletion fails.
PROTECTED = (
    ("abstract/recall-capped",        r"recall is capped"),
    ("abstract/admission-not-a-predictor", r"admission predicts security"),
    ("intro/no-transfer-beyond-setting", r"controlled setting we study"),
    ("intro/our-own-refutations",     r"false negative of our own criterion"),
    ("estimand/P-not-a-covariate",    r"measured covariate"),
    ("collider/half-is-textbook",     r"that half is textbook"),
    ("threat-model/not-adaptive",     r"sharpest thing this framework lacks"),
    ("prop1/sufficiency-only",        r"identify attack-suppression preservation"),
    ("tost/margin-not-load-bearing",  r"the one arm that needs it"),
    ("krum/admission-zero-is-a-floor", r"zero is a floor here"),
    ("krum/floor-not-load-bearing",   r"rests on Krum's admission floor"),
    ("channels/no-per-channel-decomposition", r"per-channel decomposition"),
    ("channels/table-has-a-positive-control", r"its own positive control"),
    ("reversal/no-p-value-on-difference", r"-value attaches"),
    ("mask/one-mask-is-not-the-class", r"class of all non-rescalings"),
    ("scope/box-exists",              r"label\{box:scope\}"),
)

# Sentence-final periods that are not sentence ends.
ABBREV = ("Prop", "Thm", "Cor", "Lem", "Def", "Fig", "Tab", "App", "Sec", "Eq",
          "vs", "cf", "resp", "approx", "e.g", "i.e", "Dr", "et al", "Ref")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def prose_lines(text):
    """The body's prose lines, floats and structural commands removed."""
    lines = text.split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    b = next(i for i, l in enumerate(lines)
             if any(l.startswith(m) for m in BODY_END))
    depth = 0
    out = []
    for l in lines[a:b]:
        s = l.strip()
        # Scan the WHOLE line, not just its start: `\end{tabular}\end{center}`
        # on one line left the depth counter stuck open and silently reduced the
        # body to its Introduction (14 lines of 227).
        opened = closed = 0
        for kind, env in re.findall(r"\\(begin|end)\{(\w+\*?)\}", s):
            if env in SKIP_ENVS:
                if kind == "begin":
                    opened += 1
                else:
                    closed += 1
        was_open = depth
        depth = max(depth + opened - closed, 0)
        if was_open or opened or closed or not s or s.startswith(SKIP_PREFIX):
            continue
        out.append(s)
    return out


def strip_markup(s):
    """Math, refs and citations become opaque tokens so they cannot split or
    trip the negation match ($(P4)\\not\\Rightarrow(P5)$ is not prose)."""
    s = re.sub(r"\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(?:ref|eqref|cite[a-z]*|citep|citet|label)\{[^}]*\}", " REF ", s)
    s = re.sub(r"\\(?:textbf|emph|texttt|textit|noindent|paragraph)\b", " ", s)
    s = re.sub(r"[{}~]", " ", s)
    for a in ABBREV:
        s = s.replace(a + ".", a + "<DOT>")
    s = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", s)
    return s


def sentences(lines):
    out = []
    for l in lines:
        for s in SPLIT_RE.split(strip_markup(l)):
            s = s.replace("<DOT>", ".").strip()
            # A fragment with no alphabetic run of 3+ is not a sentence.
            if re.search(r"[A-Za-z]{3,}", s):
                out.append(s)
    return out


def density(text):
    sents = sentences(prose_lines(text))
    neg = [s for s in sents if NEG_RE.search(s)]
    return len(neg), len(sents), neg


def body_window(text):
    """The raw body lines, unfiltered: floats and list items included, because
    several protected disclosures live in exactly those."""
    lines = text.split("\n")
    a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    b = next(i for i, l in enumerate(lines)
             if any(l.startswith(m) for m in BODY_END))
    return "\n".join(lines[a:b])


def protected(text):
    """(present_in_file, in_body, total, missing_names). The first is the floor."""
    body = body_window(text)
    present, in_body, missing = 0, 0, []
    for name, pat in PROTECTED:
        rx = re.compile(pat)
        if rx.search(text):
            present += 1
            if rx.search(body):
                in_body += 1
        else:
            missing.append(name)
    return present, in_body, len(PROTECTED), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/main.tex")
    ap.add_argument("--rev", help="git revision to compare against")
    ap.add_argument("--list", action="store_true", help="print negative sentences")
    args = ap.parse_args()

    path = ROOT / args.tex
    text = path.read_text()
    n, tot, neg = density(text)
    print(f"{args.tex}: {n}/{tot} body sentences carry a negation = {100*n/tot:.1f}%")

    have, in_body, ptot, missing = protected(text)
    print(f"  protected disclaimers: {have}/{ptot} present ({in_body} in the body window)")
    for name in missing:
        print(f"    MISSING: {name}")

    if args.rev:
        rel = str(Path(args.tex))
        old = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.rev}:{rel}"],
                             capture_output=True, text=True)
        if old.returncode:
            sys.exit(f"cannot read {rel} at {args.rev}: {old.stderr.strip()}")
        m, mtot, _ = density(old.stdout)
        ohave, oin_body, _, _ = protected(old.stdout)
        print(f"{args.rev}:  {m}/{mtot} = {100*m/mtot:.1f}%, "
              f"protected {ohave}/{ptot} ({oin_body} in body)")
        print(f"delta: {100*n/tot - 100*m/mtot:+.1f} points, "
              f"{n-m:+d} negative sentences, {tot-mtot:+d} sentences, "
              f"protected {have-ohave:+d} ({in_body-oin_body:+d} in body)")

    if args.list:
        for s in neg:
            print(f"  - {s[:160]}")

    # The floor, not a ratchet: the density above may fall, this may not.
    if missing:
        print(f"  FAIL: the disclaimer floor dropped to {have}/{ptot}. A caveat was "
              f"removed, not rewritten; restore it or correct its pattern.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
