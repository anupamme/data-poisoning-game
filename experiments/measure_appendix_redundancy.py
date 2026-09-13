#!/usr/bin/env python3
"""Measure how much of the appendix is actually redundant text.

Written for Round 63, in answer to a review that says *"you could remove 20-30% of the appendix
without losing scientific content"*. Three previous rounds answered that with a refusal in prose. A
refusal is not checkable; this is, and it is the reason the file exists rather than a paragraph in the
response letter. Run it, read the pairs it prints, and disagree with the numbers if they are wrong.

    python3 -m experiments.measure_appendix_redundancy            # summary + every pair, with sites
    python3 -m experiments.measure_appendix_redundancy --verbose  # full text of both members

What it does and, more usefully, what it cannot do:

* It finds *textual* near-duplicates: sentence pairs by Jaccard over 3-word shingles, paragraph pairs
  by containment of the smaller in the larger. Both thresholds are deliberately LOW (0.30 and 0.15),
  so the output over-reports. Every pair it prints was read by hand before anything was edited. At the
  Round 63 state that is 3 sentence pairs out of 1011 sentences and 12 paragraph pairs out of 209
  paragraphs, and **exactly one** pair -- of either kind -- has the same numbers in both members. The
  rest are the same sentence *shape* carrying a different `n`, a different arm, or a prose restatement
  standing beside a formal statement, which is a real finding about this appendix and not a limitation
  of the measurement. The one same-numbers pair is `:1110` against `:1526`, and `:1110`'s own first
  clause declares it: *"Both are developed in full later, but neither should be met late."*
* It cannot find *conceptual* redundancy, and it does not claim to. Twenty-seven claims that each
  carry their own tier word and their own scope hedge read as repetitive while sharing almost no
  shingles. That is what a reader is reacting to, and the honest answer to it is the reading guide,
  not a deletion.
* Comment lines are stripped before anything is measured, because roughly a third of this file's
  appendix lines are provenance comments that no reader ever sees. Counting them as prose would
  inflate every denominator. `%` inside `\\%` is not a comment; see the regex.

Environment bodies (tabular, verbatim, equation-likes) are excluded from the prose pool and reported
separately, since a table's cells legitimately repeat column headers and would dominate any shingle
measure. The split is printed so the pool is auditable rather than asserted.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parent.parent / "paper" / "main.tex"

SENT_SHINGLE = 3
PARA_SHINGLE = 3
SENT_JACCARD = 0.30
PARA_CONTAINMENT = 0.15
MIN_SENT_WORDS = 8
MIN_PARA_WORDS = 40

# Environments whose bodies are not prose. Kept as a list rather than a "starts with tab" heuristic
# so that adding one is a visible edit.
NON_PROSE_ENVS = {
    "tabular", "tabularx", "longtable", "array", "verbatim", "lstlisting", "equation", "equation*",
    "align", "align*", "gather", "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
    "tikzpicture", "figure", "figure*", "table", "table*", "thebibliography",
}


def strip_comments(text: str) -> tuple[str, int, int]:
    """Drop comment text. Returns (stripped, comment_chars, prose_chars).

    A line whose first non-space char is `%` is dropped entirely INCLUDING its newline, which is the
    same thing LaTeX does and the reason a comment block can weld two paragraphs together. An inline
    `%` not preceded by a backslash truncates its line.
    """
    out, comment_chars = [], 0
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            comment_chars += len(line)
            continue
        m = re.search(r"(?<!\\)%", line)
        if m:
            comment_chars += len(line) - m.start()
            line = line[: m.start()]
        out.append(line)
    stripped = "\n".join(out)
    return stripped, comment_chars, len(stripped)


def appendix_window(text: str) -> str:
    i = text.find("\\appendix")
    if i < 0:
        sys.exit("no \\appendix in main.tex, so the appendix window cannot be located")
    j = text.find("\\end{document}", i)
    return text[i : j if j > 0 else len(text)]


def split_envs(text: str) -> tuple[str, int]:
    """Remove non-prose environment bodies. Returns (prose, env_chars)."""
    env_chars = 0
    while True:
        m = re.search(r"\\begin\{(" + "|".join(re.escape(e) for e in NON_PROSE_ENVS) + r")\}", text)
        if not m:
            break
        end = re.search(r"\\end\{" + re.escape(m.group(1)) + r"\}", text[m.end():])
        if not end:
            # Unbalanced: refuse rather than silently swallow the rest of the appendix.
            sys.exit(f"unbalanced \\begin{{{m.group(1)}}} at offset {m.start()}")
        stop = m.end() + end.end()
        env_chars += stop - m.start()
        text = text[: m.start()] + "\n\n" + text[stop:]
    return text, env_chars


def normalize(s: str) -> str:
    s = re.sub(r"\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(ref|eqref|citep|citet|cite|label|autoref)\{[^}]*\}", " XREF ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}~\\&$^_#]", " ", s)
    return s


def words(s: str) -> list[str]:
    # Numbers are KEPT as tokens on purpose: `n=5` vs `n=20` is exactly what distinguishes the pairs
    # that look like duplicates and are not.
    return re.findall(r"[a-z0-9]+", normalize(s).lower())


def shingles(ws: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(ws[i : i + k]) for i in range(max(0, len(ws) - k + 1))}


def paragraphs(prose: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", prose) if p.strip()]


def sentences(para: str) -> list[str]:
    flat = re.sub(r"\s+", " ", para)
    # Split on . ! ? followed by space + capital, protecting the abbreviations this paper uses.
    flat = re.sub(r"\b(App|Sec|Fig|Tab|Thm|Def|Prop|Lem|Cor|Alg|vs|cf|e\.g|i\.e|resp)\.", r"\1@", flat)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", flat)
    return [p.replace("@", ".").strip() for p in parts if p.strip()]


def numbers(s: str) -> set[str]:
    """Numeric literals in a fragment, which is the deletion test.

    Two fragments of the same shape are safe to consolidate only if they say the same thing about the
    same measurement. In this appendix the near-duplicates are one disclosure template instantiated per
    arm, so their numbers differ; a pair whose numeric sets are EQUAL is the only kind that can be
    deleted without removing an arm's disclosure. Printed per pair so the claim is checkable rather
    than asserted.

    Read from the RAW fragment, deliberately not from `normalize()`. `normalize()` collapses every
    `$...$` to the token MATH, and almost every number in this paper is inside math, so normalizing
    first reported the two `Three disclosures` paragraphs -- one at n=3, one at n=5 -- as having the
    same numbers. Label, ref and cite arguments are dropped first, because arXiv ids and years are not
    measurements.
    """
    s = re.sub(r"\\(ref|eqref|label|citep|citet|cite|autoref)\{[^}]*\}", " ", s)
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def verdict(a: str, b: str) -> str:
    na, nb = numbers(a), numbers(b)
    if not na and not nb:
        return "no numbers in either: judge by hand"
    if na == nb:
        return f"SAME NUMBERS ({len(na)}): candidate for consolidation"
    only_a, only_b = sorted(na - nb)[:6], sorted(nb - na)[:6]
    return f"different numbers: only-A {only_a}, only-B {only_b}"


def source_line(raw_lines: list[str], para: str) -> int:
    """1-based source line of a paragraph, located by its leading line.

    Stripping environments destroys offsets, so the line number is recovered by search rather than
    tracked. Returns 0 if the lead line is not found verbatim, which happens when an environment
    removal spliced two fragments together; a 0 is printed as `?` rather than guessed at.
    """
    lead = para.split("\n")[0].strip()
    if len(lead) > 90:
        lead = lead[:90]
    for i, line in enumerate(raw_lines):
        if lead and line.strip().startswith(lead):
            return i + 1
    return 0


def main() -> int:
    verbose = "--verbose" in sys.argv
    raw = MAIN.read_text(encoding="utf-8")
    raw_lines = raw.split("\n")
    body, comment_chars, _ = strip_comments(raw)
    app = appendix_window(body)
    app_all = len(app)
    prose, env_chars = split_envs(app)
    prose_chars = len(prose)

    print(f"appendix window        : {app_all} non-comment chars")
    print(f"  environment bodies   : {env_chars} ({100*env_chars/app_all:.1f}%) excluded from the pool")
    print(f"  prose pool           : {prose_chars} ({100*prose_chars/app_all:.1f}%)")
    print(f"  comments stripped    : {comment_chars} chars document-wide")

    paras = [p for p in paragraphs(prose) if len(words(p)) >= MIN_PARA_WORDS]
    sents: list[tuple[int, str]] = []
    for pi, p in enumerate(paras):
        for s in sentences(p):
            if len(words(s)) >= MIN_SENT_WORDS:
                sents.append((pi, s))
    lines = [source_line(raw_lines, p) for p in paras]
    def site(i):
        return f":{lines[i]}" if lines[i] else ":?"
    print(f"  paragraphs >= {MIN_PARA_WORDS}w    : {len(paras)}")
    print(f"  sentences >= {MIN_SENT_WORDS}w      : {len(sents)}")

    sent_sets = [shingles(words(s), SENT_SHINGLE) for _, s in sents]
    hits = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            a, b = sent_sets[i], sent_sets[j]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= SENT_JACCARD:
                hits.append((jac, i, j))
    hits.sort(reverse=True)
    print(f"\nsentence pairs at Jaccard >= {SENT_JACCARD} over {SENT_SHINGLE}-shingles: {len(hits)}")
    for jac, i, j in hits:
        print(f"  J={jac:.2f}  para {sents[i][0]} ({site(sents[i][0])}) / para {sents[j][0]} ({site(sents[j][0])})")
        print(f"    A: {sents[i][1][:150] if not verbose else sents[i][1]}")
        print(f"    B: {sents[j][1][:150] if not verbose else sents[j][1]}")
        print(f"    -> {verdict(sents[i][1], sents[j][1])}")

    para_sets = [shingles(words(p), PARA_SHINGLE) for p in paras]
    phits = []
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            a, b = para_sets[i], para_sets[j]
            if not a or not b:
                continue
            cont = len(a & b) / min(len(a), len(b))
            if cont >= PARA_CONTAINMENT:
                phits.append((cont, i, j))
    phits.sort(reverse=True)
    print(f"\nparagraph pairs at containment >= {PARA_CONTAINMENT} over {PARA_SHINGLE}-shingles: {len(phits)}")
    for cont, i, j in phits:
        print(f"  C={cont:.2f}  paras {i} ({site(i)}) / {j} ({site(j)})  ({len(words(paras[i]))}w / {len(words(paras[j]))}w)")
        print(f"    A: {paras[i][:180] if not verbose else paras[i]}")
        print(f"    B: {paras[j][:180] if not verbose else paras[j]}")
        print(f"    -> {verdict(paras[i], paras[j])}")

    same = [1 for _, i, j in hits if numbers(sents[i][1]) == numbers(sents[j][1])]
    psame = [1 for _, i, j in phits if numbers(paras[i]) == numbers(paras[j])]
    print(f"\npairs whose numeric sets are EQUAL: {len(same)} sentence, {len(psame)} paragraph")
    print("Both thresholds over-report by design; read every pair before deleting anything. A pair with")
    print("different numbers is one disclosure template instantiated on a different arm, and deleting")
    print("either member removes that arm's disclosure rather than a repetition. In Round 63 the single")
    print("same-numbers pair was a preview whose own opening clause declares it as one, so nothing was")
    print("deleted and the measurement itself is what answers the reviewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
