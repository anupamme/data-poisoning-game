"""Measure the body-prose budget of paper/main.tex, and assert the page gate.

Two measures, because Round 39 established that they disagree:

  source chars  -- everything between \\section{Introduction} and
                   \\section*{Ethics statement}. Cheap, and the only thing to
                   watch *while* editing, but it converts poorly to pages: the
                   body runs ~6.4 chars per typeset word because of markup
                   density, so a 1,500-char cut bought only ~29 typeset words.

  rendered words -- alphabetic words per page from pdftotext, ~600 per body
                   page. This is the honest measure of whether a cut buys a
                   line.

The gate is STRUCTURAL, not a char count: the constraint is body prose
(Introduction through the Conclusion) <= 9 pages. Ethics and Reproducibility
are allowed to overflow past p9, but ONLY as the first content of p10. A
char-count proxy for this gate read 45 / 58 / 60 / 70 across four probes of the
same PDF and must not be used.

WHY THIS GATE WAS ONCE WRONG, AND WHAT IT NOW MEASURES INSTEAD. Until Round 51
check_gate() tested only which page the Conclusion HEADING landed on, and
printed "[OK] ... Ethics overflows to p10, which HEAD's own layout permits" --
a layout assumption that was true when written and silently went false. At the
time it was caught the Conclusion's own prose ran 167 rendered words / 14
typeset lines onto p10 ahead of the Ethics heading, so the main text was 10
pages against a strictly enforced 9-page limit, and this gate said OK through
nine rounds of an otherwise green battery. A reviewer found it by reading the
PDF. The heading test is necessary but not sufficient: what the limit
constrains is where the body's LAST LINE falls, not where its last heading
starts. So p(limit+1) is now dumped, the running header and the ICLR line-number
gutter are stripped, and any surviving text before the Ethics heading is a
FAILURE reported in rendered words. The verdict is taken on that WORD COUNT and
not on whether the residue string is empty: ETHICS_MARK omits the heading's own
small-caps-split first letter, so a clean p10 still leaves a bare "E" ahead of
the mark, and testing the string would fail every clean build.

Usage:
    python3 -m experiments.measure_body_chars                 # source chars only
    python3 -m experiments.measure_body_chars --pages         # + rendered words
    python3 -m experiments.measure_body_chars --gate          # + assert the gate
    python3 -m experiments.measure_body_chars --tex workshop_paper/main.tex
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BODY_START = "\\section{Introduction}"
# main.tex ends its body at the Ethics statement; the workshop twin, which has
# none, ends it at \appendix. First marker present wins.
BODY_END = ("\\section*{Ethics statement}", "\\appendix")

# ICLR headings render as small caps, which pdftotext extracts with an inserted
# space after the first letter: "C ONCLUSION", "E THICS STATEMENT".
CONCLUSION_MARK = "ONCLUSION"
ETHICS_MARK = "THICS STATEMENT"
# Stripped from a page before asking whether any body prose is left on it.
RUNNING_HEADER = "Under review as a conference paper at ICLR"


# A LaTeX comment renders as nothing, so it must not be counted as body prose.
# Round 61 found this the hard way: that round adds ~90 lines of provenance comments
# inside the body window, and every char measure here and in measure_clarity_load
# read them as text a reviewer has to hold in working memory. body chars "rose" 1868
# in a round that pulled the whole Ethics statement up onto p9. Worse, the (P#) and
# C0-C3 token gates counted the labels used INSIDE those comments to justify not
# renaming them, so documenting the constraint spent the budget the constraint
# protects. Everything below now measures the rendered part of each line only.
# An escaped \% (as in "88.1\%") is not a comment and is kept.
COMMENT_RE = re.compile(r"(?<!\\)%.*$")


def uncomment(line):
    """The rendered part of one source line: everything before an unescaped %."""
    return COMMENT_RE.sub("", line)


def body_chars(tex_path):
    """Chars between the Introduction heading and the Ethics statement."""
    lines = [uncomment(l) for l in Path(tex_path).read_text().split("\n")]
    try:
        a = next(i for i, l in enumerate(lines) if l.startswith(BODY_START))
    except StopIteration:
        raise SystemExit(f"{tex_path}: no line starting with {BODY_START!r}")
    b = next((i for i, l in enumerate(lines)
              if any(l.startswith(m) for m in BODY_END)), None)
    if b is None:
        raise SystemExit(f"{tex_path}: no line starting with any of {BODY_END!r}")
    return len("".join(lines[a:b])), a + 1, b + 1


def page_words(pdf_path):
    """Alphabetic words per page, 1-indexed."""
    out = subprocess.run(["pdftotext", str(pdf_path), "-"],
                         capture_output=True, text=True, check=True).stdout
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return [len(re.findall(r"[A-Za-z]{2,}", p)) for p in pages], pages


def check_gate(pdf_path, limit=9):
    """Body prose must end by page `limit`. Returns (ok, messages)."""
    _, pages = page_words(pdf_path)
    msgs, ok = [], True

    concl = next((i for i, p in enumerate(pages, 1) if CONCLUSION_MARK in p), None)
    ethics = next((i for i, p in enumerate(pages, 1) if ETHICS_MARK in p), None)

    if concl is None:
        return False, [f"FAIL no {CONCLUSION_MARK!r} heading found in {pdf_path}"]
    if ethics is None:
        return False, [f"FAIL no {ETHICS_MARK!r} heading found in {pdf_path}"]

    msgs.append(f"Conclusion heading on p{concl}, Ethics heading on p{ethics}")

    if concl > limit:
        return False, msgs + [f"FAIL body prose spills past p{limit}: "
                              f"the Conclusion heading is on p{concl}"]
    if ethics <= limit:
        msgs.append(f"[OK] body prose ends by p{limit} "
                    f"(Ethics opens on p{ethics}, at or before the limit)")
        return ok, msgs
    if ethics > limit + 1:
        return False, msgs + [f"FAIL Ethics opens on p{ethics}, so at least one "
                              f"whole page of body prose lies past p{limit}"]

    # Ethics opens on p(limit+1). Permitted ONLY if it is that page's first
    # content: any body prose ahead of it is main text past the limit. This is
    # the check the heading test used to skip.
    spill, n_words = _spill_before(pages[ethics - 1])
    if not n_words:
        msgs.append(f"[OK] body prose ends by p{limit} (Ethics is the first "
                    f"content of p{ethics}, which the limit permits; residue "
                    f"before the heading: {spill!r})")
        return ok, msgs
    head = spill if len(spill) <= 180 else spill[:177] + "..."
    return False, msgs + [
        f"FAIL {n_words} rendered words of body prose sit on p{ethics}, ahead of "
        f"the Ethics heading, so the main text is {ethics} pages against a "
        f"{limit}-page limit. First spilled text: {head!r}"]


def _spill_before(page_text, mark=ETHICS_MARK):
    """Body prose on `page_text` ahead of `mark`, minus header and gutter.

    ICLR renders a running header on every page and a line-number gutter that
    pdftotext extracts as lines holding nothing but digits. Neither is body
    prose, so both are stripped before deciding whether anything is left.
    """
    before = page_text[:page_text.find(mark)]
    kept = [l for l in before.split("\n")
            if l.strip()
            and RUNNING_HEADER not in l
            and not re.fullmatch(r"\s*\d+\s*", l)]
    text = " ".join(l.strip() for l in kept).strip()
    return text, len(re.findall(r"[A-Za-z]{2,}", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper/main.tex")
    ap.add_argument("--pdf", default=None, help="defaults to --tex with .pdf")
    ap.add_argument("--pages", action="store_true", help="rendered words per page")
    ap.add_argument("--gate", action="store_true", help="assert the page gate")
    ap.add_argument("--limit", type=int, default=9)
    args = ap.parse_args()

    tex = ROOT / args.tex
    pdf = ROOT / args.pdf if args.pdf else tex.with_suffix(".pdf")

    n, a, b = body_chars(tex)
    print(f"{args.tex}: body prose {n} chars (lines {a}-{b})")

    if args.pages or args.gate:
        if not pdf.exists():
            raise SystemExit(f"{pdf} does not exist; build first")

    if args.pages:
        counts, _ = page_words(pdf)
        body = counts[:args.limit + 1]
        print(f"{pdf.name}: {len(counts)} pages; alphabetic words per page "
              f"through p{len(body)}:")
        for i, c in enumerate(body, 1):
            print(f"  p{i:>2}: {c:>4}")

    if args.gate:
        ok, msgs = check_gate(pdf, args.limit)
        for m in msgs:
            print(m)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
