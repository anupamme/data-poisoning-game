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
are allowed to overflow past p9 -- that is HEAD's own layout, so we assert
against it rather than against a recorded number. A char-count proxy for this
gate read 45 / 58 / 60 / 70 across four probes of the same PDF and must not be
used.

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


def body_chars(tex_path):
    """Chars between the Introduction heading and the Ethics statement."""
    lines = Path(tex_path).read_text().split("\n")
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
    if ethics <= limit:
        msgs.append(f"[OK] body prose ends by p{limit} "
                    f"(Ethics opens on p{ethics}, at or before the limit)")
    else:
        # HEAD's layout: Conclusion ends p9, Ethics opens p10. Still passing.
        if concl <= limit:
            msgs.append(f"[OK] body prose ends by p{limit} "
                        f"(Conclusion on p{concl}; Ethics overflows to p{ethics}, "
                        f"which HEAD's own layout permits)")
        else:
            ok = False
            msgs.append(f"FAIL body prose spills past p{limit}: "
                        f"the Conclusion heading is on p{concl}")
    return ok, msgs


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
