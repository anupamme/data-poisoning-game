#!/usr/bin/env python3
"""Verify that every quotation in every literature-audit coding record occurs
VERBATIM in that paper's extracted full text.

Non-negotiable 5 of the frozen pre-registration says nothing is transcribed by
hand. Quotations are still assembled by a coder reading extracted text, so this
script is the check that the assembly did not drift: it re-reads the full text
from disk, whitespace-normalizes both sides, and asserts containment.

It DECIDES NOTHING about the codes. A quotation can be verbatim and still be the
wrong evidence for a criterion; that judgment stays in the record's locator and
coder_note, where a reader re-adjudicates it.

Three accommodations, all deliberate and all narrow:

  * A quotation may ELIDE, with " ... " marking the gap. Each span is checked
    separately, and the spans must occur in source order -- an out-of-order pair
    is reported, because " ... " asserts forward elision within one passage and
    two spans from different sections are a different claim.
  * A quotation may be truncated with a BRACKETED COMPLETION, e.g.
    "some malicious or an[omalous updates may still be present]". The text before
    the bracket must be verbatim AND the bracket must be checked too: with its
    brackets removed the whole span must also be verbatim. An earlier version of
    this script checked only the text before "[", and four invented completions
    passed it -- one of them, in a paper that carries a primary code, put words in
    the paper's mouth that its own sentence contradicts. A bracket whose contents
    are a numeric citation marker ("[34]", "[9, 20]") needs no completion check,
    because such a quotation is already verbatim as written.
  * Whitespace only is normalized (runs of space/newline collapse to one space).
    Nothing else is: not case, not hyphens, not quote marks, not the interior
    spaces that small-caps macros leave in extracted acronyms ("R O B AJ O L"),
    and not running-head artifacts that land mid-sentence. Those are preserved
    on purpose, because a quotation that needs them silently repaired is a
    quotation a reader cannot find in the source.

Usage:
    python3 experiments/verify_quotes_literature_audit.py            # all records
    python3 experiments/verify_quotes_literature_audit.py 2508.12978 # some records

Exit status is 1 if any quotation is missing or any full text is absent, so this
can gate a merge.
"""
import glob
import json
import os
import re
import sys

CODING = "results/literature_audit/coding"
FULLTEXT = "results/literature_audit/fulltext"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_text(paper_id):
    hits = sorted(glob.glob(os.path.join(FULLTEXT, paper_id + "*.txt")))
    if not hits:
        return None
    with open(hits[0], encoding="utf-8", errors="ignore") as fh:
        return norm(fh.read())


CITE_BRACKET = re.compile(r"\[\s*\d+[\s,;\-–]*(?:\d+[\s,;\-–]*)*\]")


def spans(quote):
    """The list of spans that must each appear verbatim, in source order."""
    return [norm(p) for p in quote.split(" ... ") if norm(p)]


def completion(span):
    """(before, before+inside) for a coder-supplied bracketed completion, else None.

    Returns None when the span has no bracket, or when every bracket in it is a
    numeric citation marker -- in that case the span is checked as written.
    """
    if "[" not in span:
        return None
    # Mask citation markers so only the coder-supplied bracket is treated as a
    # completion; their brackets are the paper's own text, and stripping them too
    # would make every reconstruction containing a citation fail.
    cites = CITE_BRACKET.findall(span)
    masked = CITE_BRACKET.sub("\x00", span)
    if "[" not in masked:
        return None

    def unmask(s):
        out, i = [], 0
        for ch in s:
            if ch == "\x00":
                out.append(cites[i])
                i += 1
            else:
                out.append(ch)
        return norm("".join(out))

    before = unmask(masked.split("[")[0])
    inlined = unmask(masked.replace("[", "").replace("]", ""))
    return before, inlined


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = sorted(glob.glob(os.path.join(CODING, "*.json")))
    if wanted:
        paths = [p for p in paths if os.path.basename(p)[:-5] in wanted]
        missing = set(wanted) - {os.path.basename(p)[:-5] for p in paths}
        for m in sorted(missing):
            print("NO RECORD: %s" % m)
        if missing:
            return 1

    n_quotes = 0
    n_spans = 0
    n_completions = 0
    failures = []
    no_text = []
    short = []
    unordered = []

    for path in paths:
        paper_id = os.path.basename(path)[:-5]
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        text = load_text(paper_id)
        if text is None:
            no_text.append(paper_id)
            continue
        for crit, body in sorted(rec.get("criteria", {}).items()):
            quote = body.get("quote", "")
            n_quotes += 1
            pieces = spans(quote)
            if not pieces:
                failures.append((paper_id, crit, "EMPTY QUOTE"))
                continue
            last = -1
            for span in pieces:
                n_spans += 1
                comp = completion(span)
                head = comp[0] if comp else span
                pos = text.find(head)
                if pos < 0:
                    failures.append((paper_id, crit, "SPAN: " + head[:100]))
                    continue
                if comp:
                    n_completions += 1
                    if comp[1] not in text:
                        failures.append(
                            (paper_id, crit, "COMPLETION: " + comp[1][-100:]))
                if pos < last:
                    unordered.append((paper_id, crit, head[:70]))
                last = pos
                if len(head) < 40:
                    # not a failure: a short fragment can be the only verbatim
                    # span the extractor preserved. Reported so a reader weighs it.
                    short.append((paper_id, crit, head))

    print("%d records, %d quotations, %d spans, %d bracketed completions checked"
          % (len(paths), n_quotes, n_spans, n_completions))
    for paper_id in no_text:
        print("NO FULL TEXT ON DISK: %s" % paper_id)
    if short:
        print("\nshort spans (<40 chars), reported not failed:")
        for paper_id, crit, frag in short:
            print("  %s %s  %r" % (paper_id, crit, frag))
    if unordered:
        print("\nspans out of source order, reported not failed "
              "(the locator must say the spans are not one passage):")
        for paper_id, crit, frag in unordered:
            print("  %s %s  %r" % (paper_id, crit, frag))
    if failures:
        print("\nNOT VERBATIM (%d):" % len(failures))
        for paper_id, crit, frag in failures:
            print("  %s %s  %s" % (paper_id, crit, frag))
        return 1
    if no_text:
        return 1
    print("\nall quotations verbatim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
