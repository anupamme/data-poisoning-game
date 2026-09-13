#!/usr/bin/env python3
"""Audit a response letter's `:NNN` source cites against the .tex files they point into.

WHY THIS EXISTS. The letters cite main.tex and supplementary.tex by SOURCE LINE NUMBER, and those
numbers move whenever a round inserts anything above them -- including a `%` provenance comment, which
is invisible to every LaTeX check and to both paper gates. Round 63 shipped 26 stale cites out of 53
that way. Two facts make eyeballing insufficient:

  1. A stale cite usually lands on real prose and so passes any "is the line non-blank" check. Only 10
     of those 26 failed that test; the other 16 read as perfectly good cites to the wrong sentence.
  2. The PDF's margin prints ICLR *rendered* line numbers, which are offset from source lines and drift.
     A cite transcribed from a pixel read is wrong even when the letter was written the same hour.

So this script runs the two passes that actually catch those, and never guesses an offset:

  (A) every cited line exists, is non-blank, and does not start with `%`
  (B) every quoted substring the letter puts on the same markdown line as a cite is present ON that
      cited line -- and when it is not, the substring is searched for across the whole file and the
      line where it really lives is reported, which is the only trustworthy source of a shift

Pass (B) is the one with teeth, and its blind spot is stated rather than hidden: a quote that wraps onto
a second markdown line is not paired with any cite, so it is counted and listed as UNCHECKED for a human
to read. Quoting text the round DELETED must not share a line with a cite; that correctly fails here.

Cites are read as `` `:NNN` `` (main.tex, the default) or `` `supplementary.tex:NNN` ``. Nothing is
written: the shift is reported for a single deliberate regex pass elsewhere, because applying shifts
sequentially double-counts any pair where one cite's new value is another's old one.

Run: python3 paper/audit_letter_cites.py paper/response_reviewer_6of10_v8.md
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# `:NNN` optionally prefixed by a filename. Two digits minimum, so a "§5.1" style token cannot match.
CITE = re.compile(r"`(?:(main|supplementary)\.tex)?:(\d{2,4})`")
# The letter's own quoting styles: *"..."*, **"..."** and bare "..." inside a table cell.
QUOTE = re.compile(r'\*{0,2}"([^"]{8,})"\*{0,2}')


def load(name):
    with open(os.path.join(HERE, f"{name}.tex")) as fh:
        return fh.read().split("\n")


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip().split("\n\n")[-1])
        return 2
    letter = argv[1] if os.path.isabs(argv[1]) else os.path.join(os.path.dirname(HERE), argv[1])
    src = {"main": load("main"), "supplementary": load("supplementary")}
    fails, n_cites, n_quotes, unchecked, shifts = [], 0, 0, [], []

    for ln, line in enumerate(open(letter).read().split("\n"), 1):
        cites = CITE.findall(line)
        quotes = QUOTE.findall(line)
        n_cites += len(cites)
        # (A) Each cited line must exist and carry rendered content.
        for fname, num in cites:
            fname = fname or "main"
            lines, i = src[fname], int(num)
            if i > len(lines):
                fails.append((ln, f"{fname}.tex:{i} past end of file ({len(lines)} lines)"))
                continue
            text = lines[i - 1]
            if not text.strip():
                fails.append((ln, f"{fname}.tex:{i} is BLANK"))
            elif text.lstrip().startswith("%"):
                fails.append((ln, f"{fname}.tex:{i} is a COMMENT: {text.strip()[:60]}"))
        # (B) Each quote on this line must appear on one of this line's cited lines.
        for q in quotes:
            n_quotes += 1
            if not cites:
                unchecked.append((ln, q[:70], "quote with no cite on the same line"))
                continue
            hit = False
            for fname, num in cites:
                fname = fname or "main"
                if q in src[fname][int(num) - 1]:
                    hit = True
                    break
            if hit:
                continue
            # Not on any cited line: find where it really is, per file, and report the true line.
            where = [(f, j + 1) for f in ("main", "supplementary")
                     for j, t in enumerate(src[f]) if q in t]
            if not where:
                fails.append((ln, f'quote ABSENT from both files: "{q[:60]}"'))
            else:
                loc = ", ".join(f"{f}.tex:{j}" for f, j in where)
                cited = ", ".join(f"{f or 'main'}.tex:{n}" for f, n in cites)
                fails.append((ln, f'quote NOT on {cited}; it is at {loc}: "{q[:50]}"'))
                for f, j in where:
                    shifts.append((cites, f, j))

    print(f"letter: {os.path.relpath(letter)}")
    print(f"  cites checked   : {n_cites}")
    print(f"  quotes checked  : {n_quotes} ({len(unchecked)} unchecked by construction)")
    for ln, q, why in unchecked:
        print(f"    letter:{ln}  {why}: \"{q}\"")
    print(f"  FAIL            : {len(fails)}")
    for ln, why in fails:
        print(f"    letter:{ln}  {why}")
    if shifts:
        print("\n  resolved anchors (apply as ONE regex pass, never sequentially):")
        for cites, f, j in shifts:
            print(f"    {', '.join((c[0] or 'main') + '.tex:' + c[1] for c in cites)}  ->  {f}.tex:{j}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
