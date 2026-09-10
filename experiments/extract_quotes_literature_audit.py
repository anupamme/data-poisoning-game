"""
Coding AID for the literature audit. Surfaces candidate sentences and their section locators from a
paper's full text, one bucket per criterion, so the coder quotes verbatim instead of paraphrasing.

THIS SCRIPT DECIDES NOTHING. It cannot code a criterion and does not try. It is a retrieval tool over
one PDF's text: the regexes below say "a sentence like this might bear on C-b", and a human reads the
sentence and decides. The distinction matters because the pre-registration's mitigation for having one
coder is that every YES and NO ships a verbatim quotation and a locator; a script that guessed the code
and pasted a nearby sentence as its evidence would defeat exactly that mitigation.

Two things it does that a coder reading a PDF by hand does worse:
  * it reports the nearest preceding SECTION HEADING for every candidate, which is the locator the
    rubric requires, and
  * it searches the WHOLE text rather than the abstract, so a C-d ablation buried in an appendix is not
    missed just because the abstract did not advertise it. C-d gets the benefit of the doubt by rule 4,
    so failing to find an ablation that exists is the one error that would INFLATE the primary rate.

Run: python3 experiments/extract_quotes_literature_audit.py 2101.02281 [2201.00763 ...]
     python3 experiments/extract_quotes_literature_audit.py --all      (every included paper)
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "literature_audit")
FT = os.path.join(OUT, "fulltext")

# One bucket per criterion. These are RECALL-oriented: a loose pattern that surfaces ten sentences the
# coder discards is cheap, a tight one that hides the only quotable sentence is not.
BUCKETS = {
    "C-a joint application": r"\b(combin\w+|integrat\w+|two[- ]stage|three[- ]stage|multi[- ]stage|"
                             r"two[- ]phase|pipeline|jointly|in tandem|followed by|"
                             r"consists? of|comprises?|components?|modules?|first .{0,40}then)\b",
    "C-b reason for pairing": r"\b(complementar\w+|because|since|motivat\w+|rationale|"
                              r"individually|respectively|each (of which|addresses|targets)|"
                              r"cover\w* (the |different |complementary )|"
                              r"synerg\w+|orthogonal|different (types?|kinds?|regions?)|"
                              r"one .{0,30}the other|whereas)\b",
    "C-c composed-system outcome": r"\b(attack success rate|\bASR\b|backdoor accuracy|"
                                   r"robust accuracy|main task accuracy|reduces? .{0,25}to \d|"
                                   r"achieves? .{0,25}\d+(\.\d+)?\s*%)\b",
    "C-d identifying contrast": r"\b(ablation|w/o|without the|leave[- ]one[- ]out|"
                                r"replac\w+ .{0,30}with|vary\w*|varying|hold\w* .{0,20}fixed|"
                                r"disabl\w+|remov\w+ (the |one )|each component|"
                                r"component[- ]wise|contribution of (the|each))\b",
    "C-e attribution claim": r"\b(attribut\w+|responsible for|thanks to|is due to|owing to|"
                             r"stems? from|the key to|explains? why|because of|"
                             r"contributes? (to|most)|accounts? for|the reason|"
                             r"we (attribute|believe|conclude) )\b",
}
BUCKETS = {k: re.compile(v, re.I) for k, v in BUCKETS.items()}

HEADING = re.compile(r"^\s*(?:(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,70})|"
                     r"([A-Z][A-Z \-]{5,50})|"
                     r"((?:Table|Figure|Fig\.|Algorithm)\s+\d+[^\n]{0,60}))\s*$")


def load(aid):
    p = os.path.join(FT, f"{aid}.txt")
    return open(p, encoding="utf8", errors="ignore").read() if os.path.exists(p) else None


def sentences_with_locators(txt):
    """Yield (sentence, locator). Locator is the nearest preceding heading, or a page-ish offset."""
    lines = txt.split("\n")
    cur, out, buf = "(before first heading)", [], []

    def flush(loc):
        if not buf:
            return
        blob = re.sub(r"\s+", " ", " ".join(buf)).strip()
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", blob):
            s = s.strip()
            if 40 <= len(s) <= 420:
                out.append((s, loc))
        buf.clear()

    for ln in lines:
        m = HEADING.match(ln.rstrip())
        if m:
            flush(cur)
            g = [x for x in m.groups() if x]
            cur = (f"Sec. {g[0]} {g[1]}" if m.group(1) else g[0]).strip()[:70]
        else:
            buf.append(ln)
    flush(cur)
    return out


def report(aid, title=None, per_bucket=6):
    txt = load(aid)
    if txt is None:
        print(f"\n########## {aid}: NO FULL TEXT ON DISK ##########")
        return
    sents = sentences_with_locators(txt)
    print(f"\n{'#' * 100}\n### {aid}  {title or ''}\n### {len(txt)} chars, {len(sents)} sentences")
    for name, rx in BUCKETS.items():
        hits = [(s, loc) for s, loc in sents if rx.search(s)]
        # Prefer sentences that are dense in the bucket's vocabulary: more distinct matches first.
        hits.sort(key=lambda h: -len(set(m.group(0).lower() for m in rx.finditer(h[0]))))
        print(f"\n--- {name}  ({len(hits)} candidates)")
        for s, loc in hits[:per_bucket]:
            print(f"  [{loc}]\n    {s}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # --per=N trims candidates shown per bucket. Coding 59 papers means reading this output 59 times,
    # and the buckets are sorted by vocabulary density, so the quotable sentence is almost always in
    # the first few. N is a display cap only: the SEARCH is always over the whole text, so lowering it
    # cannot hide an appendix ablation from the count in the bucket header.
    per = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--per=")), 6)
    if "--all" in sys.argv:
        scr = json.load(open(os.path.join(OUT, "screening.json")))
        inc = [r for r in scr["rows"] if r["screen"] == "include"]
        for r in sorted(inc, key=lambda r: r["id"]):
            report(r["id"], (r["title"] or "")[:70], per)
        return 0
    for aid in args:
        report(aid, per_bucket=per)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
