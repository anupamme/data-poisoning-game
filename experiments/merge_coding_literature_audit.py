"""
Merge the per-paper coding files into the single results/literature_audit/coding.json that
`experiments/analyze_literature_audit.py` scores.

WHY PER-PAPER FILES AND NOT ONE HAND-EDITED coding.json
Sixty papers x five criteria x (code + verbatim quotation + locator) is ~900 fields entered by hand. In
one file, a single mis-nested brace makes the whole artifact unparseable and there is no way to tell
which paper's record was damaged; a re-read of one paper means re-editing a 3000-line file. So each
paper is coded into results/literature_audit/coding/<arxiv_id>.json and this script concatenates them.
The merge is mechanical and adds nothing: it copies the screening decision, venue and year from
screening.json onto each record so those are never re-typed, and it refuses to emit a merged file if a
record contradicts the screen.

WHAT THIS SCRIPT REFUSES TO DO
  * It will not invent a criterion. A paper whose file omits C-c has no C-c, and the analyzer's
    non-negotiable 3 gate catches it there rather than here.
  * It will not merge a record whose id is not an INCLUDED paper in screening.json. Coding a paper the
    screen excluded, or coding one that is not in the frame at all, is the failure non-negotiable 7
    forbids ("no paper is added to the frame after coding begins"), so it is an error, not a warning.
  * It does not touch the `blind` flag, which is set per paper from the pre-registration's disclosure
    of the three papers known to us before the freeze.

Reads:  results/literature_audit/screening.json, results/literature_audit/coding/*.json
Writes: results/literature_audit/coding.json
Run:    python3 experiments/merge_coding_literature_audit.py
"""
import glob
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "literature_audit")
CODING = os.path.join(OUT, "coding")

# The papers known to us before the rubric was frozen. Their coding is not blind and the shipped table
# marks them. Hard-coded here rather than per file so the disclosure cannot be lost by a typo in one
# record.
#
# FOUR, NOT THREE (Amendment 2, 2026-09-10). The pre-registration originally disclosed only the three
# retrieval seeds, which conflated "seed of the frame" with "known to the coder". Every entry already in
# paper/references.bib was known to us, and one of them survives into the frame: cao2021fltrust
# (2012.13995), included precisely because main.tex:1654 asserts it is a single stage. The set is the
# title-match of the included papers against `git show 9b8a395:paper/references.bib` (37 entries) plus
# the three seeds; that match returns exactly these four, and the other 55 includes are blind.
NOT_BLIND = {"2012.13995", "2101.02281", "2201.00763", "2509.08089"}


def main():
    scr = json.load(open(os.path.join(OUT, "screening.json")))
    by_id = {r["id"]: r for r in scr["rows"]}
    included = {r["id"] for r in scr["rows"] if r["screen"] == "include"}

    files = sorted(glob.glob(os.path.join(CODING, "*.json")))
    if not files:
        print(f"No per-paper coding files in {CODING}. Nothing to merge.")
        return 1

    papers, errors = [], []
    ft = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(OUT, "fulltext"))
             if f.endswith(".txt"))
    for f in files:
        aid = os.path.splitext(os.path.basename(f))[0]
        try:
            rec = json.load(open(f))
        except json.JSONDecodeError as e:
            errors.append(f"{aid}: not valid JSON ({e})")
            continue
        if aid not in included:
            errors.append(f"{aid}: coded, but screening.json does not include it "
                          f"({by_id.get(aid, {}).get('screen', 'not in frame at all')})")
            continue
        s = by_id[aid]
        rec.update({"id": aid, "screen": "include",
                    "title": rec.get("title") or s.get("title"),
                    "venue": rec.get("venue") or s.get("venue") or "arXiv",
                    "year": rec.get("year") or s.get("year"),
                    "source": s.get("source"), "query": s.get("query"),
                    "screen_reason": s.get("screen_reason"),
                    "full_text_sought": True, "full_text_obtained": aid in ft,
                    "blind": aid not in NOT_BLIND})
        papers.append(rec)

    # Excluded papers ship too (non-negotiable 6): the analyzer prints them with their reasons, so they
    # must be present in coding.json even though they carry no criteria.
    for r in scr["rows"]:
        if r["screen"] == "exclude":
            papers.append({"id": r["id"], "title": r.get("title"), "screen": "exclude",
                           "screen_reason": r["screen_reason"], "venue": r.get("venue"),
                           "year": r.get("year"), "decided_by": r.get("decided_by")})

    if errors:
        print("REFUSING TO MERGE:")
        for e in errors:
            print(f"  {e}")
        return 1

    coded = {p["id"] for p in papers if p.get("screen") == "include"}
    missing = sorted(included - coded)
    json.dump({"note": "Merged from results/literature_audit/coding/*.json by "
                       "experiments/merge_coding_literature_audit.py. Hand input is the per-paper "
                       "files; every rate is emitted by analyze_literature_audit.py.",
               "n_included_in_frame": len(included), "n_coded": len(coded),
               "not_yet_coded": missing, "papers": papers},
              open(os.path.join(OUT, "coding.json"), "w"), indent=2)
    print(f"merged {len(coded)}/{len(included)} included papers, "
          f"{len(papers) - len(coded)} exclusions carried through")
    if missing:
        print(f"NOT YET CODED ({len(missing)}): {' '.join(missing)}")
    nft = [p["id"] for p in papers if p.get("screen") == "include"
           and not p.get("full_text_obtained")]
    if nft:
        print(f"coded WITHOUT full text on disk ({len(nft)}): {' '.join(nft)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
