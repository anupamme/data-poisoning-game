"""
Frame retrieval for the pre-registered literature audit (Round 57).

WHAT THIS IS
`experiments/pre_registration_literature_audit.md`, read with its Amendment 1, specifies a frame in
three parts. This script executes all three and writes the RAW candidate list before any screening
decision is taken, which is what non-negotiable 9 requires:

    F1   every paper/references.bib entry that could be an FL poisoning/backdoor defense
    F2   five FROZEN query strings against the arXiv HTML search UI, paginated to exhaustion
    F2b  the COMPLETE title listings of USENIX Security / NDSS / IEEE S&P / PMLR, screened by one
         frozen title regex -- an exhaustive population per venue-year, with a real denominator
    F3   one-hop snowball from three named seeds, via their own reference lists

WHY THE ENDPOINTS ARE NOT THE ONES IN THE FROZEN DOCUMENT
The frozen F2 named DBLP and Semantic Scholar. Both are unreachable from this environment (DBLP serves
an anti-bot challenge; Semantic Scholar 429s; so does the arXiv *API*, which is a different surface from
the arXiv HTML UI used here). Amendment 1 records that substitution, records that HTTP status codes and
no retrieved result forced it, and freezes the replacement endpoints. **The five query strings, the year
window, the venue list, the five criteria, the decision rules and the 10-paper floor are unchanged.**

THIS SCRIPT TAKES NO SCREENING DECISION AND CODES NOTHING. It retrieves and it counts. Screening and
coding are `experiments/analyze_literature_audit.py`, which reads this script's output and refuses to
print a rate if any coded row lacks a quotation or if fewer than 10 papers survive.

Output: results/literature_audit/candidates_raw.json  (+ funnel_retrieval.json, the per-surface counts)
Run:    python3 experiments/retrieve_literature_audit.py [--full-text ID [ID ...]]
"""
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "literature_audit")
FT = os.path.join(OUT, "fulltext")
PREREG = "experiments/pre_registration_literature_audit.md"
PREREG_COMMIT = None          # set to the Amendment 1 commit hash before running; see check_frozen()

# ---------------------------------------------------------------------------------------------
# FROZEN by the pre-registration. Editing anything in this block after the first retrieval is an
# amendment, not a fix. The five query strings are verbatim from the unamended document's F2.
# ---------------------------------------------------------------------------------------------
QUERIES = [
    "federated learning backdoor defense",
    "federated learning poisoning defense",
    "byzantine robust federated aggregation",
    "federated learning backdoor mitigation",
    "robust federated learning aggregation defense",
]
YEAR_MIN, YEAR_MAX = 2019, 2026

# F2b, frozen in Amendment 1: `federated` AND (backdoor|poison|byzantine|robust), case-insensitive.
F2B_STEM = re.compile(r"federat", re.I)
F2B_TOPIC = re.compile(r"backdoor|poison|byzantin|robust", re.I)

# Complete per-venue-year listings. Each entry is (venue, year, url, title regex). ACM CCS is absent
# on purpose: its proceedings pages return 403, which Amendment 1 records as a coverage gap rather
# than a sample. PMLR volumes are the ICML/AISTATS years reachable as static indexes.
VENUE_INDEXES = [
    ("USENIX Security", y, f"https://www.usenix.org/conference/usenixsecurity{y % 100}/technical-sessions",
     r'<a href="/conference/usenixsecurity\d+/presentation/[^"]+">([^<]+)</a>')
    for y in range(19, 26)
] + [
    ("NDSS", y, f"https://www.ndss-symposium.org/ndss{y}/accepted-papers/", None)
    for y in range(2019, 2026)
] + [
    ("IEEE S&P", y, f"https://www.ieee-security.org/TC/SP{y}/program-papers.html", None)
    for y in range(2019, 2026)
] + [
    ("PMLR", vol, f"https://proceedings.mlr.press/v{vol}/", r'<p class="title">\s*([^<]+?)\s*</p>')
    # ICML 2019/2020/2021/2022/2023/2024, AISTATS 2020/2021/2022/2023/2024
    for vol in [97, 119, 139, 162, 202, 235, 108, 130, 151, 206, 238]
]

SEEDS_F3 = {"nguyen2022flame": "2101.02281",
            "rieger2022deepsight": "2201.00763",
            "fenaux2025hammer": None}      # arXiv id resolved from the bib if present

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def check_frozen():
    """Refuse to retrieve unless the pre-registration is committed AND its tree is clean.

    Same two-part gate as run_comparability_cells.check_frozen, and for the same reason: `git log -1`
    reports the last commit that touched a file and is unchanged by uncommitted edits to it, so the
    hash check alone passes for an amendment sitting dirty. Non-negotiable 1 of the pre-registration
    says the document is committed BEFORE the first retrieval; an uncommitted amendment means the
    frame this script executes is not the frame the repository contains.
    """
    if PREREG_COMMIT is None:
        print(f"REFUSING TO RETRIEVE: PREREG_COMMIT is None.\n"
              f"  1. git add {PREREG} && git commit    (Amendment 1 must be committed first)\n"
              f"  2. set PREREG_COMMIT here to that hash\n"
              f"  3. rerun. A frame is only pre-registered if it is committed before retrieval.")
        return False
    got = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20).stdout.strip()
    if not got or not got.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO RETRIEVE: {PREREG} last touched at {got or 'UNTRACKED'}, "
              f"but PREREG_COMMIT is {PREREG_COMMIT}.")
        return False
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20).stdout.strip()
    if dirty:
        print(f"REFUSING TO RETRIEVE: {PREREG} has uncommitted changes ({dirty.split()[0]}), so the "
              f"frame is not frozen at {got} whatever `git log` says.")
        return False
    print(f"[OK] {PREREG} frozen at {got}, working tree clean")
    return True


def get(url, tries=3, pause=3.0):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
                return r.read().decode("utf8", "ignore")
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(pause * (a + 1))
    print(f"    FETCH FAILED {url}  ({type(last).__name__}: {last})")
    return None


def strip_tags(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).replace("&nbsp;", " ").strip()


def year_of(arxiv_id):
    """arXiv's new-style id encodes YYMM, which is the submission date and is always present."""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    return 2000 + int(m.group(1)) if m else None


# ------------------------------------------------------------------ F1: the on-disk seed
def frame_f1():
    bib = os.path.join(BASE, "paper", "references.bib")
    txt = open(bib, encoding="utf8", errors="ignore").read()
    out = []
    # Split on the @ that starts an entry, keeping it: entries are `@type{key,` ... `}` blocks.
    for ent in re.split(r"(?m)^@", txt)[1:]:
        key = re.match(r"\w+\s*\{\s*([^,\s]+)\s*,", ent)
        title = re.search(r"(?mi)^\s*title\s*=\s*\{(.+?)\}\s*,\s*$", ent, re.S)
        if not key or not title:
            continue
        t = re.sub(r"[{}]", "", re.sub(r"\s+", " ", title.group(1))).strip()
        aid = re.search(r"arXiv:(\d{4}\.\d{4,5})", ent) or re.search(r"(\d{4}\.\d{4,5})", ent)
        if F2B_STEM.search(t) or F2B_TOPIC.search(t):
            out.append({"source": "F1-bib", "bibkey": key.group(1), "title": t,
                        "arxiv_id": aid.group(1) if aid else None})
    return out


# ------------------------------------------------------------------ F2: the five frozen queries
def frame_f2():
    out, seen = [], set()
    for q in QUERIES:
        start, total, page = 0, None, 0
        while True:
            url = ("https://arxiv.org/search/?searchtype=all&query="
                   + urllib.parse.quote_plus(q) + f"&start={start}&size=50")
            h = get(url)
            if h is None:
                break
            if total is None:
                m = re.search(r"of ([\d,]+) results", h)
                total = int(m.group(1).replace(",", "")) if m else 0
                print(f"  q={q!r}: {total} results")
            for b in re.split(r'<li class="arxiv-result">', h)[1:]:
                aid = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", b)
                ti = re.search(r'<p class="title is-5 mathjax">(.*?)</p>', b, re.S)
                ab = re.search(r'<span class="abstract-full[^"]*"[^>]*>(.*?)</span>', b, re.S)
                if not aid or not ti:
                    continue
                if aid.group(1) in seen:
                    continue
                seen.add(aid.group(1))
                out.append({"source": "F2-arxiv", "query": q, "arxiv_id": aid.group(1),
                            "title": strip_tags(ti.group(1)),
                            "abstract": strip_tags(ab.group(1)).replace("▽ More", "").strip()
                            if ab else "",
                            "year": year_of(aid.group(1))})
            page += 1
            start += 50
            if total is None or start >= total or page > 40:
                break
            time.sleep(2.0)
    return out


# ------------------------------------------------------------------ F2b: exhaustive venue listings
def frame_f2b():
    out, funnel = [], []
    for venue, year, url, pat in VENUE_INDEXES:
        h = get(url, tries=2, pause=2.0)
        if h is None:
            funnel.append({"venue": venue, "year": year, "url": url,
                           "listed": None, "retained": None, "note": "index unreachable"})
            continue
        if pat:
            titles = [strip_tags(t) for t in re.findall(pat, h)]
        else:
            # NDSS and IEEE S&P do not expose a single stable per-title tag across years, so take
            # every plausible title-length text node and let the frozen regex do the screening. This
            # over-collects, which is the safe direction: the denominator is honest and the regex is
            # the only thing that decides retention.
            titles = [strip_tags(t) for t in re.findall(r">([^<>]{25,200})<", h)]
            titles = [t for t in titles if " " in t and not t.startswith("&")]
        keep = [t for t in titles if F2B_STEM.search(t) and F2B_TOPIC.search(t)]
        funnel.append({"venue": venue, "year": year, "url": url,
                       "listed": len(titles), "retained": len(keep)})
        print(f"  {venue} {year}: listed {len(titles):4d}  retained {len(keep)}")
        for t in keep:
            out.append({"source": "F2b-venue", "venue": venue, "year": year, "title": t,
                        "arxiv_id": None})
        time.sleep(1.0)
    return out, funnel


# ------------------------------------------------------------------ F3: one-hop snowball
def frame_f3(bib_ids):
    """Dump each seed's reference list to disk. One hop, no second hop (F3 as frozen)."""
    out = []
    for key, aid in SEEDS_F3.items():
        aid = aid or bib_ids.get(key)
        if not aid:
            print(f"  {key}: no arXiv id, snowball not taken for this seed (recorded)")
            out.append({"source": "F3-snowball", "seed": key, "status": "no arxiv id"})
            continue
        txt = fetch_fulltext(aid)
        if not txt:
            out.append({"source": "F3-snowball", "seed": key, "status": "full text unreachable"})
            continue
        tail = txt[int(len(txt) * 0.55):]
        refs = re.findall(r"\[\d{1,3}\]\s*(.{20,300}?)(?=\s*\[\d{1,3}\]|\Z)", tail, re.S)
        cand = [re.sub(r"\s+", " ", r).strip() for r in refs]
        keep = [r for r in cand if F2B_STEM.search(r) and F2B_TOPIC.search(r)]
        print(f"  {key} ({aid}): {len(cand)} references parsed, {len(keep)} match the frozen regex")
        for r in keep:
            out.append({"source": "F3-snowball", "seed": key, "title": r, "arxiv_id": None})
    return out


def fetch_fulltext(arxiv_id):
    """arxiv.org/pdf/<id> -> pdftotext. The arXiv API is 429 here; this surface is 200."""
    os.makedirs(FT, exist_ok=True)
    pdf = os.path.join(FT, f"{arxiv_id}.pdf")
    txt = os.path.join(FT, f"{arxiv_id}.txt")
    if os.path.exists(txt) and os.path.getsize(txt) > 5000:
        return open(txt, encoding="utf8", errors="ignore").read()
    if not os.path.exists(pdf) or os.path.getsize(pdf) < 10000:
        r = subprocess.run(["curl", "-sL", "-A", UA["User-Agent"], "--max-time", "90",
                            "-o", pdf, f"https://arxiv.org/pdf/{arxiv_id}"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(pdf) or os.path.getsize(pdf) < 10000:
            print(f"    PDF FAILED {arxiv_id}")
            return None
        time.sleep(2.0)
    subprocess.run(["pdftotext", pdf, txt], capture_output=True, text=True)
    if not os.path.exists(txt):
        return None
    return open(txt, encoding="utf8", errors="ignore").read()


def main():
    if "--full-text" in sys.argv:
        # Full-text fetch for papers already in the candidate list. Retrieval of full text for an
        # already-framed paper is not a change to the frame, so it is allowed after freezing.
        for aid in sys.argv[sys.argv.index("--full-text") + 1:]:
            t = fetch_fulltext(aid)
            print(f"{aid}: {'%d chars' % len(t) if t else 'FAILED'}")
        return 0

    if not check_frozen():
        return 1
    os.makedirs(OUT, exist_ok=True)

    print("\n=== F1: paper/references.bib ===")
    f1 = frame_f1()
    print(f"  {len(f1)} bib entries match the topic regex")
    bib_ids = {r["bibkey"]: r["arxiv_id"] for r in f1 if r.get("arxiv_id")}

    print("\n=== F2: five frozen queries, arXiv HTML search, paginated to exhaustion ===")
    f2 = frame_f2()
    print(f"  {len(f2)} distinct arXiv ids")

    print("\n=== F2b: exhaustive venue listings, frozen title regex ===")
    f2b, funnel_venues = frame_f2b()
    print(f"  {len(f2b)} retained titles")

    print("\n=== F3: one-hop snowball from three named seeds ===")
    f3 = frame_f3(bib_ids)
    print(f"  {len(f3)} snowball rows")

    allrows = f1 + f2 + f2b + f3
    in_window = [r for r in allrows
                 if r.get("year") is None or YEAR_MIN <= r["year"] <= YEAR_MAX]

    json.dump({"prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "queries": QUERIES, "year_window": [YEAR_MIN, YEAR_MAX],
               "f2b_regex": ["federat", "backdoor|poison|byzantin|robust"],
               "candidates": in_window},
              open(os.path.join(OUT, "candidates_raw.json"), "w"), indent=2)
    json.dump({"per_surface": {"F1": len(f1), "F2": len(f2), "F2b": len(f2b), "F3": len(f3)},
               "retrieved_total": len(allrows),
               "in_year_window": len(in_window),
               "venue_listings": funnel_venues},
              open(os.path.join(OUT, "funnel_retrieval.json"), "w"), indent=2)

    print(f"\nretrieved {len(allrows)} rows, {len(in_window)} inside {YEAR_MIN}--{YEAR_MAX}")
    print(f"wrote {os.path.join(OUT, 'candidates_raw.json')} and funnel_retrieval.json")
    print("Screening and coding are analyze_literature_audit.py. This script decided nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
