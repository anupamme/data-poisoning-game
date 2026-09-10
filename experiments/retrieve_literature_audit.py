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
import html
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
# DELIBERATELY NOT BUMPED TO cafa771. The retrieval RAN at 3c61301, before Amendments 2-5 existed, and
# this constant records when it ran rather than what the rubric says now -- so the frame was fixed before
# any adjudication rule was written, which is the strongest thing the audit can say about itself. The
# consequence is that check_frozen() now REFUSES, by design: a re-run is a new retrieval and must re-pin
# to the rubric it runs under and re-report its own funnel. Do not "fix" the refusal by editing the hash.
PREREG_COMMIT = "3c61301"     # freeze 9b8a395 + Amendment 1 (3c61301), which re-pointed F2 from DBLP
                              # and Semantic Scholar to the arXiv HTML search UI and added F2b's
                              # exhaustive venue listings. Forced by HTTP status codes; no retrieved
                              # result informed it. The queries, criteria, rules and floor are
                              # unchanged. This script refuses to run if the document drifts.

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
    # `year` is the real four-digit year in every row, so the funnel reports a year and not a URL
    # fragment; USENIX's two-digit path is derived, not stored.
    ("USENIX Security", y,
     f"https://www.usenix.org/conference/usenixsecurity{y % 100}/technical-sessions",
     r'<a href="/conference/usenixsecurity\d+/presentation/[^"]+">([^<]+)</a>')
    for y in range(2019, 2026)
] + [
    ("NDSS", y, f"https://www.ndss-symposium.org/ndss{y}/accepted-papers/", None)
    for y in range(2019, 2026)
] + [
    ("IEEE S&P", y, f"https://www.ieee-security.org/TC/SP{y}/program-papers.html", None)
    for y in range(2019, 2026)
] + [
    # PMLR is indexed by volume, so the volume is carried WITH its year rather than in place of it --
    # a funnel row reading "PMLR 202" would look like a year to every reader of the table.
    (f"PMLR v{vol} ({conf})", yr, f"https://proceedings.mlr.press/v{vol}/",
     r'<p class="title">\s*([^<]+?)\s*</p>')
    for vol, conf, yr in [(97, "ICML", 2019), (119, "ICML", 2020), (139, "ICML", 2021),
                          (162, "ICML", 2022), (202, "ICML", 2023), (235, "ICML", 2024),
                          (108, "AISTATS", 2020), (130, "AISTATS", 2021), (151, "AISTATS", 2022),
                          (206, "AISTATS", 2023), (238, "AISTATS", 2024)]
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


INLINE_TAGS = re.compile(r"</?(?:span|a|em|b|i|strong|sup|sub|code|small)\b[^>]*>", re.I)


def strip_tags(s):
    """Delete INLINE tags, space out block tags, then unescape entities.

    arXiv wraps every matched query term in its own `<span class="search-hit">`, and the terms are
    separated by real spaces in the source. Replacing an inline tag with a space therefore invents
    one: `<span>Federated</span> <span>Learning</span>:` came out as "Federated Learning :" -- the
    spurious space before punctuation that was visible in the first retrieval's titles. Inline tags
    are deleted; block tags still become a space, so words either side of a `</p>` do not fuse.
    """
    s = INLINE_TAGS.sub("", s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


# arXiv's abstract span CONTAINS nested <span class="search-hit"> elements, one per matched term, so a
# `(.*?)</span>` capture terminates at the first highlighted word. On the first retrieval that made
# every one of the 483 "abstracts" the single word "Federated". The span is the last child of
# <p class="abstract">, so terminate on the "More"/"Less" toggle anchor or on </p> instead.
ARXIV_TITLE_RE = re.compile(r'<p class="title is-5 mathjax">(.*?)</p>', re.S)
ARXIV_ABSTRACT_RE = re.compile(r'<span class="abstract-full[^"]*"[^>]*>(.*?)'
                               r'(?:<a class="is-size-7"|</span>\s*</p>|</p>)', re.S)


def parse_arxiv_results(h, q, seen):
    """Parse one arXiv search-results page into candidate rows. ONE implementation on purpose.

    The first retrieval had this block copied into both frame_f2() and retry_429(), so the abstract
    defect had to be fixed twice and could have been fixed in only one. Both callers now share this.
    """
    rows = []
    for b in re.split(r'<li class="arxiv-result">', h)[1:]:
        aid = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", b)
        ti = ARXIV_TITLE_RE.search(b)
        ab = ARXIV_ABSTRACT_RE.search(b)
        if not aid or not ti or aid.group(1) in seen:
            continue
        seen.add(aid.group(1))
        rows.append({"source": "F2-arxiv", "query": q, "arxiv_id": aid.group(1),
                     "title": strip_tags(ti.group(1)),
                     "abstract": strip_tags(ab.group(1)) if ab else "",
                     "year": year_of(aid.group(1))})
    return rows


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
            out.extend(parse_arxiv_results(h, q, seen))
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


def retry_429():
    """Re-issue any FROZEN query that returned nothing because arXiv rate-limited it, and merge.

    The first run got HTTP 429 on query 5 after ~14 page fetches, so that query contributed zero rows
    while the other four contributed 472. Leaving it there would mean the reported frame silently
    omits a fifth of its own pre-registered query list.

    This is NOT a frame change and cannot become one: it re-issues only strings already in QUERIES, it
    derives WHICH ones from their absence in the existing artifact rather than from anything retrieved,
    and it merges into the candidate list rather than replacing it. The retry is recorded in
    funnel_retrieval.json so a reader can see that a query needed one. Backoff is much longer than
    get()'s default, which is what the 429 was telling us.
    """
    raw_p = os.path.join(OUT, "candidates_raw.json")
    if not os.path.exists(raw_p):
        print("no candidates_raw.json; run the full retrieval first")
        return 1
    raw = json.load(open(raw_p))
    have = {c.get("query") for c in raw["candidates"] if c.get("source") == "F2-arxiv"}
    missing = [q for q in QUERIES if q not in have]
    if not missing:
        print("every frozen query is represented in the artifact; nothing to retry")
        return 0
    print(f"frozen queries with zero rows (rate-limited on the first pass): {missing}")
    seen = {c["arxiv_id"] for c in raw["candidates"] if c.get("arxiv_id")}
    added = []
    for q in missing:
        start, total, page = 0, None, 0
        while True:
            url = ("https://arxiv.org/search/?searchtype=all&query="
                   + urllib.parse.quote_plus(q) + f"&start={start}&size=50")
            h = get(url, tries=5, pause=45.0)          # 45/90/135/180/225s; the 429 asked for patience
            if h is None:
                print(f"  q={q!r}: STILL UNREACHABLE after 5 attempts with long backoff.")
                break
            if total is None:
                m = re.search(r"of ([\d,]+) results", h)
                total = int(m.group(1).replace(",", "")) if m else 0
                print(f"  q={q!r}: {total} results")
            added.extend({**r, "retrieved_on_retry": True}
                         for r in parse_arxiv_results(h, q, seen))
            page += 1
            start += 50
            if total is None or start >= total or page > 40:
                break
            time.sleep(8.0)                            # slower than the first pass, on purpose
    keep = [r for r in added if r["year"] is None or YEAR_MIN <= r["year"] <= YEAR_MAX]
    raw["candidates"] = raw["candidates"] + keep
    raw["retry_429"] = {"queries_retried": missing, "rows_added": len(keep),
                        "note": "Re-issue of frozen queries that returned HTTP 429 on the first pass. "
                                "No query string, year window or venue was changed."}
    json.dump(raw, open(raw_p, "w"), indent=2)
    fp = os.path.join(OUT, "funnel_retrieval.json")
    if os.path.exists(fp):
        f = json.load(open(fp))
        f["retry_429"] = raw["retry_429"]
        f["in_year_window"] = len(raw["candidates"])
        json.dump(f, open(fp, "w"), indent=2)
    print(f"merged {len(keep)} rows from {len(missing)} retried query/queries; "
          f"candidate list now {len(raw['candidates'])}")
    return 0


PARSER_FIX = {
    "id": "parser-fix-abstract-and-title",
    "what_was_wrong": [
        "The abstract capture `<span class=\"abstract-full...\">(.*?)</span>` terminated at the first "
        "NESTED <span class=\"search-hit\"> that arXiv wraps around each matched query term, so all 483 "
        "F2 'abstracts' were the single word 'Federated' (median length 9 characters). Title/abstract "
        "screening on that artifact would have been title-only while reporting itself as both.",
        "strip_tags() replaced every tag with a space, including those same inline highlight spans, "
        "which invented a space at each highlight boundary: 'Federated Learning : A Survey', "
        "'Defenses , Frameworks'. Titles were corrupted at exactly the matched terms.",
    ],
    "what_changed": "Only the HTML parser: inline tags are deleted rather than spaced, entities are "
                    "unescaped, the abstract span terminates on the More/Less anchor or </p>, and the "
                    "one block is shared by frame_f2() and retry_429() instead of copied.",
    "what_did_not_change": "No query string, year window, venue, criterion, decision rule or floor. "
                           "The same five frozen queries are re-issued against the same frozen URL "
                           "template from Amendment 1. This is not an amendment: nothing the "
                           "pre-registration freezes was edited.",
    "when": "Before any paper was screened or coded. No inclusion, exclusion or criterion decision was "
            "taken on the defective artifact, so nothing downstream needs re-doing.",
}


def reparse_f2():
    """Re-issue the five frozen queries and merge CORRECTED titles and abstracts into the artifact.

    The search-results pages were never cached -- only PDFs were -- so recovering the abstracts means
    re-fetching the same pages. Same endpoint, same frozen query strings, same pagination rule: this is
    a re-execution of F2 as pre-registered, with a parser that reads what the page actually says.

    Merge, never replace. Existing rows are matched by arXiv ID and only `title` and `abstract` are
    overwritten; `source`, `query`, `year` and any `retrieved_on_retry` marker are left alone. A row
    the first pass missed is appended and marked, so the artifact can still be diffed against itself.
    """
    raw_p = os.path.join(OUT, "candidates_raw.json")
    if not os.path.exists(raw_p):
        print("no candidates_raw.json; run the full retrieval first")
        return 1
    raw = json.load(open(raw_p))
    byid = {c["arxiv_id"]: c for c in raw["candidates"] if c.get("arxiv_id")}
    fixed_ab, fixed_ti, appended = 0, 0, []
    for q in QUERIES:
        start, total, page, seen = 0, None, 0, set()
        while True:
            url = ("https://arxiv.org/search/?searchtype=all&query="
                   + urllib.parse.quote_plus(q) + f"&start={start}&size=50")
            h = get(url, tries=5, pause=30.0)
            if h is None:
                print(f"  q={q!r}: UNREACHABLE at start={start}; partial reparse for this query")
                break
            if total is None:
                m = re.search(r"of ([\d,]+) results", h)
                total = int(m.group(1).replace(",", "")) if m else 0
                print(f"  q={q!r}: {total} results")
            for r in parse_arxiv_results(h, q, seen):
                old = byid.get(r["arxiv_id"])
                if old is None:
                    if r["year"] is None or YEAR_MIN <= r["year"] <= YEAR_MAX:
                        r["found_on_reparse"] = True
                        appended.append(r)
                        byid[r["arxiv_id"]] = r
                    continue
                if r["abstract"] and r["abstract"] != old.get("abstract"):
                    old["abstract"] = r["abstract"]
                    fixed_ab += 1
                if r["title"] and r["title"] != old.get("title"):
                    old["title"] = r["title"]
                    fixed_ti += 1
            page += 1
            start += 50
            if total is None or start >= total or page > 40:
                break
            time.sleep(8.0)
    raw["candidates"] = raw["candidates"] + appended
    raw["parser_fix"] = {**PARSER_FIX, "abstracts_repaired": fixed_ab, "titles_repaired": fixed_ti,
                         "rows_appended": len(appended)}
    json.dump(raw, open(raw_p, "w"), indent=2)
    ab = [len(c.get("abstract") or "") for c in raw["candidates"] if c.get("source") == "F2-arxiv"]
    usable = sum(1 for n in ab if n >= 200)
    fp = os.path.join(OUT, "funnel_retrieval.json")
    if os.path.exists(fp):
        f = json.load(open(fp))
        f["parser_fix"] = raw["parser_fix"]
        f["f2_rows_with_usable_abstract"] = usable
        f["f2_rows"] = len(ab)
        json.dump(f, open(fp, "w"), indent=2)
    print(f"repaired {fixed_ab} abstracts and {fixed_ti} titles; appended {len(appended)} rows")
    print(f"F2 rows with an abstract of >=200 chars: {usable}/{len(ab)}")
    return 0


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def resolve_titles():
    """Resolve arXiv IDs for already-framed candidates that carry a title but no ID.

    This is FULL-TEXT ACQUISITION, not a change to the frame. Every row it touches is already in
    candidates_raw.json; no new paper enters, and a row that fails to resolve stays in the artifact and
    is excluded later at the "full text sought" stage exactly as the unamended pre-registration
    requires. Without it the F2b venue rows -- USENIX, NDSS, IEEE S&P, PMLR, the security venues
    Amendment 1 added F2b to cover -- would all be dropped for want of an identifier, which would
    realize the arXiv bias far harder than Amendment 1 bargained for.

    Matching is STRICT: the arXiv title, normalized to alphanumerics, must equal the candidate's or
    contain it. A loose match here would attribute a quotation to the wrong paper, which is the one
    error this audit cannot absorb.
    """
    raw_p = os.path.join(OUT, "candidates_raw.json")
    raw = json.load(open(raw_p))
    todo = [c for c in raw["candidates"]
            if not c.get("arxiv_id") and c.get("source") in ("F1-bib", "F2b-venue")
            and len(c.get("title") or "") > 25]
    print(f"{len(todo)} framed rows carry a title but no arXiv id")
    known = {norm_title(c["title"]): c["arxiv_id"]
             for c in raw["candidates"] if c.get("arxiv_id") and c.get("title")}
    hit = 0
    for c in todo:
        nt = norm_title(c["title"])
        if nt in known:                          # already retrieved under another surface
            c["arxiv_id"], c["id_resolved_by"] = known[nt], "dedup-within-frame"
            hit += 1
            continue
        url = ("https://arxiv.org/search/?searchtype=all&query="
               + urllib.parse.quote_plus(c["title"][:180]) + "&start=0&size=25")
        h = get(url, tries=3, pause=20.0)
        if h is None:
            print(f"  UNREACHABLE  {c['title'][:64]}")
            continue
        got = None
        for r in parse_arxiv_results(h, "title-resolution", set()):
            rn = norm_title(r["title"])
            if rn == nt or (len(nt) > 30 and (nt in rn or rn in nt)):
                got = r
                break
        if got:
            c["arxiv_id"], c["id_resolved_by"] = got["arxiv_id"], "arxiv-title-match"
            if not c.get("abstract"):
                c["abstract"] = got["abstract"]
            hit += 1
            print(f"  {got['arxiv_id']}  {c['title'][:62]}")
        else:
            c["id_unresolved"] = True
            print(f"  NO MATCH     {c['title'][:62]}")
        time.sleep(6.0)
    raw["title_resolution"] = {
        "attempted": len(todo), "resolved": hit, "unresolved": len(todo) - hit,
        "note": "Full-text acquisition for rows already in the frame (F1-bib and F2b-venue titles that "
                "carry no arXiv identifier). No candidate was added or removed. Strict normalized-title "
                "equality or containment only. Unresolved rows remain in the artifact and are excluded "
                "at the 'full text sought' stage with that reason.",
    }
    json.dump(raw, open(raw_p, "w"), indent=2)
    print(f"resolved {hit}/{len(todo)}")
    return 0


def main():
    if "--resolve-titles" in sys.argv:
        if not check_frozen():
            return 1
        return resolve_titles()
    if "--reparse-f2" in sys.argv:
        if not check_frozen():
            return 1
        return reparse_f2()
    if "--retry-429" in sys.argv:
        if not check_frozen():
            return 1
        return retry_429()
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
