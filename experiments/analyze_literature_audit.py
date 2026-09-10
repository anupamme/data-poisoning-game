"""
Score the pre-registered literature audit against the rubric in
`experiments/pre_registration_literature_audit.md` (freeze + Amendments 1-5).

WHICH PARTS OF THAT RUBRIC ARE PRE-REGISTERED, BECAUSE NOT ALL OF IT IS
The primary quantity, the five criteria's frozen text, the four ambiguity rules, the frame and its
screening procedure, the 10-paper floor and the three admissible outcomes were fixed at 9b8a395 and
Amendment 1 at 3c61301, both before any paper was retrieved, and this script scores exactly them.
Amendments 2-3 were written with 5 rows coded and Amendments 4-5 with all 59; they are adjudication
rules for questions the criteria underdetermine, and the rubric's closing section tabulates that
chronology. Only Amendment 4a moves anything printed here: primary = 6 with it and 5 without, and the
LaTeX table is required to report both. So the hash gate below pins a document that is part
pre-registration and part post-hoc adjudication record, and the [OK] line says so rather than calling
the whole file frozen.

WHAT IS COMPUTED, AND WHAT IS HAND INPUT
The frame is retrieved by `experiments/retrieve_literature_audit.py`. The CODING is by one human coder
reading full text -- the pre-registration says so and declines to claim an inter-rater statistic it
cannot compute. So `results/literature_audit/coding.json` is hand input, and every RATE, COUNT and
LaTeX row is emitted here. Nothing downstream is transcribed. The mitigation for single-coder coding is
that each YES/NO ships a verbatim quotation and a locator, and this script REFUSES to print a rate if
any of them is missing: a reader re-adjudicates rather than trusts.

THE PRIMARY IS A CONJUNCTION AND ITS ¬C-d TERM IS THE PART THAT COULD BE TAUTOLOGICAL
    OUTCOME-GATED ATTRIBUTION := C-a AND C-b AND C-c AND C-e AND NOT C-d
C-e (an attribution claim is actually made) is what stops this from measuring "nobody runs the contrast
we invented", which would be near-100% and worthless. The C-d presence rate is therefore printed
SEPARATELY and never folded in, so a reader can see how much of the primary rides on the predictable
absence of our own contrast. The count of C-e = NO *among DESIGN papers* is printed too: those papers
have the design and make no attributional claim, and they are evidence AGAINST the sharp reading. They
are reported as such, next to the C-e = NO count over the whole frame, which is a different and much
weaker thing -- a paper with no design and no claim is evidence in neither direction.

TWO REFUSAL GATES, BOTH FROZEN
  * non-negotiable 3: no rate if any included paper codes YES or NO without a quotation AND a locator.
  * non-negotiable 4: no rate if fewer than 10 papers are included and coded. Outcome (iii) of the
    pre-registration -- "the frame was too small to support a rate" -- is a publishable result and this
    script prints it as one rather than printing a rate over 6 papers.

UNCLEAR RESOLVES AGAINST THE FINDING
Rule 1-2: any UNCLEAR criterion makes the paper's primary verdict UNCLEAR, and UNCLEAR papers leave
BOTH the numerator and the denominator. That can only shrink the measured prevalence relative to a coder
who guesses, which is the direction a pre-registration is allowed to be wrong in.

Reads:  results/literature_audit/candidates_raw.json, coding.json
Writes: results/literature_audit/summary.json  (the only artifact any LaTeX number may come from)
Run:    python3 experiments/analyze_literature_audit.py
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "literature_audit")
PREREG = "experiments/pre_registration_literature_audit.md"
PREREG_COMMIT = "cafa771"       # freeze 9b8a395, Amendment 1 at 3c61301, Amendments 2-5 at cafa771.
                                # The rubric this scores against -- five criteria, four ambiguity
                                # rules, the 10-paper floor -- is unchanged by all five amendments.
                                # A1 moved only the retrieval endpoint; A2-A3 postdate 5 coded rows,
                                # A4-A5 postdate all 59. Scoring refuses if the document drifts from
                                # this hash, which is what stops a rule being added after a rate.

CRITERIA = ["C-a", "C-b", "C-c", "C-d", "C-e"]
FLOOR = 10                      # frozen: non-negotiable 4. Not a tunable.
CODES = {"YES", "NO", "UNCLEAR"}


def check_frozen():
    """The rubric must be committed, and its working tree clean, BEFORE any rate is printed.

    Same two-part gate as the comparability runner: `git log -1` reports the last commit that touched
    the file and is blind to uncommitted edits, so the hash alone passes for a rubric edited after
    coding began -- which is precisely the failure a pre-registration exists to prevent.
    """
    if PREREG_COMMIT is None:
        print(f"REFUSING TO SCORE: PREREG_COMMIT is None. Commit {PREREG} (freeze + Amendment 1), "
              f"set PREREG_COMMIT to that hash, and rerun.")
        return False
    got = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20).stdout.strip()
    if not got or not got.startswith(PREREG_COMMIT[:7]):
        print(f"REFUSING TO SCORE: {PREREG} last touched at {got or 'UNTRACKED'}, but PREREG_COMMIT "
              f"is {PREREG_COMMIT}. A rubric that moved after coding is not a pre-registration.")
        return False
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                           cwd=BASE, capture_output=True, text=True, timeout=20).stdout.strip()
    if dirty:
        print(f"REFUSING TO SCORE: {PREREG} has uncommitted changes ({dirty.split()[0]}).")
        return False
    print(f"[OK] {PREREG} at {got}, working tree clean. Criteria, ambiguity rules, frame and floor "
          f"pre-registered at 9b8a395/3c61301; Amendments 2-3 postdate 5 coded rows and 4-5 postdate "
          f"all 59 (see that file's amendment timeline). Not a claim that the whole file predates "
          f"coding.")
    return True


def load(name, required=True):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        if required:
            print(f"MISSING: {p}")
        return None
    return json.load(open(p))


def validate(papers):
    """Every YES/NO needs a verbatim quotation AND a locator. Returns the list of violations."""
    bad = []
    for p in papers:
        if p.get("screen") != "include":
            continue
        cr = p.get("criteria", {})
        for k in CRITERIA:
            c = cr.get(k)
            if c is None:
                bad.append(f"{p['id']}: {k} not coded at all")
                continue
            code = c.get("code")
            if code not in CODES:
                bad.append(f"{p['id']}: {k} code {code!r} is not YES/NO/UNCLEAR")
                continue
            if code == "UNCLEAR":
                continue                       # a criterion with no quotable text IS UNCLEAR, by rule
            q, loc = (c.get("quote") or "").strip(), (c.get("locator") or "").strip()
            if len(q) < 15:
                bad.append(f"{p['id']}: {k}={code} has no verbatim quotation "
                           f"({len(q)} chars; a YES or NO must be quotable)")
            if not loc:
                bad.append(f"{p['id']}: {k}={code} has a quotation but no section/table locator")
    return bad


def verdicts(papers):
    """Per-paper primary verdict. UNCLEAR anywhere => UNCLEAR, and out of both numerator and
    denominator (rules 1-2). Returns (rows, counts)."""
    rows = []
    for p in papers:
        if p.get("screen") != "include":
            continue
        cr = {k: p["criteria"][k]["code"] for k in CRITERIA}
        unclear = [k for k in CRITERIA if cr[k] == "UNCLEAR"]
        if unclear:
            rows.append({**{"id": p["id"]}, "codes": cr, "primary": "UNCLEAR",
                         "unclear_on": unclear, "design": None})
            continue
        y = {k: cr[k] == "YES" for k in CRITERIA}
        rows.append({"id": p["id"], "codes": cr,
                     "primary": ("YES" if (y["C-a"] and y["C-b"] and y["C-c"] and y["C-e"]
                                           and not y["C-d"]) else "NO"),
                     "unclear_on": [],
                     "design": y["C-a"] and y["C-b"] and y["C-c"]})
    return rows


def main():
    if not check_frozen():
        return 1
    raw = load("candidates_raw.json")
    coding = load("coding.json")
    if raw is None or coding is None:
        print("Nothing to score. Run retrieve_literature_audit.py, then code into coding.json.")
        return 1

    papers = coding.get("papers", [])
    included = [p for p in papers if p.get("screen") == "include"]

    # ---- the funnel, PRISMA-style. Every stage, and the excluded lists ship (non-negotiable 6).
    cand = raw.get("candidates", [])
    funnel = {
        "retrieved": len(cand),
        "deduplicated": len({(c.get("arxiv_id") or c.get("title", "")).lower() for c in cand}),
        "screened": len(papers),
        "full_text_sought": sum(1 for p in papers if p.get("full_text_sought")),
        "full_text_obtained": sum(1 for p in papers if p.get("full_text_obtained")),
        "included": len(included),
        "excluded": len(papers) - len(included),
    }
    print("\n=== PRISMA funnel ===")
    for k, v in funnel.items():
        print(f"  {k:22s} {v}")
    ex = [p for p in papers if p.get("screen") == "exclude"]
    if ex:
        print(f"\n  excluded ({len(ex)}), each with its reason:")
        for p in ex:
            print(f"    {p['id']:28s} {p.get('screen_reason', '** NO REASON GIVEN **')}")
        noreason = [p["id"] for p in ex if not p.get("screen_reason")]
        if noreason:
            print(f"\nREFUSING TO SCORE: exclusions without a reason: {noreason}")
            return 1

    # ---- gate 1: every YES/NO quotable and located
    bad = validate(papers)
    if bad:
        print(f"\nREFUSING TO PRINT A RATE -- {len(bad)} coding rows are not auditable "
              f"(non-negotiable 3):")
        for b in bad:
            print(f"  {b}")
        return 1
    print("\n[OK] every YES/NO on every included paper ships a quotation and a locator")

    rows = verdicts(papers)
    scorable = [r for r in rows if r["primary"] != "UNCLEAR"]
    unclear = [r for r in rows if r["primary"] == "UNCLEAR"]

    # ---- gate 2: the floor. Outcome (iii) is printed as a result, not as a failure.
    if len(scorable) < FLOOR:
        print(f"\n=== OUTCOME (iii): THE FRAME IS TOO SMALL FOR A RATE ===")
        print(f"  {len(scorable)} papers are included and fully coded; the frozen floor is {FLOOR}.")
        print(f"  ({len(unclear)} further included papers are UNCLEAR on at least one criterion and")
        print(f"   are out of both numerator and denominator by rules 1-2.)")
        print(f"  NO RATE IS REPORTED. The pre-registration's outcome (iii) applies: the instance list")
        print(f"  ships, main.tex:1658's decline stands UNCHANGED, and the paper says the frame was too")
        print(f"  small to support a rate. A rate over {len(scorable)} papers would be noise dressed")
        print(f"  as a finding, so it is not printed here and must not be computed by hand.")
        json.dump({"prereg": PREREG, "prereg_commit": PREREG_COMMIT, "funnel": funnel,
                   "outcome": "iii-frame-too-small", "floor": FLOOR,
                   "n_scorable": len(scorable), "n_unclear": len(unclear),
                   "rate_reported": False,
                   "instances": [{"id": r["id"], "codes": r["codes"]} for r in rows]},
                  open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        print(f"\nWrote {os.path.join(OUT, 'summary.json')} (outcome iii; no rate)")
        return 0

    # ---- the primary and its secondaries, which are never merged into it
    n = len(scorable)
    prim = sum(1 for r in scorable if r["primary"] == "YES")
    design = sum(1 for r in scorable if r["design"])
    cd_yes = sum(1 for r in scorable if r["codes"]["C-d"] == "YES")
    # Two different counts, and the label used to conflate them. The pre-registered secondary is
    # C-e = NO *among DESIGN papers*: those have the outcome-gated design and decline to make the
    # attribution claim, which is what makes them evidence against the sharp reading. A C-e = NO row
    # that is not DESIGN is not that -- it is a paper with no design and no claim, which says nothing
    # either way. Both are printed because the difference is the whole informational content.
    ce_no_design = sum(1 for r in scorable if r["design"] and r["codes"]["C-e"] == "NO")
    ce_no_all = sum(1 for r in scorable if r["codes"]["C-e"] == "NO")

    print(f"\n=== PRIMARY: OUTCOME-GATED ATTRIBUTION (C-a & C-b & C-c & C-e & not C-d) ===")
    print(f"  {prim}/{n} = {100.0 * prim / n:.1f}%  of included, fully-coded papers")
    print(f"  ({len(unclear)} included papers are UNCLEAR and are out of BOTH numerator and "
          f"denominator, rules 1-2)")
    print(f"\n  SECONDARIES, reported alongside and never merged into the primary:")
    print(f"    DESIGN rate (C-a & C-b & C-c)          {design}/{n} = {100.0 * design / n:.1f}%")
    print(f"    C-d PRESENT (an identifying contrast)  {cd_yes}/{n} = {100.0 * cd_yes / n:.1f}%")
    print(f"      ^ read the primary against this: {n - cd_yes} of {n} papers lack the contrast, so that")
    print(f"        absence, not the attribution claim, is what most of the primary is made of.")
    print(f"    C-e = NO among DESIGN papers            {ce_no_design}/{design}")
    print(f"      ^ the pre-registered secondary: these HAVE the design and make no attributional")
    print(f"        claim, so they are EVIDENCE AGAINST the sharp reading and are reported as such.")
    print(f"    C-e = NO anywhere in the frame           {ce_no_all}/{n}")
    print(f"      ^ printed only to keep the line above from being read as this one. A row with no")
    print(f"        design and no claim is not evidence in either direction.")

    bykey = {p["id"]: p for p in papers}
    # ---- Amendment 4a's exposure, EMITTED and not transcribed (non-negotiable 13). A record whose
    # amendment_4a_dependency says its primary is NO under the coarser reading drops out of the
    # counterfactual count. The rule can move a row either way, so this is a recount, not a subtraction.
    dep4a = [r for r in scorable
             if bykey[r["id"]].get("amendment_4a_dependency", {}).get("primary_under_coarser_reading") == "NO"]
    prim_no4a = sum(1 for r in scorable
                    if r["primary"] == "YES"
                    and bykey[r["id"]].get("amendment_4a_dependency", {}).get("primary_under_coarser_reading") != "NO")
    print(f"\n  AMENDMENT 4a COUNTERFACTUAL (non-negotiable 13): primary = {prim}/{n} with the "
          f"distinctness rule, {prim_no4a}/{n} without it")
    for r in dep4a:
        d = bykey[r["id"]].get("amendment_4a_dependency", {})
        print(f"    {r['id']} turns on it via {d.get('criterion')}; primary {r['primary']} with, "
              f"{d.get('primary_under_coarser_reading')} without")
    if not dep4a:
        print(f"    no coded record declares a 4a dependency, so the two counts coincide")

    # ---- c_d_route stratification: a reporting field, a floor, and it enters no rate (Amendment 5,
    # non-negotiable 14). Printed over the C-d = YES rows only; 'absent' must equal the C-d = NO count
    # or a record's route contradicts its own code, which is worth failing loudly for.
    routes = {}
    for r in scorable:
        rt = bykey[r["id"]].get("c_d_route") or "unlabelled"
        routes.setdefault(rt, []).append(r["id"])
    n_absent = len(routes.get("absent", []))
    if n_absent != n - cd_yes:
        print(f"REFUSING TO SCORE: {n_absent} records carry c_d_route 'absent' but {n - cd_yes} code "
              f"C-d = NO. A route contradicts a code.")
        return 1
    print(f"\n  C-d ROUTE STRATIFICATION over the {cd_yes} rows that run a contrast (Amendment 5's "
          f"precedence sweep > contrast > analytic):")
    for rt in ("hyperparameter-sweep", "component-contrast", "analytic", "unlabelled"):
        if routes.get(rt):
            print(f"    {rt:22s} {len(routes[rt]):3d}/{cd_yes}")
    print(f"      ^ a FLOOR, not a census: the field records only routes a record QUOTES, so a route "
          f"present in a paper\n        but never located is not counted. Enters no rate; decides nothing.")

    print(f"\n=== per-paper table ({len(rows)} included) ===")
    print(f"  {'paper':30s} " + " ".join(f"{k:>7s}" for k in CRITERIA) + "  primary   blind")
    for r in rows:
        b = "no" if bykey[r["id"]].get("blind") is False else "yes"
        print(f"  {r['id']:30s} " + " ".join(f"{r['codes'][k]:>7s}" for k in CRITERIA)
              + f"  {r['primary']:8s} {b}")

    print("\n--- LaTeX rows (paper, venue/year, C-a..C-e, primary) ---")
    for r in rows:
        p = bykey[r["id"]]
        cells = " & ".join({"YES": "\\checkmark", "NO": "--",
                            "UNCLEAR": "?"}[r["codes"][k]] for k in CRITERIA)
        vy = f"{p.get('venue', '?')} {p.get('year', '?')}"
        star = "$^{\\dagger}$" if p.get("blind") is False else ""
        print(f"\\citet{{{p.get('bibkey', r['id'])}}}{star} & {vy} & {cells} & "
              f"{ {'YES': 'yes', 'NO': 'no', 'UNCLEAR': 'unclear'}[r['primary']] } \\\\")
    # Emitted only if some row actually carries the marker. A footnote explaining a symbol that appears
    # nowhere in its own table is the kind of thing that survives every check and confuses every reader.
    if any(bykey[r["id"]].get("blind") is False for r in rows):
        print("\\multicolumn{8}{l}{\\footnotesize $\\dagger$ known to us before the rubric was frozen; "
              "coding is not blind.} \\\\")

    json.dump({"prereg": PREREG, "prereg_commit": PREREG_COMMIT, "funnel": funnel,
               # "rate-reported" distinguishes this from outcome (iii) and NOTHING else. The
               # pre-registration deliberately set no numeric threshold between outcomes (i) and (ii),
               # so this script does not decide between them: inventing a cutoff now, with the counts
               # visible, is exactly the move the rubric's amendment timeline forbids. The prose
               # reading must quote primary, DESIGN and C-d-present together.
               "outcome": "rate-reported", "outcome_i_vs_ii_thresholded": False,
               "floor": FLOOR, "rate_reported": True,
               "primary": {"numerator": prim, "denominator": n, "pct": 100.0 * prim / n},
               "secondaries": {"design": [design, n], "c_d_present": [cd_yes, n],
                               "c_e_no_among_design": [ce_no_design, design],
                               "c_e_no_anywhere": [ce_no_all, n]},
               # Reporting-only, and the LaTeX table must say so where it prints them.
               "amendment_4a": {"primary_with": prim, "primary_without": prim_no4a,
                                "rows_that_turn_on_it": [r["id"] for r in dep4a]},
               "c_d_route_floor": {rt: len(ids) for rt, ids in sorted(routes.items())},
               "n_unclear_excluded": len(unclear),
               "papers": [{"id": r["id"], "codes": r["codes"], "primary": r["primary"],
                           "unclear_on": r["unclear_on"],
                           "blind": bykey[r["id"]].get("blind", True)} for r in rows]},
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(f"\nWrote {os.path.join(OUT, 'summary.json')}")
    print("Every number in the appendix table comes from this file. None is transcribed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
