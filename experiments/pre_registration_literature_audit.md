# Pre-registration: how often is a composed FL defense validated by an outcome-gated design?

**Round 57.** Frozen before any paper outside the three named seeds is retrieved or read. Nothing in
this document is written with knowledge of a result, with one disclosed exception recorded in
"What we already knew before the freeze" below.

## The claim this audit exists to test, and the sentence it will change

`paper/main.tex` currently declines a claim about the literature, on purpose:

> **We do not claim this is how the FL literature generally proceeds**, and we can name one external
> instance rather than a practice: \citet{fenaux2025hammer} pair a defense effective against
> large-deviation updates with one effective against small-deviation updates, and validate the pairing
> by the combined defense's attack success rate [...] The claim we do make is structural rather than
> sociological, and its clearest instance is *our own* screen.

That decline was the right call with n=1 external instance. This audit measures the quantity instead of
declining it. **The structural claim does not depend on the outcome** and is not being put at risk: the
identification result stands on its own screen either way. What is at stake is only whether the paper
can say how much published evidence the result applies to.

## The measured quantity, and why it is not the obvious one

The naive quantity -- "how many papers omit the within-defense intervention we propose" -- is
**near-tautological**. Almost no one runs a contrast this paper introduces, so a rate near 100% would
be uninformative and would deserve the dismissal it would get.

The quantity that is actually unknown, and the one Proposition `prop:identification_full` actually
constrains, is how many papers **make a mechanism-attribution claim on the strength of an
outcome-gated design.** A paper that reports "our pipeline reduces backdoor accuracy to 2%" and claims
nothing about which component is responsible is inferring nothing invalid, and must not be counted.

So the coding is five criteria, and the headline is their conjunction.

## Coding criteria, frozen. Each requires a verbatim quotation and a locator.

For each included paper, each criterion is coded YES / NO / UNCLEAR, and **every YES and every NO must
carry a verbatim quotation plus a section or table locator.** A criterion with no quotable text is
UNCLEAR.

- **C-a — joint application.** Two or more distinct defense mechanisms are applied together to the
  same training round (sequential pipeline, filter-then-aggregate, ensemble, or clip-then-noise).
  *Not* C-a: a menu of alternative defenses compared against each other; a single mechanism with
  hyperparameters; a defense compared against baselines.
- **C-b — the pairing is selected by constituent properties.** The combination is motivated or chosen
  because its constituents are individually effective, or cover complementary threats or complementary
  regions of the attack space. Quotation must show the *reason for pairing*, not merely that a pairing
  exists.
- **C-c — validated by the composed system's own outcome.** The evidence offered for the combination is
  the assembled pipeline's attack success rate / backdoor accuracy / robust accuracy.
- **C-d — an identifying contrast is present.** The paper varies the **upstream** stage while holding
  the **downstream** mechanism and the attack fixed, and reads the downstream mechanism's suppression
  off that variation. **A component-ablation or leave-one-out table is NOT C-d** unless the downstream
  mechanism is held fixed and the upstream stage is the only thing varied; dropping a component changes
  the system, not the upstream input to a fixed component. This distinction is the whole content of
  Proposition `prop:identification_full` and it is written down here, before coding, precisely so it
  cannot be decided per paper.
- **C-e — a mechanism-attribution claim is made.** The paper asserts *why* the combination works, or
  what a particular constituent contributes to the combined result, rather than only reporting the
  combined number. Quotation must contain the attributional claim.

**Primary quantity: OUTCOME-GATED ATTRIBUTION := C-a ∧ C-b ∧ C-c ∧ C-e ∧ ¬C-d**, as a fraction of
included papers.

**Secondary quantities, reported alongside and never merged into the primary:**
- **DESIGN rate := C-a ∧ C-b ∧ C-c** (the design, independent of whether an attributional claim rides
  on it);
- **C-d presence rate**, reported on its own, so a reader can see how much of the primary is carried by
  the predictable absence of a contrast this paper introduces;
- the count of papers where C-e is NO -- papers with the design and no attributional claim, which are
  **evidence against** the sharp reading and are reported as such.

## Ambiguity resolves against the finding, by rule

1. Any criterion coded UNCLEAR makes the paper's primary verdict UNCLEAR.
2. **UNCLEAR papers are excluded from the primary's numerator and denominator**, and their count is
   reported next to the rate. This can only shrink the measured prevalence relative to a coder who
   guesses, so the rule cannot inflate the finding.
3. If a quotation supports two readings, the reading that makes the paper **not** outcome-gated is
   taken.
4. C-d is given the **benefit of the doubt**: any experiment plausibly holding a downstream mechanism
   fixed while varying an upstream stage is coded C-d = YES even if imperfect, which again shrinks the
   primary.

## The frame, stated as a procedure a reviewer can re-run

`WebSearch` is not available in this environment; `WebFetch` is. The frame is therefore built from
**public APIs whose queries are reproducible verbatim**, which is a better frame than a reading list
regardless.

- **F1 — the on-disk seed.** Every entry in `paper/references.bib` that proposes or evaluates an FL
  poisoning/backdoor defense. Enumerable from the repository, so a reviewer starts where we started.
- **F2 — the query frame.** DBLP (`https://dblp.org/search/publ/api?q=<query>&h=100&format=json`) and,
  where DBLP's title-only matching is too coarse, Semantic Scholar
  (`https://api.semanticscholar.org/graph/v1/paper/search`), over this **fixed** query list:
  1. `federated learning backdoor defense`
  2. `federated learning poisoning defense`
  3. `byzantine robust federated aggregation`
  4. `federated learning backdoor mitigation`
  5. `robust federated learning aggregation defense`
  Restricted to: **years 2019--2026**, and venues **NeurIPS, ICML, ICLR, AISTATS, USENIX Security,
  NDSS, IEEE S&P, ACM CCS, TMLR, arXiv preprints**. The query strings, the year window and the venue
  list are frozen here; **adding a query after seeing results is forbidden.**
- **F3 — one-hop snowball.** Papers appearing in the defense-comparison tables or related-work of three
  named seeds: `nguyen2022flame`, `rieger2022deepsight`, `fenaux2025hammer`. One hop only; a second hop
  is not taken.

### Screening, and the funnel that gets reported

Every stage's count is reported, PRISMA-style, and the excluded lists ship with their reasons:

    retrieved -> deduplicated -> title/abstract screened -> full text sought -> full text obtained -> included -> coded

**Include at screening iff** the paper proposes or evaluates a defense **pipeline** in the C-a sense.
**Exclude** single-mechanism defenses, attack-only papers, surveys, position papers, and work outside
FL. Every exclusion carries its reason string.

**The full-text restriction, and the bias it introduces.** Coding requires quotable full text, so a
paper whose full text cannot be fetched is recorded at the "full text sought" stage and **excluded**,
with the reason. This biases the frame toward arXiv and open-access venues (USENIX, NDSS, OpenReview)
and against paywalled ones (notably ACM CCS). **That bias is stated in the paper, not buried here**, and
the count of full-text failures is reported so a reader can bound it.

## Outcomes, all three publishable, which is what makes this safe to pre-register

- **(i) The measured primary rate is substantial.** `:1658`'s decline is replaced by a measured,
  frame-scoped statement, and the paper says what the frame is in the same breath.
- **(ii) The measured primary rate is low.** **The decline stays, and is vindicated with evidence
  instead of asserted.** This is a strictly better paper than the status quo and it is not a failed
  round: it converts a hedge into a measurement.
- **(iii) Fewer than 10 papers survive to coding.** **No rate is reported at all.** The instance list
  ships, `:1658`'s decline stands unchanged, and the paper says the frame was too small to support a
  rate. A rate over 6 papers would be noise dressed as a finding.

The floor is **10 included-and-coded papers** and it is frozen here.

## What we already knew before the freeze, disclosed

Three papers were known to us before this document existed, so their coding is **not blind** and is
labelled as such in the shipped table:

- `fenaux2025hammer` -- already cited at `main.tex:1658` and `:1664` as the one external instance.
- `nguyen2022flame` and `rieger2022deepsight` -- already in the bibliography. **`main.tex:1654`
  currently asserts FLAME is "proposed and evaluated as a *single* stage", which we believe to be
  wrong** (FLAME combines clustering, adaptive clipping and adaptive noising). Verifying that against
  the paper and correcting the sentence is a defect fix that happens **regardless of this audit's
  outcome**, and it is recorded here so that finding FLAME to be a pipeline cannot later be presented
  as a discovery of the audit.

No other paper's content is known to us at freeze time.

## What this audit does NOT establish, recorded before the numbers exist

- **It is not a claim about "the FL literature".** It is a rate within a stated, reproducible frame,
  with a stated open-access bias and a stated year window. The paper must say "within this frame" every
  time it says the number.
- **A high rate is not an accusation of error.** An outcome-gated design is the natural design; the
  paper's own screen is its clearest instance, and that sentence stays. The finding would be that a
  structural limit applies widely, not that authors were careless.
- **Coding is by one coder and no inter-rater statistic is computed.** The mitigation is that this
  rubric is frozen at a commit hash before coding and **every row ships its quotation and locator**, so
  a reader re-adjudicates rather than trusts. Claiming a reliability statistic we cannot compute would
  be worse than admitting we have none.
- **A rate is not a causal claim about the field's conclusions.** It does not establish that any
  specific published conclusion is wrong. Proposition `prop:identification_full` says the design cannot
  identify the mechanism; whether each paper's conclusion happens to be true is not measured and is not
  claimed either way.
- **The audit does not touch the paper's own results.** No number in any table moves.

## Non-negotiables

1. **This document is committed before the first retrieval beyond the three disclosed seeds.** If it is
   not, the audit is post-hoc and must be reported as post-hoc.
2. The query list, year window, venue list, five criteria, decision rules and the 10-paper floor are
   **not edited after the first retrieval.** Any change is an amendment, dated, disclosed, and stating
   what was known when it was written -- the convention
   `experiments/pre_registration_comparability.md` established across its three amendments.
3. **Every coded row ships its quotations.** `experiments/analyze_literature_audit.py` refuses to print
   a rate if any included paper is missing a quotation for any criterion it codes YES or NO.
4. The analyzer refuses to print a rate if fewer than 10 papers are included and coded.
5. Every number that reaches LaTeX is emitted by the analyzer. **Nothing is transcribed by hand**, which
   is the standing rule for every table in this paper.
6. The excluded lists, with reasons, ship in the appendix table. An audit that reports only its
   inclusions is not auditable.
7. **No paper is added to the frame after coding begins**, and no paper is dropped from the frame
   because of how it coded.

---

## Amendment 1 (Round 57, 2026-09-10): the retrieval endpoint moved, and nothing else did

**Written before any paper beyond the three disclosed seeds has been coded, and before any abstract has
been read.** What is known at the time of writing is stated exhaustively below.

### What forced the change: HTTP status codes, not results

F2 above names DBLP and Semantic Scholar. **Both are unreachable from this environment.** Measured:

| surface | result |
|---|---|
| `dblp.org/search/publ/api?...&format=json` | serves an Anubis anti-bot HTML challenge, not JSON |
| `api.semanticscholar.org/graph/v1/paper/search` | HTTP **429** on 5 backoff attempts over ~75 s; also 429 via `WebFetch` |
| `export.arxiv.org/api/query` (the arXiv **API**) | **301** on http, **429** on https |
| `api.crossref.org` | 200, but `total-results` 2,808,257 on the frame's queries, and poor ML-venue coverage |
| ACM Digital Library proceedings pages | HTTP **403** |
| `arxiv.org/search/` (the **HTML UI**, not the API) | **200**, paginable, 50 titles + arXiv IDs per page |
| `arxiv.org/pdf/<id>` | **200**; FLAME's 872 KB PDF yields 110 KB of `pdftotext` text |
| `usenix.org/.../technical-sessions`, `ndss-symposium.org/.../accepted-papers`, `ieee-security.org/TC/SP*/program-papers.html`, `proceedings.mlr.press/v*/` | **200**, complete per-venue-year title listings |

**No retrieved bibliographic result caused this amendment.** The change is caused entirely by the status
codes in the right-hand column, and it would read identically had every reachable surface returned an
empty set.

### The change, stated as a substitution

**F2 is re-pointed and not re-specified.** The five query strings, the 2019--2026 year window, the venue
list, the five criteria C-a..C-e, the primary and secondary quantities, the four ambiguity rules and the
10-paper floor are **unchanged, verbatim.** Only the endpoint that answers a query changes:

- **F2 (amended).** The same five frozen query strings are issued against the **arXiv HTML search
  interface**, `https://arxiv.org/search/?searchtype=all&query=<query>&start=<n>&size=50`, paginated to
  exhaustion. Titles and arXiv IDs are extracted mechanically. Venue attribution, where arXiv does not
  carry it, is resolved against the reachable venue indexes in the table above.
- **F2b (new, and additive only).** Because keyword search over arXiv under-covers the security venues
  where composed defenses actually publish, the **complete title listings** of USENIX Security, NDSS,
  IEEE S&P (2019--2026) and PMLR (ICML, AISTATS) are additionally screened by a frozen title regex:
  `federated` AND (`backdoor` OR `poison` OR `byzantine` OR `robust`), case-insensitive. This is an
  **exhaustive population per venue-year with a real denominator**, which is a stronger frame than
  keyword search, not a weaker one. Its recall limit is stated: a paper whose *title* carries none of
  those tokens is missed by F2b, and F2 and F1 are its only routes into the frame.
- **Full text** is obtained as `arxiv.org/pdf/<id>` piped through `pdftotext`. A paper with no reachable
  full text is recorded at the "full text sought" stage and excluded with its reason, exactly as the
  unamended document already required.

### The bias this realizes, and why it is not a new one

The unamended document already stated the bias: *"This biases the frame toward arXiv and open-access
venues (USENIX, NDSS, OpenReview) and against paywalled ones (notably ACM CCS)."* Amendment 1 does not
introduce that bias; it **makes it the operative retrieval mechanism**, which is a difference of degree
worth recording. Two consequences are frozen here:

1. **ACM CCS is a coverage gap, not a sample.** Its proceedings pages are 403 and its papers enter the
   frame only if they are also on arXiv. **The paper states this in the same sentence as the rate.**
2. **The frame is arXiv-reachable published FL poisoning-defense work, and the paper says so every time
   it says the number.** It is not "the FL literature", which the unamended document already forbade
   claiming.

### What had been retrieved when this was written, disclosed exhaustively

Reachability probing necessarily returned some bibliographic content. All of it, completely:

- **Query 1 (`federated learning backdoor defense`) reports 174 arXiv results**, and the **first five
  titles** of page 1 were displayed. No abstract was opened, no PDF beyond `nguyen2022flame` was
  fetched, and no criterion was coded for any of them.
- The USENIX Security 2022 listing was extracted (256 titles) and filtered on `federated`, returning
  **three** titles, one of which is FLAME -- already a disclosed seed.
- Eight PMLR v202 titles containing `federated` were displayed. None concerns a defense pipeline on its
  title alone, and none has been coded.
- Three CrossRef titles were displayed during precision testing of a surface that is **not** adopted.

**None of this content selected a query, a criterion, a decision rule or the floor** -- all of which
predate it in a committed document (`9b8a395`) -- and none of it has been coded. It is disclosed because
non-negotiable 2 requires the amendment to state what was known when it was written, and a count of
results for one query is something known.

### Non-negotiables, extended

8. **The amended F2/F2b endpoints are themselves frozen at this commit.** The arXiv query URL template,
   the pagination-to-exhaustion rule and the F2b title regex are not edited after the first retrieval
   under this amendment. A surface that later becomes reachable (DBLP, Semantic Scholar, CCS) **is not
   added to this audit**; it is future work, because adding a surface after seeing a rate is how a frame
   gets tuned to its result.
9. **The retrieval is executed by a script, not by hand**, and the script ships: the query strings it
   issues, the pages it walked and the raw candidate list are written to `results/literature_audit/`
   before any screening decision. A frame that cannot be diffed against its own output is not
   reproducible.
10. **F2b's exhaustive denominators are reported in the funnel** -- how many titles were listed per
    venue-year, and how many the regex retained. Reporting only the retained count would present an
    exhaustive screen as if it were a search.

---

## Amendment 2 (Round 57, 2026-09-10): the blind-coding disclosure was too narrow by one paper

**Written after five papers have been coded, which is itself the disclosure this amendment exists to
make.** Nothing about the frame, the queries, the window, the five criteria, the four decision rules or
the 10-paper floor changes; non-negotiable 2 is not touched. What changes is the count on line 139
above, from three papers to **four**, and the direction of that change is against our own interest: a
row marked not-blind is a row a reader discounts.

**What went wrong.** Line 139 says "Three papers were known to us before this document existed," and
names the three retrieval seeds. That conflated *seed of the frame* with *known to the coder*. The
seeds are not the only papers we had read before the freeze -- **every paper already in
`paper/references.bib` was**, and `main.tex:1654` names six defenses by hand. One of those six survives
into the frame: **`cao2021fltrust` (arXiv 2012.13995)**, which the screen included with the explicit
reason that `:1654` asserts it is a single stage and coding it against the full text decides that. A
paper admitted in order to test a sentence we wrote is the least blind row in the audit, and it was
about to ship marked blind.

**The rule, stated so it is mechanical rather than remembered.** A paper is **not blind** iff its title
matches an entry in `paper/references.bib` *as that file stood at the pre-registration commit*
(`git show 9b8a395:paper/references.bib`, 37 entries), or it is one of the three named seeds. Matching
the 59 included papers against that frozen bibliography returns exactly four:

| arXiv id | bibkey | why not blind |
|---|---|---|
| 2012.13995 | `cao2021fltrust` | cited at `main.tex:1654`; **added by this amendment** |
| 2101.02281 | `nguyen2022flame` | named seed, already disclosed |
| 2201.00763 | `rieger2022deepsight` | named seed, already disclosed |
| 2509.08089 | `fenaux2025hammer` | named seed, already disclosed |

The other 55 included papers match no entry in the frozen bibliography and are coded blind. The set is
hard-coded in `experiments/merge_coding_literature_audit.py` (`NOT_BLIND`) rather than set per record,
so a typo in one file cannot lose a disclosure, and it is now derived rather than recalled.

**What was known when this was written.** Five papers were coded: the three seeds and 2012.13995, all
four of which code C-d = YES, plus 1909.05125, which codes C-b = NO and C-d = NO. No rate has been
computed, `analyze_literature_audit.py` has not been run on more than a merge check, and 54 of the 59
included papers were unread at the time of writing. The amendment therefore cannot have been chosen to
move a number, because no number exists yet -- but it does move one in a knowable direction, and it is
worth saying which: it moves a fourth row (1 of 59) out of the blind column, and the four not-blind
rows are precisely the four whose codes bear most directly on `main.tex:1654` and `:1658`.

### Non-negotiables, extended

11. **The not-blind set is derived, not asserted.** It is the title-match of the included set against
    the pre-registration commit's `references.bib` plus the three seeds, and the shipped table marks
    every such row. A paper does not become blind by being re-read later.

---

## Amendment 3 (Round 57, 2026-09-10): what makes a component a *defense* mechanism under C-a

**Written after five papers have been coded and before any paper in the class this amendment governs has
been coded.** The five criteria are **unchanged, verbatim**; this amendment adds no criterion and
removes none. It records how the word already in C-a -- "Two or more distinct **defense** mechanisms" --
is applied to a component whose purpose is *not* resisting poisoning, because the frame contains **22
included papers that pair a privacy, cryptographic or communication component with a robustness one**
and deciding them one at a time invites deciding them inconsistently.

**The rule.** A component counts toward C-a's "two or more" iff **the paper's own stated purpose for
including it is to resist faulty, malicious or poisoned updates.** Purpose is read off the paper and
quoted, never inferred from the operation's name.

**Why purpose and not operation.** The same operation appears on both sides of this line, so an
operation-based rule is not available:

- Gaussian noise **is** a defense component in `nguyen2022flame`, whose §4.3 adds it "to eliminate the
  remaining backdoors after applying clustering and clipping". It is **not** one in a
  differentially-private FL paper that adds the identical noise to bound a privacy loss and then bolts a
  robust aggregator on for a separate reason.
- Clustering **is** a defense component in FLAME (dynamic clustering "to remove poisoned models with
  large cosine distances"). It is **not** one in `2110.02940`, where clients are clustered so that
  secure aggregation can run over nonlinear operations it otherwise cannot.
- Compression / sparsification is essentially never a defense component: it exists to cut communication.
  `2104.06685` is explicit that compression *hurts* robustness, which is the opposite of a defense role.

**What the rule does NOT do.** It does not exclude a paper from the frame -- every one of the 22 stays
in the denominator, as non-negotiable 7 requires -- and it does not decide the whole class, because a
paper can pair a privacy component with **two** defense components and still code C-a = YES on those two
(`2505.01454`, `2512.11760`, `2601.06466` and `2603.04422` each carry two or more defense families
before their privacy family is counted at all).

**Direction of effect, stated before the class is coded.** The rule can only move rows from C-a = YES to
C-a = NO. It therefore **lowers** the primary rate and the DESIGN rate while leaving the denominator
untouched. That is against the outcome that would flatter this paper, which is the only direction an
under-specified rubric is allowed to be resolved in.

**What was known when this was written.** Five papers coded (`2012.13995`, `2101.02281`, `2201.00763`,
`2509.08089`, `1909.05125`); four code C-d = YES and one codes C-b = NO. No rate has been computed. Of
the 22 papers in the affected class, only FLAME has been coded, and it codes C-a = YES on
clustering + clipping + noising independently of this rule.

### Non-negotiables, extended

12. **Every C-a code on a paper in the 22-row class ships the purpose quotation**, not only the
    joint-application quotation -- the sentence that says what the non-robustness component is *for*.
    Without it a reader cannot re-adjudicate the code, which is the audit's only reliability mitigation.

---

## Amendment 4 (Round 57, 2026-09-10): three adjudication rules the frozen criteria do not settle

**Written after all 59 rows were coded, and recording rules that were articulated at rows 45, 47 and 53
rather than in advance. That is the opposite of pre-registration and is stated plainly here rather than
folded into the criteria**: a reader who rejects any of the three re-adjudicates the rows named below,
and this amendment exists so that the rows are named.

**Why they were not foreseen.** C-a's frozen text asks for "two or more **distinct** defense mechanisms",
and Amendment 3 says what makes a mechanism a *defense*. Neither says what makes two mechanisms
*distinct*, and neither says what to do when one purpose sentence covers several components at once. Both
gaps are invisible until a paper sits on them.

### 4a. The distinctness (granularity) rule -- articulated at row 45, `2602.16480` (SRFed)

**A component is DISTINCT when it contributes an independent discriminative signal or an independent
decision about an update, and not when it is a stage in computing one signal.**

Worked in both directions, from rows already coded:

- SRFed's layer-wise projection -> K-Means -> cosine-similarity cluster ranking -> mean of survivors is
  **one** mechanism: a single decision function computed in stages, emitting one keep-mask per client.
- SecureDyn-FL's (`2601.06466`) cluster-relative Mahalanobis score, its separate across-round trajectory
  score and its three-way accept / down-weight / reject decision over three independently maintained
  thresholds are **three**.
- STAR-FL's (`2608.14861`) spatial and temporal filters are **two**: different statistics over different
  axes (across clients; across rounds), each flagging its own set, combined conjunctively.
- Secure-CHG's (`2606.31066`) EMA reputation stage is **not distinct** from the CHG contribution signal it
  smooths, even though its stated purpose is defensive.

**Direction of effect, and the exact exposure.** Unlike Amendment 3 this rule can move a row either way,
so its exposure is stated per row rather than as a direction. **One of the six primaries depends on it:**
`2601.06466`, whose C-b -- and therefore its primary -- turns on reading its auditing stack at operation
granularity. Under the coarser reading the primary count is **five, not six**. SRFed's own C-a depends on
the rule and its primary does not (C-b = NO under either reading, C-d = YES independently). No other row's
primary turns on it.

### 4b. The distributive test for a joint purpose clause -- articulated at row 47, `2603.04422`

When a paper attributes a defense purpose to a **conjunction** of components and elsewhere **distributes**
the purposes among them, the distributive sentence governs and the joint clause is not a purpose statement
for either component alone. Direction of effect: this can only **lower** C-a. Live consequence in the
frame: none on the primary count, because `2603.04422`'s primary is NO under every reading (C-d = YES on
its Table 7 independently of C-a).

### 4c. The ordering of 4a and Amendment 3 -- articulated at row 53, `2606.31066`

**Distinctness is applied FIRST and purpose SECOND.** Amendment 3 asks whether a distinct component
counts; it presupposes that the component is distinct. Under the other order Secure-CHG's EMA reputation
stage would count toward C-a on its stated purpose ("mitigating the impact of intermittent attacks") while
contributing no independent signal, and the same would follow for any smoother, buffer or normalizer whose
paper describes it defensively. The ordering changes no row's primary in the frame.

**What was known when this was written.** All 59 rows are coded and the codes are in hand -- which
Amendments 2 and 3 could not say. Six rows code primary = YES. No rate has been emitted, because
`analyze_literature_audit.py` refuses to score while this file has uncommitted changes. Writing a rule
with the codes visible is the position an audit should least like to be in, which is why 4a's exposure is
quantified above (six primaries, or five without it) instead of being described as small.

### Non-negotiables, extended

13. **A rule articulated mid-coding names the row that forced it and every row it could move**, and
    App. G reports the primary count **both with and without it** wherever the count differs. A rule
    whose exposure is not quantified is not disclosed, only mentioned.

---

## Amendment 5 (Round 57, 2026-09-10): how `c_d_route` is recorded (a reporting field; no rate moves)

Every coded record carries a `c_d_route` field. **It is not a criterion, it enters no rate, and it decides
nothing**: C-d's code is YES or NO on the frozen text alone. The field exists so App. G can report *what
kind* of identifying contrast the papers that run one actually run. It was added during coding without a
written rule, and by row 58 two rows satisfying more than one route had been labelled inconsistently
(`2608.08574` and `2609.03064`), which is what forced the rule.

**The three routes.** `hyperparameter-sweep`: a graded dose of an **upstream** stage's own parameter, with
an adversarial outcome read at each setting and the downstream mechanism and the attack held fixed.
`component-contrast`: an upstream stage removed, substituted or cumulatively built up across discrete
arms, downstream fixed. `analytic`: the variation is characterized by proof rather than run.

**Precedence when a row satisfies more than one: sweep > contrast > analytic.** The other routes are named
in the record's C-d locator. The precedence is fixed rather than deferred to which finding the paper calls
central, so the stratification does not depend on an author's emphasis; and it ranks the graded dose
highest because that is the closest external analogue to this paper's own screen.

**Two clauses in the sweep definition that do real work, with the rows they exclude:**

- *The dose must be an upstream **defense** parameter.* A sweep of the **attack's** strength is not an
  upstream dose: `2201.00763`'s poisoned-data-rate figure and `2603.04422`'s malicious-fraction figure are
  component contrasts crossed with attack strength, and both keep `component-contrast`.
- *An adversarial outcome must be read at **each** setting.* `2511.09294`'s server-dataset-size sweep and
  `2603.04422`'s distillation-temperature sweep report no per-setting outcome under attack and do not
  qualify.

**Applied backward on 2026-09-10 across all 46 C-d = YES rows.** Sixteen name more than one route; four
moved from `component-contrast` to `hyperparameter-sweep`: `2409.01435` (LASA), `2502.00587` (RKD),
`2505.10297` (FeRA), `2604.03862` (SecureAFL). The stratification moves from 25 / 17 / 4 to **21 / 21 / 4**
(contrast / sweep / analytic). **No rate moves**: primary, DESIGN, C-d presence and C-e-among-DESIGN are
all unchanged, and the relabelled rows' C-d codes and locators' substance are unchanged.

**The lower-bound disclosure, which matters more than the precedence rule.** The field records only routes
that the record **quotes**. A route present in a paper but never located and quoted is not recorded, so
the sweep column is a **floor, not a census**. LASA is the proof: its appendix sweep of the upstream
sparsification level -- with accuracy under three attacks at every rung and the downstream filter fixed --
was present in the paper and merely *pointed at* in one clause of the record until the backward pass
quoted it, at which point the row's label changed.

### Non-negotiables, extended

14. **`c_d_route` decides nothing, and App. G says so where it reports it** -- as a floor, with the
    backward pass and its four relabels disclosed, and with the precedence rule stated so a reader can
    recompute the stratification under a different precedence from the locators alone.

---

## The amendment timeline, and what "pre-registered" therefore means here

This section is written last, immediately before the commit that lets `analyze_literature_audit.py`
score, and it exists so that no reader has to reconstruct the chronology from five amendment headers
that all carry the same date.

| what | committed / written | coded rows in hand at that moment |
|---|---|---|
| Base rubric: five criteria, four ambiguity rules, the 10-paper floor, the 12 non-negotiables | committed `9b8a395` | **0** |
| Amendment 1 (retrieval endpoint) | committed `3c61301` | **0** |
| Amendment 2 (blind-coding disclosure widened by one paper) | written after coding began | **5** |
| Amendment 3 (purpose test for C-a) | written after coding began | **5** |
| Amendment 4 (4a distinctness, 4b distributive test, 4c ordering) | written after coding finished | **59** |
| Amendment 5 (`c_d_route` routes and precedence) | written after coding finished | **59** |

**What is pre-registered without qualification:** the primary quantity and its definition, all five
criteria's frozen text, the four ambiguity rules, the frame and its screening procedure, the 10-paper
floor, the three admissible outcomes, and the single-coder limitation. **None of those changed.** The
sentence the audit was built to change, the prohibition on sampling, and the outcome mapping were all
fixed at `9b8a395` before any paper was retrieved.

**What is not:** Amendments 2 through 5. They are adjudication rules for questions the frozen criteria
underdetermine, and every one of them was written with some codes visible. Amendments 2 and 3 postdate
5 rows; **Amendments 4 and 5 postdate all 59**, which is the weakest position an amendment can occupy
and is why each one states the row that forced it and quantifies its exposure. The exposures, restated
in one place: 4a is the only rule any headline number depends on (**primary = 6 with it, 5 without**);
4b and 4c can only lower C-a and move no primary; Amendment 2 changed a disclosure and no code;
Amendment 3 raises three codes and lowers none in the frame; Amendment 5 enters no rate at all.

**What this does not license.** No amendment may be written after this commit. If a further question
arises during drafting, the answer is to report the row as it stands and name the question in App. G,
not to add Amendment 6. And no amendment relaxed a criterion to admit a row: every one of the five
either widened a disclosure or resolved a granularity or purpose question in the direction that keeps
the frame's coding uniform.

### Non-negotiables, extended

15. **App. G reproduces this timeline table** -- which amendments postdate how many coded rows, and
    4a's six-or-five exposure -- and does not describe the audit as pre-registered without it. The
    scoring script pins the commit hash of *this* file including this section, so a reader can verify
    that no rule was added after the rate was emitted.
