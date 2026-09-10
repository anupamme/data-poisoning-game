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
