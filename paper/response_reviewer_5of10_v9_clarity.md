# Response to the fifth review (5/10, Borderline / Weak Reject, confidence 0.78)

**Scope of this response.** The review scores Clarity 5 and supplies twenty clarity items, a five-item
priority list, and its own acceptance test:

> If a reviewer can answer these five questions after ~3 pages, I would give the paper 8/10 clarity.
> Right now, I think they can answer them after ~10–15 pages.

**This round targets clarity/presentation only, and does not claim the overall 5/10 moves.** No new
experiment, dataset, seed, `N`, `f` or attack was run. The estimand is not sharpened further and the
regime is not widened; of the review's two *overall* levers we take only the prevalence half, and we say
below where that leaves the other half.

**The headline result, and it is now machine-checked rather than asserted.** All five of the review's
questions resolve to **rendered page 1** of `paper/main.pdf`, against the review's own threshold of page 3
and its estimate of pages 10–15. The check is a new measure in `experiments/measure_clarity_load.py`, not
a claim in prose; its output is quoted in §3.

**Which build the review read.** The overall-score half cites `main(4).pdf` (Aug. 31, thirteen days
stale — the same PDF the fourth review read); the clarity half cites `main(20260911-085716).pdf` (Sep. 11)
plus the Aug. 26, Aug. 28 and Sep. 8 builds. So the clarity charge cannot be dismissed as stale, and we do
not try to. But fifteen of the twenty items were already implemented in the current source, several of them
after every PDF the review had. Those are listed first, each with a line you can check, because they decide
three quarters of the response.

---

## 1. Already in the paper before this round — fifteen items, with a check-site each

| review's ask | state in current `paper/main.tex` |
|---|---|
| #3 stop introducing *the criterion* before explaining why it fails | The token `Criterion` occurs **0** times in the body (lines 103–568). The word is fixed once, in lower case, as `:169`'s "our own composability rule" |
| #5 kill C0/C1/C2/C3 from the main narrative | Measured by the gate: **C0:0 C1:2 C2:1 C3:0**, three tokens over 53 body paragraphs, against a target of ≤8 |
| #6 don't make the reader track two *Criterion 3*s | Same measurement: the token is absent from the body entirely |
| #14 reduce equation density; add a one-line interpretation | **Exactly one** display in the body, the τ(T) estimand, and the paragraph that follows it interprets it: `:204` says what is *not* estimated |
| #15 give every experiment a one-sentence question | Already the pattern at `:169`, `:356`, `:372`, `:493`, `:511`, `:520` and `:547`. §3 was the one exception, and B3 below fixes it |
| #16 make ΔS / ΔA / ΔASR explicit | Table 1 is exactly that — aggregate, decision, admission, influence and ASR columns, each row with its own `n` (`:426` is one such row) |
| #19 the introduction should end with exactly three contributions | The list at `:147`–`:151` has **exactly three** items |
| #8 move the 42-pair story later | It is already §6.3, the last subsection: `:538` |
| #13 Established / Supported / Not established | Appendix §"Evidence status: what is proved, what is supported, what is not" (`:2261`) plus §"The three tiers of evidence" (`:726`), and the body's scope box points at both |
| #18 theory should come after the evidence | Thm. 8 is **appendix-only**: "Bounded reweighting preserves the downstream discriminative mechanism" is stated at `:1022`. The body carries no mechanism-preservation theorem |
| #12 cut the *we had ourselves predicted* narrative | Two sentences remain in the body. `:153` is the paper's own refutation disclosure: "we had predicted collapses". `reviewer` and any rendered "Round N" return **0** hits |
| #2 put the conclusion in the first paragraph | `:109` is a single question and nothing else: "Everything below is an answer to that one question" |
| Priority 1: put the sign reversal on page 1 | Already on p1 in the abstract, at **better** numbers than the review quotes — see the note below this table. `:66` carries the pair with "intervals excluding zero" |
| Priority 5: aim for a ~9-page main paper | **The main text is 9 pages.** `measure_body_chars --gate` reports body prose ends p9, with the Ethics heading the first content of p10. The 45 pages the review measured is the appendix |
| Q1: what exactly is the causal estimand? | `:196` names treatment, outcome and $P$, and states $P$ is "not a measured covariate" — the exact estimand-vs-covariate distinction Q1 asks for. It landed Sep. 12–13, **after every PDF the review read** |

**On Priority 1's two numbers.** The review asks for "−0.272 vs +0.098" on page 1. In the current draft
those two values are from **different arms**: `+0.098` is `coord_median`'s ΔASR at $n{=}5$ (`:426`, and the
refutation it grounds at `:476`). The sign-reversal pair, on one cell, is **−0.273 outcome-gated against
+0.125 within-defense at $n{=}20$**, both 95% intervals excluding zero — a larger contrast at four times
the seeds. That pair is in the abstract on p1 (`:66`), restated at `:129`, and reported in full with both
intervals at `:484`. So Priority 1 is satisfied, at superseding numbers, and B2 below is what put the
body's restatement on p1 as well.

**Q2's remedy — "audit 15–30 papers".** A pre-registered audit of **59** papers already exists
(`:2047`), twice the requested scale. Its result was invisible to a main-text reader, which is the
single largest gap this round closes; see B1.

---

## 2. The five edits this round

| # | edit | site | check |
|---|---|---|---|
| **B1** | **The audit's prevalence is now in the body, as a number.** The sentence claimed the gate was a field convention without measuring it. It now reads that "its prevalence is measured, not asserted": 6 of 59 coded composed-defense papers (10.2%) gate attribution on the composed outcome while running no identifying contrast; 46 (78%) do run one | `:213` | `app:audit` is referenced from the body **once**, where before this round it was zero |
| **B2** | **Q5's two numbers moved onto rendered p1.** The sentence was reordered so "outcome-gated against" its within-defense counterpart precedes the mechanism explanation. Both values now sit at rendered columns 37 and 66 of a p1 line; before, the sentence began on p1 and both numbers wrapped to p2 | `:129` | The new gate scores Q5 at **p1** |
| **B3** | **§3 gains the italic lead question the other seven sections have.** "Preserved in which sense?" now opens it, absorbed into the existing first clause rather than added ahead of it | `:286` | Rendered line 238 of p5 |
| **B4** | **The preservation-chain figure is pointed at from p1.** The paragraph stating the two measured facts now ends "both drawn in Fig." A5, App. J — so a reader meets the paper in one picture on p1 instead of p3 | `:125` | Renders on p1 as a parenthetical pointing at Fig. A5, App. J |
| **B5** | **§2 and §4 retitled to say what they do.** "The Testability Boundary: Why the Gated Test Fails" and "A Causal Protocol: Intervene Upstream, Not Across Defenses" | `:166`, `:353` | Both fit **one** rendered line (137 and 295) — checked in pixels at 300 dpi, not by character count, because a wrapped heading costs a full line for no content |

**Funding.** The paper had zero page slack and every ratchet at cap, so each added line was paid for by
moving numbers into appendix tables that **already hold them** — never by dropping a claim:

| body site | number moved out | its home, verified by grep before the cut |
|---|---|---|
| `:397` | the $n{=}20$ TOST interval | `:1930`, a `krum/scaling` row of Table A12. The claim "The margin is not load-bearing" stays, with its pointer |
| `:386` | the permutation $p$ for the increasing leg | `:1724`, whose row is labelled "H-admission" — a row of the very table the sentence cites |
| `:511` | the confounded ladder's per-arm values | §6.1's appendix home; the sentence keeps the one-sided $p$ against its frozen threshold and still says it "refutes that suite's pooled prediction" |

Body prose is **39758** chars against a 39902 baseline: the round is net shorter, and the page count is
unchanged.

---

## 3. The review's acceptance test is now a gate

`experiments/measure_clarity_load.py` gained a tenth measure, `q5_pages`, which locates one distinctive
rendered phrase per question in `main.pdf` and reports the page it lands on, gated at the review's own
threshold of ≤3. Adding a measure is permitted by that file's contract; raising an existing ratchet is not,
and none was raised. Current output:

```
  first proposition on:     p3   (the central claim's rendered page)
  five questions by   :     p1   (the review's own 8/10 test; its threshold is p3)
                          Q1 what problem?   p1
                          Q2 what answer?    p1
                          Q3 why?            p1
                          Q4 how shown?      p1
                          Q5 what happens?   p1
```

Both failure branches were falsification-probed rather than assumed: against a stale PDF the gate reports
the test cannot be scored, and with the threshold lowered to 0 it reports "answered only by p1 > p0". A
phrase that is edited away therefore fails the gate instead of silently passing.

Every other ratchet held at or below its cap: longest body paragraph 846/850, bold 39/39, italic 62/62,
cross-refs per paragraph 5/5, `(P#)` tokens 12/12, first proposition p3/p3. `measure_body_chars --gate`
passes with the same residue as before ("residue before the heading: 'E'"). Negation density is unchanged
at 90/176 = 51.1%, which is the tripwire for a concession softened by compression; the numerator did not
move.

---

## 4. Five items declined, each with a machine reason rather than a preference

1. **#18's §2↔§3 swap (put the preservation hierarchy first).** Declined. `measure_clarity_load` gates
   `prop_page ≤ 3`, and §3 is about two rendered pages, so the swap pushes Prop. 1 to roughly p5 — which is
   verbatim the defect that file exists to prevent. **This review's own five-question test sides with the
   current order**, since Q1–Q3 are the why-the-evaluation-fails questions and it wants them answered
   inside three pages; they are now on p1. Raising a ratchet to accommodate the swap is prohibited by that file's own contract.
2. **#5/#6/#10 and #17 in full: renaming P1–P5 or C0–C3.** Declined. `figures/modeS_causal.pdf` and
   `figures/story_chain.pdf` **bake "P1"–"P5" into rendered pixels**, and the Mode-S generator also writes
   into a pinned companion document; a retired label inside a shared figure PDF passes every LaTeX check
   and every gate. Separately the ask is ~90% satisfied by measurement already: `Criterion` 0, C0 0, C3 0,
   C1 2, C2 1. The definition list at `:305` ("Value invariance") is where the names are fixed, once.
3. **Priority 5 / #20 "cut 25–35%, aim for 9 pages".** Declined as stated, because the main text **is** 9
   pages and cutting it further would cost disclosures. What we do instead is disclose rather than let the
   number pass: `main.pdf` is now **70** pages in total, of which the main text is 9, the references end on
   p12, and the appendix runs from its own reading guide on p13 to p70; the supplement is a separate
   11-page document. That is up from the 45 pages the review measured. ICLR limits neither the appendix nor
   the supplement, and no claim in the body rests on a reader reaching them.
4. **#12 / Priority 4 "move the revision history to an appendix".** Partly declined. Of the two body
   sentences left, `:511` was thinned this round, and `:153` is the paper's own refutation disclosure and
   is protected from deletion. The category the review objects to is otherwise already gone.
5. **#9's "boxed paragraph or prominent figure" for the sign reversal.** Declined as a box, delivered as
   B2's reorder. Bold runs are at 39/39 and a float costs at least six rendered lines against zero slack;
   the reorder achieves the stated goal — the numbers on page 1 — at negative cost.

---

## 5. Deviations from the plan, reported rather than buried

- **B4 moved from the Roadmap paragraph to `:125`.** The Roadmap at `:160` already carries five
  cross-references, exactly the per-paragraph cap, and it is one of three paragraphs whose pointers may not
  be cut. `:125` is on p1, has no other reference, and is the paragraph that states the two breaks the
  figure draws — a better home, reached for a mechanical reason.
- **Two planned compressions were not made.** `:532` and `:536` were left intact because the three
  compressions above already funded the additions and the page gate passed; compressing further would have
  removed numbers for no gain.
- **`results/literature_audit/` was rewritten by re-running the audit's own analyzer.** This is a repair,
  not churn: the only value that changed is a deduplicated-record count, 595 → 593, and
  `experiments/analyze_literature_audit.py` records 595 in its own comment as a known-wrong value from a
  looser dedup key. The paper already printed 593. Nothing cited 595, and a second run is byte-identical.
- **§2's title now differs from the companion document's.** The companion is pinned and was not edited,
  which is deliberate; the two are separate documents and no shared float depends on the title.

---

## 6. What did not change

No section was reordered and no float was promoted from the appendix. The paper title, §5, P1–P5, C0–C3,
Mode S/A/M and the words *admission*, *influence* and *attenuation* keep their names. No scope condition,
limitation, withdrawal or disclosure was removed — the three compressions moved numbers into tables that
already held them, and every one was verified present in its destination by grep, matching the referent and
not merely the digits, before the body sentence was cut. Neither equivalence convention was touched: the
ResNet18 arm keeps its withheld phrase and the EMNIST arm keeps its licensed one. The four frozen result
artifacts hash unchanged, the margin ladder's binding arm stays where it was, and both documents build to a
fixpoint with zero LaTeX errors, undefined references, multiply-defined labels, `??` marks or overfull
boxes.

**One thing the review asked for that we have not done.** Its second overall lever — demonstrating that the
gated pattern is *prevalent* — came out measured **low**, and a low rate was pre-registered in advance as
one of the three possible outcomes. So the audit stays in the appendix, the contributions list gains no
clause from it, and `:213` now says plainly that "A low rate was pre-registered as outcome (ii)". The
decline to call the gate common is kept and evidenced instead of hedged. Nothing in the paper implies the
identifying contrast is unrun: 46 of the 59 coded papers do run one.

---

*Every `:NNN` in this letter is a source line of `paper/main.tex` as of this round, resolved by grepping
current content rather than by shifting an earlier letter's numbers. Checked by
`python3 paper/audit_letter_cites.py paper/response_reviewer_5of10_v9_clarity.md`, which reports 46 cites
and 26 quotes with **FAIL: 0**. The seven quotes it lists as unchecked by construction are quotations of
the review itself and of gate output, not of the paper, so no source claim is in its blind spot.*
