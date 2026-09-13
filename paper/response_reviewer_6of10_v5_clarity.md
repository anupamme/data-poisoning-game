# Response to Reviewer (6/10, Weak Accept; confidence 4/5; clarity 6/10)

Thank you, and specifically for the second half of the review. You wrote that *"clarity/writing is the
easiest dimension to move from ~6 → 8 in this version. The technical content is already there; the
problem is that the reader has to do too much work to discover the simple story."* We agree, we treated
that as the whole assignment for this revision, and every change below is a presentation change: **no new
experiment, no new claim, no number moved.** The four frozen artifact md5s are unchanged
(`dose_femnist/summary.json` `a16ef13f`, `all_compositions/summary.json` `2f4d9920`,
`headline_seed_topup/summary.json` `a0717893`, `comparability_six_cells.json` `2e8cced3`).

Two things bound what a clarity round can do here, and they are worth stating because they explain the
shape of the answers:

1. **The main text is at a hard 9-page limit with under one rendered line of slack.** Splitting long
   sentences and long paragraphs — the core of what you asked for — *costs* page space: each paragraph
   break spends `\parskip` plus the ragged last line of the paragraph it ends, about one rendered line
   per split. This round adds **nine** paragraph splits. Prose is flat (the body is 40072 chars at HEAD
   against 38952 now, and the split-induced overflow was ~12 rendered lines), so the splits had to be
   funded from named sites. Part 6 is that ledger.
2. **Nothing was deleted to pay for it.** Standing rule since Round 48: consolidation moves a claim's
   home, it never removes one. Every clause demoted below is quoted with the appendix line that already
   carried it, and the body keeps a pointer.

---

## Part 1: the seven clarity changes

| # | your change | disposition | where to check |
|---|---|---|---|
| 1 | Don't lead with causal vocabulary; go FL problem → evaluation failure → causal explanation → corrected design | **done.** The two measured FL facts and the FL grounding now precede *positivity* and *counterfactual*. §1's causal-vocabulary token count is **13 → 7**, and a gate now asserts that the first one appears *after* the FL problem | `:92`–`:120`; `measure_clarity_load` line `§1 causal vocabulary` |
| 2 | §3's opening is the "single biggest clarity problem": plain words before P1–P5 | **done.** The opening is now five senses in plain words, ordered *from what the defense computes to what the attacker gets*, then the centred plain-word chain `statistic → ordering → decision → admission → suppression`, then Def. 1. 868 → **736** chars, with the chain added inside that | `:272`–`:283`, rendered p5 |
| 3 | The four-word vocabulary as a two-column table | **done**, exactly as you specified: *statistic* = what the downstream defense computes; *decision* = which client or coordinate it selects (P3); *admission* = whether adversarial input enters its aggregate at all (P4); *suppression* = whether the attack still succeeds (P5) | `:308`–`:317`, rendered p6 |
| 4 | Give the reader one memorable sentence | **done, in your words.** `:161` now closes: *"The problem is not insufficient sample size: the required counterfactual is absent by construction."* | `:161`, rendered p3 |
| 5 | Reduce to three contributions, one claim each | **already three**; the defect was length, so the three items are **2122 → 1295 chars** and every demoted detail is restated in the section the item cites | `:137`–`:139`, rendered p2 |
| 6 | Move failed-hypothesis history into one table; cut the meta-history from the main text | **done by demotion, not deletion.** `tab:status` (`:2078`) already carried every withdrawal; this round moves the *provenance* clauses out of `box:scope` and §5–§6 into the appendix homes that already stated them. The disclosures themselves all remain — see Part 4 on why | Part 6's ledger |
| 7 | Rewrite the abstract with shorter sentences | **done, with every hedge kept.** 7 sentences → **14**; longest sentence **334 → 179 chars**; total **1594 chars** (was 1595). Your draft dropped two hedges — *we do not claim admission predicts security*, and the emergent blind spot — and both are back in | `:66`, rendered p1 |

Two of your asks that were not on the numbered list, and are also done:

- **Name both dissociations in the narrative, not only inside the contribution list.** §1 now states Krum's
  decision flipping in $0.733$ of rounds at unchanged admitted mass with $|\Delta|{=}0.026$, and
  `cos_krum`'s bit-identical selection with ASR falling $0.173$ (`:114`, rendered p1).
- **Move the adaptive caveat earlier.** It appeared first on p6; the word *committed* is now in §1's first
  statement of the finding, and `box:scope` (p5) adds *"we claim nothing about adaptive robustness"*.

## Part 2: six asks that were already satisfied, with the evidence

These are reported rather than claimed, because knowing them changes what the review's priorities buy:

| your ask | state, measured |
|---|---|
| the "ideal main-paper structure" (intro → why the comparison fails → what preservation means → protocol → experiments → what the screen can and cannot do) | **already the section order**: §1, §2 *The Testability Boundary*, §3 *Which Level a Composition Preserves*, §4 *A Causal Evaluation Protocol*, §5 *Statistic Preservation Does Not Identify Attack-Suppression Preservation*, §6 *What the Boundary Licenses*, §7 *Conclusion* |
| reduce to three contributions | **already three** (see Part 1, row 5) |
| put the failed-hypothesis history in one table | **`tab:status` exists** (`:2078`), and says on its face that it is hand-maintained |
| the reader needs the intuition before the formal proposition | **`:161` is already `\textbf{Intuition.}`**, immediately before Prop. 1, and Prop. 1 still lands on **p3** (gated) |
| Fig. 2 should follow the proposition | it already does, on p3 |
| treat the literature audit as supporting evidence, and avoid "X% of the literature" | the audit has **zero main-text footprint** beyond one appendix pointer, and `grep -c "of the literature" paper/main.tex` returns **0** |

## Part 3: the four §21 priorities that are not clarity edits

- **The ML-novelty objection ("probably the most important reviewer objection").** It had no address in the
  paper; it now has a titled paragraph, `\paragraph{Why this is more than a standard positivity
  violation.}` at `:202` (rendered p4). It concedes the standard part first — *"The violation itself is
  standard and we claim no credit for it"* — and then names the four things around it that are not: the
  estimand is *which level of a preservation hierarchy an upstream transform leaves intact*, a hierarchy
  with no analogue outside this setting; the emptied cell comes from the field's own eligibility
  convention rather than an analyst's slip; the channel that reaches the outcome without passing through
  the statistic has to be closed by construction; and a real selection methodology — ours — reverses the
  sign of its own conclusion once the gated comparison is replaced.
- **Equivalence-margin sensitivity (§21.3: "show pass/fail at ±0.05, 0.10, 0.15, 0.20").** The analysis
  existed and was stronger than the ask, but it lived in a *different document*, so no main-paper reviewer
  would have found it. Two changes: your ladder is now emitted by
  `experiments/analyze_margin_sensitivity.py` part (D) and printed as **Table A22** (`:2291`, rendered
  p64), and one body sentence surfaces it — *"the binding arm's $m^\ast$ is $0.034$, so every equivalence
  reading in the paper survives any margin above that, $\pm0.05$ included"* (`:370`, rendered p7). Part
  (D) reads only the frozen artifact and writes no artifact; the script's binding-arm staleness guard is
  intact, which is what makes that $m^\ast$ falsifiable. The one arm that fails at every rung
  (reputation / scaling, interval $[-0.112,+0.469]$) is named in the caption *and* in the text, because
  the paper makes no equivalence claim there and a bare $\times$ row would look like a contradiction.
- **The transportable rule (§21.6: the reusable artifact is the recipe, not the Krum result).** Stated once,
  in the Conclusion: *"to attribute an outcome to an upstream mechanism in a composed pipeline, hold the
  downstream mechanism and the threat fixed and intervene on the upstream component"* (`:506`). Its
  negative half is the next paragraph's imperative — *do not select the pairs you compare on the outcome
  you are about to attribute, the design that reversed our own sign* — which is where an evaluator acts.
- **Figure 1, panel (c).** You asked for the two sign-reversal rows to be legible rather than counted. All
  seven dumbbells stay; the two rows the artifact classifies `SIGN REVERSAL` now carry a heavier connector
  and print both means on the panel ($-0.272$ / $+0.098$ and $-0.213$ / $+0.071$), the panel title names
  the finding, and the caption's panel-(c) sentence was rewritten in the same edit so a caption that
  counts cannot describe a panel that names. The reversal rows are read from
  `results/comparability_six_cells.json`, never as literals.

## Part 4: two declines, and one thing we did not rename

- **Renaming *admission*.** Declined. Def. 1 itself states on its face that *admission* is the pre-registered term for level (P4)
  (`:294`); renaming it now would put the paper's vocabulary out of step with a frozen
  pre-registration that a reader can check. The two-column glossary (Part 1, row 3) is our answer to the
  underlying problem, which is that the word was never defined where it is first used.
- **Renaming P1–P5, and *preserved attack suppression* → *suppression invariance*.** Declined. Those labels
  are baked into the rendered pixels of `figures/modeS_causal.pdf` and `figures/story_chain.pdf`, which
  are shared with a second document; a rename that a LaTeX build cannot see is exactly the class of defect
  that has bitten this paper before. The plain-word chain in Part 1, row 2 gives the reader the meanings
  without the labels.
- **"Ordering", not "ranking".** Your draft chain says *ranking*; the paper's chain says *ordering*, because
  Def. 1 (P2) is stated as *"$S(T(U)) = \varphi(S(U))$ for some $\varphi$ preserving the ordering $S$
  induces on clients"*. We kept the definition's own word rather than introducing a synonym.

## Part 5: the clarity scorecard, as numbers

Gated by `python3 -m experiments.measure_clarity_load --gate`, so a regression fails a build rather than
being noticed in a later review:

| counter | before | now | gate |
|---|---|---|---|
| longest abstract sentence (chars) | 334 | **179** | ≤ 200 |
| abstract sentences | 7 | **14** | — |
| §1 causal-vocabulary tokens | 13 | **7** | ≤ 8, and none before the FL problem |
| longest body paragraph (chars) | 1137 | **846** | ≤ 850 |
| bold runs / italic runs in the body | 41 / 71 | **39 / 62** | ≤ 39 / ≤ 62 |
| max cross-references in one paragraph | 5 | **5** | ≤ 5 (held, not improved) |
| body paragraphs | 45 | **53** | — |
| comment-joined paragraphs | 0 | **0** | 0 |
| negation density (tripwire, not a target) | 45.8% | **50.0%** | must not fall |

Two of these are honest failures to improve: cross-references per paragraph is **held at 5**, not
reduced, and the contribution items are 1295 chars rather than the ~1150 we aimed at.

## Part 6: what the nine paragraph splits cost, and what paid for it

Sites demoted, each with the appendix line that already carried the claim verbatim:

| moved out of the body | home it already had |
|---|---|
| the recall arithmetic beside Prop. 2 ($|\mathcal E|/|\mathcal S|{=}1/5$, the $80\%$ cap against a measured $40\%$) | `app:boundary`, which states both numbers in one sentence |
| the frozen-threshold recital and $R^2$ in §5's replication paragraph | `app:targeted_full` and limitation (L2) |
| the four per-aggregator instantiations of *admission* | `app:channel_table`, whose subject they are |
| Theorem 8's $\rho$-vacuity and the $3.24 \to 1.01$ arithmetic | `app:proofs` and (L5) |
| the nine-variant frontier recital | Table `app:frontier`'s own body and footnotes |
| `box:scope`'s two research-process clauses (C0 was post hoc; the instruments are measurement devices) | (L4); and six other sites, including `tab:tiers`'s oracle row and the supplement's limitation 2 |
| §5's channel decomposition and freeze-commit recital (two paragraphs consolidated into one) | `app:score_only`, which names channels (a)–(d), splits the $0.892$, records the freeze commit, and gives $-0.023$ against $-0.026$ |
| §6's EMNIST margin clause | the general statement at `:370` plus Table A22's own row |
| §6.2's $+0.762 \to +0.178$ shrinkage numbers | `app:score_only`, which adds the one-sided Welch $p$ and the admission and decision moves |

One clause had **no** second home, so it was *moved* rather than demoted: *"A preserved statistic is not
thereby uninformative; it is not sufficient"* now sits in the appendix beside the proposition whose
reading it guards, and §5 keeps a pointer to it.

**One deviation from our own plan, reported rather than buried.** All three of §6's subsection headings
wrapped onto a second rendered line, costing three lines of page 9 for no content, so they were shortened
(*"The confounded ladder that motivated the instrument, and where its sign disagrees"* →
*"The confounded ladder, and where its sign disagrees"*, and similarly for §6.2 and §6.3). This round's
plan had "no retitling of any section" as an explicit non-goal. We took the deviation because the
alternative was deleting three lines of prose, and every clause dropped from a heading is the first
sentence of the paragraph it heads.

**One defect this round's own page-by-page read caught, and it was ours.** Demoting the recall arithmetic
left p4 saying *"so the $40\%$ is an operating point, not a ceiling"* with no $40\%$ anywhere earlier in
the main text — a definite reference to a number the same edit had removed. The bound's own sentence now
names its referent, *"which is what puts the bound above the recall we report"*, and the operating-point
sentence points back at it: *"so that recall is an operating point, not a ceiling"* (both on p4). We
mention it because it is the exact failure mode of consolidation, and a green build did not see it.

## Part 7: what this revision does not do

- **No new experiments.** Your §6/W2 request for larger $N$/$K$/seed counts and a second composition
  dataset is compute, not clarity, and we do not pretend otherwise.
- **No renaming** of C0–C3, Mode S/A/M, *influence* or *attenuation*, for the pixel reason above.
- **No claim, scope condition, limitation, withdrawal or disclosure deleted.** You scored transparency 9/10
  and called it *"exactly the kind of thing that makes a skeptical reviewer more comfortable"*; we read
  that as a reason to keep every disclosure and move only where it lives.
- **No code or data package accompanies this submission.** Unchanged and stated in the paper.
