# Response to the 6/10 Borderline Accept review (confidence 4/5)

This review read `main(9).pdf` and scored Novelty 7, Technical correctness 7, Empirical evidence 6,
Significance 7, ML relevance 6, **Experimental breadth 5**, Clarity 6, Reproducibility 7. Its own §19
simulation lands 5.5–6.0 and names two blockers: breadth, and the seed count on the result it calls
"the result I would build the entire paper around."

**We do not claim this round reaches 8/10.** It hardens that centerpiece from n=5 to n=20 on both
legs, closes three presentational gaps the review names, and answers the length request with a
measurement rather than a third refusal. It adds **no** new dataset, architecture or `N`, so the
breadth score is untouched by design and we say so rather than implying otherwise.

Every `` `:NNN` `` below is a line of `paper/main.tex` (or `paper/supplementary.tex` where marked) in
the current draft, resolved by grep and machine-checked to be non-blank and not a comment line. They
are **not** the numbers printed in the PDF's margin: ICLR's margin numbers are rendered line numbers
and run about four ahead of the source.

---

## 1. The round's main product: the sign reversal is now n=20 on both legs

The review's §10 asked for more seeds and its §21.1 asked us to build the paper around the sign
reversal. Those two asks met on the same cell: the reversal was **n=5 on both legs** while the earlier
n=20 top-up covered only the *equivalence* arm. That is now fixed.

**Pre-registered before any run, and the run refuses to start otherwise.**
`experiments/pre_registration_reversal_seed_topup.md`, git-committed at **`fb94d3a`**, amended three
times (`115b804`, `1efe5ec`, `89beed0`). `experiments/run_reversal_seed_topup.py` carries a
`PREREG_COMMIT` guard that aborts unless the document is committed at that hash with a clean tree.

**The demotion clause, quoted verbatim from the pre-registration, because naming the downside in
advance is the only thing that licenses adding seeds:**

> **Reversal demoted at n = 20 (the demotion clause):** if **either** interval contains zero, **or**
> the two intervals overlap, the paper reports the sign reversal as **not established at n = 20** and
> demotes it to a design *disagreement*. That demotion is made in the abstract, in §1, in Figure 1(a)
> and 1(c), in §5 and in the Conclusion — not in a footnote and not only in the appendix. The n = 5
> result is reported alongside it, and the disagreement between the two n's is the finding.

**Harness check before any scored run.** `--harness-check` recomputed three published values for this
arm at seed 42 and matched all three **bit-identically** (|d| = 0.000e+00 in both ASR and clean
accuracy, against `results/dose_response/` at κ=0 and κ=2 and `results/dose_replication/` at κ=0). At
κ=0 the transform returns the update list unwrapped, so the identity rung is one computation shared by
both legs; that is what the check establishes rather than assumes.

**The run.** Seeds 47–61 (15 new seeds, contiguous, fixed in advance, no interim look, no extension),
endpoint rungs κ∈{0,2} only, both designs, 45 runs, 7.9 h wall clock, 0 runs below the 0.35 accuracy
floor. Progress was judged by counting `per_seed` entries in the artifact, not by the runner's
`[i/N]` index, which counts resumed-and-skipped runs.

**The result: the reversal survives and narrows.**

| | confounded (outcome-gated) | controlled (Mode S) |
|---|---|---|
| n=5, as published | −0.2721 [−0.4217, −0.1224] | +0.0981 [+0.0178, +0.1783] |
| **n=20, both legs** | **−0.2733 [−0.3337, −0.2128]** | **+0.1251 [+0.0954, +0.1547]** |

Signs opposite, both intervals excluding zero, intervals disjoint: the demotion clause does **not**
fire, and the cell is now **equal-n** (20/20), which is stronger than the krum/scaling cell's 5/20.
Endpoint p-values are one-sided paired t at n=20: p↓ = 6.4×10⁻⁹ and p↑ = 1.9×10⁻⁸.

**What is deliberately *not* extended.** The interior rungs (κ=0.5, 1.0) were not run, so this cell's
four-rung Jonckheere–Terpstra trend statistics stay at n=5 and stay `pre_registered: false`. No
four-rung display prints a mixed-n grid without a per-rung `n`; `tab:comparability`'s caption states
which rows are n=20 (the outcome rows) and which are the frozen n=5 (the mechanism rows), and why
(`:859`).

**Where it landed.** Abstract `:66`; §1 `:129`; §5 `:446`; `tab:comparability`, caption `:859` and effect row `:882` (regenerated,
byte-identical to `build_comparability_table.py`'s emitted `latex`); the seven-cell table's row 4,
`20/20` `:1880`; the appendix discussion `:1704`; Figure 1(c), which reads the artifact and was
regenerated; and the supplement at `supplementary.tex:386` and `supplementary.tex:464`.

**Every site quoting −0.272/+0.098 or −0.273/+0.125 now prints its own `n`.** The pre-registration
fixed the split before the result existed, because `+0.098` has two referents sourced from the same
directory: the comparability cell's controlled leg (**moves to n=20**) and Table 1 row 4's Mode-S
endpoint ΔASR (**stays at its own frozen n=5**, because Table 1's rows are n=5 throughout and mixing
one row would make them incomparable). Table 1 row 4 prints `5` in its own `n` column (`:404`).

**The reader trap this created, and the clause that closes it.** `:438` (frozen, n=5) and `:446`
(n=20) sit eight lines apart in §5 and both quote what is arithmetically one estimand. `:446`
therefore states in one clause: *"The n=5 above is the same estimand at its frozen threshold's own n:
the two differ in n, not in quantity."*

**Artifact hashes.** `results/comparability_six_cells.json` moves by design, `2e8cced3` →
**`6e25ef3a`**. The move is disclosed in rendered text at `supplementary.tex:432` and
`supplementary.tex:438`; in `main.tex` it is recorded only as a source comment (`:2337`–`:2338`, the two
comment lines cited in this letter, flagged as such because the rule above says every other citation is
non-comment). The other three frozen md5s are unchanged and were re-verified on disk: `a16ef13f`
(`results/dose_femnist/summary.json`), `2f4d9920` (`results/all_compositions/summary.json`),
`a0717893` (`results/headline_seed_topup/summary.json`) — each is the md5 of that one file, not of
the directory. `workshop_paper/figures/modeS_causal.pdf` is still `76c29dcb`. None of those four hashes is printed in
either paper; they are internal verification, and only the sixth-cell artifact's is published, because
only it changed a published number.

---

## 2. What the review asked for that the paper already carried

Verified against the current source, not from memory. Each row cites where a reviewer can check it.

| review item | where it already is |
|---|---|
| §21.1 make the sign reversal the centerpiece | abstract `:66`, §1 `:129`, §5 `:446`, Conclusion `:524`, and Fig. 1(c), which marks the reversed rows and prints each row's own `n` |
| §15 "does not lose power, it gets the sign wrong" | `:446` verbatim: *"here the confounded design does not lose power, it points the other way"* |
| §21.2 demote the screen | `:149`: *"the screen that follows is a consequence we price rather than a third contribution"*; the contributions are three |
| §21.4 bring the controls forward | Table 1 (`:399`–`:406`) carries all six rows in the **main text**, including score-only Krum and the EMNIST replication |
| §14 re-tier the theory | the body states exactly two propositions (`:174`, `:219`); Thm. 8 is stated only in the appendix and appears in the body twice as a pointer (`:237`, `:348`) |
| §20 state the novelty fast | `:213`, *"Why this is more than a standard positivity violation"*, four items |
| §7 make the thesis FL-specific | abstract sentence 3, and `:213`'s *"that hierarchy has no analogue outside this setting"* |
| §9 the straw man | `:169`, which states the natural test and why it fails, before any of our machinery |
| §17 remove the historical-evolution narration | one non-comment "Round N" sentence survives in `main.tex`'s 2,624 lines (`:2334`), a git-hash provenance fact |

---

## 3. The three presentational gaps, now closed

**B1 (review §12): P4 is a causal intervention variable, not a proposed security metric.** The review
asked for this in the first paragraph where P4 is introduced. `:341` now says it there: *"It is a
causal intervention variable, not a security metric a deployed defense could compute: both
instruments set it by reading adversary identity."* The oracle fact previously lived only in the
Conclusion and the Ethics statement. Nothing was removed to make room.

**B2 (review §11): why ±0.15 is safe.** We do **not** supply a rationale after the fact — `:2334`
still states plainly that the margin was frozen at `5130cec` *"with no written rationale"*, and that
refusal stands. What `:382` now imports is the one fact the pre-freeze record does support: the frozen
rule **conjoins** the margin with *every rung below 0.5*, so a generous margin cannot certify a
high-ASR arm. The sensitivity the review asked for over [0.10, 0.20] was already stronger than
requested and is unchanged: the binding arm's m\* is 0.034, so every equivalence reading in the paper
survives any margin above 0.034, ±0.05 included.

**B3 (review §10): make the evidentiary hierarchy unmistakable where the numbers are.** Main-text
Table 1 carried no `n` at all. It now has an `n` column (`:399`), so the reader meets the seed count
in the same row as the effect: 5, 5, 8, 5, 3, 5. The seven-cell table goes further and prints `n` as
**confounded/controlled** (5/20, 5/8, 5/5, **20/20**, 5/5, 5/5, 5/5), so the strongest row and the
weakest are distinguishable on the page.

---

## 4. The length request, answered with a measurement

The review says 20–30% of the appendix could go without losing scientific content. We measured it
instead of asserting otherwise. `experiments/measure_appendix_redundancy.py`, runnable by a reviewer:

- Window: the appendix, 340,362 non-comment chars. Environment bodies (tables, figures, proofs) are
  excluded from the pool: 71,249 chars (20.9%), leaving a 269,199-char prose pool.
- Units: 1,038 sentences of ≥8 words and 210 paragraphs of ≥40 words.
- Similarity: Jaccard over **3-shingles** for sentences (threshold 0.30) and containment over
  3-shingles for paragraphs (threshold 0.15). Both thresholds over-report on purpose.
- Found: **3 sentence pairs and 12 paragraph pairs.** Each was read individually.
- Of those 15 pairs, **exactly one** has an equal numeric set — a *sentence* pair, one member in the
  paragraph at `:1152` and the other in the paragraph at `:1572` (0 of the 12 paragraph pairs match on
  numbers).

Every other pair is one disclosure template instantiated on a **different arm** — different `n`,
different aggregator, prose against a formal statement — so deleting either member deletes that arm's
disclosure rather than a repetition. The script prints each pair's numeric difference so this is
checkable rather than asserted.

The single same-numbers pair is a **declared preview**. Its host paragraph at `:1152` opens *"Two
boundary conditions, stated alongside the primary result. Both are developed in full later, but neither
should be met late,"* and `:1572` is that full development — the shared sentence is the α=0.1
heterogeneity result (fg→rfa 0.569, rep→cm 0.548), stated once as a warning beside the primary result
and once where it is analysed. Deleting the first removes the early warning; deleting the second
removes the analysis.

**So the applied de-duplication yield is 0 characters, against the ~456 we projected. We report the
projection and the miss rather than the projection alone.** The length a reader is reacting to is not
duplication: it is that every empirical claim carries its own tier word, its own `n` and its own scope
hedge, which is the discipline the tiers appendix imposes (`:683`–`:685`) rather than repetition. We have no
defensible count of those claims — the evidence-status ledger is one dense paragraph (`:2212`), not an
enumerated list — so we do not quote one.

**What we did instead** is treat the complaint as navigation rather than volume. The reading guide now
carries a per-top-level-section index (`:663`–`:681`, under the reading guide at `:611`), one line each in document order, saying what
each section licenses and whether it is skippable. This **adds** to an appendix the review asked to
shorten; that trade is made knowingly and stated in the source.

---

## 5. Four declines, with reasons

1. **The title (§16).** The review prefers its Option C and says *"this isn't essential."* Declined:
   the current title names the estimand, and no scope condition, term or claim in the paper is keyed
   to a renaming.
2. **Moving the novelty paragraph `:213` into §1 (§20).** Declined: §1's causal-vocabulary count is at
   its cap of 6/6, and §1 already carries the plain-words version at `:129`. Relocating the paragraph
   would either raise a ratchet or strand its plain-words twin.
3. **A fourth hedge on the masking result (§13).** Declined: it is already hedged three times (`:491`
   *"One mask is not the class of all non-rescalings"*, the appendix, and the abstract's *"a second
   invariance class"*). A fourth would add a negation to a draft whose negation density is already a
   monitored 50.6%.
4. **Cutting 20–30% of the appendix (§18).** Declined on the measurement in §4, not on preference. No
   scope condition, limitation, withdrawal or disclosure was deleted this round; consolidation moves a
   home, it never removes one.

---

## 6. Deviations from our own plan, reported rather than buried

1. **B5's funding premise was false.** We planned to fund B2 by compressing `:382`'s clause *"an
   evaluation-design tolerance … not a claim that 15 ASR points are universally negligible"* on the
   ground that the appendix restates it. It does not: `:2334` carries the *conjunction* fact and the
   no-rationale refusal, not that clause. Compressing would have left the claim with no home, so B2
   landed as an addition and B5 was not taken.
2. **B3 cost table geometry.** Table 1's `n` column was bought with `\tabcolsep`; the overfull-hbox
   count is the arbiter and it is 0.
3. **C1 yielded 0 chars against a planned 456** (§4 above).
4. **The appendix grew by one page** — the per-section index, plus this round's new disclosures.
5. **One site was reclassified after the classes were fixed**, which non-negotiable 10 requires be
   reported as such. Amendment 2 assigned the line now at `:1664` to Class B on the stated ground that
   it "quotes the frozen replication arm or a frozen table row." It quotes neither: its `0.098` is a
   **mean ASR at an amplified Mode A rung**, a level and not a paired difference, in a different
   experiment and a different table. Amendment 3 moves it **out** of scope into the named-collision
   list (now three members), keeping the split exhaustive and disjoint at 6 Class A + 13 Class B + 3
   collisions = 22. The correction is derivable from `paper/main.tex` alone with no n=20 number as an
   input, and it moves a site out of scope, so no result could make it attractive.
6. **Our own Amendment 3 contained a misstatement, which we correct here.** It listed the site now at
   `:1782` as satisfying the print-your-own-`n` rule via the sentence *"at n=5 a null from this test is
   not evidence of flatness."* That `n=5` belongs to disclosure (i), about the Jonckheere–Terpstra
   test; the `+0.098` is in disclosure (iii). The site does print its `n` — beside the rise, in
   (iii), which is where a reader looks — but not in the clause the amendment named.
7. **A published p-value changed meaning, not just value.** `tab:comparability`'s effect p was a
   four-rung Jonckheere–Terpstra trend p and is now the one-sided **endpoint paired t** at n=20 on the
   same per-seed differences as the interval beneath it. The interior rungs were not topped up, so the
   trend test stays at n=5 where it was frozen and its values are reported in the seven-cell section
   instead of mixed into that column. The caption says all of this (`:859`).
8. **`tab:sixcell`'s tabular geometry changed** (a wrapping `p{}` column for the CIFAR-100 label);
   checked in pixels on the rendered page, no collision.
9. **A second consumer of the artifact was stale and is now re-run.**
   `experiments/analyze_margin_sensitivity.py` also reads `comparability_six_cells.json`. After the
   top-up, `coord_median`/pixel's m\* moved **0.178 → 0.155**, so `tab:margin_ladder`'s cell, its
   caption and its provenance comment were updated. A field-by-field JSON diff confirms exactly one
   row moved. **The binding equivalence arm did not move** (krum/scaling, EMNIST at m\* = 0.0343,
   headroom +0.1157 — the number the supplement quotes), the script's own "BINDING ARM MOVED" warning
   did not fire, the AGREE tolerance is still inert over all of [0,1], and all seven published
   verdicts still reproduce.
10. **A published disclosure had gone false and is replaced by a stronger one, not edited away.**
    `supplementary.tex` §`app:sensitivity` (label `supplementary.tex:415`) asserted, of the artifact's first hash move, that
    *"no value inside the six published cells moved"* (`supplementary.tex:427`). After this top-up that is
    false. A new bolded clause at
    `supplementary.tex:432` states that the hash has moved a second time and that this move **did**
    change a published cell, names the cell and the moved m\*, and gives both hashes. The dated
    original sentence stays.
11. **The scope box's global seed counts were wrong twice, both caught in a pixel read of p5.** It
    stated a single global n=5 after this round made one cell n=20; and then, once corrected, *"n=20 on
    the sign-reversal cell"* was ambiguous, because there are now **two** sign reversals at different
    `n` (CIFAR-10 at 20, CIFAR-100 at 5). It now reads *"n=20 on the **CIFAR-10** sign-reversal cell"*
    (`:245`).
12. **§5's replication sentence quoted the CIFAR-100 pair with no `n`**, three rendered lines below an
    "(n=20)". It now prints `(n=5)` (`:450`), funded inside the same paragraph so the page budget is
    unchanged.
13. **A reader-checkable count in the Reproducibility Statement had gone false because of this
    round's own work.** The sentence spans `:563`–`:565` and promised *"the 20 pre-registration documents … which is the count of
    `experiments/pre_registration_*.md` and is checkable as one rather than taken on trust."* Phase A0
    created a 21st. Corrected to **21** at `:563`. No gate, artifact hash or build could see this: the
    emitter is the filesystem.
14. **Five prose compressions were spent to buy back three rendered lines** (`:245`, `:370`, `:446`,
    `:450`) after the n=20 rewrite pushed the main text onto an eleventh page. Each is a compression,
    not a deletion; no claim, hedge or citation was dropped. The page gate's deficit is reported in
    spilled words but is paid in rendered lines, and the cheapest line to buy back is a short final
    line of a paragraph.
15. **The pre-registration guard hash advanced across amendments while the work was in flight.** The
    harness check ran with the document frozen at `fb94d3a`; the scored run recorded `1efe5ec`; the
    runner's constant is now `89beed0` because Amendment 3 was committed after the run finished. The
    artifact records the hash that was in force when it was written, which is the honest record, and
    Amendment 3 changed a **classification**, not a decision rule or a threshold.
16. **A provenance comment recited a measurement that this round's own edits moved** (1,011 sentences
    and 209 paragraphs → 1,038 and 210). Corrected, and re-pointed at the script as the authority.

---

## 7. Verification state of the draft this letter describes

- **Fixpoint build**: 3× (`pdflatex main` + `bibtex main` + `pdflatex supplementary` + `bibtex
  supplementary`), then 3× alternating. Zero occurrences of LaTeX Error, undefined reference,
  undefined citation, multiply-defined, `??`, "Rerun to get", Overfull hbox and Overfull vbox in both
  logs. Main 68 pages, supplement 11.
- **Page limit**: `measure_body_chars --gate` exits 0; the body prose ends on p9 and the Ethics
  heading is the first content of p10.
- **Clarity ratchets**: `measure_clarity_load --gate` exits 0 with **no ratchet raised** — longest body
  paragraph 846/850, bold 39/39, italic 62/62, cross-refs per paragraph 5/5, abstract 1591/1600, §1
  causal tokens 6/6, comment-joined paragraphs 0. Negation density 50.6%, which is monitored so that
  it does not fall: a fall means a concession was softened.
- **Numbers**: `tab:comparability` is byte-identical to its generator's emitted LaTeX. The one
  word-stated magnitude near a moved number, Λ_a's "2.4×", re-divides correctly (0.08093/0.03323 =
  2.436).
- **Read in pixels, pp. 1–10 end to end**, plus Figure 1 panel by panel at 300 dpi against the
  generator's own draw predicates, plus the rendered seven-cell table. Deviations 11, 12 and 13 above
  were found this way and by no gate.
- **Anonymity and punctuation**: `pdfinfo` Title/Author/Subject/Keywords empty; no `\iclrfinalcopy`,
  no `\thanks`, no URL; zero U+2014/2013/2212 and zero unprotected `\texttt{…}\texttt{…}` joins in all
  three sources.

## 8. What this round does not do

It adds no dataset, no architecture and no new `N`; **Experimental breadth is not addressed and we do
not claim it is.** At n=20 this remains one aggregator, one attack, one dataset, one architecture and
one synthetic instrument that reads adversary identity. The top-up buys precision on the sign reversal
and nothing else — it is not evidence about any other cell, and Mode S is not a defense.

No code or data package accompanies this submission.
