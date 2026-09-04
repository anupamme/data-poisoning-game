# Response to Reviewer (6/10, Weak Accept / Borderline; confidence 4/5)

Thank you. This is the most useful report the paper has had, for one reason we want to state before
anything else.

## 0. There is no version mismatch to declare, and nothing to discount

Our two earlier replies each had to open by naming a draft the reviewer had read that was two
revisions behind. **Yours is a report of the current source.** You quote the 2 × 2 factorial that
finished immediately before you wrote (full −0.039, score-only −0.030, emit-only +0.046, additive
+0.016, residual −0.054 against the frozen 0.05 margin, "one seed carries the whole result"), the
R² = 0.239 pooled model, ρ_s = 0.351 at p = 0.091, the 40% recall ceiling, `cos_krum`'s 0.173 at
p = 0.044 with 5/8 seeds, the contents of both `(L·)` lists, and Theorem 8 by its current number.
Every criticism below is against what a reader will actually see, so we treat all ten as live.

**We also accepted your framing of what to do about breadth.** You wrote that the marginal value of
more experiments is now lower than the marginal value of making the conceptual contribution
unmistakable, and you named the non-goals explicitly: no more defenses, no hyperparameter sweeps, no
more seeds, no third dataset, no fourth attack, no new defense, no further predictive model. **We ran
no new experiments.** Every number in the paper is unchanged and still emitted by its generator; the
only script we touched is a figure generator, and only its annotation labels. The revision is
entirely writing, one figure, and the eight defects we found while doing it.

---

## Priority 1: the generality argument is now explicit, in the body, in §1

You asked for a subsection that separates what is general, what is mechanism-specific and what is
empirical. **The paper already contained that table, and it was invisible.** Table A2 (appendix p15)
has exactly your three rows: General (Prop. 1 and Prop. 2, "proved, and not specific to FL, to
backdoors, or to our conditions"), Formal (Theorem 8 and the invariance classes, "a laboratory for
composition, not a model of FL defenses in general"), Empirical (the arms, CIFAR-10 and one EMNIST
arm, n = 5). A reader who forms the judgement "narrow FL paper" does so in the introduction and never
reaches p15.

So the fix is promotion, not authoring. A new closing paragraph of §1 states the three tiers in
prose, and **inverts the framing**: Table A2 is written as what each claim does *not* license, and the
body version leads with what each tier *establishes*, closing each with its scope limit in one clause.
It ends by pointing once at the three inventories rather than restating them. Verbatim:

> **What is general here, and what is a federated-learning instantiation.** The paper makes three
> kinds of claim and never pools them. **General:** Prop. 1 and Prop. 2 are statements about the
> *shape of a screen*, proved with no dataset, architecture or scale in them, so what carries beyond
> this paper is the design they license and *not* our screen. **Mechanism-specific:** Thm. 8 and the
> invariance classes (Prop. 10) are proved for *positive per-client rescalings* into named rank and
> geometric aggregators, a laboratory clean enough to settle invariance algebraically rather than a
> model of FL defenses in general, and they **bound no ASR**. **Empirical:** the witnesses are four
> aggregators of three structural kinds under two committed attacks, on CIFAR-10 with one arm
> replicated on EMNIST-byclass, at n = 5 seeds (n = 8 for `cos_krum`): "replicated on a second
> dataset and architecture", never "generalizes". **The middle tier is both the one an FL reader will
> recognise and the narrowest of the three**, and that is deliberate: a broken implication has to be
> exhibited where invariance is decidable rather than argued. Claims do not transfer upward.

It is prose and not a table on purpose: a fourth float on p3 would cost more vertical space than its
content, and this document has twice had one enlarged float exile every later float of its class.

## Priority 2: (P4) is "adversarial contribution"; "admission" is the operationalization

You asked for the level to be named "adversarial contribution" with "admission" kept as the
aggregator-specific instrument. **The definition already read that way** ("Adversarial contribution
(admission) invariance"), and the paper already carried the sentence that (P4) is a level and not one
measurable quantity, with the four per-family instruments named. What was inconsistent was usage. So
this was a consistency pass, not a rename, and the rule we applied is:

- the **level** is "adversarial contribution": the abstract's closing sentence, the (P4) row of the
  hierarchy table, both definitions, the Figure 1 caption, and the conclusion's imperative;
- a **measurement, column, reading or frozen rule name** stays "admission": Δ adm., the channel
  tables, H-admission, "the admission reading", "admission-level change".

Every frozen pre-registration name and every generator string is byte-identical, because relabelling a
scored quantity after the fact is exactly what the pre-registrations exist to prevent. The body now
has 5 level-naming sites, all reading "adversarial contribution", and 18 measurement sites, all
reading "admission", and we checked each of the 18 individually rather than trusting a count.

## Priority 3: Figure 1 is now readable without the caption

Your test was 30 seconds. The panel already had your sketched geometry; two things defeated it. The
arm each broken arrow belongs to was stated only in prose, and the caption spent about 380 characters
explaining that fact.

**In the figure**, each annotation now names its own scope:

| before | after |
|---|---|
| `flips 0.80 of rounds` | `Krum: flips 0.80` |
| `support unchanged (0/240)` | `unchanged, 0/240, 4 arms` |
| `cos_krum ASR falls 0.173` | `cos_krum: falls 0.173` |

All three stay inside the 25-character budget the layout requires (the segment midpoints are only
about 0.2 apart at 4.6 pt, which broke an earlier one-line attempt), and every number is still a live
generator variable, with `0.173` still differencing the two *published rounded* rung means under the
existing assert that pins the two conventions to < 1e−3.

**In the caption**, the first two sentences now state the whole chain and both breaks in plain words
with no counts and no citations, and one sentence replaces the 380-character explanation: the three
annotations are three scopes and not one experiment. It went from 1933 to 1749 characters.

**One thing we got wrong on the first attempt, and caught.** Our draft label for the middle segment
was `Krum: support 0/240`. That is false. The generator computes that count by summing over the four
CIFAR-10 rows of the channel table (4 arms × 4 rungs × 15 rounds = 240), so attributing it to Krum
would have put one arm's name on a pooled quantity, which is the same defect class a previous round
caught in an arrow annotation. The label reads `4 arms` instead, the reasoning is recorded in a
comment in the generator, and the pooling arithmetic that used to live only in the caption
(100 of those 240 rounds carry adversarial mass at baseline, 48 per arm for reputation and
`coord_median`) is now written into the appendix section the caption points at, rather than deleted
with the caption text.

---

## The ten numbered weaknesses

**1. Experimental breadth.** Answered as you asked, by sharpening generality rather than adding arms.
Priority 1 above is the whole of our answer, and we accept that it either works or the score stands.

**2. The final recommendation reads as a metric.** Adopted. The old imperative ("evaluate composed
defenses by measuring the adversarial contribution that survives the downstream mechanism") invited a
deployment reading, with the qualifier arriving afterwards. It is now scoped as an evaluation
principle, with the obstruction in the same sentence:

> **The implication, which is an evaluation principle and not a metric.** To evaluate a composed
> defense causally, manipulate or measure the adversarial contribution channel rather than using
> preservation of the statistic that feeds it as a proxy for suppression. That channel is (P4), and
> Mode S reads it only by knowing adversary identity, so what we offer is a design for identifying
> it, not a quantity a deployment could screen on. That is also what leaves the open question: can
> (P4) be estimated without knowing which clients are malicious?

**3. Too many numbered objects.** The logic diagram you sketched is Figure 1(a), now annotated so it
carries the argument alone (Priority 3). We did not add a second diagram, because the problem was that
the existing one could not be read without prose, not that one was missing.

**4. The factorial is fragile and the rhetoric overstates it.** Adopted, and this is the change we
think most improves the paper's honesty. The body asserted that "a pre-registered 2 × 2 factorial
answers it no", which claims non-additivity on an arm we ourselves report as not decisive. It now
reads:

> **Whether those channels decompose the *suppression* effect is a separate question, and a
> pre-registered 2 × 2 factorial does not establish that they do**: at n = 5 the additive prediction
> reverses the sign of the measured joint effect, a residual of −0.054 against a frozen 0.05 margin,
> which we report as **not decisive** rather than as an established interaction. Either way we claim
> no per-channel decomposition of ΔASR anywhere in this paper.

The withdrawal it licenses is unchanged and still stands: no per-channel decomposition of ΔASR is
claimed anywhere, and the re-selection/rescaling split is presented as a decomposition of a
*displacement* only. We also confirmed that appendix (L1)'s "not decisive" sentence still agrees with
the body after the edit.

**5. The 0.173 witness leans on its p-value.** Adopted, in your words. The rung is now called a
**sign-consistent witness**, and the p-value is labelled as supportive rather than load-bearing:

> the first accuracy-valid rung is a **sign-consistent witness** of a 0.173 suppression change under
> bit-identical selection and admission (one-sided p = 0.044, 5/8 seeds, *supportive* rather than what
> the identification claim rests on, which is the bit-identical antecedent)

That is also the honest description of the design: what the identification claim rests on is the
antecedent being exactly zero, which the figure generator now asserts at all four rungs for both
channels rather than assuming.

**6. "Admission" is overloaded.** Priority 2 above.

**7. Title and abstract.** Adopted, including your suggested title. It is now *What Does Preserving a
Defense **Statistic** Preserve? Causal Identification for Federated-Learning Defense Composition*,
which is one word inserted rather than a new title, so the paper stays findable against the earlier
version. The statistic/mechanism distinction has moved into the abstract's first three sentences:
"upstream processing can destroy the *statistic* the downstream defense reads. The natural repair is
to require the upstream stage to **preserve that statistic**. For *positive per-client rescalings*,
where invariance is settled algebraically, **preserving the statistic is not preserving the
defense**." We did not rewrite the abstract, because it carries 14 numeric literals that all have to
survive.

**8. Over-defensive.** Adopted, and this is the largest deletion. The body's `(L1)-(L6)` enumerate is
**gone**, 2470 characters of it, replaced by a pointer: the caveats are inventoried once, by kind of
claim in Table A2 and item by item in the Limitations appendix, and "no claim above is hedged twice
and none is hedged nowhere, which is what a single inventory is for."

**Relocate-never-delete applies to disclosures, so we gated the cut on a per-item conservation check
rather than assuming the appendix already said it.** Four of the six items were verified conserved
elsewhere and cut. Two were not, and were moved first, which we name rather than implying they had
always been there:

- into appendix (L1): *"we ran neither DBA nor Neurotoxin through the intervention"*, *"the score-only
  control that closes the magnitude channel is a single arm, one aggregator and one attack"*, and
  *"the pre-registered PASS predictions hold at N = 100/ResNet18 but far from converged accuracy, so
  they bound nothing about converged training, and the negative was never measured at that scale"*;
- into appendix (L5), which also gained a title: *"neither ρ < r\* nor Δ'_sep > 2(R'_B + δ') is
  readable off a (d1, d2) definition pair: both need a pilot measurement."*

We then verified mechanically that no numeric literal was lost: 0 of the source's distinct literals
disappeared, and every literal whose count fell (48, 60, 3.24, 1.01, 40, 0.239, 26 and others) was a
duplicate we located at its surviving sites individually.

**9. The foundation-model reach.** Removed, at both sites. §1 no longer ends a sentence with "and it
bites hardest in decentralized pipelines, foundation models included"; it now stops at "with no single
owner". The assert-then-disclaim pair is collapsed into one clause: "The identification problem we
isolate is defined independently of model scale, so we study it in a controlled federated setting and
claim no transfer beyond it." The trustless/decentralized motivation is load-bearing for the framing
and is kept; only the reach to foundation-model training is gone.

**10. Position Related Work around causal evaluation.** Adopted as your three moves, and the third one
was genuinely missing:

> **Where the novelty sits, as three moves.** *Standard*, and textbook outside this literature:
> conditioning on a common descendant of treatment and outcome destroys identification. *New*: an
> inherited-suppression protocol does not merely risk that structure, it **necessarily creates** it,
> by the constitutive argument just given, while reading as a reasonableness filter (only study
> defenses that work) rather than as the step that removes identification. That is the reading we held
> ourselves. *Further new*: the repairing intervention then exhibits **empirical non-implications
> between adjacent preservation levels**, (P3) ⇏ (P4) and (P4) ⇏ (P5), which the collider argument
> does not predict and which is what makes the hierarchy an empirical object rather than a taxonomy.

---

## Seven defects your report did not have to reach, and one presentation defect

We found these while doing the above, and none of them is visible to a clean LaTeX log, an
undefined-reference check or a duplicate-label check. All are fixed.

1. **Two different `(L1)-(L6)` lists shared the same tags.** The body list (witnesses narrow / attack
   fixed / (P4) per-family / scale is a prediction / theorem conditions / screen not validated) and the
   appendix Limitations list (limited scope / criterion is a prioritizer / per-round switching only /
   no falsifiable prospective positive / Thm (1) not checkable / disturbance not transferable) agreed
   on item 5 alone. Every `(L·)` reference in the paper resolved to whichever list the reader guessed.
   **Neither list defines its tags with `\label`; both print them as bold text**, which is precisely
   why no reference check could see the collision. Deleting the body list for weakness 8 dissolved it,
   and exactly one list now survives.
2. **Two references to a nonexistent `(L7)`**, both inside Table A2, the float Priority 1 promotes.
   Rewired to the items whose content they describe: the recall clause to (L2), where the 40% ceiling
   is discussed, and the C0 clause to (L4), where C0's retrospective formulation is disclosed. A third
   collision on the same tag, `(L2)` used to mean the L2 *norm*, is now `ℓ₂`.
3. **One `(L·)` reference pointed at the wrong item under both lists.** Table A2's "the γ ≈ 0.66
   boundary is architecture-conditional (L3, L6)" is a claim about architecture-conditional boundaries,
   which is (L1) under the surviving list and was neither L3 nor L6 under either. Now `(L1)`. We found
   this only by reading all sixteen `(L·)` sites against the item text, not by counting them.
4. **One surviving `(L·)` reference pointed at an item that did not state the claim it was cited for.**
   §2's theorem paragraph ends "the limit at which the bound goes vacuous (L5)", but appendix (L5) was
   entirely about the *checkability* of the two conditions and never mentioned that the geometric
   margin goes empty under extreme heterogeneity. The vacuity is measured and stated in three other
   appendix places, so nothing was missing from the paper; what was missing was that the item the body
   sends the reader to said it. **We extended (L5) rather than rewiring the reference**, because
   "conditions that are not merely awkward to check but empty in the regime that matters" is the same
   limitation, and (L5) is where a reader looks for it: it now carries the 3.24 → 1.01 collapse at
   Dirichlet α = 0.1 and the point at which Corollary 2 certifies nothing, with pointers to §C.2 and
   §D.4(E) where both are measured. This is appendix-only; the `.aux` diff after it shows zero page
   moves and zero label changes.
5. **A round count in the body was attributed to a population it does not cover.** The sentence
   defining the Δ adm. column said the support "never moves in any of the 60 measured rounds of any
   arm". 60 is right for each CIFAR-10 arm (4 rungs × 15 rounds) and wrong for the FEMNIST
   replication, which has 14 rounds per rung and so 56. The measured fact is unchanged, 0 changed
   rounds everywhere, but "of any arm" made a per-arm count universal. It now reads "in any measured
   round of any arm (60 per CIFAR-10 arm, 56 on the FEMNIST replication)". We found this by reading
   the body page against the generator's own per-row `n_rounds`, which is the only place the 14 is
   visible.
6. **The companion supplement's title was one revision stale.** Its title page still read *What Does
   Preserving a Defense Preserve?*, without the word *Statistic* that Priority 2 inserted into the main
   title, so the two documents disagreed on their own name on their own first pages. Both titles are now
   byte-identical after normalizing the manual line breaks. Nothing in either build could catch this:
   `xr` couples the two documents through labels and object numbers, never through the title.
7. **The supplement cited a work with no reference list of its own.** Its DBA transfer section carries
   the paper's only supplement-side `\citep`, and the supplement had no bibliography, yet it rendered
   "(Xie et al., 2020)" correctly and produced no warning. The reason is that `xr` reading `main.aux`
   imports natbib's `\bibcite` entries along with the labels, so the key resolved off the main paper's
   bibliography invisibly. Adding a bibliography to the supplement made the key *multiply* defined, so
   we did the opposite: the supplement's opening note now states that citations, like unprefixed section
   and table numbers, refer to the main paper and that their full entries are in its reference list. The
   dependency was already total, since all of the supplement's cross-document references resolve the
   same way; it is now declared to the reader instead of being accidental.
8. **A cross-reference printed "App. D, D".** Two labels sat on the same appendix section, and one
   sentence referenced both. The redundant reference and the alias label are removed.

We also caught two presentation defects of our own making before they shipped. The lengthened title
hyphenated across the line break as "Pre-serve?"; the manual line breaks are rebalanced into three
even lines. And the near-miss on the pooled figure label is described under Priority 3 above.

## Verification of this revision

Verified on a clean build, because an incremental build once reported 46 pages here when the truth was
45: **the Ethics statement heading is still on p9**, against the ICLR body limit; 45 pages total plus
the 8 page companion supplement. Both documents build with 0 errors, 0 undefined references, 0
multiply-defined labels, 0 overfull boxes and no `??` in the output. Float placement was checked by
diffing the `.aux` against a pre-revision copy rather than by trusting a clean log: **no object number
changed and no label was added or lost** apart from the removed alias, with three appendix page shifts
as the only difference. Figure 1 was rendered at 300 dpi and inspected visually, since annotation
collision is that generator's own documented failure mode and no text check can see it. Anonymity
re-checked after the title change, including that the PDF's Title and Author metadata are still empty.

## What we have not done

- **No new experiments, arms, seeds, datasets, attacks or defenses**, per your own non-goals. Nothing
  was rescored and no frozen number moved.
- **No code or data package.** The submission's statement that no code or data package accompanies it
  remains literally true, and we would rather leave it true than promise an artifact we are not
  releasing at submission time.
- **We did not restore the persistence and mixing stratum**, which an earlier reviewer called the
  strongest part of the paper. It stays relocated to the appendix and supplement, intact.
- **The page count is at the limit, not comfortably inside it.** The revision is net negative in body
  characters by design: the 2470 character deletion funds the tier paragraph, and the appendix
  relocations cost body pages nothing. Any further body addition still requires a compensating cut.
- **The factorial remains not decisive at n = 5.** We softened the claim it supported and did not run
  more seeds to rescue it, because low power is a reason to stop asserting a decomposition and never a
  reason to assert one.
