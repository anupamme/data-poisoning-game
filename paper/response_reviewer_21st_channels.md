# Response to the twenty-first review (5/10, borderline / weak reject, confidence 4/5)

Thank you for naming, precisely, the test that would decide this paper:

> *the paper needs to convince the reviewer that the observed sign reversal is genuinely about
> identification of mechanism, rather than simply another manifestation of the fact that the upstream
> intervention changes the malicious clients' effective contribution.*

We ran that test on the cell the paper is built on. **It went against us, and §0 below is what it
returned.** Your *Toward Strong Reject* branch is therefore the one this round reports on, not the
*Toward Accept* branch, and we would rather you read it from us than find it later.

Source line cites in this letter are `main.tex` **source** lines, or `supplementary.tex:NNN`. They are
not the PDF's margin numbers, which count rendered lines and do not coincide with these. Everything
below is this round's diff against the version you reviewed; where a fact predates your review we say
so rather than claim it.

---

## 0. Your most important remaining issue, run as a pre-registered control, and it fired

Your §16 names four candidate confounds for the reversal, and puts *differences in adversarial
weighting* first. The paper's answer at the time was that Mode S pins the adversarial coefficient, so
weighting is held. That answer was **incomplete on exactly the channel your §4 quotes from Table 2**:
Mode S pins the coefficients, and the *aggregated update's magnitude and geometry* still move.

So this round closed that channel on the headline cell. One rung, twenty seeds, frozen before the
instrument existed, with all three branches written down in advance.

**The result: the within-defense leg reverses its own sign.** "$\Delta_B = \mathbf{-0.1722}$, paired $95\%$ CI $[-0.2119, -0.1325]$, which excludes zero on the negative side." `:2457`, against the published within-defense $+0.1251$ on the same twenty seeds. The secondary
estimand is "$\Delta_{\mathrm{mag}} = \mathbf{+0.2973}$, CI $[+0.2527, +0.3418]$, mean ASR $0.6034$ uncontrolled against $0.3061$ controlled" `:2457`.

And the consequence for the claim you were probing, stated as the freeze required rather than as we
would have preferred: "the published reversal does not reproduce with the magnitude channel closed" `:2459`. Set beside both published legs, "outcome-gated $-0.2733$ $[-0.3337,-0.2128]$ and within-defense $+0.1251$ $[+0.0954,+0.1547]$, $\Delta_B$ overlaps neither" `:2459`, and we
refuse to lean on that non-overlap, because "its non-overlap with the outcome-gated interval is by $0.0009$" `:2459`. What the cell now shows is two designs agreeing in **sign** and differing in
magnitude by an amount twenty seeds barely resolve.

Two things we will not do with this number. We do not call it a decomposition: it is "\textbf{not} described as the magnitude channel's share of $\Delta$ and is not summed with anything" `:2457`,
because no $2\times2$ factorial exists on this cell. And we do not let it revise the formal results,
which it does not touch: "Proposition~\ref{prop:nonidentifiability}'s non-identification and the exact invariances are algebra about what an outcome-gated test can distinguish" `:2461`.

**The freeze also recorded a prior, and the prior was wrong.** "The freeze's own stated prior was that this arm was more likely to confirm than to refute" `:2463`, on the arithmetic that refuting
"needed the magnitude channel to account for roughly $0.095$ of the $0.125$ rise, and the measured effect was three times that" `:2463`. We print it as written, "because a pre-registration that records only the priors it got right is not one" `:2463`.

**The mirror arm, on a selector, goes the other way, and that is the honest shape of the finding.** The
published score-only Krum control was extended to the same twenty seeds: "$\Delta = \mathbf{-0.0088}$ at $n{=}20$, paired $95\%$ CI $[-0.0335, +0.0160]$" `:2123`, inside the frozen $\pm0.15$ and not
excluding zero, against $-0.010$ uncontrolled. So: "The channel is inert on a selector and decisive on an order statistic, and neither result transfers." `:979` We state that as a dissociation across
aggregators, not as one result at two seed counts `:2463`.

This is now in the abstract, against us: "With the aggregate magnitude held fixed the $+0.125$ is $-0.172$ and both agree in sign." `:100` In §1's opening evidence: "a pre-registered control that holds it fixed on the first row's cell turns the within-defense $+0.125$ into $-0.172$" `:181`. And as
a row of the page-2 claims table, with its status in your vocabulary: "The sign disagreement is not a magnitude artifact & Measured, $n{=}20$ & refuted, by us" `:351`.

---

## 1. §15.A: reframe the central contribution

Done at the title, which is now your sentence rather than ours: "Statistic Preservation Does Not Identify" `:54` mechanism preservation in composed FL defenses. The abstract's first sentence is the
identification claim and no longer an aphorism: "Whether the downstream mechanism caused a composition's suppression is unidentified by any test admitting it because a constituent suppresses the attack." `:100`

The screen is demoted in the same motion, as its own table row: "Our screen predicts a menu's best compositions & Retrospective & not established" `:352`. See §8 below for the rest of that demotion.

## 2. §15.B: make the causal estimand absolutely explicit

Three changes, because your ask has three parts.

**The diagram.** Figure 1(a) is now that diagram, and its caption says what the arrows mean: "The chain a preservation check reasons along, and the map of this paper's vocabulary, at the levels of Def.~\ref{def:levels}: solid links hold by definition, each dashed link is marked $\nRightarrow$ for not implied, and the lower arc is the attenuation channel Mode~S cuts by construction." `:269`

**The estimand, written down.** "The causal quantity is $\tau(T)=\mathbb{E}[Y\mid\mathrm{do}(T),d_2,a]-\mathbb{E}[Y\mid\mathrm{do}(T{=}\mathrm{id}),d_2,a]$." `:532`

**Total effect against mechanism-specific effect**, which is your §16.3 and the part we think was
genuinely missing: "$\tau(t)$ is the total effect of the assigned transform at that one $(d_2,a)$: it separates into a statistic channel and an admission channel only when one of them is pinned by construction" `:1601`. And a new paragraph separates the three layers you asked us to separate,
in your order: "What is general is the estimand, an intervention on the upstream transform" `:506`.

## 3. §15.C: stop calling Mode S a clean causal intervention

**Conceded in substance, declined as a rename**, and we want to be explicit about which is which.

The substance is a paragraph that now states your objection before any reader can form the wrong
impression: "Mode~S identifies the intervention's effect at a fixed adversarial coefficient share, not statistic disturbance in isolation, so what it yields is a \emph{dissociation}, not proof that disturbance is inert." `:974` It then names the open channel and quantifies it with the number your
§4 quotes: "It pins the adversarial coefficients alone, so the dose still moves the aggregated update's magnitude and geometry, a third channel reading $\Delta$~agg.$={}0.892$ here." `:974`

**The rename we decline**, for one reason: *Mode S* names the instrument in over a hundred places
across the two documents and in $25$ of the $38$ pre-registration documents, so renaming it in prose
would make the paper disagree with the freezes and artifacts that fixed its rules, while changing no
claim. What the label *means* is now stated where the label is introduced, which is the part that was
actually missing.

**One correction to your reading of Table 2, because the table invited it.** You quote Krum's
$\Delta$ aggregate of $0.892$ as evidence the channel is large. The score-only row reads $0.986$ (larger),
and that is the channel *closed*, not opened. The caption now says why: "$0.892$ is the vector sum of that same $0.986$ re-selection component and a $0.973$ rescaling component, which partly cancel, so a larger number in this column is a \emph{narrower} channel and not a wider one" `:2058`. A
reader who compared the two rows and concluded the control made things worse was reading the table as
printed, and that was our defect rather than theirs.

## 4. §15.D: put the oracle limitation in the abstract

It was already there in the version you read, and we checked rather than assumed it: "The first two need an oracle no server has: identification, not a deployable test." `:100` We have not strengthened
it, because we could not find a shorter form that keeps *identification* and *not deployable* in the
same clause inside the abstract's frozen character budget.

What did change is that the oracle is now load-bearing in a second place. Arm B removes none of it:
"The oracle is untouched." `:2463`

## 5. §15.E: make the adaptive-attacker result central

Moved into the body's threat-model paragraph, with the number and the boundary in the same breath:
"raises FG$\to$RFA to ASR $0.157 \pm 0.073$ at $n{=}30$, $2.4\times$ its seed-matched $0.064$" `:780`, and then, in your own framing, "Inherited robustness is not adaptive robustness" `:780`, with
the concession that "a protocol stated against an adaptive adversary is the sharpest thing this framework lacks" `:780`.

## 6. §15.F: reduce the literature-audit emphasis

The audit's rubric, retrieval frame and all $59$ coded rows are in the supplement. Four other blocks
went with it, and the body now carries a pointer so the material is findable rather than merely gone:
"Five blocks are in the supplement rather than here, and this is where a reader looking for them finds them." `:1430`

The criterion we applied is narrower than *least important*, and we state it: "What moved them is that no claim in the body is stated only there, not that any of them is unimportant" `:1430`. One of the
five is explicitly not skippable, and saying so is the price of moving it: the equivalence testing,
"read it if you doubt the margin, since it is where each arm's binding $m^\ast$ and the one-sided bound behind every non-increase are computed" `:1430`.

## 7. §15.G and §16.1: a non-oracle intervention on a second dataset

This is the ask we cannot answer with a result, and we will not dress up what we have. It is a
different ask from your §16.2, which we *can* answer: §8 below reports a composition menu on a second
dataset. What is missing here is specifically the *oracle-free upstream intervention* on a second
dataset, and no arm delivers that.

The arm you describe (a second dataset, same downstream defense, same attack, upstream intervention,
no adversary identity, independently pre-registered, $n\ge20$) is pre-registered at `9cbe359`, and its
instrument's harness check passed all five of its checks, with the verdict persisted in the artifact
rather than printed to a terminal. One naming note before the numbers, because the ask says FEMNIST and
the paper says otherwise: our `femnist` dataset key loads torchvision `EMNIST(split="byclass")` under
this paper's own label-Dirichlet split, not LEAF FEMNIST's partition by writer, so every table and every
sentence reads EMNIST-byclass and only the artifact path, the pre-registration filename and the JSON key
keep the older spelling `:2078`. Those five checks cover three of the freeze's six numbered gates: gate 1
(bit-identity) recomputed a published EMNIST-byclass cell two independent ways and returned a difference
of exactly `(+0.00e+00, +0.00e+00)` on both; gate 2 (the bisection invariant) and gate 3 (the share gate)
were measured live on EMNIST-byclass update stacks, where all $3$ of $3$ crossing rounds flipped the shipped
selection and the supremum coefficient-share gap over all $30$ nonempty proper participant subsets
reached $8.727\times10^{-8}$ against a `SHARE_TOL` of $10^{-6}$, on seed 42.

We are precise about which gates those are because the count invites a wrong reading. **The freeze's
gates 4, 5 and 6 (crossing rounds per run, the accuracy gate, and the ceiling/floor gate) are gates
on the scored contrast, and none of the three is cleared.** Gate 4 records crossing rounds per run and
it did that for the 11 runs that exist and for none of arm B's; gates 5 and 6 were never evaluated,
because each is a condition on both arms' means. Five checks passed; three of six gates are cleared;
the arm is not gated through.

**The contrast is not reported, because the arm did not reach its pre-registered $n{=}20$ on both
arms.** Its freeze fixes $n{=}20$ per arm and contains no stop-early provision, and the arm's own
banner forbids a mixed-$n$ comparison, so there is no pre-specified form in which a partial version of
it could be scored. We compute no $\Delta$ at any $n$ from it and we report no single leg as a
finding. What exists today is an instrument with a passing harness check and an unrun contrast, and the paper
says exactly that.

For the three attempts that *are* reported, the honest tally is two against us and one premise
failure: "A second, RFA into Krum at $n{=}20$, refuted our flatness prediction ($+0.297$), attenuation left open" `:1052`, and a third, oracle-free at coefficient share $10^{-7}$, where "a decision flip alone carries $+0.322$, so the negative is not established without an oracle" `:1052`.

## 8. §8, §16.2 and §16.5: a more restrained claim about the screening criterion, and it now fails on a second dataset

We agree with your reading, including the part that is against the paper, and the text now leads with
it where the number is computed: "Read against its base rate that figure is nearly uninformative, which is why it appears here, where it is computed, and in no summarizing position." `:2523` The
constant predictor sits beside it in the table itself, not in a footnote: "Constant-HIGH baseline on the same 42 pairs: 37/42 = 88.1\%." `:2541`

And §6.3 goes further than demotion, to a negative recommendation: "do not screen on inherited suppression at all" `:1187`, because "what it discards is FG$\to$RFA, the menu's lowest-ASR composition" `:1187`.

**Your §16.2 asked for the composition menu on a second dataset, and that arm ran. It went against
the screen harder than the CIFAR-10 arm did.** The 42-pair menu was re-run on CIFAR-100 with
`cifar_cnn` against both committed attacks, pre-registered and committed alone at `f559b85`, capped by
its own freeze to wave 1 before the first run. The result is not a lower agreement figure; it is that
the screen never reaches a LOW prediction for any pair at all. "on CIFAR-100 \textbf{no single defense suppresses the committed pixel backdoor alone}, and a pair is predicted LOW only when C1 is met for \emph{every} committed attack" `:2779`, so "the screen returns HIGH on every scored pair and is therefore the constant-HIGH predictor on this menu cell for cell rather than merely equal to it in aggregate" `:2779`.

The figures, with the base rate in the same sentence as the freeze requires: "The screen agrees with the observed class on $7$ of the $9$ scored pairs, $77.8\%$, against a constant-HIGH predictor's $7$ of $9$ on the same pairs, $77.8\%$, which is also the base rate, with $2$ of the $11$ complete pairs excluded below the floor and $0$ excluded for a missing baseline." `:2785` Those two figures are equal by
construction rather than by coincidence, and we say so rather than letting a reader treat the
coincidence as informative. "Precision on the LOW class is undefined because the screen predicts LOW for no pair at all, and recall on the LOW class is $0$" `:2785`.

**Why it fails is a measurement, and the failure is attack-specific rather than dataset-wide.** The
seven aggregators the screen reads "give fourteen standalone readings, and \textbf{exactly one falls below the screen's own C1 threshold of $0.3$}" `:2787`: `reputation` against committed scaling. The
attack with no standalone suppressor at all is the pixel backdoor, "its seven readings run from \texttt{coord\_median}'s $0.673$ to $0.888$" `:2787`, and because a pair is predicted LOW only when C1
is met for every committed attack, that one scaling suppressor cannot yield a LOW pair. **We report the
part of the precondition that held rather than absorbing it into the headline:** "\texttt{reputation} does suppress committed scaling alone, at ASR $0.005$" `:2779`, and on two scored pairs C1 and C2 both
hold for that attack, so "the screen reaches a LOW per-attack prediction twice and still predicts no LOW pair" `:2779`.
This is the same shape as EMNIST-byclass, where the minimum single-defense pixel ASR is $0.609$. Both
datasets that leave CIFAR-10 at the composition level fail C1's precondition **on the same attack**, so
this is a scope condition on the criterion rather than an artifact of one menu, and §6.3's negative
recommendation now rests on two datasets rather than one.

**What this arm is not.** It is not the full 42-pair menu and the paper never calls it one: "the freeze capped the arm to wave~1's $18$ pairs before the first run, so wave~2's $24$ held-out pairs are deliberately unrun and no held-out generalization claim rests on this arm" `:2781`. Two complete pairs
sit below the frozen clean-accuracy floor and are counted rather than dropped, with their accuracies
printed `:2783`. One thing departed from the frozen run order (the pairs were taken in LOW-first
slices rather than list order, so that the menu's lowest-ASR composition was reached before the
deadline), and we disclose it as a deviation with the determinism check that licenses recombining the
slices, rather than amending the freeze `:2789`.

## 9. §12: the statistical concerns

**Small $n$.** The two arms this round adds are both $n{=}20$ on exactly seeds $42$--$61$, which is
the seed count of the headline reversal, so the new evidence is not the thin part of the paper. The
$n{=}3$ results you flag are unchanged and still labelled row by row; we did not top any of them up
this round and we do not claim to have.

**Multiplicity.** Both new arms were frozen with their branches named in advance, and in both cases
the branch that fired is the adverse one, which is the only evidence we can offer that the freezes
are doing work rather than decorating a search. Arm B's freeze additionally recorded a wrong prior and
we printed it `:2463`.

**Equivalence margins.** No margin test is run on Arm B, deliberately, and the reason is a rule and
not a convenience: its claim is about a sign and a size rather than a non-increase, and "Importing a margin afterwards would be choosing a rule after the numbers." `:2455`

## 10. §13: length

Partly acted on and partly declined, with the accounting stated rather than implied.

Acted on: five blocks left the main paper for the supplement (§6 above), the contributions list became
a table, and the main text's body ends by page 9 under a gate that measures its last rendered line.

Declined: we did not delete a scope condition, limitation, withdrawal or disclosure to buy pages. Each
moved block keeps every number it had and has a pointer from its old home. Your 30--40% figure is a
reduction we cannot reach without deleting disclosures, and between a long paper and a shorter one
that has quietly dropped a caveat we choose the long one. If there is a *specific* block whose removal
you would accept, we would act on it directly.

---

## 11. What we decline, and why

**The rename of Mode S** (§3 above): the label is not the claim, every characterization of it is now
yours, and renaming it would desynchronize the prose from the artifacts and the freezes.

**Any statement that the negative holds or fails without an oracle.** Your §5 is conceded and not
closed. Three oracle-free attempts exist; the second and third went against us; the fourth has a
passing harness check and no contrast. One aggregator, one attack and one dataset license no general statement
in either direction, and the paper refuses both: they "license no statement of the form \emph{the negative fails without an oracle} in general, any more than they would have licensed its converse" `:2386`.

**Reading Arm B as a magnitude decomposition of the reversal** (§0 above). It is one of four cells of a
factorial we did not run, and the paper already withdraws channel-split-as-decomposition readings.

---

## 12. On the score

Your §16 offered two branches and we report on the one that went against us. Of your §15, A, B, D, E
and F are done; C is conceded in substance and declined as a rename; G is pre-registered with a passing
harness check, three of six gates cleared and an unrun contrast, and we say so plainly. Of your §16's
five, the second is run and reported against us (§8 above), the third is now explicit in the text, the
fifth is done, and the first is the one compute did not deliver.

We think the paper is now more clearly a negative result than it was: an outcome-gated design cannot
identify mechanism attribution, a within-defense design can reach the opposite sign, and on the one
cell where we closed the channel you named, the opposite sign was carried by that channel rather than
by the statistic. That last sentence costs us the headline we had. It is also the first thing in this
paper we would believe if someone else wrote it.
