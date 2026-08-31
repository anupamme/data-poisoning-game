# Response to Reviewer (5/10, Weak Reject; confidence 4/5)

Thank you for a report that was harder to answer than a higher score would have been. Your §16
diagnosis, that the strongest paper inside the manuscript was a causal methodology rather than a
criterion, is the one we acted on, and it restructured the submission. We also adopted your suggested
title verbatim, your §19 changes #1 through #5, and your recommendation in W9 about the word
"security."

We have not adopted everything, and we say where below. Two of your suggested experiments were not
run.

---

## The four changes that restructured the paper

1. **The identification result is the main contribution** (§19 #1). The paper leads with it, the
   criterion is a downstream consequence, and the abstract's central claim is that no screen of this
   form identifies the mechanism's contribution **at any sample size**.
2. **The criterion is replaced by an intervention protocol** (§19 #2). The paper's stated methodological
   output is: do not select across defenses; hold the downstream defense and attack fixed, intervene on
   the upstream transformation, and measure the channels separately.
3. **90.5% is no longer sold as validation** (§19 #4). It is absent from the abstract and appears only
   where it is computed, beside its base rate.
4. **The paper was cut by more than your 40-50% target** (§19 #5, W10). Main text is **9 pp** against
   ICLR's limit, from the 74 pp PDF you read. The criterion-validation stratum moved to an 8 pp
   companion supplement linked by `xr`, so cross-document numbers are read from the other document's
   `.aux` and cannot drift. **Nothing was deleted:** 665/665 distinct numeric literals were verified
   conserved across the split, none reduced in count.

Your suggested title, *What Does Preserving a Defense Preserve? Causal Identification for
Federated-Learning Defense Composition*, is the paper's title.

---

## W1. Novelty: what is new beyond applying a known identification principle

This is the concern we can answer least comfortably, and we would rather sharpen it than resist it.

**We concede the collider argument is standard.** The paper says the proof is three lines and does not
present the selection-bias phenomenon as new. What we claim is new is not the principle but three
things downstream of it:

- **The design class is derived, not assumed.** Prop. 1 does not merely observe that outcome gating
  biases estimates; it identifies the cell that is *structurally absent* under an inherited-suppression
  gate and therefore characterises which experimental designs can and cannot identify a composition
  effect. That derived class is what licenses the interventions, and the paper is organised around that
  derivation rather than around the bias.
- **A second, non-collider boundary.** Prop. 2 (Coverage) is a distinct result: suppression that is
  emergent in a composition and absent from both constituents cannot satisfy an inherited-suppression
  gate, so recall is capped structurally at `1 − |E|/|S|`. It is not selection bias, and it converts
  the recall complaint of W7 into a scope theorem whose bound binds at exactly the reported 40%.
- **The identification result bounds other people's criteria too, including one that scooped us.**
  A literature check for this revision found `fenaux2025hammer` (arXiv 2509.08089), which *does* provide
  a predictive criterion for FL defense composition, categorising defenses by adversarial deviation.
  The paper previously claimed *"no prior work provides a predictive criterion for which FL defense
  compositions succeed."* **That claim is now explicitly withdrawn in the text.** Their criterion is a
  *design* theory where ours is an *evaluation* theory, and Prop. 1 applies to theirs and ours alike:
  it bounds what validating evidence of that form can establish, not whether either criterion is right.
  We also added `karimireddy2022byzantine` (Bucketing, ICLR 2022) as the closest structural precedent
  in FL, an upstream reshuffling step ahead of a fixed robust aggregator, which is literally our object
  and which the paper had failed to cite at all.

We do not claim this settles novelty for the main track. It is the sharpest answer we can make without
overstating.

## W2. The criterion is not validated

Adopted in full, as the paper's own position rather than as a caveat. Every item on your list survives
in the text: pooled R² = 0.239, cross-arm ordering refuted, C3 with no incremental predictive value
(and now demoted from a gate to a margin diagnostic), no emergent-synergy discovery, and an explicit
statement that the criterion is not a security screen. The criterion is presented as an evaluation
prioritizer whose predictive validity is disclaimed at first use.

Your sentence, *"the paper no longer has a strong positive methodological contribution,"* is the
sentence we tried to answer with the restructuring rather than with more validation: the positive
contribution is the protocol, and the criterion is what the protocol replaces.

## W3. The 90.5% figure

Adopted, including your specific recommendation to remove it from the abstract. It is not in the
abstract. Where it appears, the paper prints alongside it that a constant HIGH predictor scores
37/42 = 88.1% on the same 42 pairs at a 12% base rate, so 90.5% beats it by one pair and the tally is
carried by correct negatives; that **the tallies should not be read as discrimination**; and that the
informative quantities are precision 2/2 and recall 40%.

**Your third point, that the genuinely falsifiable evidence is much smaller, is now quantified rather
than acknowledged.** The paper's only prospective and only fully out-of-sample evidence is a 20-pair
suite over three never-used defenses (FLTrust, Krum, Multi-Krum), labels and mechanisms committed at
`262cf35` before any run existed and never revised, with a predicted-FAIL sanity gate. **It scores
13/20, and the paper does not offer that as validation of the screen.** The reason is printed in the
same breath: FLTrust suppresses both committed attacks alone at max-committed ASR 0.048, so all 13
pairs containing it land within 0.030 of FLTrust alone in either slot, and seven of those fall outside
a scope condition **we added retrospectively rather than pre-registering it**. Only the seven
FLTrust-free pairs could have surprised us (5/7), and none is predicted-LOW, so they bear on
specificity alone. Two further pre-registered validation attempts were defeated by the same confound,
which makes them evidence for Prop. 1 rather than for the screen.

## W4. (P4) has different operational definitions per defense

Accepted as a limitation rather than repaired by formalization, and stated as such: the four admission
instruments are not commensurable, so **the (P4) result is four within-family results, not one
cross-defense statement.** The paper says this in the limitations block and at each use, and (P4) is
described as a conceptual level instantiated defense-specifically (selected-client indicator /
coefficient share / coordinate share) rather than as a scalar.

We did not attempt the tighter cross-defense formalization you ask for. We do not currently know how to
define admission commensurably across selector, order-statistic and weighted-averager families, and
inventing a definition to satisfy the objection would be the kind of unvalidated construction the rest
of the paper argues against.

## W5. The Krum floor effect

Accepted, and now measured rather than acknowledged. Krum admits zero adversarial clients in 60 of 60
rounds **with zero adversarial mass present at baseline**, so its zero is a floor and the paper says
so. What makes the non-event a measured one is reputation and coordinate median, which carry
adversarial mass at baseline in **48 of 60 rounds each**. Pooled over the four CIFAR arms: 240 rounds,
0 with any change in the support of admitted adversarial mass, 100 with mass present. Both the pooled
count and the explicit statement that Krum's own zero is a floor appear at every site that reports it,
including the Figure 1 caption.

## W6. The score-only control, and the factorial you proposed

Your factorial table is the design we implemented, and it is **half-run**.

- The cell that exists is the pre-registered score-only Krum control: Krum scores the transformed stack
  and therefore makes the same selection, but the **untransformed** selected update is aggregated, so
  the decision channel is preserved and the magnitude and geometry change is removed. Rules were frozen
  at `35788d9` before any score-only ASR existed. Result: Δ = −0.023 (0.062 → 0.039) against −0.026
  uncontrolled on the identical cell, at the same decision change of 0.733, with the refuting threshold
  0.212 nowhere approached and every rung clearing the accuracy floor.
- The complementary emit-only cell exists only as code. Its harness is implemented and its thresholds
  are drafted (additive point prediction −0.0029; separability iff |ΔASR| < 0.05 and additivity
  residual < 0.05), but the pre-registration document is **not yet git-committed** and neither the
  harness check nor the arm has been run. By this paper's own standard that is not a pre-registration,
  so we do not call it one, and the paper claims nothing about the cell. Your point that 0.986 re-selection and 0.973 rescaling displacements are not causal
  decompositions in the interventional sense is correct, and that decomposition is now presented as
  algebra, not as a causal claim.

## W7. The positive result is internally undermined

This is now the paper's argument rather than an objection to it. The recall-40% complaint is a theorem
(Prop. 2, above), and FG→RFA is its witness with the bound binding at exactly 1/5. The paper's positive
claim is the protocol; the criterion's positive predictive direction is stated as not established.

## W8. Experimental diversity

Accepted as a limitation and stated first among them: the identification result involves no dataset,
architecture or scale, but **every witness for it here is CIFAR-10 with a 4-layer CNN, N=10, K=5,
f=0.2, two committed attacks**, replicated on EMNIST-byclass with a different CNN. We did not run DBA
or Neurotoxin through the intervention. The generality of the headline claim is reduced accordingly:
the abstract scopes the empirical claim to *"the broad class of positive per-client rescaling
compositions"* and the limitations say the empirical claims are not general even where the boundary
results are.

## W9. "Security" is the wrong word

Adopted verbatim. **`security preservation` occurs zero times in the paper.** The limitations state:
*"What we establish is attack-suppression preservation under a fixed committed attack; the protocol has
no clause for an adversary that reoptimizes against the pipeline."* The theorem is labelled a boundary
theorem and not a security theorem in the same breath as its statement, and the abstract says the
sufficient conditions certify **mechanism preservation, not security**, and bound no ASR.

The adaptive evaluation was **not** substantially expanded, and we are not claiming it was. What exists
is the one composition-aware white-box stress test you cite (0.051 → 0.167, a 3.3× rise), plus
projection-adaptive and criterion-aware joint-constraint arms, all presented as the boundary the
methodology does not cross rather than as evidence for it.

## W10. Length

See above: 9 pp main text plus an appendix, and an 8 pp supplement. On your specific observation that
the paper *"spends more space explaining why an experiment cannot establish something than presenting
the result"*: the caveats are now collected in a single numbered block with an explicit rule stated in
the text, *"so that no claim above is hedged twice and none is hedged nowhere."* The repetition you
counted was the target of that consolidation.

## W11. Terminology load

Partially addressed, and we will not claim more. The layers that left the body are real: `mixing` (2),
`Nash` (1) and `VoPD` (2) now occur only in the appendix, the persistence model is withdrawn, and
"security preservation" is retired. What remains is P1-P5, C0-C3, Mode S / Mode A, and the
committed-versus-adaptive distinction. **We did not reduce those four**, because each is load-bearing:
P1-P5 is the object of the negative result, C0-C3 is the criterion the identification result is about,
Mode S / Mode A are the two interventions, and the attack distinction is what W9 required us to make
explicit. We accept that this remains dense.

## W12. Claims framed more broadly than the evidence

Adopted. The title no longer leads with the broad negative, which is your §21 recommendation. The
abstract carries the scope restriction in its second sentence. Every claim in the paper is tagged
*general*, *formal* or *empirical* at the point it is made, with a scope table stating the condition
under which each transfers, and the limitations state the narrow reading you propose in substance.

## §15. Theorem 8's quantities are hard to estimate operationally

You wrote that the theory is elegant but its required quantities are difficult to measure, and that the
heterogeneity experiment shows the bound can go vacuous. **This revision found that the problem was
worse than you said, and fixes part of it.**

Part (2)'s condition `Δ'_sep > 2(R'_B + δ')` contains `δ' = ‖μ' − μ'_B‖`, the displacement caused by
including the adversarial points. Evaluating it **requires running the composition the criterion is
supposed to screen**. In a screening criterion that is a defect, not an inconvenience.

**Corollary 1 (new) removes it.** Since the geometric median minimises `F(x) = Σ_i ‖x − u'_i‖`,
comparing `F(μ') ≤ F(μ'_B)` gives `(n_b − n_a) δ' ≤ 2 S'_B`, hence `δ' ≤ C_A R̄'_B` with
`C_A := 2 n_b / (n_b − n_a)` and `R̄'_B` the **mean** transformed benign residual. The condition holds
whenever `Δ'_sep > 2 R'_B + 2 C_A R̄'_B`, in which no quantity comes from the composed aggregate.
Honest majority `n_b > n_a` is exactly what this needs, and it enters on the displacement rather than
on the weight ratio.

Measured at α = 0.5 over the 17 positive-weight rounds: the displacement bound holds 17/17,
conservative by at least 10.5× (mean 215×); the condition is satisfied in 16/17, being 15/15 at
`n_a = 1` and 1/2 at `n_a = 2` where `C_A = 6.00` binds. **At α = 0.1 it holds in 5/12**, so your
vacuity observation survives the fix and the paper says so rather than reporting only α = 0.5.

We also attempted the fully pilot-free version, a threshold on `ρ` alone, and **it is empty on our
data**: it requires `Δ_sep > 4(1+C_A) R_B`, the measured ratio reaches at most 1.015 of that, so
`ρ* ≤ 1` in 16/17 rounds at α = 0.5 and in all 12 at α = 0.1 while measured `ρ` never falls below
1.034 and 1.015 respectively. No `ρ` satisfies it. Reported as vacuous.

**One consequence went against us and is now in the paper.** The operational protocol previously
claimed the theorem-backed class needed *"no calibration runs."* For the rank aggregators that holds.
**For the geometric case it was false**, and the paper states that it was false: Corollary 1 removes the
unscreenable quantity, but the benign spread replacing it is measured on transformed points, so RFA
carries the same short-pilot cost as every other family and is no longer exempted.

All three inequalities are validated independently of any FL run on 4000 random configurations
(`experiments/verify_margin_inequalities.py`): 4000/4000 on each.

---

## §20. What we did not run, and why

- **Experiment A (clean factorial): half done.** The score-only cell is run and genuinely
  pre-registered (frozen at `35788d9`). The emit-only cell is implemented but unrun, and its
  pre-registration is not yet committed. We report the cell we have and claim nothing about the other.
- **Experiment B (multiple attacks through the intervention): not run.** Estimated ~18 h on this
  hardware. One Mode-S arm measures 27 minutes per run on this machine, so this is a real cost rather
  than a scheduling excuse.
- **Experiment C (four downstream defenses under an identical design): not run.** Estimated ~40 h plus
  a harness generalization the current runner does not support.
- **Experiment D (adaptive attacker): partially, and framed as you asked.** The existing adaptive arms
  are presented as the boundary of what the methodology can say, not as validation.

We weighed B and C against your own instruction, *"I would not recommend adding another 30
experiments... The problem isn't lack of experiments,"* and against the record that four consecutive
prior rounds of added experiments did not move the score. We chose the restructuring you prioritised
over the compute. If the committee's view is that B or C is decisive, we would rather be told that than
guess.

## One more thing we changed that no reviewer asked for

The re-centering left behind **orphaned claims**: prose whose supporting evidence had moved out of the
document, including a reference to a "Heuristic 1" that existed in neither the paper nor its
supplement, two symbols used but never defined, and a limitation scoping an experiment the draft did
not contain. None of these were caught by cross-reference checking, which reported zero undefined
references throughout. They are fixed by retraction where the claim was unsupported and by definition
where the object was real, and every retraction is written into the text rather than performed
silently.

The reproducibility statement was also corrected: it previously promised a supplementary code and
artifact package. **No code or data package accompanies this submission**, and the statement now says
so, describes what the paper itself provides (pre-registration commit hashes, runners that refuse to
start unless their recorded commit matches, generator-emitted tables, per-seed values), and states that
code and artifacts will be released on publication.
