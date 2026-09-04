# Response to Reviewer (6/10, Weak Accept / Borderline; moderate confidence)

Thank you for a detailed and technically specific report. Concern #2 in particular identified a real
defect in the theorem, and it has changed the paper.

## 0. A version mismatch we should state first

Your report names the file it read: `main(20260809-151339).pdf`. That draft is two revisions behind the
current submission, and several of the things you describe are no longer in the paper. We state this as
fact, not as an argument that the concerns were invalid: five of your eight concerns were live in the
draft you read, three of them are things we had already changed, and #2 was a genuine error that a
later revision fixed independently of your report.

| What the report describes | The current draft |
|---|---|
| `PASS precision: 100%` **in the abstract** | Absent from the abstract. The phrase survives at exactly one appendix site, where it is computed, printed beside its base rate and the constant-HIGH baseline it beats by one pair. |
| Main text ~12 pp, references beginning on p13 | Ethics statement heading on **p9**; references precede `\appendix`. 45 pp total including appendix, plus an 8 pp companion supplement. |
| Title containing *"Under Persistence"* | *What Does Preserving a Defense Statistic Preserve? Causal Identification for Federated-Learning Defense Composition* (the word *Statistic* was inserted after a later review) |
| Theorem **2** proves the RFA case via weighted majority, `ρ < n_b/n_a` | The object is now **Theorem 8**; its part (2) assumes `Δ'_sep > 2(R'_B + δ')`, and its proof says explicitly that no weighted-majority assumption is used or would be appropriate. |
| Persistence / VoPD / Nash / temporal mixing as co-primary contributions | `mixing` appears 2 times, `Nash` 1, `VoPD` 2, all in the appendix. The body declares temporal mixing out of scope. |

The renumbering of the theorem from 2 to 8 is itself a useful check: the object you are describing sat
in a body section that no longer exists in that form.

**One thing we want to say plainly, because it cuts against us.** You called the persistence and
mixing story *"probably the strongest part of the paper"* and scored Empirical 8/10 substantially on
it. A second reviewer, at confidence 4/5, priced their score on removing exactly that stratum and
re-centering the paper on the causal identification result. We took the re-centering, because it was
the review of the later draft. **The persistence and mixing results were relocated, not retracted.**
They are intact in the appendix and the companion supplement, every number preserved, and we can
restore any of them to the body if the committee prefers your reading of the paper's center of
gravity. We are aware that the two reviews ask for different papers, and we would rather say so than
pretend the choice was costless.

---

## Concern #2 (Theorem 2 / now Theorem 8, geometric-median step): you were right, and we went further

This is the concern that changed the paper, so we take it first.

Your diagnosis was that step 3 does not follow: weighted majority `ρ < n_b/n_a` does not establish
that the adversarial update retains the largest geometric-median residual, and what is needed is
explicit benign/adversarial separation plus a bound on benign spread. That is correct. A revision
between the draft you read and this one had already replaced the weight-ratio condition with your
**Option A**: part (2) now assumes the transformed-point separation condition
`Δ'_sep > 2(R'_B + δ')`, and the proof states that no weighted-majority assumption is used.

That fix left a second defect, which your report did not have to reach but which a screening criterion
cannot survive: **`δ' = ‖μ' − μ'_B‖` is the displacement caused by including the adversarial points, so
evaluating the condition requires running the very composition the criterion is supposed to screen.**
This revision removes it.

**Corollary 1 (checkable separation margin), new in this revision.** Since `μ'` minimises
`F(x) = Σ_i ‖x − u'_i‖`, comparing `F(μ') ≤ F(μ'_B)` and applying the triangle inequality termwise
gives `(n_b − n_a) δ' ≤ 2 S'_B` where `S'_B := Σ_b ‖u'_b − μ'_B‖`. Writing `R̄'_B := S'_B / n_b` for
the **mean** transformed benign residual,

> `δ' ≤ C_A · R̄'_B` with `C_A := 2 n_b / (n_b − n_a)`,

so part (2)'s condition holds whenever `Δ'_sep > 2 R'_B + 2 C_A R̄'_B`. No quantity in that inequality
is computed from the composed aggregate.

**This vindicates your instinct about honest majority while correcting where it belongs.** `C_A`
diverges as `n_a → n_b`, so `n_b > n_a` is exactly what the geometric case needs. But it enters on the
**displacement**, not as a bound on the weight ratio. The paper now says so, and retracts the earlier
weight-ratio summary in the text rather than quietly dropping it.

**Measured, at Dirichlet α = 0.5, on the 17 rounds with positive adversarial weight:** the
displacement bound holds in 17/17, conservative by at least 10.5× (mean 215×). The sufficient
condition is satisfied in 16/17: 15/15 at `n_a = 1`, where `C_A = 2.67` and the requirement is
`Δ'_sep > 4.29 R'_B` against a measured ratio never below 5.88; and 1/2 at `n_a = 2`, where
`C_A = 6.00` tightens the requirement past what one of the two rounds supplies. **At α = 0.1 it holds
in only 5/12.** The checkable form certifies less exactly where the condition it replaces also stops
binding, and we say so rather than reporting only the favourable setting.

**We also attempted the stronger result you did not ask for, and it failed.** The natural next step is
part (1)'s analogue: a threshold on `ρ` alone, computable from pre-composition geometry, so no pilot is
needed. Bounding `Δ'_sep ≥ w_min[Δ_sep − (ρ−1) N_A]` and `R'_B ≤ 2 w_min[ρ R_B + (ρ−1)‖μ_B‖]` makes
`w_min` cancel and yields a critical `ρ*`. **It is empty on our data**: it requires
`Δ_sep > 4(1+C_A) R_B`, the measured ratio reaches at most 1.015 of that requirement, and so `ρ* ≤ 1`
in 16/17 rounds at α = 0.5 and in **all 12** at α = 0.1, while measured `ρ` is never below 1.034 and
1.015 in the two settings. No value of `ρ` satisfies it. That is a stronger negative than a merely
conservative threshold, and the paper reports it as vacuous. We claim **no pilot-free screen for the
geometric class.**

All three inequalities are validated independently of any FL run on 4000 random configurations
(`experiments/verify_margin_inequalities.py`, float64): 4000/4000 on each.

**A consequence we did not want but have accepted.** The operational calibration protocol previously
claimed the theorem-backed class needs *"no calibration runs."* For the rank aggregators that is still
true: part (1)'s `ρ < r*` needs only `ρ` and coordinate-wise separation, both read off the baseline
runs. **For the geometric case it was false, and the paper now says it was false.** Corollary 1 removes
the quantity that required running the composition, but the benign spread that replaces it is measured
on transformed points, so RFA carries the same short-pilot cost as every other family and is no longer
exempted. This is also the direct answer to the other reviewer's note that Theorem 8's required
quantities are hard to estimate operationally.

---

## Concern #1: the 100% PASS precision claim

Agreed, and already changed before your report. The abstract does not contain `100%`, `90.5%`, or any
tally. `PASS precision: 100%` survives at one appendix site, where it is computed, immediately beside
the sentence that a constant HIGH predictor scores 37/42 = 88.1% on the same pairs so the 90.5% figure
beats it by exactly one pair, and that the informative quantities are precision 2/2 and recall 40%.
The paper states in the body that the tallies *"should not be read as discrimination."*

**On your deeper point, that no clean PASS prediction was independently tested, we did more than
reword.** The paper's only fully out-of-sample and only prospective evidence is now surfaced in the
body: 20 pairs over three defenses the paper had never used (FLTrust, Krum, Multi-Krum), predicted
label **and mechanism** recorded for all 20 and git-committed at `262cf35` before any run existed and
never revised, with the predicted-FAIL sanity gate the earlier battery lacked. **It scores 13/20, and
we do not offer that as validation of the screen** in the paper or here. The reason is stated in the
same breath as the score: FLTrust suppresses both committed attacks alone at max-committed ASR 0.048,
so all 13 pairs containing it land within 0.030 of FLTrust alone in either slot, seven of them outside
a scope condition **we added retrospectively rather than pre-registering**. Only the seven FLTrust-free
pairs could have surprised us (5/7), and none is predicted-LOW, so they bear on specificity alone.

We think this is the honest version of the result you asked for, and it is weaker than the wording you
proposed. We did not adopt your suggested sentence, because *"transferred across DBA, N=100/ResNet18
and adversarial-fraction shifts"* describes composition transfer rather than independent samples of the
prediction space, which is the distinction your own report draws two paragraphs earlier.

## Concern #3: the criterion is partly tautological

Adopted, and taken further than reframing. The paper now argues the criterion **cannot** be validated
the obvious way, and that this is a structural fact rather than a sample-size problem: a criterion
gated on the outcome it predicts conditions on a descendant of both the mechanism and the downstream
defense's standalone strength, so no screen of this form identifies the mechanism's contribution at
any sample size (Prop. 1, with the general form in the appendix). The criterion is presented as an
**evaluation prioritizer**, not a predictive theory, and C3 is demoted to a margin diagnostic because
it shows no incremental predictive value on our data. Your framing, *"a theory-backed composability
diagnostic,"* is close to the paper's current position; we would put it more negatively still.

## Concern #4: the FG→RFA accuracy tradeoff

Adopted. The paper states that FG→RFA imposes a substantial clean-accuracy cost (0.50 vs 0.74 for RFA
alone), attributes that cost to FoolsGold's per-client downweighting rather than to composition, and
places the pair on an explicit accuracy/security Pareto frontier table alongside the alternatives. The
phrase *"deployable stack"* does not occur. The paper makes no deployment recommendation at all: its
stated output is an evaluation ordering.

## Concern #5: "composition dominates mixing" is too broad

The claim is **gone**, not qualified. `composition dominates mixing` occurs zero times. Because the
supporting mixing evidence moved out of the body in the re-centering, leaving the claim asserted would
have been worse than broad, so Related Work now says outright that an earlier version made that
comparison, that it is not part of this submission, and that no claim here rests on it. The NormClip +
Reputation counterexample you cite (scaling ASR ≈ 0.807) is retained in the appendix and is now
referenced from the body as evidence for the C2 condition.

## Concern #6: the persistence boundary is not quantitatively characterized

Agreed, and we have withdrawn the claim rather than defend it. The paper no longer advances the
`(γ, α, p_eff)` model, and states explicitly: *"We make no formal claim about persistence."* The
withdrawal is written into Related Work rather than being a silent deletion, because the previous draft
also referenced a "Heuristic 1" that existed in neither document. What survives is `γ` as a measured
descriptive quantity and the DBA persistence replication, which returns an intermediate verdict and
establishes no boundary. The architecture-conditional ResNet18 contrast you flag (0.097 vs 0.865) is
retained, labelled architecture-conditional, and now carries a pointer to its per-seed values.

## Concern #7: selection dependence in the rep30/tm70 survivor

Moot in this draft: that result and its winner-bias discussion are part of the relocated mixing
stratum, and the body makes no randomization-survives claim. The general principle behind your concern
is now stated as a scope condition rather than a caveat: the strongest development-set figures are
labelled *fit*, not prediction.

## Concern #8: the paper is doing too much

Adopted, and this was the largest single change. Your "Paper A / Paper B" diagnosis is exactly what a
second reviewer independently demanded, and the resolution was to keep Paper A. The main text is 9 pp
against the ICLR limit, the criterion-validation stratum moved to an 8 pp companion supplement linked
by `xr` so cross-document numbers are read from the other document's `.aux` and cannot drift, and
nothing was deleted in the process: 665/665 distinct numeric literals were verified conserved across
the split, none reduced in count.

---

## Since drafting this reply: the 2 × 2 factorial ran, and refuted our own channel attribution

An earlier version of this letter listed the factorial as unrun. It has since been pre-registered, run
and reported, and **the outcome went against us**, so it belongs here rather than only in an appendix.
Rules were frozen in `experiments/pre_registration_emit_only.md` at commit `b995f1b` before any
emit-only ASR existed, with the runner refusing to start unless its own recorded commit matches that
hash. All three harness checks the freeze required pass at 0.00e+00 across five published arms and the
score-only control, so the refactor that added the fourth cell perturbed no published computation, and
the κ = 0 rung is imported bit-identically rather than re-run, because at the identity the transform
returns the stack unwrapped and emit-only *is* Krum alone. The fourth cell, **emit-only Krum**, scores
the untransformed stack (pinning the decision to the identity rung's) and aggregates the transformed
selected update, so it closes the statistic channel exactly where score-only closes the magnitude one.

Seed-matched at the primary rung, which the 0.35 clean-accuracy floor **forced** to κ = 1 rather than
κ = 2 (Section 5 of the pre-registration named κ = 2 as the rung at risk, and named it for this cell
specifically, before any ASR existed):

| Cell | Statistic channel | Magnitude channel | Mean ASR | ΔASR |
|---|---|---|---|---|
| κ = 0 identity (imported) | closed | closed | 0.062 | (baseline) |
| Score-only | **open** | closed | 0.032 | −0.030 |
| Emit-only | closed | **open** | 0.107 | +0.046 |
| Full Mode S | **open** | **open** | 0.023 | −0.039 |
| additive prediction Δ_SO + Δ_EO | | | | **+0.016** |
| residual against measured Δ_full (frozen margin 0.05) | | | | **−0.054** |

The magnitude channel **is** inert on its own, |+0.046| falling inside the frozen 0.05 inert margin,
which is what we had predicted. But the additive prediction +0.016 sits against a measured −0.039: a
residual of −0.054, outside the frozen margin, and the additive model does not merely mis-size the
joint effect, **it reverses its sign**, the same failure mode this paper's headline result charges the
outcome-gated ladder with (−0.272 against +0.098). So **no per-channel decomposition of ΔASR is claimed
anywhere in the paper**, and we withdraw the re-selection/rescaling split as a decomposition of
*suppression*. It stands as what it says it is: a decomposition of a *displacement*, which makes no
claim about suppression in its own terms.

We report the arm as **not decisive**, and act on it only in the direction that withdraws a claim.
Three reasons, all of them ours to report. (i) It is knife-edge on both margins at once: |Δ_EO| falls
inside the inert margin by 0.004 and the residual falls outside the additivity margin by 0.004. (ii)
Both pre-registered secondaries are null: per-seed sign count 3 up against 2 down, and
Jonckheere-Terpstra across the four rungs gives z = −0.437, p = 0.669. (iii) One seed carries the whole
result: seed 42 contributes ΔASR = +0.272 against −0.007, +0.007, +0.013 and −0.058, and excluding it
would give Δ_EO = −0.011 with a residual of −0.003, which would read as **separable**. **We do not
exclude it.** Section 5 fixes n = 5 on seeds 42-46 and applies the accuracy floor to a rung's *mean*
rather than per seed, so no per-seed exclusion is licensed, and dropping the one seed that produces the
unwelcome answer is exactly what pre-registration exists to prevent. Section 6.1 of the same document
declared, before the run, that at n = 5 only a large |Δ_EO| would be decisive, and |Δ_EO| is not large.
Low power is a reason to stop asserting a decomposition, never a reason to assert one, which is why a
knife-edge result is safe to act on in this direction and in no other.

## What we have not done

- **The factorial is a within-family result at n = 5, not a general one.** It is defined only for the
  two selectors whose decision can be transplanted (`krum`, `cos_krum`) and the harness raises for any
  other `d2`, so it is one attack on one aggregator family, one dataset and one architecture, and it is
  reported as not decisive in (L1) rather than as an established interaction.
- **The page count is at the limit, not comfortably inside it.** The Ethics heading sits on p9. Any
  further body addition requires a compensating cut.
- **`ρ*`, the pilot-free geometric screen, is not available.** We report it vacuous rather than
  presenting a conservative threshold.
