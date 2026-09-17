# Response to the ninth review (a line-by-line technical audit, no score given)

Thank you. This is the first review to audit the mathematics line by line, and it found **three real errors
in one theorem**. All three are ours, all three are correct catches, and none of them is cosmetic. We lead
with them, state the wrong claim, the correct algebra, and **which conclusion survives and why** — because
in two of the three cases the corrected statement is *stronger* than what we had written, and in the third
it removes a claim we should never have made.

The review's line numbers are PDF margin numbers of a pre-Round-68 build (it calls the heterogeneity sweep
Appendix D.6, which is now §E), so every item below was resolved by grepping current content rather than
by assuming an offset. Source line cites in this letter are `main.tex` source lines and are **not** the
PDF's margin numbers.

---

## 1. Three errors in Theorem 8, conceded

### 1a. Norm clipping does not make consensus distance collapse (part 3, and the C2 exposition)

**What we wrote.** That clipping every client to a common $\tau$ makes the consensus distance
$\|u_i-\hat\mu\|$ collapse to a client-independent constant, so adversarial and benign updates become
equidistant from consensus.

**Why it is wrong.** The reviewer is right and the algebra is one line. With $\|u_i\|=\tau$ for every $i$,

$$\|u_i-\hat\mu\|^2 = \tau^2 + \|\hat\mu\|^2 - 2\langle u_i,\hat\mu\rangle,$$

and the first two terms are common to all clients while **the inner product is not**. Equalizing norms
removes the *radial* part of the separation and leaves the *angular* part untouched. Nothing collapses.

**The correction, and why it is stronger.** Part (3) now separates the two cases it had conflated. A
$\sigma_2$ keyed on the update norm *does* become constant and C2 fails outright. A $\sigma_2$ keyed on
consensus distance does not: the statement now carries the identity and says
*"so client dependence survives through the inner product"* `:1225`, and the proof reaches the same place —
clipping *"compresses the separation onto its angular component instead of annihilating it"* `:1232`. C2
then fails once the residual angular separation drops below $d_2$'s threshold, which is
*"a margin condition of the same shape as parts (1) and (2)"* `:1232`. That matters structurally: the old
part (3) was a degenerate outlier in a theorem whose other two parts are margin conditions, and the
corrected part (3) is a margin condition too.

**What survives, and one thing that improves.** The empirical conclusion is measured, not derived:
NormClip$\to$reputation leaves model-scaling ASR at $0.807 \pm 0.048$ and the pixel backdoor at
$0.842 \pm 0.047$ `:1169`, against reputation alone at $0.017$. The corrected algebra explains that pair
*better* than the collapse claim did, and the C2 exposition now says so: for a purely radial attack the
radial part is the whole of the separation, and *"For a purely radial attack that is the whole of it, and
model-scaling is one"* `:1169`, since scaling multiplies the update by a constant that clipping undoes
exactly; where the adversarial direction also differs, *"C2 fails by margin compression rather than by
annihilation"* `:1169`. The scope condition that kept this case away from our runs is unchanged and still
stands: at $\tau{=}5$ only 1--2 of 5 clients clip per round `:1332`.

### 1b. An adversarial coordinate large in absolute value cannot lie between the benign values

**What we wrote.** That magnitude alone is insufficient without the common-sign assumption *because* a large
adversarial coordinate could sit between the benign values.

**Why it is wrong.** Self-contradictory, as the review says. Since $\rho \ge 1$, the magnitude condition
gives $|u_{a,k}| > \max_b|u_{b,k}| \ge \max(\max_b u_{b,k},\, -\min_b u_{b,k})$, which forces
$u_{a,k} > \max_b u_{b,k}$ or $u_{a,k} < \min_b u_{b,k}$. It **cannot** lie between them.

**The correction.** The statement now gives the algebra and names the real role of the assumption: the
justification is prefixed *"the reason is not that a large adversarial coordinate could sit between the
benign values"* `:1219`, and what magnitude leaves open is *which* of the two tails, so
*"Mixed adversarial signs split the adversaries across both tails"* `:1219` while condition (ii) is a
one-tail condition and (i)'s count budget is written against a single tail. We added the check that makes
the assumption legible: *"With $n_a{=}1$ the sign assumption is therefore vacuous"* `:1219`.

**What survives.** Both conditions and the bound $\rho < r^\star$ are unchanged. Only the justifying
sentence was wrong; the assumption it justifies is load-bearing for a different and correct reason.

### 1c. NormClip's weight ratio is finite, not unbounded

**What we wrote.** That NormClip is not a bounded reweighting because the induced per-coordinate weight
ratio is unbounded, so no $\rho$-margin applies.

**Why it is wrong.** $w_i = \tau/\|u_i\|$ is finite for every non-zero update, so $\max_i w_i/\min_i w_i$
is a finite scalar and a $\rho$-margin formally applies.

**The correction.** Part (3) now says the ratio *"is finite for every non-zero update"* `:1225` and that
what actually fails is checkability: $\rho$ is set by the realized norms, so no margin fixed *before* the
round bounds it. The genuinely unbounded case is a weight of exactly zero, and
*"weight exactly zero is the genuinely unbounded case"* `:1225`, which is already the zero-weight
annihilation lemma (FoolsGold), so the correction cross-references existing theory and adds none.

**What survives, and a corroboration we should have noticed.** The paper already argued this correctly
elsewhere: the restriction of $\rho$ to strictly positive weights exists precisely because FoolsGold's
normalization sends 2 of 5 clients to exactly zero, which would make $\rho$
*"infinite in every round"* `:1283`. So the correct treatment of the unbounded case was already in the
appendix while the theorem's part (3) contradicted it. That inconsistency is now gone.

### 1d. No downstream result read any of the three deleted claims

We checked rather than assumed: every non-comment site citing Theorem 8 reads the bound $\rho < r^\star$,
the separation condition, or the theorem's scope. None reads norm-collapse, the between-the-benign-values claim, or an unbounded
ratio, so no corollary, no limitation entry and no empirical claim moves.

---

## 2. Two further theory items, both correct

**An unbound index.** $r^\star$ was defined with the adversary index $a$ free while the theorem admits
$n_a$ adversaries. It is now $r^\star := \min_k \min_a \min_b |u_{a,k}|/|u_{b,k}|$, glossed in place as
*"which minimizes over adversaries as well so that the bound binds on the least extreme one"* `:1219`.

**$S$ carried three referents, and fixing it exposed an internal inconsistency the review did not name.**
$S$ was the downstream statistic, the benign-residual sum, and the attack's coordinate extremeness. The
theorem itself already calls the third quantity $r^\star$, so the fix cost no new symbol: extremeness is
now $r^\star$ everywhere, including in the notation table, where the row reads
*"the attack's coordinate extremeness"* `:2567`. $S$ now means one thing,
*"the downstream statistic $d_2$ reads"* `:2568`, and $S'_B$ is untouched. The inconsistency this removes
is that the theorem and its own limitation entry had been calling one quantity by two names.

This is a rename of a **symbol**, not of any of the paper's vocabulary: the title, §5, P1--P5, C0--C3,
Mode S/A/M and *admission*/*influence*/*attenuation* are all unchanged.

---

## 3. Our own accuracy floor was applied inconsistently, and applying it weakens a limitation of ours

The review is right. We fix a $0.35$ clean-accuracy floor below which a cell is called uninterpretable
rather than suppressed, and at Dirichlet $\alpha{=}0.1$ we then counted fg$\to$rfa's $0.569$ ASR as one of
two of the three PASS pairs breaking — at clean accuracy $0.20$, below our own floor.

Applying the rule **removes a datapoint from a limitation we state against ourselves**, so we print both
readings rather than quietly taking the softer one.

| pair | ASR at $\alpha{=}0.1$ | clean acc | gate-applied reading |
|---|---|---|---|
| rep$\to$cm | $0.548$ | $0.48$ | crossing, at interpretable accuracy |
| fg$\to$rfa | $0.569$ | $0.20$ | **uninterpretable**, not scored either way |
| fg$\to$cm | $0.433$ | $0.40$ | survives |

The body's boundary condition now states both counts in its own heading,
*"two of three on the raw reading, one of three once our own accuracy floor is applied"* `:1782`, and the
appendix says why both appear: *"Our own accuracy floor changes that count, and it changes it in our
favour, which is why both readings stand here"* `:2207`. The table marks the cell and the marker's footnote
records that it *"is scored as neither a crossing nor a survival. Raw reading kept above."* `:2203`.

We also state the two things the gate does **not** buy, because a softening that is allowed to spread is
worse than the original error: *"the cell is uninterpretable in both directions"* `:2207`, so it is no
evidence that the criterion holds at $\alpha{=}0.1$ either, and rep$\to$cm's crossing is unaffected by the
floor, so extreme heterogeneity still breaks a PASS pair at accuracy we are willing to read. The reason for
printing the raw count beside the gated one is stated in the paper rather than left to us:
*"a reader should be able to see that softening rather than inherit it"* `:2207`.

Nothing was deleted. The cell is reclassified under the paper's own rule and both readings are carried.

---

## 4. A DAG of the composed pipeline (new appendix figure)

The review asks twice for a formal DAG of the composed pipeline and notes that our Figure 2 is not one.
That is correct: Figure 2 is a DAG of the **evaluation design** (the statistic, the gate, the $do(T)$
repair), and it says nothing about what a client update passes through. We added the figure the review asked
for, `paper/figures/pipeline_dag.py` → `figures/pipeline_dag.pdf`, referenced from its host section
*"which is the composed pipeline as a DAG rather than the evaluation design of Figure"* `:1283` and
labelled `:1295`.

It draws the two arcs the paper's whole argument turns on: the statistic path
($T_1 \to \sigma_2 \to$ decision $\to$ admission, i.e. levels P1--P4 in order), and the **bypass arc** by
which $d_1$'s weights reach aggregate adversarial influence $\Lambda_a$ without passing through $\sigma_2$
at all. Mode S is drawn as a cut across that arc rather than as a stage, because pinning every adversary at
$c{=}1$ removes the arc. The caption states what the reader is meant to take from the pair,
*"which is the whole of why (P1)--(P3) do not deliver (P5)"* `:1294`.

Two disciplines it observes. It is an **appendix** float, not a third body figure, because the body's own
roadmap inventory says the main text carries two figures and a third would falsify a self-count no build can
see. And **no result number is baked into the PDF** — the two papers number the same results differently, so
a baked number would be wrong in one document while still resolving to a real result in both; the numbers
live in the caption where `\ref` reaches them.

---

## 5. Three new pre-registered arms, the third a correction of an error of ours

All three pre-registrations were committed **before** their output directories existed, one file per commit,
and each runner refuses to start unless its `PREREG_COMMIT` matches: `580854a` for
`experiments/pre_registration_criterion_aware_topup.md`, `ca89f96` for
`experiments/pre_registration_dose_resnet18_converged.md`, and `784a201` for
`experiments/pre_registration_consensus_shift_fix.md`. All three runners are orchestration only — each
imports `run_one` from the frozen runner rather than copying it, and each carries a `--harness-check` that
re-runs a frozen seed through the imported function and asserts **bit-equality by value** against the stored
row. An md5 of an artifact would not be that proof: this repository has twice shipped a hook that silently
never fired.

**5a. The adaptive arm, $n{=}5 \to n{=}30$ (COMPLETE, and it moved a headline number down).** The review is
right that the $3.7\times$ adaptive gain rested on 5 seeds. It has run, and **we lead with the change rather
than the settled figure**: the ratio falls from $3.7\times$ to $2.4\times$, and every site in the paper that
quoted it now reads $2.4\times$.

The freeze named this trap before the runs, which is why it is a relabelling and not a surprise: the
published $3.7\times$ was a seed-matched $n{=}5$ ratio $0.1668/0.0452$, and at $n{=}30$ **both legs move**,
because the denominator becomes the base pixel arm's own $n{=}30$ mean. Both legs were therefore recomputed
at the 30 shared seeds rather than one, which is the whole point: a top-up that moves only the numerator
hides a mixed-$n$ comparison inside the denominator.

| leg | $n{=}5$ (published) | $n{=}30$ (now) |
|---|---|---|
| criterion-aware, $\epsilon{=}1.0$, decorrelated | $0.1668$ | $0.1571$ (sd $0.0730$, median $0.1587$, max $0.3528$) |
| committed-pixel base, same seeds | $0.0452$ | $0.0644$ (median $0.0517$) |
| descriptive ratio | $3.687\times$ | $2.439\times$ |

**The attack barely weakened; the denominator stopped being lucky.** The numerator moved $0.1668 \to 0.1571$
while the denominator rose $0.0452 \to 0.0644$, and the base arm's $n{=}30$ distribution is right-skewed
(mean $0.0644$ against median $0.0517$), so seeds 42--46 sat in its left tail. Holding the numerator at its
$n{=}5$ value and recomputing the denominator alone would already give $2.588\times$ — so the fall is almost
entirely the denominator, which the paper now states at the site: *"almost all of the fall is the denominator
leaving the left tail"* `:2248`.

**The pre-registered primary confirms the gain at $n{=}30$.** The paired mean difference over the 30 shared
seeds is $+0.0927$ with a 95% Student-$t$ interval $[+0.0659, +0.1195]$, which excludes zero from above —
the frozen confirming branch. The body reports it beside the ratio: *"$2.4\times$ its seed-matched $0.064$
(paired $+0.093$, $[+0.066, +0.120]$)"* `:489`. The ratio stays **descriptive only**, because a ratio of
means has no interval we are willing to defend at these sample sizes.

Four things we are careful not to claim. (i) **The denominator was reused, not re-run**, and the reuse was
verified rather than assumed: the worst absolute per-seed deviation between the reused rows and the stored
base rows is $0.0$ on both ASR and clean accuracy. (ii) **No re-ranking.** Only `ca_eps1_decorr` is measured
at $n{>}5$; the other five adaptive conditions stay at $n{=}5$, so no statement of the form "X is the
strongest adaptive attack" is licensed at this $n$, and the table now prints each row's own $n$. (iii) **The
$n{=}5$ pair stays visible** rather than being overwritten — the appendix keeps both readings and says
*"so the ratio falls from $3.7\times$ to $2.4\times$ once the denominator leaves its left tail"* `:2215`.
(iv) **The gain is still not a utility trade**: mean clean accuracy is $0.549$ for the adaptive arm against
$0.492$ for the pixel arm at the same 30 seeds, with no seed below the paper's $0.35$ accuracy floor.

What survives is weaker than what we shipped, and stated as such: *"The top-up moved that ratio down and we
report the move rather than the settled figure alone"* `:2248`.

**5b. The converged ResNet18 arm, 50 $\to$ 200 rounds (COMPLETE: 6/6 runs, 51.4 h).** The review is right that we evaluate
security on a ResNet18 at $0.536$--$0.557$ clean accuracy. The same cell is re-run at a $4\times$ longer
horizon behind **two gates that decide different things**, and the separation is deliberate:

- **Validity gate, imported from the parent arm unchanged.** Standalone Krum must suppress the committed
  scaling attack on this architecture at usable accuracy: identity-rung mean ASR $< 0.5$ with mean clean
  accuracy $\ge 0.35$, and every rung $\ge 0.35$. Failing it makes the arm **void, not negative**.
- **Convergence *label* gate, new, $0.65$.** That number is not invented for the occasion: it is the bottom
  of the $0.66$--$0.76$ band this paper's own ResNet18 mini-sweep attains in this regime, rounded down. It
  decides only what the arm may be **called**. If it fails, the pre-registered finding is that four times the
  round budget does not bring Krum-aggregated ResNet18 to that band, reported as the result. **The
  $\kappa{=}2$ leg runs either way**: $0.35$ decides validity, $0.65$ decides only the label.

One design change is disclosed in the freeze rather than dropped quietly. We had planned a two-sided
admissibility window requiring identity-rung ASR inside $[0.15,0.85]$ so neither leg is saturated. It is
**not adopted**, because it contradicts the parent arm's own frozen validity gate: that gate requires
identity ASR $< 0.5$, the published value is $0.0707$, and a window demanding ASR $\ge 0.15$ would void an
arm that passed the rule it was frozen against. The saturation concern is real, so it survives as an
**informativeness condition** that changes the reporting rather than voiding anything: benign accuracy moves
by less than $0.07$ across rungs, so if identity-rung ASR at 200 rounds is below $0.15$ then $|\Delta| <
0.15$ is attainable by arithmetic alone, and the arm is then reported as replicating **with the margin not
binding** — never as a strengthened claim.

The three pre-registered outcomes on $\Delta$ are fixed in advance: $|\Delta| < 0.15$, the negative survives
a $4\times$ longer horizon; $\Delta > +0.15$, the negative is horizon-specific and our central claim must be
narrowed; $\Delta < -0.15$, an attenuation-side fall, **indeterminate and explicitly not scored in our
favour**. The freeze also states that the comparison is **horizon-confounded** — 200 rounds is four times the
backdoor injection exposure — so the arm answers whether the dissociation survives near convergence and
**not** what ResNet18's converged ASR is relative to the 50-round arm. No cross-horizon $\Delta$ is reported.

**Outcome: the negative survives, and the margin does not bind because the identity rung has less headroom
than the margin.** All six runs completed. The harness check passed first, reproducing a stored 50-round
per-seed row through the same imported `run_one` to $< 10^{-9}$.

| pre-registered quantity | result |
|---|---|
| validity gate (identity ASR $< 0.5$, all rungs acc $\ge 0.35$) | **passed** |
| convergence *label* gate (identity-rung mean acc $\ge 0.65$) | **passed**, $0.652$ |
| mean clean accuracy | $0.652$ at $\kappa{=}0$, $0.688$ at $\kappa{=}2$ |
| $\Delta = $ mean ASR($\kappa{=}2$) $-$ mean ASR($\kappa{=}0$) | $\mathbf{-0.022}$ ($0.023$ against $0.045$), $n{=}3$ |
| pre-registered verdict on $\Delta$ | $\lvert\Delta\rvert < 0.15$: **the negative survives a $4\times$ longer horizon** |
| paired $95\%$ interval, $m^{\ast}$ | $[-0.182,+0.138]$, $m^{\ast} = 0.182$ |
| informativeness condition (identity ASR $\ge 0.15$) | **failed**, $0.045$ — as the freeze predicted |
| wall time | $51.4$ h over six runs ($6.6$--$14.3$ h each) against a budget of $40$--$54$ h |

Four things we are **not** claiming from it, three of them because the freeze said so in advance:

- **Not a strengthened replication.** An identity rung at $0.045$ cannot fall by more than $0.045$, so
  $\lvert\Delta\rvert < 0.15$ was attainable by arithmetic alone. The freeze predicted this condition would
  fail and fixed the reporting consequence before the runs, which is the only reason a favourable outcome
  here means anything at all.
- **No equivalence at any tighter margin.** $m^{\ast} = 0.182$ exceeds the frozen $0.15$, so at $n{=}3$ this
  arm is a replication of a negative and not an equivalence result — the same limit its 50-round parent has,
  unimproved by the longer horizon. The per-seed differences are $-0.096$, $+0.015$, $+0.016$: the mean is
  carried by seed 42, the one seed whose identity rung was appreciably backdoorable.
- **No cross-horizon $\Delta$.** The identity rung reads $0.045$ at 200 rounds and $0.071$ at 50 rounds. We
  print them side by side because the freeze requires the reader to see the headroom, and we take **no**
  difference between them, because a longer horizon lengthens the adversaries' poisoning exposure as well as
  the training.
- **Not a converged model in any formal sense.** The gate is a label about a measured accuracy band this
  architecture reaches in this regime, and the $\kappa{=}0$ rung at $0.652$ in fact sits just below that
  band's lower end.

One process detail worth reporting, because it is the trap that cost us a claim two rounds ago: at 4/6 and
5/6 completed runs the analyzer **refused to score $\Delta$ at all**, printing that the two endpoints had
been scored on different seed sets and that their difference would mix $n$. It scored in one pass at 6/6.
That refusal is in the runner, not in our discipline.

The arm's full report is the new paragraph block in the appendix section that already houses the 50-round
arm, and disclosure (v) there is re-scoped. It claimed the frozen margin was load-bearing for this arm and for no
other arm in the paper, which this arm falsifies; it is now scoped to the arms of the margin ladder. The converged
arm is deliberately **not** added to the margin-sensitivity ladder, whose emitter reads only the frozen
artifacts.

**5c. A published null that was never measured: the consensus-shift row is withdrawn and re-run (COMPLETE: 10/10 runs, 3.8 h).**
The review does not raise this. We found it while auditing the numbers this round, and we report it because a
null we never measured is worse than any criticism the review made.

`tab:fg_rfa_flagship` printed a consensus-shift row at `$0.045 \pm 0.012$`, and the surrounding
paragraph counts it among the arms that *"the other three sit level with the base"* `:2248`. **The
perturbation was never applied.** The runner builds the attack as `get_attack('backdoor_pixel')`, and
`BackdoorPixelAttack` overrides only `poison_dataset` and `cost`, so it inherits the base class's identity
`manipulate_update`. The Phase-1 update is then `update + shift_rate * (update - update)`, which is exactly
`update` **for every shift rate**, and the Phase-2 injection is likewise the identity. The run silently
reduced to the base committed-pixel composition.

The artifact witnesses this without reference to the code, which is how we are confident it is the code path
and not a coincidence: `rate_0.01`, `rate_0.05` and `base_composition/committed_pixel` agree on **all five
seeds, in both ASR and clean accuracy, to twelve decimal places**. Two shift rates a factor of five apart
cannot produce bit-identical clean accuracy under a live perturbation. The printed
`$0.045 \pm 0.012$` is the base pixel arm's own $n{=}5$ figure re-labelled as an attack result.

- **The row is withdrawn regardless of what the re-run finds.** That is written into the freeze, not decided
  after seeing an outcome. No result makes the shipped number retrospectively defensible.
- **The re-run moves exactly one factor: the attack object.** Config, composition, seeds, rates and metric are
  imported from the existing runner, never copied. The two semantics that make the attack a *small* effect on
  this composition — one shared attack instance, so its counter advances per adversary-update event, and
  `scale_factor = 1.0`, so Phase 2 is ordinary committed pixel — are **frozen rather than corrected**, because
  correcting them would make the arm incomparable to the published rep$+$tm arm it sits beside. The freeze
  therefore predicts a small effect *before* the runs.
- **The harness check is deliberately pointed at the base row, not at the consensus-shift rows.** The stored
  consensus-shift rows are void, so agreeing with them would prove nothing. Reproducing the *base* seed-42
  row through the new loop to $< 10^{-9}$ proves the loop, aggregator, partition, metric and RNG consumption
  are unchanged, which isolates the attack object as the single moving factor.
- **The buggy function is left in place** as the record of the defect rather than silently repaired, and the
  void cells stay on disk so the error remains auditable.
- **Not affected, checked rather than assumed.** The rep$+$tm consensus-shift figures in (L3) come from a
  different runner that constructs the real attack class and calls its live hook; they are sound and are not
  re-run. The projection and Neurotoxin rows of the same table differ from the base arm per seed, so their
  hooks fire.

This is the **second** hook in this repository that never fired and failed silently. That is why every check
in this round is a per-seed value comparison and never an md5 of a results file: an md5 would have called this
arm reproducible.

**Outcome (all ten runs complete).** The harness check passed exactly: the new loop reproduces the stored
seed-42 base row at $|\Delta\mathrm{acc}| = |\Delta\mathrm{ASR}| = 0$, which isolates the attack object as
the single moving factor. With the live attack:

| rate | mean ASR ($\pm$ sd, $n{=}5$) | paired difference vs.\ the pixel arm, $95\%$ CI | mean acc |
|---|---|---|---|
| $0.01$ | $0.048 \pm 0.010$ | $+0.0031$, $[-0.0005, +0.0067]$ | $0.495$ |
| $0.05$ | $0.045 \pm 0.010$ | $-0.0006$, $[-0.0042, +0.0029]$ | $0.497$ |

Both intervals contain zero, so the **pre-registered attack-ineffective branch fires**: the row is restored
with the honest figures and the count sentence stands unchanged:
*"Three of the six adaptive strategies beat the base attacks"* `:2248`.
Consensus-shift stays among the arms level with the base — now for a measured reason rather than by accident.
Three consequences we report rather than leave implicit:

- **The corrected mean is $0.045$, the same three-decimal figure the void row printed.** Only the sd
  ($0.012 \to 0.010$), the max ($0.059 \to 0.058$) and the accuracy ($0.495 \to 0.497$) move. That near-identity
  is *why* the defect survived review by us for as long as it did, and it is also why the disclosure and not
  the row carries the correction: a reader diffing the table alone would see almost nothing.
- **A $5$-seed null on this composition is weak evidence of a null**, and the freeze said so before the runs.
  The base arm's own $n{=}30$ distribution is right-skewed and seeds $42$--$46$ sit in its left tail. The arm
  establishes that the attack does not beat the base on the five seeds the table reports, not that it cannot.
- **Part of the null is mechanical.** At `scale_factor = 1.0` the attack's second phase is ordinary committed
  pixel, so the only distinguishing mechanism is Phase-1 drift. The freeze predicted a small effect from that
  fact in advance, which is what makes the prediction cheap rather than impressive.

The paper now carries the defect in the table caption — the void figure, the identity-hook mechanism, the
twelve-decimal-place agreement, the re-run directory and its prereg hash — plus a row in the
withdrawn-claims ledger, whose header would have gone false without it:
*"Every claim we have withdrawn or refuted, enumerated in one place"* `:2452`.

**One erratum in our own freeze**, recorded here rather than by editing a committed file: its caveat 3 says
the FG$\to$RFA table prints two shift rates. It prints one, at rate $0.05$. The reporting rule that followed
from it is unaffected — both rates were run, scored and reported — but the restored table keeps the single row
it had, because the count sentence beside it enumerates table configurations, and a second row would silently
turn a count of three of six into three of seven. Rate $0.01$ is reported in the prose and in the table above.

---

## 6. Five edits at the body sites the review points at

- **Mode S is previewed where the protocol first needs it.** The protocol clause that forces the
  intervention now names the device and its price in the same breath:
  *"which requires knowing which clients are adversarial and is therefore a laboratory instrument rather
  than a deployable defense"* `:478`. This closes the methodological-flow item and the oracle-boundary item
  together.
- **The Conclusion now names what holding the threat fixed costs.** The rule is now
  followed by a clause naming its price:
  *"it is a diagnostic under a committed attack and not a guarantee against an adversary that adapts"* `:756`.
  Its positive half — that the dissociation survives one adaptive adversary at $N{=}10$/$K{=}5$ —
  is carried at its §6.2 home rather than duplicated here, under the paper's one-home-per-claim rule.
- **The excluded transformation class is named by example.** The sentence that already said the theory has a
  boundary now says which canonical defenses lie outside it: *"nothing we prove speaks to a transform
  outside it"* `:378`, and Bulyan and FLAME are named there as composed defenses that mix client updates
  across the network and so admit no per-client $T$. Bulyan is a new bibliography entry, cited from
  `main.tex` only.
- **A cross-reference the reviewer read as broken.** `p\pageref{...}` rendered as "p5" and looked like a
  malformed reference; it now reads *"the attacks and seed counts of page"* `:644` followed by the page
  number.
- **The word replicates is not allowed to be read as equivalence.** The ResNet18 interval is wider than the margin
  itself, and the consequence is now explicit in the same clause: *"so equivalence there is indeterminate at
  $n{=}3$ rather than established"* `:702`.

The §6 recital cut we had reserved to pay for these lines **proved unnecessary**: all five fit inside the
body's page budget, with the Ethics section still opening on page 9 after a fixpoint build. So no
limitation, scope condition or disclosure was shortened to fund them, and none was deleted this round.

---

## 7. Fourteen local corrections, each at the site the review names

| # | the item | what we did |
|---|---|---|
| 1 | an undefined $\Delta$mix column | **Deleted the column.** It was blank on 6 of 9 rows, its three values are quoted nowhere else, and no emitter defines it. Back-defining a number whose emitter cannot be found would be invention |
| 2 | Bulyan missing from the bibliography | Added and cited at `:378` |
| 3 | literature-audit coverage gaps | The CCS HTTP 403 gap was already disclosed; the disclosure now says *"and CCS is not the only gap"* `:2311` and adds the two we had not named, since *"OpenReview is unscreened"* `:2311` and so are IEEE venues other than S&P |
| 4 | an appendix paragraph describing panel (a) under a panel-less figure | Both figures are now named by `\ref`, and the arc it describes is the one that reaches ASR *"without passing through $d_2$'s statistic at all"* `:2530` |
| 5 | randomization tested only against a static attack | Correct and previously unanswered. The scope sentence now says *"the attack is committed in both arms, which is the scope condition that bites hardest"* `:2954`, and that randomization's theoretical value is against an adversary that adapts, since *"a menu the adversary cannot predict is a menu it cannot optimize against"* `:2956`. No rerun: the arm bounds the cost of randomizing and measures none of the benefit |
| 6 | an unrolled gradient-based adaptive attack | Declined as an experiment; the footnote's argument now names what an analytical construction at the binding constraint cannot exclude, gradient obfuscation and optimization instability |
| 7 | a column of a $3\times3$ box used to describe a table **row** | Both sites now name the axes and which column: the $\ddagger$ row's three pairs *"are all three upstream defenses against the single downstream defense"* `:2012`, which is one column of a census over $d_1 \times d_2$ `:2041`. The same audit found a **false count** at the second site, which claimed two thirds of the box had never been run; it is two cells, and the sentence now reads *"the two cells of that box which had never been run are what this wave measured"* `:2042` |
| 8 | $\Lambda_a$ used before definition | A forward reference at the first use, in Definition 1's (P4) `:387` |
| 9 | a garbled clause about the refuting branches | Now *"and neither of the pre-registered refuting branches fired"* `:704` |
| 10 | a title truncated to an unreadable stem | Answered **in the caption**, because the table body is machine-written from the audit records and we did not edit a `results/` artifact this round: all four truncated stems are expanded there, and *"We extend them here rather than in the table, whose body is machine-written"* `:2354` |
| 11 | an undefined "mix (NC/rep)" header | Expanded in that table's caption; the header row itself is *"ResNet18 mean & cifar\_cnn mean"* `:2825`, whose second column had also been mislabelled `R50` — a name with no referent anywhere in the repository |
| 12 | Proposition 2 rendering as suppression of a emergent | **A real fix, not a no-change.** In source the symbol is already math, so a source grep passes; but in the italic proposition font $a$ is visually indistinguishable from the article "a", and the line rendered without the copula. A copula now separates them at both of the proposition's homes, and we confirmed the fix by reading the rendered page rather than the source |
| 13 | `simple cnn` / `cifar cnn` written with spaces | No defect. Both read `\texttt{simple\_cnn}` and `\texttt{cifar\_cnn}` in source; the space is introduced by PDF text extraction, which renders `\_` as a space, so we verified in pixels as well as in source |
| 14 | an overbar over a decimal point | No defect. There is exactly one `\bar` in the file and it is well-formed; confirmed by reading the rendered page at high resolution before touching anything |

---

## 8. Six declines, each with the site that answers it

1. **An anonymous code repository.** Declined, and the cost is recorded rather than argued away:
   *"No code or data package accompanies this"* `:792` submission. This is blocked on rotating two live
   plaintext access tokens and rewriting history, which is not an action we can take inside the paper.
2. **Converged re-verification of the ResNet18 *mix* sweep.** The *dose* arm is being re-run converged
   (§5b); the 15-run mix sweep is not, and the architecture-conditional boundary is already scoped as a
   measured swing rather than a law.
3. **An adaptive top-up beyond one condition.** Only `ca_eps1_decorr` goes to $n{=}30$; the other five stay
   at $n{=}5$ and no re-ranking among adaptive strategies is claimed.
4. **A discrete margin ladder.** Still delivered as $m^\ast$, the smallest surviving margin, which is
   strictly stronger than sampled thresholds; and the margin is reported as not load-bearing —
   *"The margin is not load-bearing"* `:532` for any reading of the ladder.
5. **Retitling.** Nothing in the paper's vocabulary is renamed. §2's rename is of a **symbol** to the name
   the theorem already used.
6. **New theory.** No new proposition, corollary or lemma. §1 corrects justifications inside an existing
   theorem and adds no result.

---

## 9. What this round did not touch

Novelty framing is untouched by design, as it was in the previous round. The bibliography has not been
re-verified end to end since an earlier round, and this round adds one entry to `main.tex`'s bibliography
only, which is a deliberate divergence from the workshop bibliography and is recorded as such. No existing
`results/` directory was written, no existing figure was regenerated, and no existing pre-registration was
edited.

The workshop version **was** touched, at exactly the five sites that carried the superseded $3.7\times$ and
$0.167$ figures, so that the two documents do not disagree about a number one of them has moved. One of those
five quoted a ratio whose legs sat at different $n$; it now quotes both legs at $n{=}30$. The workshop's
NormClip $\rho$ range shares the digits $3.7$ but not the referent and is deliberately unchanged.
