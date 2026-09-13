# Response to Reviewer (6/10, Weak Accept; confidence 4/5; novelty 6)

Thank you. Two things need saying before any point-by-point reply, and the first is ours to own rather
than yours.

**The PDF you were given is not the paper under review.** Your review reads `main(41).pdf`, dated
**3 July 2026**. The submitted source is two months and one full restructure later. The consequence is
not that a few line numbers moved: **the object your review is about no longer exists in the paper.**
The July draft argued for a VoPD / defense-randomization / persistence-boundary result. The current
paper argues that no outcome-gated test can identify a composed defense's mechanism, and proves it. Of
your twelve asks, **nine target text that returns zero occurrences in the current source**, and we give
the greps below so this is checkable rather than asserted. This is the **fifth consecutive review to be
served a superseded PDF**, and we state that once, as a process fact, because it is the single largest
determinant of how informative our reviews have been. It is not a complaint about you.

**The one thing in your review that survives the restructure is the reason you withheld a 7, and it is
the reason we accept.** You write that *"the core theoretical contribution is modest and the strongest
empirical phenomenon is still demonstrated primarily in a relatively constrained FL setting."* Both
halves land on the current paper, unchanged by the restructure, and **novelty has scored 6 to 6.5 in
every review this paper has had, across two entirely different framings**, which tells us the problem
is not the framing. So this revision spends itself on exactly those two clauses:

1. A **pre-registered systematic audit** of the design our identification result is about, converting a
   declined claim into a measured prevalence over 59 coded papers. *This audit's first act was to
   delete our own paper's one named external instance.*
2. A **third dataset** for the one cell where the sign reversal actually lives, because you are right
   that it lived on one dataset with one architecture, and the paper's own table concealed that by
   spanning two datasets on a *different* cell.

Neither was asked for by your review. Both answer it.

---

## Part 1: the nine void asks, with line numbers so you can check us

Every count below is `grep -c` against the submitted `paper/main.tex`.

| your ask | target | occurrences now | disposition |
|---|---|---|---|
| W1 | `POMDP` | **0** | quantity withdrawn; the direction error is conceded on the record; see Part 2 |
| W1/W2/§17 | `realized VoPD` | **1**, in a withdrawal | renamed *and* withdrawn; see Part 2 |
| W3 | `value of information` | **0** | abstract no longer leads with a theorem at all |
| W4/W6 | `persistence boundary`, `phase boundary` | **0** each | persistence claim withdrawn by name (`:1672`) |
| W6 | `single-crossing`, `argmax divergence` | **0** each | gone with the framing |
| W7 | `200 rounds`, `horizon`, `long-horizon` | **0** each | the low-$f$/long-horizon probe is not in this paper |
| W8 | `K/N`, `clients_per_round`, `N-sweep` | **0** each | no $N$-sweep claim is made; see below |
| W9 | `dichotomy` | **0** | no dichotomy is asserted |
| §14 | `27/30` | **0** | see Part 3 |
| §18 | the title | n/a | already changed, on an earlier reviewer's advice (`:43`) |

Two of these deserve more than a table row, because "zero hits" is the *weaker* answer in both cases
and we would rather give you the stronger one.

**W3 (the abstract should not be built on a classical VOI theorem).** Agreed, and the current abstract
is built on neither VOI nor a theorem. `:52` opens: *"A preserved statistic is not preserved
suppression, and no test that admits a composition because one constituent already suppresses the attack
can identify whether the downstream mechanism is responsible for the composition's success."* The
paper's headline empirical claim is a **sign reversal between two designs on one identical cell**
($-0.272$ against $+0.098$), not a theorem restatement.

**W8 ($N$ is confounded with $K/N$ in the scale sweep).** The sweep is gone as a claim. What is left of
it is a single $N{=}100$ table at `:2033`, explicitly labelled *"Residual material from an earlier
framing, recorded rather than dropped"* and prefaced with *"Nothing in the screen, the identification
result or the dose--response evidence depends on it."* We did not delete it, because deleting material a
reviewer has criticised is how a criticism becomes invisible; we demoted it and said so. Your confound
is real and it is now a confound in a table that carries no claim.

---

## Part 2: W1, conceded in the paper and not only in this letter

This is the one place your review is right about the **mathematics**, and it is right in a way that
survives the framing change even though the quantity does not. The July draft said its
oracle-minus-committed difference **lower**-bounds the gap realizable by an adversary with no
information about the server's draw. That is backwards, for exactly the reason you give: the committed
policies are a **subset** of that adversary's available policies, so the unrestricted best is at least
as good as the best committed one, and subtracting the larger quantity makes the difference an **upper**
bound.

We could have let this go, since the quantity was deleted in the restructure and no reviewer of the
current draft would ever see it. We think that would be the wrong call, so it is now on the record at
`main.tex:2033`, in the paper, in these words:

> **This is not the quantity an earlier draft called "realized VoPD", and that draft described its own
> quantity with the bound direction inverted, which is worth recording rather than quietly dropping.**
> There the reported difference was oracle-minus-committed, and it was said to *lower*-bound the gap an
> adversary with no information about the server's draw could realize. It does the opposite: the
> committed policies are a *subset* of that adversary's available policies, so its best policy is at
> least as good as the best committed one, and subtracting the larger quantity makes oracle-minus-committed
> an *upper* bound on the gap. That quantity is withdrawn with its framing, nothing in this paper uses
> it, and the VoPD defined above is a different object, non-negative by construction and sharing only
> the name.

That paragraph also settles **W2** and **§17** together. W2 asks us to rename "realized VoPD" because
population VoPD is non-negative while the estimate can go negative. The name is retired; the surviving
`VoPD` is redefined as **value of policy diversity**, the equilibrium strategy's gain over the best
*pure* defense, so it is zero exactly when the equilibrium is pure. That is non-negative by
construction and cannot reproduce the defect you identified.

---

## Part 3: W5, W6 and §14, all live, all already in the paper

**W5: lean into deterministic composition dominating randomization.** Done, and it is now a
pre-registered experiment rather than an observation. App. O reports a compute-matched randomization
test with the degeneracy confound removed by construction: on model scaling
$\bar{\Delta} = +0.1012$ with $5/5$ concordant signs; on the pixel backdoor $\bar{\Delta} = +0.0663$
with $5/5$ signs. The frozen materiality rule required **both** $\bar{\Delta} \geq 0.10$ and $\geq 4/5$
signs, so we report **material on scaling and not material on pixel**, and we say in the paper that
*"a threshold met by $1.2\%$ of itself is met, and it is also the weakest way a threshold can be met, so
we do not describe the scaling result as large"* and that *"the stable part of the result is the sign
structure, $10/10$ across both attacks, and not either magnitude."* A third arm breached the clean-accuracy
floor and is reported as uninterpretable rather than compared. We did not lean into this as far as your
review invites, and the reason is in the numbers: the effect is real in sign and weak in magnitude.

**W6: external validity, and stop calling it a boundary.** The word was audited this round. **53
occurrences on 42 lines, every one read and classified.** Most are a *formal object* and stay: the
testability boundary of Lemma `lem:testability`, $d_2$'s rejection boundary, C3 boundary compression,
the identification boundary of Prop. `prop:identification_full`, the $m \to 1$ boundary of
Cor. `cor:aggregate_mass`. Four more are explicit **denials** that a boundary was established. `:385`
says the scale and architecture checks *"establish no boundary of their own"* and `:2066` says its sweep
*"establishes no boundary of its own"*, `:818` says the $0.5$
threshold is *"not a theoretically privileged security boundary"*, `:1672` withdraws the $\gamma \to 1$
heuristic boundary outright. The empirical kind you object to occurs at **ten sites**, `:530`, `:1273`
(twice), `:1275`, `:1816` (twice), `:1822`, `:1824`, `:2052` and `:2066`, and **every one already carries
its scope condition in the same sentence**: `:530`, `:1816` and `:2052` say *"architecture-conditional"*,
`:1275` and `:1822` say the decision boundary *"may shift with output dimensionality"*, `:1824` says C3's
threshold *"in practice requires calibration runs"*, `:1273` marks its own reading as *"a post-hoc
measurement of a known failure, not a pre-registered prediction of it"*, and `:2066` says the `cifar_cnn` collapse boundary
*"is therefore a property of the smaller-architecture"* run. The ResNet18 disagreement you cite is
reported as the primary fact about that sweep, not a footnote. **No loose usage was found, so no edit was
made**; we are telling you the audit came back clean rather than inventing a change to report.

**§14: lead with the effect size, not $27/30$ and $p<10^{-5}$.** Agreed as a principle and audited as a
rule this round. `27/30` does not appear. Every claim we could find that carries a test statistic leads
with the effect: `:1403` opens *"The outcome is $\Delta = +0.098$: the admission ordering is refuted"*;
L6 at `:1838` leads with the refutation and reports $\rho_s = +0.351$, $p = 0.091$ after it. The one
place a $p$-value leads is a TOST **equivalence** test, where the margin *is* the effect size and is
printed with it ($\pm 0.15$, with the per-arm effects $+0.178$, $-0.425$, $+0.098$ alongside).

**W7: the underpowered $n{=}3$ probe.** The probe is gone. Where $n{=}3$ still appears it is disclosed
as frozen rather than chosen: `:1464` records *"(i) $n{=}3$ seeds per rung, not $5$, fixed in advance to
match the EMNIST-byclass payoff matrix's own trial budget"*, and adds that the frozen verdict is the
$n{=}3$ one. **W9** is likewise not asserted in any form we can find.

---

## Part 4: what this round added in answer to *"the core theoretical contribution is modest"*

The July draft's claim about the literature was a decline: it named **one** external instance and said
so. That is thin, and it is thin in exactly the place your novelty score is about. So we pre-registered
an audit, froze the rubric at a commit **before retrieving a single paper**, and ran it.

**Frame and funnel.** Three routes in (the bibliography, five frozen query strings over 2019 to 2026, and
four venues' complete title listings under a frozen regex), plus a one-hop snowball with no second hop.
$622$ records retrieved, $595$ after deduplication, $593$ screened, $59$ full texts obtained, $59$
included, $534$ excluded with a reason string each. Five criteria, each requiring a **verbatim quotation
plus a locator**; the outcome-gated verdict is
C-a $\wedge$ C-b $\wedge$ C-c $\wedge$ C-e $\wedge \neg$ C-d.

**The result is a low rate, and a low rate was pre-registered as meaning the decline is kept and
evidenced rather than replaced.** Primary **6 of 59 = 10.2%**. The design itself
(C-a $\wedge$ C-b $\wedge$ C-c) holds for **33 of 59**. And the number that constrains us hardest: an
identifying contrast is **present in 46 of 59 = 78.0%**, and among the 33 design papers **27 run one**.

**Which is why the audit's first casualty is our own sentence.** `fenaux2025hammer` was the paper's one
named external instance. It codes **C-d = YES on strict grounds**. Its Table 3 holds the Type-2 stage
fixed and varies the upstream Type-1 defense across six choices, so the instance is **withdrawn by our
own audit**, and the paper says so at `:1662` before it says anything else the audit found. The residual
claim is now narrow and stated as such: **no row in the frame measures whether the ordering of two
downstream arms survives an upstream stage, with intervals over seeds.**

**What we are not claiming.** Not that the contrast is rare (78% run one, so any sentence implying
otherwise is false against our own audit), not that the six papers' conclusions are wrong, and not that
this is an estimate of "the FL literature": the frame is arXiv-reachable, preprint-heavy and 2026-heavy,
ACM CCS is a coverage gap because its proceedings pages return HTTP 403, and there is one coder and no
inter-rater statistic. **The mitigation is the quotation, not a claim of reliability**: all 59 records,
295 quotations and 334 spans are machine-verified verbatim against each paper's own extracted text, the
checker exits non-zero on any failure, and its first whole-set run **failed 28 quotations in 15 records**,
and the repair log is committed, no code changed in any repair, and two repairs produced *stronger*
evidence for the code already recorded, which is the uncomfortable part, since a reader checking the
argument would not have caught paraphrases and only the machine did.

**Not all of the rubric predates the coding, and the appendix's first table says so.** Amendments 4 and
5 were written with all 59 rows in hand. Exactly one moves a number: under Amendment 4a the primary is
$6/59$, without it $5/59$, the single row that turns on it is `2601.06466`, and **both rates are
reported everywhere either is**.

Artifacts: `experiments/pre_registration_literature_audit.md` (the frozen rubric),
`results/literature_audit/coding/*.json` (one record per paper),
`experiments/analyze_literature_audit.py` (every count in the paper is emitted by it, including the
59-row table body, so nothing is transcribed by hand), App. G `app:audit` (rubric, frame, funnel, and all
59 rows with their five codes).

---

## Part 5: *"demonstrated primarily in a relatively constrained FL setting"*: the sharper version of your point

On inspection your objection is **stronger than your review states**, and we say so rather than
defending the table. Before this round the comparability table spanned two datasets and two
architectures. But
**the sign reversal itself does not.** `coord_median` / `committed_pixel`, the only cell where the two
designs' signs differ *and* both intervals exclude zero, exists solely on CIFAR-10 with `cifar_cnn`.
The cell that added a second dataset was `krum` / scaling, a *different* cell, and that cell is
explicitly **not** counted as a reversal because both of its intervals contain zero. So the honest
statement of the July evidence, and of the September evidence before this round, is: **one cell, one
dataset, one architecture.**

Cell 7 moves that exact cell, with the same aggregator, attack, rung grid and seeds, to **CIFAR-100
with `cifar_cnn`**, a 100-way task that changes the backdoor base rate and the clean-accuracy regime
together. Amendment 4 to `experiments/pre_registration_comparability.md` was committed **before the
runner was pointed at it**, and it fixes in advance:

- a **two-sided admissibility gate** read off the first five runs (identity-rung mean clean accuracy
  $\geq 0.35$ and mean ASR in $[0.15, 0.85]$), because a saturated rung kills the *rise* leg exactly as
  EMNIST's 0.027 rung killed the *fall* leg. That is a defect this paper already discloses against itself, and
  reproducing it on a new dataset would be worse the second time;
- **no fallback dataset.** The only other configurations the harness supports cost 29.5 h and 65 h for
  40 runs. If CIFAR-100 fails the gate, the recorded finding is *"no third dataset within this compute
  budget admits this cell"*;
- **no prediction from the withdrawn mechanism.** H-ADMISSION-GATED was refuted by cell 6 and stays
  refuted; cell 7's $\Delta\Lambda_a$ entry is an absent-value cell, not a number, and cell 7 must not
  be used to resurrect it;
- **a null is a result.** If cell 7 agrees, or disagrees without reversing, that is the answer and it
  goes in the table. *"The reversal on CIFAR-10 is not retracted by a null elsewhere, and it is not
  generalised by a hit elsewhere"*. Either way the reversal is **dataset-conditional**, which is a scope
  condition we owe you.

**The run has landed, and the reversal replicates.** CIFAR-100 passed the gate on both sides
(identity-rung mean clean accuracy $0.393$, mean ASR $0.703$, means over seeds 42 to 46), so the
inadmissibility finding was not the one we had to report. On the frozen primary contrast, the
$\kappa{=}0$ to $\kappa{=}2$ endpoint, the outcome-gated ladder **falls** $-0.213$ $[-0.284, -0.141]$
where the within-defense instrument **rises** $+0.071$ $[+0.031, +0.110]$: opposite signs with both
intervals excluding zero, which under the same definition applied to every other row of the table is a
**second sign reversal**. Table A14 in `app:sixcell` now has seven rows, the two designs disagree on
five of them, and two of those disagreements are sign reversals.

Four things we will not let that result be read as more than it is, all of them from Amendment 4
rather than from hindsight:

- **The scope after it is three datasets and two architectures, and no more.** The reversal is
  **dataset-conditional** on exactly that evidence. A hit on a third dataset does not make the
  phenomenon generic and we do not claim it does.
- **The $\Delta\Lambda_a$ entry stays absent, not $0.0000$.** The withdrawn rule would in fact have
  predicted this cell correctly, and that is precisely why we record no prediction for it: a mechanism
  refuted out of sample is not revived by a later cell that happens to agree with it.
- **The cell is scored on the two endpoint rungs, which is exactly what the primary contrast reads,**
  so the row is complete as printed. All four rungs of the frozen grid are now run on both designs, and
  **an earlier version of this bullet said the cell carried no Jonckheere and Terpstra trend statistic
  because its interior rungs were not run.** That was true when written and went false when the runner
  finished, so we correct it here rather than let it stand on a technicality. The trend is **post hoc**
  -- nothing in the comparability pre-registration registers a trend secondary -- and it is reported as
  a description: the outcome-gated ladder falls monotonically ($z = -2.89$, $p = 0.002$) while the
  instrument rises monotonically ($z = +1.41$), so the reversal spans the whole ladder and not just its
  endpoints. **The rising leg does not clear 0.05 as a rank trend** ($p = 0.079$, permutation 0.084)
  even though its endpoint interval $[+0.031, +0.110]$ excludes zero, and we state that next to the
  falling leg rather than quoting only the significant half. The analyzer prints a rung-coverage line
  unconditionally, including when coverage is complete, so no cell's grid can be misquoted in either
  direction.
- **CIFAR-100 was chosen on wall time.** At 660 s per run it was the third dataset that fit an overnight
  budget, not the third that was most informative. We record that as the limitation it is.

One safeguard fired while the run was still live, and we record it because it is the same defect class
as an earlier amendment of this very document. Mid-run, cell 7 held five confounded seeds and four
controlled ones, and the analyzer scored it anyway and printed a row that looked finished: an $n{=}5$
leg against an $n{=}4$ leg, the mixed-$n$ error we had already fixed once, re-entering through a
completeness test instead of through a top-up. A cell whose frozen seed set is not fully landed on
**both** legs now prints as INCOMPLETE, is excluded from every count, and has its LaTeX row withheld
rather than emitted with an empty verdict column. The numbers above are the fully paired $n{=}5$.

One reader-side disclosure, because it moves a number you can check. Cell 7's controlled interval
$[+0.031, +0.110]$ lies inside the pre-registered $\pm 0.15$ equivalence margin **and** excludes zero,
which is a real joint TOST outcome. Scored naively it would have become the binding equivalence arm of
the sensitivity appendix and moved its headline headroom from $+0.116$ to $+0.040$, on the strength of
an equivalence reading the paper never makes about that arm. The margin sweep now requires an interval
to contain zero before it counts as an equivalence reading, the published binding arm is unchanged
(Krum on EMNIST-byclass), and §S12 states the exclusion and its reason rather than leaving the
arithmetic to speak.

---

## What we did not do

- **No compression.** The paper is long and this round made it longer. We judged that a reviewer asking
  for external validity and a measured prevalence is better served by the audit and the third dataset
  than by a shorter paper, and we would rather tell you that was a choice than present the length as
  unavoidable.
- **No front-matter claim from the audit.** The pre-registration made a contributions-list clause
  conditional on a *high* prevalence. The prevalence is low, so the clause is dropped and the audit
  lives in the appendix. We are not putting a $10.2\%$ into the abstract.
- **No resurrection of the withdrawn $\Delta\Lambda_a$ rule**, and no new dataset chosen after seeing a
  gate.

## Still owed to you

Two concerns from the immediately preceding review are unrecoverable from our records and remain
unanswered; that is a gap in our bookkeeping, not a decision. And **if there is any way to ensure the
next reviewer receives the submitted build rather than a July PDF, that single change would improve the
next round's review more than anything in this letter.**
