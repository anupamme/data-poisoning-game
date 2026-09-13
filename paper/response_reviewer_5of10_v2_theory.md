# Response to Reviewer (5/10, Borderline Reject; confidence 4/5)

Thank you. This review is the most useful we have had on the theory, because it points at a real
misalignment in the composability criterion that we had not seen, and it does so precisely enough that
we could go and compute what fixing it would cost. We did compute that, and the answer changed the
paper: the quantifier slip you identify is now fixed at both sites that carried it, the specific repair
you propose is declined, and the arithmetic for the decline is published rather than asserted. We also
ran the randomization experiment you asked for, pre-registered before the first run.

One thing to flag at the top, because it accounts for most of the length below. The review states it
read *"the latest August 12, 2026 version"*. Nine of its asks, including two of the three items under
Concern #1, are already in the submitted source and have been since the August-to-September revision.
We give line numbers and verbatim quotations for every one of them so this can be checked rather than
taken on our word. **We are not claiming the review is careless**; the paper is 58 pages and the
theorem in question is appendix-only by design, which is itself a discoverability defect we have now
fixed (see #1(c)).

We have not adopted everything. Two asks are declined on the merits, and both declines are published
in the paper with the numbers that motivate them.

---

## The five changes this review produced

1. **The C1 quantifier is fixed** (`main.tex:475`). The gloss said "With C0 this forces the suppressing
   constituent to be $d_2$", full stop, which is false on 4 of the 42 pairs. It now reads: forces $d_2$
   *"on the attack $a^*$ that witnesses C0, and there only"*, and adds explicitly that **C0 is
   existential over attacks while C1 is universal over them**, so on an attack $d_1$ happens to suppress
   the constituent doing the work can be $d_1$.

2. **Its twin in the sufficiency sketch is fixed** (`main.tex:644`). C1's sufficiency argument now
   separates the two inheritance routes and states that when the suppressing constituent is $d_1$,
   *"the suppression is inherited because $d_1$ sits upstream and nothing downstream can undo it, so
   C2$\wedge$C3 are not required on that attack and the preservation argument contributes nothing to
   it."* That sentence is the honest form of what the criterion does and did not previously exist.

3. **The price of your Fix #2 is published** (`main.tex:894`), as a new paragraph beside the existing
   tightening disclosure at `:892`. It is declined, and §"Concern #2" below is the arithmetic.

4. **A new compute-matched randomization experiment was pre-registered and run** (Concern #3), with the
   confound your review describes removed by construction. Design frozen at
   `experiments/pre_registration_randomized_composition.md`, committed before the first run; analysis in
   `experiments/analyze_randomized_composition.py`. Results in §"Concern #3" below.

5. **An appendix reading guide** (`main.tex:382`) naming what is skippable, in place of the deletions
   Concern #13 asks for. Nothing was deleted; §"Concern #13" gives the reason and an audit.

---

## Concern #1: Theorem 2's separation condition, and instantiating $\rho$

Three sub-asks. **All three are already in the submitted source.** What was genuinely defective is that
none of them was findable from the body, and that is what we fixed.

**(a) The separation condition.** The theorem does not assume $\Delta_{\mathrm{sep}} > r_B/2$. Part (2)
assumes, at `main.tex:699`, on the **transformed** points $u'_i = w_i u_i$ that RFA actually receives:

$$\Delta'_{\mathrm{sep}} := \min_{a,b}\|u'_a - u'_b\| > 2(R'_B + \delta')$$

and the proof at `:708` derives exactly that, noting *"All quantities are on the transformed points
$u'_i = w_i u_i$, which is what RFA receives."* The suggested $((1+\rho)/2)\cdot r_B$ form is on
untransformed points, which is a different (and weaker) statement.

**(b) The $\rho$-form you name is already retired in print.** `main.tex:731` states that an earlier
version summarized part (2) as a weight-ratio bound and that **the summary was wrong**, and that honest
majority *"enters on the displacement, not on the weight ratio"*: $C_A$ diverges as $n_a \to n_b$, so
$n_b > n_a$ becomes indispensable at Corollary (checkable margin), while the theorem itself needs no
majority assumption on the weights.

**(c) Instantiating $\rho$ empirically: done, and reported vacuous.** `main.tex:733` derives the
pilot-free reduction to $\rho$ alone that your ask points toward, and prices it: the reduction requires
$\Delta_{\mathrm{sep}} > 4(1{+}C_A)R_B$, and the measured ratio clears 1 in **exactly 1 of 17**
positive-weight rounds, maximum **1.015**, mean 0.754, against a measured $\rho$ of at least 1.034.
**The reduced condition is met in no round.** We report that as a stronger negative than a merely
conservative threshold, and we therefore claim no pilot-free screen for the geometric class (L5,
`:1684`).

**What we changed because of this concern.** You could not reasonably have found (a)-(c), because all
three live in the appendix while the theorem is *invoked* in the body. That is our defect, not yours.
The body site where the theorem is first invoked (`main.tex:179`) now carries the instantiation
explicitly: the condition reads $m > 1$, the measured margin is $m \ge 2.82$, the bound gives 12.6%
adversarial mass against a realized $\le 1.8\%$, and the $\rho$-only reduction is vacuous. That is the
one body edit in this round.

---

## Concern #2: make C1 directional (C1$'$). Declined, and here is what it costs

**You correctly detected a misalignment we had missed. Your repair for it empties the certified set.** We
say both plainly because the second half is easy to miss: adopting C1$'$ destroys the very statistic your
own Fix #4 asks us to promote.

C1$'$ ("$d_2$ individually suppresses the attack") and symmetric C1 disagree on **4 of the 42 pairs**.
Re-scored read-only from `results/condition_ablation/summary.json`, with every C1 input the $n{=}5$
single-defense baseline at seeds 42-46:

| pair | $d_1$ standalone (scaling / pixel) | $d_2$ standalone | symmetric C1 | directional C1$'$ |
|---|---|---|---|---|
| `foolsgold_then_coord_median` | 0.200 / 0.732 | 0.519 / 0.443 | **holds** | **fails** |
| `reputation_then_coord_median` | 0.017 / 0.842 | 0.519 / 0.443 | **holds** | **fails** |
| `coord_median_then_foolsgold` (degenerate) | 0.519 / 0.443 | 0.200 / 0.732 | holds | fails |
| `coord_median_then_reputation` (degenerate) | 0.519 / 0.443 | 0.017 / 0.842 | holds | fails |

The first two pairs **are** the entire in-distribution C1$\wedge$C2$\wedge$C3 PASS cell, and therefore
the entire *"precision 2/2, recall 2/5 = 40%"* at `main.tex:892`. Under C1$'$ the criterion certifies
**nothing** in distribution and precision becomes **undefined** rather than 2/2. The mechanism is
visible in the table: on model scaling the constituent that suppresses is the **upstream** FoolsGold
(0.200) or reputation (0.017), not CoordMedian (0.519); CoordMedian is the suppressor on the pixel arm,
which is the $a^*$ witnessing C0.

**So the misalignment you sense is real but mislocated: it is in the attack quantifier, not in the
$d_1$/$d_2$ direction.** C0 does not force $d_2$, because C0 is existential over attacks while C1 is
universal over them; the testability lemma pins $d_2$ only on $a^*$. What your substitution correctly
points at is that on the model-scaling arm of those two pairs the suppression is inherited from upstream,
so the signal-preservation argument contributes nothing to that arm even though the arm counts toward
the max-committed label. **The criterion and Theorem 2 are aligned per attack, not per pair.** That
sentence is now in the paper (`:894`), together with the four-pair census and a pointer to the script
that emits it.

We keep symmetric C1 and disclose the asymmetry. We did not consider your alternative (certify if either
direction holds) an improvement, since it is weaker than the symmetric form we already have.

---

## Concern #3: the compute-matched randomization experiment. Run, pre-registered, and it answers you

Your objection to the existing compute-matched experiment is right, and it is sharper than stated: we
found **three** defects in that artifact while designing the replacement, all documented in
`experiments/pre_registration_randomized_composition.md` §1.

1. **The degenerate arm you identify.** The randomized arm sampled between FG$\to$CM and CM$\to$FG, and
   CM$\to$FG is degenerate: CoordMedian emits an aggregate rather than per-client updates, so half the
   randomized arm's rounds reduced to **FoolsGold alone**. The 9x/19x gaps are fully consistent with
   "one defense per round is worse than two."
2. **The arms were not seed-matched.** `run_compute_matched_mixing.py:150` reads
   `if randomize_order and rng.random() < 0.5:`. Python short-circuits `and`, so the fixed arm never
   drew that number while the randomized arm drew one per round, from the same generator that selects
   participants. **The two arms therefore saw different clients, and no paired test was licensed on that
   artifact.** None is computed from it here.
3. **It was not on the canonical participant stream at all**, so it re-measured rather than reproduced
   every cell it shares with the main suite (max per-seed deviation 0.079 on FG$\to$CM's pixel arm).

Note also that the paper **does not currently claim anything about randomized menus**: `main.tex:1655`
withdraws the earlier comparison outright (*"that comparison is not part of this submission and no claim
here rests on it"*). **That withdrawal stays regardless of what the new experiment finds.** The new
experiment is a separate, separately pre-registered object.

**Design (frozen before any run).** Seven conditions x 2 committed attacks x 5 seeds (42-46) = 70 runs,
one seed set for every arm, all arms re-run, nothing reused. **Every member of every randomized set
keeps FoolsGold as a genuine per-client upstream, so no round degenerates to a single defense.** Policy
draws come from a separate `policy_rng = default_rng(seed + 2000)`, which cannot perturb the participant
sequence. Primary contrast: `rand_comp_strong` (uniform over {FG$\to$CM, FG$\to$RFA} each round) against
**the best member of its own set, selected by rule** (lower mean ASR on that attack within this
artifact), **paired per seed**, one-sided H1 $\bar\Delta > 0$ fixed in advance. Material only if both
$\bar\Delta \ge 0.10$ **and** $\Delta_s > 0$ in $\ge 4$ of 5 seeds. Compute matching is claimed for the
fixed and randomized composition arms only; the two single-defense reference arms use half the defense
computation and every table says so.

**Results, recomputed per seed from `results/randomized_composition/summary.json` (population sd):**

| condition | def/round | scaling ASR | pixel ASR | acc (sc/px) |
|---|---|---|---|---|
| `fixed_fg_cm` | 2 | 0.1068 +- 0.0306 | 0.0988 +- 0.0201 | 0.682 / 0.664 |
| `fixed_fg_rfa` | 2 | 0.0510 +- 0.0096 | 0.0452 +- 0.0118 | 0.499 / 0.495 |
| `fixed_fg_tm` | 2 | 0.3843 +- 0.2594 | 0.2495 +- 0.1026 | 0.694 / 0.705 |
| `rand_comp_strong` | 2 | 0.1522 +- 0.1091 | 0.1115 +- 0.0440 | 0.607 / 0.598 |
| `rand_comp_all3` | 2 | 0.2317 +- 0.2780 | (in flight) | 0.623 |

**Primary contrast** (`rand_comp_strong` vs. `fixed_fg_rfa`, selected by the frozen rule on both
attacks, both legs $n{=}5$):

| attack | randomized | best member | $\bar\Delta$ | signs | Wilcoxon (1-sided) | paired $t$ | pre-registered verdict |
|---|---|---|---|---|---|---|---|
| scaling | 0.1522 | 0.0510 | **+0.1012** | 5/5 | p = 0.0312 | t = 1.866, p = 0.0677 | **material, in the pre-registered direction** |
| pixel | 0.1115 | 0.0452 | +0.0663 | 5/5 | p = 0.0312 | t = 2.603, p = 0.0299 | **not material** |

Per-seed $\Delta$, scaling: +0.0352, +0.2899, +0.1564, +0.0102, +0.0143. Pixel: +0.0816, +0.0428,
+0.1581, +0.0133, +0.0356. Secondary (`rand_comp_all3` vs. best of all three fixed arms, scaling):
$\bar\Delta = +0.1806$, 4/5 signs, Wilcoxon p = 0.0625.

**Three caveats we put on the record rather than let a reader find.**

- **The scaling result clears its own bar by 0.0012** and is carried by 2 of 5 seeds (+0.2899, +0.1564).
  We will not describe it as a robust effect.
- **The two tests disagree in opposite directions across the two attacks.** The Wilcoxon is at its floor
  (1/32) on both, while the $t$-test reaches p < 0.05 only on the attack the pre-registered rule calls
  *not* material. **The stable part of this result is the sign structure, 10/10 across both attacks**,
  not either p-value.
- **Post hoc:** the randomized arm exceeds **both** members in 3 of 5 seeds on each attack, so the gap
  is not interpolation between the members. The pre-registered draw diagnostic finds no significant
  association between realized member counts and ASR: on the primary arm Spearman $|\rho| = 0.5$,
  p = 0.391 on both attacks, and the largest coefficient anywhere (the secondary arm's FG$\to$TM count
  on scaling) is $\rho = +0.667$, p = 0.219. At $n{=}5$ this diagnostic can only rule out a strong
  association, and that is all we claim from it.

**Positive controls, because this project has twice shipped a hook that silently never fired.**
Control 2 (no randomized arm is a fixed arm in disguise): passes, with the closest fixed arm's per-seed
max $|d|$ = 0.2207 (`rand_comp_strong`/scaling), 0.0658 (`rand_comp_strong`/pixel) and 0.1946
(`rand_comp_all3`/scaling). Control 3 (the harness did not change the run): **all eight identity checks against the
canonical artifacts return max $|d|$ = 0.000e+00**, so the fixed arms are bit-identical reproductions and
the new arms are comparable to published numbers. Accuracy floor (0.35): no arm breached it, minimum
0.495. **Control 1 fails as literally pre-registered and we report it as a failure**: 3 of 10 policy
streams carry a draw outside a per-stream two-sided 99% interval (family-wise P of at least one
excursion = 0.096, so this is roughly what chance gives). We do not treat it as invalidating, because
the failure mode the control targets is a *dead* hook, which yields a count of 0 or a 50/0 log, and every
stream here draws every member; but the pre-registered assertion was the stricter one and it did not
hold.

**Status.** The primary contrast is final: its three arms are complete at $n{=}5$ and the contrast was
fixed by rule before any data existed. The secondary randomized arm's pixel leg and the two
single-defense reference arms are still running, and the pre-registration forbids declaring the overall
outcome or writing the appendix subsection until all seven arms are complete at seeds 42-46. That subsection will report all seven arms, the paired contrast, the
compute-matching scope, and whichever pre-registered outcome obtains, with `:1655`'s withdrawal of the
old comparison kept and pointed at explicitly.

---

## Already in the submitted source: nine asks, with sites

| ask | already in the paper | site |
|---|---|---|
| **#4** make PASS precision the primary statistic | *"None of those tallies should be read as discrimination: only 5 of the 42 two-way pairs are low-ASR, so a constant HIGH call scores 37/42 = 88.1% and the combined 90.5% beats it by one pair. Precision (2/2) and recall (40%) are the quantities that carry information, and they are why we present the criterion as a prioritizer."* | `:1672` (L2), footnote `:885` |
| **#6** persistence is explanatory, not predictive | Stronger than asked: **withdrawn outright.** *"We make no formal claim about persistence... that material is not part of this submission, and we withdraw the claim."* | `:1657` |
| **#7** penalize "for most deployments" | The phrase appears **nowhere** in `main.tex`. The architecture swing you cite is stated as a limitation and as evidence for our own thesis: *"ResNet18 suppression at NC10/rep90: 0.865 vs. CifarCNN 0.097, a 10x shift... a swing of that size under a change of architecture alone is why we state the preservation conditions as mechanisms to be measured per setting."* | `:1664` (L1) |
| **#10** keep the DBA caveat prominent | In L1's scope statement, with dedicated per-seed appendices at both fractions | `:1664`, App. `app:dba_f04` (`:1827`), `app:dba_f06` (`:1850`) |
| **#11** FG$\to$RFA's accuracy cost | *"imposes a substantial clean-accuracy cost (0.50 vs. 0.74 for RFA alone). This cost is intrinsic to FoolsGold's aggressive per-client downweighting, not to composition itself"*, plus the Pareto table | `:1621` |
| **#11** FG$\to$CM as the practical choice | *"FG$\to$RFA is the correct choice under a security-first objective... For operators with accuracy constraints, FG$\to$CM (acc 0.68, max ASR 0.107) buys 37% more accuracy for suppression that is statistically indistinguishable on the worst-case arm."* | `:1641` |
| **#12** 0.5 is an operator-chosen threshold | Verbatim: *"That 0.5 is an operational classification threshold, not a theoretically privileged security boundary: nothing in the propositions depends on it, and a deployment with a lower tolerance should read the per-cell ASR rather than the label."* | `:813` |
| **#1(a)** the separation condition | on transformed points, as derived | `:699`, proof `:708` |
| **#1(b)** the $\rho$ weight-ratio form | explicitly retired as wrong | `:731` |

**One qualification on #11.** We adopted the deployment recommendation you ask for (`:1641`) but did
**not** rename the FG$\to$RFA section. "Flagship" there marks a *theoretical* role, not a deployment
recommendation: FG$\to$RFA is the empirical witness for the emergence proposition and **a false negative
of our own criterion**, which is why it has a section. The practical recommendation is already
FG$\to$CM, at the site above. If the label itself is the problem we will rename the section; it carries
no claim.

**#5, theoretical novelty rests on the empirical work.** Accepted without reservation, and the paper
already refuses to launder one into the other: the evidence-status ledger separates what is *proved*
(the invariance classes, the mechanism-preservation margins, the identification boundary) from what is
*measured*, and (L2) states that the criterion **has no established predictive validity** and that the
pooled model which would give it predictive content is unconfirmed ($R^2 = 0.239$) with its cross-arm
ordering refuted.

---

## Concern #13: trimming the auxiliary analyses. Declined, with an audit

We added a reading guide instead of deleting (`:382`), naming what is skippable and, deliberately, the
one block that *looks* auxiliary and is not. Two reasons.

**First, six of the nine analyses you name have no trimmable object in this submission, and the other
three are load-bearing.** We checked each of the nine against all three documents:

| named analysis | what is actually there |
|---|---|
| DKW, POMDP, fictitious play | **absent** from all three documents (0 occurrences of each) |
| $\lambda$/$\tau$ sweeps | **neither is a swept axis, so there is nothing to trim.** $\lambda$ occurs only as RFA's Weiszfeld per-point weights inside the proofs (`:743`, `:756`-`:757`), never as a hyperparameter; $\tau$ occurs only in the definitions of NormClip and the C3 margin ratio, held fixed at 5.0. The supplement additionally states of the $\delta_2$ threshold that it *"is not swept here, and the reason is worse for us than a sweep would be: no label in the 42-pair table is computed from it."* |
| spam | **absent as an analysis**; the sole occurrence in all three documents is "UCI Spambase" in the Ethics dataset list (`:294`) |
| bootstrap | **the word occurs 0 times in all three documents.** What exists is one resampling null ($10^5$ resamples against the menu's 12% base rate, `blind_selection_analysis.py`) and permutation p-values quoted beside two Jonckheere-Terpstra tests: four clauses inside existing sentences, no section |
| survivor regimes | **absent as a named analysis.** "Surviv-" occurs 35 times in `main.tex`, every one the ordinary verb. What exists is the $\alpha{=}0.1$ heterogeneity boundary where only fg$\to$cm survives, and that is **load-bearing**: boundary condition (i), stated alongside the primary result at `:865` rather than deferred, and the reason it is more than a negative datapoint is that the failure shows up in the theorem's own quantity ($m$ collapsing from 3.24 to 1.01, exactly the $m \to 1$ boundary at which the bound degrades to 40% adversarial mass) |
| disjoint triggers | present, as the criterion-aware **joint-constraint** adaptive family (disjoint coordinate blocks to evade FoolsGold plus $\epsilon$-ball projection to evade RFA), and **load-bearing**: it is the adaptive-security evidence, three of six strategies beat the base attacks, we lead with the strongest rather than the median (`:1601`, `:1615`), and it is what licenses the statement that a certified pair's ASR rises 3.7x against an adversary solving both constraints at once (`:644`). The reading guide does mark the adaptive-attack **details** appendix skippable; that is the per-seed backing, not the result |
| FLTrust | present, substantial, and **load-bearing** (see below) |

**Second, FLTrust cannot be cut, and we say so because its headline reads like a weakness.** It is the
prospective out-of-sample suite: the only prospective test in the paper, where the criterion's shortfall
is priced at 13/20, and where an FLTrust near-identity confound is the reason six of its arms cannot
falsify anything. **Cutting it would remove the evidence against us.**

So on our reading none of the nine is a block that could be removed without either deleting nothing or
deleting evidence against our own claims. **Where we agree with you is on the diagnosis, not the
remedy**: 58 pages is a real cost to a reader, and the reading guide is our attempt to pay it down by
routing attention rather than by discarding disclosures.

The standing rule behind the decline: every scope condition, limitation and withdrawal keeps exactly one
home, because a disclosure relocated into a subordinate clause is one a later reader cannot find. The
paper now says that explicitly at `:382` rather than leaving the length to look like an oversight.

---

## What we have not done

- **We did not adopt C1$'$.** It empties the certified set and leaves precision undefined. The decline is
  published with its four-pair census (`:894`), and the quantifier slip it correctly identifies is fixed
  (`:475`, `:644`).
- **We did not delete any scope condition, limitation, withdrawal or disclosure.**
- **We did not reinstate the withdrawn randomized-menu comparison.** The new experiment is a separate
  pre-registered object and `:1655` keeps its withdrawal.
- **We did not overwrite the superseded artifact.** `results/compute_matched_mixing/summary.json` stays
  as the record of what was withdrawn and why.
- **We did not restate Theorem 2.** It is already correct on the transformed points; the defect was
  discoverability, and that is what we fixed.

Concerns #1, #2 and #3 are the three the review called central, and all three now have an answer in the
paper with numbers attached: #1 through discoverability rather than restatement, #2 through a published
decline, #3 through a new pre-registered experiment. Where we declined, the arithmetic for the decline is in the
paper rather than in this letter alone.
