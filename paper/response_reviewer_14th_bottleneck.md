# Response to the fourteenth review (5/10, Borderline / Weak Reject, confidence 4/5)

We thank the reviewer for a review that does something the previous ones did not: it names **one**
experiment, states a conditional on it, and tells us plainly that anything else we spend the round on
will not move the score. We took that literally. We ran that experiment and nothing else.

**It went against us, and this letter leads with that rather than burying it.**

---

## 1. The named experiment, and its verdict

The ask was *one stronger, non-floor-effect causal intervention with substantially more seeds and less
dependence on oracle adversary identity*, with the conditional *if that experiment reproduces the
central dissociation, I could realistically move the review to 6–7/10*.

We ran it. **It did not reproduce the dissociation. It refuted our pre-registered prediction.** The arm
is now App. D.7 (`:2063`), and the body carries it at `:870`.

| what the review asked for | what the arm did |
|---|---|
| non-floor-effect cell | Krum under the committed pixel backdoor, baseline admitted adversarial mass 0.333 in 4 of 12 adversary rounds, recomputed by the arm itself: `:2070` says "the floor the flagship arm is charged with is the attack's rather than the aggregator's" |
| substantially more seeds | n=20 (seeds 42–61), fixed before the first run with no top-up clause; 60 runs, 17.3 h |
| less dependence on oracle adversary identity | **zero** dependence in the intervention: the upstream transform is RFA, whose coefficients come from the update stack alone, and the two controls are the pre-registered `score_only` / `emit_only` off-diagonal cells, so `:2068` needs "no new transformation class and no new methodology" |
| reproduces the dissociation | **no.** Leg 1 was frozen as \|ΔASR\| *inside* the ±0.15 margin. It reads **+0.2969** (sd 0.4243, paired 95% CI [+0.0983, +0.4955], n=20) — outside the margin *and* excluding zero, "which is the exact conjunction the pre-registration named in advance as contradicting this paper's headline reading" (`:2076`) |

We report it as the freeze requires: **a contradiction, not a scope condition.** The pre-registration
(`experiments/pre_registration_oracle_free_channels.md`, committed alone at `4e7090b` before either
output path existed) named this branch in advance precisely so that we could not relabel it afterwards.
We have not amended the freeze and we have not written the sentence that the negative reproduces
oracle-free anywhere in the paper.

Four things about the number, all in App. D.7:

1. **It is not an averaging artifact.** The cell is bimodal; eight seeds move from the low mode to the
   high one and **none** move the other way:
   "directional seed by seed and not an artifact of averaging" (`:2078`).
   The identity arm has 6 of 20 seeds above ASR 0.5 and 12 below 0.10;
   the statistic arm has 12 above and **none** below 0.10.
2. **The premise passed before any ASR existed.** RFA moves Krum's decision in 8 of 15 rounds (0.533) at
   admitted adversarial mass change 0.000, aggregate displacement 0.868, on a three-way rule frozen
   before its numbers were read and applied to the *published* Mode S/Krum arm first, where it types that
   arm the same way — "so this is not an instrument that condemns only what is new" (`:2074`). So this is
   a valid test that refutes, not a broken test.
3. **Leg 2 is unresolved, which is not a zero.** ΔASR = −0.0167 (CI [−0.0588, +0.0254], n=19; one rung
   void below the 0.35 accuracy floor, and both verdicts unchanged if it is retained). The half-width is
   ±0.042, so it is not evidence of absence. We also say that
   "Its failure was foreseeable from an artifact we already had" (`:2080`), and that we should have said
   so in the freeze.
4. **The arm is not the clean probe the freeze took it for, and we found that out only afterwards.** RFA
   pins nothing: the adversarial share of coefficient mass moves in **12 of 12** adversary rounds, by up
   to 0.0859, against the frozen tolerance of 1e-6 that Mode S holds to 3.5e-7, with implied Δ_c mean
   −0.089 over [−0.537, +0.147]. So App. D.7 states that this arm
   "is not a clean probe of the statistic channel alone" (`:2082`).
   Its refutation therefore does not transfer to the published Mode S result, and the defect is
   ours: our own freeze recorded the tolerance and the runner recorded the quantity every round, and the
   gate was still not written.

**What the arm establishes is narrower than either prediction and is not good news for us.** App. D.7
says "the oracle is load-bearing for our negative rather than a laboratory convenience" (`:2084`), and that we
have not produced an oracle-free measurement of the statistic channel on any cell. The formal claim is
untouched — it is an existence claim over exhibited counterexamples — but the *generalization* a reader
would want is what fails, and App. D.7 tells that reader to read the arm as evidence that the negative
does not travel without an oracle.

**On the conditional itself.** By the reviewer's own terms, the 6–7/10 route conditioned on reproduction
is closed. We are not asking for it. What we claim is narrower: the named experiment was run at the named
scale on the one cell where both of its objections are answerable, its result is pre-registered, and the
paper now states it against ourselves. App. D.7 fixes the scope:
"One aggregator, one attack, one dataset and one upstream defense license no statement" (`:2090`)
of the general form, in either direction.

---

## 2. Two of the three headline weaknesses were answered before this review was written

Said without complaint, because it changes what is worth arguing about. The review was written against
a 70-page file (commit `5a81b04`, 2026-09-13); the current paper is 80 pages, and two answers landed
after that file.

| weakness | status in the reviewed file | status now |
|---|---|---|
| §5: Krum's zero admission may be a floor effect | absent | `:735` carries the receipt: "that floor is the attack's, not Krum's", with baseline mass 0.333 in 4 of 12 adversary rounds under the pixel backdoor against 0 of 12 under scaling. App. D.7 then *hosts* the new arm on that exact cell |
| §4: oracle dependence; one experiment where the same conclusion can be obtained without adversary identity | absent | App. D.6's impossibility at `:2059`: "For this family neutrality and informativeness are mutually exclusive", so "no oracle-free member of Theorem" 9's positive-rescaling class is a clean statistic probe. App. D.7 is the positive attempt, and §1 above is its verdict |
| §10: the audit implies the pathology is widespread | already there | `:2683` already states that "any sentence implying the contrast is rare" in this literature is false against our own audit, with an identifying contrast present in 46 of 59 included papers (78.0%) |

App. D.6 and App. D.7 are the two halves of one answer: the impossibility says where an oracle-free
probe *cannot* live, and the arm says what happens when you build one outside that requirement anyway.

---

## 3. The novelty concession, unhedged

The review's §3 and §14 say Prop. 6's collider logic is standard causal reasoning rather than a new
result. **We agree, and the paper already says so.** We do not claim the identification argument is
novel as causal statistics; we claim it has not been applied to composed-defense evaluation in this
literature, and the audit is the evidence for that second claim rather than for the first.

That puts the novelty burden on the empirical arms. We accept it, and we accept that App. D.7 — the
strongest of them by seed count and the only oracle-free one — **came out against the reading it was
built to extend.** That is the honest state of the contribution as of this revision.

---

## 4. Declines, each with its site

| review item | disposition |
|---|---|
| §4 deployability of the instruments | **Declined and disclosed, not answered.** The controls reach inside the aggregator, so App. D.7 remains an instrument. `:1010` now separates the two reasons our instruments are "measurement devices, not deployable defenses" — Mode S and the payload-share instrument need the mask, score-only Krum does not but modifies the aggregator — which is a correction this round's arm forced |
| §6 the sign reversal is not a pure statistic-only intervention | **Already disclosed in the paper's own words**, and App. D.7 is the clean version the review asked for. Its post hoc measurement (item 4 of §1) shows that even the oracle-free version is not pure, and we state that rather than let the arm stand as one |
| §7 / §16.3 demote the screening criterion | **Already maximal.** It is appendix-only and its own heading calls it "an evaluation prioritizer, never a security screen" (`:1230`). We added nothing to it this round, on the review's own advice |
| §9 Theorem 8 is supporting, not major | **Already so ranked** in Table A2 (`:1183`), whose Formal row carries the scope conditions. We flag a genuine cross-review tension: review 13 asked us to state the positive direction and we did; review 14 asks us not to overclaim it. The existing text carries both limits in the same breath |
| §10 audit prevalence | **Already exactly as asked** (`:2683`) |
| §11 adaptive attacks | **Already the scope box's first denial** (`:470`: "about adaptive robustness, our instruments read adversary identity"), and §4 states the boundary as "attack-suppression preservation under a fixed committed attack and nothing wider" (`:639`), with one whitebox adaptive stress test measuring it |
| §12 elevate FoolsGold→RFA | **Already the body's second paragraph** (`:160`) and the abstract's last sentence |
| §13 / §16.5 meta-narration; reduce dramatically | **Declined with two measurements.** The main text is 9 pages — the 70 pages the review counted are appendix — and the body's research-history framing gate reads 0. The narration is appendix-only, and appendix disclosures are the paper's audit trail; we do not delete them to shrink a page count |
| §16.1 reorder the abstract | **Done.** The identification claim now leads: `:72` opens with "identify whether the downstream mechanism caused it". Character-neutral |
| §16.2 the sign reversal is buried | **Already on pp. 1–2** and Figure 1 is the Mode S causal figure |
| §16.4 three claims as an organizing principle | **Already Table A2 plus the scope box's three denials on p4.** Not promoted further: the main-text page budget has no slack, and the new body clause at `:870` had first call on it |
| §8 n=3 / n=5 cells | **Partly answered.** App. D.7 is n=20. The cell-7 seed top-up stays written and unrun, disclosed rather than promised |
| expanding the screening criterion; more defense/attack combinations | **Not done, on the review's explicit advice** that neither would move the score |

---

## 5. What changed in the paper this round

* **New App. D.7** (`:2063`), 13 paragraphs: the design, the host cell's headroom, code-enforced
  oracle-freeness, the prospective premise, both frozen verdicts read separately as sign and margin, the
  distribution behind the +0.297, the void rung, the post hoc Δ_c measurement, what the arm establishes,
  the bit-level control assertions ("neither control is taken on its docstring", `:2086`), three harness
  checks whose verdicts are persisted in the artifact ("All three verdicts are persisted", `:2088`), and
  what the arm does not establish.
* **One body clause** folded into the paragraph that already carried the floor receipt, reading
  "removing the oracle reopens attenuation, so the negative is not established without one" (`:870`).
* **Abstract reordered** so the identification claim leads (`:72`), character-neutral. Its one-clause
  account of the oracle question was also corrected in the same line, because it had been written for
  App. D.6's premise failure alone and now covers two attempts with different failure modes. It reads
  "two oracle-free attempts did not carry, one refuting our prediction" (`:72`). Without that last word
  the abstract would have called a refutation inconclusive.
* **Ethics statement corrected** (`:1010`) because the new arm falsified it: score-only Krum does not
  need adversary identity, and the statement said all three instruments do.
* **One additive keyword** on a shared runner, proven bit-neutral against a frozen published cell at
  Δ = (0, 0), with the verdict written into the artifact rather than left in a terminal.
* No existing runner, pre-registration, results directory, frozen threshold or published verdict was
  edited. One commit was made: the pre-registration, alone, before either output path existed.

## 6. What we did not fix

* The oracle-free measurement of the statistic channel does not exist on any cell, including this one,
  and the paper now says so.
* Our own pre-registration omitted the coefficient-share gate that App. D.6 uses to demote an arm on
  exactly this condition. The freeze recorded the tolerance and the runner recorded the quantity; the
  gate was still not written, and by the paper's own precedent this leg would have been typed confounded
  in the attenuation channel had it been included.
* The premise's admission-flatness measurement is 15 rows at rounds 0–2, not the 50-round ladder behind
  the ASR figures, and App. D.7 says which claim rests on which.
* No composed-pair ASR result on a second dataset exists anywhere in this paper; this arm is on CIFAR-10
  and does not change that.
