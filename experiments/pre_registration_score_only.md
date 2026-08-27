# Pre-registration: the score-only control (magnitude channel closed, Krum, CIFAR-10)

**Frozen before any score-only ASR exists.** `results/score_only/` does not exist at the time this file
is committed. `experiments/run_score_only_control.py` refuses to start until this file is git-committed
and its `PREREG_COMMIT` is set to that hash.

## 1. What is being tested, and why the published arm does not already test it

The paper's central negative lives in one cell: Mode S into Krum under model-scaling on CIFAR-10, where
the dose changes Krum's decision in up to 80% of rounds, changes the adversarial mass Krum admits in
exactly 0% of rounds, and does not move suppression (published Δ = **−0.026**).

**The objection this arm answers.** Mode S pins every adversary's coefficient at `c_adv = 1.0` exactly,
which closes the *attenuation* channel by construction. It does **not** hold the benign coefficients
fixed, and it was never claimed to. So the dose also changes:

  (a) Krum's pairwise distances, hence the statistic it reads;
  (b) *which* benign client Krum selects — measured at 0.533 / 0.800 / 0.733 of rounds at κ = 0.5/1/2;
  (c) the **magnitude and geometry of the update that is actually aggregated**, because the selected
      client's update enters the model scaled by its own coefficient `c_sel`;
  (d) and therefore the model trajectory.

Channel (c) is already measured and already published: the `Δ agg.` column of the paper's channel table
is `||agg(T(U)) − agg(U)|| / ||agg(U)||` (`experiments/build_channel_table.py`), and for this cell it
reads **0.892** at κ = 2. The published arm therefore establishes that *statistic and decision
disturbance, together with a 0.892 displacement of the emitted aggregate, do not move suppression when
the adversarial coefficient share is held fixed.* It does **not** establish anything with (c) closed.

**This arm closes (c).** Krum **scores** on the transformed stack — so the statistic and the decision are
disturbed exactly as before — but **aggregates the untransformed selected update**. The update entering
training is then bit-for-bit the one a client produced. The dose can change *which* client contributes
and nothing about *what* that client contributes.

This is the reviewer-suggested control stated in its own terms: *hold the selected benign update's
magnitude and direction fixed while perturbing the statistic used for selection independently.*

## 2. The intervention, precisely, and why it cannot silently change the published arms

`experiments/run_all_compositions.py:generic_compose` gains a **defaulted** `score_only=False`
parameter, threaded through a defaulted `score_only=False` on
`experiments/run_targeted_dose.py:run_one`. When it is `False` — every pre-existing call site — the
computation is unchanged. When it is `True` and `d2 ∈ {krum, cos_krum}`:

```
transformed = apply_d1_transform(updates, d1)          # the dose, unchanged
sel, _      = krum_selection(flatten(transformed), cosine=(d2 == "cos_krum"))
return        server.aggregate([updates[sel]], method="fedavg")   # UNTRANSFORMED
```

`krum_selection` is imported from `experiments/verify_cos_invariance.py` — the same single-sourced mirror
`experiments/measure_admission.py` uses — so this control and the channel measurement cannot drift apart
in what they call "the statistic". The final line is `_fedavg` over the single selected update, exactly
as `fl_core/federated.py:_krum` ends, so the returned object takes the same form and dtype path as the
uncontrolled arm's.

**Verified before this file was committed.** `python3 experiments/run_targeted_dose.py --harness-check`
passes on all five frozen arms with a largest absolute deviation of **exactly 0.00e+00** against a
1e−6 tolerance (mode S krum/reputation/cos_krum, mode A reputation/coord_median, seed 42), so every
pre-existing computation is bit-identical after the kwarg was added. And on a synthetic five-client
stack, with no training: at κ=0 `score_only=True` is bit-identical to the uncontrolled arm for **both**
selectors; at κ=2 the returned object is **verbatim** `updates[sel]` for the `sel` that
`krum_selection` picks on the transformed stack, and differs from the uncontrolled arm's output; and
`reputation`, `coord_median` and `rfa` each raise `ValueError` rather than silently doing something
else.

## 3. The primary rule

On **Δ = mean ASR(κ=2) − mean ASR(κ=0)**, n = 5 seeds {42, 43, 44, 45, 46}, `krum` /
`committed_scaling` / CIFAR-10 / `cifar_cnn`, rungs κ ∈ {0, 0.5, 1, 2} (ρ = 1, 2.72, 7.39, 54.60),
against the suite's existing `EQUIV_MARGIN = 0.15`. **No new constant is introduced**; the margin, the
rung ladder, the accuracy floor and the trend test are all imported from the frozen suite's own scorers.
The published Mode-S Δ is recomputed from `results/targeted_dose/summary.json` per seed at run time,
never transcribed.

| outcome | verdict |
|---|---|
| \|Δ\| < 0.15 | **THE NEGATIVE SURVIVES WITH THE MAGNITUDE CHANNEL CLOSED.** Substantial statistic and decision disturbance does not produce a corresponding change in suppression even when the aggregated update's magnitude and direction are held exactly fixed. The central claim is then instrumented against two independent confounds — attenuation (Mode S) and magnitude (here) — rather than one. |
| Δ > +0.15 | **THE PUBLISHED FLAT RESULT WAS PARTLY AN ARTIFACT OF THE MAGNITUDE CHANNEL.** With magnitude closed, statistic disturbance *does* move suppression. The paper's central claim narrows to "with benign magnitudes free to move", stated in the body next to the result, not in a limitation. |
| Δ < −0.15 | A fall with the attenuation channel already closed and magnitude also closed. **INDETERMINATE**, reported as such and **not** scored in our favour. |

**Headroom of the two branches, disclosed with the rule rather than after it.** The identity rung is
Krum alone at mean ASR **0.0618** (recomputed from the five per-seed rows: 0.1149, 0.0209, 0.0469,
0.0567, 0.0694). So the **refuting** branch needs mean ASR(κ=2) > **0.2118** against a ceiling of 1.0 —
fully reachable, and 3.4× the identity rung. The **indeterminate** branch would need mean ASR(κ=2) <
−0.088, which is unreachable; that branch cannot fire on this cell, and equivalence here is to the
identity rung, not to a low absolute ASR.

## 4. Secondary and gates

- **Secondary (trend):** Jonckheere–Terpstra across all four rungs, specified to detect a monotone
  **RISE**. That is why the arm runs 4 rungs and not 2. **Pre-registered caveat:** at n = 5 the JT test
  is weak; a null from it is **not** evidence of flatness and will not be reported as such. A downward
  trend is not the direction this test was specified to detect and will be reported, not scored.
- **Accuracy gate:** every rung must hold mean clean accuracy ≥ `ACC_FLOOR = 0.35`. If any rung fails,
  the cell is uninterpretable and **no verdict stands** — a low ASR at collapsed accuracy is not
  suppression. (The uncontrolled arm's four rungs sit at 0.565 / 0.665 / 0.692 / 0.637, so the gate is
  not expected to bind; expectation is not a substitute for the check.)
- **Harness sanity, explicitly not a hypothesis test:** at κ = 0 the dose returns the update stack
  unwrapped, so score-only Krum *is* Krum alone and the κ=0 rung must reproduce the published identity
  rung **bit-identically**. It is therefore imported from `results/targeted_dose/summary.json` rather
  than re-run, and `--harness-check` verifies the claim at one seed. A gap of tens of points means the
  two paths disagree about the protocol and **the arm is void rather than interesting**.

## 5. The `cos_krum` corollary: an assertion checked at one seed, not a five-seed arm

`cos_krum`'s statistic is exactly invariant under positive per-client rescaling (Proposition 1(a)), and
the paper measures this: **0 of 120 rounds change selection, maximum relative score drift 1.1e−4**. So
under score-only `cos_krum` the selection is unchanged at every rung *and* the aggregated update is
untransformed — the trajectory must therefore be **bit-identical to κ=0**, and Δ must be exactly 0.000.

This is a **test of the paper's existing explanation**, not a new result. The uncontrolled `cos_krum` arm
falls **0.173** by the first gate-passing rung with a decision change of exactly 0.000, and the paper
attributes that fall to the **magnitude** channel — the only channel left open. Score-only closes it, so:

- **Bit-identical trajectory (Δ = 0.000 exactly):** the magnitude attribution is correct, and the
  0.173 fall is confirmed as a magnitude effect.
- **Not bit-identical:** the magnitude attribution is **wrong**, and that is reported in the body in the
  same paragraph, with the same prominence, and the 0.173 explanation is withdrawn.

Because the prediction is bit-identity rather than a mean over seeds, it is checked at **one seed** and
costs one run, not five. This is decided now, before the check, so that a null cannot later be sold as
a five-seed replication.

## 6. Declared limitations, before the run rather than after

- **The control is defined for selectors only.** A weighted averager (`reputation`) and a
  coordinate-wise order statistic (`coord_median`) emit no single selected client, so there is no "the
  selected update" whose magnitude could be held fixed. `generic_compose` **raises** for them rather
  than silently doing something else. This arm therefore covers Krum, and the `coord_median` arm that
  carries the +0.098 rise is **not** controlled for magnitude. Say so in the paper.
- **Score-only Krum is not a defense.** It requires aggregating an update the scoring stage did not see,
  which no deployment would do, on top of Mode S already reading adversary identity to pin `c_adv = 1`.
  This is an instrument. Nothing here is proposed for deployment.
- **One cell.** One dataset, one architecture, one attack, one defense, n = 5. This closes one channel on
  the flagship cell; it is not a breadth claim and does not extend to FEMNIST, which is not re-run.
- **Closing (c) does not close (d).** The model trajectory still diverges across rungs, because *which*
  client is selected still changes. No design that leaves the decision free to move can hold the
  trajectory fixed, and this one deliberately leaves the decision free — that is the whole point.
  The claim this arm licenses is about magnitude, not about trajectory.
- **15 new runs, ≈ 6.9 h.** κ=0 is imported bit-exactly, so 3 rungs × 5 seeds are new compute. The
  per-run cost is taken from the five runs of the harness check above, on this machine and this
  configuration: 1658 s / 1737 s / 1600 s / 1692 s / 1601 s, mean **≈ 1658 s**. An earlier estimate of
  ~730 s/run, read off an unrelated sweep log, was wrong by 2.3× and is corrected here rather than
  after the fact; the Section 5 corollary adds 4 runs, ≈ 1.8 h. This is a budget decision made before
  any result exists, and the budget is not a reason to shorten the ladder or drop a seed.

## 7. Non-negotiables

1. **No frozen artifact is modified.** `results/admission_measurement.json`,
   `results/targeted_dose/`, `results/dose_response/`, `results/dose_replication/` and
   `results/dose_femnist/` are read-only here. This arm writes only `results/score_only/summary.json`
   and `results/score_only/scored.json`.
2. **`run_one` is imported, not copied**, and `score_only` is a defaulted parameter, so every frozen
   computation is bit-identical by construction rather than by inspection —
   `run_targeted_dose.py --harness-check` is the check, and it runs before any new ASR.
3. **Every comparison number is recomputed from per-seed rows at run time**, including the published
   Mode-S Δ and the identity rung. Nothing is transcribed.
4. **Both outcomes are written above and neither is renegotiated afterwards.** If this narrows the
   central claim it goes in the body as a narrowing, with this rule quoted, in the same place a
   confirmation would have gone.
5. **Wording is fixed now.** A confirmation is reported as *"the negative survives with the magnitude
   channel closed"* — never as *"statistic disturbance is causally irrelevant to suppression"*, which
   asserts a null no equivalence test can deliver, and never as *"the intervention isolates the
   statistic"*, which is false: the intervention isolates statistic and decision disturbance from
   **adversarial coefficient attenuation** and, here, from **benign magnitude**, and those are the two
   channels named.
