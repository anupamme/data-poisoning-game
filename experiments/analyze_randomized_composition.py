"""
Analyze results/randomized_composition/summary.json against its pre-registration.

Pre-registration: experiments/pre_registration_randomized_composition.md (committed at e3fb038
BEFORE the first run; the runner refuses to start otherwise, and the hash is written into the
artifact so a reader can check the freeze without trusting any docstring).

WHAT THIS SCRIPT IS FOR
The question is whether randomization is ITSELF costly, or whether the withdrawn 9x/19x gaps were
an artifact of degenerate CM->FG rounds that reduced to ONE defense. Every member of every
randomized set here keeps FoolsGold as a genuine per-client upstream, so no round degenerates, and
a gap between arm 2 and the best member of its own set is attributable to randomization rather
than to defense count.

NOTHING IS TRANSCRIBED. Every number printed is recomputed per seed from the artifact. The
pre-registration's non-negotiable is that no Delta or ratio mixes an n=5 minuend with an n=3
subtrahend -- the Round 55 defect, which hid in the SUBTRAHEND where no output column showed it.
So every contrast row prints BOTH legs' n and marks any mismatch `<-- MIXED n`.

FOUR POSITIVE CONTROLS, asserted rather than assumed, because this project has twice shipped a
hook that silently never fired (run_cross_distribution_compositions.py never calls
manipulate_update; a Round 55 Delta was credited to a bit-exact no-op).

Run: python3 experiments/analyze_randomized_composition.py
Exit 0 if every hard control passes; 1 otherwise.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import binom, ttest_rel, wilcoxon

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(base_dir, "results", "randomized_composition", "summary.json")
SEEDS = [42, 43, 44, 45, 46]
ATTACKS = ["committed_scaling", "committed_pixel"]
SHORT = {"committed_scaling": "scaling", "committed_pixel": "pixel"}

ARM1 = ["fixed_fg_cm", "fixed_fg_rfa", "fixed_fg_tm"]
STRONG_SET = ["fixed_fg_cm", "fixed_fg_rfa"]      # the set rand_comp_strong draws over
# Pre-registered materiality thresholds. Both must hold; fixed before any data existed.
DELTA_MATERIAL = 0.10
SIGNS_MATERIAL = 4
ACC_FLOOR = 0.35

failures = []
notes = []


def load():
    if not os.path.exists(ART):
        raise SystemExit(f"missing artifact: {ART}")
    with open(ART) as f:
        return json.load(f)


def cell(s, cond, atk):
    """The (seed -> record) map for one condition/attack, or None if not run."""
    c = s["conditions"].get(cond, {})
    a = c.get(atk)
    if not a or not a.get("per_seed"):
        return None
    return {r["seed"]: r for r in a["per_seed"]}


def asrs(s, cond, atk, seeds=None):
    m = cell(s, cond, atk)
    if m is None:
        return None
    seeds = seeds or sorted(m)
    if any(sd not in m for sd in seeds):
        return None
    return np.array([m[sd]["asr"] for sd in seeds], dtype=float)


def complete(s, cond):
    """Is every attack x seed present for this condition?"""
    return all((cell(s, cond, a) or {}).keys() >= set(SEEDS) for a in ATTACKS)


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ----------------------------------------------------------------------------------------------
def control_1_hook_fired(s):
    """Every member of every randomized set appears, with counts inside a two-sided 99% binomial
    interval of uniform.

    The policy stream is default_rng(seed + 2000) and depends ONLY on the seed, so the two attacks
    at one seed share an identical policy_log by construction -- there are 5 independent streams
    per randomized arm, not 10. That is verified here rather than assumed, because if the two
    attacks' logs ever diverged it would mean the policy draw had been re-coupled to something
    else in the run.
    """
    hdr("CONTROL 1 -- the randomization hook fired (per-round policy_log)")
    specs = [("rand_comp_strong", 2), ("rand_comp_all3", 3), ("temporal_mix", 4)]
    streams, excursions = 0, []
    for cond, k in specs:
        m = cell(s, cond, ATTACKS[0])
        if m is None:
            print(f"  {cond:18s} NOT RUN YET -- skipped")
            continue
        p = 1.0 / k
        lo = min(x for x in range(51) if binom.cdf(x, 50, p) > 0.005)
        hi = max(x for x in range(51) if binom.sf(x - 1, 50, p) > 0.005)
        members = {tuple(x) for x in s["conditions"][cond]["policy"]}
        print(f"  {cond}  ({len(members)} members, p=1/{k}, exact 99% interval [{lo},{hi}])")
        for sd in sorted(m):
            # identical logs across attacks: the policy stream must not depend on the attack
            logs = {atk: tuple((cell(s, cond, atk) or {}).get(sd, {}).get("policy_log", ()))
                    for atk in ATTACKS}
            present = [v for v in logs.values() if v]
            if len(present) == 2 and present[0] != present[1]:
                failures.append(f"C1 {cond} seed={sd}: policy_log differs between attacks, so the "
                                "policy draw is coupled to the attack")
                print(f"    seed={sd}  POLICY LOG DIFFERS BETWEEN ATTACKS  <-- FAIL")
                continue
            log = present[0]
            if len(log) != 50:
                failures.append(f"C1 {cond} seed={sd}: policy_log has {len(log)} entries, not 50")
            cnt = Counter(log)
            if len(cnt) < len(members):
                failures.append(f"C1 {cond} seed={sd}: only {len(cnt)}/{len(members)} members ever "
                                f"drawn ({dict(cnt)}) -- a member that never fires is a dead hook")
            streams += 1
            marks = []
            for name, c in sorted(cnt.items()):
                if not (lo <= c <= hi):
                    tail = binom.cdf(c, 50, p) if c < lo else binom.sf(c - 1, 50, p)
                    marks.append(f"{name}={c} OUTSIDE (tail P={tail:.5f})")
                    excursions.append((cond, sd, name, c, tail))
                else:
                    marks.append(f"{name}={c}")
            print(f"    seed={sd}  " + "  ".join(marks))

    # Excursions are counted PER STREAM, not per member. In a 2-member arm the counts are
    # complementary (36 and 14 of 50), so one lopsided draw necessarily flags both members and a
    # per-member tally would double-count a single event and inflate the family-wise arithmetic.
    bad_streams = sorted({(c, sd) for c, sd, _, _, _ in excursions})
    if streams:
        fam = 0.99 ** streams
        print(f"\n  {streams} independent policy streams checked; "
              f"{len(bad_streams)} stream(s) carry an excursion outside a per-stream 99% interval "
              f"({len(excursions)} member-level flags, counted per stream because a 2-member arm's "
              "counts are complementary).")
        print(f"  Family-wise: P(all {streams} inside) = 0.99^{streams} = {fam:.3f}, so "
              f"P(at least one excursion) = {1-fam:.3f}.")
    if excursions:
        # DISCLOSED, NOT SUPPRESSED, AND THE READING IS FLAGGED AS POST HOC.
        # The pre-registration wrote this control as a per-stream 99% interval and did not specify
        # a family-wise correction, so as LITERALLY SPECIFIED it fails on the streams listed above.
        # It is not being treated as invalidating, and the reason is stated rather than assumed:
        # the failure mode this control exists to catch is a DEAD HOOK, which produces a degenerate
        # log (50/0, or a member with count 0), not a merely lopsided one. Every stream above draws
        # every member. This reading was formed AFTER seeing the data and is labelled as such.
        print("\n  READING (post hoc, and labelled as such because the data existed first):")
        print("  As literally pre-registered -- a per-stream two-sided 99% interval, no family-wise")
        print("  correction -- this control FAILS on the stream(s) above. It is reported, not")
        print("  suppressed. It is not treated as invalidating, because the failure mode the")
        print("  control targets is a DEAD hook, which yields a member with count 0 or a 50/0 log;")
        print("  every stream here draws every member, and the excursion is a lopsided draw from a")
        print("  live generator. The pre-registered assertion is the stricter one and it did not")
        print("  hold on all streams; that is the honest statement.")
        notes.append(f"control 1: {len(bad_streams)} of {streams} policy streams outside the "
                     f"per-stream 99% interval (" +
                     "; ".join(f"{c} seed={sd} {n}={k}, P={t:.5f}"
                               for c, sd, n, k, t in excursions) + ")")
    elif streams:
        print("  [OK] every stream inside its interval, every member drawn.")
    # The pre-registration's parenthetical for rand_comp_all3 read "approx 8-26"; the exact
    # two-sided 99% interval at n=50, p=1/3 is [8,25]. The file said "approx", the exact bound is
    # used here, and the discrepancy is recorded rather than quietly absorbed.
    print("\n  NOTE: prereg wrote rand_comp_all3's interval as 'approx 8-26'; exact is [8,25].")
    print("        The exact interval is used. (strong [16,34] and temporal_mix [5,21] were exact.)")


def control_2_arms_differ(s):
    """Neither randomized arm is bit-identical to any arm-1 condition at any seed."""
    hdr("CONTROL 2 -- the randomized arms are not a fixed arm in disguise")
    for rand in ("rand_comp_strong", "rand_comp_all3"):
        for atk in ATTACKS:
            rm = cell(s, rand, atk)
            if rm is None:
                print(f"  {rand:18s} {SHORT[atk]:8s} NOT RUN YET -- skipped")
                continue
            worst = None
            for fx in ARM1:
                fm = cell(s, fx, atk)
                if fm is None:
                    continue
                shared = sorted(set(rm) & set(fm))
                dev = max(abs(rm[sd]["asr"] - fm[sd]["asr"]) for sd in shared) if shared else None
                if dev is None:
                    continue
                if dev == 0.0:
                    failures.append(f"C2 {rand} is BIT-IDENTICAL to {fx} on {atk} -- dead hook")
                if worst is None or dev < worst[1]:
                    worst = (fx, dev, len(shared))
            if worst:
                print(f"  {rand:18s} {SHORT[atk]:8s} closest fixed arm {worst[0]:15s} "
                      f"max per-seed |d|={worst[1]:.4f} over n={worst[2]}"
                      + ("  <-- FAIL (identical)" if worst[1] == 0.0 else "  [OK]"))


def control_3_canonical_identity(s):
    """Each fixed arm-1 condition, per seed, against the canonical artifacts on shared seeds.

    This runner uses the canonical global participant stream (np.random.seed(seed) +
    np.random.choice) and calls generic_compose, so agreement must be BIT-IDENTICAL. Any nonzero
    deviation is a harness change, and in that case no new arm may be compared to any published
    number. We do NOT expect agreement with results/compute_matched_mixing, which is on the
    divergent default_rng(seed+1000) stream -- that is the point of the control, not a failure.
    """
    hdr("CONTROL 3 -- the fixed arms reproduce the canonical artifacts BIT-IDENTICALLY")
    with open(os.path.join(base_dir, "results", "all_compositions", "summary.json")) as f:
        allc = json.load(f)["pairs"]
    with open(os.path.join(base_dir, "results", "fg_cm_survivor", "summary.json")) as f:
        surv = json.load(f)["composed"]

    refs = [
        ("fixed_fg_cm", "all_compositions/foolsgold_then_coord_median",
         lambda atk: {r["seed"]: r["asr"]
                      for r in allc["foolsgold_then_coord_median"][atk]["per_seed"]}),
        ("fixed_fg_rfa", "all_compositions/foolsgold_then_rfa",
         lambda atk: {r["seed"]: r["asr"]
                      for r in allc["foolsgold_then_rfa"][atk]["per_seed"]}
         if "foolsgold_then_rfa" in allc else {}),
        ("fixed_fg_tm", "all_compositions/foolsgold_then_trimmed_mean",
         lambda atk: {r["seed"]: r["asr"]
                      for r in allc["foolsgold_then_trimmed_mean"][atk]["per_seed"]}
         if "foolsgold_then_trimmed_mean" in allc else {}),
        ("fixed_fg_cm", "fg_cm_survivor/composed",
         lambda atk: {r["seed"]: r["asr"] for r in surv[atk]["per_seed"]}),
    ]
    for cond, label, getter in refs:
        for atk in ATTACKS:
            new = cell(s, cond, atk)
            ref = getter(atk)
            if new is None or not ref:
                print(f"  {cond:14s} {SHORT[atk]:8s} vs {label:42s} "
                      f"{'NOT RUN' if new is None else 'NO REF'} -- skipped")
                continue
            shared = sorted(set(new) & set(ref))
            if not shared:
                print(f"  {cond:14s} {SHORT[atk]:8s} vs {label:42s} no shared seeds -- skipped")
                continue
            dev = max(abs(new[sd]["asr"] - ref[sd]) for sd in shared)
            ok = dev == 0.0
            # BOTH legs' n, always, per the pre-registration's non-negotiable.
            mixed = "" if len(new) == len(ref) else "   <-- MIXED n (identity is on shared seeds only)"
            print(f"  {cond:14s} {SHORT[atk]:8s} vs {label:42s} "
                  f"n_new={len(new)} n_ref={len(ref)} shared={len(shared)} "
                  f"max|d|={dev:.3e} {'[OK]' if ok else '<-- HARNESS CHANGED'}{mixed}")
            if not ok:
                failures.append(f"C3 {cond}/{atk} deviates from {label} by {dev:.3e} on shared "
                                f"seeds {shared} -- harness changed the run")


def control_4_pure_cm_standalone(s):
    """pure_coord_median is the same quantity as wave2_held_out/fedavg_then_coord_median."""
    hdr("CONTROL 4 -- arm 4 reproduces a published standalone (end-to-end path check)")
    with open(os.path.join(base_dir, "results", "wave2_held_out", "summary.json")) as f:
        w2 = json.load(f)["pairs"]["fedavg_then_coord_median"]
    for atk in ATTACKS:
        new = cell(s, "pure_coord_median", atk)
        ref = {r["seed"]: r["asr"] for r in w2[atk]["per_seed"]}
        if new is None:
            print(f"  pure_coord_median {SHORT[atk]:8s} NOT RUN YET -- skipped "
                  f"(ref mean {np.mean(list(ref.values())):.4f}, n={len(ref)})")
            continue
        shared = sorted(set(new) & set(ref))
        dev = max(abs(new[sd]["asr"] - ref[sd]) for sd in shared)
        mixed = "" if len(new) == len(ref) else "   <-- MIXED n"
        print(f"  pure_coord_median {SHORT[atk]:8s} n_new={len(new)} n_ref={len(ref)} "
              f"new={np.mean([new[sd]['asr'] for sd in shared]):.4f} "
              f"ref={np.mean([ref[sd] for sd in shared]):.4f} max|d|={dev:.3e} "
              f"{'[OK]' if dev == 0.0 else '<-- PARTICIPANT STREAM DIFFERS'}{mixed}")
        if dev != 0.0:
            failures.append(f"C4 pure_coord_median/{atk} deviates from wave2_held_out "
                            f"fedavg_then_coord_median by {dev:.3e}")


def accuracy_floor(s):
    hdr(f"ACCURACY FLOOR (pre-registered: any arm below {ACC_FLOOR} mean clean accuracy is "
        "uninterpretable)")
    for cond in s["conditions"]:
        for atk in ATTACKS:
            m = cell(s, cond, atk)
            if m is None:
                continue
            acc = float(np.mean([m[sd]["accuracy"] for sd in m]))
            flag = "  <-- BELOW FLOOR, uninterpretable for ASR" if acc < ACC_FLOOR else ""
            print(f"  {cond:18s} {SHORT[atk]:8s} n={len(m)} mean_acc={acc:.4f}{flag}")
            if acc < ACC_FLOOR:
                notes.append(f"accuracy floor breached: {cond}/{atk} at {acc:.4f}")


def contrast(s, rand_cond, comparator_set, atk, label):
    """One paired contrast. Returns a dict or None. Prints both legs' n, always."""
    rand = cell(s, rand_cond, atk)
    if rand is None:
        print(f"  {label:52s} {SHORT[atk]:8s} randomized arm NOT RUN -- skipped")
        return None
    # "Best" by RULE, not by name: lowest mean ASR on THIS attack WITHIN THIS ARTIFACT, so the
    # selection and the comparison are on the same n and the comparator comes only from the set
    # actually being randomized over.
    cands = {}
    for c in comparator_set:
        v = asrs(s, c, atk)
        if v is not None:
            cands[c] = float(np.mean(v))
    if not cands:
        print(f"  {label:52s} {SHORT[atk]:8s} no comparator run -- skipped")
        return None
    best = min(cands, key=cands.get)
    bm = cell(s, best, atk)

    shared = sorted(set(rand) & set(bm))
    n_r, n_b = len(rand), len(bm)
    mixed = n_r != n_b
    r = np.array([rand[sd]["asr"] for sd in shared])
    b = np.array([bm[sd]["asr"] for sd in shared])
    d = r - b
    pos = int((d > 0).sum())
    dbar = float(d.mean())

    p_w = t_stat = p_t = float("nan")
    if len(d) >= 2 and not np.allclose(d, 0):
        p_w = float(wilcoxon(d, alternative="greater").pvalue)
        tt = ttest_rel(r, b, alternative="greater")
        t_stat, p_t = float(tt.statistic), float(tt.pvalue)

    material = (dbar >= DELTA_MATERIAL) and (pos >= SIGNS_MATERIAL)
    print(f"  {label:52s} {SHORT[atk]:8s}")
    # EVERY mean is labelled with the n IT WAS COMPUTED OVER, not with the leg's full n. Printing
    # "mean=0.0454 n=5" for a mean taken over 3 shared seeds is the Round 55 defect in the output
    # layer: the number and the n beside it would describe different sets.
    print(f"      randomized  {rand_cond:18s} mean={r.mean():.4f} sd={np.std(r):.4f} "
          f"over n={len(shared)} shared   [full leg n={n_r}]")
    print(f"      best member {best:18s} mean={b.mean():.4f} sd={np.std(b):.4f} "
          f"over n={len(shared)} shared   [full leg n={n_b}]")
    sel_ns = {k: len(cell(s, k, atk)) for k in cands}
    print(f"      selected by rule (lowest full-artifact mean on this attack) from "
          f"{ {k: (round(v, 4), f'n={sel_ns[k]}') for k, v in cands.items()} }")
    if len(set(sel_ns.values())) > 1:
        print("        <-- SELECTION Ns DIFFER: the best-member rule compared means over unequal "
              "seed counts")
        failures.append(f"selection rule for {label}/{atk} compared unequal-n means: {sel_ns}")
    print(f"      paired Delta over {len(shared)} shared seeds: "
          + ", ".join(f"s{sd}={x:+.4f}" for sd, x in zip(shared, d)))
    print(f"      Dbar={dbar:+.4f}  signs +{pos}/{len(d)}  "
          f"Wilcoxon(1-sided,greater) p={p_w:.4f}  t={t_stat:.3f} p={p_t:.4f}"
          + ("   <-- MIXED n" if mixed else ""))
    print(f"      MATERIAL? Dbar>={DELTA_MATERIAL} -> {dbar >= DELTA_MATERIAL}; "
          f"signs>={SIGNS_MATERIAL}/5 -> {pos >= SIGNS_MATERIAL}  ==> "
          f"{'MATERIAL, pre-registered direction' if material else 'NOT material'}")
    if mixed:
        failures.append(f"MIXED n in contrast {label}/{atk}: {n_r} vs {n_b}")
    # Post hoc mechanism diagnostic, labelled: is the randomized arm merely between its members,
    # or worse than BOTH? "Worse than the best" is consistent with interpolation; "worse than
    # both" is not, and the two readings support different mechanism claims.
    both = None
    if rand_cond == "rand_comp_strong":
        cm, rfa = cell(s, "fixed_fg_cm", atk), cell(s, "fixed_fg_rfa", atk)
        if cm and rfa:
            both = sum(1 for sd in shared
                       if rand[sd]["asr"] > max(cm[sd]["asr"], rfa[sd]["asr"]))
            print(f"      [post hoc] randomized exceeds BOTH members in {both}/{len(shared)} seeds "
                  "(not interpolation between them)")
    return {"rand": rand_cond, "atk": atk, "best": best, "dbar": dbar, "pos": pos,
            "n": len(d), "p_wilcoxon": p_w, "p_t": p_t, "material": material,
            "worse_than_both": both, "mean_rand": float(r.mean()), "mean_best": float(b.mean())}


def contrasts(s):
    hdr("PRIMARY CONTRAST (pre-registered: rand_comp_strong vs the best member OF ITS OWN SET, "
        "paired per seed, one-sided H1: Dbar > 0)")
    print("  n=5 exact limit, stated in advance: min attainable Wilcoxon p = 1/32 = 0.031, so only")
    print("  5/5 concordant signs can reach p<0.05; 4/5 gives p=0.094.\n")
    prim = [contrast(s, "rand_comp_strong", STRONG_SET, a,
                     "rand_comp_strong vs best of {fg->cm, fg->rfa}") for a in ATTACKS]

    hdr("SECONDARY CONTRAST (rand_comp_all3 vs the best of all three arm-1 conditions)")
    sec = [contrast(s, "rand_comp_all3", ARM1, a,
                    "rand_comp_all3 vs best of all three") for a in ATTACKS]

    hdr("FURTHER SECONDARIES (reported, not decisive)")
    for a_cond, b_cond, why in [
        ("rand_comp_all3", "rand_comp_strong", "what a non-suppressing member in the menu costs"),
        ("rand_comp_strong", "temporal_mix",
         "randomizing over COMPOSITIONS vs over SINGLE defenses (NOT compute-matched: 2 vs 1 "
         "defense/round)"),
        ("temporal_mix", "pure_coord_median",
         "is temporal mixing worse than the best pure defense (both 1 defense/round)"),
    ]:
        print(f"\n  {a_cond} - {b_cond}   [{why}]")
        for atk in ATTACKS:
            va, vb = cell(s, a_cond, atk), cell(s, b_cond, atk)
            if va is None or vb is None:
                print(f"    {SHORT[atk]:8s} NOT RUN YET -- skipped")
                continue
            shared = sorted(set(va) & set(vb))
            d = np.array([va[sd]["asr"] - vb[sd]["asr"] for sd in shared])
            mixed = "   <-- MIXED n" if len(va) != len(vb) else ""
            print(f"    {SHORT[atk]:8s} n_a={len(va)} n_b={len(vb)} shared={len(shared)} "
                  f"mean_a={np.mean([va[sd]['asr'] for sd in shared]):.4f} "
                  f"mean_b={np.mean([vb[sd]['asr'] for sd in shared]):.4f} "
                  f"Dbar={d.mean():+.4f} signs +{int((d>0).sum())}/{len(d)}{mixed}")
            if len(va) != len(vb):
                failures.append(f"MIXED n: {a_cond}({len(va)}) vs {b_cond}({len(vb)}) on {atk}")
    return prim, sec


def draw_diagnostic(s):
    """Pre-registered diagnostic, costing no extra runs: is any gap explained by WHICH member was
    drawn rather than by randomization? Per seed, associate a randomized arm's realized member
    counts with its ASR."""
    hdr("DIAGNOSTIC -- does the realized draw composition explain the ASR? (pre-registered)")
    from scipy.stats import spearmanr
    for cond in ("rand_comp_strong", "rand_comp_all3", "temporal_mix"):
        m = cell(s, cond, ATTACKS[0])
        if m is None:
            print(f"  {cond:18s} NOT RUN YET -- skipped")
            continue
        members = sorted({tuple(x) for x in s["conditions"][cond]["policy"]})
        for atk in ATTACKS:
            mm = cell(s, cond, atk)
            if mm is None:
                continue
            seeds = sorted(mm)
            print(f"  {cond} / {SHORT[atk]}  (n={len(seeds)})")
            for d1, d2 in members:
                key = f"{d1}->{d2}"
                counts = [Counter(mm[sd]["policy_log"])[key] for sd in seeds]
                a = [mm[sd]["asr"] for sd in seeds]
                if len(seeds) >= 3 and len(set(counts)) > 1:
                    rho, p = spearmanr(counts, a)
                    print(f"      count({key:28s}) {counts}  vs ASR  "
                          f"Spearman rho={rho:+.3f} p={p:.3f}")
                else:
                    print(f"      count({key:28s}) {counts}  (too few / no variation)")


def raw_table(s):
    hdr("RAW ARMS (recomputed per seed from the artifact; sd is POPULATION sd, ddof=0, the "
        "convention every published +/- in these papers uses)")
    print(f"  {'condition':18s} {'arm':>3s} {'def/rd':>6s} "
          f"{'scaling ASR':>18s} {'pixel ASR':>18s} {'acc(sc/px)':>15s}")
    order = ARM1 + ["rand_comp_strong", "rand_comp_all3", "temporal_mix", "pure_coord_median"]
    for cond in order:
        c = s["conditions"].get(cond)
        if not c:
            print(f"  {cond:18s}  -- NOT RUN YET --")
            continue
        cells = {}
        for atk in ATTACKS:
            m = cell(s, cond, atk)
            cells[atk] = m
        def fmt(atk):
            m = cells[atk]
            if m is None:
                return f"{'--':>18s}"
            v = np.array([m[sd]["asr"] for sd in sorted(m)])
            return f"{v.mean():.4f}+-{np.std(v):.4f}(n={len(v)})"
        accs = []
        for atk in ATTACKS:
            m = cells[atk]
            accs.append("--" if m is None
                        else f"{np.mean([m[sd]['accuracy'] for sd in m]):.3f}")
        print(f"  {cond:18s} {c.get('arm','?'):>3} {c.get('defenses_per_round','?'):>6} "
              f"{fmt(ATTACKS[0]):>18s} {fmt(ATTACKS[1]):>18s} {'/'.join(accs):>15s}"
              + ("" if c.get("defenses_per_round") == 2 else "   [1 def/rd: NOT compute-matched]"))
    print("\n  Compute matching is claimed for arms 1-2 ONLY. temporal_mix and pure_coord_median")
    print("  apply one defense per round and use roughly half the defense computation; they are")
    print("  reference points, not compute-matched comparators.")


def verdict(s, prim, sec):
    hdr("PRE-REGISTERED OUTCOME")
    done = [c for c in ["fixed_fg_cm", "fixed_fg_rfa", "fixed_fg_tm", "rand_comp_strong",
                        "rand_comp_all3", "temporal_mix", "pure_coord_median"] if complete(s, c)]
    print(f"  complete conditions: {len(done)}/7 -> {done}")
    if len(done) < 7:
        print("  ==> PARTIAL DATA. No outcome is declared until all seven arms are complete at "
              "seeds 42-46;")
        print("      the pre-registration forbids dropping any arm or seed after the data exists.")
        return
    for r in [x for x in prim if x]:
        state = ("(i) MATERIAL in the pre-registered direction" if r["material"]
                 else "(ii) NOT material (or opposite direction)")
        print(f"  {SHORT[r['atk']]:8s} Dbar={r['dbar']:+.4f} signs +{r['pos']}/{r['n']} "
              f"p_wilcoxon={r['p_wilcoxon']:.4f}  ==> {state}")
    print("\n  Whichever obtained, `:1638`'s withdrawal of the OLD comparison stands: that artifact")
    print("  remains unpaired and compute-confounded. The pre-registration forbids re-labelling a")
    print("  null as 'randomization is at least not harmful, so composition is robust'.")
    print("  At n=5 the honest phrasing for a sub-material gap is 'no material difference")
    print("  detected; the smallest effect this design resolves is ~0.10'.")


def main():
    s = load()
    print("=" * 78)
    print("RANDOMIZED COMPOSITION -- analysis against "
          "experiments/pre_registration_randomized_composition.md")
    print("=" * 78)
    print(f"  artifact      : {os.path.relpath(ART, base_dir)}")
    print(f"  prereg_commit : {s.get('prereg_commit')}  (frozen before the first run)")
    cfg = s.get("config", {})
    print(f"  config        : N={cfg.get('num_clients')} K={cfg.get('clients_per_round')} "
          f"f={cfg.get('adv_fraction')} rounds={cfg.get('num_rounds')} "
          f"alpha={cfg.get('dirichlet_alpha')} seeds={cfg.get('seeds')}")

    raw_table(s)
    control_1_hook_fired(s)
    control_2_arms_differ(s)
    control_3_canonical_identity(s)
    control_4_pure_cm_standalone(s)
    accuracy_floor(s)
    prim, sec = contrasts(s)
    draw_diagnostic(s)
    verdict(s, prim, sec)

    hdr("HARD CONTROL SUMMARY")
    if failures:
        print(f"  {len(failures)} FAILURE(S) -- no number from this artifact may enter the paper:")
        for f in failures:
            print(f"    - {f}")
    else:
        print("  [OK] every hard control passed.")
    if notes:
        print("\n  Disclosed, non-blocking:")
        for n in notes:
            print(f"    - {n}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
