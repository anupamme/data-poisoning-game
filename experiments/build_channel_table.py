"""
The cross-aggregator channel table: one row per aggregator, one column per link in the chain.

WHY THIS EXISTS. The paper's central negative -- that disturbing a defense's downstream statistic
changes its decisions without changing the adversarial mass it admits, and without changing
suppression -- has always been reported one arm at a time, in three different tables, on three
different pages. A reader therefore cannot see the thing that makes it a mechanism claim rather than a
Krum anecdote: the SAME pattern across four aggregators of three structurally different kinds. Every
number below is already in the frozen artifacts. Nothing new is computed and nothing is trained; this
script only assembles.

THE CHAIN, and what each column is:

  aggregate   ||agg(T(U)) - agg(U)|| / ||agg(U)||, the relative displacement of the aggregate the
              defense EMITS -- i.e. of the update that actually enters training. It is NOT the value of
              a scoring statistic, and it is not small: a criterion that preserves a downstream
              statistic does not thereby preserve what the defense emits, which is the point.
  decision    the fraction of rounds the defense's own internal choice changes -- selected index
              (krum, cos_krum), weight ordering (reputation), or median-attaining client
              (coord_median). This is what Round 11 put on its abscissa.
  admission   the fraction of rounds the SUPPORT of the adversarial mass changes, i.e. an adversary
              gets in where it had not, or is shut out where it had been.
  influence   the mean absolute change in the adversarial mass itself, Lambda_a. For a selector the
              mass is 0 or 1, so influence and admission coincide by construction and the table says
              so; for a weighted averager or an order statistic the mass is graded and they differ.
  ASR         mean ASR at the top rung minus mean ASR at the identity rung, recomputed per seed.

Lambda_a, per aggregator, is the aggregator's own notion of admitted adversarial mass -- there is no
single formula, because a selector has no weights:

  krum, cos_krum   1 if the selected update is adversarial, else 0
  reputation       sum_A w / sum w, the adversarial weight share (this is Lambda_a exactly)
  coord_median     the fraction of output coordinates whose median-attaining client is adversarial

SOURCES, all read-only:
  results/admission_measurement.json   channels for the CIFAR-10 arms (frozen, POST-HOC for `dose`)
  results/femnist_admission.json       channels for the FEMNIST arm, if it has been measured
  results/targeted_dose/summary.json   ASR for the three Round-12 Mode-S arms
  results/dose_replication/summary.json  ASR for the Round-15 coord_median arm
  results/dose_femnist/summary.json     ASR for the FEMNIST arm, if it has been run
  results/displacement_decomposition.json  the score-only row's aggregate displacement
  results/score_only/summary.json          the score-only row's ASR, read separately (see below)

THE SCORE-ONLY ROW is the one row that is not a plain Mode-S ladder, and it is included because it
closes the channel the Delta agg. column of the Krum row measures. Under score-only, Krum SCORES on
the transformed stack and AGGREGATES the untransformed selected update, so:

  decision / admission / influence  are the SAME measurement as the Krum row -- the selection is
                                    computed on the transformed stack in both arms, so these three
                                    columns are read from the Krum row rather than re-measured, and
                                    this script asserts they come from that row.
  aggregate                         is NOT the Krum row's 0.892. What score-only emits is u_{s1}
                                    untransformed, so the displacement is the re-selection component
                                    alone -- `score_only_displacement` in
                                    results/displacement_decomposition.json, computed there by the
                                    same rel_disp on the same rounds.
  ASR                               comes from results/score_only/summary.json, the pre-registered
                                    arm (experiments/pre_registration_score_only.md).

Every cell printed is recomputed here from per-round / per-seed rows; no aggregate is transcribed.
Rows whose ASR is not yet available print the channels and leave the ASR column blank rather than
being dropped, so a reader can see what is measured and what is pending.

Output: results/channel_table.json plus LaTeX on stdout.
Run: python3 experiments/build_channel_table.py
"""
import json, os, sys
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.measure_admission import DECISION_KEY

R = os.path.join(base, "results")
ADM = os.path.join(R, "admission_measurement.json")
ADM_FEMNIST = os.path.join(R, "femnist_admission.json")
ASR_SOURCES = [os.path.join(R, "targeted_dose", "summary.json"),
               os.path.join(R, "dose_replication", "summary.json"),
               os.path.join(R, "dose_femnist", "summary.json")]
# NOT in ASR_SOURCES: score_only/summary.json and emit_only/summary.json reuse run_targeted_dose's
# cell_key on the same dataset, so merging either in would overwrite the published Krum row's ASR with
# a controlled arm's. They are read only by score_only_row() / emit_only_row(), and _asr_cells() now
# asserts no source can shadow another.
DECOMP = os.path.join(R, "displacement_decomposition.json")
SCORE_ONLY = os.path.join(R, "score_only", "summary.json")
EMIT_ONLY = os.path.join(R, "emit_only", "summary.json")

OUT = os.path.join(R, "channel_table.json")
TOP = 2.0          # the top Mode-S rung, rho = exp(4) = 54.6
IDENTITY = 0.0

# The adversarial-mass field each arm's channel measurement records, and whether that mass is binary.
# A binary mass means admission and influence are the same measurement, which the caption must say.
MASS = {
    "krum":         ("krum_admits_adv", True),
    "cos_krum":     ("cos_krum_admits_adv", True),
    "reputation":   ("reputation_adv_weight_share", False),
    "coord_median": ("coord_median_adv_argmedian_frac", False),
}
KIND = {
    "krum":         "selector (pairwise distance)",
    "cos_krum":     "selector (cosine distance)",
    "reputation":   "weighted averager (cross-round state)",
    "coord_median": "coordinate-wise order statistic",
}
# (label, arm, attack, channel source). One row per aggregator; the FEMNIST row is the same arm as the
# first, on a different dataset and architecture, which is the only reason it is a separate row.
ROWS = [
    ("Krum",              "krum",         "committed_scaling", ADM),
    ("Reputation",        "reputation",   "committed_scaling", ADM),
    ("Cosine-Krum",       "cos_krum",     "committed_pixel",   ADM),
    ("Coord.\\ median",   "coord_median", "committed_pixel",   ADM),
    ("Krum (FEMNIST)",    "krum",         "committed_scaling", ADM_FEMNIST),
]


def channel_rows(path, arm, attack, rung):
    """Per-round rows of one arm's Mode-S rung from a channel-measurement file."""
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    return [r for r in d["per_round"]
            if r["family"] == "doseS" and abs(r["rung"] - rung) < 1e-12 and r["attack"] == attack]


def channels(path, arm, attack, rung):
    """Delta aggregate / decision / admission / influence for one arm at one rung.

    Recomputed from the per-round rows rather than read from the file's own summary, so this script
    and the measurement cannot disagree about what the columns mean.
    """
    rows = channel_rows(path, arm, attack, rung)
    if not rows:
        return None
    field, binary = MASS[arm]
    disp, dec, adm, infl = [], [], [], []
    for r in rows:
        disp.append(float(r[f"agg_disp_{arm}"]))
        v = r[DECISION_KEY[arm]]
        dec.append(float(v))                       # bool for the selectors, a fraction for the median
        b, p = float(r[f"base_{field}"]), float(r[f"post_{field}"])
        adm.append(float((b > 0.0) != (p > 0.0)))  # did the SUPPORT of the adversarial mass change
        infl.append(abs(p - b))                    # how much the mass itself moved
    return {"n_rounds": len(rows), "d_agg_disp": float(np.mean(disp)),
            "d_decision": float(np.mean(dec)), "d_admission": float(np.mean(adm)),
            "d_influence": float(np.mean(infl)), "mass_is_binary": binary,
            "mean_mass_base": float(np.mean([float(r[f"base_{field}"]) for r in rows])),
            "mean_mass_post": float(np.mean([float(r[f"post_{field}"]) for r in rows]))}


def _asr_cells():
    """{(dataset, cell_key): cell} over every ASR source that exists.

    The dataset is part of the key on purpose. run_dose_femnist.py reuses run_targeted_dose's
    cell_key, so the FEMNIST krum arm and the CIFAR-10 krum arm have the SAME string key -- merging
    the files into one flat dict would silently let one overwrite the other.
    """
    cells = {}
    for p in ASR_SOURCES:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ds = d.get("dataset", "cifar10")
        for k, c in d.get("cells", {}).items():
            # Keying on (dataset, key) is not enough on its own: two suites can share BOTH, as the
            # score-only arm shares them with targeted_dose. A shadowed cell is a silently wrong
            # published number, so refuse rather than resolve it by file order.
            assert (ds, k) not in cells, (
                f"cell {(ds, k)} appears in {cells[(ds, k)]['_source']} and "
                f"{os.path.relpath(p, base)}; one would shadow the other")
            cells[(ds, k)] = dict(c, _source=os.path.relpath(p, base))
    return cells


def mean_asr(cell):
    """Recomputed from per-seed rows: imported identity cells carry rows but no stored aggregate."""
    rows = cell.get("per_seed", [])
    return (float(np.mean([r["asr"] for r in rows])) if rows else float("nan"),
            float(np.mean([r["accuracy"] for r in rows])) if rows else float("nan"),
            len(rows))


def asr_delta(arm, attack, dataset="cifar10"):
    cells = _asr_cells()
    lo = cells.get((dataset, f"doseS_kappa{IDENTITY}_then_{arm}|{attack}"))
    hi = cells.get((dataset, f"doseS_kappa{TOP}_then_{arm}|{attack}"))
    if lo is None or hi is None:
        return None
    a0, c0, n0 = mean_asr(lo)
    a2, c2, n2 = mean_asr(hi)
    return {"identity_asr": a0, "top_asr": a2, "delta_asr": a2 - a0,
            "identity_acc": c0, "top_acc": c2, "n_seeds": min(n0, n2),
            "source": hi.get("_source")}


def _dasr(row):
    return "---" if row["asr"] is None else f"{row['asr']['delta_asr']:+.4f}"


def _controlled_asr(path):
    """The identity -> top ASR contrast of a controlled arm's own summary file.

    Shared by both factorial controls so the two rows cannot come to mean different contrasts.
    """
    if not os.path.exists(path):
        return None
    cells = json.load(open(path)).get("cells", {})
    lo = cells.get(f"doseS_kappa{IDENTITY}_then_krum|committed_scaling")
    hi = cells.get(f"doseS_kappa{TOP}_then_krum|committed_scaling")
    if not lo or not hi:
        return None
    a0, c0, n0 = mean_asr(lo)
    a2, c2, n2 = mean_asr(hi)
    return {"identity_asr": a0, "top_asr": a2, "delta_asr": a2 - a0, "identity_acc": c0,
            "top_acc": c2, "n_seeds": min(n0, n2), "source": os.path.relpath(path, base)}


def score_only_row(table):
    """The score-only control's row, derived from the Krum row plus two read-only artifacts.

    Not re-measured: score-only leaves the SELECTION untouched (Krum still scores the transformed
    stack), so decision, admission and influence are literally the Krum row's numbers and are taken
    from it rather than recomputed under a second name. What differs is what is emitted -- the
    untransformed selected update -- so the aggregate column is the re-selection component alone.
    """
    krum = next((t for t in table if t["label"] == "Krum"), None)
    if krum is None or not os.path.exists(DECOMP) or not os.path.exists(SCORE_ONLY):
        return None
    dec = json.load(open(DECOMP))["summary"].get(f"doseS|krum|{TOP}")
    if dec is None:
        return None
    asr = _controlled_asr(SCORE_ONLY)
    # The Krum row's own total must equal the published displacement the decomposition split, or the
    # re-selection component being borrowed here belongs to a different measurement.
    assert abs(dec["total"] - krum["d_agg_disp"]) < 1e-9, (dec["total"], krum["d_agg_disp"])
    return dict(krum, label="Krum, score-only", arm="krum_score_only",
                kind="selector, magnitude channel closed",
                channel_source=os.path.relpath(DECOMP, base),
                d_agg_disp=float(dec["score_only_displacement"]),
                identity_displacement=0.0, asr=asr,
                borrowed_from="Krum (decision/admission/influence are the same measurement)")


def emit_only_row(table):
    """The emit-only control's row: the mirror of score-only and the fourth factorial cell.

    Emit-only scores the UNTRANSFORMED stack and aggregates the TRANSFORMED selected update, so:

    * Delta dec., Delta adm. and Delta Lambda_a are 0.000 **by construction, not borrowed**. The
      transformation has no path to the selection at all, so on every round Krum picks exactly what
      Krum-alone picks on that round's stack, the admitted set is that same client, and for a binary
      mass the admitted mass is that same indicator. Precisely: this is a same-round, same-stack
      counterfactual. It does NOT claim the run follows the identity rung's trajectory -- it cannot,
      since a different update is emitted from round 1 onward -- only that within this arm the
      statistic channel is closed. The score-only row borrows these three from Krum because there the
      selection genuinely does change; here there is nothing to borrow.
    * Delta agg. is the *rescaling* component of the algebraic decomposition, exactly as the
      score-only row's is the *re-selection* component. Both are components of one bookkeeping
      identity and neither is a causal quantity, which is the whole reason the ASR column of these
      two rows -- which IS measured by running the counterfactual -- exists.
    """
    krum = next((t for t in table if t["label"] == "Krum"), None)
    if krum is None or not os.path.exists(DECOMP) or not os.path.exists(EMIT_ONLY):
        return None
    dec = json.load(open(DECOMP))["summary"].get(f"doseS|krum|{TOP}")
    if dec is None:
        return None
    assert abs(dec["total"] - krum["d_agg_disp"]) < 1e-9, (dec["total"], krum["d_agg_disp"])
    return dict(krum, label="Krum, emit-only", arm="krum_emit_only",
                kind="selector, statistic channel closed",
                channel_source=os.path.relpath(DECOMP, base),
                d_agg_disp=float(dec["rescaling"]),
                d_decision=0.0, d_admission=0.0, d_influence=0.0,
                identity_displacement=0.0, asr=_controlled_asr(EMIT_ONLY),
                borrowed_from=None,
                closed_by_construction="decision/admission/influence: the transform has no path "
                                       "to the selection (same-round, same-stack counterfactual)")


def main():
    print("=== CROSS-AGGREGATOR CHANNEL TABLE (Mode S, identity rung -> kappa = 2, rho = 54.6) ===")
    print("    Assembled from frozen artifacts. No ASR is computed here and no model is trained.\n")

    table = []
    for label, arm, attack, src in ROWS:
        ch = channels(src, arm, attack, TOP)
        if ch is None:
            print(f"  -- {label}: no channel measurement in {os.path.relpath(src, base)}; row skipped")
            continue
        # The identity rung's own displacement must be exactly zero: T is the identity there, so a
        # nonzero value would mean the measurement is not comparing what it claims to compare.
        base_ch = channels(src, arm, attack, IDENTITY)
        dataset = "femnist" if src == ADM_FEMNIST else "cifar10"
        asr = asr_delta(arm, attack, dataset)
        table.append({"label": label, "arm": arm, "attack": attack, "kind": KIND[arm],
                      "dataset": dataset, "channel_source": os.path.relpath(src, base),
                      "identity_displacement": None if base_ch is None else base_ch["d_agg_disp"],
                      **ch, "asr": asr})

    so = score_only_row(table)
    if so is None:
        print("  -- Krum, score-only: needs displacement_decomposition.json and "
              "score_only/summary.json; row skipped")
    else:
        table.append(so)

    eo = emit_only_row(table)
    if eo is None:
        print("  -- Krum, emit-only: needs displacement_decomposition.json and "
              "emit_only/summary.json; row skipped (the arm has not been run)")
    else:
        table.append(eo)

    hdr = (f"  {'aggregator':22s} {'kind':34s} {'dAgg':>8s} {'dDec':>7s} {'dAdm':>7s} "
           f"{'dInfl':>7s} {'dASR':>8s} {'n':>4s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in table:
        a = t["asr"]
        dasr = "     --- " if a is None else f"{a['delta_asr']:+8.3f}"
        n = "  - " if a is None else f"{a['n_seeds']:>4d}"
        print(f"  {t['label'].replace(chr(92), ''):22s} {t['kind']:34s} {t['d_agg_disp']:8.3f} "
              f"{t['d_decision']:7.3f} {t['d_admission']:7.3f} {t['d_influence']:7.3f} {dasr} {n}")

    print("\n  dAgg   relative displacement of the EMITTED aggregate at the top rung -- the")
    print("         update that enters training, not the value of a scoring statistic")
    print("  dDec   fraction of rounds the defense's own decision changes")
    print("  dAdm   fraction of rounds the SUPPORT of the adversarial mass changes")
    print("  dInfl  mean absolute change in the adversarial mass Lambda_a")
    print("  dASR   mean ASR(kappa=2) - mean ASR(identity), recomputed per seed")
    if so is not None:
        print("\n  The score-only row emits the UNTRANSFORMED selected update, so its dAgg is the")
        print("  re-selection component alone (displacement_decomposition.json); its dDec/dAdm/dInfl")
        print("  are the Krum row's own numbers, since score-only does not change the selection.")
    if eo is not None:
        print("\n  The emit-only row is its mirror: it emits the RESCALED selected update but scores")
        print("  the untransformed stack, so its dAgg is the rescaling component alone and its")
        print("  dDec/dAdm/dInfl are 0.000 BY CONSTRUCTION -- the transform has no path to the")
        print("  selection -- rather than borrowed from any row.")
    if so is not None and eo is not None:
        krum = next(t for t in table if t["label"] == "Krum")
        base_asr = krum["asr"]["identity_asr"] if krum["asr"] else float("nan")
        print("\n=== THE 2x2 FACTORIAL (dASR at the top rung; the identity rung is the shared base) ===")
        print(f"  base ASR at the identity rung: {base_asr:.4f}")
        print(f"  {'':28s} {'emitted unchanged':>20s} {'emitted rescaled':>20s}")
        print(f"  {'decision unchanged':28s} {'0.000 (identity)':>20s} "
              f"{_dasr(eo):>20s}   <- emit-only")
        print(f"  {'decision changed':28s} {_dasr(so):>20s} {_dasr(krum):>20s}   <- full Mode S")
        print("                                    ^ score-only")
        d_so, d_eo, d_full = (r["asr"]["delta_asr"] if r["asr"] else None for r in (so, eo, krum))
        if None not in (d_so, d_eo, d_full):
            print(f"\n  additivity: full {d_full:+.4f} = score-only {d_so:+.4f} + emit-only "
                  f"{d_eo:+.4f} + residual {d_full - (d_so + d_eo):+.4f}")
            print("  The pre-registered thresholds and the verdict live in "
                  "results/emit_only/summary.json;")
            print("  this block reports the arithmetic, not the verdict.")
    # Separated by " | ", not ", ": two of these labels contain a comma, so a comma-joined list
    # reads as more arms than there are.
    bins = [t["label"].replace(chr(92), "") for t in table if t["mass_is_binary"]]
    print(f"\n  Binary-mass arms (dAdm == dInfl by construction): {' | '.join(bins)}")

    print("\n=== SANITY: the identity rung must displace nothing ===")
    for t in table:
        v = t["identity_displacement"]
        print(f"  {t['label'].replace(chr(92), ''):22s} identity displacement "
              + ("missing" if v is None else f"{v:.2e}"
                 + ("  OK" if v is not None and v < 1e-9 else "  <- NOT ZERO")))

    print("\n=== THE FULL LADDERS, for the appendix ===")
    for t in [t for t in table if "borrowed_from" not in t]:
        print(f"\n  -- {t['label'].replace(chr(92), '')} ({t['kind']}, {t['dataset']}) --")
        print(f"  {'kappa':>6s} {'dAgg':>8s} {'dDec':>7s} {'dAdm':>7s} {'dInfl':>7s} "
              f"{'ASR':>8s} {'acc':>6s}")
        cells = _asr_cells()
        for k in (0.0, 0.5, 1.0, 2.0):
            ch = channels(os.path.join(base, t["channel_source"]), t["arm"], t["attack"], k)
            c = cells.get((t["dataset"], f"doseS_kappa{k}_then_{t['arm']}|{t['attack']}"))
            a, ac, _ = mean_asr(c) if c else (float("nan"), float("nan"), 0)
            if ch is None:
                continue
            print(f"  {k:>6} {ch['d_agg_disp']:8.3f} {ch['d_decision']:7.3f} "
                  f"{ch['d_admission']:7.3f} {ch['d_influence']:7.3f} {a:8.3f} {ac:6.3f}")

    # LaTeX. Emitted here so the paper's table cannot drift from the numbers above.
    print("\n=== LATEX (copy verbatim; regenerate rather than edit) ===\n")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Aggregator & Kind & $\Delta$ agg. & $\Delta$ dec. & $\Delta$ adm. "
             r"& $\Delta \Lambda_a$ & $\Delta$ ASR \\", r"\midrule"]
    for t in table:
        a = t["asr"]
        dasr = "---" if a is None else f"{a['delta_asr']:+.3f}"
        lines.append(f"{t['label']} & {t['kind']} & {t['d_agg_disp']:.3f} & "
                     f"{t['d_decision']:.3f} & {t['d_admission']:.3f} & "
                     f"{t['d_influence']:.3f} & {dasr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(lines))

    json.dump({"description": "Cross-aggregator channel table, Mode S identity rung -> kappa=2. "
                              "Assembled read-only from frozen artifacts; no ASR computed here. "
                              "d_agg_disp is the relative displacement of the EMITTED aggregate; it "
                              "was named d_statistic through Round 20, which mislabelled it.",
               "top_rung": TOP, "identity_rung": IDENTITY,
               "mass_definition": {k: v[0] for k, v in MASS.items()},
               "mass_is_binary": {k: v[1] for k, v in MASS.items()},
               "sources": [os.path.relpath(p, base) for p in [ADM, ADM_FEMNIST] + ASR_SOURCES
                           if os.path.exists(p)],
               "rows": table,
               "latex": "\n".join(lines)}, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
