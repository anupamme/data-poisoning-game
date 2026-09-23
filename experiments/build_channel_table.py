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
  results/resnet18_admission.json      channels for the ResNet18 arm, measured PROSPECTIVELY
  results/targeted_dose/summary.json   ASR for the three Round-12 Mode-S arms
  results/dose_replication/summary.json  ASR for the Round-15 coord_median arm
  results/dose_femnist/summary.json     ASR for the FEMNIST arm, if it has been run
  results/dose_resnet18/summary.json    ASR for the ResNet18 arm, if it has been run
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
ADM_RESNET18 = os.path.join(R, "resnet18_admission.json")
ASR_SOURCES = [os.path.join(R, "targeted_dose", "summary.json"),
               os.path.join(R, "dose_replication", "summary.json"),
               os.path.join(R, "dose_femnist", "summary.json"),
               # Safe to merge in ONLY because the key carries the model: this arm is CIFAR-10 and
               # reuses run_targeted_dose's cell_key, so it collides with the flagship row on
               # (dataset, key) and is separated from it by `model` alone. See _asr_cells().
               os.path.join(R, "dose_resnet18", "summary.json"),
               # Safe to merge in: Mode M's cells are keyed `doseM_m<m>_then_...`, a disjoint namespace
               # from `doseS_kappa<k>_then_...`, and its score-only control carries a `|score_only`
               # suffix. _asr_cells() asserts non-shadowing anyway, so a future collision fails loudly
               # rather than silently overwriting a published number.
               os.path.join(R, "dose_mask", "summary.json")]
# NOT in ASR_SOURCES: score_only/summary.json and emit_only/summary.json reuse run_targeted_dose's
# cell_key on the same dataset, so merging either in would overwrite the published Krum row's ASR with
# a controlled arm's. They are read only by score_only_row() / emit_only_row(), and _asr_cells() now
# asserts no source can shadow another.
DECOMP = os.path.join(R, "displacement_decomposition.json")
SCORE_ONLY = os.path.join(R, "score_only", "summary.json")
EMIT_ONLY = os.path.join(R, "emit_only", "summary.json")

MASK = os.path.join(R, "mask_admission.json")

OUT = os.path.join(R, "channel_table.json")
TOP = 2.0          # the top Mode-S rung, rho = exp(4) = 54.6
IDENTITY = 0.0
# The two committed attacks, for the floor cross-check below. Order is fixed so the printed columns
# cannot silently swap; both are present in results/admission_measurement.json.
ATTACKS_XCHECK = ("committed_scaling", "committed_pixel")
MASK_TOP = 0.8     # the top Mode-M rung, a drop rate -- NOT a weight ratio, and never pooled with TOP

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
# (label, arm, attack, channel source). One row per aggregator; the second-dataset row is the same arm
# as the first, on a different dataset and architecture, which is the only reason it is a separate row.
# The LABEL says EMNIST-byclass because that is what fl_core/data_loader.py:52-58 loads for the
# "femnist" key: torchvision EMNIST split="byclass" under this paper's own label-Dirichlet partition,
# NOT LEAF FEMNIST's by-writer partition. The dataset KEY, the artifact paths and ADM_FEMNIST keep the
# `femnist` spelling because they are frozen; only the display string is corrected.
#
# The ResNet18 row varies ARCHITECTURE alone: dataset, N, K, f, alpha, attack and defense are the
# frozen CIFAR-10 values, so it is the only row in the table that differs from row 1 in one factor.
# Its channels were measured PROSPECTIVELY -- results/resnet18_admission.json states "No ASR. Freezes
# nothing." and was written before run_dose_resnet18.py produced any ASR -- which is why the row can
# be read as a prediction that was then scored rather than as a fit.
ROWS = [
    ("Krum",                     "krum",         "committed_scaling", ADM),
    ("Reputation",               "reputation",   "committed_scaling", ADM),
    ("Cosine-Krum",              "cos_krum",     "committed_pixel",   ADM),
    ("Coord.\\ median",          "coord_median", "committed_pixel",   ADM),
    ("Krum (EMNIST-byclass)",    "krum",         "committed_scaling", ADM_FEMNIST),
    ("Krum (ResNet18)",          "krum",         "committed_scaling", ADM_RESNET18),
]
# (dataset, model) of each channel source, so the ASR lookup uses the same triple the artifact is
# stored under. Keyed on the source path because that is what a ROWS entry names.
SRC_REGIME = {ADM: ("cifar10", "cifar_cnn"),
              ADM_FEMNIST: ("femnist", "simple_cnn"),
              ADM_RESNET18: ("cifar10", "resnet18")}

# Mode M, coordinate masking: the SECOND transformation class, emitted as its own block rather than
# appended to the rows above. Two reasons, both substantive. (i) The top rung is a drop rate m=0.8, not
# a weight ratio rho=e^4, so a single table under one "identity -> kappa=2" header would misdescribe
# four of its rows. (ii) `prop:invariance` classifies aggregators under positive rescaling only, so a
# Mode-M row is not a second measurement of the same predicted quantity -- it is a measurement outside
# the family the prediction is about. The attack assignment is the frozen one: class-(c) arms on
# committed_scaling, class-(a)/(b) arms on committed_pixel, identical to the Mode-S block.
# Only `krum` has an ASR ladder; the other three are CHANNEL results and print `---` for Delta ASR,
# which is the existing convention for a row whose ASR is not available rather than a reason to drop it.
MASK_ROWS = [
    ("Krum (mask)",              "krum",         "committed_scaling"),
    ("Reputation (mask)",        "reputation",   "committed_scaling"),
    ("Cosine-Krum (mask)",       "cos_krum",     "committed_pixel"),
    ("Coord.\\ median (mask)",   "coord_median", "committed_pixel"),
]


def channel_rows(path, arm, attack, rung, family="doseS"):
    """Per-round rows of one arm's rung from a channel-measurement file.

    `family` selects the transformation class: "doseS" for positive rescaling, "doseM" for coordinate
    masking. results/mask_admission.json records exactly the same per-round field names as the Mode-S
    measurement -- it is produced by the same loop, passed a different rungs list -- so only the family
    filter and the rung grid differ, and nothing about the columns' meaning changes.
    """
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    return [r for r in d["per_round"]
            if r["family"] == family and abs(r["rung"] - rung) < 1e-12 and r["attack"] == attack]


def channels(path, arm, attack, rung, family="doseS"):
    """Delta aggregate / decision / admission / influence for one arm at one rung.

    Recomputed from the per-round rows rather than read from the file's own summary, so this script
    and the measurement cannot disagree about what the columns mean.
    """
    rows = channel_rows(path, arm, attack, rung, family)
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
    # BASELINE LEVEL, conditional on an adversary actually participating in the round. The Delta adm.
    # column above is a CHANGE and is 0.000 on every row, which is the dissociation; a reader cannot
    # tell from it whether an arm had any room to change. That is the twelfth review's floor objection
    # and it is right about Krum. The conditioning is not cosmetic: in a round where no adversary was
    # sampled the defense has no adversarial mass to admit, so those rounds are structural zeros and
    # pooling them understates every arm's room (coord_median reads 0.2284 unconditionally against
    # 0.2855 here). `mean_mass_base` is left exactly as it was so no emitted number moves.
    adv = [r for r in rows if int(r["n_adv_in_round"]) > 0]
    return {"n_rounds": len(rows), "d_agg_disp": float(np.mean(disp)),
            "d_decision": float(np.mean(dec)), "d_admission": float(np.mean(adm)),
            "d_influence": float(np.mean(infl)), "mass_is_binary": binary,
            "mean_mass_base": float(np.mean([float(r[f"base_{field}"]) for r in rows])),
            "mean_mass_post": float(np.mean([float(r[f"post_{field}"]) for r in rows])),
            "n_rounds_adv_present": len(adv),
            "base_adm_adv": (float(np.mean([float(r[f"base_{field}"]) for r in adv]))
                             if adv else float("nan")),
            # Rounds whose baseline admission is nonzero, i.e. the arm admitted SOME adversarial mass
            # before any transform. This is the per-arm decomposition of the 100-of-240 count the
            # provenance paragraph already reports pooled across the four CIFAR-10 rows.
            "n_rounds_base_nonzero": sum(1 for r in rows if float(r[f"base_{field}"]) > 0.0)}


def _asr_cells():
    """{(dataset, model, cell_key): cell} over every ASR source that exists.

    The dataset is part of the key on purpose. run_dose_femnist.py reuses run_targeted_dose's
    cell_key, so the FEMNIST krum arm and the CIFAR-10 krum arm have the SAME string key -- merging
    the files into one flat dict would silently let one overwrite the other.

    The MODEL is part of the key for the same reason and one step further: the ResNet18 arm varies
    architecture alone, so it shares the flagship arm's dataset AND its cell_key and is distinguished
    by nothing else. `cifar_cnn` is the default because the two sources that predate the field
    (targeted_dose, dose_replication) are both CIFAR-10 `cifar_cnn` arms; widening the key cannot
    move an existing row, since a row is looked up by the same triple it is stored under.
    """
    cells = {}
    for p in ASR_SOURCES:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        ds = d.get("dataset", "cifar10")
        md = d.get("model", "cifar_cnn")
        for k, c in d.get("cells", {}).items():
            # Keying on (dataset, key) is not enough on its own: two suites can share BOTH, as the
            # score-only arm shares them with targeted_dose. A shadowed cell is a silently wrong
            # published number, so refuse rather than resolve it by file order.
            assert (ds, md, k) not in cells, (
                f"cell {(ds, md, k)} appears in {cells[(ds, md, k)]['_source']} and "
                f"{os.path.relpath(p, base)}; one would shadow the other")
            cells[(ds, md, k)] = dict(c, _source=os.path.relpath(p, base))
    return cells


def mean_asr(cell):
    """Recomputed from per-seed rows: imported identity cells carry rows but no stored aggregate."""
    rows = cell.get("per_seed", [])
    return (float(np.mean([r["asr"] for r in rows])) if rows else float("nan"),
            float(np.mean([r["accuracy"] for r in rows])) if rows else float("nan"),
            len(rows))


def asr_delta(arm, attack, dataset="cifar10", family="doseS", top=None, suffix="",
              model="cifar_cnn"):
    """Identity -> top-rung ASR contrast for one arm, in one transformation family.

    `family` picks the cell-key spelling, which differs because the two dials are different quantities:
    Mode S's is a weight ratio (`doseS_kappa<k>`), Mode M's is a drop rate (`doseM_m<m>`). They are
    never pooled, and passing the wrong family simply finds no cell rather than mixing them.
    """
    cells = _asr_cells()
    keyfmt = ("doseS_kappa{r}_then_{a}|{k}" if family == "doseS" else "doseM_m{r}_then_{a}|{k}")
    hi_rung = TOP if top is None else top
    lo = cells.get((dataset, model, keyfmt.format(r=IDENTITY, a=arm, k=attack) + suffix))
    hi = cells.get((dataset, model, keyfmt.format(r=hi_rung, a=arm, k=attack) + suffix))
    if lo is None or hi is None:
        return None
    a0, c0, n0 = mean_asr(lo)
    a2, c2, n2 = mean_asr(hi)
    # delta_asr stays a difference of rung MEANS, unchanged, because that is what this table's column
    # has always been. The paired difference is reported alongside, with a flag, because the two agree
    # only when both rungs carry the same seeds -- which a partly-run ladder does not.
    s0 = {int(r["seed"]): float(r["asr"]) for r in lo.get("per_seed", [])}
    s2 = {int(r["seed"]): float(r["asr"]) for r in hi.get("per_seed", [])}
    shared = sorted(set(s0) & set(s2))
    return {"identity_asr": a0, "top_asr": a2, "delta_asr": a2 - a0,
            "identity_acc": c0, "top_acc": c2, "n_seeds": min(n0, n2),
            "seeds_match": set(s0) == set(s2), "n_shared": len(shared),
            "paired_delta_asr": (float(np.mean([s2[s] - s0[s] for s in shared]))
                                 if shared else None),
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
        # The baseline admission level is a PRE-transform quantity measured on the same raw updates at
        # every rung, so it must not depend on which rung reports it. If it did, the "baseline" the
        # paper prints would silently be a property of the top rung. Checked, not assumed.
        if base_ch is not None and not (np.isnan(ch["base_adm_adv"])
                                        or np.isnan(base_ch["base_adm_adv"])):
            assert abs(ch["base_adm_adv"] - base_ch["base_adm_adv"]) < 1e-12, (
                f"{label}: baseline admission differs between the identity rung "
                f"({base_ch['base_adm_adv']}) and rung {TOP} ({ch['base_adm_adv']}), so it is not a "
                "baseline")
        dataset, model = SRC_REGIME[src]
        asr = asr_delta(arm, attack, dataset, model=model)
        table.append({"label": label, "arm": arm, "attack": attack, "kind": KIND[arm],
                      "dataset": dataset, "model": model,
                      "channel_source": os.path.relpath(src, base),
                      "family": "doseS", "top_rung": TOP, "rungs": [0.0, 0.5, 1.0, 2.0],
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

    # DISPLAY ORDER IS NOT `ROWS` ORDER, and the reason is the paper rather than the data. Four sites
    # index the score-only control as "row 6" of this table (main.tex:378 in the body, plus the two
    # captions and the provenance paragraph), so the ResNet18 row is emitted AFTER it: inserting it
    # beside the other replication, where it belongs by subject, would silently move score-only to
    # row 7 and falsify all four. Reordering here rather than in ROWS keeps the channel loop, the
    # round counter that imports ROWS, and the two controls that read the Krum row all unaffected.
    LAST_ROWS = ("Krum (ResNet18)",)
    table = ([t for t in table if t["label"] not in LAST_ROWS]
             + [t for t in table if t["label"] in LAST_ROWS])

    # The Mode-M block, built separately and never appended to `table`: see MASK_ROWS for why.
    mask_table = []
    for label, arm, attack in MASK_ROWS:
        ch = channels(MASK, arm, attack, MASK_TOP, family="doseM")
        if ch is None:
            print(f"  -- {label}: no Mode-M channel measurement in "
                  f"{os.path.relpath(MASK, base)}; row skipped")
            continue
        base_ch = channels(MASK, arm, attack, IDENTITY, family="doseM")
        mask_table.append({"label": label, "arm": arm, "attack": attack, "kind": KIND[arm],
                           "dataset": "cifar10", "model": "cifar_cnn",
                           "channel_source": os.path.relpath(MASK, base),
                           "family": "doseM", "top_rung": MASK_TOP,
                           "rungs": [0.0, 0.2, 0.5, 0.8],
                           "identity_displacement": (None if base_ch is None
                                                     else base_ch["d_agg_disp"]),
                           **ch,
                           "asr": asr_delta(arm, attack, family="doseM", top=MASK_TOP)})

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

    if mask_table:
        print(f"\n=== SECOND TRANSFORMATION CLASS: Mode M, coordinate masking "
              f"(identity rung -> m = {MASK_TOP}) ===")
        print("    NOT of the form c_i * u_i, so prop:invariance and thm:bounded_reweight make no")
        print("    prediction here -- by construction. The dial is a drop rate, not a weight ratio,")
        print("    and it is never pooled with the Mode-S rungs above.")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for t in mask_table:
            a = t["asr"]
            dasr = "     --- " if a is None else f"{a['delta_asr']:+8.3f}"
            n = "  - " if a is None else f"{a['n_seeds']:>4d}"
            print(f"  {t['label'].replace(chr(92), ''):22s} {t['kind']:34s} {t['d_agg_disp']:8.3f} "
                  f"{t['d_decision']:7.3f} {t['d_admission']:7.3f} {t['d_influence']:7.3f} "
                  f"{dasr} {n}")
        print("\n    Only Krum has a Mode-M ASR ladder; the other three rows are CHANNEL results and")
        print("    print --- for dASR rather than being dropped. The cos_krum/pixel mask ladder was")
        print("    considered and dropped for compute before any ASR existed "
              "(pre_registration_dose_mask.md).")
        ck = next((t for t in mask_table if t["arm"] == "cos_krum"), None)
        ck_s = next((t for t in table if t["arm"] == "cos_krum"), None)
        if ck is not None and ck_s is not None:
            print(f"\n    THE ADJUDICATING CELL: cos_krum is exactly invariant under rescaling "
                  f"(dDec {ck_s['d_decision']:.3f})")
            print(f"    and not invariant under masking (dDec {ck['d_decision']:.3f}, dAdm "
                  f"{ck['d_admission']:.3f}). One class certifies")
            print("    what the other cannot, on the same aggregator, attack and seeds.")

    # A row whose two rungs do not carry the same seeds has a difference-of-means dASR that is not the
    # paired difference. Said out loud, because a partly-run ladder produces exactly that.
    mismatched = [t for t in table + mask_table
                  if t["asr"] is not None and not t["asr"].get("seeds_match", True)]
    if mismatched:
        print("\n=== SEED-SET MISMATCH: dASR is a difference of rung MEANS, not a paired difference ===")
        for t in mismatched:
            a = t["asr"]
            paired = "--" if a["paired_delta_asr"] is None else f"{a['paired_delta_asr']:+.4f}"
            print(f"  {t['label'].replace(chr(92), ''):22s} dASR(means) {a['delta_asr']:+.4f}  "
                  f"paired on the {a['n_shared']} shared seeds {paired}")
        print("  A ladder still running produces this. Quote the paired figure, not the column.")

    print("\n=== SANITY: the identity rung must displace nothing ===")
    for t in table + mask_table:
        v = t["identity_displacement"]
        print(f"  {t['label'].replace(chr(92), ''):22s} identity displacement "
              + ("missing" if v is None else f"{v:.2e}"
                 + ("  OK" if v is not None and v < 1e-9 else "  <- NOT ZERO")))

    print("\n=== THE FULL LADDERS, for the appendix ===")
    for t in [t for t in table + mask_table if "borrowed_from" not in t]:
        fam = t.get("family", "doseS")
        dial = "kappa" if fam == "doseS" else "m"
        keyfmt = ("doseS_kappa{r}_then_{a}|{k}" if fam == "doseS" else "doseM_m{r}_then_{a}|{k}")
        print(f"\n  -- {t['label'].replace(chr(92), '')} ({t['kind']}, {t['dataset']}) --")
        print(f"  {dial:>6s} {'dAgg':>8s} {'dDec':>7s} {'dAdm':>7s} {'dInfl':>7s} "
              f"{'ASR':>8s} {'acc':>6s}")
        cells = _asr_cells()
        for k in t.get("rungs", [0.0, 0.5, 1.0, 2.0]):
            ch = channels(os.path.join(base, t["channel_source"]), t["arm"], t["attack"], k, fam)
            c = cells.get((t["dataset"], t.get("model", "cifar_cnn"),
                           keyfmt.format(r=k, a=t["arm"], k=t["attack"])))
            a, ac, _ = mean_asr(c) if c else (float("nan"), float("nan"), 0)
            if ch is None:
                continue
            # `--`, not `nan`, for a rung with no ASR runs: this block is headed "for the appendix" and
            # a printed nan is exactly the kind of thing that gets transcribed. No Mode-S rung is
            # affected -- all of theirs exist -- so this changes only rungs that have nothing to report.
            sa = "     --" if np.isnan(a) else f"{a:8.3f}"
            sc = "    --" if np.isnan(ac) else f"{ac:6.3f}"
            print(f"  {k:>6} {ch['d_agg_disp']:8.3f} {ch['d_decision']:7.3f} "
                  f"{ch['d_admission']:7.3f} {ch['d_influence']:7.3f} {sa} {sc}")

    # LaTeX. Emitted here so the paper's table cannot drift from the numbers above.
    # Signed values are wrapped in $...$ deliberately: a bare "-0.026" in a tabular cell sets a
    # text-mode HYPHEN, which is visibly shorter than the true minus every other signed number in
    # both papers gets from math mode. The absent-value cell stays "---", which is the papers'
    # convention for a missing tabular entry and is correct in text mode.
    # The `n` column is emitted too. It was added to both paper tables by hand in Round 63, which left
    # the one number in them that this script could not check -- and n is load-bearing here, because
    # rows carry 3, 5 and 8 seeds. Only the ROWS are copy-verbatim: the two paper tables use different
    # tabular preambles on purpose (tab:channels is tight-set with @{}, tab:channels_full is centred),
    # so the preamble below matches neither and is not meant to be.
    # === BASELINE ADMISSION, the floor question ===
    # Every number the paper quotes about "room to change" is printed here with its referent, because
    # the three quantities in play are easy to swap: the LEVEL (conditional on an adversary being
    # sampled), the count of rounds with NONZERO baseline, and the count of rounds where an adversary
    # was sampled at all. They read 0.2855 / 48 / 48 for coord_median and 0.0833 / 4 / 48 for
    # cos_krum, so quoting the wrong one would turn a near-floor arm into a roomy one.
    print("\n=== BASELINE ADVERSARIAL ADMISSION (pre-transform level, the floor question) ===")
    print("    The Delta adm. column is a CHANGE and is 0.000 on every row. These are LEVELS: how")
    print("    much adversarial mass each arm admitted before any transform, so how much room its")
    print("    admission had to move at all.\n")
    print(f"  {'aggregator':22s} {'baseline':>9s} {'nonzero':>9s} {'adv seen':>9s} {'rounds':>7s}"
          f"  {'floor?':>7s}")
    print("  " + "-" * 68)
    for t in table:
        if np.isnan(t.get("base_adm_adv", float("nan"))):
            continue
        b = t["base_adm_adv"]
        nz, na, nr = t["n_rounds_base_nonzero"], t["n_rounds_adv_present"], t["n_rounds"]
        # "at the floor" is exactly b == 0: the arm admitted no adversarial mass in any round, so
        # unchanged admission is arithmetically forced and carries no information.
        flag = "AT FLOOR" if b == 0.0 else ("near" if nz <= na // 4 else "no")
        print(f"  {t['label'].replace(chr(92), ''):22s} {b:9.4f} {nz:9d} {na:9d} {nr:7d}  {flag:>7s}")
    print("\n  baseline  mean pre-transform adversarial mass, over rounds where an adversary was")
    print("            sampled; rounds with no adversary are structural zeros and are excluded")
    print("  nonzero   rounds whose baseline admission is > 0 (this arm's share of the 100/240)")
    print("  AT FLOOR  baseline is exactly 0: unchanged admission is forced, not evidence")

    # === IS THE FLOOR A PROPERTY OF THE AGGREGATOR OR OF THE ATTACK? ===
    # The block above reports each row at ITS OWN committed attack, which is what the paper's tables
    # do, and on that reading every Krum row reads 0.0000. That invites the inference that Krum never
    # admits an adversary. This block holds the aggregator fixed and varies the attack instead, which
    # is the only way to tell the two explanations apart. It reads only the frozen CIFAR-10 artifact
    # and adds no row to either paper table.
    print("\n=== THE SAME AGGREGATORS, BOTH ATTACKS (is the floor the aggregator or the attack?) ===")
    print(f"  {'aggregator':16s}", end="")
    for a in ATTACKS_XCHECK:
        print(f" {a.replace('committed_', ''):>22s}", end="")
    print()
    print("  " + "-" * 62)
    for arm in ("krum", "cos_krum", "reputation", "coord_median"):
        print(f"  {arm:16s}", end="")
        for atk in ATTACKS_XCHECK:
            ch = channels(ADM, arm, atk, TOP)
            if ch is None or np.isnan(ch["base_adm_adv"]):
                print(f" {'--':>22s}", end="")
                continue
            print(f" {ch['base_adm_adv']:11.4f} ({ch['n_rounds_base_nonzero']:2d}/"
                  f"{ch['n_rounds_adv_present']:2d})", end="")
        print()
    print("\n  Read as: level (rounds with nonzero baseline / rounds an adversary was sampled).")
    print("  A row that is 0.0000 under one attack and nonzero under the other has no aggregator-level")
    print("  floor: the zero belongs to the attack. Krum is the case that matters, since the paper's")
    print("  adjudicating arm commits to scaling, and a scaling attack IS a norm outlier.")

    print("\n=== LATEX (copy the ROWS verbatim; regenerate rather than edit) ===\n")
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Aggregator & Kind & $\Delta$ agg. & $\Delta$ dec. & $\Delta$ adm. "
             r"& $\Delta \Lambda_a$ & $\Delta$ ASR & $n$ \\", r"\midrule"]
    for t in table:
        a = t["asr"]
        dasr = "---" if a is None else f"${a['delta_asr']:+.3f}$"
        ncol = "---" if a is None else f"{a['n_seeds']}"
        lines.append(f"{t['label']} & {t['kind']} & {t['d_agg_disp']:.3f} & "
                     f"{t['d_decision']:.3f} & {t['d_admission']:.3f} & "
                     f"{t['d_influence']:.3f} & {dasr} & {ncol} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(lines))

    # ONE CELL PAIR IN THE PAPER DELIBERATELY DIFFERS FROM THIS OUTPUT, and "regenerate rather than
    # edit" above would silently revert it. pre_registration_score_only_kappa2_n20.md (15d02e4) requires
    # the score-only row to report n=20, and the n=20 contrast cannot be assembled from
    # results/score_only/ alone: it needs the 15 new kappa=2 runs in results/score_only_kappa2_topup/
    # plus the imported kappa=0 leg, which analyze_score_only_kappa2_topup.py merges with both legs'
    # seed sets asserted. Rather than reimplement that three-source merge here, where a divergence from
    # the analyzer would be a silently wrong published number, this script keeps printing the published
    # n=5 contrast and NAMES the difference. Both values stand at their own seed counts.
    print("\n  NOTE on the score-only row, so a regeneration cannot silently revert the paper:")
    print("  the row above prints this script's results/score_only/ contrast at its own seed count,")
    print("  and main.tex's tab:channels / tab:channels_full print the n=20 top-up instead, as")
    print("  pre_registration_score_only_kappa2_n20.md requires. Take that cell from")
    print("  `PYTHONPATH=. python3 -m experiments.analyze_score_only_kappa2_topup`, not from here.")

    # A SECOND tabular, not extra rows in the first. The Mode-S LaTeX above is therefore unchanged
    # byte for byte by this round's addition, which is what lets the paper's existing tab:channels stay
    # as it is while the new class gets a table whose header states its own rung.
    mask_lines = []
    if mask_table:
        mask_lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
                      r"Aggregator & Kind & $\Delta$ agg. & $\Delta$ dec. & $\Delta$ adm. "
                      r"& $\Delta \Lambda_a$ & $\Delta$ ASR \\", r"\midrule"]
        for t in mask_table:
            a = t["asr"]
            dasr = "---" if a is None else f"${a['delta_asr']:+.3f}$"
            mask_lines.append(f"{t['label']} & {t['kind']} & {t['d_agg_disp']:.3f} & "
                              f"{t['d_decision']:.3f} & {t['d_admission']:.3f} & "
                              f"{t['d_influence']:.3f} & {dasr} \\\\")
        mask_lines += [r"\bottomrule", r"\end{tabular}"]
        print(f"\n=== LATEX, Mode M block (identity rung -> m = {MASK_TOP}) ===\n")
        print("\n".join(mask_lines))

    json.dump({"description": "Cross-aggregator channel table, Mode S identity rung -> kappa=2. "
                              "Assembled read-only from frozen artifacts; no ASR computed here. "
                              "d_agg_disp is the relative displacement of the EMITTED aggregate; it "
                              "was named d_statistic through Round 20, which mislabelled it.",
               "top_rung": TOP, "identity_rung": IDENTITY, "mask_top_rung": MASK_TOP,
               "mass_definition": {k: v[0] for k, v in MASS.items()},
               "mass_is_binary": {k: v[1] for k, v in MASS.items()},
               "sources": [os.path.relpath(p, base) for p in [ADM, ADM_FEMNIST, MASK] + ASR_SOURCES
                           if os.path.exists(p)],
               "rows": table,
               # Mode M kept in its own list and its own LaTeX field, so nothing consuming `rows` or
               # `latex` sees a row measured at a different dial without asking for it.
               "mask_rows": mask_table,
               "latex": "\n".join(lines),
               "latex_mask": "\n".join(mask_lines)}, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
