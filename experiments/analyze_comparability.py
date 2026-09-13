"""
Score the confounded-vs-controlled comparability table against the rules frozen in
`experiments/pre_registration_comparability.md` (c986ef4, Amendments 1-4, latest 9b8a395).

HOW MANY CELLS: ask the artifact, not this docstring. `CELLS` defines them and `n_cells` counts the
COMPLETE ones. This header said "six-cell" through Round 56 and a seventh was added in Round 57; a
count written in prose beside the list it describes is a count that will be wrong.

WHAT THIS DECIDES AND WHAT IT CANNOT
Each cell contrasts two estimates of the SAME quantity on the SAME arm, attack and seeds: the
outcome-gated (confounded) ladder `dose_kappa<k>`, in which the adversary is free to be attenuated,
and the controlled `doseS_kappa<k>` intervention, in which every adversary is pinned at c=1. The cell
says whether the two designs reach the same conclusion. **It does not say which is right about the
world** -- Mode S is an oracle instrument that reads adversary identity, so a cell where the designs
agree is not a licence to use the confounded design in general, only a report that here they coincide.

FOUR CELLS ARE TRAINING DATA AND ARE LABELLED AS SUCH
The frozen rule was read off the four cells that already existed, so they cannot also be evidence for
it. They are printed as TRAINING. Cells 5 and 6 are the out-of-sample test of the frozen rule and are
the only rows that can confirm or refute it. Cell 7 (Amendment 4) is out-of-sample too but tests a
DIFFERENT question -- whether the sign reversal itself replicates on a third dataset -- and the rule
makes no prediction about it, so it carries no confirm/refute verdict and must never be counted as one.

THE GATING QUANTITY IS ΔΛ_a, NOT THE SUPPORT CHANGE
Amendment 1 settles this: `d_admission` (whether the SUPPORT of the adversarial mass moved) is
identically 0.0000 on all four training cells -- that constancy is the paper's own headline finding --
so it has no discriminating power. The rule thresholds `d_influence`, the admission-LEVEL change ΔΛ_a,
at zero. Both quantities are printed side by side so the choice stays visible.

EACH DESIGN IS SCORED AT ITS OWN FULL n, AND AMENDMENT 3 EXPLAINS WHY
The obvious-looking alternative -- intersect the two designs' seed sets and compare them on shared
seeds -- was tried, and it made the four PUBLISHED cells fail to reproduce: krum/scaling's controlled
ladder moves -0.010 (n=20) to -0.026 when cut to the confounded side's 5 seeds, and cos_krum/pixel
-0.425 (n=8) to -0.272. Non-negotiable 1's bit-identical reproduction gate rejected it, which is proof
that full-n is the convention c986ef4 scored. "Paired" in the freeze means paired across a design's two
RUNGS at identical seeds within that design, which `_endpoint` does.
This mattered: shared-seed scoring would have flipped cell 6 to AGREE and rescued H-ADMISSION-GATED.
It is retained as a disclosed robustness check, printed in the unequal-n warning, and is NEVER the
verdict. The mechanism is refuted and stays refuted.

t_crit IS IMPORTED, NEVER `T95[n-1]`
`analyze_headline_cis.T95` is a literal table that stops at df=9, so reading it directly raises
KeyError the moment an arm reaches n=11. The flagship controlled ladder is at n=20.

Reads frozen artifacts and results/comparability_cells/. Adds no runs. Writes exactly one artifact,
results/comparability_six_cells.json, which is what Fig. 1(c) draws -- the figure recomputes nothing
and quotes no number this script did not emit.
Run: python3 experiments/analyze_comparability.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_headline_cis import t_crit                      # noqa: E402  NOT T95[n-1]
from experiments.build_channel_table import channels, ADM                # noqa: E402
# Imported, never reimplemented: the dose--response ladders, this table and the paper's captions must
# not be able to quote two differently-derived J or z for the same four rungs. Same reason
# analyze_tost_existing.py imports t_crit instead of restating a table.
from experiments.analyze_dose_response import jonckheere                 # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = "experiments/pre_registration_comparability.md"
PREREG_COMMIT = "9b8a395"          # freeze c986ef4 + Amendments 1, 2, 3 + Amendment 4 (cell 7, the
                                   # reversal cell on a third dataset, and its admissibility gate)
FEMNIST_ADM = os.path.join(BASE, "results", "femnist_admission.json")

LO, HI = "0.0", "2.0"              # the ladder endpoints every existing arm is scored on
NEAR_ZERO = 0.05                   # frozen: "both point estimates within +-0.05 of zero" => AGREE
# The frozen rung grid. The SCORED primary is the LO-to-HI endpoint contrast and reads none of the
# interior rungs, so a cell whose interior is still running scores correctly and completely -- but it
# is NOT a four-rung ladder, and the difference is invisible in every column this script prints.
# `rung_coverage` exists to make it visible: an endpoint-only cell has no monotone-trend evidence and
# its Jonckheere--Terpstra secondary is unavailable, not null. Reporting a two-rung cell as the
# four-rung ladder its neighbours are is the same misreport as reporting a smaller n as the full one.
GRID = ("0.0", "0.5", "1.0", "2.0")

CONF = "dose_kappa{k}_then_{d2}|{atk}{sfx}"
CTRL = "doseS_kappa{k}_then_{d2}|{atk}{sfx}"

# label, d2, attack, [(dir, keyfmt, suffix)] for confounded, same for controlled, training?, adm source
CELLS = [
    ("krum / scaling",             "krum",         "committed_scaling",
     [("dose_response", CONF, "")], [("targeted_dose", CTRL, ""), ("dose_seed_topup", CTRL, "")],
     True,  (ADM, "krum", "committed_scaling")),
    ("cos_krum / pixel",           "cos_krum",     "committed_pixel",
     [("dose_response", CONF, "")], [("targeted_dose", CTRL, "")],
     True,  (ADM, "cos_krum", "committed_pixel")),
    ("reputation / scaling",       "reputation",   "committed_scaling",
     [("dose_response", CONF, "")], [("targeted_dose", CTRL, "")],
     True,  (ADM, "reputation", "committed_scaling")),
    # Round 63. The sign reversal is the cell the paper is built on and it was the ONLY cell still at
    # n=5 on BOTH legs, so `reversal_seed_topup` extends both to n=20 at the endpoint rungs. The top-up
    # directory is appended AFTER the published ones because `series` is first-writer-wins: seeds 42--46
    # keep their published values and 47--61 are added, so the published contrast stays recoverable as
    # the n=5 subset (PUB_SEEDS below) rather than being averaged away. Frozen in
    # experiments/pre_registration_reversal_seed_topup.md before the first new run.
    ("coord_median / pixel",       "coord_median", "committed_pixel",
     [("dose_response", CONF, ""), ("reversal_seed_topup", CONF, "")],
     [("dose_replication", CTRL, ""), ("reversal_seed_topup", CTRL, "")],
     True,  (ADM, "coord_median", "committed_pixel")),
    # --- out of sample ---
    ("coord_median / scaling",     "coord_median", "committed_scaling",
     [("comparability_cells", CONF, "")], [("comparability_cells", CTRL, "")],
     False, (ADM, "coord_median", "committed_scaling")),
    ("krum / scaling, EMNIST",     "krum",         "committed_scaling",
     [("comparability_cells", CONF, "|emnist")],
     [("dose_femnist", CTRL, ""), ("comparability_cells", CTRL, "|emnist")],
     False, (FEMNIST_ADM, "krum", "committed_scaling")),
    # Amendment 4 (Round 57): the reversal cell itself, moved to a third dataset. `admsrc` is None on
    # purpose and is NOT an omission. H-ADMISSION-GATED was REFUTED in Amendment 3, so the frozen rule
    # generates no prediction for this cell; there is also no CIFAR-100 admission artifact and none is
    # computed, because computing one now and reading a threshold off it is exactly how a withdrawn
    # mechanism gets rescued. dLam_a and `predicted` therefore print as absent-value cells, which is
    # what Amendment 4 pre-registered. This cell tests ONE thing: whether the sign reversal replicates.
    ("coord_median / pixel, CIFAR-100", "coord_median", "committed_pixel",
     [("comparability_cells", CONF, "|cifar100")], [("comparability_cells", CTRL, "|cifar100")],
     False, None),
]

# Cells whose seed set is frozen by an amendment, keyed by label. BOTH designs must carry EVERY seed
# listed here or the cell is INCOMPLETE and leaves every count -- `conf`/`ctrl` being merely non-None is
# not completeness. Without this, a cell mid-run scores: cell 7's controlled kappa=2 rung passed through
# n=4 of 5 while the run was live, and at that moment `complete` was true, a verdict was formed, and the
# two legs were compared at n=5 against n=4. That is the mixed-n defect Amendment 2 was written about,
# reappearing through a completeness test rather than through a top-up.
# The six earlier cells are deliberately absent: their seed sets are heterogeneous by publication
# history (n=20, 8, 5, 3) and asserting one here would break the reproduction gate below.
REQUIRED_SEEDS = {
    "coord_median / pixel, CIFAR-100": (42, 43, 44, 45, 46),      # Amendment 4, "seeds 42--46, frozen"
}

# The four published contrasts, transcribed from the pre-registration's own table so the re-score can
# be checked against the document rather than against itself. Asserted, never printed as a result.
PUB = {"krum / scaling":       (-0.008, -0.010),
       "cos_krum / pixel":     (-0.405, -0.425),
       "reputation / scaling": (+0.762, +0.178),
       "coord_median / pixel": (-0.272, +0.098)}
PUB_TOL = 5e-3                      # the prereg prints 3 decimals

# The seed set each published contrast in PUB was computed on, declared ONLY for cells a later top-up
# has since added seeds to. Absent means the cell's full n still IS its published n, so the guard reads
# the full-n contrast as before.
#
# Why this exists rather than an edit to PUB. Round 63's top-up moves coord_median/pixel from n=5 to
# n=20 on both legs, so its full-n contrast is a DIFFERENT quantity from the published one and the
# assertion below would fail. The tempting repair -- retyping (-0.272, +0.098) as whatever the n=20
# re-score prints -- would destroy the only check that this script reproduces the document, since the
# tuple would then be asserted against the number it was copied from. So the published pair stays
# verbatim and is asserted against the n=5 SUBSET, which the merge deliberately keeps recoverable, and
# the n=20 pair is reported as a new result rather than as a reproduction.
PUB_SEEDS = {
    "coord_median / pixel": {"confounded": (42, 43, 44, 45, 46),
                             "controlled": (42, 43, 44, 45, 46)},
}

# What `paper/main.tex` prints for the topped-up cell, so a later re-score cannot drift from the
# document. Left None until the run finishes and the paper quotes it: a value written here before the
# run would be a prediction, and a value copied from the artifact and asserted against that same
# artifact would check nothing. Fill it from the paper, not from this script's output.
PAPER_N20 = {"coord_median / pixel": None}


def cells_of(dirname):
    p = os.path.join(BASE, "results", dirname, "summary.json")
    return json.load(open(p)).get("cells", {}) if os.path.exists(p) else {}


def series(sources, d2, atk, kappa):
    """{seed: asr} merged over the directories that carry this ladder; first writer wins."""
    out = {}
    for dirname, keyfmt, sfx in sources:
        c = cells_of(dirname).get(keyfmt.format(k=kappa, d2=d2, atk=atk, sfx=sfx))
        if not c:
            continue
        for r in c.get("per_seed", []):
            out.setdefault(int(r["seed"]), float(r["asr"]))
    return out


def rung_coverage(conf_src, ctrl_src, d2, atk):
    """Which frozen rungs this cell holds on BOTH designs, and whether the interior is complete.

    Read-only over the same artifacts `contrast` reads. A rung counts as present only if both designs
    have at least one seed at it, because a rung present on one side alone cannot enter any contrast.
    """
    present = [k for k in GRID
               if series(conf_src, d2, atk, k) and series(ctrl_src, d2, atk, k)]
    return {"rungs_present": present, "rungs_frozen": list(GRID),
            "endpoint_primary_available": LO in present and HI in present,
            "full_grid": len(present) == len(GRID),
            "trend_secondary_available": len(present) == len(GRID)}


# Above this total N the permutation leg is skipped: jt_statistic is a pure-Python quadruple loop, so
# 20000 permutations of krum/scaling's 4x20 controlled ladder is ~48M comparisons. The normal
# approximation is computed for every cell regardless, and `trend_report` PRINTS which cells got a
# permutation p and which did not, because a silently-skipped leg reads as an absent effect.
PERM_MAX_N = 24


def trend(conf_src, ctrl_src, d2, atk, rungs):
    """Post hoc Jonckheere--Terpstra trend across the four frozen rungs, per design.

    POST HOC, AND LABELLED SO EVERYWHERE. `pre_registration_comparability.md` never registers a trend
    secondary for any cell of this table -- the string "Jonckheere" does not occur in it, and the frozen
    primary is the LO-to-HI endpoint contrast alone. So this is a DESCRIPTION of the ladder's shape, not
    a scored secondary, and no verdict in this script reads it. It is computed and printed because the
    interior rungs exist: the paper's own disclosure used to rest the absence of a trend statistic on
    the rungs not being run, and once they are run that ground is gone. Withholding a statistic that is
    now computable would be the worse of the two failures.

    Not paired, unlike `_endpoint`: JT is a between-group rank test, so each design contributes its own
    seeds at each rung and the two designs are never pooled.

    TWO VERSIONS, because an endpoint-only top-up makes the four rungs unequal in n. A cell topped up at
    kappa=0 and kappa=2 only has rungs of size [20, 5, 5, 20], and JT's null distribution is a function
    of those group sizes, so the mixed-n statistic is not the statistic the frozen grid was described
    with. `all_seeds` is the mixed-n version and `common_seeds` restricts every rung to the seeds all
    four share -- which is the pre-top-up block, and therefore reproduces the published trend values
    exactly. For every cell that was never topped up the two are identical and `equal_n` is True, so
    this adds a distinction only where one exists. The paper quotes `common_seeds`; the mixed-n version
    is emitted so that a reader can see it was computed and deliberately not quoted.
    """
    if not rungs["trend_secondary_available"]:
        return None
    out = {}
    for side, src in (("confounded", conf_src), ("controlled", ctrl_src)):
        per_rung = [series(src, d2, atk, k) for k in GRID]
        common = sorted(set.intersection(*[set(s) for s in per_rung]))
        variants = {"all_seeds": [[v for _, v in sorted(s.items())] for s in per_rung],
                    "common_seeds": [[s[x] for x in common] for s in per_rung]}
        # Not named "common_seeds": that key holds the statistics variant below, and a dict whose value
        # is sometimes a seed list and sometimes a stats block is how a reader quotes the wrong thing.
        block = {"pre_registered": False, "common_seed_block": common,
                 "equal_n": len({len(g) for g in variants["all_seeds"]}) == 1}
        for name, groups in variants.items():
            n_tot = sum(len(g) for g in groups)
            J, z, p_inc, p_dec, p_perm = jonckheere(groups,
                                                    n_perm=(20000 if n_tot <= PERM_MAX_N else 0))
            block[name] = {"rung_means": [float(np.mean(g)) for g in groups],
                           "n_per_rung": [len(g) for g in groups], "J": float(J), "z": z,
                           "p_increasing": p_inc, "p_decreasing": p_dec,
                           "p_perm_increasing": (p_perm if n_tot <= PERM_MAX_N else None)}
        # The quoted one, so downstream readers of this JSON cannot pick the wrong variant by default.
        block.update(block["common_seeds" if not block["equal_n"] else "all_seeds"])
        block["quoted_variant"] = "common_seeds" if not block["equal_n"] else "all_seeds"
        out[side] = block
    return out


def _endpoint(lo, hi, seeds):
    """Paired endpoint contrast over exactly `seeds`, with a 95% t interval."""
    if len(seeds) < 2:
        return None
    d = np.array([hi[s] - lo[s] for s in seeds], dtype=float)
    n = len(d)
    m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)); hw = t_crit(n) * se
    return {"mean": m, "n": n, "lo": m - hw, "hi": m + hw, "seeds": list(seeds)}


def contrast(conf_src, ctrl_src, d2, atk, pub_seeds=None):
    """Each design's endpoint contrast at its own full n (SCORED), and on the shared seeds (CHECK).

    Amendment 3. `unpaired_*` is the scored pair -- each design at its own full n, the frozen
    convention, and the only one under which the four published cells reproduce. `conf`/`ctrl` restrict
    both designs to the seeds they share and exist solely so the unequal-n warning can disclose what a
    shared-seed comparison would have said. The caller must not score on them.

    Round 63. `pub_*` restricts each design to the seeds its PUBLISHED contrast was computed on, and is
    present only for a cell a top-up has extended. It is what the reproduction guard asserts against and
    what the paper reports beside the new n, so the smaller published verdict stays a live quantity
    instead of being replaced by the larger one.
    """
    cl, ch = series(conf_src, d2, atk, LO), series(conf_src, d2, atk, HI)
    tl, th = series(ctrl_src, d2, atk, LO), series(ctrl_src, d2, atk, HI)
    own_c = sorted(set(cl) & set(ch))
    own_t = sorted(set(tl) & set(th))
    shared = sorted(set(own_c) & set(own_t))
    out = {"conf": _endpoint(cl, ch, shared), "ctrl": _endpoint(tl, th, shared),
           "unpaired_conf": _endpoint(cl, ch, own_c), "unpaired_ctrl": _endpoint(tl, th, own_t),
           "shared": shared, "dropped_conf": sorted(set(own_c) - set(shared)),
           "dropped_ctrl": sorted(set(own_t) - set(shared))}
    if pub_seeds:
        out["pub_conf"] = _endpoint(cl, ch, [s for s in pub_seeds["confounded"] if s in own_c])
        out["pub_ctrl"] = _endpoint(tl, th, [s for s in pub_seeds["controlled"] if s in own_t])
        # Declared but unformable is a hard error at the call site, not a silent fallback: a subset that
        # quietly went missing would make the reproduction guard pass by having nothing to compare.
        out["pub_seeds_declared"] = {k: list(v) for k, v in pub_seeds.items()}
    return out


def verdict(a, b):
    """The frozen classification. AGREE / DISAGREE / SIGN REVERSAL, in those exact terms."""
    if a is None or b is None:
        return None
    opposite = (a["mean"] < 0) != (b["mean"] < 0)
    disjoint = a["hi"] < b["lo"] or b["hi"] < a["lo"]
    both_near_zero = abs(a["mean"]) < NEAR_ZERO and abs(b["mean"]) < NEAR_ZERO
    if opposite and a["lo"] * a["hi"] > 0 and b["lo"] * b["hi"] > 0:
        return "SIGN REVERSAL"
    if both_near_zero and not disjoint:
        return "AGREE"
    if disjoint or opposite:
        return "DISAGREE"
    return "AGREE"


def check_prereg():
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", PREREG],
                         cwd=BASE, capture_output=True, text=True, timeout=20)
    a = out.stdout.strip()
    ok = a and a.startswith(PREREG_COMMIT[:7])
    print(f"[{'OK' if ok else 'MISMATCH'}] {PREREG}: recorded {PREREG_COMMIT}, "
          f"{'committed at ' + a if a else 'UNTRACKED'}")
    return bool(ok)


def main():
    if not check_prereg():
        print("Refusing to score: the frozen rules are not at the recorded commit.")
        return 1

    rows, problems, unequal = [], [], []
    pub_reproduced = {}      # topped-up cells: the published contrast on its own seed subset
    for (label, d2, atk, csrc, tsrc, training, admsrc) in CELLS:
        con = contrast(csrc, tsrc, d2, atk, PUB_SEEDS.get(label))
        # The SCORED contrast is each design at its own full n -- the frozen convention. Amendment 3
        # records why: under cross-design pairing the four published cells do not reproduce (krum
        # /scaling's controlled ladder moves -0.010 -> -0.026 when its n=20 is cut to the confounded
        # side's 5 seeds), which proves full-n is what c986ef4 scored. The shared-seed contrast is
        # printed as a disclosed robustness check and never as the verdict.
        a, b = con["unpaired_conf"], con["unpaired_ctrl"]
        if con["dropped_conf"] or con["dropped_ctrl"]:
            unequal.append((label, con))
        # admsrc is None for cells the withdrawn rule makes no prediction about (Amendment 4). An
        # absent admission source must produce an absent-value cell, never a computed-on-the-fly one.
        ch = None
        if admsrc is not None:
            admpath, admarm, admatk = admsrc
            ch = channels(admpath, admarm, admatk, 2.0) if os.path.exists(admpath) else None
        lam = ch["d_influence"] if ch else None
        sup = ch["d_admission"] if ch else None
        pred = None if lam is None else ("DISAGREE" if lam > 0.0 else "AGREE")
        # A frozen seed set not yet fully present on BOTH legs means the ladder is still running. No
        # verdict is formed from it -- not a provisional one, not a parenthesised one -- because the
        # only thing a partial ladder can do here is be quoted.
        want = REQUIRED_SEEDS.get(label)
        missing = {} if not want else {
            side: sorted(set(want) - set(e["seeds"] if e else []))
            for side, e in (("confounded", a), ("controlled", b))
            if sorted(set(want) - set(e["seeds"] if e else []))}
        got = None if missing else verdict(a, b)
        rg = rung_coverage(csrc, tsrc, d2, atk)
        rows.append({"label": label, "conf": a, "ctrl": b, "lam": lam, "sup": sup, "pred": pred,
                     "got": got, "training": training, "con": con, "missing_seeds": missing,
                     "rungs": rg, "trend": trend(csrc, tsrc, d2, atk, rg)})
        if training and label in PUB:
            pc, pt = PUB[label]
            # The guard reads the seed set the published number was computed on. For a cell no top-up
            # has touched that is the full n and nothing changes; for a topped-up cell it is the n=5
            # subset, and the full-n pair below is a new result rather than a reproduction.
            if label in PUB_SEEDS:
                ga, gb, on = con.get("pub_conf"), con.get("pub_ctrl"), "the published seed subset"
                if ga is None or gb is None:
                    problems.append(f"{label}: PUB_SEEDS declares "
                                    f"{con.get('pub_seeds_declared')} but the subset cannot be formed "
                                    f"from the merged artifacts, so the published contrast is "
                                    f"unverifiable. Refusing to treat an absent check as a pass.")
                    ga = gb = None
            else:
                ga, gb, on = a, b, "the full n"
            if ga and gb and (abs(ga["mean"] - pc) > PUB_TOL or abs(gb["mean"] - pt) > PUB_TOL):
                problems.append(f"{label}: re-scored on {on} ({ga['mean']:+.3f}, {gb['mean']:+.3f}) "
                                f"n({ga['n']},{gb['n']}) != pre-registered ({pc:+.3f}, {pt:+.3f})")
            elif ga and gb and label in PUB_SEEDS:
                pub_reproduced[label] = {
                    "confounded": {k: ga[k] for k in ("mean", "lo", "hi", "n")},
                    "controlled": {k: gb[k] for k in ("mean", "lo", "hi", "n")},
                    "verdict": verdict(ga, gb),
                    "note": "The published contrast, recomputed on its own seed subset after the "
                            "top-up. Reported alongside the full-n pair, never replaced by it."}
            # What the paper prints, checked against what this script emits. None until the paper
            # quotes the new n; see PAPER_N20.
            want = PAPER_N20.get(label)
            if want and a and b and (abs(a["mean"] - want[0]) > PUB_TOL
                                     or abs(b["mean"] - want[1]) > PUB_TOL):
                problems.append(f"{label}: paper prints ({want[0]:+.3f}, {want[1]:+.3f}) but the "
                                f"re-score emits ({a['mean']:+.3f}, {b['mean']:+.3f})")

    if problems:
        print("\nREFUSING TO PRINT A VERDICT -- the published contrasts do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("Published contrasts reproduce to the pre-registration's printed precision.\n")

    # Topped-up cells. Printed BEFORE the table, like the unequal-n warning, so the two n cannot be read
    # as one. The published verdict is not superseded by the larger n; where they disagree, that
    # disagreement is the result (pre_registration_reversal_seed_topup.md, non-negotiable 6).
    for label, pr in pub_reproduced.items():
        r = next(x for x in rows if x["label"] == label)
        a, b = r["conf"], r["ctrl"]
        pa, pb = pr["confounded"], pr["controlled"]
        print(f"  ** TOPPED-UP CELL: {label} **")
        print(f"    published  n={pa['n']:<3d} confounded {pa['mean']:+.4f} "
              f"[{pa['lo']:+.4f},{pa['hi']:+.4f}]   controlled {pb['mean']:+.4f} "
              f"[{pb['lo']:+.4f},{pb['hi']:+.4f}]   -> {pr['verdict']}")
        print(f"    scored     n={a['n']:<3d} confounded {a['mean']:+.4f} "
              f"[{a['lo']:+.4f},{a['hi']:+.4f}]   controlled {b['mean']:+.4f} "
              f"[{b['lo']:+.4f},{b['hi']:+.4f}]   -> {r['got']}"
              + ("" if a["n"] == b["n"] else f"   ** LEGS UNEQUAL: n={a['n']} vs {b['n']} **"))
        if pr["verdict"] != r["got"]:
            print("    ** THE TWO n DISAGREE. Both are reported and the disagreement is the result; the")
            print("       larger n does not silently replace the published verdict.")
        if a["n"] != b["n"]:
            print("    The top-up runs both legs at identical seeds, so unequal n here means the run is")
            print("    mid-seed or a leg is short. It is not a finished cell.")
        print()

    # The freeze required this warning and an earlier version of this script did not emit it, which is
    # how cell 6 came to be scored across non-identical seed sets. It is printed BEFORE the table so a
    # reader cannot reach a verdict without having seen it.
    if unequal:
        print("  ** UNEQUAL-N WARNING (frozen clause: 'a partial run is not a verdict') **")
        print("    These cells' two designs do not run at identical seeds. The SCORED contrast is each")
        print("    design at its own full n, which is the frozen convention; the shared-seed contrast")
        print("    below is a robustness check and is never the verdict (Amendment 3).")
        for label, con in unequal:
            print(f"    {label}: designs share seeds {con['shared']}; confounded also has "
                  f"{con['dropped_conf'] or '[]'}, controlled also has {con['dropped_ctrl'] or '[]'}.")
            s_c, s_t = con["conf"], con["ctrl"]
            if s_c and s_t:
                flip = ((s_c["mean"] < 0) != (s_t["mean"] < 0)) != \
                       ((con["unpaired_conf"]["mean"] < 0) != (con["unpaired_ctrl"]["mean"] < 0))
                print(f"      shared-seed check: confounded {s_c['mean']:+.4f} "
                      f"[{s_c['lo']:+.4f},{s_c['hi']:+.4f}] n{s_c['n']} vs controlled "
                      f"{s_t['mean']:+.4f} [{s_t['lo']:+.4f},{s_t['hi']:+.4f}] n{s_t['n']}"
                      + ("   ** SIGN AGREEMENT DIFFERS FROM THE SCORED CONTRAST **" if flip else ""))
        print()

    print(f"=== {len(rows)}-CELL COMPARABILITY: outcome-gated vs controlled, same arm/attack/seeds ===")
    print(f"  {'cell':31s} {'confounded':>22s} {'controlled':>22s} {'dLam_a':>7s} "
          f"{'pred':>9s} {'observed':>14s}")
    for r in rows:
        f = lambda x: "--" if x is None else f"{x['mean']:+.3f}[{x['lo']:+.3f},{x['hi']:+.3f}]n{x['n']}"
        lam = "--" if r["lam"] is None else f"{r['lam']:.4f}"
        tag = "" if r["training"] else "  <= OUT OF SAMPLE"
        # `--`, not `None`: an absent value that prints as a Python literal reads like a bug in the
        # analyzer rather than a cell the rule says nothing about, and the LaTeX row already uses `---`.
        print(f"  {r['label']:31s} {f(r['conf']):>22s} {f(r['ctrl']):>22s} {lam:>7s} "
              f"{r['pred'] or '--':>9s} {r['got'] or '--':>14s}{tag}")

    tr = [r for r in rows if r["training"]]
    oos = [r for r in rows if not r["training"]]
    print(f"\n  TRAINING ({len(tr)} cells, the rule was read off these and they are NOT evidence for it):")
    print(f"    the support change is {'constant at 0.0000' if all(r['sup'] == 0 for r in tr if r['sup'] is not None) else 'NOT constant'} "
          f"across them, which is why Amendment 1 gates on dLam_a instead.")

    scored = [r for r in oos if r["pred"] and r["got"]]
    print(f"\n  OUT-OF-SAMPLE VERDICT ({len(scored)} of {len(oos)} cells scorable)")
    if not scored:
        print("    Not yet scorable: the new ladders are incomplete. No verdict is formed, and a")
        print("    partial ladder must not be quoted as one.")
        return 0

    hits = [r for r in scored if (r["got"] == r["pred"]) or
            (r["pred"] == "DISAGREE" and r["got"] == "SIGN REVERSAL")]
    misses = [r for r in scored if r not in hits]
    for r in scored:
        ok = r in hits
        print(f"    {r['label']:24s} predicted {r['pred']:9s} observed {r['got']:14s} "
              f"{'CONFIRMS' if ok else '** REFUTES'}")
    if not misses:
        print(f"\n    H-ADMISSION-GATED SURVIVES its out-of-sample test at {len(hits)}/{len(scored)}.")
        print("    The disagreement between the two designs is STRUCTURED, not universal: it appears")
        print("    where the defense admits adversarial mass and not otherwise. This is a claim about")
        print("    WHEN the confounded design misleads, and it is not a claim that the sign reversal")
        print("    replicates -- it does not, and the agreeing cells are the built-in control.")
    else:
        print(f"\n    ** H-ADMISSION-GATED IS REFUTED ** by {len(misses)} of {len(scored)} cells.")
        print("    Per the frozen clause the mechanism is withdrawn and the four-cell pattern is")
        print("    demoted from a mechanism to a description. The table is still reported in full;")
        print("    the observation survives, only the explanation is retracted. No fitted threshold")
        print("    is introduced to rescue it.")

    # Completeness is both legs present AND every frozen seed landed. The second conjunct is what makes
    # a live run look like a live run instead of like a smaller cell.
    complete = [r for r in rows if r["conf"] and r["ctrl"] and not r["missing_seeds"]]
    n_dis = sum(1 for r in rows if r["got"] in ("DISAGREE", "SIGN REVERSAL"))
    n_rev = sum(1 for r in rows if r["got"] == "SIGN REVERSAL")
    if len(complete) != len(rows):
        miss = [r["label"] for r in rows if r not in complete]
        print(f"\n  ** {len(rows) - len(complete)} CELL(S) INCOMPLETE and excluded from every count "
              f"below: {miss} **")
        for r in rows:
            if r["missing_seeds"]:
                print(f"    {r['label']}: frozen seeds still missing "
                      + ", ".join(f"{s} {v}" for s, v in sorted(r['missing_seeds'].items()))
                      + " -- no verdict is formed for this cell.")
        print("    A partial ladder is not a cell. Any count quoted from a run in progress is wrong,")
        print("    and the artifact records the incomplete labels so this cannot be read as absence.")
    print(f"\n  ACROSS ALL {len(complete)} COMPLETE CELLS: {n_dis} disagree, of which {n_rev} are "
          f"sign reversals.")
    print("    Report this as the fraction it is. 'The sign reversal replicates' would be a misreport.")

    # Rung coverage, printed for every cell that is not on the full frozen grid. The scored primary is
    # the endpoint contrast, so an endpoint-only cell is fully scored and its row above is final -- but
    # it has no interior and therefore no trend evidence, and the row cannot show that.
    # Printed UNCONDITIONALLY. The previous version printed only when some cell was off the grid, so the
    # moment the last interior rung landed this section fell silent -- and silence is exactly what the
    # stale paper sentence claimed ("the interior rungs are not run"). A coverage report that disappears
    # when coverage becomes complete cannot be used to check a sentence about coverage.
    partial = [r for r in rows if not r["rungs"]["full_grid"]]
    print(f"\n  RUNG COVERAGE: {len(rows) - len(partial)} of {len(rows)} cells are on the full frozen "
          f"grid {list(GRID)}.")
    for r in rows:
        g = r["rungs"]
        print(f"    {r['label']:31s} rungs {g['rungs_present']} on both designs; endpoint primary "
              f"{'AVAILABLE' if g['endpoint_primary_available'] else 'UNAVAILABLE'}; trend "
              f"{'available' if g['trend_secondary_available'] else 'UNAVAILABLE'}.")
    if partial:
        print("    The endpoint contrast reads only the two endpoints, so these rows are scored and")
        print("    final. Do not describe such a cell as a four-rung ladder and do not report a trend")
        print("    for it; an absent interior is not a flat interior.")

    # The trend itself. POST HOC for every cell: no trend secondary is pre-registered anywhere in
    # pre_registration_comparability.md, so this describes the ladders' shape and scores nothing. It is
    # printed because it is computable, not because any frozen rule asks for it.
    withtrend = [r for r in rows if r["trend"]]
    if withtrend:
        print(f"\n  POST HOC TREND across the four rungs ({len(withtrend)} cells). NOT PRE-REGISTERED:")
        print("    no trend secondary appears in pre_registration_comparability.md, so no verdict above")
        print("    reads these, and they must be quoted as a post hoc description of shape.")
        print("    A cell whose rungs are unequal in n prints TWO lines per design: the quoted one is")
        print("    the common seed block all four rungs share, because JT's null depends on the group")
        print("    sizes, so a mixed-n statistic is not the one the frozen grid was described with.")

        def _line(tag, t):
            pp = "--" if t["p_perm_increasing"] is None else f"{t['p_perm_increasing']:.3f}"
            return (f"      {tag:22s} means " + " ".join(f"{m:.3f}" for m in t["rung_means"])
                    + f"  n{t['n_per_rung']}  J={t['J']:.1f} z={t['z']:+.3f} "
                      f"p_up={t['p_increasing']:.3f} p_down={t['p_decreasing']:.3f} perm_up={pp}")

        for r in withtrend:
            print(f"    {r['label']}")
            for side in ("confounded", "controlled"):
                t = r["trend"][side]
                if t["equal_n"]:
                    print(_line(side, t["all_seeds"]))
                else:
                    print(_line(f"{side} QUOTED", t["common_seeds"]))
                    print(_line(f"{side} mixed-n", t["all_seeds"])
                          + "   <- NOT QUOTED: unequal rungs")
        # Which cells are mixed, named, so a reader of this block does not have to diff the n lists.
        mixed = [r["label"] for r in withtrend
                 if not (r["trend"]["confounded"]["equal_n"] and r["trend"]["controlled"]["equal_n"])]
        if mixed:
            print(f"    mixed-n rungs on {len(mixed)} cell(s) ({', '.join(mixed)}): an endpoint-only "
                  "top-up moved kappa=0 and kappa=2 only.")
        # The permutation note reads the QUOTED variant, since that is the one whose absence a reader
        # would otherwise misread as an absent effect.
        skipped = [r["label"] for r in withtrend
                   if r["trend"]["controlled"]["p_perm_increasing"] is None
                   or r["trend"]["confounded"]["p_perm_increasing"] is None]
        if skipped:
            print(f"    perm_up=-- on {len(skipped)} cell(s) ({', '.join(skipped)}): total N above "
                  f"PERM_MAX_N={PERM_MAX_N}, permutation leg SKIPPED for runtime, not absent for cause.")

    print("\n--- LaTeX rows (cell, n, confounded, controlled, dLam_a, predicted, observed) ---")
    print("%   The n column is emitted, not optional: the seven rows run at n=5, 8 and 20, the caption's")
    print("%   'each design at its own full n' does not say WHICH, and after the endpoint top-up the")
    print("%   strongest row and the weakest look identical on the page without it.")
    for r in rows:
        f = lambda x: "---" if x is None else f"${x['mean']:+.3f}$ $[{x['lo']:+.3f},{x['hi']:+.3f}]$"
        # confounded/controlled, in the column order the two estimate columns appear in. Printed as a
        # pair rather than one number because they differ on three of the seven rows.
        ns = ("---" if r["conf"] is None or r["ctrl"] is None
              else f"${r['conf']['n']}/{r['ctrl']['n']}$")
        lam = "---" if r["lam"] is None else f"${r['lam']:.4f}$"
        star = "" if r["training"] else "$^{\\ast}$"
        # No paste-ready row for an incomplete cell. The row carries neither leg's n, so a mid-run pair
        # (here confounded n=5 against controlled n=4) is INVISIBLE once it is in the table -- the
        # observed column would read `---` and every other column would look finished. Emitting a
        # commented-out placeholder keeps the cell's existence visible without shipping its numbers.
        if r["missing_seeds"]:
            print(f"% {r['label']}: MID-RUN, row withheld. Missing "
                  + "; ".join(f"{s} {v}" for s, v in sorted(r["missing_seeds"].items()))
                  + ". Re-run this script when the ladder finishes.")
            continue
        print(f"{r['label'].replace('_', chr(92)+'_')}{star} & {ns} & {f(r['conf'])} & {f(r['ctrl'])} & "
              f"{lam} & {r['pred'] or '---'} & {r['got'] or '---'} \\\\")
    print(f"\\multicolumn{{7}}{{l}}{{\\footnotesize $\\ast$ out-of-sample; the other {len(tr)} are the "
          f"cells the rule was read off.}} \\\\")

    # The figure's artifact. `assertions` carries the premises Fig. 1(c) asserts visually, so the panel
    # can refuse to draw if a re-score ever stops supporting them -- the same guard the two-bar version
    # had, widened to every cell in CELLS.
    out = {
        "description": f"{len(rows)}-cell confounded-vs-controlled comparability, scored against the "
                       f"rules frozen in {PREREG}. Each design at its own full n (Amendment 3). NOTE: "
                       f"the filename says 'six_cells' and is historical -- it is the path three "
                       f"readers already resolve, and renaming it would silently strand whichever one "
                       f"was missed. Trust n_cells, never the filename.",
        "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
        "cells": [{"label": r["label"], "training": r["training"],
                   "confounded": None if not r["conf"] else
                       {k: r["conf"][k] for k in ("mean", "lo", "hi", "n")},
                   "controlled": None if not r["ctrl"] else
                       {k: r["ctrl"][k] for k in ("mean", "lo", "hi", "n")},
                   "d_influence": r["lam"], "d_admission": r["sup"],
                   "predicted": r["pred"], "observed": r["got"],
                   "missing_frozen_seeds": r["missing_seeds"] or None,
                   "rung_coverage": r["rungs"],
                   # Present only for a topped-up cell: the published contrast recomputed on its own
                   # seed subset, so every site quoting the smaller n can quote an emitted number and
                   # the two n can never collapse into one.
                   "published_subset": pub_reproduced.get(r["label"]),
                   "legs_equal_n": (None if not (r["conf"] and r["ctrl"])
                                    else r["conf"]["n"] == r["ctrl"]["n"]),
                   # Emitted so the paper's post hoc trend sentence quotes an artifact rather than this
                   # script's stdout. Every entry carries "pre_registered": false; nothing above reads
                   # it, and no verdict in this file is a function of it.
                   "trend_post_hoc": r["trend"]} for r in rows],
        "assertions": {
            "published_cells_reproduce": True,
            # n_cells counts COMPLETE cells -- both designs present -- because that is the set Fig. 1(c)
            # draws and the number the paper counts. Fig. 1(c) filters incomplete cells out and then
            # asserts len(drawn) == n_cells, so defining n_cells as len(rows) would make the panel
            # refuse to draw for the whole time a new ladder is mid-run: a live run would look exactly
            # like a corrupt artifact. The defined and incomplete counts are carried separately so a
            # partial ladder is visible rather than absent.
            "n_cells": len(complete),
            "n_cells_defined": len(rows),
            "n_cells_incomplete": len(rows) - len(complete),
            "incomplete_cells": [r["label"] for r in rows if r not in complete],
            "n_disagree": n_dis,
            "n_sign_reversal": n_rev,
            "n_out_of_sample_scored": len(scored),
            "n_out_of_sample_refuting": len(misses),
            "mechanism_refuted": bool(misses),
            # Round 57. `sign_reversal_cell` was a SINGLE label because exactly one cell had ever been
            # one, and Fig. 1(c)'s guard compared its row list against `[that label]`. Cell 7 exists to
            # ask whether the reversal replicates, so a second one is a possible outcome -- and under
            # the old schema that outcome would have made the panel refuse to draw, i.e. the success
            # case looked identical to a corrupted artifact. Both keys are emitted: the list is
            # authoritative, the scalar is kept for any reader not yet updated and is None once there
            # is more than one, so a stale reader degrades to "no reversal" rather than to a wrong one.
            "sign_reversal_cells": [r["label"] for r in rows if r["got"] == "SIGN REVERSAL"],
            "sign_reversal_cell": (lambda L: L[0] if len(L) == 1 else None)(
                [r["label"] for r in rows if r["got"] == "SIGN REVERSAL"]),
        },
    }
    path = os.path.join(BASE, "results", "comparability_six_cells.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
