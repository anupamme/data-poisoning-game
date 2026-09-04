"""
Score the six-cell confounded-vs-controlled comparability table against the rules frozen in
`experiments/pre_registration_comparability.md` (c986ef4, Amendments 1-3, latest f16083b).

WHAT THIS DECIDES AND WHAT IT CANNOT
Each cell contrasts two estimates of the SAME quantity on the SAME arm, attack and seeds: the
outcome-gated (confounded) ladder `dose_kappa<k>`, in which the adversary is free to be attenuated,
and the controlled `doseS_kappa<k>` intervention, in which every adversary is pinned at c=1. The cell
says whether the two designs reach the same conclusion. **It does not say which is right about the
world** -- Mode S is an oracle instrument that reads adversary identity, so a cell where the designs
agree is not a licence to use the confounded design in general, only a report that here they coincide.

FOUR CELLS ARE TRAINING DATA AND ARE LABELLED AS SUCH
The frozen rule was read off the four cells that already existed, so they cannot also be evidence for
it. They are printed as TRAINING. Cells 5 and 6 are the out-of-sample test and are the only rows that
can confirm or refute anything.

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

Reads frozen artifacts and results/comparability_cells/. Writes nothing. Adds no runs.
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = "experiments/pre_registration_comparability.md"
PREREG_COMMIT = "f16083b"          # freeze c986ef4 + Amendment 1 + Amendment 2 + Amendment 3
FEMNIST_ADM = os.path.join(BASE, "results", "femnist_admission.json")

LO, HI = "0.0", "2.0"              # the ladder endpoints every existing arm is scored on
NEAR_ZERO = 0.05                   # frozen: "both point estimates within +-0.05 of zero" => AGREE

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
    ("coord_median / pixel",       "coord_median", "committed_pixel",
     [("dose_response", CONF, "")], [("dose_replication", CTRL, "")],
     True,  (ADM, "coord_median", "committed_pixel")),
    # --- out of sample ---
    ("coord_median / scaling",     "coord_median", "committed_scaling",
     [("comparability_cells", CONF, "")], [("comparability_cells", CTRL, "")],
     False, (ADM, "coord_median", "committed_scaling")),
    ("krum / scaling, EMNIST",     "krum",         "committed_scaling",
     [("comparability_cells", CONF, "|emnist")],
     [("dose_femnist", CTRL, ""), ("comparability_cells", CTRL, "|emnist")],
     False, (FEMNIST_ADM, "krum", "committed_scaling")),
]

# The four published contrasts, transcribed from the pre-registration's own table so the re-score can
# be checked against the document rather than against itself. Asserted, never printed as a result.
PUB = {"krum / scaling":       (-0.008, -0.010),
       "cos_krum / pixel":     (-0.405, -0.425),
       "reputation / scaling": (+0.762, +0.178),
       "coord_median / pixel": (-0.272, +0.098)}
PUB_TOL = 5e-3                      # the prereg prints 3 decimals


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


def _endpoint(lo, hi, seeds):
    """Paired endpoint contrast over exactly `seeds`, with a 95% t interval."""
    if len(seeds) < 2:
        return None
    d = np.array([hi[s] - lo[s] for s in seeds], dtype=float)
    n = len(d)
    m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)); hw = t_crit(n) * se
    return {"mean": m, "n": n, "lo": m - hw, "hi": m + hw, "seeds": list(seeds)}


def contrast(conf_src, ctrl_src, d2, atk):
    """Each design's endpoint contrast at its own full n (SCORED), and on the shared seeds (CHECK).

    Amendment 3. `unpaired_*` is the scored pair -- each design at its own full n, the frozen
    convention, and the only one under which the four published cells reproduce. `conf`/`ctrl` restrict
    both designs to the seeds they share and exist solely so the unequal-n warning can disclose what a
    shared-seed comparison would have said. The caller must not score on them.
    """
    cl, ch = series(conf_src, d2, atk, LO), series(conf_src, d2, atk, HI)
    tl, th = series(ctrl_src, d2, atk, LO), series(ctrl_src, d2, atk, HI)
    own_c = sorted(set(cl) & set(ch))
    own_t = sorted(set(tl) & set(th))
    shared = sorted(set(own_c) & set(own_t))
    return {"conf": _endpoint(cl, ch, shared), "ctrl": _endpoint(tl, th, shared),
            "unpaired_conf": _endpoint(cl, ch, own_c), "unpaired_ctrl": _endpoint(tl, th, own_t),
            "shared": shared, "dropped_conf": sorted(set(own_c) - set(shared)),
            "dropped_ctrl": sorted(set(own_t) - set(shared))}


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
    for (label, d2, atk, csrc, tsrc, training, admsrc) in CELLS:
        con = contrast(csrc, tsrc, d2, atk)
        # The SCORED contrast is each design at its own full n -- the frozen convention. Amendment 3
        # records why: under cross-design pairing the four published cells do not reproduce (krum
        # /scaling's controlled ladder moves -0.010 -> -0.026 when its n=20 is cut to the confounded
        # side's 5 seeds), which proves full-n is what c986ef4 scored. The shared-seed contrast is
        # printed as a disclosed robustness check and never as the verdict.
        a, b = con["unpaired_conf"], con["unpaired_ctrl"]
        if con["dropped_conf"] or con["dropped_ctrl"]:
            unequal.append((label, con))
        admpath, admarm, admatk = admsrc
        ch = channels(admpath, admarm, admatk, 2.0) if os.path.exists(admpath) else None
        lam = ch["d_influence"] if ch else None
        sup = ch["d_admission"] if ch else None
        pred = None if lam is None else ("DISAGREE" if lam > 0.0 else "AGREE")
        got = verdict(a, b)
        rows.append({"label": label, "conf": a, "ctrl": b, "lam": lam, "sup": sup,
                     "pred": pred, "got": got, "training": training, "con": con})
        if training and label in PUB and a and b:
            pc, pt = PUB[label]
            if abs(a["mean"] - pc) > PUB_TOL or abs(b["mean"] - pt) > PUB_TOL:
                problems.append(f"{label}: re-scored ({a['mean']:+.3f}, {b['mean']:+.3f}) != "
                                f"pre-registered ({pc:+.3f}, {pt:+.3f})")

    if problems:
        print("\nREFUSING TO PRINT A VERDICT -- the published contrasts do not reproduce:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("Published contrasts reproduce to the pre-registration's printed precision.\n")

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

    print("=== SIX-CELL COMPARABILITY: outcome-gated vs controlled, same arm/attack/seeds ===")
    print(f"  {'cell':24s} {'confounded':>22s} {'controlled':>22s} {'dLam_a':>7s} "
          f"{'pred':>9s} {'observed':>14s}")
    for r in rows:
        f = lambda x: "--" if x is None else f"{x['mean']:+.3f}[{x['lo']:+.3f},{x['hi']:+.3f}]n{x['n']}"
        lam = "--" if r["lam"] is None else f"{r['lam']:.4f}"
        tag = "" if r["training"] else "  <= OUT OF SAMPLE"
        print(f"  {r['label']:24s} {f(r['conf']):>22s} {f(r['ctrl']):>22s} {lam:>7s} "
              f"{str(r['pred']):>9s} {str(r['got']):>14s}{tag}")

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

    n_dis = sum(1 for r in rows if r["got"] in ("DISAGREE", "SIGN REVERSAL"))
    n_rev = sum(1 for r in rows if r["got"] == "SIGN REVERSAL")
    print(f"\n  ACROSS ALL {len(rows)} CELLS: {n_dis} disagree, of which {n_rev} are sign reversals.")
    print("    Report this as the fraction it is. 'The sign reversal replicates' would be a misreport.")

    print("\n--- LaTeX rows (cell, confounded, controlled, dLam_a, predicted, observed) ---")
    for r in rows:
        f = lambda x: "---" if x is None else f"${x['mean']:+.3f}$ $[{x['lo']:+.3f},{x['hi']:+.3f}]$"
        lam = "---" if r["lam"] is None else f"${r['lam']:.4f}$"
        star = "" if r["training"] else "$^{\\ast}$"
        print(f"{r['label'].replace('_', chr(92)+'_')}{star} & {f(r['conf'])} & {f(r['ctrl'])} & "
              f"{lam} & {r['pred'] or '---'} & {r['got'] or '---'} \\\\")
    print("\\multicolumn{6}{l}{\\footnotesize $\\ast$ out-of-sample; the other four are the cells the "
          "rule was read off.} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
