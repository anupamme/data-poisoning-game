"""Emit the 2x2 factorial's pre-registered SECONDARY measures, and one labelled sensitivity check.

run_emit_only_control.py froze and emitted the PRIMARY verdict (results/emit_only/summary.json).
It did not emit the two secondaries that Section 5 of experiments/pre_registration_emit_only.md
names -- "the per-seed sign count, and the same Jonckheere-Terpstra monotone-trend test across kappa
the score-only arm reports" -- so they are computed here, from the frozen artifacts only.

This script is READ-ONLY on results/. It re-derives the primary verdict from the per-seed cells and
asserts it reproduces the frozen one, so the secondaries cannot be read off a different computation
than the verdict they qualify.

The final block is a POST-HOC sensitivity check and is labelled as such at every print. The
pre-registration fixes n=5 on seeds 42-46 ("No seed addition") and applies the 0.35 accuracy floor to
a rung's MEAN clean accuracy, not per seed. Dropping a seed is therefore NOT licensed by the frozen
rules and does NOT override the verdict. It is computed and printed because one seed turns out to
carry the whole result, and a reader who is not told that would be misled.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_dose_response import jonckheere  # noqa: E402

SUMMARY = "results/emit_only/summary.json"
FULL = "results/targeted_dose/summary.json"
SCORE = "results/score_only/summary.json"
KEY = "doseS_kappa{k}_then_krum|committed_scaling"
KAPPAS = [0.0, 0.5, 1.0, 2.0]
SEEDS = [42, 43, 44, 45, 46]


def load_cells(path):
    d = json.load(open(path))
    return d.get("cells", d)


def per_seed(cells, kappa):
    """{seed: (asr, acc)} for one rung, or None if the artifact has no such cell."""
    c = cells.get(KEY.format(k=kappa))
    if c is None:
        return None
    return {p["seed"]: (p["asr"], p["accuracy"]) for p in c["per_seed"]}


def deltas(cells, rung, seeds):
    """Seed-matched dASR from the kappa=0 baseline to `rung`, in `seeds` order."""
    base, top = per_seed(cells, 0.0), per_seed(cells, rung)
    return np.array([top[s][0] - base[s][0] for s in seeds])


def main():
    frozen = json.load(open(SUMMARY))
    v = frozen["verdict"]
    rung = v["primary_rung"]
    eo, full, score = load_cells(SUMMARY), load_cells(FULL), load_cells(SCORE)

    print("=== 2x2 FACTORIAL: pre-registered secondaries (primary verdict is frozen) ===")
    print(f"    prereg {frozen['prereg_commit']}; primary rung kappa={rung}"
          f"{'  [FORCED by the accuracy floor]' if v['primary_rung_forced'] else ''}")

    # ---- fidelity: the secondaries must qualify the same computation as the verdict ----
    d_eo, d_full, d_so = (deltas(c, rung, SEEDS) for c in (eo, full, score))
    resid = d_full.mean() - (d_so.mean() + d_eo.mean())
    for name, got, want in [("d_asr_emit_only", d_eo.mean(), v["d_asr_emit_only"]),
                            ("d_asr_full_mode_s", d_full.mean(), v["d_asr_full_mode_s"]),
                            ("d_asr_score_only", d_so.mean(), v["d_asr_score_only"]),
                            ("additivity_residual", resid, v["additivity_residual"])]:
        assert abs(got - want) < 1e-9, f"{name}: recomputed {got!r} != frozen {want!r}"
    print(f"    fidelity: all four frozen quantities reproduce from per-seed cells (<1e-9)")

    # ---- pre-registered secondary 1: per-seed sign count ----
    print("\n=== SECONDARY 1 (frozen): per-seed sign count ===")
    print("    dASR per seed, kappa=0 -> kappa={:g}, all three cells of the factorial".format(rung))
    print(f"    {'seed':>6} {'emit-only':>12} {'score-only':>12} {'full Mode S':>12}"
          f"  {'EO acc':>8}")
    eo_acc = per_seed(eo, rung)
    for i, s in enumerate(SEEDS):
        flag = "  * below 0.35 floor" if eo_acc[s][1] < 0.35 else ""
        print(f"    {s:>6} {d_eo[i]:>+12.6f} {d_so[i]:>+12.6f} {d_full[i]:>+12.6f}"
              f"  {eo_acc[s][1]:>8.4f}{flag}")
    for name, d in [("emit-only", d_eo), ("score-only", d_so), ("full Mode S", d_full)]:
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        print(f"    {name:>12}: {pos} up, {neg} down, {len(d) - pos - neg} exactly zero"
              f"   (mean {d.mean():+.6f})")

    # ---- pre-registered secondary 2: Jonckheere-Terpstra across kappa ----
    print("\n=== SECONDARY 2 (frozen): Jonckheere-Terpstra monotone trend across kappa ===")
    groups, labels = [], []
    for k in KAPPAS:
        ps = per_seed(eo, k)
        groups.append([ps[s][0] for s in SEEDS])
        labels.append(f"kappa={k:g}")
    _, z, p_inc, p_dec, p_perm = jonckheere(groups)
    print(f"    ordered rungs: {', '.join(labels)}  (n=5 each, emit-only ASR)")
    print(f"    z={z:+.3f}  p_increasing={p_inc:.4g}  p_decreasing={p_dec:.4g}"
          f"  p_perm_increasing={p_perm:.4g}")
    floor = [f"{k:g}" for k in KAPPAS
             if np.mean([per_seed(eo, k)[s][1] for s in SEEDS]) < 0.35]
    if floor:
        print(f"    NOTE the trend spans rung(s) {', '.join(floor)} whose MEAN clean accuracy is"
              f" below 0.35, which the")
        print(f"         pre-registration calls uninformative. The test is reported over all four"
              f" rungs as frozen,")
        print(f"         but a trend that runs through a collapsed rung is not evidence about"
              f" suppression.")

    # ---- POST-HOC sensitivity, not part of the frozen rule ----
    print("\n=== POST-HOC SENSITIVITY -- NOT PRE-REGISTERED, DOES NOT OVERRIDE THE VERDICT ===")
    below = [s for s in SEEDS if eo_acc[s][1] < 0.35]
    if not below:
        print("    No emit-only seed at the primary rung is below the floor; nothing to check.")
    else:
        print(f"    The pre-registration applies the 0.35 floor to a rung's MEAN accuracy and fixes")
        print(f"    n=5 on seeds 42-46, so no per-seed exclusion is licensed. Seed(s) {below} sit")
        print(f"    below the floor individually at kappa={rung:g} while the rung mean"
              f" ({np.mean([eo_acc[s][1] for s in SEEDS]):.4f}) clears it.")
        keep = [s for s in SEEDS if s not in below]
        k_eo, k_full, k_so = (deltas(c, rung, keep) for c in (eo, full, score))
        k_resid = k_full.mean() - (k_so.mean() + k_eo.mean())
        print(f"\n    {'':<22}{'as frozen (n=5)':>18}{'excl. ' + str(below) + f' (n={len(keep)})':>22}")
        for name, a, b in [("dASR emit-only", d_eo.mean(), k_eo.mean()),
                           ("dASR score-only", d_so.mean(), k_so.mean()),
                           ("dASR full Mode S", d_full.mean(), k_full.mean()),
                           ("additivity residual", resid, k_resid)]:
            print(f"    {name:<22}{a:>+18.6f}{b:>+22.6f}")
        print(f"\n    verdict under each:")
        for tag, de, rs in [("as frozen (n=5)", d_eo.mean(), resid),
                            (f"excl. {below}", k_eo.mean(), k_resid)]:
            inert = abs(de) < v["inert_margin"]
            sep = abs(rs) < v["additivity_margin"]
            print(f"      {tag:<18} magnitude channel {'INERT' if inert else 'ACTIVE'}"
                  f" (|{de:+.4f}| vs {v['inert_margin']}),"
                  f" channels {'SEPARABLE' if sep else 'INTERACT'}"
                  f" (|{rs:+.4f}| vs {v['additivity_margin']})")
        print("\n    Read this as a statement about POWER, not as a corrected result: prereg"
              " Section 6.1 declared")
        print("    before the run that at n=5 only a large |dASR_EO| would be decisive, and"
              " |dASR_EO| is not large.")

    # ---- LaTeX table rows, so no number in the paper is hand-typed ----
    print("\n=== LATEX TABLE ROWS (paste target: tab:factorial) ===")
    base = np.mean([per_seed(eo, 0.0)[s][0] for s in SEEDS])
    rows = [("$\\kappa{=}0$ identity (imported)", "closed", "closed", base, None),
            ("Score-only", "\\textbf{open}", "closed",
             np.mean([per_seed(score, rung)[s][0] for s in SEEDS]), d_so.mean()),
            ("Emit-only", "closed", "\\textbf{open}",
             np.mean([per_seed(eo, rung)[s][0] for s in SEEDS]), d_eo.mean()),
            ("Full Mode~S", "\\textbf{open}", "\\textbf{open}",
             np.mean([per_seed(full, rung)[s][0] for s in SEEDS]), d_full.mean())]
    for name, sc, mg, asr, d in rows:
        dcell = "---" if d is None else f"${d:+.3f}$"
        print(f"{name} & {sc} & {mg} & ${asr:.3f}$ & {dcell} \\\\")
    print("\\midrule")
    print("\\multicolumn{4}{l}{additive prediction $\\Delta_{\\mathrm{SO}}"
          f"+\\Delta_{{\\mathrm{{EO}}}}$}} & ${d_so.mean() + d_eo.mean():+.3f}$ \\\\")
    print("\\multicolumn{4}{l}{residual against measured $\\Delta_{\\mathrm{full}}$"
          f" (frozen margin ${v['additivity_margin']}$)}} & $\\mathbf{{{resid:+.3f}}}$ \\\\")

    print(f"\n=== FROZEN PRIMARY VERDICT (unchanged, from {SUMMARY}) ===")
    print(f"    {v['verdict']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
