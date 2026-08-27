"""The confounded ladder vs the Mode-S instrument, side by side, on the ONE arm that has both.

WHY THIS EXISTS. The paper reports a sign reversal -- the same aggregator, the same attack, the same
seeds, the same nominal dose, and an ASR effect of -0.272 under the confounded ladder against +0.098
under the Mode-S instrument -- and asks the reader to attribute the difference to the adversarial
coefficient channel. Until now that argument was spread across two tables and a paragraph, so a reader
could not check the one thing that makes it an argument rather than an assertion: that the two designs
are matched on everything else. This script puts the two columns next to each other, recomputes every
cell from per-seed and per-round rows, and asserts the shared-origin identity.

WHAT THE COMPARISON RESTS ON, and what it does not:

  matched      the aggregator (coord_median), the attack (committed_pixel), the seeds (42-46), the
               nominal dose (kappa 0 -> 2, rho 1 -> 54.6), the identity rung ITSELF (the two designs
               share their kappa=0 cell numerically, asserted below to 1e-12), and -- at the top rung --
               the displacement of the emitted aggregate, the decision-change rate, and the mean clean
               accuracy. Matched accuracy is what rules out "the ladder's fall is an accuracy collapse".
  differing    the adversarial coefficient channel: the ladder lets the adversaries' coefficients be
               drawn from the same widening spread as everyone's, so their share of the coefficient mass
               falls from 0.267 to 0.187 and Delta_c drifts to -0.336; Mode S pins c_adv = 1.0 exactly,
               holding the share constant and Delta_c at 0.
  NOT matched  aggregate adversarial influence Lambda_a moves in BOTH designs, 0.081 in the ladder
               against 0.033 in the instrument. Neither column is a clean statistic-only intervention:
               what Mode S pins exactly is the adversarial COEFFICIENT share, and for a coordinate-wise
               order statistic that does not force Lambda_a to be constant, because which client attains
               the median per coordinate still moves. The ladder's movement is 2.4x the instrument's and
               in the same direction. This is disclosed in the table itself rather than in a footnote,
               and it is why the contrast is stated as "differ chiefly in the coefficient channel"
               rather than "differ only in it".

               VOCABULARY, so this table cannot contradict the channel table. The frozen field
               results/admission_measurement.json '<family>|coord_median|<rung>|admission' is the change
               in coord_median_adv_argmedian_frac -- the adversarial share of selected coordinates --
               which is exactly build_channel_table.py's Delta Lambda_a column (0.033 in both scripts),
               NOT its Delta adm. column. That column counts rounds in which the SUPPORT of the
               adversarial mass changes -- whether any adversarial input enters at all -- and reads
               0.000 for every row. So this row is labelled Delta Lambda_a: what moves is how much
               weight the adversary's direction carries, not whether it enters. For a selector the mass
               is binary and the two coincide.

This is a two-column contrast on one arm, not a controlled experiment over many. It licenses the claim
that the adversarial coefficient channel is where the two designs differ; it does not by itself measure
that channel's effect size, and no p-value is attached to the difference between the two columns.

SOURCES, all read-only:
  results/dose_response/summary.json      ASR for the confounded ladder (Round 11)
  results/dose_replication/summary.json   ASR for the Mode-S arm (Round 15)
  results/admission_measurement.json      aggregate displacement / decision / admission, both families
  results/coefficient_targeting.json       Delta_c and the adversarial coefficient share, both families

Output: results/comparability_table.json plus LaTeX on stdout.
Run: python3 experiments/build_comparability_table.py
"""
import json, os, sys
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# The same trend test the frozen scorers use, so this table and the papers cannot disagree about
# the quoted p-values: the published p_down = 0.0015 and p_up = 0.069 are Jonckheere-Terpstra over all
# FOUR rungs, not two-rung t-tests. The two-rung Welch statistic is computed too, but it is recorded in
# the JSON as a post-hoc diagnostic and deliberately kept out of the LaTeX so no new number enters the
# papers.
from experiments.analyze_dose_response import jonckheere  # noqa: E402
from experiments.analyze_targeted_dose import welch_less  # noqa: E402

R = os.path.join(base, "results")
LADDER = os.path.join(R, "dose_response", "summary.json")
MODES = os.path.join(R, "dose_replication", "summary.json")
ADM = os.path.join(R, "admission_measurement.json")
COEF = os.path.join(R, "coefficient_targeting.json")
OUT = os.path.join(R, "comparability_table.json")

ARM, ATTACK = "coord_median", "committed_pixel"
KAPPAS = [0.0, 0.5, 1.0, 2.0]
LO, HI = KAPPAS[0], KAPPAS[-1]
SHARED_TOL = 1e-12      # the two designs must share their identity cell NUMERICALLY, not approximately

# (column label, family prefix in the channel files, ASR source, the d1 name pattern)
DESIGNS = [("Confounded ladder", "dose", LADDER, "dose_kappa{}"),
           ("Instrument (Mode S)", "doseS", MODES, "doseS_kappa{}")]


def asr_rungs(path, pattern):
    """{kappa: (per-seed asr list, per-seed acc list, seeds)} recomputed from per-seed rows."""
    if not os.path.exists(path):
        return {}
    cells = json.load(open(path))["cells"]
    out = {}
    for k in KAPPAS:
        c = cells.get(f"{pattern.format(k)}_then_{ARM}|{ATTACK}")
        if c is None:
            continue
        rows = sorted(c["per_seed"], key=lambda r: r["seed"])
        out[k] = ([r["asr"] for r in rows], [r["accuracy"] for r in rows],
                  [r["seed"] for r in rows])
    return out


def channel(family, kappa, field):
    if not os.path.exists(ADM):
        return None
    return json.load(open(ADM))["summary"].get(f"{family}|{ARM}|{kappa}|{field}")


def coeff(family, kappa, field):
    if not os.path.exists(COEF):
        return None
    return json.load(open(COEF))["summary"].get(f"{family}|{kappa}|{field}")


def share(family, kappa):
    if not os.path.exists(ADM):
        return None
    return json.load(open(ADM))["adv_coeff_share"].get(f"{family}|{kappa}")


def fmt(v, p=3, signed=False):
    if v is None:
        return "---"
    if abs(v) and abs(v) < 10 ** -(p + 1):
        return f"{v:.1e}"
    return f"{v:+.{p}f}" if signed else f"{v:.{p}f}"


def tex_exp(v, p=0):
    """A float in scientific notation as LaTeX math, so the emitted table needs no hand-editing."""
    m, e = f"{v:.{p}e}".split("e")
    return rf"{m}\mathrm{{e}}{{-}}{abs(int(e)):02d}" if int(e) < 0 else \
        rf"{m}\mathrm{{e}}{{+}}{int(e):02d}"


def main():
    for p in (LADDER, MODES, ADM, COEF):
        if not os.path.exists(p):
            sys.exit(f"missing {os.path.relpath(p, base)} -- this table only assembles frozen "
                     "artifacts and computes nothing new")

    cols = []
    for label, family, src, pattern in DESIGNS:
        a = asr_rungs(src, pattern)
        if LO not in a or HI not in a:
            sys.exit(f"{label}: the {ARM}/{ATTACK} arm is missing its identity or top rung")
        lo_asr, lo_acc, seeds = a[LO]
        hi_asr, hi_acc, seeds_hi = a[HI]
        if seeds != seeds_hi:
            sys.exit(f"{label}: rung seeds differ ({seeds} vs {seeds_hi}); the arm is not seed-matched")
        t, p_welch_down = welch_less(hi_asr, lo_asr)   # post-hoc two-rung diagnostic, JSON only
        # The published trend test: JT across all four rungs, in the frozen rung order.
        J, z, p_up, p_down, p_perm_up = jonckheere([a[k][0] for k in KAPPAS if k in a])
        cols.append({
            "label": label, "family": family, "arm": ARM, "attack": ATTACK, "seeds": seeds,
            "asr_source": os.path.relpath(src, base),
            "kappa_range": [LO, HI], "rho_range": [float(np.exp(2 * LO)), float(np.exp(2 * HI))],
            "asr_identity": float(np.mean(lo_asr)), "asr_top": float(np.mean(hi_asr)),
            "asr_identity_per_seed": lo_asr, "asr_top_per_seed": hi_asr,
            "asr_effect": float(np.mean(hi_asr) - np.mean(lo_asr)),
            "acc_identity": float(np.mean(lo_acc)), "acc_top": float(np.mean(hi_acc)),
            "jt_J": J, "jt_z": z, "jt_p_rise": p_up, "jt_p_fall": p_down,
            "jt_p_rise_perm": p_perm_up,
            "welch_t_two_rung": t, "welch_p_fall_two_rung": p_welch_down,
            "welch_note": "post-hoc two-rung diagnostic; the papers quote the JT p-values",
            "agg_disp_top": channel(family, HI, "agg_disp"),
            "decision_top": channel(family, HI, "decision"),
            "admission_top": channel(family, HI, "admission"),
            "share_identity": share(family, LO), "share_top": share(family, HI),
            "delta_c_identity": coeff(family, LO, "delta_c"),
            "delta_c_top": coeff(family, HI, "delta_c"),
            "all_rungs": {str(k): {"asr": float(np.mean(a[k][0])), "acc": float(np.mean(a[k][1])),
                                   "agg_disp": channel(family, k, "agg_disp"),
                                   "decision": channel(family, k, "decision"),
                                   "admission": channel(family, k, "admission"),
                                   "share": share(family, k),
                                   "delta_c": coeff(family, k, "delta_c")}
                          for k in KAPPAS if k in a},
        })

    L, S = cols[0], cols[1]

    print("=== THE CONFOUNDED LADDER vs THE MODE-S INSTRUMENT "
          f"({ARM}, {ATTACK.replace('committed_', '')}, seeds {L['seeds']}) ===")
    print("    Assembled read-only from frozen artifacts. Nothing is trained and no ASR is "
          "recomputed here.\n")
    w = 26
    rows = [
        ("downstream $d_2$", ARM.replace("_", " "), ARM.replace("_", " ")),
        ("attack", ATTACK.replace("committed_", ""), ATTACK.replace("committed_", "")),
        ("seeds", str(L["seeds"]), str(S["seeds"])),
        ("nominal dose", f"kappa {LO}->{HI}, rho 1->{L['rho_range'][1]:.1f}",
         f"kappa {LO}->{HI}, rho 1->{S['rho_range'][1]:.1f}"),
        ("benign coefficients", "mean 1 over ALL clients", "mean 1 over BENIGN clients"),
        ("adversarial coefficients", "same spread (attenuated)", "pinned c=1.0 exactly"),
        ("adv. coefficient share", f"{fmt(L['share_identity'])} -> {fmt(L['share_top'])}",
         f"{fmt(S['share_identity'], 6)} -> {fmt(S['share_top'], 6)}"),
        ("Delta_c", f"{fmt(L['delta_c_identity'], signed=True)} -> "
                    f"{fmt(L['delta_c_top'], signed=True)}",
         f"{fmt(S['delta_c_identity'], signed=True)} -> {fmt(S['delta_c_top'], signed=True)}"),
        ("aggregate displacement", f"0.000 -> {fmt(L['agg_disp_top'])}",
         f"0.000 -> {fmt(S['agg_disp_top'])}"),
        ("decision change", f"0.000 -> {fmt(L['decision_top'])}",
         f"0.000 -> {fmt(S['decision_top'])}"),
        ("adversarial influence dLambda_a", f"0.000 -> {fmt(L['admission_top'])}",
         f"0.000 -> {fmt(S['admission_top'])}"),
        ("mean clean accuracy", f"{fmt(L['acc_identity'])} -> {fmt(L['acc_top'])}",
         f"{fmt(S['acc_identity'])} -> {fmt(S['acc_top'])}"),
        ("ASR", f"{fmt(L['asr_identity'])} -> {fmt(L['asr_top'])}",
         f"{fmt(S['asr_identity'])} -> {fmt(S['asr_top'])}"),
        ("ASR EFFECT", f"{fmt(L['asr_effect'], signed=True)}  (JT p_fall={L['jt_p_fall']:.4f})",
         f"{fmt(S['asr_effect'], signed=True)}  (JT p_rise={S['jt_p_rise']:.3f})"),
        ("JT trend z", f"{L['jt_z']:+.3f}", f"{S['jt_z']:+.3f}"),
    ]
    print(f"  {'':26s} {L['label']:>30s} {S['label']:>30s}")
    print("  " + "-" * 88)
    for name, a, b in rows:
        print(f"  {name:26s} {a:>30s} {b:>30s}")

    print("\n=== ASSERTIONS ===")
    same_seeds = L["seeds"] == S["seeds"]
    gap = abs(L["asr_identity"] - S["asr_identity"])
    per_seed_gap = max(abs(x - y) for x, y in zip(L["asr_identity_per_seed"],
                                                 S["asr_identity_per_seed"])) if same_seeds else None
    print(f"  seed sets identical: {same_seeds} ({L['seeds']})")
    print(f"  the two designs share their identity cell: mean gap {gap:.2e}"
          + (f", worst per-seed gap {per_seed_gap:.2e}" if per_seed_gap is not None else "")
          + f"  (tolerance {SHARED_TOL:g})  {'OK' if gap < SHARED_TOL else '<- NOT SHARED'}")
    ok_shared = gap < SHARED_TOL and (per_seed_gap is None or per_seed_gap < SHARED_TOL)
    if not ok_shared:
        print("  ! The identity cell is NOT shared, so the two columns do not start from the same "
              "point and the contrast as stated does not hold.")
    for nm, key, tol in (("aggregate displacement", "agg_disp_top", 0.10),
                         ("decision change", "decision_top", 0.10),
                         ("mean clean accuracy", "acc_top", 0.05)):
        d = abs((L[key] or 0) - (S[key] or 0))
        print(f"  matched at the top rung on {nm:24s}: |{fmt(L[key])} - {fmt(S[key])}| = {d:.3f} "
              f"{'(matched)' if d <= tol else f'(NOT matched within {tol})'}")
    d_share = abs((L["share_top"] or 0) - (S["share_top"] or 0))
    d_dc = abs((L["delta_c_top"] or 0) - (S["delta_c_top"] or 0))
    print(f"  differing on the adversarial coefficient channel: share gap {d_share:.3f}, "
          f"Delta_c gap {d_dc:.3f}")
    print(f"  trend tests (Jonckheere-Terpstra over all four rungs, the quantity the papers "
          f"quote):\n    ladder z={L['jt_z']:+.3f}, p_fall={L['jt_p_fall']:.6f}; "
          f"instrument z={S['jt_z']:+.3f}, p_rise={S['jt_p_rise']:.4f}")
    print(f"    post-hoc two-rung Welch (JSON only, not in the papers): ladder "
          f"p_fall={L['welch_p_fall_two_rung']:.4f}")
    print(f"  sign reversal: ladder {fmt(L['asr_effect'], signed=True)} vs instrument "
          f"{fmt(S['asr_effect'], signed=True)} -- "
          f"{'opposite signs' if L['asr_effect'] * S['asr_effect'] < 0 else 'SAME SIGN'}")
    print(f"  NON-MATCH, disclosed: adversarial influence Lambda_a moves in BOTH designs -- ladder "
          f"{fmt(L['admission_top'])} vs instrument {fmt(S['admission_top'])} "
          f"({L['admission_top'] / max(S['admission_top'], 1e-12):.1f}x). Mode S pins the "
          f"coefficient share\n    exactly, which for a coordinate-wise order statistic does not "
          "force admitted mass constant. Neither column is a clean statistic-only intervention.")

    print("\n=== BOTH FULL LADDERS, for the appendix ===")
    for c in cols:
        print(f"\n  -- {c['label']} ({c['family']}) --")
        print(f"  {'kappa':>6} {'ASR':>7} {'acc':>7} {'aggDisp':>8} {'dDec':>7} {'dAdm':>7} "
              f"{'share':>9} {'Delta_c':>9}")
        for k in KAPPAS:
            r = c["all_rungs"].get(str(k))
            if r is None:
                continue
            print(f"  {k:>6} {r['asr']:7.3f} {r['acc']:7.3f} {fmt(r['agg_disp']):>8} "
                  f"{fmt(r['decision']):>7} {fmt(r['admission']):>7} {fmt(r['share'], 6):>9} "
                  f"{fmt(r['delta_c'], signed=True):>9}")

    # LaTeX. Emitted here so the paper's table cannot drift from the numbers above.
    print("\n=== LATEX (copy verbatim; regenerate rather than edit) ===\n")
    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r"& Confounded ladder & Instrument (Mode~S) \\",
           r"& \texttt{dose\_kappa} & \texttt{doseS\_kappa} \\", r"\midrule",
           rf"downstream $d_2$ & \texttt{{coord\_median}} & \texttt{{coord\_median}} \\",
           rf"attack & committed pixel & committed pixel \\",
           rf"seeds & {L['seeds'][0]}--{L['seeds'][-1]} & {S['seeds'][0]}--{S['seeds'][-1]} \\",
           rf"nominal dose & $\kappa\,{LO}\!\to\!{HI}$, $\rho\,1\!\to\!{L['rho_range'][1]:.1f}$ "
           rf"& $\kappa\,{LO}\!\to\!{HI}$, $\rho\,1\!\to\!{S['rho_range'][1]:.1f}$ \\",
           r"\midrule",
           r"benign coefficients & mean $1$ over \emph{all} clients "
           r"& mean $1$ over \emph{benign} clients \\",
           r"adversarial coefficients & drawn from the same spread & pinned $c=1.0$ exactly \\",
           rf"adv.\ coefficient share & ${fmt(L['share_identity'])} \to "
           rf"\mathbf{{{fmt(L['share_top'])}}}$ & ${fmt(S['share_identity'], 6)}$, constant \\",
           rf"$\Delta_c$ & ${fmt(L['delta_c_identity'], signed=True)} \to "
           rf"\mathbf{{{fmt(L['delta_c_top'], signed=True)}}}$ & $0$ by construction "
           rf"(read back ${tex_exp(abs(S['delta_c_top']))}$) \\",
           r"\midrule",
           rf"aggregate displacement & $0.000 \to {fmt(L['agg_disp_top'])}$ "
           rf"& $0.000 \to {fmt(S['agg_disp_top'])}$ \\",
           rf"decision change & $0.000 \to {fmt(L['decision_top'])}$ "
           rf"& $0.000 \to {fmt(S['decision_top'])}$ \\",
           rf"adversarial influence $\Delta\Lambda_a$ & $0.000 \to \mathbf{{{fmt(L['admission_top'])}}}$ "
           rf"& $0.000 \to \mathbf{{{fmt(S['admission_top'])}}}$ \\",
           rf"mean clean accuracy & ${fmt(L['acc_identity'])} \to {fmt(L['acc_top'])}$ "
           rf"& ${fmt(S['acc_identity'])} \to {fmt(S['acc_top'])}$ \\",
           rf"ASR & ${fmt(L['asr_identity'])} \to {fmt(L['asr_top'])}$ "
           rf"& ${fmt(S['asr_identity'])} \to {fmt(S['asr_top'])}$ \\",
           r"\midrule",
           rf"\textbf{{ASR effect}} & $\mathbf{{{fmt(L['asr_effect'], signed=True)}}}$ "
           rf"($p_{{\downarrow}}={L['jt_p_fall']:.4f}$) "
           rf"& $\mathbf{{{fmt(S['asr_effect'], signed=True)}}}$ "
           rf"($p_{{\uparrow}}={S['jt_p_rise']:.3f}$) \\",
           r"\bottomrule", r"\end{tabular}"]
    print("\n".join(tex))

    json.dump({"description": "The confounded ladder vs the Mode-S instrument on the one arm that has "
                              "both. Assembled read-only from frozen artifacts; nothing is trained.",
               "arm": ARM, "attack": ATTACK, "kappas": KAPPAS,
               "sources": [os.path.relpath(p, base) for p in (LADDER, MODES, ADM, COEF)],
               "columns": cols,
               "assertions": {"seed_sets_identical": same_seeds,
                              "identity_cell_mean_gap": gap,
                              "identity_cell_worst_per_seed_gap": per_seed_gap,
                              "shared_tolerance": SHARED_TOL,
                              "identity_cell_shared": ok_shared,
                              "sign_reversal": bool(L["asr_effect"] * S["asr_effect"] < 0),
                              "top_rung_gaps": {
                                  "agg_disp": abs((L["agg_disp_top"] or 0) - (S["agg_disp_top"] or 0)),
                                  "decision": abs((L["decision_top"] or 0) - (S["decision_top"] or 0)),
                                  "clean_accuracy": abs(L["acc_top"] - S["acc_top"]),
                                  "adv_coeff_share": d_share, "delta_c": d_dc}},
               "disclosed_non_match": (
                   f"aggregate adversarial influence Lambda_a moves in both designs: "
                   f"{L['admission_top']:.3f} in the ladder against {S['admission_top']:.3f} in the "
                   "instrument. Mode S pins the adversarial COEFFICIENT share exactly; for a "
                   "coordinate-wise order statistic that does not force Lambda_a to be constant. "
                   "Neither column is a clean statistic-only intervention, so the contrast is 'differ "
                   "chiefly in the coefficient channel', not 'only in it'. This is the same quantity as "
                   "build_channel_table.py's Delta Lambda_a column, not its Delta adm. column (support "
                   "of the adversarial mass), which reads 0.000 for every row."),
               "scope": ("A two-column contrast on one arm. It licenses the claim that the "
                         "adversarial coefficient channel is where the two designs differ; it does "
                         "not measure that channel's effect size, and no p-value is attached to the "
                         "difference between the two columns."),
               "latex": "\n".join(tex)}, open(OUT, "w"), indent=1)
    print(f"\nWrote {OUT}")
    return 0 if ok_shared else 1


if __name__ == "__main__":
    sys.exit(main())
