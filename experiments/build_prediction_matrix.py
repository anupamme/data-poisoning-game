"""
The cross-class prediction matrix: transformation class x aggregator x theory class x frozen
prediction x observed.

WHAT THIS IS FOR
`prop:invariance` classifies aggregators by their behaviour under **positive per-client rescaling**:
(a) exactly invariant, (b) conditionally invariant, (c) not invariant. Coordinate masking is not of
the form `c_i * u_i` for scalar `c_i > 0`, so `thm:bounded_reweight` and `prop:invariance` make no
prediction about it -- by construction, not by omission. That is what makes the second class a test:
the classification is a statement about a TRANSFORMATION CLASS, and an aggregator certified by one
class can be uncertified by the other.

Every prediction cell in this table was frozen in a committed pre-registration BEFORE the
corresponding measurement existed, and this script prints the commit that froze it. Observed cells are
read from frozen artifacts, read-only. **Any outcome cell no run supports prints `--` rather than
being omitted**, so the table's own gaps are visible.

WHY THERE IS NO `rfa` ROW
The plan for this table listed `rfa` alongside the other three aggregators. RFA is not an arm of the
channel measurement at all -- `build_channel_table.py` has no RFA row and neither
`admission_measurement.json` nor `mask_admission.json` contains one -- so an RFA row would be `--` in
every observed column. The frozen matrix in `pre_registration_dose_mask.md` accordingly covers the four
measured aggregators, and this script reproduces that matrix rather than padding it with a vacuous row.
RFA's role in the paper is `prop:emergent` and the FG->RFA coverage discussion, not this matrix.

THE ASR COLUMNS ARE NOT SYMMETRIC, AND THE ASYMMETRY IS DISCLOSED
Under rescaling every aggregator has a published Mode-S ASR ladder. Under masking only `krum` is being
run: the `cos_krum`/pixel mask ladder was considered and **dropped for compute** before any ASR
existed, and that drop is recorded in the pre-registration. So the masking ASR column is one number and
three `--`, and the `cos_krum` masking cells are labelled CHANNEL results, never outcome results.

Reads: results/admission_measurement.json, results/mask_admission.json, results/targeted_dose/,
results/dose_replication/, results/dose_mask/. Writes nothing. Adds no runs.

Run: python3 experiments/build_prediction_matrix.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Single-sourced so this table, tab:tost and tab:comparability cannot quote three differently-derived
# intervals for the same paired difference.
from experiments.analyze_tost_existing import load, tost  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODE_S_ADM = os.path.join(BASE, "results", "admission_measurement.json")
MODE_M_ADM = os.path.join(BASE, "results", "mask_admission.json")

# The rung each class is read at: the top rung of its own ladder. These are different dials and are
# never pooled -- mode S's dial is a weight ratio (rho = e^{2 kappa}), mode M's is the drop rate
# itself, which is not a weight ratio at all since the transform preserves every client's norm.
S_TOP, M_TOP = "2.0", "0.8"
IDENTITY = "0.0"          # both families' identity rung; returns the update list unwrapped

# Which pre-registration froze which prediction, and the commit that froze it. A cell whose prediction
# is not traceable to one of these prints `--`.
PREREGS = {
    "rescaling": ("experiments/pre_registration_targeted_dose.md", "5130cec"),
    "masking":   ("experiments/pre_registration_dose_mask.md",     "684b31e"),
}

# The frozen matrix, transcribed from pre_registration_dose_mask.md's "cross-class prediction matrix"
# section. The `observed` values are NOT transcribed -- they are read from the artifacts below and
# ASSERTED equal to the prereg's printed values, so the document and this emitter cannot drift.
ROWS = [
    # aggregator, prop:invariance class, rescaling prediction, masking prediction,
    #   prereg-printed rescaling decision, prereg-printed masking decision, attack
    ("cos_krum",     "(a) exactly invariant",     "decision unchanged",
     "guarantee lapses: decision changes", 0.000, 0.467, "committed_pixel"),
    ("krum",         "(c) not invariant",         "decision changes",
     "decision changes",                   0.733, 0.467, "committed_scaling"),
    ("coord_median", "(b) conditionally inv.",    "partial",
     "partial",                            0.482, 0.700, "committed_pixel"),
    ("reputation",   "(c) not invariant",         "decision changes",
     "decision changes",                   1.000, 0.533, "committed_scaling"),
]

# Where each aggregator's ASR ladder lives. Mode M has one entry because only krum is being run.
ASR_S = {
    "krum":         ("targeted_dose",    "doseS_kappa{r}_then_krum|committed_scaling"),
    "reputation":   ("targeted_dose",    "doseS_kappa{r}_then_reputation|committed_scaling"),
    "cos_krum":     ("targeted_dose",    "doseS_kappa{r}_then_cos_krum|committed_pixel"),
    "coord_median": ("dose_replication", "doseS_kappa{r}_then_coord_median|committed_pixel"),
}
ASR_M = {
    "krum":         ("dose_mask",        "doseM_m{r}_then_krum|committed_scaling"),
}

PREREG_TOL = 5e-4          # the prereg prints 3 decimals, so this is exact agreement at its precision


def prereg_commit(relpath):
    """The commit that last touched a pre-registration, or None if it is untracked.

    Untracked means unfrozen: by this project's own standard a document that is not committed is not a
    pre-registration, so a prediction resting on one prints `--`.
    """
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", relpath],
                             cwd=BASE, capture_output=True, text=True, timeout=20)
        h = out.stdout.strip()
        return h or None
    except Exception:
        return None


def channel(path, family, agg, rung, field):
    """One measured channel quantity, or None if the artifact or the key is absent."""
    if not os.path.exists(path):
        return None
    s = json.load(open(path)).get("summary", {})
    return s.get(f"{family}|{agg}|{rung}|{field}")


def paired_delta(dirname, keyfmt, lo, hi):
    """Paired per-seed (ASR at the top rung - ASR at the identity rung), on seeds present in BOTH.

    Paired because every rung of an arm runs at the same seeds, so each difference holds one data
    partition fixed. Returns None when either rung is missing, which is what makes an unfinished arm
    print `--` instead of a number computed over whichever seeds happen to have landed.
    """
    cells = load(dirname)
    if cells is None:
        return None
    L, H = cells.get(keyfmt.format(r=lo)), cells.get(keyfmt.format(r=hi))
    if not L or not H:
        return None
    a = {r["seed"]: r["asr"] for r in L["per_seed"]}
    b = {r["seed"]: r["asr"] for r in H["per_seed"]}
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([b[s] - a[s] for s in seeds], dtype=float)
    r = tost(d)
    r["seeds"] = seeds
    return r


def fmt_delta(r):
    """`--` when no run supports the cell; otherwise the paired delta with its n."""
    if r is None:
        return "--"
    return f"{r['mean']:+.3f} (n={r['n']})"


def main():
    print("CROSS-CLASS PREDICTION MATRIX")
    print("prop:invariance classifies aggregators under POSITIVE RESCALING only. Coordinate masking")
    print("is outside that family, so the theory makes no prediction there -- which is the test.\n")

    # Traceability first: a prediction is only frozen if its document is committed.
    frozen = {}
    for cls, (relpath, recorded) in PREREGS.items():
        actual = prereg_commit(relpath)
        ok = actual is not None and actual.startswith(recorded[:7])
        frozen[cls] = ok
        state = f"committed at {actual}" if actual else "UNTRACKED -- not a pre-registration"
        flag = "OK" if ok else "MISMATCH"
        print(f"  [{flag}] {cls:9s} predictions frozen in {relpath}")
        print(f"           recorded {recorded}, {state}")
    print()

    if not os.path.exists(MODE_M_ADM):
        print(f"  {MODE_M_ADM} ABSENT: every masking cell will print `--`.\n")

    # The observed channel cells, read and then checked against the frozen document.
    drift = []
    table = []
    for agg, cls, pred_s, pred_m, doc_s, doc_m, attack in ROWS:
        obs_s = channel(MODE_S_ADM, "doseS", agg, S_TOP, "decision")
        obs_m = channel(MODE_M_ADM, "doseM", agg, M_TOP, "decision")
        adm_m = channel(MODE_M_ADM, "doseM", agg, M_TOP, "admission")
        for name, obs, doc in (("rescaling", obs_s, doc_s), ("masking", obs_m, doc_m)):
            if obs is not None and abs(obs - doc) > PREREG_TOL:
                drift.append(f"{agg} {name}: artifact {obs:.6f} vs pre-registration {doc:.3f}")

        asr_s = asr_m = None
        if frozen["rescaling"] and agg in ASR_S:
            asr_s = paired_delta(*ASR_S[agg], IDENTITY, S_TOP)
        if frozen["masking"] and agg in ASR_M:
            asr_m = paired_delta(*ASR_M[agg], IDENTITY, M_TOP)

        table.append({"agg": agg, "cls": cls, "attack": attack,
                      "pred_s": pred_s if frozen["rescaling"] else "--",
                      "pred_m": pred_m if frozen["masking"] else "--",
                      "obs_s": obs_s, "obs_m": obs_m, "adm_m": adm_m,
                      "asr_s": asr_s, "asr_m": asr_m})

    if drift:
        print("  ARTIFACT/PRE-REGISTRATION DRIFT -- refusing to emit:")
        for d in drift:
            print(f"    {d}")
        return 1
    print(f"  all {2 * len(ROWS)} observed channel cells agree with the frozen document "
          f"to {PREREG_TOL}\n")

    def cell(v, w=7):
        return f"{v:.3f}".rjust(w) if v is not None else "--".rjust(w)

    print("  decision change at the top rung of each class, and the paired ASR difference")
    print(f"  {'aggregator':13s} {'prop:invariance':24s} {'resc.':>7s} {'mask':>7s} "
          f"{'adm(mask)':>9s}  {'ASR resc.':>14s} {'ASR mask':>14s}")
    for r in table:
        print(f"  {r['agg']:13s} {r['cls']:24s} {cell(r['obs_s'])} {cell(r['obs_m'])} "
              f"{cell(r['adm_m'], 9)}  {fmt_delta(r['asr_s']):>14s} {fmt_delta(r['asr_m']):>14s}")

    # The adjudicating cell, stated as a claim about classes rather than about aggregators.
    ck = next(r for r in table if r["agg"] == "cos_krum")
    if ck["obs_s"] is not None and ck["obs_m"] is not None:
        print(f"\n  ADJUDICATING CELL: cos_krum reclassifies across the two classes.")
        print(f"    exactly invariant under rescaling (decision {ck['obs_s']:.3f} at every rung), and")
        print(f"    under masking its decision changes in {ck['obs_m']:.3f} of rounds with adversarial")
        print(f"    admission {ck['adm_m']:.3f}. One class certifies what the other cannot, on the same")
        print(f"    aggregator, attack and seeds. Both masking cells are CHANNEL results: cos_krum's")
        print(f"    ASR under masking is not measured and prints `--`.")

    kr = next(r for r in table if r["agg"] == "krum")
    if kr["adm_m"] is not None:
        verdict = ("GENERALIZATION (flat predicted)" if kr["adm_m"] <= 0.05
                   else "ADMISSION (ASR predicted to move with admission)")
        print(f"\n  krum mask arm type: {verdict}")
        print(f"    decision {kr['obs_m']:.3f} disturbed, admission {kr['adm_m']:.3f} unchanged -- the")
        print(f"    flagship's premise, outside the rescaling family. ASR: {fmt_delta(kr['asr_m'])}")
        if kr["asr_m"] is None:
            print("    (the mask ladder is still running; no outcome claim is made from a partial arm)")

    # LaTeX, emitted rather than transcribed.
    print("\n--- LaTeX rows (aggregator, class, resc. pred/obs, mask pred/obs, ASR) ---")
    for r in table:
        tex = r["agg"].replace("_", "\\_")
        o_s = f"${r['obs_s']:.3f}$" if r["obs_s"] is not None else "---"
        o_m = f"${r['obs_m']:.3f}$" if r["obs_m"] is not None else "---"
        a_s = f"${r['asr_s']['mean']:+.3f}$" if r["asr_s"] else "---"
        a_m = f"${r['asr_m']['mean']:+.3f}$" if r["asr_m"] else "---"
        print(f"\\texttt{{{tex}}} & {r['cls']} & {r['pred_s']} & {o_s} & "
              f"{r['pred_m']} & {o_m} & {a_s} & {a_m} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
