"""Emit the per-seed LaTeX rows for the two magnitude-channel arms (A and B).

WHY THIS EXISTS. The Reproducibility statement promises per-seed values for EVERY arm, and it is a
whole-file promise with no gate behind it: nothing in the build, the clarity gate or the page gate can
see an arm that was added without its per-seed table. These two arms are that case, so the rows are
emitted here rather than transcribed.

WHAT IT READS, AND FROM WHOSE READER.
    Arm A  krum / committed_scaling, score-only, kappa=0 -> kappa=2
           the three legs are read through analyze_score_only_kappa2_topup's own load()/leg(), so a
           second implementation cannot land on a different cell key.
    Arm B  coord_median / committed_pixel, coordinate-wise magnitude control, kappa=0 -> kappa=2
           the kappa=0 and uncontrolled kappa=2 legs are read through the RUNNER's published(), which
           is the same function the analyzer uses.

Both arms are paired on seeds 42-61. A missing seed on any leg is a hard exit, never a short row:
a table with 19 rows under a caption saying n=20 is the failure this refuses to print.

    PYTHONPATH=. python3 -m experiments.emit_perseed_magnitude_arms
"""

import os
import sys
import json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from experiments.analyze_score_only_kappa2_topup import (load, leg, ARM, PUB,      # noqa: E402
                                                         BASE_TOPUP, CELL_K2_TOPUP,
                                                         CELL_K2_PUB, CELL_K0, SEEDS)
from experiments.run_score_only_coordmedian import published, KAPPA, OUT           # noqa: E402

CELL_B = f"doseS_kappa{KAPPA}_then_coord_median|committed_pixel|score_only"


def arm_a():
    """seed -> (asr_kappa0, asr_kappa2_score_only)."""
    k2 = dict(leg(load(ARM), CELL_K2_TOPUP, ARM))
    pub = load(PUB)
    k2.update(leg(pub, CELL_K2_PUB, PUB))
    k0 = dict(leg(load(BASE_TOPUP), CELL_K0, BASE_TOPUP))
    k0.update(leg(pub, CELL_K0, PUB))
    out = {}
    for s in SEEDS:
        if s not in k0 or s not in k2:
            sys.exit(f"REFUSING TO EMIT: Arm A is missing seed {s} on one leg "
                     f"(kappa=0 {s in k0}, kappa=2 {s in k2}).")
        out[s] = (k0[s][0], k2[s][0])
    return out


def arm_b():
    """seed -> (asr_kappa0, asr_kappa2_control, asr_kappa2_uncontrolled)."""
    pub = published()
    p = os.path.join(OUT, "summary.json")
    if not os.path.exists(p):
        sys.exit(f"REFUSING TO EMIT: {p} is absent.")
    ctrl = {int(r["seed"]): float(r["asr"])
            for r in json.load(open(p))["cells"][CELL_B]["per_seed"]}
    out = {}
    for s in SEEDS:
        missing = [n for n, d in (("kappa=0", pub.get(0.0, {})),
                                  ("kappa=2 uncontrolled", pub.get(KAPPA, {})),
                                  ("kappa=2 control", ctrl)) if s not in d]
        if missing:
            sys.exit(f"REFUSING TO EMIT: Arm B is missing seed {s} on {missing}.")
        out[s] = (pub[0.0][s][1], ctrl[s], pub[KAPPA][s][1])
    return out


def main():
    a, b = arm_a(), arm_b()
    print("% Emitted by experiments/emit_perseed_magnitude_arms.py. Copy the ROWS verbatim;")
    print("% regenerate rather than edit. Every column is one leg's ASR at that seed.")
    for s in SEEDS:
        a0, a2 = a[s]
        b0, b2, bu = b[s]
        print(f"${s}$ & ${a0:.3f}$ & ${a2:.3f}$ & ${a2 - a0:+.3f}$ "
              f"& ${b0:.3f}$ & ${b2:.3f}$ & ${b2 - b0:+.3f}$ & ${bu:.3f}$ \\\\")
    n = len(SEEDS)
    ma0 = sum(a[s][0] for s in SEEDS) / n
    ma2 = sum(a[s][1] for s in SEEDS) / n
    mb0 = sum(b[s][0] for s in SEEDS) / n
    mb2 = sum(b[s][1] for s in SEEDS) / n
    mbu = sum(b[s][2] for s in SEEDS) / n
    print("\\midrule")
    print(f"mean & ${ma0:.3f}$ & ${ma2:.3f}$ & ${ma2 - ma0:+.4f}$ "
          f"& ${mb0:.3f}$ & ${mb2:.3f}$ & ${mb2 - mb0:+.4f}$ & ${mbu:.3f}$ \\\\")
    print()
    print("% Cross-check against the two analyzers' own primaries, which must agree to 1e-9:")
    print(f"%   Arm A Delta = {ma2 - ma0:+.9f}    Arm B Delta_B = {mb2 - mb0:+.9f}")
    print(f"%   Arm B Delta_mag = {mbu - mb2:+.9f}")


if __name__ == "__main__":
    main()
