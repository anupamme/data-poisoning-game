"""Emit the per-seed LaTeX rows for Arm D's displayed CIFAR-100 menu pair, and check them in place.

WHY THIS EXISTS. The Reproducibility statement promises that every table cell is "emitted by a
generator that reads only frozen artifacts rather than being transcribed", and it promises per-seed
values for every arm. Both are whole-file promises with no gate behind them: nothing in the build, the
clarity gate or the page gate can see a table that was typed by hand, and a hand-typed table is exactly
what a new arm produces if nobody writes its emitter. This arm was that case. It is the same obligation
`emit_perseed_magnitude_arms.py` discharges for the two magnitude-channel arms, and it is discharged
the same way.

WHAT IT READS. `results/cifar100_composition_suite/summary.json` only, through the FROZEN runner's own
`out_path`, `pair_key`, `SEEDS` and `ATTACKS`, so a second implementation cannot land on a different
cell key or a different seed set. It computes nothing the paper reports as an estimand: no fraction, no
base rate, no precision, no recall. Those come from the frozen runner's own `--report` and from
nowhere else. What it does compute is the display rounding of measured floats, plus the two per-attack
standard deviations the paragraph after the table quotes.

A missing seed is a hard exit, never a short row: a five-row table under a caption saying n=5 whose
artifact holds four runs is the failure this refuses to print.

    PYTHONPATH=. python3 -m experiments.emit_perseed_cifar100_menu
    PYTHONPATH=. python3 -m experiments.emit_perseed_cifar100_menu --check

`--check` asserts every emitted row appears verbatim in the paper -- either document, since Round 83
moved this arm's section into the supplement -- so the pasted table is
verified against the artifact mechanically rather than by eye. Run it after any re-merge: this arm's
coverage changes as slices land, and the displayed pair's rows must not drift with them.
"""

import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from experiments.run_cifar100_composition_suite import (  # noqa: E402
    ATTACKS, SEEDS, out_path, pair_key)

# The pair the paper displays: the more clearly LOW of the two the screen misses. Named here as a
# (d1, d2) tuple and keyed through the frozen pair_key, not written out as a string.
PAIR = ("foolsgold", "coord_median")

# The paper a reviewer receives is two xr-linked documents, and Round 83 relocated this arm's section
# from the main paper's appendix into the supplement. --check therefore reads BOTH and requires the row
# in one of them: absent from both is still a hard exit, so the check is scoped, not softened. It also
# prints WHICH document carried the rows, because a silent move between the two is what this would
# otherwise hide.
DOCS = (os.path.join(BASE, "paper", "main.tex"),
        os.path.join(BASE, "paper", "supplementary.tex"))


def cells():
    """{attack: {seed: (accuracy, asr)}} for PAIR, straight from the frozen artifact."""
    if not os.path.exists(out_path):
        sys.exit(f"REFUSING TO EMIT: {out_path} is absent, so there is no artifact to read.")
    k = pair_key(*PAIR)
    pairs = json.load(open(out_path)).get("pairs", {})
    if k not in pairs:
        sys.exit(f"REFUSING TO EMIT: {k} is not in the artifact's pairs.")
    out = {}
    for att in ATTACKS:
        rows = {int(r["seed"]): (float(r["accuracy"]), float(r["asr"]))
                for r in pairs[k].get(att, {}).get("per_seed", [])}
        missing = [s for s in SEEDS if s not in rows]
        if missing:
            sys.exit(f"REFUSING TO EMIT: {k} / {att} is missing seed(s) {missing}. A short row under a "
                     f"caption claiming n={len(SEEDS)} is the failure this refuses to print.")
        out[att] = rows
    return k, out


def rows(c):
    """The tabular's body lines, exactly as they must appear in main.tex."""
    a, b = ATTACKS
    body = [f"{s} & ${c[a][s][0]:.3f}$ & ${c[a][s][1]:.3f}$ "
            f"& ${c[b][s][0]:.3f}$ & ${c[b][s][1]:.3f}$ \\\\" for s in SEEDS]
    mean = [float(np.mean([c[x][s][i] for s in SEEDS])) for x in (a, b) for i in (0, 1)]
    body.append("mean & " + " & ".join(f"${m:.3f}$" for m in
                                       (mean[0], mean[1], mean[2], mean[3])) + " \\\\")
    return body


def main():
    k, c = cells()
    a, b = ATTACKS
    body = rows(c)

    print(f"=== per-seed rows for {k}, from {os.path.relpath(out_path, BASE)} ===")
    for line in body:
        print("  " + line)

    # The figures the paragraph after the table quotes. mean_asr is the frozen runner's own field and
    # is NOT recomputed here; only its display rounding is.
    ma = {x: float(np.mean([c[x][s][1] for s in SEEDS])) for x in (a, b)}
    sd = {x: float(np.std([c[x][s][1] for s in SEEDS], ddof=1)) for x in (a, b)}
    mx = max(ma.values())
    print(f"\n  max_committed_asr = {mx:.4f}   (the larger of the two mean ASRs)")
    print(f"  per-attack ASR sd at n={len(SEEDS)}, ddof=1: "
          + ", ".join(f"{x} {sd[x]:.3f}" for x in (a, b)))
    print(f"  (population sd, ddof=0, for contrast: "
          + ", ".join(f"{np.std([c[x][s][1] for s in SEEDS], ddof=0):.3f}" for x in (a, b)) + ")")

    if "--check" not in sys.argv:
        print("\n  --check asserts these rows appear verbatim in the paper (either document).")
        return 0

    texts = {os.path.relpath(p, BASE): open(p).read() for p in DOCS}
    print(f"\n=== --check against {', '.join(texts)} ===")
    bad, where = [], {}
    for line in body:
        hit = [rel for rel, text in texts.items() if line in text]
        if hit:
            where.setdefault(", ".join(hit), 0)
            where[", ".join(hit)] += 1
        else:
            bad.append(line)
    if bad:
        print("  ROWS NOT PRESENT VERBATIM. The pasted table does not match the artifact:")
        for line in bad:
            print("    " + line)
        sys.exit(f"REFUSING TO PASS: {len(bad)} of {len(body)} emitted rows are in neither document. "
                 "Paste this emitter's output over the table; do not edit the numbers by hand.")
    for rel, n in sorted(where.items()):
        print(f"  [OK] {n} of {len(body)} emitted rows appear verbatim in {rel}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
