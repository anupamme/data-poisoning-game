"""How many MEASURED ROUNDS is the channel table's `Delta adm. = 0.000` actually zero in?

WHY THIS EXISTS. A reviewer objected to the paper's phrase "admission stays exactly $0.000$". The
objection is well aimed at the FORM of the claim: `Delta adm.` is not a constant of the construction,
it is the MEAN of a per-round indicator (build_channel_table.py: `(b > 0.0) != (p > 0.0)`), so a
printed 0.000 could in principle be a small nonzero mean rounded down at three decimals. "Exactly"
would then be a typographic accident dressed up as a guarantee.

It is not one. This script re-derives the same indicator round by round and reports the COUNT, which
is what the paper should say: not "the mean was 0.000" but "the support of the adversarial mass changed
in none of the N rounds we measured". A count is both honest and stronger than a rounded mean.

It also reports, per arm, how many of those rounds had adversarial mass PRESENT at baseline. That
number is the one that decides whether the zero is informative: a run in which no adversary is ever
admitted at the identity rung has nothing for the dose to change, so its zero is vacuous. Rounds with
mass present are rounds where a change was possible and did not happen.

WHAT IS NOT CLAIMED. The zero holds for each arm under ITS OWN committed attack -- the cells the
channel table actually prints. The same indicator is NOT zero everywhere off that diagonal (krum under
the pixel backdoor moves in 9 of 96 rounds), which is exactly why the arm/attack pairing is asserted
here rather than pooled. Pooling across attacks was a mistake made once while checking this and it
produced a number that contradicted the table.

Nothing is trained, nothing is written. Reads results/admission_measurement.json and
results/femnist_admission.json, both READ-ONLY, and reuses build_channel_table's own row loader and
mass-field map so this script and the published table cannot disagree about what a round is.

Run: python3 experiments/count_admission_rounds.py
"""
import os, sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
# Single-sourced from the emitter of the published table: same per-round rows, same indicator, same
# adversarial-mass field per aggregator. A second copy is how the count and the table drift apart.
from experiments.build_channel_table import (  # noqa: E402
    ROWS, MASS, SRC_REGIME, channel_rows,
)

RUNGS = (0.0, 0.5, 1.0, 2.0)


def per_rung(path, arm, attack, rung):
    """(n_rounds, n_support_changed, n_mass_present_at_baseline) for one arm at one rung."""
    rows = channel_rows(path, arm, attack, rung)
    if not rows:
        return None
    field, _ = MASS[arm]
    changed = present = 0
    for r in rows:
        b, p = float(r[f"base_{field}"]), float(r[f"post_{field}"])
        changed += int((b > 0.0) != (p > 0.0))   # the published indicator, verbatim
        present += int(b > 0.0)
    return len(rows), changed, present


def main():
    print("=== ADMISSION: THE COUNT BEHIND `Delta adm. = 0.000` ===")
    print("    Per-round support-change indicator, re-derived from the frozen channel measurements.")
    print("    Each arm under ITS OWN committed attack -- the cells the channel table prints.\n")
    # The MODEL is part of the key, not decoration. The ResNet18 row varies architecture alone, so it
    # shares (arm, attack, dataset) with the flagship CIFAR-10 Krum row: keying on those three alone
    # let it overwrite that row in `totals`, which silently dropped 60 rounds from the pooled count.
    print(f"  {'aggregator':16s} {'attack':18s} {'dataset':8s} {'model':10s} {'rounds':>7s} "
          f"{'changed':>8s} {'mass present':>13s}")
    print("  " + "-" * 85)

    totals = {}
    for label, arm, attack, src in ROWS:
        dataset, model = SRC_REGIME[src]
        n = c = p = 0
        missing = False
        for rung in RUNGS:
            got = per_rung(src, arm, attack, rung)
            if got is None:
                missing = True
                continue
            n += got[0]; c += got[1]; p += got[2]
        if missing and n == 0:
            print(f"  {arm:16s} {attack:18s} {dataset:8s} {model:10s} {'--':>7s}   "
                  f"no channel measurement")
            continue
        print(f"  {arm:16s} {attack:18s} {dataset:8s} {model:10s} {n:7d} {c:8d} {p:13d}")
        assert (arm, dataset, model) not in totals, (
            f"{(arm, dataset, model)} counted twice; one row would shadow the other")
        totals[(arm, dataset, model)] = (n, c, p)

    n = sum(v[0] for v in totals.values())
    c = sum(v[1] for v in totals.values())
    p = sum(v[2] for v in totals.values())
    print("  " + "-" * 85)
    print(f"  {'ALL ROWS':16s} {'':18s} {'':8s} {'':10s} {n:7d} {c:8d} {p:13d}")

    print(f"\n  rounds        measured rounds pooled over rungs {list(RUNGS)}")
    print("  changed       rounds in which the SUPPORT of the adversarial mass changed")
    print("  mass present  rounds with adversarial mass at baseline, i.e. a change was possible")

    print("\n=== WHAT THE PAPER MAY SAY ===")
    for (arm, dataset, model), (rn, rc, rp) in totals.items():
        tag = f"{arm} ({dataset}/{model})"
        if rc == 0 and rp > 0:
            print(f"  {tag:26s} \"changed in none of the {rn} measured rounds\" "
                  f"({rp} with mass present, so a change was possible)")
        elif rc == 0:
            print(f"  {tag:26s} zero, but VACUOUS: mass never present at baseline -- do not report "
                  f"as evidence")
        else:
            print(f"  {tag:26s} NOT zero: {rc}/{rn} rounds changed. The claim does not hold here.")

    print("\n=== OFF-DIAGONAL CONTROL: the indicator is not zero everywhere ===")
    print("    Same arm, a different attack -- not a channel-table cell, shown so the arm/attack")
    print("    pairing above reads as a scope condition rather than as a convenient filter.")
    from experiments.build_channel_table import ADM  # noqa: E402
    for arm, attack in (("krum", "committed_pixel"), ("cos_krum", "committed_scaling")):
        n = c = 0
        for rung in RUNGS:
            got = per_rung(ADM, arm, attack, rung)
            if got:
                n += got[0]; c += got[1]
        if n:
            print(f"  {arm:16s} {attack:18s} {n:7d} rounds, {c} changed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
