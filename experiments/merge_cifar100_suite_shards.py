"""Merge Arm D's shard artifacts into the canonical suite artifact, and refuse if the merge is unsound.

WHY THIS FILE EXISTS. run_cifar100_composition_suite.py walks wave 1 in one serial loop and is not
editable. At the measured per-run cost that loop reaches about pair 16 of 18 before the submission
deadline, and the pair it drops is wave-1 position 18, foolsgold_then_rfa -- the lowest-ASR
composition in the CIFAR-10 menu and one of the five LOW pairs the estimand is computed over. So the
remaining pairs were run as disjoint slices by run_cifar100_suite_shard.py, serially, with the LOW
slices taken first. This file puts the slices back together.

WHAT LICENSES THE MERGE. run_one sets torch.manual_seed(seed) and np.random.seed(seed) as its first
two statements, so a run's (accuracy, asr) is a deterministic function of (seed, d1, d2, attack) and
is independent of execution order. That is an argument, not evidence, so shard 8-9 deliberately
re-ran one already-banked run and asserted bit-identity against the canonical artifact; its verdict
is carried in every shard's "determinism_check" field and is re-checked here. This file additionally
asserts that EVERY (pair, attack, seed) appearing in more than one source is bit-identical across
sources. Read that second assertion for what it is: the slices are disjoint by construction and the
determinism re-run is stored in "determinism_check" rather than in "pairs", so the overlap is normally
EMPTY and an empty overlap is a property of the design, not a second passed check. The assertion exists
to catch a mis-specified slice, and the deliberate re-run is what actually measures the premise.

WHAT THIS FILE DOES NOT DO. It computes no fraction, no base rate, no precision and no recall. The
union is handed to the FROZEN runner's own save(), so every derived number in the merged artifact is
produced by frozen scoring code, and the arm is then read with that runner's own --report. This file
also does not edit the freeze: the sharding and the LOW-first ordering are DEVIATIONS from the frozen
run order and are recorded as such, in the artifact, in the shard files and in the paper.

    PYTHONPATH=. python3 -m experiments.merge_cifar100_suite_shards [--dry-run]

It is idempotent: the pre-merge canonical artifact is kept beside the merged one as
summary.pre_merge.json and is preferred as the merge base whenever it exists, so re-running merges
the shards into the ORIGINAL rows rather than into its own output.

ONE CAUTION, from a defect this codebase has already shipped. The frozen save() writes a fixed set of
keys, so the two provenance keys this file adds (merge_provenance, sharding_deviation) would be
DELETED by a later `--harness-check` or by a bare re-run of the suite, both of which call save().
`--report` does not call save() and is safe. After the merge, read the arm with --report only.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_cifar100_composition_suite import (  # noqa: E402
    ACC_FLOOR, ATTACKS, DATASET, MODEL, PAIRS, PREREG_COMMIT, SEEDS, WAVE_OF, base, check_frozen,
    load_probe, out_dir, out_path, pair_key, prereg_md5, probe_verdict, report, save)

SHARD_GLOB = os.path.join(base, "results", "cifar100_composition_suite_shard_*", "summary.json")
PRE_MERGE = os.path.join(out_dir, "summary.pre_merge.json")


def die(msg):
    sys.exit(f"REFUSING TO MERGE: {msg}")


def load_base():
    """The merge base: the pre-merge snapshot when one exists, so re-running is idempotent."""
    src = PRE_MERGE if os.path.exists(PRE_MERGE) else out_path
    if not os.path.exists(src):
        die(f"no canonical artifact at {out_path}. The shards inherit their gates from it.")
    d = json.load(open(src))
    x = d.get("cross_check")
    if x is None or not x.get("all_passed"):
        die(f"{src} records no passing CIFAR-10 cross-check.")
    pv = probe_verdict(load_probe())
    if not (pv["complete"] and pv["go"]):
        die("gate 0's probe is not both complete and GO. The suite has no licence to hold these runs.")
    return src, d


def check_shard_agrees_with_the_freeze(path, s):
    """A shard may differ from the suite in WHICH pairs it holds and in nothing else."""
    for field, mine in (("prereg_commit", PREREG_COMMIT), ("prereg_md5", prereg_md5()),
                        ("dataset", DATASET), ("model", MODEL), ("acc_floor", ACC_FLOOR)):
        if s.get(field) != mine:
            die(f"{path}: {field} is {s.get(field)!r}, the frozen value is {mine!r}.")
    if list(s.get("attacks", [])) != list(ATTACKS):
        die(f"{path}: attacks are {s.get('attacks')}, frozen {list(ATTACKS)}.")
    if list(s.get("seeds", [])) != list(SEEDS):
        die(f"{path}: seeds are {s.get('seeds')}, frozen {list(SEEDS)}.")
    # The determinism check is run ONCE, by whichever shard was given --determinism-check; the
    # chained slices carry determinism_check: null. So a null here is permitted and a FAILED one is
    # not, and main() separately requires that at least one shard carried a passing check.
    det = s.get("determinism_check")
    if det is not None and not det.get("bit_identical"):
        die(f"{path}: its determinism check did not return bit-identity "
            f"({det.get('delta')}), so execution order changed a number and the slices may not be "
            "combined. Fall back to serial execution in the frozen order.")
    return det


def rows_of(cell, att):
    return {int(r["seed"]): r for r in cell.get(att, {}).get("per_seed", [])}


def main():
    check_frozen()
    dry = "--dry-run" in sys.argv
    src, d = load_base()
    pairs = d.get("pairs", {})
    print("=== MERGE: Arm D shard artifacts into the canonical suite artifact ===")
    print(f"    base: {os.path.relpath(src, base)}")

    shard_paths = sorted(glob.glob(SHARD_GLOB))
    if not shard_paths:
        die(f"no shard artifacts match {os.path.relpath(SHARD_GLOB, base)}. Nothing to merge.")

    # Provenance per row, so a conflict names both sources rather than just failing.
    origin = {}
    for k, cell in pairs.items():
        for att in ATTACKS:
            for seed in rows_of(cell, att):
                origin[(k, att, seed)] = os.path.relpath(src, base)

    added, overlap, conflicts, shard_note = 0, 0, [], []
    det_passed = []
    for path in shard_paths:
        s = json.load(open(path))
        rel = os.path.relpath(path, base)
        det = check_shard_agrees_with_the_freeze(rel, s)
        if det is not None:
            det_passed.append({"shard_file": rel, **det})
        sh = s.get("shard", {})
        here = 0
        for k, scell in s.get("pairs", {}).items():
            d1, d2 = scell["d1"], scell["d2"]
            if pair_key(d1, d2) != k or (d1, d2) not in WAVE_OF:
                die(f"{rel}: pair key {k!r} is not a menu pair.")
            cell = pairs.setdefault(k, {"d1": d1, "d2": d2, "wave": WAVE_OF[(d1, d2)],
                                        "dataset": DATASET, "model": MODEL})
            for att in ATTACKS:
                have = rows_of(cell, att)
                for seed, r in sorted(rows_of(scell, att).items()):
                    if seed in have:
                        overlap += 1
                        old = have[seed]
                        # Bit-identity, not a tolerance. float equality is the whole point.
                        if (old["accuracy"], old["asr"]) != (r["accuracy"], r["asr"]):
                            conflicts.append(
                                f"{k} / {att} / seed {seed}: "
                                f"{origin[(k, att, seed)]} has "
                                f"acc={old['accuracy']!r} asr={old['asr']!r}; "
                                f"{rel} has acc={r['accuracy']!r} asr={r['asr']!r}")
                        continue
                    have[seed] = {"seed": int(seed), "accuracy": float(r["accuracy"]),
                                  "asr": float(r["asr"])}
                    origin[(k, att, seed)] = rel
                    added += 1
                    here += 1
                    # The frozen loop's own two derived fields, recomputed by its own expressions so
                    # a merged cell is indistinguishable in shape from a serially written one.
                    cell[att] = {"per_seed": [have[x] for x in sorted(have)]}
                    cell[att]["mean_asr"] = float(
                        np.mean([q["asr"] for q in cell[att]["per_seed"]]))
                    cell["max_committed_asr"] = max(
                        (cell[a]["mean_asr"] for a in ATTACKS
                         if a in cell and "mean_asr" in cell[a]), default=None)
        print(f"    shard {sh.get('from')}-{sh.get('to')}: {s.get('n_runs_here')} runs, "
              f"{here} new here  [{rel}]")
        shard_note.append({"file": rel, "from": sh.get("from"), "to": sh.get("to"),
                           "pairs": sh.get("pairs"), "n_runs_here": s.get("n_runs_here"),
                           "carried_the_determinism_check": det is not None})

    if not det_passed:
        die("no shard carried a passing determinism check, so nothing measured the premise that "
            "execution order cannot change a number. Re-run one banked cell with "
            "`--determinism-check` before merging, or fall back to serial execution.")

    if conflicts:
        print("\n  BIT-IDENTITY FAILED on the overlap. The merge is unsound and nothing is written.")
        for c in conflicts:
            print(f"    {c}")
        die(f"{len(conflicts)} of {overlap} overlapping runs disagree. Execution order changed a "
            "number, so the shards may not be combined: fall back to serial execution in the frozen "
            "order and report the pre-registered prefix.")

    n_rows = sum(len(c[a]["per_seed"]) for c in pairs.values() for a in ATTACKS if a in c)
    print("\n    " + (f"overlap {overlap} runs, all bit-identical"
                      if overlap else
                      "overlap EMPTY (slices are disjoint by construction; the premise is measured by "
                      "the determinism re-run, not by this zero)")
          + f"; {added} rows added; {n_rows} rows total")

    # A row that the canonical artifact holds TODAY and this merge's result does not would be
    # DESTROYED by the write below, and `added` cannot see it because `added` counts insertions. This
    # has happened once, costing 8 measured runs: the merge rebuilds from the pre-merge snapshot plus
    # the shard files, so a row whose only home was a shard file that a second invocation of its own
    # slice truncated is covered by neither source. See run_cifar100_suite_shard.load_own(), which
    # closes the hole at the writing end; this closes it at the reading end, because a guard that
    # depends on the other file being correct is not a guard.
    now = json.load(open(out_path)).get("pairs", {}) if os.path.exists(out_path) else {}
    lost = []
    for k, cell in now.items():
        for att in ATTACKS:
            gone = sorted(set(rows_of(cell, att)) - set(rows_of(pairs.get(k, {}), att)))
            if gone:
                lost.append(f"{k} / {att}: seed(s) {gone} are in "
                            f"{os.path.relpath(out_path, base)} and not in this merge's result")
    if lost:
        print("\n  THIS MERGE WOULD DESTROY MEASURED RUNS. Nothing is written.")
        for msg in lost:
            print(f"    {msg}")
        die(f"{len(lost)} cell(s) would lose rows. {os.path.relpath(PRE_MERGE, base)} plus the shard "
            "files no longer cover what the canonical artifact holds, which is what a truncated shard "
            "file looks like from here. Restore those rows to their shard file, or re-run them -- "
            "which is safe only because bit-identical determinism was measured -- and merge again. Do "
            "NOT copy them out of a run log: the logs carry 4 decimal places and the artifact carries "
            "the float that was measured.")

    if dry:
        print("    --dry-run: nothing written.")
        return 0

    # Snapshot the pre-merge artifact before the first write, and never overwrite that snapshot.
    if not os.path.exists(PRE_MERGE):
        json.dump(json.load(open(out_path)), open(PRE_MERGE, "w"), indent=2)
        print(f"    pre-merge snapshot kept at {os.path.relpath(PRE_MERGE, base)}")

    before = {k: d.get(k) for k in ("run_scope", "cross_check", "prereg_commit", "prereg_md5")}
    save(pairs, d["cross_check"], load_probe())

    # Additive only: the two keys below are appended to what the frozen save() wrote, and no field
    # save() produced is touched. Asserted rather than trusted, because run_scope is sticky through a
    # read of this very file and stickiness is easy to break.
    m = json.load(open(out_path))
    for k, v in before.items():
        if m.get(k) != v:
            die(f"the frozen save() changed {k!r} across the merge. Restore "
                f"{os.path.relpath(PRE_MERGE, base)} over {os.path.relpath(out_path, base)}.")
    m["merge_provenance"] = {
        "merged_by": "experiments/merge_cifar100_suite_shards.py",
        "base": os.path.relpath(src, base),
        "pre_merge_snapshot": os.path.relpath(PRE_MERGE, base),
        "shards": shard_note,
        "determinism_checks": det_passed,
        "overlapping_runs_checked": overlap,
        "overlapping_runs_disagreeing": 0,
        "rows_added": added,
        "rows_total": n_rows,
        "scored_by": "the frozen runner's own score(), called through its own save(). This file "
                     "computes no fraction, base rate, precision or recall.",
        "caution": "The frozen save() writes a fixed key set, so a later --harness-check or a bare "
                   "re-run of the suite would DELETE this field and sharding_deviation. --report "
                   "does not call save() and is safe."}
    m["sharding_deviation"] = (
        "DEVIATION FROM THE FROZEN RUN ORDER, disclosed and not legislated away. The freeze fixes "
        "wave 1's pairs in the imported list order so that any partial completion is a pre-specified "
        "prefix. ONE thing departed from that order, for the submission deadline: the slices were "
        "taken LOW-PAIRS-FIRST rather than in list order. Execution was SERIAL, one run at a time; an "
        "earlier attempt at two concurrent workers was killed by the box under memory pressure in its "
        "first run and banked nothing, so concurrency is not part of the deviation. "
        "Obeying the list order would have stopped at about pair 16 of 18 and dropped wave-1 position 18, "
        "foolsgold_then_rfa, whose CIFAR-10 max_committed_asr of 0.0454 is the lowest in the menu and "
        "which is one of the five LOW pairs (positions 8, 9, 12, 13, 18) the estimand is computed "
        "over. No rule, threshold, seed, attack, gate or menu entry was changed; only the order was. "
        "What licenses recombining the slices: run_one seeds torch and numpy as its first two "
        "statements, so a run is a deterministic function of (seed, d1, d2, attack). That is an "
        "argument, so it was also measured: the deliberate re-run recorded in determinism_checks "
        "recomputed an already-banked cell and returned exact equality, not a tolerance. "
        + (f"Beyond it, {overlap} run(s) appeared in more than one source and every one was "
           "bit-identical across sources."
           if overlap else
           "Beyond it NO run appeared in more than one source, because the slices are disjoint by "
           "construction and the determinism re-run is recorded in determinism_checks rather than in "
           "pairs; so overlapping_runs_checked is 0 and that zero is a property of the design and "
           "NOT a second passed check.")
        + " The freeze itself is not edited.")
    json.dump(m, open(out_path, "w"), indent=2)
    print(f"    written to {os.path.relpath(out_path, base)}")

    print("\n=== The frozen runner's own report, over the merged rows ===")
    report(m["pairs"], load_probe())
    print("\nRead this arm ONLY with: PYTHONPATH=. python3 -m "
          "experiments.run_cifar100_composition_suite --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
