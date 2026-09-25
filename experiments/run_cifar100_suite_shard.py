"""One SLICE of Arm D's wave-1 menu, so the LOW pairs the estimand needs finish before the deadline.

WHY THIS FILE EXISTS, AND WHAT IT DEVIATES FROM
-----------------------------------------------
`experiments/pre_registration_cifar100_composition_suite.md` freezes the run ORDER -- wave 1's 18
pairs in the imported list order, attacks in ATTACKS order, seeds ascending -- so that "any partial
completion is a pre-specified subset rather than an arbitrary truncation". Obeying that order to the
submission deadline would have stopped at about pair 16 of 18, and the pair it would have dropped is
position 18, `foolsgold_then_rfa`, whose CIFAR-10 `max_committed_asr` of 0.0454 is the LOWEST in the
menu. The five CIFAR-10 LOW pairs sit at wave-1 positions 8, 9, 12, 13 and 18, and precision and
recall over the LOW class ARE the estimand, so serial obedience would have cost the most informative
cell the suite contains.

So the menu is cut into disjoint slices and the LOW slices are taken FIRST. That reordering is the
ONE departure from the frozen order, and it is disclosed as such in this docstring, in every shard
artifact's `deviation_from_the_frozen_run_order` field, in the merged artifact, and in the paper.

Execution is SERIAL: one worker, one run at a time. An earlier attempt ran two disjoint slices on two
concurrent workers and the box killed both in their first run under memory pressure, banking nothing,
so concurrency was abandoned and is not part of the deviation. The slices are therefore chained, not
parallel, and the only thing that distinguishes this scheduler from the frozen loop is which pair it
reaches first.

The reordering is not a change to any rule: the menu, the attacks, the seeds, the gates, the
thresholds and the scoring all come from the frozen module by import and are not restated here.
Nothing in the freeze is edited. If the wave completes, the completed set is the whole of wave 1 and
the order is moot; if a slice is interrupted, the completed set is stated exactly, with the unrun
pairs named, rather than described as a prefix it is not.

WHY A MERGE IS SOUND
--------------------
`run_cifar100_composition_suite.run_one` sets `torch.manual_seed(seed)` and `np.random.seed(seed)` as
its first two statements, so a run's result is a deterministic function of (seed, d1, d2, attack) and
carries no state from whatever ran before it. Execution order therefore cannot change a number, and
the union of disjoint slices is bit-identical to the serial run. `--determinism-check` does not take
that argument on trust: it re-runs one cell this repository has already computed and refuses to
continue unless the pair comes back bit-identical.

WHAT IT REFUSES TO DO
---------------------
It never writes the canonical artifact. Each shard owns
`results/cifar100_composition_suite_shard_<from>_<to>/summary.json`, so no two processes write one
file and the canonical artifact is read-only for the whole sharded phase. A shard artifact is not a
suite and says so in its own `run_scope`; only `merge_cifar100_suite_shards.py` may combine them, and
only the frozen runner's own `--report` may score the result.

  one shard:  PYTHONPATH=. python3 -m experiments.run_cifar100_suite_shard --from 8 --to 13
"""

import json
import os
import sys
import time

# The frozen module, imported whole. Every rule-bearing constant below is ITS constant, read through
# this import and never restated here, which is what makes this file a scheduler rather than a second
# definition of the experiment.
from experiments.run_cifar100_composition_suite import (  # noqa: E402
    ACC_FLOOR, ATTACKS, DATASET, MODEL, PAIRS, PREREG_COMMIT, SEEDS, WAVE_OF, base,
    check_frozen, load, load_probe, out_path as CANONICAL_PATH, pair_key, prereg_md5,
    probe_verdict, run_one)

# Wave 1 in the frozen order. PAIRS is wave 1 + wave 2 by construction, so filtering preserves the
# order the freeze fixed; it is not re-sorted and not re-listed.
WAVE1 = [p for p in PAIRS if WAVE_OF[p] == 1]

# The one cell re-run by --determinism-check. Chosen because the canonical artifact already holds it
# complete, so the comparison costs one run rather than a cell.
DET_PAIR, DET_ATTACK, DET_SEED = ("reputation", "coord_median"), "committed_scaling", 42


def parse_args():
    """--from and --to are 1-indexed, inclusive, positions in the frozen wave-1 order."""
    def val(flag):
        if flag not in sys.argv:
            sys.exit(f"REFUSING TO RUN: {flag} is required. Positions are 1-indexed and inclusive "
                     f"in the frozen wave-1 order, so --from 8 --to 13 is six pairs.")
        return int(sys.argv[sys.argv.index(flag) + 1])
    a, b = val("--from"), val("--to")
    if not (1 <= a <= b <= len(WAVE1)):
        sys.exit(f"REFUSING TO RUN: --from {a} --to {b} is not a slice of 1..{len(WAVE1)}.")
    return a, b


def inherited_gates():
    """Gate 0 and the CIFAR-10 cross-check, READ from the canonical artifact, never re-decided.

    A shard applies the frozen gates; it does not get to re-run them and it does not get to assume
    them. If the canonical artifact does not record both as passing, there is nothing for this shard
    to inherit and it must not spend runs.
    """
    if not os.path.exists(CANONICAL_PATH):
        sys.exit(f"REFUSING TO RUN: {CANONICAL_PATH} does not exist, so gate 0 and the CIFAR-10 "
                 "cross-check have no recorded verdict for this shard to inherit.")
    pairs, xcheck = load()
    if xcheck is None or not xcheck.get("all_passed"):
        sys.exit("REFUSING TO RUN: the canonical artifact records no passing CIFAR-10 cross-check. "
                 "Run the frozen runner's --harness-check first; a shard does not get to skip it.")
    pv = probe_verdict(load_probe())
    if not pv.get("complete") or not pv.get("go"):
        sys.exit("REFUSING TO RUN: gate 0's probe is not recorded complete-and-go in the canonical "
                 f"artifact ({pv.get('decision')!r}). The frozen decision rule, not this shard, "
                 "decides whether the suite runs at all.")
    return pairs, xcheck, pv


def determinism_check(canonical):
    """Re-run one already-computed cell and require BIT-identity, or stop.

    This is the premise the whole merge rests on, so it is measured rather than argued. A near miss
    is not a pass: the comparison is `==` on the floats the artifact stores.
    """
    d1, d2 = DET_PAIR
    k = pair_key(d1, d2)
    rows = {r["seed"]: r for r in canonical.get(k, {}).get(DET_ATTACK, {}).get("per_seed", [])}
    if DET_SEED not in rows:
        sys.exit(f"REFUSING TO RUN: --determinism-check wants {k}/{DET_ATTACK}/s{DET_SEED} from the "
                 "canonical artifact and it is not there, so there is nothing to compare against.")
    want = rows[DET_SEED]
    print(f"=== DETERMINISM CHECK: {k} / {DET_ATTACK} / seed {DET_SEED} ===", flush=True)
    print(f"    canonical: acc={want['accuracy']:.16f} asr={want['asr']:.16f}")
    print("    Re-running it here. The merge is sound only if execution order cannot change a "
          "number,\n    and the pass criterion is exact equality, not a tolerance.", flush=True)
    t = time.time()
    acc, asr = run_one(DET_SEED, d1, d2, DET_ATTACK)
    d = (acc - want["accuracy"], asr - want["asr"])
    exact = (acc == want["accuracy"]) and (asr == want["asr"])
    print(f"    recomputed: acc={acc:.16f} asr={asr:.16f}")
    print(f"    d=({d[0]:+.2e}, {d[1]:+.2e})  -> {'BIT-IDENTICAL' if exact else 'NOT IDENTICAL'}"
          f"  ({time.time() - t:.0f}s)", flush=True)
    if not exact:
        sys.exit("REFUSING TO CONTINUE: the recomputation is not bit-identical, so runs are NOT "
                 "order-independent on this box and a sharded union is not the serial artifact. "
                 "Kill every shard, run the frozen runner serially in its frozen order, and report "
                 "the pre-registered prefix.")
    return {"pair": k, "attack": DET_ATTACK, "seed": DET_SEED,
            "canonical": {"accuracy": want["accuracy"], "asr": want["asr"]},
            "recomputed": {"accuracy": acc, "asr": asr},
            "delta": {"accuracy": d[0], "asr": d[1]}, "bit_identical": exact,
            "what_it_licenses": "Execution order does not change a number, so the union of disjoint "
                                "shards is bit-identical to the serial run. Exact equality was the "
                                "pass criterion; 1e-09 was not."}


def load_own(shard_path, my_pairs):
    """This shard's OWN rows from a previous invocation of the SAME slice, carried forward.

    Without this the file is truncated on every re-invocation, and that destroys data. The sequence
    that did it once: invocation 1 banks runs here; the merge copies them into the canonical artifact;
    invocation 2 of the same slice sees them in the canonical artifact and SKIPS them, so they never
    re-enter `rows`; `save()` then overwrites this file with the smaller set. The rows now live only in
    the canonical artifact, and the merge rebuilds that from `summary.pre_merge.json` plus the shard
    files, so the next merge silently drops them. Eight runs of `foolsgold_then_rfa` were lost that
    way. A shard file is therefore APPEND-ONLY over its own slice, which is the invariant the merge's
    reconstruction already assumes.
    """
    if not os.path.exists(shard_path):
        return {}
    mine = {pair_key(*p) for p in my_pairs}
    prior = json.load(open(shard_path)).get("pairs", {})
    return {k: v for k, v in prior.items() if k in mine}


def save(shard_path, a, b, my_pairs, rows, det):
    """A shard artifact, written after every run, that cannot be mistaken for the suite."""
    n = sum(len(c[att]["per_seed"]) for c in rows.values() for att in ATTACKS if att in c)
    json.dump({
        "description":
            f"SHARD {a}-{b} of Arm D's wave-1 menu on {DATASET}/{MODEL}: wave-1 positions {a} to {b} "
            f"inclusive in the frozen order, against both committed attacks at seeds "
            f"{SEEDS[0]}-{SEEDS[-1]}. This is a SLICE, not a suite.",
        "run_scope":
            f"NOT A SUITE AND NOT SCOREABLE ALONE. This file holds {n} runs over "
            f"{len(my_pairs)} of the 18 wave-1 pairs. It must be merged with the canonical artifact "
            "by experiments/merge_cifar100_suite_shards.py and scored only by the frozen runner's "
            "own --report. No fraction, base rate or baseline may be computed from this file.",
        "deviation_from_the_frozen_run_order":
            "The freeze fixes wave 1's pairs in the imported list order so that any partial "
            "completion is a pre-specified prefix. ONE thing here departs from that order, for the "
            "submission deadline: the slices are taken LOW-PAIRS-FIRST rather than in list order. "
            "Obeying the list order would have stopped at about pair 16 of 18 and dropped wave-1 "
            "position 18, foolsgold_then_rfa, whose CIFAR-10 max_committed_asr of 0.0454 is the "
            "lowest in the menu and which is one of the five LOW pairs (positions 8, 9, 12, 13, 18) "
            "the estimand is computed over. Execution is SERIAL, one worker, one run at a time: a "
            "first attempt ran two concurrent workers and the box killed both in their first run "
            "under memory pressure, banking nothing, so concurrency is not part of this deviation. "
            "No rule, threshold, seed, attack, gate or menu entry is changed; only the order is. The "
            "deviation is reported against the freeze and the freeze is not edited.",
        "shard": {"from": a, "to": b, "pairs": [pair_key(*p) for p in my_pairs],
                  "positions_are": "1-indexed, inclusive, in the frozen wave-1 order"},
        "prereg": "experiments/pre_registration_cifar100_composition_suite.md",
        "prereg_commit": PREREG_COMMIT, "prereg_md5": prereg_md5(),
        "gates_inherited_not_rerun":
            "Gate 0's probe verdict and the CIFAR-10 cross-check are READ from "
            "results/cifar100_composition_suite/summary.json and this shard refuses to start unless "
            "both are recorded as passing. Gate 1 (adversary_hook verify=True) and gate 2 (the "
            "accuracy floor) are inside the imported run_one and apply unchanged.",
        "determinism_check": det,
        "dataset": DATASET, "model": MODEL, "acc_floor": ACC_FLOOR,
        "attacks": list(ATTACKS), "seeds": list(SEEDS),
        "n_runs_here": n,
        "pairs": rows,
    }, open(shard_path, "w"), indent=2)


def main():
    check_frozen()
    a, b = parse_args()
    my_pairs = WAVE1[a - 1:b]
    canonical, xcheck, pv = inherited_gates()

    shard_dir = os.path.join(base, "results", f"cifar100_composition_suite_shard_{a}_{b}")
    shard_path = os.path.join(shard_dir, "summary.json")
    os.makedirs(shard_dir, exist_ok=True)

    print(f"=== ARM D, SHARD {a}-{b}: {len(my_pairs)} of the 18 wave-1 pairs ===")
    print(f"    {DATASET}/{MODEL}, attacks {list(ATTACKS)}, seeds {SEEDS[0]}-{SEEDS[-1]}")
    print(f"    rules frozen at {PREREG_COMMIT}, md5 {prereg_md5()}")
    print(f"    gate 0 inherited: {pv.get('decision')}")
    print("    pairs: " + ", ".join(f"{a + i}.{pair_key(*p)}" for i, p in enumerate(my_pairs)))
    print("    LOW-pairs-first, SERIAL. This is a DEVIATION from the frozen run order, taken for the "
          "deadline\n    and disclosed in the artifact. No rule, seed, attack, gate or threshold is "
          "changed.", flush=True)

    det = None
    if "--determinism-check" in sys.argv:
        det = determinism_check(canonical)
        print()

    # Rows already in the canonical artifact are skipped, not recomputed: the canonical file is read
    # here and never written. Rows this slice banked on an earlier invocation are carried forward
    # rather than truncated, for the reason load_own() states.
    rows = load_own(shard_path, my_pairs)
    carried = sum(len(c[att]["per_seed"]) for c in rows.values() for att in ATTACKS if att in c)
    skipped, todo = 0, []
    for p in my_pairs:
        k = pair_key(*p)
        have = {att: {r["seed"] for r in canonical.get(k, {}).get(att, {}).get("per_seed", [])}
                | {r["seed"] for r in rows.get(k, {}).get(att, {}).get("per_seed", [])}
                for att in ATTACKS}
        for att in ATTACKS:
            for s in SEEDS:
                if s in have[att]:
                    skipped += 1
                else:
                    todo.append((p, att, s))
    total = len(todo)
    print(f"  {total} runs to do; {skipped} already in the canonical artifact or in this shard's own "
          f"file and skipped rather than recomputed\n  ({carried} of them carried forward from an "
          f"earlier invocation of this same slice).\n", flush=True)

    save(shard_path, a, b, my_pairs, rows, det)
    t0 = time.time()
    for i, ((d1, d2), att, seed) in enumerate(todo, 1):
        k = pair_key(d1, d2)
        t = time.time()
        acc, asr = run_one(seed, d1, d2, att)
        cell = rows.setdefault(k, {"d1": d1, "d2": d2, "wave": WAVE_OF[(d1, d2)],
                                   "dataset": DATASET, "model": MODEL})
        per = {r["seed"]: r for r in cell.get(att, {}).get("per_seed", [])}
        per[seed] = {"seed": int(seed), "accuracy": acc, "asr": asr}
        cell[att] = {"per_seed": [per[s] for s in sorted(per)]}
        save(shard_path, a, b, my_pairs, rows, det)
        print(f"  [{i}/{total}] {k:34s} {att:18s} s{seed}: acc={acc:.4f} asr={asr:.4f}"
              + ("  * below acc floor" if acc < ACC_FLOOR else "")
              + f"  ({time.time() - t:.0f}s)", flush=True)
    print(f"\nShard {a}-{b} done: {total} runs in {(time.time() - t0) / 3600:.1f} h")
    print(f"Saved to {shard_path}")
    print("NOT SCOREABLE ALONE. Merge with experiments.merge_cifar100_suite_shards, then score with "
          "the frozen runner's --report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
