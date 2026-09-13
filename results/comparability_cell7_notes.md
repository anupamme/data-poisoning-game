# Cell 7 run notes (Round 57)

Companion to `results/comparability_cell7_run.log`. Nothing here is an amendment: Amendment 4 is
committed at `9b8a395` and **must not be edited while the run is resumable**, because
`run_comparability_cells.py`'s `check_frozen()` refuses to start if that one file has uncommitted
changes (`git status --porcelain -- PREREG`, `:189-193`). The gate is scoped to the pre-registration
alone, so other working-tree files may change between resumes.

## The `[69/108]` then `[36/108]` counter oddity, diagnosed

The log shows two startup banners. The first reports `resuming: 68 runs already done, 108 planned` and
numbers the next run `[69/108]`; the second reports `resuming: 69 runs already done, 108 planned` and
numbers the next run `[36/108]`, which looks like the counter went backwards after a run completed.

It did not. `i` at `:262-264` is the run's **position in the plan**, incremented before the resume-skip
at `:268`, so it never counts runs executed. The plan's ORDER changed between the two invocations: the
endpoint-first stable sort at `:257-258` was installed in the restart, deliberately, for the reason
documented at `:246-256` (so that a truncated night lands on the pre-registered endpoints-only
fallback rather than on one complete ladder with nothing to contrast).

Both numbers are reproduced exactly by re-deriving the two orderings from `CELLS`, `KAPPAS`, `SEEDS`:

| run | natural order | endpoint-first order |
|---|---|---|
| cell7 / confounded / k=0.0 / s42 | **69** | 35 |
| cell7 / confounded / k=0.0 / s43 | 70 | **36** |

34 endpoint runs belong to the six published cells, so cell 7's first confounded k=0 seed sits at 35
and the second at 36 under the sort. The bolded cells are the two indices in the log. **No run was
lost, repeated or renumbered mid-plan; the two invocations simply printed positions in two different
plan orders.**

### Why the reorder cannot change a result

- `run_one()` re-seeds on entry, before any data or model exists: `torch.manual_seed(seed);
  np.random.seed(seed)` at `:123`, and `get_federated_dataset(dataset, ..., seed)` at `:125` takes the
  seed explicitly. No run's value depends on what ran before it.
- The resume key is `(cell, family, kappa, seed)` via `cell_key()` at `:265` plus the seed test at
  `:268`, all of which are order-independent.

### Verified against the saved artifact, not just the code

- `results/comparability_cells/summary.json` holds **75 per-seed runs** (68 published + 7 cell-7) with
  **zero duplicate seeds in any cell**, so the restart double-counted nothing.
- Both cell-7 keys carry the `|cifar100` suffix
  (`dose_kappa0.0_then_coord_median|committed_pixel|cifar100` and the `kappa2.0` twin), and **no
  `cifar100` result sits in an unsuffixed key**. The pooling firewall the suffix exists for (`:92`)
  holds.

## Timing caveat for the log's seconds column

The first row, `k=0.0 s42`, reports **1370s** against ~650-723s for every subsequent run. It is the
only row measured in the pre-sort invocation and the only one whose wall time includes first-touch
CIFAR-100 setup. **Do not use that row in any per-run cost estimate**; the 650-723s band from the five
subsequent runs is the comparable figure, and it is consistent with the 660s/run parsed from
`results/comparability_run.log` for CIFAR-10 + `cifar_cnn`.

## Admissibility gate (Amendment 4): PASSED

Applied to the first five confounded / k=0 runs, as the amendment defines it:

| seed | accuracy | ASR |
|---|---|---|
| 42 | 0.3914 | 0.7143 |
| 43 | 0.4040 | 0.5493 |
| 44 | 0.3926 | 0.7557 |
| 45 | 0.3950 | 0.8269 |
| 46 | 0.3826 | 0.6708 |

Rung mean accuracy **0.3931** >= 0.35, rung mean ASR **0.7034** in [0.15, 0.85]. Both gates pass, so
the cell is admissible and the ladder proceeds. Had either failed, the recorded outcome would have been
that the cell is inadmissible on headroom, with these numbers, and **no substitute cell** (non-negotiable 4).

## Progress, and what is not yet decidable

The `kappa=2.0` confounded rung is running: n=2 at mean ASR **0.4526**, mean accuracy 0.4011, against
the identity rung's 0.7034. **Nothing about the sign reversal can be read off this.** The reversal is a
contrast BETWEEN designs, and not one `controlled` run exists yet. A fall in the confounded arm alone
is the expected dose-response and is not the quantity `app:sixcell` reports.

## The κ=0 bit-exact family agreement is expected here, and the falsifier is named

Cell 7's first `controlled` run reproduces its `confounded` twin exactly (`k=0.0 s42`:
acc 0.3914, ASR 0.7143 in both). That is the signature of a silent family collapse -- the defect that
made `committed_scaling` bit-identical to `committed_pixel` in `run_cross_distribution_compositions`
because `manipulate_update` was never called -- so it was checked against the published cells rather
than assumed benign.

Per-seed comparison of the two families at each rung, over `results/comparability_cells/summary.json`:

| cell | κ=0 | κ=0.5 | κ=1.0 | κ=2.0 |
|---|---|---|---|---|
| `coord_median` × `committed_scaling` × cifar10 | **identical** | differs | differs | differs |
| `krum` × `committed_scaling` × femnist | differs | differs | differs | differs |
| `coord_median` × `committed_pixel` × cifar100 (cell 7) | **identical** | pending | pending | pending |

So κ=0 family agreement is **not** universal, and its presence is not by itself evidence of a bug: the
two ladders differ only in `d1` and `adv_mask`, κ scales `d1`, and at κ=0 `d1` is the identity, leaving
`adv_mask` as the only difference. Coordinate-wise median is insensitive to which clients the mask
names under an identity `d1`; Krum, which selects a single update, is not. Cell 7 uses `coord_median`
and reproduces the `coord_median` row's behaviour, not the `krum` row's.

**The falsifier, stated before the data exist:** if cell 7's κ=0.5, 1.0 and 2.0 rungs also come back
bit-identical across families, the cell is void as a family collapse and must be reported as such --
not as a null. The `confounded` κ=2.0 rung has already returned a mean ASR of 0.4905 against the
identity rung's 0.7034, so the remaining question is whether `controlled` κ=2.0 differs from **that**,
which is exactly the contrast `app:sixcell` reports.
