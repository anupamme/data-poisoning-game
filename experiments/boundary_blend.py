"""The boundary blend: a decision-channel intervention that is uniformly epsilon-neutral.

WHY THIS EXISTS. App. D.7's oracle-free arm moved two channels at once -- Krum's decision AND the
adversarial coefficient share, the latter in 12 of 12 adversary rounds by up to 0.0859 against a
frozen tolerance of 1e-6 -- so its refutation does not transfer to the published Mode S result.
experiments/screen_oracle_free_transforms.py then asks whether any DEPLOYED oracle-free family avoids
that, and reports what it finds. This module supplies the instrument for the case where none does.

THE CONSTRUCTION. Blend the identity toward rfa's coefficients along a one-parameter path:

    c_i(t) = (1 - t) * 1 + t * c_i^rfa  =  1 + t * (c_i^rfa - 1),      t in [0, 1]

c(0) is all ones, the exact identity; c(1) is rfa's own coefficient vector, read back from the
transformed norms the way measure_admission.py:152 already reads it rather than recomputed by a
second code path. Two facts about this path are what make it an instrument:

  * Krum's decision is a step function of t; the coefficient share is continuous in t. The argmin of
    a finite set of continuous scores changes only at isolated t. So at the SMALLEST t on a grid
    where the selection flips, call it t*, the adjacent pair

        arm A at t* - h        arm B at t*

    differs in Krum's selection BY CONSTRUCTION -- that is what "flip" means and it is asserted per
    round, not assumed -- while its coefficient vectors differ by exactly h * (c^rfa - 1). The share
    gap across the pair is therefore O(h) and is driven below any tolerance by shrinking h.

  * The construction is UNIFORMLY epsilon-neutral, which is strictly stronger than neutral for one
    adversary set. The coefficient difference h * (c^rfa - 1) is the same vector whatever the
    adversary mask is, so EVERY adversary subset's share moves by O(h). Adversary identity is read
    nowhere: c^rfa comes from the update stack alone, the grid is fixed in advance, and the flip
    search compares selections rather than labels.

WHAT THIS DOES NOT CONTRADICT. App. D.6 proves no oracle-free member of the positive per-client
rescaling class is both EXACTLY share-neutral and informative: oracle-freeness forces share
preservation for every possible adversary set, which forces c constant, which leaves Krum's argmin
invariant. The blend is NOT exactly neutral -- the gap is O(h) > 0 -- so it is consistent with that
theorem rather than a counterexample to it. What it shows is the tolerance-parameterized version:
APPROXIMATE uniform neutrality is achievable at any tolerance, and the price is paid in
informativeness, which is confined to rounds where a flip lies on the path. That price is measured,
not waved at: n_flip_rounds is recorded per run, and rounds with no flip on the path contribute a
bit-identical pair and therefore no dose at all.

WHAT THE CONTRAST IS, STATED SO IT CANNOT BE OVERREAD. Arms A and B are both nearly-rfa stacks at
adjacent points on the blend path. The contrast is therefore LOCAL, at the decision boundary: "just
below this round's first selection flip" against "just above it". It is not "identity against a
statistic disturbance", and it must not be reported as one. What it isolates is the decision channel,
because everything else across the pair is held to O(h) by construction.

COST. Krum's scores at any t follow in closed form from the raw Gram matrix G = U U^T, computed once
per round:

    ||c_i u_i - c_j u_j||^2 = c_i^2 G_ii - 2 c_i c_j G_ij + c_j^2 G_jj

so each grid point costs O(K^2) scalar operations and a 1001-point search is free. The Gram path is
never trusted on its algebra: gram_krum_selection is asserted equal to the shipped krum_selection at
t=0 and t=1 on live stacks, with krum_selection imported from experiments/verify_cos_invariance.py --
the same import measure_admission_oracle_free.py uses -- rather than reimplemented here.

This module defines the construction and its self-checks only. It runs no ladder and writes no
artifact; experiments/run_oracle_free_decomposition.py is the runner that uses it.

Self-check:  PYTHONPATH=. python3 -m experiments.boundary_blend --check
"""

import itertools, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, base)
from experiments.run_all_compositions import apply_d1_transform  # noqa: E402
# The shipped statistic, imported rather than mirrored, so the Gram shortcut is validated against the
# same function that types every published Krum row.
from experiments.verify_cos_invariance import krum_selection, flatten  # noqa: E402
# Mode S's own tolerance, imported and never restated here, so a neutrality claim made with it means
# the same thing it means everywhere else in the paper.
from experiments.measure_admission import SHARE_TOL  # noqa: E402

# Grid spacing. 1001 points over [0, 1] gives h = 1e-3, so the coefficient gap across the reported
# pair is 1e-3 * (c^rfa - 1) and the realized share gap is smaller than that by the share's own
# derivative. This constant is the instrument's tolerance dial and belongs in the pre-registration
# before any ASR exists; it is defined here so that both the runner and the freeze cite one value.
GRID_POINTS = 1001
H = 1.0 / (GRID_POINTS - 1)

# Step cap for the shipped-predicate bisection in refined_boundary_pair. Each step costs one shipped
# krum_selection on a materialized stack (measured 66.0 ms at D = 1,117,354), so the cap bounds the
# arm's search overhead: 40 runs x 50 rounds x 40 steps is the worst case and the measured typical is
# 14-15 steps. The cap is a cost bound, not a tolerance -- the tolerance is SHARE_TOL, and a round
# that exhausts the cap without reaching it FAILS the share gate rather than being let through.
SHIPPED_BISECT_MAX_STEPS = 40

# The bisection's stopping rule is set on the GATED quantity itself -- the mask-free supremum share gap
# -- rather than on a proxy for it, and it aims a factor of 10 under the gate so the instrument has
# margin instead of sitting on its own threshold. SHARE_TOL is imported; only the divisor is new.
SHARE_GAP_TARGET_DIVISOR = 10


def rfa_coefficients(ups, tau=5.0):
    """c^rfa, read back from the transformed norms rather than recomputed.

    Reads the coefficient the way measure_admission.py:152 does -- transformed norm over raw norm,
    per client -- so the vector this module blends toward is the one the shipped rfa branch actually
    applied. A second implementation of Weiszfeld here could drift from apply_d1_transform's and the
    blend would then interpolate toward something the paper never ran.
    """
    raw = flatten(ups)
    t = apply_d1_transform(ups, "rfa", tau=tau, dose_key=None, adv_mask=None)
    st = flatten(t)
    c = (st.norm(dim=1) / raw.norm(dim=1).clamp(min=1e-12))
    return c.detach().cpu().double().numpy(), raw


def blend(c_rfa, t):
    """c(t) = 1 + t (c^rfa - 1). Exactly ones at t=0; exactly c^rfa at t=1."""
    return 1.0 + float(t) * (np.asarray(c_rfa, dtype=np.float64) - 1.0)


def gram_matrix(raw):
    """G = U U^T in float64, from the raw stack. Computed once per round."""
    U = raw.detach().cpu().double()
    return (U @ U.T).numpy()


def gram_krum_selection(G, c):
    """Krum's selected index at coefficient vector c, from the Gram matrix alone.

    Reproduces krum_selection(stack, cosine=False) exactly in structure: pairwise Euclidean
    distances, f = max(1, n // 5), each row sorted with its own zero dropped and the next n - f - 1
    taken, argmin over the row sums. The only difference is that the distances come from G rather
    than from cdist on a materialized scaled stack, which is what makes a 1001-point grid free.
    """
    n = G.shape[0]
    d = np.diag(G) * c * c
    D2 = d[:, None] + d[None, :] - 2.0 * np.outer(c, c) * G
    D = np.sqrt(np.clip(D2, 0.0, None))
    np.fill_diagonal(D, 0.0)
    f = max(1, n // 5)
    scores = [float(np.sort(D[i])[1:n - f].sum()) for i in range(n)]
    return int(min(range(n), key=lambda i: scores[i])), scores


def first_flip(G, c_rfa, grid_points=GRID_POINTS):
    """The smallest grid t at which Krum's selection differs from its t=0 selection.

    Returns (t_star, sel_low, sel_high, n_evaluated) or (None, sel0, sel0, n) when no flip lies on
    the path. A round with no flip yields a bit-identical pair and therefore no dose; that outcome is
    reported rather than dropped, because the count of flip-rounds is what the instrument's power
    rests on.

    The search is over selections, never over adversary labels, which is why the construction stays
    oracle-free.
    """
    ts = np.linspace(0.0, 1.0, grid_points)
    sel0, _ = gram_krum_selection(G, blend(c_rfa, 0.0))
    for i in range(1, grid_points):
        sel, _ = gram_krum_selection(G, blend(c_rfa, ts[i]))
        if sel != sel0:
            return float(ts[i]), sel0, sel, i + 1
    return None, sel0, sel0, grid_points


def boundary_pair(ups, tau=5.0, grid_points=GRID_POINTS):
    """The (A, B) coefficient vectors for one round, plus the receipts that type the round.

    A sits at t* - h and B at t*, so B's selection differs from A's by construction on a flip round
    and the two are identical on a non-flip round. Everything a verdict could rest on is returned
    rather than printed: the realized selections, the coefficient gap, and whether this round carries
    a dose at all.
    """
    c_rfa, raw = rfa_coefficients(ups, tau=tau)
    G = gram_matrix(raw)
    h = 1.0 / (grid_points - 1)
    t_star, sel_low, sel_high, n_eval = first_flip(G, c_rfa, grid_points)
    if t_star is None:
        c = blend(c_rfa, 1.0)
        sel, _ = gram_krum_selection(G, c)
        return {"flip": False, "t_star": None, "h": h,
                "c_A": c.tolist(), "c_B": c.tolist(),
                "sel_A": sel, "sel_B": sel,
                "coeff_gap_linf": 0.0, "n_grid_evaluated": n_eval,
                "c_rfa": c_rfa.tolist(),
                "note": ("no selection flip on the blend path for this round: both arms are the "
                         "same coefficient vector, so this round carries no dose")}
    t_a = t_star - h
    c_a, c_b = blend(c_rfa, t_a), blend(c_rfa, t_star)
    sel_a, _ = gram_krum_selection(G, c_a)
    sel_b, _ = gram_krum_selection(G, c_b)
    return {"flip": True, "t_star": float(t_star), "h": h,
            "c_A": c_a.tolist(), "c_B": c_b.tolist(),
            "sel_A": int(sel_a), "sel_B": int(sel_b),
            "coeff_gap_linf": float(np.max(np.abs(c_b - c_a))),
            "n_grid_evaluated": n_eval,
            "c_rfa": c_rfa.tolist(),
            "note": "arms straddle this round's first selection flip"}


def share_gap_sup(c_a, c_b):
    """The supremum of |share_B(S) - share_A(S)| over every nonempty proper adversary subset S.

    Needs no adversary mask, which is the point: it bounds the realized gap for whatever the true
    adversary set is, so a neutrality claim made with it is UNIFORM rather than conditional on one
    mask. With K = 5 there are 30 such subsets, so the enumeration is exact rather than sampled.
    """
    a = np.asarray(c_a, dtype=np.float64); b = np.asarray(c_b, dtype=np.float64)
    ta, tb = float(a.sum()), float(b.sum())
    if ta <= 0 or tb <= 0:
        return float("nan"), None
    worst, arg = 0.0, None
    for r in range(1, len(a)):
        for S in itertools.combinations(range(len(a)), r):
            g = abs(sum(b[i] for i in S) / tb - sum(a[i] for i in S) / ta)
            if g > worst:
                worst, arg = float(g), list(S)
    return worst, arg


def shipped_selection(ups, c):
    """Krum's selected index at coefficient vector c, through the SHIPPED statistic on a materialized
    float32 stack. This is the function that types every published Krum row in this paper."""
    s, _ = krum_selection(flatten(apply_coefficients(ups, c)), False)
    return int(s)


def refined_boundary_pair(ups, tau=5.0, share_tol=SHARE_TOL,
                          gap_target_divisor=SHARE_GAP_TARGET_DIVISOR,
                          max_steps=SHIPPED_BISECT_MAX_STEPS, coarse_points=GRID_POINTS):
    """The boundary pair found by bisecting the SHIPPED float32 predicate, not the float64 Gram grid.

    WHY THIS EXISTS, AND IT IS A CORRECTION. boundary_pair above locates the flip on a fixed
    1001-point float64 Gram grid, so its pair straddles the crossing of the FLOAT64 statistic. The
    shipped statistic is float32 and its crossing sits at a slightly different t, which has two
    measured consequences at this cell (results/oracle_free_decomposition_premise.json):

      * at the fixed grid's h = 1e-3 the pair's coefficient gap is ~3e-4 and the adversarial share
        gap exceeds SHARE_TOL on 11 of 12 dosed adversary rounds, with the mask-free supremum
        exceeding it on 13 of 13. The arm would be confounded in the attenuation channel -- the exact
        defect App. D.7 disclosed about itself.
      * shrinking h while still bracketing with the float64 predicate does NOT fix it: the pair then
        straddles the wrong crossing and the shipped selection stops differing at all, on 13 of 13
        rounds. That is not a resolution floor, it is a wrong search predicate.

    Bisecting the shipped predicate makes the selection difference true BY CONSTRUCTION at every
    step, at any tolerance. The stopping rule is the gated quantity itself rather than a proxy for
    it: bisect until the mask-free supremum share gap is at or below share_tol / gap_target_divisor.
    Measured across 5 seeds x 5 rounds, both requirements then hold on 8 of 8 flip rounds at a
    supremum gap of ~1.5e-08 against a tolerance of 1e-06.

    The pair straddles A selection crossing, not necessarily the first one on the path: bisection on
    a predicate that is not monotone in t converges to some crossing, and which one is not claimed.
    What is asserted per round is that the two arms' selections differ under the shipped statistic and
    that their coefficients are uniformly share-neutral to tolerance.
    """
    c_rfa, raw = rfa_coefficients(ups, tau=tau)
    target = share_tol / float(gap_target_divisor)
    sel0 = shipped_selection(ups, blend(c_rfa, 0.0))
    sel1 = shipped_selection(ups, blend(c_rfa, 1.0))
    n_shipped = 2
    lo, hi = None, None
    if sel1 != sel0:
        # A crossing is guaranteed on [0, 1] by the shipped predicate itself.
        lo, hi = 0.0, 1.0
    else:
        # The endpoints agree, so any crossing comes in pairs. Use the free float64 Gram scan to
        # PROPOSE a bracket, then require the shipped predicate to confirm it before bisecting. A
        # bracket the shipped statistic does not confirm is not a dose.
        G = gram_matrix(raw)
        t_star, _, _, _ = first_flip(G, c_rfa, coarse_points)
        if t_star is not None:
            h = 1.0 / (coarse_points - 1)
            a, b = max(0.0, t_star - h), t_star
            sa = shipped_selection(ups, blend(c_rfa, a))
            sb = shipped_selection(ups, blend(c_rfa, b))
            n_shipped += 2
            if sa != sb:
                lo, hi = a, b
    if lo is None:
        c = blend(c_rfa, 1.0)
        return {"flip": False, "t_star": None, "c_A": c.tolist(), "c_B": c.tolist(),
                "sel_A": sel0, "sel_B": sel0, "coeff_gap_linf": 0.0,
                "share_gap_sup": 0.0, "argmax_subset": None, "n_bisect_steps": 0,
                "n_shipped_evaluations": n_shipped, "c_rfa": c_rfa.tolist(),
                "search": "shipped_predicate_bisection",
                "note": ("no selection crossing the shipped statistic confirms: both arms are the "
                         "same coefficient vector, so this round carries no dose")}
    sel_lo = shipped_selection(ups, blend(c_rfa, lo)); n_shipped += 1
    steps = 0
    sup, arg = share_gap_sup(blend(c_rfa, lo), blend(c_rfa, hi))
    while sup > target and steps < max_steps:
        mid = 0.5 * (lo + hi)
        s = shipped_selection(ups, blend(c_rfa, mid)); n_shipped += 1
        if s == sel_lo:
            lo = mid
        else:
            hi = mid
        steps += 1
        sup, arg = share_gap_sup(blend(c_rfa, lo), blend(c_rfa, hi))
    c_a, c_b = blend(c_rfa, lo), blend(c_rfa, hi)
    sel_a = shipped_selection(ups, c_a); sel_b = shipped_selection(ups, c_b); n_shipped += 2
    return {"flip": True, "t_star": float(hi), "t_lo": float(lo),
            "c_A": c_a.tolist(), "c_B": c_b.tolist(),
            "sel_A": sel_a, "sel_B": sel_b,
            "coeff_gap_linf": float(np.max(np.abs(c_b - c_a))),
            "share_gap_sup": float(sup), "argmax_subset": arg,
            "n_bisect_steps": steps, "n_shipped_evaluations": n_shipped,
            "c_rfa": c_rfa.tolist(), "search": "shipped_predicate_bisection",
            "target_share_gap_sup": target,
            "note": "arms straddle a selection crossing of the shipped float32 statistic"}


def apply_coefficients(ups, c):
    """Scale each client's update by its coefficient, the same per-client form every d1 in this
    paper takes (Proposition 11's class). Returns a new list; `ups` is not mutated."""
    out = []
    for i, u in enumerate(ups):
        w = float(c[i])
        out.append({k: u[k] * w for k in u})
    return out


def adv_share(c, adv_rows):
    """The adversarial share of coefficient mass. Used only to MEASURE the pair's neutrality after
    the fact; the construction above never calls it, which is what keeps the blend oracle-free."""
    c = np.asarray(c, dtype=np.float64)
    tot = float(c.sum())
    if tot <= 0 or not adv_rows:
        return float("nan")
    return float(sum(c[a] for a in adv_rows) / tot)


def self_check(seeds=(42,), rounds=3, verbose=True, pair_fn=None):
    """Assert the Gram shortcut against the shipped krum_selection on live update stacks.

    pair_fn selects WHICH pair is checked, and the default is the superseded one. Pass
    refined_boundary_pair to check the instrument the ladder uses; pass nothing (or boundary_pair) to
    check the fixed-grid pair that bed6562 froze. The endpoint legs -- 1, 2 and 4 below -- are about the
    path and hold either way; leg 3 is about the pair and its verdict depends on which pair was asked
    for, which is exactly why the caller must choose rather than inherit.

    Four things are asserted, each against a computation this module does not own:

      1. gram_krum_selection at t=0 equals krum_selection(raw), the untransformed selection.
      2. gram_krum_selection at t=1 equals krum_selection(rfa stack), the shipped transform's
         selection. Together with 1 this pins both ends of the path to the real statistic.
      3. On a flip round the two arms' selections differ, asserted through the SHIPPED
         krum_selection on materialized scaled stacks rather than through the Gram scores that
         chose the pair. A flip found only by the shortcut and not reproduced by the real
         statistic is a shortcut bug, and this is what would catch it.
      4. c(0) is exactly ones, so the t=0 end is the identity rather than approximately it.
    """
    from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
    from attacks import get_attack
    if pair_fn is None:
        pair_fn = boundary_pair
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows, ok = [], True
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        cd, _, nc = get_federated_dataset("cifar10", 10, 0.5, seed)
        srv = FederatedServer(get_model("cifar_cnn", nc), dev)
        atk = get_attack("backdoor_pixel")
        adv = set(range(2))
        cl = [FederatedClient(i, atk.poison_dataset(cd[i]) if i in adv else cd[i], dev)
              for i in range(10)]
        for rnd in range(rounds):
            pids = np.random.choice(10, 5, replace=False)
            ups = []
            for cid in pids:
                u = cl[cid].train(srv.global_model, 1, 0.01, 64)
                if cid in adv:
                    u = atk.manipulate_update(u, srv.global_model)
                ups.append(u)
            adv_rows = [i for i, cid in enumerate(pids) if cid in adv]
            c_rfa, raw = rfa_coefficients(ups)
            G = gram_matrix(raw)

            sel_raw_shipped, _ = krum_selection(raw, False)
            sel_raw_gram, _ = gram_krum_selection(G, blend(c_rfa, 0.0))
            rfa_stack = flatten(apply_d1_transform(ups, "rfa", tau=5.0, dose_key=None,
                                                  adv_mask=None))
            sel_rfa_shipped, _ = krum_selection(rfa_stack, False)
            sel_rfa_gram, _ = gram_krum_selection(G, blend(c_rfa, 1.0))

            pair = pair_fn(ups)
            # Assertion 3 goes through the shipped statistic on materialized stacks, not through the
            # Gram scores that chose the pair.
            sel_a_ship, _ = krum_selection(flatten(apply_coefficients(ups, pair["c_A"])), False)
            sel_b_ship, _ = krum_selection(flatten(apply_coefficients(ups, pair["c_B"])), False)

            r = {
                "seed": int(seed), "round": int(rnd),
                "t0_gram_equals_shipped_raw": bool(sel_raw_gram == sel_raw_shipped),
                "t1_gram_equals_shipped_rfa": bool(sel_rfa_gram == sel_rfa_shipped),
                "identity_is_exact_ones": bool(np.all(blend(c_rfa, 0.0) == 1.0)),
                "flip": pair["flip"], "t_star": pair["t_star"],
                "coeff_gap_linf": pair["coeff_gap_linf"],
                "sel_A_shipped": int(sel_a_ship), "sel_B_shipped": int(sel_b_ship),
                "pair_selections_differ_under_shipped_statistic": bool(sel_a_ship != sel_b_ship),
                "share_A": adv_share(pair["c_A"], adv_rows),
                "share_B": adv_share(pair["c_B"], adv_rows),
                "n_adv_in_round": len(adv_rows),
                "rho_rfa": float(np.max(c_rfa) / max(np.min(c_rfa), 1e-12)),
            }
            r["share_gap"] = (abs(r["share_B"] - r["share_A"])
                              if not (np.isnan(r["share_A"]) or np.isnan(r["share_B"]))
                              else float("nan"))
            # The mask-free supremum is recorded beside the realized gap rather than instead of it, so
            # the two can be compared. The realized gap can pass while the supremum fails, which means
            # neutral for the mask this round happened to draw rather than neutral.
            sup, sup_arg = share_gap_sup(pair["c_A"], pair["c_B"])
            r["share_gap_sup"] = float(sup) if sup == sup else float("nan")
            r["argmax_subset"] = sup_arg
            r["sup_within_tol"] = bool(sup <= SHARE_TOL) if sup == sup else True
            r["pair_fn"] = pair_fn.__name__
            # A non-flip round is allowed to have identical selections; a flip round is not.
            legs = [r["t0_gram_equals_shipped_raw"], r["t1_gram_equals_shipped_rfa"],
                    r["identity_is_exact_ones"]]
            if pair["flip"]:
                legs.append(r["pair_selections_differ_under_shipped_statistic"])
            r["all_legs_pass"] = bool(all(legs))
            ok = ok and r["all_legs_pass"]
            rows.append(r)
            if verbose:
                print(f"  seed {seed} rnd {rnd}: t*={r['t_star']} flip={r['flip']} "
                      f"gram==shipped at t0/t1: {r['t0_gram_equals_shipped_raw']}/"
                      f"{r['t1_gram_equals_shipped_rfa']}  "
                      f"sel A/B={r['sel_A_shipped']}/{r['sel_B_shipped']}  "
                      f"share gap={r['share_gap']:.3e}  coeff gap={r['coeff_gap_linf']:.3e}")
            srv.apply_update(srv.aggregate(ups))
    n_flip = sum(1 for r in rows if r["flip"])
    gaps = [r["share_gap"] for r in rows if r["flip"] and not np.isnan(r["share_gap"])]
    sups = [r["share_gap_sup"] for r in rows if r["flip"] and not np.isnan(r["share_gap_sup"])]
    return {"all_pass": bool(ok), "n_rows": len(rows), "n_flip_rounds": n_flip,
            "pair_fn": pair_fn.__name__,
            "max_share_gap_on_flip_rounds": float(max(gaps)) if gaps else float("nan"),
            "max_share_gap_sup_on_flip_rounds": float(max(sups)) if sups else float("nan"),
            "all_flip_rounds_within_tol_on_supremum": bool(all(s <= SHARE_TOL for s in sups))
                                                       if sups else None,
            "all_flip_rounds_differ_under_shipped": bool(all(
                r["pair_selections_differ_under_shipped_statistic"] for r in rows if r["flip"]))
                                                     if n_flip else None,
            "grid_points": GRID_POINTS, "h": H, "per_round": rows,
            "claim": ("The Gram-based selection equals the shipped krum_selection at both ends of "
                      "the blend path; the identity end is exactly ones; and on every flip round "
                      "the reported pair's selections differ under the shipped statistic while its "
                      "coefficient vectors differ by at most h * max|c_rfa - 1|.")}


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("=== boundary blend self-check (Gram shortcut against the shipped statistic) ===")
        print(f"grid {GRID_POINTS} points, h = {H:g}")
        res = self_check()
        print(f"\nall_pass={res['all_pass']}  flip rounds {res['n_flip_rounds']}/{res['n_rows']}  "
              f"max share gap on flip rounds {res['max_share_gap_on_flip_rounds']:.3e}")
    else:
        print(__doc__)
