#!/usr/bin/env python3
"""Randomized verification of the three inequalities behind Corollary (checkable
separation margin) and the vacuous rho-reduction remark in the appendix.

These are pure geometry claims about the geometric median, independent of any FL
run, so they are checked here on random configurations rather than on artifacts:

  (A)  (n_b - n_a) * ||mu' - mu'_B||  <=  2 * sum_b ||u'_b - mu'_B||      [Corollary]
  (B1) Delta'_sep  >=  w_min * [ Delta_sep - (rho - 1) * N_A ]            [Step B]
  (B2) R'_B        <=  2 * w_min * [ rho * R_B + (rho - 1) * ||mu_B|| ]   [Step B]

A single violation means the derivation in the appendix is wrong.  The reported
slack on (A) is bound/realised under random placement; it is large because the
realised displacement is set by a force balance among the residuals, not by the
worst case the bound must cover.  It is a validity check, not a tightness claim.

Usage:  python3 experiments/verify_margin_inequalities.py [n_trials]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch

from measure_transformed_theorem_quantities import weiszfeld

SEED = 0
TOL = 1e-6  # Weiszfeld is run to 1e-6, so allow that much numerical slop


def trial(rng):
    d = int(rng.integers(2, 40))
    n_b = int(rng.integers(2, 12))
    n_a = int(rng.integers(1, n_b))            # honest majority, as assumed
    scale = float(10.0 ** rng.uniform(-2, 2))  # exercise the scale-free claims

    u_b = torch.tensor(rng.normal(0, scale, (n_b, d)), dtype=torch.float64)
    # place adversaries anywhere, near or far: the bounds must not assume separation
    u_a = torch.tensor(rng.normal(rng.uniform(-3, 3) * scale, scale, (n_a, d)),
                       dtype=torch.float64)
    w = torch.tensor(rng.uniform(0.05, 1.0, n_b + n_a), dtype=torch.float64)

    raw = torch.cat([u_b, u_a], 0)
    tr = raw * w.unsqueeze(1)
    tb, ta = tr[:n_b], tr[n_b:]

    mu_B_p = weiszfeld(tb)
    mu_all_p = weiszfeld(tr)
    resid_p = torch.norm(tb - mu_B_p.unsqueeze(0), dim=1)
    S_B_p, R_B_p = resid_p.sum().item(), resid_p.max().item()
    delta_p = torch.norm(mu_all_p - mu_B_p).item()
    Delta_sep_p = torch.cdist(ta, tb).min().item()

    mu_B = weiszfeld(u_b)
    R_B = torch.norm(u_b - mu_B.unsqueeze(0), dim=1).max().item()
    Delta_sep = torch.cdist(u_a, u_b).min().item()
    mu_B_norm = torch.norm(mu_B).item()
    N_A = torch.norm(u_a, dim=1).max().item()
    w_min, rho = w.min().item(), (w.max() / w.min()).item()

    # (A) displacement bound; slack = bound / realised
    lhs_A = (n_b - n_a) * delta_p
    rhs_A = 2.0 * S_B_p
    slack_A = rhs_A / max(lhs_A, 1e-30)

    # (B1) transformed separation lower bound
    rhs_B1 = w_min * (Delta_sep - (rho - 1.0) * N_A)

    # (B2) transformed benign radius upper bound
    rhs_B2 = 2.0 * w_min * (rho * R_B + (rho - 1.0) * mu_B_norm)

    ref = max(R_B_p, Delta_sep_p, 1e-30)
    return {
        'A': lhs_A <= rhs_A * (1 + TOL) + TOL * ref,
        'B1': Delta_sep_p >= rhs_B1 - TOL * ref,
        'B2': R_B_p <= rhs_B2 * (1 + TOL) + TOL * ref,
        'slack_A': slack_A,
        'n_a': n_a, 'n_b': n_b,
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rng = np.random.default_rng(SEED)
    bad = {'A': [], 'B1': [], 'B2': []}
    slacks = []
    for i in range(n_trials):
        r = trial(rng)
        for k in bad:
            if not r[k]:
                bad[k].append((i, r['n_a'], r['n_b']))
        slacks.append(r['slack_A'])

    print(f"seed={SEED}  trials={n_trials}  (float64, Weiszfeld tol 1e-6)")
    for k, name in [('A', 'Corollary displacement bound  (n_b-n_a)d <= 2S'),
                    ('B1', "Step B lower bound on Delta'_sep            "),
                    ('B2', "Step B upper bound on R'_B                  ")]:
        n_v = len(bad[k])
        print(f"  {name}: {n_trials - n_v}/{n_trials} hold"
              + (f"   *** {n_v} VIOLATIONS: {bad[k][:5]} ***" if n_v else ""))
    print(f"\n  (A) bound/realised slack: min {min(slacks):.3f}x  "
          f"median {np.median(slacks):.2f}x  max {max(slacks):.1f}x")
    print("  slack is bound/realised: large values mean the bound is valid and "
          "conservative here, not loose in the worst case.")

    if any(bad.values()):
        sys.exit("DERIVATION IS WRONG -- re-derive before citing these bounds")
    print("\nall three inequalities hold on every trial")


if __name__ == '__main__':
    main()
