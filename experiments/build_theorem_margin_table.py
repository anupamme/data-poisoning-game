#!/usr/bin/env python3
"""Emit every number the checkable-margin corollary and its remark report.

Reads only frozen artifacts (results/theorem_margins*.json) and prints the
figures verbatim as they must appear in main.tex.  Nothing is transcribed by hand.

Usage:  python3 experiments/build_theorem_margin_table.py
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = {0.5: 'results/theorem_margins.json',
       0.1: 'results/theorem_margins_alpha0.1.json'}


def load(path):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        return None
    return json.load(open(full))


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    missing = [p for p in SRC.values() if load(p) is None]
    if missing:
        sys.exit(f"missing artifact(s): {missing}; run experiments/measure_theorem_margins.py first")

    print("=" * 74)
    print("COROLLARY (checkable separation margin) -- Step A")
    print("=" * 74)
    for alpha in sorted(SRC, reverse=True):
        d = load(SRC[alpha])
        pos = [r for r in d['per_round'] if not r['w_adv_is_zero']]
        n = len(pos)
        print(f"\n  alpha = {alpha}   ({d['n_rounds_total']} rounds, {n} with w_a > 0)")
        print(f"    delta' <= C_A * Rbar'_B holds in            "
              f"{sum(1 for r in pos if r['delta_bound_mean_holds'])}/{n} rounds")
        print(f"    bound/realised slack                       "
              f"min {d['delta_bound_slack_min']:.1f}x, mean {d['delta_bound_slack_mean']:.0f}x")
        print(f"    Rbar'_B / R'_B                             "
              f"mean {d['Rbar_over_R_B_prime_mean']:.3f} "
              f"[{min(r['Rbar_over_R_B_prime'] for r in pos):.3f}, "
              f"{max(r['Rbar_over_R_B_prime'] for r in pos):.3f}]")
        print(f"    condition met, mean-radius form            {d['stepA_mean_radius_met']}/{n}")
        print(f"    condition met, max-radius form             {d['stepA_max_radius_met']}/{n}")
        for na, g in sorted(d['per_na'].items()):
            print(f"      n_a={na} (n_b={g['n_benign']}, C_A={g['C_A']:.3f}): "
                  f"mean-radius {g['stepA_mean_radius_met']}/{g['n_rounds']}, "
                  f"max-radius {g['stepA_max_radius_met']}/{g['n_rounds']}")
        ratios = [r['Delta_sep_prime'] / r['R_B_prime'] for r in pos]
        print(f"    measured Delta'_sep / R'_B                 "
              f"[{min(ratios):.2f}, {max(ratios):.2f}], mean {mean(ratios):.2f}")

    print()
    print("=" * 74)
    print("REMARK (reduction to rho alone) -- Step B: VERDICT")
    print("=" * 74)
    for alpha in sorted(SRC, reverse=True):
        d = load(SRC[alpha])
        pos = [r for r in d['per_round'] if not r['w_adv_is_zero']]
        n = len(pos)
        rs = [r['stepB_rho_star'] for r in pos]
        finite = [x for x in rs if x > -1e30]
        print(f"\n  alpha = {alpha}")
        print(f"    rho* > 1 in                                "
              f"{sum(1 for x in rs if x > 1.0)}/{n} rounds")
        print(f"    vacuous (rho* <= 1: no rho satisfies it)   {d['stepB_vacuous_rounds']}/{n} rounds")
        print(f"    condition met (rho < rho*)                 {d['stepB_met']}/{n} rounds")
        if finite:
            print(f"    rho* range                                 "
                  f"[{min(finite):.3f}, {max(finite):.3f}]")
        print(f"    measured rho                               "
              f"[{min(r['rho'] for r in pos):.3f}, {max(r['rho'] for r in pos):.3f}]")
        need = [r['stepB_B_const'] * r['R_B'] for r in pos]
        have = [r['Delta_sep'] for r in pos]
        print(f"    Step B needs Delta_sep > B*R_B; ratio       "
              f"Delta_sep/(B*R_B) in [{min(h/x for h, x in zip(have, need)):.3f}, "
              f"{max(h/x for h, x in zip(have, need)):.3f}] (must exceed 1)")
        for na in sorted(set(r['n_adv_in_round'] for r in pos)):
            g = [r['Delta_sep'] / (r['stepB_B_const'] * r['R_B'])
                 for r in pos if r['n_adv_in_round'] == na]
            print(f"      n_a={na} (C_A="
                  f"{[r['C_A'] for r in pos if r['n_adv_in_round'] == na][0]:.3f}): "
                  f"ratio max {max(g):.3f}, mean {mean(g):.3f}, n={len(g)}")

    print()
    print("=" * 74)
    print("SENTENCES FOR main.tex (copy verbatim)")
    print("=" * 74)
    d5, d1 = load(SRC[0.5]), load(SRC[0.1])
    p5 = [r for r in d5['per_round'] if not r['w_adv_is_zero']]
    g1 = d5['per_na'].get('1')
    g2 = d5['per_na'].get('2')
    print(f"\n  Step A, alpha=0.5: satisfied in {d5['stepA_mean_radius_met']} of {len(p5)} "
          f"positive-weight rounds")
    if g1:
        print(f"    n_a=1: {g1['stepA_mean_radius_met']}/{g1['n_rounds']}, C_A = {g1['C_A']:.3f}, "
              f"requirement Delta'_sep > {2 + 2 * g1['C_A'] * d5['Rbar_over_R_B_prime_mean']:.2f} R'_B "
              f"at the mean Rbar'_B/R'_B")
    if g2:
        print(f"    n_a=2: {g2['stepA_mean_radius_met']}/{g2['n_rounds']}, C_A = {g2['C_A']:.3f} "
              f"(the condition binds here -- report it)")
    print(f"  bound is conservative by min {d5['delta_bound_slack_min']:.1f}x "
          f"(mean {d5['delta_bound_slack_mean']:.0f}x)")
    print(f"  Step B verdict: VACUOUS -- rho* <= 1 in {d5['stepB_vacuous_rounds']}/{len(p5)} rounds "
          f"at alpha=0.5")
    if d1:
        pp1 = [r for r in d1['per_round'] if not r['w_adv_is_zero']]
        print(f"  alpha=0.1 (heterogeneous): Step A satisfied in "
              f"{d1['stepA_mean_radius_met']}/{len(pp1)} positive-weight rounds, "
              f"Step B vacuous in {d1['stepB_vacuous_rounds']}/{len(pp1)}")


if __name__ == '__main__':
    main()
