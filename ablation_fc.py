"""
Ablation under the FIXED-CONFIDENCE protocol.

The shipped ablation (`run_simulation.py`) scores F1 against sample count. This drives
the same algorithm classes from `algorithms.py` with the stopping rule of this paper
instead: the exact subset GLR with the time-uniform threshold beta(V, delta) of
thm:beta, identical for every variant, so only the DESIGN RULE varies. This is the
same isolation the manuscript already uses for the oracle/ecc comparison.

Variants (all from the authors' own supplementary code, unmodified):
    ECC_AHT              full method (pairwise QP design)
    ECC_AHT_SimpleDiff   No-QP: selects streams but uses a plain difference vector
    RandomSparseProjection  No-Active: action chosen at random
    ECC_AHT_Diagonal     No-Correlation: ignores Sigma in the design
    ECC_AHT_CostFree     budget-free upper reference

Run:  python ablation_fc.py --sigma identity --delta 0.05 --seeds 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SUPP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "30725_Active_Hypothesis_Testin_Supplementary Material", "ECC-AHT")
sys.path.insert(0, SUPP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_rate_validation import SIGNAL, all_alternatives, beta_threshold, make_sigma

VARIANTS = ("ECC_AHT", "ECC_AHT_SimpleDiff", "RandomSparseProjection",
            "ECC_AHT_Diagonal", "ECC_AHT_CostFree",
            # baselines: same BaseAlgorithm interface, so the same harness drives them
            "BaseArm_CombGapE", "TTTS_Challenger")


def load_classes():
    import algorithms as A
    out = {}
    for name in VARIANTS:
        cls = getattr(A, name, None)
        if cls is not None:
            out[name] = cls
        else:
            print(f"  [skip] {name} not present in algorithms.py")
    return out


def trial(cls, Sigma, U, s_star, K, n, B, delta, rng, max_steps, burn):
    mu_true = np.zeros(K)
    mu_true[list(s_star)] = SIGNAL
    # mu_signal must be a SCALAR: algorithms.py:143 does delta_t[i_star] =
    # self.delta_signal (a scalar slot), while update() does C_t * self.delta_signal
    # (elementwise). Only a scalar satisfies both, and it matches the uniform-Delta
    # model used throughout this paper. Passing a length-K vector raises
    # "setting an array element with a sequence".
    algo = cls(K, n, Sigma, B, mu_signal=SIGNAL)
    n_alts = len(U)
    L = np.zeros(n_alts)
    V = np.zeros(n_alts)
    for t in range(1, max_steps + 1):
        c = np.asarray(algo.select_action(), dtype=float).ravel()
        if c.size != K or not np.any(c):
            return None, None
        var = max(float(c @ Sigma @ c), 1e-12)
        y = rng.normal(float(c @ mu_true), np.sqrt(var))
        algo.update(c, y)

        b = U @ c                                # (c' u_{S'}) for every alternative
        a = y - float(c @ mu_true)
        L += (b ** 2 - 2.0 * a * b) / (2.0 * var)
        V += b ** 2 / var
        if t > burn and float((L - beta_threshold(V, delta, n_alts)).min()) >= 0.0:
            # DECISION = the exact GLR's argmax, NOT algo.get_S_hat().
            #
            # The stopping rule certifies S* via L_t(S*, S') for every S'. Scoring the
            # algorithm's own marginal belief instead conflates two statistics: a
            # trace showed paths where the rule fires correctly and get_S_hat() is
            # confidently wrong, because the per-stream independent beliefs and the
            # subset GLR can disagree. That is a real property of the algorithm, but it
            # is not what an ablation of the DESIGN RULE should measure. Holding the
            # decision rule fixed (ML of the GLR: min_i L_i >= 0) leaves the design as
            # the only varying factor, which is the point of the ablation.
            return t, bool(L.min() >= 0.0)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sigma", default="identity")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--B", type=float, default=2.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--s-star", default="0,1")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=5)
    args = ap.parse_args()

    Sigma = make_sigma(args.K, args.sigma, args.rho)
    s_star = np.array(sorted(int(x) for x in args.s_star.split(",")))
    U = all_alternatives(args.K, args.n, s_star, signal=SIGNAL)
    classes = load_classes()
    if not classes:
        print("no algorithm classes importable -- check numpy/cvxpy versions")
        return

    print(f"\nK={args.K} n={args.n} sigma={args.sigma}({args.rho}) B={args.B} "
          f"delta={args.delta} seeds={args.seeds} |A|={len(U)}")
    print(f"{'variant':<24}{'E[tau]':>9}{'err':>8}{'timeouts':>10}")
    for name in VARIANTS:
        if name not in classes:
            continue
        taus, errs, to = [], [], 0
        for s in range(args.seeds):
            rng = np.random.default_rng(7000 + s)
            try:
                tau, ok = trial(classes[name], Sigma, U, s_star, args.K, args.n,
                                args.B, args.delta, rng, args.max_steps, args.burn)
            except Exception as e:
                print(f"  {name}: FAILED -- {type(e).__name__}: {e}")
                taus = []
                break
            if tau is None:
                to += 1
                continue
            taus.append(tau)
            errs.append(0.0 if ok else 1.0)
        if taus:
            print(f"{name:<24}{np.mean(taus):>9.1f}{np.mean(errs):>8.3f}{to:>10}")


if __name__ == "__main__":
    main()
