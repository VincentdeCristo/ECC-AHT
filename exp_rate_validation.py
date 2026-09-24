"""
Rate-validation experiment under the fixed-confidence protocol.

This is the protocol the paper reports, replacing an F1>=0.95 criterion that is
not a fixed-confidence test. It is not a migration of the old protocol.

For each (Sigma, delta) it measures, over many seeds:
    tau        = number of measurements to stop
    err_rate   = P(S_hat != S*), which must be <= delta
    ratio      = tau * Gamma* / log(1/delta), which should approach 1
                 for a procedure attaining the information rate.

Methods:
    oracle    S* known; exact subset GLR; explicit threshold of Thm 2.
              This is the reference curve -- the best any procedure can do.
    ecc       the shipped ECC-AHT: pairwise QP design, pseudo-likelihood beliefs,
              exact-GLR stopping rule with the same threshold.

Gamma* is computed EXACTLY by the SDP of corrected-theory.tex (convex), so the
ratio is not contaminated by a searched lower bound. rank(W*) is reported because
it decides (Prop. 3) whether a single-design surrogate is valid on that Sigma.

CLI (SLURM-array friendly -- one config per invocation):
    python exp_rate_validation.py --sigma identity --delta 0.05 --seeds 40
    python exp_rate_validation.py --selftest

Output: results/{sigma}{rho}_K{K}_d{delta}_{method}.json
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import cvxpy as cp
import numpy as np

SIGNAL = 1.0
B_BUDGET = 2.0
BURN_TRIES = 40          # exploration phase for `ecc` before the GLR rule applies


# --------------------------------------------------------------------------
# problem construction
# --------------------------------------------------------------------------

def make_sigma(K: int, kind: str, rho: float = 0.5) -> np.ndarray:
    idx = np.arange(K)
    if kind == "identity":
        return np.eye(K)
    if kind == "equicorr":
        return (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    if kind == "toeplitz":
        return rho ** np.abs(idx[:, None] - idx[None, :])
    if kind == "rbf":
        return np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2.0 * (rho * K) ** 2))
    raise ValueError(kind)


def all_alternatives(K: int, n: int, s_star: np.ndarray,
                     signal: float = SIGNAL) -> np.ndarray:
    """mu_{S'} - mu_{S*} for every alternative S' (shape: #alts, K).

    `signal` is the per-stream shift Delta. Divergences are homogeneous of degree 2
    in Delta at fixed direction, so Delta rescales Gamma* by Delta^2; only the
    scale-free ratio tau*Gamma*/log(1/delta) should be compared across a Delta sweep.
    """
    target = set(s_star.tolist())
    rows = []
    for combo in combinations(range(K), n):
        if set(combo) == target:
            continue
        v = np.zeros(K)
        for k in combo:
            v[k] = signal
        for k in s_star:
            v[k] -= signal
        rows.append(v)
    return np.array(rows)


def gamma_star_sdp(Sigma: np.ndarray, U: np.ndarray) -> tuple[float, int]:
    """Exact Gamma* and rank(W*), by the convex SDP of corrected-theory.tex."""
    K = Sigma.shape[0]
    Linv = np.linalg.inv(np.linalg.cholesky(Sigma))
    V = U @ Linv.T
    W = cp.Variable((K, K), symmetric=True)
    g = cp.Variable()
    cons = [W >> 0, cp.trace(W) == 1] + [cp.quad_form(v, W) >= g for v in V]
    cp.Problem(cp.Maximize(g), cons).solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    Wv = np.asarray(W.value)
    Wv = (Wv + Wv.T) / 2
    ev = np.linalg.eigvalsh(Wv)
    rank = int((ev > 1e-6 * ev.max()).sum())
    return float(g.value) / 2, rank


# --------------------------------------------------------------------------
# the explicit threshold of Theorem 2
# --------------------------------------------------------------------------

def beta_threshold(V: np.ndarray, delta: float, n_alts: int, tau: float = 1.0) -> np.ndarray:
    """Stopping threshold for the rule 'stop when min_{S'} L_t(S_hat,S') >= beta_t'.

    Under H_{S*}, L_t(S*,S') = V_t/2 - D_t with D_t = sum rho_s z_s a martingale of
    variance V_t and mean 0.  An error (stopping with S_hat != S*) requires
    L_t(S_hat,S*) >= beta_t, i.e. -L_t(S*,S_hat) >= beta_t, i.e.
        D_t >= V_t/2 + beta_t.
    Ville's inequality bounds P(exists t : D_t >= c(V_t, delta')) <= delta' for
        c(V,d) = (1/tau) sqrt(2(1+tau^2 V)[log(1/d) + log(1+tau^2 V)/2]),
    so we need V/2 + beta = c, i.e.

        beta_t = c(V_t, delta/|A|) - V_t/2.

    SIGN MATTERS.  c(V,delta') alone is a LOWER confidence bound on L_t (so
    P(L_t <= V/2 - c) <= delta'), which is the opposite of what a stopping rule
    needs.  Using V/2 - c here makes the threshold NEGATIVE for small V, so the
    rule fires at t = 1 on zero evidence.  Correct limits: beta(0) = c(0,delta') > 0
    (no premature stop) and beta -> -infinity for large V (rule becomes easy)."""
    d = delta / max(n_alts, 1)
    c = (1.0 / tau) * np.sqrt(
        2.0 * (1.0 + tau ** 2 * V) * (np.log(1.0 / d) + 0.5 * np.log(1.0 + tau ** 2 * V)))
    return c - V / 2.0


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def _step_llr(c, y, mu_true, U, Sigma):
    """Exact subset-GLR increments and the V-increments, for every alternative."""
    var = max(float(c @ Sigma @ c), 1e-12)
    a = y - float(c @ mu_true)
    b = U @ c
    return (b ** 2 - 2.0 * a * b) / (2.0 * var), b ** 2 / var


def run_trial(Sigma, U, s_star, n, K, delta, method, rng, max_steps, burn=BURN_TRIES,
              signal=SIGNAL, B=B_BUDGET):
    """One trial. Returns (stopped?, tau, correct?)."""
    mu_true = np.zeros(K)
    mu_true[s_star] = signal
    delta_sig = np.full(K, signal)
    n_alts = len(U)

    log_odds = np.full(K, np.log((n / K) / (1 - n / K)))
    # Ranking statistic. CUSUM and Shiryaev-Roberts baselines:
    # under this paper's assumptions (mu_0, Sigma, Delta, n all known) the exact LLR is
    # available, so those classical detectors are admissible and must be compared
    # against. We keep the design rule and the stopping rule IDENTICAL and swap only
    # the per-stream inference statistic, which isolates that change as the variable.
    #   ecc/pseudo : score = log-odds (the shipped pseudo-likelihood)
    #   cusum      : score = per-stream CUSUM,  S_k <- max(0, S_k + llr_k)
    #   sr         : score = log Shiryaev-Roberts, R_k <- (1+R_k) exp(llr_k)
    score = log_odds.copy()
    cusum = np.zeros(K)
    sr = np.zeros(K)
    L = np.zeros(n_alts)
    V = np.zeros(n_alts)

    c_var = cp.Variable(K)
    d_par = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(c_var, cp.psd_wrap(Sigma))),
                      [d_par @ c_var == 1, cp.norm1(c_var) <= B])

    for t in range(1, max_steps + 1):
        order = np.argsort(score)
        s_hat, rest = order[-n:], order[:-n]
        i_s = int(s_hat[np.argmin(log_odds[s_hat])])
        j_s = int(rest[np.argmax(log_odds[rest])])
        d = np.zeros(K); d[i_s], d[j_s] = signal, -signal
        try:
            d_par.value = d
            prob.solve(solver=cp.OSQP, warm_start=True)
            c = np.asarray(c_var.value).ravel()
            nrm = np.linalg.norm(c, 1)
            c = c / nrm * B if nrm > 1e-9 else d / np.linalg.norm(d, 1) * B
        except cp.error.SolverError:
            c = d / np.linalg.norm(d, 1) * B

        y = rng.normal(float(c @ mu_true), np.sqrt(max(float(c @ Sigma @ c), 1e-12)))

        inc, vinc = _step_llr(c, y, mu_true, U, Sigma)
        L += inc
        V += vinc

        if t > burn:
            gap = float((L - beta_threshold(V, delta, n_alts)).min())
            if gap >= 0.0:
                # The oracle KNOWS S*, so its output is S* by construction; what
                # it measures is the best achievable stopping time. Scoring it on
                # the pseudo-likelihood ranking (as an earlier version did) would
                # test the wrong thing -- that ranking is the `ecc` method's
                # inference step, not the oracle's.
                if method == "oracle":
                    return True, t, True
                return True, t, set(order[-n:].tolist()) == set(s_star.tolist())

        # belief update (pseudo-likelihood) -- drives action selection
        var = max(float(c @ Sigma @ c), 1e-12)
        # exact per-stream log-likelihood increment for the single-stream test
        # H_k : y ~ N(c'mu_0 + Delta_k c_k, var)  vs  H_0 : y ~ N(c'mu_0, var)
        llr = ((y - c * delta_sig) ** 2 - y ** 2) / (-2.0 * var)
        log_odds += llr
        if method == "cusum":
            cusum = np.maximum(0.0, cusum + llr)
            score = cusum
        elif method == "sr":
            sr = (1.0 + sr) * np.exp(llr)
            score = np.log(np.maximum(sr, 1e-300))
        else:
            score = log_odds

    return False, max_steps, False


def sweep(args) -> dict:
    Sigma = make_sigma(args.K, args.sigma, args.rho)
    if args.s_star:
        s_star = np.array(sorted(int(x) for x in args.s_star.split(",")))
        assert len(s_star) == args.n and s_star.max() < args.K
    else:
        rng0 = np.random.default_rng(args.seed)
        s_star = np.sort(rng0.choice(args.K, args.n, replace=False))
    U = all_alternatives(args.K, args.n, s_star, signal=args.signal)
    gamma, rank = gamma_star_sdp(Sigma, U)

    taus, errs, timeouts = [], [], 0
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 * args.seed + s)
        ok, tau, correct = run_trial(Sigma, U, s_star, args.n, args.K,
                                     args.delta, args.method, rng, args.max_steps, args.burn,
                                     signal=args.signal, B=args.B)
        if not ok:
            timeouts += 1
        taus.append(tau); errs.append(0.0 if correct else 1.0)

    mean_tau = float(np.mean(taus))
    err = float(np.mean(errs))
    return dict(sigma=args.sigma, rho=args.rho, K=args.K, n=args.n, B=args.B,
                signal=args.signal,
                delta=args.delta, seeds=args.seeds, method=args.method,
                s_star=[int(x) for x in s_star], gamma_star=gamma, rank_W=rank,
                mean_tau=mean_tau, err_rate=err, timeouts=timeouts,
                ratio=mean_tau * gamma / np.log(1.0 / args.delta),
                err_within_delta=bool(err <= args.delta + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sigma", default="identity")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--signal", type=float, default=SIGNAL,
                    help="per-stream mean shift Delta; Gamma* scales as Delta^2")
    ap.add_argument("--B", type=float, default=B_BUDGET,
                    help="l1 sensing budget; the rate is provably independent of it")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s-star", default="",
                    help="comma-separated anomaly indices, e.g. '0,1'. Empty = draw "
                         "uniformly at random. Use '0,1' vs '0,5' to contrast the "
                         "adjacent (rank-2) and separated (rank-1) regimes.")
    ap.add_argument("--method", choices=["oracle", "ecc", "cusum", "sr"], default="oracle")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--burn", type=int, default=5,
                    help="burn-in steps before the GLR rule may fire; keep well below"
                         " the expected tau or it dominates the ratio")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        base = dict(K=8, n=2, sigma="identity", rho=0.5, delta=0.20, seeds=40,
                    seed=0, max_steps=20000, burn=3)
        for meth in ("oracle", "ecc"):
            a = argparse.Namespace(**{**vars(args), **base, "method": meth})
            r = sweep(a)
            print(f"selftest [{meth:<6}] err={r['err_rate']:.3f} (delta={r['delta']}) "
                  f"gamma*={r['gamma_star']:.4f} rank={r['rank_W']} "
                  f"tau={r['mean_tau']:.1f} ratio={r['ratio']:.3f} "
                  f"timeouts={r['timeouts']}")
            if meth == "oracle":
                # the oracle is correct by construction; it must simply stop
                assert r["err_rate"] == 0.0, "oracle must always be correct"
                assert r["timeouts"] == 0, "oracle must not time out"
                assert r["rank_W"] == 1, "identity should give a rank-1 W*"
            else:
                # NOT asserted: the pseudo-likelihood ranking is known to be a
                # biased statistic, so ecc may (and does) violate delta. That is
                # a FINDING to report, not a bug in this script.
                if not r["err_within_delta"]:
                    print(f"           ^ NOTE: ecc violates its nominal delta "
                          f"({r['err_rate']:.3f} > {r['delta']}) -- expected, see "
                          f"REVISION-NOTES.md on the 1.10x pseudo-likelihood inflation")
        print("selftest ok: harness runs, oracle stops and is correct by construction")
        return

    res = sweep(args)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir,
                        f"{args.sigma}{args.rho}_K{args.K}_d{args.delta}"
                        f"_s{args.signal}_B{args.B}_{args.method}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"{res['method']:<7} {res['sigma']}({res['rho']}) delta={res['delta']} "
          f"gamma*={res['gamma_star']:.4f} rank={res['rank_W']} "
          f"tau={res['mean_tau']:.1f} err={res['err_rate']:.3f} "
          f"ratio={res['ratio']:.3f} -> {path}")


if __name__ == "__main__":
    main()
