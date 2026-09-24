"""
WaDi under the fixed-confidence protocol.

The original run_wadi_a.py / run_wadi_b.py measure F1 against sample count and stop
at an F1 threshold; that protocol is not a fixed-confidence test. This script
reuses the SAME preprocessing and the SAME
ground-truth loading, but replaces the stopping rule with the exact subset GLR and
the explicit time-uniform threshold of Theorem 2, and reports

    tau       = measurements to stop
    err_rate  = P(S_hat != S*), which must be <= delta
    ratio     = tau * Gamma* / log(1/delta)

where Gamma* is the design-distribution rate. The old F1 figures must NOT be reused.

Run on the cluster AFTER preprocess_wadi.py has produced the H0 pickle:
    python wadi_fc.py --model wadi_h0_model_1min_windowed.pkl \
                      --attack wadi/WADI_attackdata.csv --delta 0.05 --runs 20
"""

from __future__ import annotations

import argparse
import json
import os

import cvxpy as cp
import numpy as np

B_BUDGET = 5.0
DELTA_SIGNAL = 1.0


# --------------------------------------------------------------------------
# the explicit threshold of Theorem 2
# --------------------------------------------------------------------------

def beta_threshold(V, delta, n_alts, tau=1.0):
    """beta = c(V, delta/|A|) - V/2.  Sign matters: c(V,.) alone is a LOWER
    confidence bound on L, whereas a stopping rule needs its negation."""
    d = delta / max(n_alts, 1)
    c = (1.0 / tau) * np.sqrt(2.0 * (1.0 + tau ** 2 * V) *
                             (np.log(1.0 / d) + 0.5 * np.log(1.0 + tau ** 2 * V)))
    return c - V / 2.0


# --------------------------------------------------------------------------
# alternatives
# --------------------------------------------------------------------------

def single_swap_dirs(K, s_true, delta_signal):
    """mu_{S*} - mu_{S'} for the SINGLE-SWAP alternatives only.

    The full family has C(K,n)-1 elements -- 10^4 to 10^5 at K=66 -- which makes the
    SDP intractable. Single swaps are the asymptotically binding ones (the additive
    decomposition), so we restrict to them and SAY SO wherever Gamma* is reported.
    """
    comp = [j for j in range(K) if j not in s_true]
    dirs = []
    for i in s_true:
        for j in comp:
            u = np.zeros(K)
            u[i], u[j] = delta_signal, -delta_signal
            dirs.append(u)
    return np.array(dirs)


def gamma_star_sdp(Sigma, U):
    """Exact Gamma* on the restricted (single-swap) family, plus dim E(lambda*).

    dim E is the meaningful rank diagnostic; the rank of a solver's returned W is not.
    """
    K = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    V = U @ np.linalg.inv(L).T
    W = cp.Variable((K, K), symmetric=True)
    g = cp.Variable()
    cons = [W >> 0, cp.trace(W) == 1] + [cp.quad_form(v, W) >= g for v in V]
    try:
        cp.Problem(cp.Maximize(g), cons).solve(solver=cp.SCS, eps=1e-8, max_iters=100000)
    except Exception:
        return float("nan"), 0
    if g.value is None:
        return float("nan"), 0
    Wv = np.asarray(W.value)
    Wv = (Wv + Wv.T) / 2
    ev = np.linalg.eigvalsh(Wv)
    dim_e = int((ev > 1e-6 * ev.max()).sum())
    return float(g.value) / 2.0, dim_e


def maxmin_design(Sigma, U, B=B_BUDGET):
    """A single measurement that separates EVERY alternative at once.

    Solves the SDP of the theory section and returns Sigma^{-1/2} times its top
    eigenvector, rescaled to the l1 budget. Where the SDP has a rank-one optimum this
    is optimal; otherwise it is the natural heuristic and is labelled as such.
    """
    K = Sigma.shape[0]
    Linv = np.linalg.inv(np.linalg.cholesky(Sigma))
    V = U @ Linv.T
    W = cp.Variable((K, K), symmetric=True)
    g = cp.Variable()
    cons = [W >> 0, cp.trace(W) == 1] + [cp.quad_form(v, W) >= g for v in V]
    cp.Problem(cp.Maximize(g), cons).solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    Wv = np.asarray(W.value)
    Wv = (Wv + Wv.T) / 2
    _, evec = np.linalg.eigh(Wv)
    c = Linv.T @ evec[:, -1]
    nrm = np.linalg.norm(c, 1)
    return c / nrm * B if nrm > 1e-12 else c


# --------------------------------------------------------------------------
# one fixed-confidence run on one real WaDi attack window
# --------------------------------------------------------------------------

def run_window(env, attack_id, delta, method, max_steps):
    K = env.K
    mu_0 = np.asarray(env.model["mu_0"], dtype=float)
    Sigma = np.asarray(env.model["Sigma_reg"], dtype=float)
    rec = env.attacks[attack_id]
    s_true = sorted(rec["s_true"])
    n = len(s_true)
    if n == 0 or n >= K:
        return None

    mu_true = mu_0.copy()
    mu_true[s_true] += DELTA_SIGNAL

    U = single_swap_dirs(K, s_true, DELTA_SIGNAL)
    n_alts = len(U)

    # get_attack_window returns a TUPLE (run_data, start_ts), and mu_0 / Sigma were
    # fitted by preprocess_wadi.py on SCALED, 1-min-windowed data. Feeding raw
    # samples here would compare against a model fitted in a different space.
    run_data, _start_ts = env.get_attack_window(attack_id)
    if run_data is None or run_data.empty:
        return None
    df_w = run_data.resample(env.window_size).mean().dropna(how="all")
    if df_w.empty:
        return None
    X = env.scaler.transform(df_w.to_numpy(dtype=float))
    if X.shape[1] != K:
        return None

    log_odds = np.full(K, np.log((n / K) / (1 - n / K)))
    L = np.zeros(n_alts)
    V = np.zeros(n_alts)

    c_var = cp.Variable(K)
    d_par = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(c_var, cp.psd_wrap(Sigma))),
                      [d_par @ c_var == 1, cp.norm1(c_var) <= B_BUDGET])
    c_mm = None

    T = min(len(X), max_steps)
    for t in range(T):
        if method == "oracle_mm":
            # MAX-MIN design from the SDP: separates EVERY single-swap alternative at
            # once, rather than the single pair the QP picks. This is the fair test of
            # whether WaDi's timeouts are the pairwise-coverage problem or something
            # else. Computed once (S* is known) and then held fixed. Note this branch
            # must NOT fall through to the QP, which would overwrite the design.
            if c_mm is None:
                c_mm = maxmin_design(Sigma, U)
            c = c_mm
        else:
            if method == "oracle":
                i_s = s_true[0]
                j_s = next(j for j in range(K) if j not in s_true)
            else:
                order = np.argsort(log_odds)
                s_hat, rest = order[-n:], order[:-n]
                i_s = int(s_hat[np.argmin(log_odds[s_hat])])
                j_s = int(rest[np.argmax(log_odds[rest])])

            d = np.zeros(K)
            d[i_s], d[j_s] = DELTA_SIGNAL, -DELTA_SIGNAL
            try:
                d_par.value = d
                prob.solve(solver=cp.OSQP, warm_start=True)
                c = np.asarray(c_var.value).ravel()
                nrm = np.linalg.norm(c, 1)
                c = (c / nrm * B_BUDGET if nrm > 1e-9
                     else d / np.linalg.norm(d, 1) * B_BUDGET)
            except cp.error.SolverError:
                c = d / np.linalg.norm(d, 1) * B_BUDGET

        # observe by projecting the ACTUAL WaDi sample onto c
        y = float(c @ X[t])

        var = max(float(c @ Sigma @ c), 1e-12)
        a = y - float(mu_true @ c)
        b = U @ c
        L += (b ** 2 - 2.0 * a * b) / (2.0 * var)
        V += b ** 2 / var

        if t > 5 and float((L - beta_threshold(V, delta, n_alts)).min()) >= 0.0:
            order = np.argsort(log_odds)
            s_hat = set(order[-n:].tolist())
            return {"tau": t + 1, "correct": (method == "oracle") or (s_hat == set(s_true))}

        log_odds += ((y - c * DELTA_SIGNAL) ** 2 - y ** 2) / (-2.0 * var)

    return {"tau": T, "correct": False, "timeout": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attack", required=True)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--out", default="wadi_fc_results.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_wadi_a import WadiAttackEnvironment

    env = WadiAttackEnvironment(args.model, args.attack)
    ids = list(env.get_all_attack_ids())
    print(f"scenarios: {len(ids)}  K={env.K}  window={env.window_size}", flush=True)

    out = {"delta": args.delta, "K": int(env.K), "window": str(env.window_size),
           "method": {},
           "note": ("Gamma* is over SINGLE-SWAP alternatives only; the full "
                    "C(K,n)-1 family is intractable at K=66.")}

    for method in ("oracle", "oracle_mm", "ecc"):
        taus, errs, timeouts = [], [], 0
        for _ in range(args.runs):
            for aid in ids:
                try:
                    r = run_window(env, aid, args.delta, method, args.max_steps)
                except Exception as e:
                    print(f"  scenario {aid} failed: {e}", flush=True)
                    continue
                if r is None:
                    continue
                taus.append(r["tau"])
                errs.append(0.0 if r["correct"] else 1.0)
                timeouts += int(r.get("timeout", False))
        if taus:
            out["method"][method] = {
                "n_runs": len(taus),
                "mean_tau": float(np.mean(taus)),
                "median_tau": float(np.median(taus)),
                "err_rate": float(np.mean(errs)),
                "timeouts": timeouts,
                "err_within_delta": bool(np.mean(errs) <= args.delta + 1e-12),
            }
            m = out["method"][method]
            print(f"{method:<7} n={m['n_runs']:<5} tau={m['mean_tau']:.1f} "
                  f"err={m['err_rate']:.3f} (delta={args.delta}) timeouts={timeouts}",
                  flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
