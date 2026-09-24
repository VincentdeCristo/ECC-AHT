"""
Numerical check of the LOWER BOUND  E[tau] >= d(delta,1-delta) / Gamma*.

The claim is information-theoretic: it holds for ANY delta-correct procedure, so
the check runs the `oracle` rule (exact subset GLR + the Theorem-2 threshold),
which is delta-correct by construction (err = 0 <= delta).

Three links are checked separately, because a failure in any one would sink the
theorem and they fail for different reasons:

  (A) per-pair, under the NULL measure:   E_{S*}[ V_tau^(S') ] >= 2*d(delta,1-delta)
      from data processing on the stopped measures:
         KL(P_S*|F_tau || P_S'|F_tau) = E[V/2 - D] = E[V_tau]/2
      (E[D_tau] = 0 by optional stopping), and KL >= kl(P(E)||P'(E)) >= d.
      NB the expectation is under P_{S*} -- the ALTERNATIVE-measure version of
      Kaufmann's lemma gives one measure per S' and those cannot be averaged.
  (B) pathwise, for every t:              sum_{S'} lam*_{S'} V_t^(S') <= t * gamma*
      since sum_s (w'Abar(lam)w)/||w||^2 <= t * lambda_max(Abar(lam)).
  (C) the conclusion:                     E[tau] >= d / Gamma*,  Gamma* = gamma*/2.

(A) and (B) are under the same measure, which is exactly what makes the average
in (C) legal -- the step the earlier "expectation route cannot prove it" note got
wrong (it paired each per-pair bound with that pair's own cap max_c D_{S'},
yielding the weaker min-max rate).
"""
from __future__ import annotations

import argparse

import cvxpy as cp
import numpy as np

from exp_rate_validation import SIGNAL, all_alternatives, make_sigma

GAMMA_CERT = {20: 5.0 / 18.0}   # exact rational certificate, K=20 S*={0,1} Sigma=I


def binary_kl(lam_a: float, lam_b: float) -> float:
    """kl(a||b), guarding the 0 log 0 endpoints."""
    out = 0.0
    if lam_a > 0:
        out += lam_a * np.log(lam_a / lam_b)
    if lam_a < 1:
        out += (1 - lam_a) * np.log((1 - lam_a) / (1 - lam_b))
    return out


def d_delta(delta: float) -> float:
    return binary_kl(delta, 1.0 - delta)


def dual_gamma_star(Sigma: np.ndarray, U: np.ndarray):
    """gamma* = min_lam lambda_max(Abar(lam)) over the simplex, plus the argmin.

    Solved as an SDP rather than by reading a primal solver's duals, so the
    returned lam* is certified feasible: t*I - Abar(lam*) is PSD by construction.
    """
    K = Sigma.shape[0]
    Linv = np.linalg.inv(np.linalg.cholesky(Sigma))
    V = U @ Linv.T                       # rows are v_{S'}
    m = V.shape[0]
    lam = cp.Variable(m, nonneg=True)
    t = cp.Variable()
    A = sum(lam[i] * np.outer(V[i], V[i]) for i in range(m))
    cp.Problem(cp.Minimize(t),
               [lam >= 0, cp.sum(lam) == 1, t * np.eye(K) - A >> 0]).solve(
                   solver=cp.SCS, eps=1e-9, max_iters=200000)
    return float(t.value), np.asarray(lam.value).ravel()


def run_trial_traced(Sigma, U, s_star, n, K, delta, rng, max_steps, burn=5):
    """run_trial, but returning (tau, V_tau) with the full V vector at stopping."""
    from exp_rate_validation import _step_llr, beta_threshold

    mu_true = np.zeros(K)
    mu_true[s_star] = SIGNAL
    delta_sig = np.full(K, SIGNAL)
    n_alts = len(U)
    log_odds = np.full(K, np.log((n / K) / (1 - n / K)))
    score = log_odds.copy()
    L = np.zeros(n_alts)
    V = np.zeros(n_alts)

    c_var = cp.Variable(K)
    d_par = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(c_var, cp.psd_wrap(Sigma))),
                      [d_par @ c_var == 1, cp.norm1(c_var) <= 2.0])

    for t in range(1, max_steps + 1):
        order = np.argsort(score)
        s_hat, rest = order[-n:], order[:-n]
        i_s = int(s_hat[np.argmin(log_odds[s_hat])])
        j_s = int(rest[np.argmax(log_odds[rest])])
        d = np.zeros(K)
        d[i_s], d[j_s] = SIGNAL, -SIGNAL
        try:
            d_par.value = d
            prob.solve(solver=cp.OSQP, warm_start=True)
            c = np.asarray(c_var.value).ravel()
            nrm = np.linalg.norm(c, 1)
            c = c / nrm * 2.0 if nrm > 1e-9 else d / np.linalg.norm(d, 1) * 2.0
        except cp.error.SolverError:
            c = d / np.linalg.norm(d, 1) * 2.0

        y = rng.normal(float(c @ mu_true), np.sqrt(max(float(c @ Sigma @ c), 1e-12)))
        inc, vinc = _step_llr(c, y, mu_true, U, Sigma)
        L += inc
        V += vinc

        if t > burn and float((L - beta_threshold(V, delta, n_alts)).min()) >= 0.0:
            return t, V

        var = max(float(c @ Sigma @ c), 1e-12)
        llr = ((y - c * delta_sig) ** 2 - y ** 2) / (-2.0 * var)
        log_odds += llr
        score = log_odds
    return None, None


def check(args) -> None:
    Sigma = make_sigma(args.K, args.sigma, args.rho)
    s_star = np.array(sorted(int(x) for x in args.s_star.split(",")))
    U = all_alternatives(args.K, args.n, s_star)
    m = len(U)

    gamma, lam = dual_gamma_star(Sigma, U)
    gamma_star = gamma / 2.0
    assert lam.min() >= -1e-9, "dual returned an infeasible lambda"
    cert = GAMMA_CERT.get(args.K) if args.sigma == "identity" else None
    if cert is not None:
        assert abs(gamma_star - cert) < 1e-6, (
            f"SDP disagrees with the exact rational certificate: "
            f"{gamma_star} vs {cert}")

    thr = 2.0 * d_delta(args.delta)
    print(f"K={args.K} n={args.n} S*={list(s_star)} sigma={args.sigma}")
    print(f"  Gamma* = {gamma_star:.6f}   (exact certificate: {cert})")
    print(f"  delta={args.delta}  d(delta,1-delta)={d_delta(args.delta):.4f}   "
          f"2d={thr:.4f}   bound E[tau] >= d/Gamma* = {d_delta(args.delta)/gamma_star:.3f}")
    print(f"  lam*: {np.count_nonzero(lam > 1e-6)}/{m} alternatives active, "
          f"max={lam.max():.4f}")

    taus, Vs, viol_a, viol_b = [], [], 0, 0
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 * args.seed + s)
        tau, V_final = run_trial_traced(Sigma, U, s_star, args.n, args.K,
                                        args.delta, rng, args.max_steps, args.burn)
        if tau is None:
            continue
        taus.append(tau)
        Vs.append(V_final)
        if np.any(V_final < thr):
            viol_a += 1
        if float(lam @ V_final) > tau * gamma + 1e-6:
            viol_b += 1

    if not taus:
        print("  no completed trials -- increase --max-steps")
        return
    Vs = np.array(Vs)
    taus = np.array(taus)
    worst = Vs.mean(axis=0).min()
    print(f"\n  (A) per-pair E[V_tau^(S')] >= 2d, over n={len(taus)} trials:")
    print(f"      min over S' of E[V_tau^(S')] = {worst:.3f}   (needs >= {thr:.3f})"
          f"   {'PASS' if worst >= thr else 'FAIL'}")
    print(f"      trials with some pair below 2d: {viol_a}/{len(taus)}"
          f"   (individual trials may violate; the MEAN must not)")
    print(f"  (B) pathwise sum lam* V_tau <= tau*gamma*: violations {viol_b}/{len(taus)}"
          f"   {'PASS' if viol_b == 0 else 'FAIL'}")
    print(f"      E[sum lam* V_tau] = {float(lam @ Vs.mean(axis=0)):.3f}   "
          f"2d = {thr:.3f}   E[tau]*gamma* = {taus.mean()*gamma:.1f}")
    print(f"  (C) E[tau] = {taus.mean():.2f}  vs  bound d/Gamma* = "
          f"{d_delta(args.delta)/gamma_star:.3f}   "
          f"{'PASS' if taus.mean() >= d_delta(args.delta)/gamma_star else 'FAIL'}")
    print(f"      ratio E[tau]*Gamma*/log(1/delta) = "
          f"{taus.mean()*gamma_star/np.log(1/args.delta):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--sigma", default="identity")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s-star", default="0,1")
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--burn", type=int, default=5)
    args = ap.parse_args()
    check(args)


if __name__ == "__main__":
    main()
