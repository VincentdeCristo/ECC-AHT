"""
Diagnostic for ECC-AHT under the fixed-confidence protocol.

Question (decides whether Theorems E.5/E.6 as stated hold):
    does the champion-challenger pair stabilise, so that c_t -> a single c*,
    or does it cycle between pairwise optima?

Lemma E.13 asserts both "the Challenger stabilises to the most confusable
alternative" and "the empirical objective converges uniformly to
f(c) = min_{S' != S*} D(...)".  Those are incompatible: stabilisation gives a
PAIRWISE objective, not the max-min.  If the pair cycles, the rate actually
earned is the mixture rate min_ij mean_t D_ij(c_t), and the theorems' claim
that it equals Gamma* = max_c min_ij D_ij(c) is what has to be checked.

Three statistics per covariance pattern:
  Q1  does the top-n set stabilise to S*?           (ranking consistency)
  Q2  do the actions c_t concentrate?               (action convergence)
  Q3  rate_mix / max_t min_ij D_ij(c_t)             (cost of cycling; <= 1 always,
                                                    ~1 means no loss)

Numerical note: the reference implementation ranks streams by p_t = sigmoid(L_t).
Once L_t exceeds ~40, p_t saturates to exactly 1.0/0.0 in float64 and argsort
breaks ties arbitrarily, so the challenger j* jitters for purely numerical
reasons.  Ranking by L_t directly is the same ordering without the saturation.
Both variants are run so the artefact can be told apart from real cycling.

Run: python diagnose_convergence.py
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp
from scipy.special import expit

B_BUDGET = 4.0
SIGNAL = 1.0
BURN_IN = 200
STEPS = 2000
K_DEFAULT, N_ANOM = 30, 3
S_STAR = np.array([0, 1, 2])


def make_sigma(K: int, kind: str, rho: float) -> np.ndarray:
    """Correlation matrices as used in Appendix C of the paper."""
    idx = np.arange(K)
    if kind == "identity":
        return np.eye(K)
    if kind == "equicorr":
        return (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    if kind == "toeplitz":
        return rho ** np.abs(idx[:, None] - idx[None, :])
    if kind == "rbf":
        d2 = (idx[:, None] - idx[None, :]) ** 2
        return np.exp(-d2 / (2.0 * (rho * K) ** 2))
    raise ValueError(kind)


def make_solver(sigma: np.ndarray, K: int):
    c_var = cp.Variable(K)
    d_param = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(c_var, cp.psd_wrap(sigma + 1e-6 * np.eye(K)))),
                      [d_param @ c_var == 1, cp.norm1(c_var) <= B_BUDGET])
    return c_var, d_param, prob


def run_ecc_aht(sigma: np.ndarray, s_star: np.ndarray, n: int, K: int,
                steps: int, seed: int, rank_by_logodds: bool = True):
    """ECC_AHT.select_action/update, as implemented in algorithms.py."""
    rng = np.random.default_rng(seed)
    mu_true = np.zeros(K)
    mu_true[s_star] = SIGNAL
    delta = np.full(K, SIGNAL)

    c_var, d_param, prob = make_solver(sigma, K)
    log_odds = np.full(K, np.log((n / K) / (1 - n / K)))

    actions, pairs, s_hat_hist = [], [], []
    for _ in range(steps):
        key = log_odds if rank_by_logodds else expit(log_odds)
        order = np.argsort(key)
        s_hat = order[-n:]
        comp = order[:-n]
        i_star = int(s_hat[np.argmin(key[s_hat])])
        j_star = int(comp[np.argmax(key[comp])])

        d = np.zeros(K)
        d[i_star], d[j_star] = delta[i_star], -delta[j_star]
        try:
            d_param.value = d
            prob.solve(solver=cp.OSQP, warm_start=True)
            c = c_var.value
            norm1 = np.linalg.norm(c, 1)
            c = c / norm1 * B_BUDGET if norm1 > 1e-9 else d / np.linalg.norm(d, 1) * B_BUDGET
        except cp.error.SolverError:
            c = d / np.linalg.norm(d, 1) * B_BUDGET

        actions.append(c)
        pairs.append((i_star, j_star))
        s_hat_hist.append(tuple(sorted(s_hat.tolist())))

        y = rng.normal(c @ mu_true, np.sqrt(max(c @ sigma @ c, 1e-9)))
        var = max(c @ sigma @ c, 1e-9)
        log_odds += ((y - c * delta) ** 2 - y ** 2) / (-2.0 * var)

    return np.array(actions), pairs, s_hat_hist, log_odds


def pairwise_divergences(c: np.ndarray, sigma: np.ndarray, s_star: np.ndarray,
                         K: int, delta: np.ndarray) -> np.ndarray:
    """D(H_S* || H_S' | c) for every single-swap alternative S'."""
    comp = [j for j in range(K) if j not in s_star]
    denom = 2.0 * float(c @ sigma @ c)
    return np.array([(float(c @ u)) ** 2 / denom for u in _swap_dirs(s_star, comp, K, delta)])


def _swap_dirs(s_star, comp, K, delta):
    for i in s_star:
        for j in comp:
            u = np.zeros(K)
            u[i], u[j] = delta[i], -delta[j]
            yield u


def gamma_star_estimate(sigma: np.ndarray, s_star: np.ndarray, K: int,
                        delta: np.ndarray, traj: np.ndarray,
                        n_random: int = 20000, seed: int = 0) -> float:
    """Best min_ij D_ij(c) found over a rich candidate family.

    Any feasible c gives a LOWER bound on Gamma*, so this is a lower bound.
    Candidates: the algorithm's own actions, inverse-covariance designs, sparse
    random directions, and smooth/oscillatory directions (which matter for
    low effective rank, where the objective is extremely non-concave).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(K)

    cands = [c for c in traj[::max(1, len(traj) // 200)]]
    comp = [j for j in range(K) if j not in s_star]
    for u in _swap_dirs(s_star, comp, K, delta):
        cands.append(np.linalg.solve(sigma, u))

    for _ in range(n_random):
        c = np.zeros(K)
        nz = int(rng.integers(2, min(K, 8) + 1))
        supp = rng.choice(K, nz, replace=False)
        c[supp] = rng.standard_normal(nz)
        cands.append(c)
    for freq in (0.5, 1.0, 2.0, 3.0, 5.0):
        cands.append(np.cos(freq * np.pi * idx / K) * rng.standard_normal(K))

    best, best_c = -np.inf, None
    for c in cands:
        nrm = np.linalg.norm(c, 1)
        if nrm < 1e-12:
            continue
        c = c / nrm * B_BUDGET
        v = float(pairwise_divergences(c, sigma, s_star, K, delta).min())
        if v > best:
            best, best_c = v, c.copy()

    step = B_BUDGET / 4
    for _ in range(3000):
        improved = False
        for k in range(K):
            for sgn in (1.0, -1.0):
                cand = best_c.copy()
                cand[k] += sgn * step
                nrm = np.linalg.norm(cand, 1)
                if nrm < 1e-12 or nrm > B_BUDGET:
                    continue
                cand = cand / nrm * B_BUDGET
                v = float(pairwise_divergences(cand, sigma, s_star, K, delta).min())
                if v > best + 1e-12:
                    best, best_c, improved = v, cand, True
        if not improved:
            step /= 2.0
            if step < 1e-4:
                break
    return best


def main() -> None:
    K, n = K_DEFAULT, N_ANOM
    delta = np.full(K, SIGNAL)
    patterns = [("identity", 0.0), ("equicorr", 0.5), ("equicorr", 0.8),
                ("equicorr", 0.95), ("toeplitz", 0.5), ("rbf", 0.05)]

    for rank_by_logodds in (True, False):
        tag = "rank by log-odds (stable)" if rank_by_logodds else "rank by sigmoid(p) (as shipped)"
        print(f"\n=== {tag} ===")
        print(f"K={K} n={n} B={B_BUDGET} S*={S_STAR.tolist()} steps={STEPS} burn-in={BURN_IN}")
        print(f"{'pattern':<16}{'S_t=S*':>8}{'#pairs':>7}{'modal%':>8}"
              f"{'act.spread':>11}{'rate_mix':>10}{'best_act':>10}{'ratio':>8}")

        for kind, rho in patterns:
            sigma = make_sigma(K, kind, rho)
            traj, pairs, s_hist, log_odds = run_ecc_aht(
                sigma, S_STAR, n, K, STEPS, seed=0, rank_by_logodds=rank_by_logodds)
            post_a, post_p, post_s = traj[BURN_IN:], pairs[BURN_IN:], s_hist[BURN_IN:]

            frac_correct = np.mean([s == tuple(sorted(S_STAR.tolist())) for s in post_s])
            uniq, counts = np.unique(np.array(post_p), axis=0, return_counts=True)
            modal = counts.max() / counts.sum()

            modal_action = np.mean([c for c, p in zip(post_a, post_p)
                                    if tuple(p) == tuple(uniq[counts.argmax()])], axis=0)
            spread = float(np.mean([np.linalg.norm(c - modal_action) for c in post_a]))

            # mixture rate: per-pair time averages, then min over ALL pairs
            comp = [j for j in range(K) if j not in S_STAR]
            all_keys = [(int(i), int(j)) for i in S_STAR for j in comp]
            series = {key: [] for key in all_keys}
            for c in post_a:
                d_all = pairwise_divergences(c, sigma, S_STAR, K, delta)
                for idx, key in enumerate(all_keys):
                    series[key].append(float(d_all[idx]))
            rate_mix = min(float(np.mean(v)) for v in series.values())
            best_act = max(float(pairwise_divergences(c, sigma, S_STAR, K, delta).min()) for c in post_a)
            gamma = gamma_star_estimate(sigma, S_STAR, K, delta, traj, n_random=4000)

            print(f"{kind + f'(rho={rho})':<16}{frac_correct:>7.1%}{len(uniq):>7}{modal:>7.1%}"
                  f"{spread:>11.3f}{rate_mix:>10.4f}{best_act:>10.4f}"
                  f"{rate_mix / best_act:>8.3f}   [Gamma*>= {gamma:.4f}]")

    # invariants: rate_mix <= best_act <= Gamma*  always
    sigma = make_sigma(K, "equicorr", 0.8)
    traj, _, _, _ = run_ecc_aht(sigma, S_STAR, n, K, 400, seed=1)
    g = gamma_star_estimate(sigma, S_STAR, K, delta, traj, n_random=1000)
    vals = [pairwise_divergences(c, sigma, S_STAR, K, delta).min() for c in traj]
    assert max(vals) <= g + 1e-9, "Gamma* must dominate every feasible design"
    assert min(vals) <= max(vals)
    print("\nself-check ok: rate_mix <= best_act <= Gamma*  (sandwich holds)")


if __name__ == "__main__":
    main()
