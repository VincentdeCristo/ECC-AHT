"""
Decisive test for Theorem E.5/E.6 of the ECC-AHT paper.

Three rates are in play, and the paper silently identifies all three:

  Gamma_single = max_c  min_j D_j(c)          paper's Gamma*; ONE fixed measurement
                                              must separate every alternative at once
  Gamma_mix    = max_pi min_j E_{c~pi}[D_j]   information rate of the design
                                              sequence the algorithm actually runs
  rate_gap     = d/dt min_{i in S*, j not in S*} (L_t(i) - L_t(j))
                                              growth rate of the statistic the
                                              STOPPING RULE thresholds

The theorems need rate_gap == Gamma_single.  This script measures rate_gap from
simulation, computes Gamma_mix directly from the realised action sequence, and
compares both against the best single design found by search.

If rate_gap > Gamma_single the algorithm beats the paper's claimed rate and
"limsup E[tau]/log(1/delta) = 1/Gamma*" is false.
If rate_gap < Gamma_single the upper bound fails in the other direction.

Run: python measure_achieved_rate.py
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

B_BUDGET = 2.0
SIGNAL = 1.0
BURN = 2000
STEPS = 6000
SEEDS = 5


def make_sigma(K: int, kind: str, rho: float) -> np.ndarray:
    idx = np.arange(K)
    if kind == "identity":
        return np.eye(K)
    if kind == "equicorr":
        return (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    if kind == "toeplitz":
        return rho ** np.abs(idx[:, None] - idx[None, :])
    raise ValueError(kind)


def swap_dirs(s_star, comp, K, delta):
    for i in s_star:
        for j in comp:
            u = np.zeros(K)
            u[i], u[j] = delta[i], -delta[j]
            yield u


def true_divergences(c, sigma, dirs):
    """D_j(c) = (c^T u_j)^2 / (2 c^T Sigma c) for every alternative."""
    denom = 2.0 * float(c @ sigma @ c)
    return np.array([(float(c @ u)) ** 2 / denom for u in dirs])


def subset_diffs(K, n, s_star):
    """(mu_S' - mu_S*) / SIGNAL for every alternative S' of size n, S' != S*.

    Returns an array of shape (#alternatives, K).
    """
    from itertools import combinations
    rows = []
    s_set = set(s_star.tolist())
    for combo in combinations(range(K), n):
        if set(combo) == s_set:
            continue
        v = np.zeros(K)
        for k in combo:
            v[k] = SIGNAL
        for k in s_star:
            v[k] -= SIGNAL
        rows.append(v)
    return np.array(rows)


def run_once(sigma, s_star, n, K, steps, seed, sub_diffs):
    """ECC-AHT as implemented, tracking the decision statistic and actions."""
    rng = np.random.default_rng(seed)
    mu_true = np.zeros(K)
    mu_true[s_star] = SIGNAL
    delta = np.full(K, SIGNAL)
    comp = np.array([j for j in range(K) if j not in s_star])

    c_var = cp.Variable(K)
    d_param = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(c_var, cp.psd_wrap(sigma + 1e-6 * np.eye(K)))),
                      [d_param @ c_var == 1, cp.norm1(c_var) <= B_BUDGET])

    log_odds = np.full(K, np.log((n / K) / (1 - n / K)))
    gaps, actions, ok = [], [], 0
    true_glr = np.zeros(sub_diffs.shape[0])   # exact subset-level GLR, same trajectory
    glr_min = []

    for _ in range(steps):
        order = np.argsort(log_odds)                 # rank by L, not sigmoid(L)
        s_hat, rest = order[-n:], order[:-n]
        i_star = int(s_hat[np.argmin(log_odds[s_hat])])
        j_star = int(rest[np.argmax(log_odds[rest])])
        ok += int(set(s_hat.tolist()) == set(s_star.tolist()))

        d = np.zeros(K)
        d[i_star], d[j_star] = delta[i_star], -delta[j_star]
        try:
            d_param.value = d
            prob.solve(solver=cp.OSQP, warm_start=True)
            c = np.asarray(c_var.value).ravel()
            nrm = np.linalg.norm(c, 1)
            c = c / nrm * B_BUDGET if nrm > 1e-9 else d / np.linalg.norm(d, 1) * B_BUDGET
        except cp.error.SolverError:
            c = d / np.linalg.norm(d, 1) * B_BUDGET

        # the statistic the GLR stopping rule thresholds (Lemma E.16: single swaps
        # are the worst case, so the min over all S' is this pairwise min)
        gaps.append(float(log_odds[s_star].min() - log_odds[comp].max()))
        actions.append(c)

        var = max(float(c @ sigma @ c), 1e-9)
        y = rng.normal(c @ mu_true, np.sqrt(var))
        log_odds += ((y - c * delta) ** 2 - y ** 2) / (-2.0 * var)

        # exact subset GLR on the SAME trajectory: increment_j = (b_j^2 - 2 a b_j)/(2 var)
        # with a = y - c'mu_{S*} and b_j = c'(mu_{S'} - mu_{S*}).
        a = y - float(c @ mu_true)
        b = sub_diffs @ c
        true_glr += (b ** 2 - 2.0 * a * b) / (2.0 * var)
        glr_min.append(float(true_glr.min()))

    return np.array(gaps), np.array(actions), ok / steps, np.array(glr_min)


def slope(x, y):
    """Least-squares slope of y on x."""
    return float(np.polyfit(x, y, 1)[0])


def gamma_single_search(sigma, dirs, K, cands, rounds=3000):
    """Best min_j D_j(c) over a candidate set, then coordinate refinement."""
    best, best_c = -np.inf, None
    for c in cands:
        nrm = np.linalg.norm(c, 1)
        if nrm < 1e-12:
            continue
        c = c / nrm * B_BUDGET
        v = float(true_divergences(c, sigma, dirs).min())
        if v > best:
            best, best_c = v, c.copy()
    step = B_BUDGET / 4
    for _ in range(rounds):
        improved = False
        for k in range(K):
            for sgn in (1.0, -1.0):
                cand = best_c.copy()
                cand[k] += sgn * step
                nrm = np.linalg.norm(cand, 1)
                if nrm < 1e-12 or nrm > B_BUDGET:
                    continue
                cand = cand / nrm * B_BUDGET
                v = float(true_divergences(cand, sigma, dirs).min())
                if v > best + 1e-12:
                    best, best_c, improved = v, cand, True
        if not improved:
            step /= 2.0
            if step < 1e-4:
                break
    return best


def main() -> None:
    K, n = 20, 2
    s_star = np.array([0, 1])
    delta = np.full(K, SIGNAL)
    comp = [j for j in range(K) if j not in s_star]
    dirs = list(swap_dirs(s_star, comp, K, delta))
    print(f"K={K} n={n} B={B_BUDGET} S*={s_star.tolist()} "
          f"#alternatives={len(dirs)} steps={STEPS} seeds={SEEDS}")
    sub_diffs = subset_diffs(K, n, s_star)
    print(f"#alternatives(all subsets)={len(sub_diffs)}")
    print(f"{'Sigma':<16}{'rate_pseudo':>12}{'rate_trueGLR':>13}{'Gamma_mix':>11}"
          f"{'Gamma_sgl':>11}{'pseudo/sgl':>11}{'true/sgl':>10}{'S_t=S*':>8}")

    for kind, rho in [("identity", 0.0), ("equicorr", 0.5), ("toeplitz", 0.5)]:
        sigma = make_sigma(K, kind, rho)
        slope_seeds, true_seeds, mix_seeds, ok_seeds, all_actions = [], [], [], [], []
        for seed in range(SEEDS):
            gaps, actions, ok, glr = run_once(sigma, s_star, n, K, STEPS, seed, sub_diffs)
            t = np.arange(BURN, STEPS)
            slope_seeds.append(slope(t, gaps[BURN:]))
            true_seeds.append(slope(t, glr[BURN:]))
            D = np.array([true_divergences(c, sigma, dirs) for c in actions[BURN:]])
            mix_seeds.append(float(D.mean(axis=0).min()))
            ok_seeds.append(ok)
            if seed < 3:
                all_actions.append(actions[::max(1, len(actions) // 300)])

        rate_gap = float(np.mean(slope_seeds))
        rate_true = float(np.mean(true_seeds))
        gamma_mix = float(np.mean(mix_seeds))

        rng = np.random.default_rng(0)
        cands = [c for arr in all_actions for c in arr]
        cands += [np.linalg.solve(sigma, u) for u in dirs]
        cands += [rng.standard_normal(K) for _ in range(5000)]
        gamma_sgl = gamma_single_search(sigma, dirs, K, cands)

        print(f"{kind + f'({rho})':<16}{rate_gap:>12.4f}{rate_true:>13.4f}"
              f"{gamma_mix:>11.4f}{gamma_sgl:>11.4f}"
              f"{rate_gap / gamma_sgl:>11.3f}{rate_true / gamma_sgl:>10.3f}"
              f"{np.mean(ok_seeds):>7.1%}")

    # self-check: the exact subset GLR must grow at Gamma_mix, by construction.
    # rate_trueGLR = min_{S'} mean_t D(H_S*||H_S'|c_t) = Gamma_mix.
    sigma = make_sigma(K, "identity", 0.0)
    gaps, actions, _, glr = run_once(sigma, s_star, n, K, 4000, 0, sub_diffs)
    t = np.arange(BURN, 4000)
    measured = slope(t, glr[BURN:])
    D = np.array([true_divergences(c, sigma, dirs) for c in actions[BURN:]])
    predicted = float(D.mean(axis=0).min())
    assert abs(measured - predicted) / max(predicted, 1e-9) < 0.25, (measured, predicted)
    print(f"\nself-check ok: rate_trueGLR {measured:.4f} matches Gamma_mix {predicted:.4f} "
          f"(within 25%)")


if __name__ == "__main__":
    main()
