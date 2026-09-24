"""
Is Gamma* = max_c min_{S' != S*} D(H_S* || H_S' | c) the right information rate
for an ADAPTIVE procedure?

Two candidate rates:
  Gamma_single = max_c  min_j  D_j(c)            (one fixed measurement, which must
                                                  separate every alternative at once)
  Gamma_mix    = max_pi min_j  E_{c~pi}[D_j(c)]  (adaptive/mixture: different
                                                  measurements at different times)

Claim under test: Gamma_mix > Gamma_single is possible, in which case the paper's
Gamma* is not the achievable rate, its lower bound is not tight, and
"limsup E[tau]/log(1/delta) = 1/Gamma*" is false.

Method: discretise the l1-sphere into M directions, build A[j, a] = D_j(c_a), then
  Gamma_single >= max_a min_j A[j, a]
  Gamma_mix    >= LP: max t s.t. sum_a pi_a A[j,a] >= t for all j, sum pi = 1, pi >= 0
The LP optimises exactly over mixtures of the sampled directions, so the comparison
is meaningful even though both numbers are lower bounds.

Run: python gamma_single_vs_mixture.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

B_BUDGET = 2.0
SIGNAL = 1.0


def swap_directions(s_star: np.ndarray, K: int) -> np.ndarray:
    comp = [j for j in range(K) if j not in s_star]
    rows = []
    for i in s_star:
        for j in comp:
            u = np.zeros(K)
            u[i], u[j] = SIGNAL, -SIGNAL
            rows.append(u)
    return np.array(rows)


def divergence_matrix(dirs: np.ndarray, cands: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """A[j, a] = (c_a^T u_j)^2 / (2 c_a^T Sigma c_a).  Shape: (#alternatives, #cands)."""
    proj = cands @ dirs.T                                     # (#cands, #alternatives)
    quad = np.einsum("ak,kl,al->a", cands, sigma, cands)      # (#cands,)
    A = (proj ** 2) / (2.0 * np.maximum(quad, 1e-300))[:, None]
    A = A.T
    assert A.shape == (dirs.shape[0], cands.shape[0]), A.shape
    return A


def sample_directions(K: int, M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((M, K))
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12) / B_BUDGET
    for i in range(K):                     # single-pair optimal designs included
        for j in range(i + 1, K):
            u = np.zeros(K)
            u[i], u[j] = SIGNAL, -SIGNAL
            C = np.vstack([C, u / np.linalg.norm(u, 1) * B_BUDGET])
    return C


def gamma_single(A: np.ndarray) -> float:
    """max over columns a of (min over alternatives j) -- A is J x M."""
    return float(A.min(axis=0).max())


def gamma_mix(A: np.ndarray) -> float:
    """LP: maximise t s.t. A^T pi >= t*1, sum(pi)=1, pi >= 0."""
    J, M = A.shape
    c = np.zeros(M + 1)
    c[-1] = -1.0                                    # minimise -t
    A_ub = np.hstack([-A, np.ones((J, 1))])         # -A pi + t <= 0
    b_ub = np.zeros(J)
    A_eq = np.hstack([np.ones((1, M)), np.zeros((1, 1))])
    b_eq = np.array([1.0])
    bounds = [(0, None)] * M + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    assert res.success, res.message
    return float(res.x[-1])


def main() -> None:
    K, n = 6, 2
    s_star = np.array([0, 1])
    dirs = swap_directions(s_star, K)
    print(f"K={K} n={n} B={B_BUDGET} S*={s_star.tolist()}  #alternatives={len(dirs)}")
    print(f"{'Sigma':<22}{'Gamma_single':>14}{'Gamma_mix':>12}{'ratio':>9}")

    idx = np.arange(K)
    sigmas = {
        "identity": np.eye(K),
        "equicorr(0.5)": 0.5 * np.eye(K) + 0.5 * np.ones((K, K)),
        "equicorr(0.9)": 0.1 * np.eye(K) + 0.9 * np.ones((K, K)),
        "toeplitz(0.5)": 0.5 ** np.abs(idx[:, None] - idx[None, :]),
    }
    C = sample_directions(K, M=50000, seed=0)
    for name, sigma in sigmas.items():
        A = divergence_matrix(dirs, C, sigma)
        gs, gm = gamma_single(A), gamma_mix(A)
        print(f"{name:<22}{gs:>14.3e}{gm:>12.5f}{gm / max(gs,1e-12):>9.3f}")

    # decisive minimal check: two designs that each cover one alternative only.
    # max_c min_j is then small, but the 50/50 mixture covers both.
    C2 = np.zeros((2, K))
    C2[0, 0], C2[0, 3] = SIGNAL, -SIGNAL
    C2[1, 1], C2[1, 4] = SIGNAL, -SIGNAL
    A2 = divergence_matrix(dirs, C2, np.eye(K))
    gs2, gm2 = gamma_single(A2), gamma_mix(A2)
    print(f"\nhand-built counterexample (2 designs): Gamma_single={gs2:.4f} "
          f"Gamma_mix={gm2:.4f}")
    assert gm2 > gs2 + 1e-6, "mixture must beat a fixed measurement here"
    print("self-check ok: Gamma_mix strictly exceeds Gamma_single")


if __name__ == "__main__":
    main()
