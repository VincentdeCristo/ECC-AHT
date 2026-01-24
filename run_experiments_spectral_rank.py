import numpy as np
from scipy.linalg import toeplitz, circulant
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import random
from algorithms import ECC_AHT
from environment import SimulationWorld

N_runs = 20
SEEDS = [42 + i for i in range(N_runs)]

def calculate_f1(S_hat, S_true):
    S_hat, S_true = set(S_hat), set(S_true)
    if not S_hat or not S_true:
        return 0.0
    tp = len(S_hat & S_true)
    p = tp / len(S_hat)
    r = tp / len(S_true)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

def run_single_experiment(env, algo, T_steps):
    f1_scores = []
    for t in range(T_steps):
        if algo.is_terminated():
            f1_scores.append(f1_scores[-1] if f1_scores else 0.0)
            continue
        C_t = algo.select_action()
        if C_t is None:
            f1_scores.append(f1_scores[-1] if f1_scores else 0.0)
            continue
        y_t = env.get_observation(C_t)
        algo.update(C_t, y_t)
        f1_scores.append(calculate_f1(algo.get_S_hat(), env.S_true))
    return f1_scores

def effective_rank_shannon(Sigma, eps=1e-12):
    """
    Shannon effective rank:
    r_eff = exp( - sum p_i log p_i ), p_i = lambda_i / sum lambda
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.maximum(eigvals, eps)
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def effective_rank_pr(Sigma, eps=1e-12):
    """
    Participation ratio:
    r_eff = (sum lambda)^2 / sum lambda^2
    """
    eigvals = np.linalg.eigvalsh(Sigma)
    eigvals = np.maximum(eigvals, eps)
    return (eigvals.sum() ** 2) / np.sum(eigvals ** 2)

def make_covariance(K, rho, mode):
    if mode == "Toeplitz":
        return toeplitz(rho ** np.arange(K))

    if mode == "Equicorrelation":
        return (1 - rho) * np.eye(K) + rho * np.ones((K, K))

    if mode == "Block":
        block = 16
        Sigma = np.eye(K)
        for i in range(0, K, block):
            end = min(i + block, K)
            Sigma[i:end, i:end] = (1 - rho) * np.eye(end - i) + rho
        return Sigma

    if mode == "Circulant":
        c = rho ** np.minimum(np.arange(K), K - np.arange(K))
        return circulant(c)

    if mode == "Graph":
        G = nx.erdos_renyi_graph(K, p=0.05, seed=0)
        A = nx.to_numpy_array(G)
        # Get the maximum eigenvalue (spectral radius)
        lam_max = np.max(np.abs(np.linalg.eigvalsh(A)))
        # Make sure rho_eff < 1/lam_max. Here rho is mapped to (0, 0.95/lam_max)
        rho_eff = (rho * 0.95) / lam_max
        
        Sigma_inv = np.eye(K) - rho_eff * A
        Sigma = np.linalg.inv(Sigma_inv)
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(Sigma))
        return Sigma / np.outer(d, d)
    
    if mode == "Exponential":
        # Assume arm is on a 1D sequence
        idx = np.arange(K)
        # dist[i,j] = |i-j|
        dist = np.abs(idx[:, None] - idx[None, :])
        return rho ** dist
    
    if mode == "RBF":
        idx = np.arange(K)
        dist_sq = (idx[:, None] - idx[None, :])**2
        # Map rho to length_scale. The larger the rho, the stronger the correlation.
        length_scale = K * (1 - rho + 1e-5) 
        return np.exp(-dist_sq / (2 * length_scale**2))
    
    if mode == "Kronecker":
        # Decompose K into the product of two small matrices
        K1, K2 = 16, 8 # 128 = 16 * 8
        S1 = (1 - rho) * np.eye(K1) + rho * np.ones((K1, K1))
        S2 = toeplitz(rho ** np.arange(K2))
        return np.kron(S1, S2)

    raise ValueError("Unknown correlation mode")

def experiment_1_effective_rank_patterns():
    K = 128
    rho = 0.8

    patterns = [
        "Toeplitz",
        "Equicorrelation",
        "Block",
        "Circulant",
        "Graph",
        "Exponential",
        "RBF",
        "Kronecker"
    ]

    print("\n=== Experiment 1: Effective Rank of Correlation Patterns ===")
    print("Pattern\t\tShannon ER\tPR ER")

    results = {}

    for mode in patterns:
        Sigma = make_covariance(K, rho, mode)
        er_shannon = effective_rank_shannon(Sigma)
        er_pr = effective_rank_pr(Sigma)

        results[mode] = (er_shannon, er_pr)

        print(f"{mode:15s}\t{er_shannon:8.2f}\t{er_pr:8.2f}")

    return results

def make_spectral_mixture(K, alpha, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=K)
    u = u / np.linalg.norm(u)
    Sigma_low_rank = np.outer(u, u)
    Sigma = (1 - alpha) * np.eye(K) + alpha * Sigma_low_rank
    return Sigma

def experiment_2_f1_vs_effective_rank():
    K = 128
    n = 5
    mu_signal = 3.0
    B = 5.0
    T_steps = 2000

    alphas = np.linspace(0.95, 1, 30)

    algo_name = "ECC-AHT (Ours)"
    algo_class = ECC_AHT

    f1_means = []
    f1_ci_lower = []
    f1_ci_upper = []
    er_values = []

    for alpha in alphas:
        Sigma = make_spectral_mixture(K, alpha)
        er = effective_rank_shannon(Sigma)
        er_values.append(er)

        f1_runs = []

        for run in range(N_runs):
            np.random.seed(SEEDS[run])
            random.seed(SEEDS[run])
            env = SimulationWorld(K, n, mu_signal, Sigma)
            algo = algo_class(K, n, Sigma, B, mu_signal)

            scores = run_single_experiment(env, algo, T_steps)
            f1_runs.append(scores[-1])  # final F1

        data_arr = np.array(f1_runs)
        f1_means.append(np.mean(data_arr))

        res = bootstrap((data_arr,), np.mean, confidence_level=0.95, 
                            n_resamples=10000, method='BCa')
        f1_ci_lower.append(res.confidence_interval.low)
        f1_ci_upper.append(res.confidence_interval.high)

    # ===== Plot =====
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'grid.alpha': 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        'mathtext.fontset': 'stix',
        "font.family": "Times New Roman"
    })
    plt.figure(figsize=(10, 6))
    plt.plot(er_values, f1_means, marker='o', lw=2, label=algo_name)
    plt.fill_between(er_values, f1_ci_lower, f1_ci_upper, alpha=0.2, label='95% BCa CI')
    plt.xlabel("Effective Rank (Shannon)")
    plt.ylabel("Final F1 Score")
    plt.title("F1 vs Effective Rank (Spectral Mixing)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("Exp2_F1_vs_EffectiveRank.pdf", bbox_inches="tight")

def experiment_3_rbf_lengthscale():
    K = 128
    n = 5
    mu_signal = 3.0
    B = 5.0
    T_steps = 2000

    length_scales = np.concatenate([np.linspace(0.1, 5, 30), np.linspace(10, 75.0, 10)])

    f1_means = []
    er_values = []
    f1_ci_lower = []
    f1_ci_upper = []

    idx = np.arange(K)

    for ell in length_scales:
        dist_sq = (idx[:, None] - idx[None, :]) ** 2
        Sigma = np.exp(-dist_sq / (2 * ell ** 2))

        algo_name = "ECC-AHT (Ours)"
        algo_class = ECC_AHT

        er = effective_rank_shannon(Sigma)
        er_values.append(er)

        f1_runs = []

        for run in range(N_runs):
            np.random.seed(SEEDS[run])
            random.seed(SEEDS[run])
            env = SimulationWorld(K, n, mu_signal, Sigma)
            algo = algo_class(K, n, Sigma, B, mu_signal)

            scores = run_single_experiment(env, algo, T_steps)
            f1_runs.append(scores[-1])

        data_arr = np.array(f1_runs)
        f1_means.append(np.mean(data_arr))
        res = bootstrap((data_arr,), np.mean, confidence_level=0.95, 
                            n_resamples=10000, method='BCa')
        f1_ci_lower.append(res.confidence_interval.low)
        f1_ci_upper.append(res.confidence_interval.high)

    # ===== Plot =====
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'grid.alpha': 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        'mathtext.fontset': 'stix',
        "font.family": "Times New Roman"
    })
    plt.figure(figsize=(10, 6))
    plt.plot(er_values, f1_means, marker='s', lw=2, label=algo_name)
    plt.fill_between(er_values, f1_ci_lower, f1_ci_upper, alpha=0.2, label='95% BCa CI')
    plt.xlabel("Effective Rank (Shannon)")
    plt.ylabel("Final F1 Score")
    plt.title("F1 vs Effective Rank (RBF Kernel)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("Exp3_RBF_F1_vs_EffectiveRank.pdf", bbox_inches="tight")

def main():
    experiment_1_effective_rank_patterns()
    experiment_2_f1_vs_effective_rank()
    experiment_3_rbf_lengthscale()


if __name__ == "__main__":
    main()