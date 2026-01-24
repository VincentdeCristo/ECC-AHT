import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, circulant
import seaborn as sns
from scipy.stats import bootstrap
import random
import networkx as nx
from environment import SimulationWorld
from algorithms import ECC_AHT, RandomSparseProjection, BaseArm_CombGapE, TTTS_Challenger, CombGapE, TTTS, RoundRobin

# =========================
# Global parameters
# =========================
N_runs = 20
SEEDS = [42 + i for i in range(N_runs)]

# =========================
# Tool functions
# =========================
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

# =========================
# Correlation matrix factory
# =========================
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

# =========================
# Plotting
# =========================
def plot_results(results, title, name):
    """Generate the final comparison plot using SciPy's BCa Bootstrap 95% confidence interval."""
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

    fig, ax = plt.subplots(figsize=(10, 6))

    algo_names = list(results.keys())
    colors = sns.color_palette("colorblind", n_colors=len(algo_names))

    for idx, (algo_name, scores) in enumerate(results.items()):
        scores = np.asarray(scores)  # shape = (N_runs, T_steps)

        mean_scores = scores.mean(axis=0)
        T_steps = mean_scores.shape[0]
        t_axis = np.arange(T_steps)

        ci_lower = np.zeros(T_steps)
        ci_upper = np.zeros(T_steps)

        for t in range(T_steps):
            column = scores[:, t]  # All runs at time step t

            try:
                res = bootstrap(
                    data=(column,),
                    statistic=np.mean,
                    confidence_level=0.95,
                    n_resamples=10000,
                    method="BCa",
                    vectorized=False
                )
                ci_lower[t] = res.confidence_interval.low
                ci_upper[t] = res.confidence_interval.high
            except Exception as e:
                # Fallback to percentile method if bootstrap fails
                ci_lower[t] = np.percentile(column, 2.5)
                ci_upper[t] = np.percentile(column, 97.5)

        ax.plot(
            t_axis, mean_scores,
            label=algo_name,
            lw=2,
            color=colors[idx]
        )

        ax.fill_between(
            t_axis, ci_lower, ci_upper,
            alpha=0.25,
            color=colors[idx]
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel("Number of Samples (t)")
    ax.set_ylabel("F1-Score")
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_ylim(0, 1.05)
    
    metadata_str = f"N_runs={N_runs}, Seeds={SEEDS[0]}..{SEEDS[-1]}"
    props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='lightgrey')
    ax.text(0.5, 0.02, metadata_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='center', bbox=props)

    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

# =========================
# Main experiment
# =========================
def main():
    K = 128
    n = 5
    mu_signal = 3.0
    B = 5.0
    T_steps = 2000
    rho = 0.8

    correlation_modes = [
        "Toeplitz",
        "Equicorrelation",
        "Block",
        "Circulant",
        "Graph",
        "Exponential",
        "RBF",
        "Kronecker"
    ]

    algorithms = {
        "ECC-AHT (Ours)": (ECC_AHT, {}),
        "BaseArm-CombGapE": (BaseArm_CombGapE, {}),
        "TTTS-Challenger": (TTTS_Challenger, {}),
        "Random Sparse Projection": (RandomSparseProjection, {}),
        "Round Robin": (RoundRobin, {}),
        "CombGapE (Nakamura and Sugiyama, 2025)": (CombGapE, {}),
        "TTTS (Russo, 2016)": (TTTS, {})
    }

    for mode in correlation_modes:
        print(f"\n=== Correlation Pattern: {mode} ===")
        Sigma = make_covariance(K, rho, mode)
        results = {name: [] for name in algorithms}

        for run in range(N_runs):
            np.random.seed(SEEDS[run])
            random.seed(SEEDS[run])
            env = SimulationWorld(K, n, mu_signal, Sigma)

            for name, (algo_class, params) in algorithms.items():
                algo = algo_class(K, n, Sigma, B, mu_signal, **params)
                scores = run_single_experiment(env, algo, T_steps)
                results[name].append(scores)

        plot_results(
            results,
            rf"Correlation Patterns ({mode}, $\rho={rho}$)",
            f"CorrPattern_{mode}.pdf"
        )

if __name__ == "__main__":
    main()
