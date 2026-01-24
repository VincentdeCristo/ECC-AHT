import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from joblib import Parallel, delayed
import seaborn as sns
from scipy.stats import bootstrap
import random

from environment import SimulationWorld
from algorithms import ECC_AHT, RandomSparseProjection, RoundRobin, ECC_AHT_SimpleDiff, ECC_AHT_CostFree, ECC_AHT_Diagonal

N_runs = 20    # Number of repetitions
SEEDS = [42 + i for i in range(N_runs)]

def calculate_f1(S_hat, S_true):
    """Calculate the F1-Score given estimated and true sets of anomalies."""
    S_hat_set = set(S_hat)
    S_true_set = set(S_true)
    
    true_positives = len(S_hat_set.intersection(S_true_set))
    
    if true_positives == 0:
        return 0.0
        
    precision = true_positives / len(S_hat_set)
    recall = true_positives / len(S_true_set)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def run_single_experiment(env_class, algo_class, K, n, mu_signal, Sigma, B, T_steps, seed):
    """
    Run a complete simulation (one algorithm, one world)
    Returns a list of F1-scores of length T_steps.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    env = env_class(K, n, mu_signal, Sigma)
    algo = algo_class(K, n, Sigma, B, mu_signal)
    
    f1_scores = []
    
    for t in range(T_steps):
        # 1. Algorithm selects action.
        C_t = algo.select_action()
        
        # 2. Environment provides observations.
        y_t = env.get_observation(C_t)

        # 3. Algorithm updates beliefs.
        algo.update(C_t, y_t)

        # 4. Evaluate F1-Score
        S_hat = algo.get_S_hat()
        f1 = calculate_f1(S_hat, env.S_true)
        f1_scores.append(f1)
        
    return f1_scores

def plot_results(results, title, name):
    """Plotting the final comparison chart, using SciPy's BCa Bootstrap 95% confidence intervals"""
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

        # Perform BCa Bootstrap at each time step
        for t in range(T_steps):
            column = scores[:, t]  # The t-th time step of all runs

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

        # Plotting the mean curve
        ax.plot(
            t_axis, mean_scores,
            label=algo_name,
            lw=2,
            color=colors[idx]
        )

        # Draw BCa CI shaded area
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

# Running multiple experiments of a single algorithm in parallel
def run_algorithm_multiple_times(algo_name, algo_class, K, n, mu_signal, Sigma, B, T_steps, N_runs, seeds):
    """
    Run an algorithm N_runs times, using parallel processing for acceleration.
    """
    print(f"Running {algo_name} with {N_runs} runs in parallel...")
    
    # Run N_runs times in parallel using joblib
    # n_jobs=-1 indicates that all CPU cores are being used.
    # verbose=5 display progress bar
    all_scores = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_single_experiment)(
            SimulationWorld, algo_class, K, n, mu_signal, Sigma, B, T_steps, seeds[run_idx]
        )
        for run_idx in range(N_runs)
    )
    
    return all_scores

def main():
    # --- Experimental parameters ---
    K = 100         # 100 streams
    n = 3           # 3 anomalies
    mu_signal = 5.0 # Signal strength
    B = 4.0         # L1 constraint (sparsity)
    T_steps = 500   # Total observation steps

    algorithms = {
        "ECC-AHT (Ours)": ECC_AHT,
        "Random Sparse (RSP)": RandomSparseProjection,
        "Round Robin (RR)": RoundRobin
    }

    # --- Experiment 1: Independent streams (Sigma = I) ---
    print("=" * 60)
    print("--- Running Experiment 1 (Sigma = I) ---")
    print("=" * 60)
    Sigma_1 = np.eye(K)
    results_exp1 = {name: [] for name in algorithms}
    
    # For each algorithm, run it in parallel N_runs times.
    for name, algo_class in algorithms.items():
        results_exp1[name] = run_algorithm_multiple_times(
            name, algo_class, K, n, mu_signal, Sigma_1, B, T_steps, N_runs, SEEDS
        )
    
    plot_results(results_exp1, r"Sample Efficiency (Independent Streams, $\Sigma = I$)", "Experiment_1_1.pdf")

    # --- Experiment 2: Correlated Streams (Toeplitz) ---
    print("\n" + "=" * 60)
    print("\n--- Running Experiment 1 (Correlated Sigma) ---")
    print("=" * 60)
    correlation = 0.5
    Sigma_2 = toeplitz(correlation ** np.arange(K))
    results_exp2 = {name: [] for name in algorithms}

    for name, algo_class in algorithms.items():
        results_exp2[name] = run_algorithm_multiple_times(
            name, algo_class, K, n, mu_signal, Sigma_2, B, T_steps, N_runs, SEEDS
        )

    plot_results(results_exp2, r"Sample Efficiency (Correlated Streams, $\Sigma_{ij} = 0.5^{|i-j|}$)", f"Experiment_1_2.pdf")

    # --- Experiment 3: Large-scale ---
    print("\n--- Running Experiment 1 (Large-scale experiments) ---")
    K = 1000         # 1000 streams
    n = 10           # 10 anomalies
    mu_signal = 3.0 # Signal strength
    B = 10.0         # L1 constraint (sparsity)
    T_steps = 5000   # Total observation steps
    correlation = 0.8
    Sigma_3 = toeplitz(correlation ** np.arange(K))
    results_exp3 = {name: [] for name in algorithms}
    
    for name, algo_class in algorithms.items():
        results_exp3[name] = run_algorithm_multiple_times(
            name, algo_class, K, n, mu_signal, Sigma_3, B, T_steps, N_runs, SEEDS
        )

    plot_results(results_exp3, r"Sample Efficiency (Large-scale Streams, $\Sigma_{ij} = 0.8^{|i-j|}$)", f"Experiment_1_3.pdf")

    # --- Experiment 4: Ablation Study ---
    K = 100
    n = 3
    mu_signal = 5.0
    B = 4.0
    T_steps = 500

    # Strongly correlated matrix
    correlation = 0.8
    Sigma_corr = toeplitz(correlation ** np.arange(K))

    # Register all participants
    ablation_algorithms = {
        "Full ECC-AHT (Ours)": ECC_AHT,
        "Ablation (No-QP / Simple-Diff)": ECC_AHT_SimpleDiff,
        "Ablation (No-Active / RSP)": RandomSparseProjection,
        "Ablation (No-Correlation / Diagonal)": ECC_AHT_Diagonal,
        "Benchmark (Cost-Free / No-Sparsity)": ECC_AHT_CostFree
    }

    print("--- Running Experiment 2 (Ablation Study) ---")
    results_exp4 = {name: [] for name in ablation_algorithms}

    for name, algo_class in ablation_algorithms.items():
        results_exp4[name] = run_algorithm_multiple_times(
            name, algo_class, K, n, mu_signal, Sigma_corr, B, T_steps, N_runs, SEEDS
        )

    plot_results(results_exp4, r"Ablation Study (Correlated Streams, $\Sigma_{ij} = 0.8^{|i-j|}$)", f"Experiment_2.pdf")

if __name__ == "__main__":
    main()