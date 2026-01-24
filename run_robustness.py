import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import time
import seaborn as sns
from scipy.stats import bootstrap
import random
from environment import SimulationWorld
from algorithms import ECC_AHT, RandomSparseProjection, RoundRobin

N_runs = 20    # Number of repetitions
SEEDS = [42 + i for i in range(N_runs)]

def calculate_f1(S_hat, S_true):
    """Calculate the F1-Score"""
    S_hat_set = set(S_hat)
    S_true_set = set(S_true)
    
    true_positives = len(S_hat_set.intersection(S_true_set))
    
    if len(S_hat_set) == 0 or len(S_true_set) == 0:
        return 0.0
    
    precision = true_positives / len(S_hat_set)
    recall = true_positives / len(S_true_set)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def run_single_robustness_exp(env_class, algo_class, K, n, mu_signal, Sigma, B, T_max):
    """
    Single experiment function: Run until F1-Score >= 0.95 or T_max is reached
    """
    env = env_class(K, n, mu_signal, Sigma)
    algo = algo_class(K, n, Sigma, B, mu_signal)
    
    for t in range(T_max):
        # 1. Algorithm selects action.
        C_t = algo.select_action()
        
        # 2. The environment provides observations.
        y_t = env.get_observation(C_t)
        
        # 3. The algorithm updates beliefs.
        algo.update(C_t, y_t)
        
        # 4. Evaluate the F1-Score
        S_hat = algo.get_S_hat()
        f1 = calculate_f1(S_hat, env.S_true)
        
        # 5. Check the stopping conditions.
        if f1 >= 0.95:
            return t + 1 # Returns the number of steps taken (t starts from 0).
            
    return T_max # If the target is not reached within T_max steps, return T_max.

def plot_robustness_results(results, param_name, param_list, title, name):
    """Plotting sensitivity analysis (t vs. param) — using SciPy's BCa Bootstrap 95% CI"""
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

    for idx, (algo_name, data) in enumerate(results.items()):
        data = np.asarray(data)        # data.shape = (N_runs, len(param_list))
        mean_t = np.mean(data, axis=0)
        
        n_params = len(param_list)
        ci_lower = np.zeros(n_params)
        ci_upper = np.zeros(n_params)

        # Perform BCa Bootstrap for each parameter value.
        for p in range(n_params):
            column = data[:, p]

            try:
                # SciPy bootstrap requires a tuple as input.
                res = bootstrap(
                    data=(column,),
                    statistic=np.mean,
                    confidence_level=0.95,
                    n_resamples=10000,
                    method="BCa",
                    vectorized=False
                )
                ci_lower[p] = res.confidence_interval.low
                ci_upper[p] = res.confidence_interval.high
            except Exception as e:
                # If BCa fails, use the standard percentile method.
                ci_lower[p] = np.percentile(column, 2.5)
                ci_upper[p] = np.percentile(column, 97.5)

        ax.plot(
            param_list, mean_t,
            label=algo_name,
            lw=2, marker='o',
            color=colors[idx]
        )

        ax.fill_between(
            param_list, ci_lower, ci_upper,
            alpha=0.25,
            color=colors[idx]
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel(f"Parameter: {param_name}")
    ax.set_ylabel("Avg. Samples (t) to reach F1=0.95")
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_xticks(param_list)
    ax.set_xticklabels([str(p) for p in param_list])
    
    metadata_str = f"N_runs={N_runs}, Seeds={SEEDS[0]}..{SEEDS[-1]}"
    props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='lightgrey')
    ax.text(0.5, 0.02, metadata_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='center', bbox=props)

    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # --- Experiment 3: Robustness Analysis ---
    
    # Benchmark Configuration
    K_base = 100
    n_base = 3
    mu_signal_base = 3.0
    rho_base = 0.6
    B_base = 5.0
    
    T_max = 20000000   # Maximum number of steps (Time limit exceeded)
    
    algorithms = {
        "ECC-AHT (Ours)": ECC_AHT,
        "RSP (Baseline)": RandomSparseProjection,
        "RR (Baseline)":RoundRobin
    }
    
    # --- 3.1: vs. Signal strength ---
    print("--- Running Robustness 3.1: Varying mu_signal ---")
    param_name_1 = "Signal Strength (mu_signal)"
    param_list_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    results_1 = {name: np.zeros((N_runs, len(param_list_1))) for name in algorithms}
    
    for r in range(N_runs):
        seed = SEEDS[r]
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Run {r+1}/{N_runs}, seed={seed}")
        for i, mu in enumerate(param_list_1):
            Sigma = toeplitz(rho_base ** np.arange(K_base))
            for name, algo_class in algorithms.items():
                t_needed = run_single_robustness_exp(
                    SimulationWorld, algo_class, K_base, n_base, mu, Sigma, B_base, T_max
                )
                results_1[name][r, i] = t_needed
    plot_robustness_results(results_1, param_name_1, param_list_1, "Robustness vs. Signal Strength", "Experiment_3_SignalStrength.pdf")

    # --- 3.2: vs. Correlation ---
    print("\n--- Running Robustness 3.2: Varying rho (Correlation) ---")
    param_name_2 = "Correlation (rho)"
    param_list_2 = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_2 = {name: np.zeros((N_runs, len(param_list_2))) for name in algorithms}
    
    for r in range(N_runs):
        seed = SEEDS[r]
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Run {r+1}/{N_runs}, seed={seed}")
        for i, rho in enumerate(param_list_2):
            Sigma = toeplitz(rho ** np.arange(K_base)) if rho > 0 else np.eye(K_base)
            for name, algo_class in algorithms.items():
                t_needed = run_single_robustness_exp(
                    SimulationWorld, algo_class, K_base, n_base, mu_signal_base, Sigma, B_base, T_max
                )
                results_2[name][r, i] = t_needed
    plot_robustness_results(results_2, param_name_2, param_list_2, "Robustness vs. Stream Correlation (rho)", "Experiment_3_StreamCorrelation.pdf")

    # --- 3.3: vs. Number of Anomalies ---
    print("\n--- Running Robustness 3.3: Varying n (Number of Anomalies) ---")
    param_name_3 = "Number of Anomalies (n)"
    param_list_3 = [1, 2, 3, 4, 5, 6]
    results_3 = {name: np.zeros((N_runs, len(param_list_3))) for name in algorithms}
    
    for r in range(N_runs):
        seed = SEEDS[r]
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Run {r+1}/{N_runs}, seed={seed}")
        for i, n in enumerate(param_list_3):
            Sigma = toeplitz(rho_base ** np.arange(K_base))
            for name, algo_class in algorithms.items():
                t_needed = run_single_robustness_exp(
                    SimulationWorld, algo_class, K_base, n, mu_signal_base, Sigma, B_base, T_max
                )
                results_3[name][r, i] = t_needed
    plot_robustness_results(results_3, param_name_3, param_list_3, "Robustness vs. Number of Anomalies (n)", "Experiment_3_NumberofAnomalies.pdf")

    # --- 3.4: vs. L1 Budget ---
    print("\n--- Running Robustness 3.4: Varying B (L1 Budget) ---")
    param_name_4 = "L1 Budget (B)"
    param_list_4 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    results_4 = {name: np.zeros((N_runs, len(param_list_4))) for name in algorithms}
    
    for r in range(N_runs):
        seed = SEEDS[r]
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"Run {r+1}/{N_runs}, seed={seed}")
        for i, B in enumerate(param_list_4):
            Sigma = toeplitz(rho_base ** np.arange(K_base))
            for name, algo_class in algorithms.items():
                t_needed = run_single_robustness_exp(
                    SimulationWorld, algo_class, K_base, n_base, mu_signal_base, Sigma, B, T_max
                )
                results_4[name][r, i] = t_needed
    plot_robustness_results(results_4, param_name_4, param_list_4, "Robustness vs. L1 Budget (B)", "Experiment_3_L1Budget.pdf")

if __name__ == "__main__":
    main()