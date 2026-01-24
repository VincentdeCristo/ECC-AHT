import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import seaborn as sns
from scipy.stats import bootstrap
import random

from environment import SimulationWorld
from algorithms import ECC_AHT
from algorithms import HDS_Gafni # SOTA Baseline
from algorithms import ECC_AHT_Restricted

N_runs = 20    # Number of repetitions
SEEDS = [42 + i for i in range(N_runs)]

def calculate_f1(S_hat, S_true):
    """Calculate the F1-Score given estimated and true sets of anomalies."""
    S_hat_set = set(S_hat)
    S_true_set = set(S_true)
    
    if not S_hat_set or not S_true_set:
        return 0.0
    
    true_positives = len(S_hat_set.intersection(S_true_set))
    
    precision = true_positives / len(S_hat_set)
    recall = true_positives / len(S_true_set)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def run_single_experiment(env, algo, T_steps):
    """
    Run a complete simulation (one algorithm, one world)
    Returns a list of F1-Scores of length T_steps
    """
    f1_scores = []
    
    for t in range(T_steps):
        if algo.is_terminated():
            # The algorithm terminated prematurely; the final score was used to fill in the remaining values.
            f1_scores.append(f1_scores[-1] if f1_scores else 0.0)
            continue
            
        # 1. Algorithm selects action.
        C_t = algo.select_action()
        if C_t is None: # HDS may return None.
             continue
            
        # 2. Environment gives observation
        y_t = env.get_observation(C_t)
        
        # 3. Algorithm updates belief
        algo.update(C_t, y_t)
        
        # 4. Evaluate F1-Score
        S_hat = algo.get_S_hat()
        f1 = calculate_f1(S_hat, env.S_true)
        f1_scores.append(f1)
        
    return f1_scores

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
            column = scores[:, t]  # The t-th time step of all runs

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

def main():
    # --- Experiment 4: Continuous Space vs. Discrete Tree (HDS Challenge) ---
    K = 128         # Must be a power of 2
    n = 5           
    mu_signal = 5.0
    B = 5.0         
    T_steps = 1500  # Increase the number of steps
    N_runs = 20     
    
    # The experimental setting is Sigma=I (independent)
    # This is the fairest scenario for HDS
    Sigma = np.eye(K)
    
    # HDS specific parameters
    hds_params = {
        'K_l': 5, # Internal nodes sample 5 times per child
        'Theta1_hypotheses': {mu_signal - 2, mu_signal, mu_signal + 2} # Composite hypothesis
    }

    algorithms = {
        "ECC-AHT (Ours, Continuous)": (ECC_AHT, {}),
        "HDS (Gafni '23, Tree-Discrete)": (HDS_Gafni, hds_params),
        "ECC-AHT (Ours, Tree-Restricted)": (ECC_AHT_Restricted, {})
    }

    print("--- Running Experiment 4 (HDS Showdown) ---")
    results_exp4 = {name: [] for name in algorithms}

    for run in range(N_runs):
        print(f"HDS Showdown, Run {run+1}/{N_runs}")
        
        np.random.seed(SEEDS[run])
        random.seed(SEEDS[run])
        
        # Create a fixed world for this run
        env = SimulationWorld(K, n, mu_signal, Sigma)
        
        for name, (algo_class, params) in algorithms.items():
            
            # Instantiate the algorithm
            algo = algo_class(K, n, Sigma, B, mu_signal, **params)
            
            scores = run_single_experiment(env, algo, T_steps)
            
            results_exp4[name].append(scores)

    plot_results(results_exp4, r"Continuous vs. Discrete Action Space ($\Sigma = I$)", "Experiment_4.pdf")

if __name__ == "__main__":
    main()