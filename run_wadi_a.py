import numpy as np
import pandas as pd
import pickle
import time
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
import seaborn as sns
from scipy.stats import bootstrap
from algorithms import ECC_AHT, ECC_AHT_Diagonal, TTTS_Challenger, RandomSparseProjection, ECC_AHT_SimpleDiff, BaseArm_CombGapE, RoundRobin
from preprocess_wadi import read_wadi_csv, coerce_and_fill 

# --- Configuration ---
MODEL_PATH = f"wadi_h0_model_{'1min'}_windowed.pkl" 
ATTACK_CSV_PATH = "./WADI.A1_9 Oct 2017/WADI_attackdata.csv"

T_MAX_DELAY = 20161    
N_RUNS = 20        # Number of runs
B_BUDGET = 5.0     
DELTA_SIGNAL = 1.0 
F1_ALARM_THRESHOLD = 0.95

# Explicit Random Seeds for Reproducibility
SEEDS = [42 + i for i in range(N_RUNS)]

# Ignore the warning.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helper functions ---

def calculate_f1(S_hat, S_true_set):
    """Calculate the F1-Score"""
    S_hat_set = set(S_hat)
    if not S_hat_set or not S_true_set:
        return 0.0
    true_positives = len(S_hat_set.intersection(S_true_set))
    precision = true_positives / len(S_hat_set)
    recall = true_positives / len(S_true_set)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def normalize_sensor_name(name):
    """
    Normalize sensor names for matching purposes.
    Remove all non-alphanumeric characters and convert to uppercase.
    For example: "1_MV_001" -> "1MV001", "1-P-005" -> "1P005"
    """
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())

def bootstrap_ci(data, ci=95, n_boot=10000):
    """
    Return (error_low, error_high) for the mean of the data using bootstrap.
    Use SciPy's built-in BCa (Bias-Corrected and Accelerated) confidence interval.
    """
    data = np.asarray(data)
    if len(data) < 2:
        return 0.0, 0.0

    # Convert CI percentage to a value between 0 and 1.
    confidence_level = ci / 100.0

    # SciPy's bootstrap function requires a tuple of the form (data,) as input.
    res = bootstrap(
        data=(data,),
        statistic=np.mean,
        confidence_level=confidence_level,
        n_resamples=n_boot,
        method='BCa',
        vectorized=False     # np.mean is not vectorized
    )

    low = res.confidence_interval.low
    high = res.confidence_interval.high
    mean = np.mean(data)

    return mean - low, high - mean

# --- Manual Attack List ---
'''
Some attacks were removed here 
because their corresponding columns were deleted in preprocess_wadi.py, 
making their detection meaningless.
'''
MANUAL_ATTACKS = [
    (1.0, "2017-10-09 19:25:00", "2017-10-09 19:50:16", "1_MV_001 OPEN, Overflow of primary grid tank", 1),
    (2.0, "2017-10-10 10:24:10", "2017-10-10 10:34:00", "1_FIT_001 False readings", 1),
    (3.0, "2017-10-10 10:55:00", "2017-10-10 11:24:00", "2_LT_002 Stealthy attack", 1),
    (4.0, "2017-10-10 11:07:46", "2017-10-10 11:12:15", "1_AIT_001 False readings, Raw water tank drain", 1),
    #(5.0, "2017-10-10 11:30:40", "2017-10-10 11:44:50", "2_MCV_101, 2_MCV_201, 2_MCV_301, 2_MCV_401, 2_MCV_501, 2_MCV_601", 6),
    #(6.0, "2017-10-10 13:39:30", "2017-10-10 13:50:40", "2_MCV_101, 2_MCV_201 Open to 50%", 2),
    (7.0, "2017-10-10 14:48:17", "2017-10-10 14:53:44", "1_AIT_002, Supply contaminated water", 1),
    (7.1, "2017-10-10 14:53:44", "2017-10-10 14:59:55", "1_AIT_002 and 2_MV_003, Supply contaminated water", 2),
    (7.2, "2017-10-10 14:59:55", "2017-10-10 15:00:32", "2_MV_003, Supply contaminated water", 1),
    #(8.0, "2017-10-10 17:40:00", "2017-10-10 17:49:40", "2_MCV_007 Water leakage", 1),
    (9.0, "2017-10-11 10:55:00", "2017-10-11 10:56:27", "1-P-005 Pipe bursts", 1),
    (10.0, "2017-10-11 11:17:54", "2017-10-11 11:31:20", "1_MV_001 Randomized attack", 1),
    #(11.0, "2017-10-11 11:36:31", "2017-10-11 11:47:00", "2_MCV_007 Set at 50%, Booster never on", 1),
    #(12.0, "2017-10-11 11:59:00", "2017-10-11 12:05:00", "2_MCV_007 Set at 100%, Waste water", 1),
    (13.0, "2017-10-11 12:07:30", "2017-10-11 12:10:52", "2_PIC_003 Set point to 0.25 bar", 1),
    (14.0, "2017-10-11 12:16:00", "2017-10-11 12:25:36", "1-P-001 and 1-P-003 OFF, Stop chemical dosing", 2),
    (15.0, "2017-10-11 15:26:30", "2017-10-11 15:37:00", "2_LT_002 Stealthy attack, Overflow ER tank", 1)
]


# --- 1. WaDi Environmental Category ---

class WadiAttackEnvironment:
    def __init__(self, model_path, attack_csv_path):
        print(f"Loading WaDi H0 model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.K = len(self.model['mu_0'])
        self.window_size = self.model['window_size']
        self.model_cols = self.model['sensor_columns']
        self.scaler = self.model['scaler']
        
        print(f"Model loaded: K={self.K}, Window={self.window_size}")
        print(f"Loading and pre-processing attack data from {attack_csv_path}...")
        df_raw = read_wadi_csv(attack_csv_path)
        self.data = coerce_and_fill(df_raw, self.model_cols)

        # --- Robust Ground Truth Matching ---
        print(f"Loading MANUALLY defined attack descriptions...")
        self.attacks = {}
        
        # Precompute all column names in normalized form for faster lookup
        normalized_cols = {normalize_sensor_name(col): i for i, col in enumerate(self.model_cols)}
        
        for (s_no, start_str, end_str, desc, n_val) in MANUAL_ATTACKS:
            try:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)
                
                if start_ts < self.data.index.min() or end_ts > self.data.index.max():
                    continue
                
                # 1. Extract short names
                possible_names = re.findall(r"([12](?:_|-|)[A-Z]{1,3}(?:_|-|)\d{3})", desc)
                
                s_true = set()
                # 2. Normalize and match
                for p_name in possible_names:
                    norm_p_name = normalize_sensor_name(p_name)
                    matched = False
                    for col_norm, col_idx in normalized_cols.items():
                        if norm_p_name in col_norm:
                            s_true.add(col_idx)
                            matched = True
                    
                    if not matched: 
                         pass

                # 3. Validate n
                if len(s_true) < n_val:
                     print(f"  Warning: S.No {s_no} (Desc: {desc[:30]}...) Expected n={n_val}, but only matched {len(s_true)} columns.")
                
                self.attacks[s_no] = {
                    'id': s_no,
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'n': n_val,
                    's_true': s_true
                }
            except Exception as e:
                print(f"  Skipping attack {s_no}: {e}")
        
        print(f"Successfully loaded {len(self.attacks)} scenarios with ground truth.")

    def get_attack_window(self, attack_id):
        attack = self.attacks[attack_id]
        start_ts = attack['start_ts']
        window_duration = pd.to_timedelta(self.window_size) 
        pre_start_ts = start_ts
        max_end_ts = start_ts + (T_MAX_DELAY * window_duration)
        
        if pre_start_ts < self.data.index.min(): pre_start_ts = self.data.index.min()
        if max_end_ts > self.data.index.max(): max_end_ts = self.data.index.max()
            
        run_data = self.data.loc[pre_start_ts : max_end_ts].copy()
        return run_data, start_ts

    def get_all_attack_ids(self):
        return sorted(list(self.attacks.keys()))

# --- 2. Run Scenarios ---

def run_scenario(env, algo_class, params, attack_id):
    try:
        run_data, start_ts = env.get_attack_window(attack_id)
        if run_data.empty: return np.nan
    except: return np.nan

    df_windows_raw = run_data.resample(env.window_size).mean().dropna(how='all')
    X_windows_scaled = env.scaler.transform(df_windows_raw.to_numpy())
    
    constructor_params = params.copy()
    
    algo = algo_class(
        K=env.K, mu_0=env.model['mu_0'], Sigma=env.model['Sigma_reg'], **constructor_params
    )
    
    s_true_set = env.attacks[attack_id].get('s_true', set())
    t_start_idx = -1
    
    for t, (timestamp, x_t) in enumerate(zip(df_windows_raw.index, X_windows_scaled)):
        if t_start_idx == -1 and timestamp >= start_ts:
            t_start_idx = t
            
        C_t = algo.select_action()
        if C_t is None:
            if algo.is_terminated():
                return t - t_start_idx if t_start_idx != -1 else 0
            else: continue 
        
        y_t = np.dot(C_t, x_t)
        algo.update(C_t, y_t) 
        
        # F1 Check
        alarm = False
        if hasattr(algo, 'get_S_hat'):
            S_hat = algo.get_S_hat()
            f1 = calculate_f1(S_hat, s_true_set)
            if f1 >= F1_ALARM_THRESHOLD:
                alarm = True
        
        if alarm and t_start_idx != -1:
            return t - t_start_idx
            
    return T_MAX_DELAY

# --- 3. Main Program ---

def main():
    t_start = time.time()
    env = WadiAttackEnvironment(MODEL_PATH, ATTACK_CSV_PATH)
    
    ALGORITHMS_TO_TEST = {
        r"ECC-AHT (Full $\Sigma$)": (ECC_AHT, {"B": B_BUDGET}),
        r"ECC-AHT (Diagonal $\Sigma$)": (ECC_AHT_Diagonal, {"B": B_BUDGET}),
        "TTTS Challenger": (TTTS_Challenger, {"B": B_BUDGET}),
        "Random Sparse Projection": (RandomSparseProjection, {"B": B_BUDGET}),
        "ECC-AHT SimpleDiff": (ECC_AHT_SimpleDiff, {"B": B_BUDGET}),
        "BaseArm CombGapE": (BaseArm_CombGapE, {"B": B_BUDGET}),
        "Round Robin": (RoundRobin, {"B": B_BUDGET})
    }

    results = {name: [] for name in ALGORITHMS_TO_TEST}
    attack_ids = env.get_all_attack_ids()
    
    print(f"\n--- Phase 3a: Robustness Evaluation (N_runs={N_RUNS}) ---")
    
    # Using a random seed for the loop.
    for run in tqdm(range(N_RUNS), desc="Total Runs"):
        
        # Set a global random seed to ensure reproducibility.
        current_seed = SEEDS[run]
        np.random.seed(current_seed)
        random.seed(current_seed)
        
        for attack_id in tqdm(attack_ids, desc="Scenarios", leave=False):
            n_val = env.attacks[attack_id].get('n', 1)
            
            for name, (algo_class, base_params) in ALGORITHMS_TO_TEST.items():
                params = base_params.copy()
                params["n"] = n_val
                params["delta_signal"] = DELTA_SIGNAL
                
                delay = run_scenario(env, algo_class, params, attack_id)
                
                if not np.isnan(delay):
                    results[name].append(delay)

    print("\n--- Generating Visualization ---")
    
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
    data_list = [results[name] for name in algo_names]
    
    colors = sns.color_palette("colorblind", n_colors=len(algo_names))
        
    x_positions = np.arange(len(algo_names))
    
    for idx, (name, data) in enumerate(zip(algo_names, data_list)):
        if not data: continue
        
        mean_val = np.mean(data)
        err_low, err_high = bootstrap_ci(data)
        
        ax.fill_between(
            [x_positions[idx] - 0.25, x_positions[idx] + 0.25],
            mean_val - err_low,
            mean_val + err_high,
            color=colors[idx],
            alpha=0.25,
            linewidth=0
        )
        
        ax.plot(
            x_positions[idx], mean_val,
            marker='o',
            markersize=10,
            markerfacecolor='white',
            markeredgecolor=colors[idx],
            markeredgewidth=2.2,
            zorder=5
        )
        
        ci_text = f"{mean_val:.1f}\n[{mean_val-err_low:.1f}, {mean_val+err_high:.1f}]"
        ax.text(x_positions[idx], mean_val + err_high + 1,
                ci_text, 
                ha='center', va='bottom', 
                fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=colors[idx], alpha=0.2))
        
        print(f"{name}: Mean={mean_val:.2f}, 95% CI=[{mean_val-err_low:.2f}, {mean_val+err_high:.2f}]")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([n.replace(" ", "\n") for n in algo_names])
    ax.set_ylabel(rf"Avg. Identification Delay (F1 $\geq$ {F1_ALARM_THRESHOLD}, Window={env.window_size})")
    ax.set_xlabel("Algorithms")
    
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_axisbelow(True)
    
    title_str = f"Anomaly Identification Delay (WADI)"
    metadata_str = f"N_runs={N_RUNS}, Seeds={SEEDS[0]}..{SEEDS[-1]}, Budget={B_BUDGET}"
    ax.set_title(title_str, fontweight='bold', pad=20)
    
    props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='lightgrey')
    ax.text(0.98, 0.02, metadata_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    out_file = "wadi_a_results.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"\nThe resulting image has been saved as {out_file}")
    print(f"Total time: {(time.time() - t_start) / 60.0:.2f} minutes")

if __name__ == "__main__":
    main()