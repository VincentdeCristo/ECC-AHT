import numpy as np
import pandas as pd
import pickle
import time
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
from scipy.stats import bootstrap
import seaborn as sns
from algorithms import ECC_AHT, CombGapE
from preprocess_wadi import read_wadi_csv, coerce_and_fill 

# --- Configuration ---
WINDOW_STR = "1min"
WINDOW_SECONDS = 60 

MODEL_PATH = f"wadi_h0_model_{WINDOW_STR}_windowed.pkl"
ATTACK_CSV_PATH = "./WADI.A1_9 Oct 2017/WADI_attackdata.csv"

# Timeout setting (seconds)
MAX_DELAY_SECONDS = 1209601
MAX_STEPS_ECC = 20161 # Corresponds to 20161 windows
MAX_STEPS_COMB = MAX_DELAY_SECONDS 

N_RUNS = 20
B_BUDGET = 5.0
DELTA_SIGNAL = 1.0 # Maintain consistency with a
F1_ALARM_THRESHOLD = 0.95

# Random seed
SEEDS = [42 + i for i in range(N_RUNS)]

# Ignore the warning.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helper Functions ---

def calculate_f1(S_hat, S_true_set):
    S_hat_set = set(S_hat)
    if not S_hat_set or not S_true_set: return 0.0
    tp = len(S_hat_set.intersection(S_true_set))
    p = tp / len(S_hat_set)
    r = tp / len(S_true_set)
    if p + r == 0: return 0.0
    return 2 * p * r / (p + r)

def normalize_sensor_name(name):
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())

def bootstrap_ci(data, ci=95, n_boot=10000):
    data = np.asarray(data)
    if len(data) < 2: return 0.0, 0.0
    res = bootstrap((data,), np.mean, confidence_level=ci/100.0, 
                    n_resamples=n_boot, method='BCa', vectorized=False)
    mean = np.mean(data)
    return mean - res.confidence_interval.low, res.confidence_interval.high - mean

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

# --- 1. Dual Stream Environment Class ---

class WadiGrandChallengeEnv:
    def __init__(self, model_path, attack_csv_path):
        print(f"Loading H0 Model from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.scaler = model['scaler']
        self.mu_0 = model['mu_0']
        self.Sigma_reg = model['Sigma_reg']
        self.sensor_columns = model['sensor_columns']
        self.K = len(self.sensor_columns)
        
        print("Loading Attack CSV...")
        df_raw = read_wadi_csv(attack_csv_path)
        
        print("Preprocessing (Coerce & Fill)...")
        df_clean = coerce_and_fill(df_raw, self.sensor_columns)
        
        print("Scaling Raw Data...")
        X_raw = df_clean.to_numpy()
        X_scaled = self.scaler.transform(X_raw)
        self.df_1sec_scaled = pd.DataFrame(X_scaled, index=df_clean.index, columns=df_clean.columns)
        
        print(f"Generating Windowed ({WINDOW_STR}) Stream...")
        self.df_windowed_scaled = self.df_1sec_scaled.resample(f"{WINDOW_SECONDS}s").mean().dropna(how='all')
        
        print("Matching Ground Truth...")
        self.attacks = {}
        normalized_cols = {normalize_sensor_name(col): i for i, col in enumerate(self.sensor_columns)}
        
        for (s_no, start_str, end_str, desc, n_val) in MANUAL_ATTACKS:
            try:
                start_ts = pd.to_datetime(start_str)
                end_ts = pd.to_datetime(end_str)
                
                if start_ts < self.df_1sec_scaled.index.min() or end_ts > self.df_1sec_scaled.index.max():
                    continue

                possible_names = re.findall(r"([12](?:_|-|)[A-Z]{1,3}(?:_|-|)\d{3})", desc)
                s_true = set()
                for p_name in possible_names:
                    norm_p_name = normalize_sensor_name(p_name)
                    for col_norm, col_idx in normalized_cols.items():
                        if norm_p_name in col_norm:
                            s_true.add(col_idx)
                
                self.attacks[s_no] = {
                    'start_ts': start_ts,
                    'n': n_val,
                    's_true': s_true
                }
            except Exception as e:
                print(f"Error loading attack {s_no}: {e}")

        print(f"Env Ready. Loaded {len(self.attacks)} attacks.")

    def get_window_obs(self, ts, C_t):
        if ts not in self.df_windowed_scaled.index: return 0.0
        x = self.df_windowed_scaled.loc[ts].to_numpy()
        return np.dot(C_t, x)

    def get_raw_obs(self, ts, C_t):
        idx = self.df_1sec_scaled.index.asof(ts)
        if pd.isna(idx): return 0.0
        x = self.df_1sec_scaled.loc[idx].to_numpy()
        return np.dot(C_t, x)

# --- 2. Pipeline Logic ---

def run_pipeline_ecc(env, attack_id):
    """Pipeline 1: ECC-AHT (Windowed) - Fixed Start Logic"""
    attack = env.attacks[attack_id]
    s_true = attack['s_true']
    start_ts = attack['start_ts']
    n = attack['n']
    
    start_idx = env.df_windowed_scaled.index.searchsorted(start_ts)
    if start_idx > 0 and start_idx < len(env.df_windowed_scaled):
        if env.df_windowed_scaled.index[start_idx] > start_ts:
            start_idx -= 1
            
    if start_idx >= len(env.df_windowed_scaled): return np.nan
    
    # Instantiate the algorithm (while maintaining parameter consistency)
    algo = ECC_AHT(env.K, n, env.Sigma_reg, B_BUDGET, 
                   mu_0=env.mu_0, delta_signal=DELTA_SIGNAL)
    
    curr_idx = start_idx
    steps = 0
    
    while steps < MAX_STEPS_ECC:
        if curr_idx >= len(env.df_windowed_scaled): break
        
        ts = env.df_windowed_scaled.index[curr_idx]
        
        C_t = algo.select_action()
        y_t = env.get_window_obs(ts, C_t)
        algo.update(C_t, y_t)
        
        S_hat = algo.get_S_hat()
        f1 = calculate_f1(S_hat, s_true)
        if f1 >= F1_ALARM_THRESHOLD:
            # Return seconds
            return (steps + 1) * WINDOW_SECONDS
            
        curr_idx += 1
        steps += 1
        
    return MAX_DELAY_SECONDS

def run_pipeline_comb(env, attack_id):
    """Pipeline 2: CombGapE (Raw 1s)"""
    attack = env.attacks[attack_id]
    s_true = attack['s_true']
    start_ts = attack['start_ts']
    n = attack['n']
    
    # Raw data start index
    start_idx = env.df_1sec_scaled.index.searchsorted(start_ts)
    if start_idx >= len(env.df_1sec_scaled): return np.nan
    
    algo = CombGapE(env.K, n)
    
    curr_idx = start_idx
    steps = 0
    
    while steps < MAX_STEPS_COMB:
        if curr_idx >= len(env.df_1sec_scaled): break
        
        ts = env.df_1sec_scaled.index[curr_idx]
        
        C_t = algo.select_action()
        y_t = env.get_raw_obs(ts, C_t)
        algo.update(C_t, y_t)
        
        S_hat = algo.get_S_hat()
        f1 = calculate_f1(S_hat, s_true)
        if f1 >= F1_ALARM_THRESHOLD:
            return float(steps + 1)
            
        curr_idx += 1
        steps += 1
        
    return MAX_DELAY_SECONDS

# --- 3. Main Program ---

def main():
    t_main_start = time.time()
    env = WadiGrandChallengeEnv(MODEL_PATH, ATTACK_CSV_PATH)
    
    target_attack_ids = [aid for aid in env.attacks]
    
    print(f"\n=== Phase 3b: Grand Challenge ===")
    print(f"Comparing Pipelines on {len(target_attack_ids)} scenarios.")
    
    results = {
        "ECC-AHT\n(Windowed)": [],
        "CombGapE\n(Raw 1s)": []
    }
    
    for run in tqdm(range(N_RUNS), desc="Total Runs"):
        seed = SEEDS[run]
        np.random.seed(seed)
        random.seed(seed)
        
        for aid in tqdm(target_attack_ids, desc="Scenarios", leave=False):
            # Pipeline 1
            d_ecc = run_pipeline_ecc(env, aid)
            if not np.isnan(d_ecc):
                results["ECC-AHT\n(Windowed)"].append(d_ecc)
            
            # Pipeline 2
            d_comb = run_pipeline_comb(env, aid)
            if not np.isnan(d_comb):
                results["CombGapE\n(Raw 1s)"].append(d_comb)

    # --- Visualization ---
    print("\n--- Visualization ---")
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
    
    names = list(results.keys())
    data_list = [results[n] for n in names]
    colors = sns.color_palette("colorblind", n_colors=len(results))
    
    for i, data in enumerate(data_list):
        if not data: continue
        mean_val = np.mean(data)
        err_low, err_high = bootstrap_ci(data)
        
        ax.fill_betweenx(
            [mean_val - err_low, mean_val + err_high],
            i - 0.25,
            i + 0.25,
            color=colors[i],
            alpha=0.25,
            linewidth=0
        )
        
        ax.plot(
            i, mean_val,
            marker='o',
            markersize=10,
            markerfacecolor='white',
            markeredgecolor=colors[i],
            markeredgewidth=2.2,
            zorder=5
        )
        
        ci_text = f"{mean_val:.1f}\n[{mean_val-err_low:.1f}, {mean_val+err_high:.1f}]"
        ax.text(i, mean_val + err_high + 1,
                ci_text, 
                ha='center', va='bottom', 
                fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=colors[i], alpha=0.2))
        
        print(f"{names[i]}: Mean={mean_val:.2f}, 95% CI=[{mean_val-err_low:.2f}, {mean_val+err_high:.2f}]")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(rf"Wall-Clock Detection Delay (F1 $\geq$ {F1_ALARM_THRESHOLD}, Seconds)")
    ax.set_xlabel("Algorithms")
    ax.set_title("Pipeline Challenge (WADI)", fontweight='bold', pad=20)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax.set_axisbelow(True)
    
    metadata_str = f"N_runs={N_RUNS}, Seeds={SEEDS[0]}..{SEEDS[-1]}, Budget={B_BUDGET}"
    props = dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='lightgrey')
    ax.text(0.98, 0.02, metadata_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Label the winner
    m1 = np.mean(data_list[0])
    m2 = np.mean(data_list[1])
    if m1 < m2:
        speedup = m2 / m1
        ax.text(0.5, 0.9, f"ECC-AHT is {speedup:.2f}x Faster!", 
                ha='center', transform=ax.transAxes, fontsize=12, fontweight='bold', color=colors[0])

    plt.tight_layout()
    plt.savefig("wadi_b_results.pdf", bbox_inches='tight')
    print("\nThe results image has been saved as phase3b_results.pdf")
    print(f"Total time: {(time.time() - t_main_start)/60:.2f} min")

if __name__ == "__main__":
    main()