"""
WADI-2017 Cleaning + H0 Modeling Pipeline
Feature Overview:
- Automatically reads raw WADI CSV data
- Automatically identifies sensor/actuator columns
- PV/STATUS separation and filling strategy
- Uses RobustScaler
- Output 1: 'wadi_h0_data_clean_1sec.pkl'
    - Contains an *unscaled*, 1-second resolution DataFrame for model-agnostic algorithms like CombGapE.
- Output 2: 'wadi_h0_model_windowed.pkl'
    - Contains mu_0 and Sigma_reg trained on *windowed* (e.g., 5 min) and *scaled* data.
    - Includes scaler for ECC-AHT during testing.
"""

import os
import sys
import time
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# -----------------------------
# Configuration parameters
# -----------------------------
NORMAL_DATA_PATH = "./WADI.A1_9 Oct 2017/WADI_14days.csv"
WINDOW_SIZE = "1min"
REG_PARAM = 1e-6
SAMPLE_FREQ = "s"
MIN_VALID_COLUMNS = 3
VERBOSE = True

# Output filename
MODEL_WINDOWED_OUTFILE = f"wadi_h0_model_{WINDOW_SIZE}_windowed.pkl"
DATA_1SEC_OUTFILE = "wadi_h0_data_clean_1sec.pkl"

# Identify keywords in the column (used for automatic identification)
SENSOR_KEYWORDS = ["PV", "STATUS", "CMD", "SP", "PV1", "PV2"]


# -----------------------------
# Helper functions
# -----------------------------
def verbose_print(*args, **kwargs):
    """
    Docstring for verbose_print
    
    :param args: Description
    :param kwargs: Description
    """
    if VERBOSE:
        print(*args, **kwargs)

def read_wadi_csv(path: str) -> pd.DataFrame:
    """
    Docstring for read_wadi_csv
    
    :param path: Description
    :type path: str
    :return: Description
    :rtype: DataFrame
    """
    verbose_print(f"Reading CSV: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    # WADI-2017 (14 days) The first 4 lines are metadata; the table header starts from the 5th line.
    skip_candidates = [4, 3, 2, 1, 0]

    last_error = None
    df = None

    for s in skip_candidates:
        try:
            df = pd.read_csv(path, skiprows=s)
            # Check if the correct columns (including Date + Time) were actually read.
            cols_lower = [c.lower() for c in df.columns]
            if "date" in cols_lower and "time" in cols_lower:
                verbose_print(f"Successfully read CSV with skiprows={s}.")
                break
        except Exception as e:
            last_error = e

    if df is None:
        raise RuntimeError(f"Failed to read CSV header correctly. Last error: {last_error}")

    # Subsequent steps remain unchanged: Identify Date + Time
    date_col = None
    time_col = None
    for c in df.columns:
        if c.lower() == "date":
            date_col = c
        if c.lower() == "time":
            time_col = c

    if date_col and time_col:
        verbose_print("Merging 'Date' and 'Time' into Timestamp index.")
        df["Timestamp"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                                         errors="coerce", dayfirst=False)
        df = df.set_index("Timestamp")
        df = df.drop(columns=[date_col, time_col], errors="ignore")
    else:
        raise ValueError("CSV does not contain proper Date/Time columns.")

    df = df.sort_index()
    verbose_print(f"Read dataframe with index from {df.index.min()} to {df.index.max()}. Shape: {df.shape}")
    return df

def auto_select_sensor_columns(df: pd.DataFrame) -> List[str]:
    """
    Docstring for auto_select_sensor_columns
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: List[str]
    """
    verbose_print("Auto-detecting sensor/actuator columns using keywords:", SENSOR_KEYWORDS)
    cols = [c for c in df.columns if any(k in c for k in SENSOR_KEYWORDS)]
    verbose_print(f"Found {len(cols)} candidate columns by keyword.")
    valid_cols = []
    for c in cols:
        series = df[c]
        if series.isna().all():
            continue
        if series.dropna().nunique() <= 1:
            continue
        valid_cols.append(c)
    verbose_print(f"After removing all-NaN/constant columns: {len(valid_cols)} columns.")
    if len(valid_cols) < MIN_VALID_COLUMNS:
        verbose_print("Too few auto-detected columns; falling back to any non-empty column set.")
        valid_cols = [c for c in df.columns if not df[c].isna().all()]
    return valid_cols

def is_status_column(name: str, series: pd.Series) -> bool:
    """
    Docstring for is_status_column
    
    :param name: Description
    :type name: str
    :param series: Description
    :type series: pd.Series
    :return: Description
    :rtype: bool
    """
    name_up = name.upper()
    if "STATUS" in name_up or "CMD" in name_up or name_up.endswith("_STATE") or name_up.endswith("_STS"):
        return True
    nunique = series.dropna().nunique()
    total = len(series)
    if total == 0:
        return False
    if nunique <= 10:
        non_numeric_frac = series.dropna().apply(lambda v: not _looks_like_number(v)).mean()
        if non_numeric_frac > 0.3:
            return True
    return False

def _looks_like_number(v) -> bool:
    """
    Docstring for _looks_like_number
    
    :param v: Description
    :return: Description
    :rtype: bool
    """
    try:
        float(v)
        return True
    except Exception:
        return False

def coerce_and_fill(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """
    Docstring for coerce_and_fill
    
    :param df: Description
    :type df: pd.DataFrame
    :param sensor_cols: Description
    :type sensor_cols: List[str]
    :return: Description
    :rtype: DataFrame
    """
    df_sel = df[sensor_cols].copy()
    verbose_print(f"Coercing columns to numeric where possible and applying PV/STATUS strategies...")
    status_cols = []
    pv_cols = []
    for c in df_sel.columns:
        if is_status_column(c, df_sel[c]):
            status_cols.append(c)
        else:
            pv_cols.append(c)
    verbose_print(f"Detected {len(pv_cols)} PV-like cols and {len(status_cols)} STATUS-like cols.")

    for c in pv_cols:
        df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
    for c in status_cols:
        s = df_sel[c]
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().sum() >= (len(s) * 0.5):
            df_sel[c] = s_num
        else:
            nonnull = s.dropna()
            if nonnull.empty:
                df_sel[c] = s
            else:
                cats, uniques = pd.factorize(nonnull.astype(str), sort=False)
                mapping = {v: i for i, v in enumerate(uniques)}
                df_sel[c] = s.astype(str).map(mapping)
                df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")

    start_ts = df_sel.index.min()
    end_ts = df_sel.index.max()
    full_index = pd.date_range(start=start_ts, end=end_ts, freq=SAMPLE_FREQ)
    if len(full_index) != len(df_sel.index):
        verbose_print(f"Index has gaps or irregular sampling. Reindexing to continuous 1s from {start_ts} to {end_ts}.")
        df_sel = df_sel.reindex(full_index)

    if pv_cols:
        verbose_print("Interpolating PV-like columns using time interpolation...")
        df_sel[pv_cols] = df_sel[pv_cols].interpolate(method="time", limit_direction="both")
    if status_cols:
        verbose_print("Filling STATUS-like columns with forward/backward fill...")
        df_sel[status_cols] = df_sel[status_cols].ffill().bfill()

    df_sel = df_sel.apply(pd.to_numeric, errors="coerce")
    
    # Final cleanup: Drop columns that are still entirely NaN, and fill the remaining scattered NaNs with 0.
    all_nan_cols = [c for c in df_sel.columns if df_sel[c].isna().all()]
    if all_nan_cols:
        verbose_print(f"Dropping {len(all_nan_cols)} columns still all-NaN after filling.")
        df_sel = df_sel.drop(columns=all_nan_cols)
        
    # Fill any remaining NaN values ​​with 0 (for example, where bfill/ffill failed at the beginning/end)
    df_sel = df_sel.fillna(0)

    verbose_print(f"After cleaning: shape = {df_sel.shape}")
    return df_sel

def robust_h0_estimate(X: np.ndarray, reg_param: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Docstring for robust_h0_estimate
    
    :param X: Description
    :type X: np.ndarray
    :param reg_param: Description
    :type reg_param: float
    :return: Description
    :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    """
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])
    M = Sigma.shape[0]
    Sigma_reg = Sigma + reg_param * np.eye(M)
    return mu, Sigma_reg

# -----------------------------
# Main process
# -----------------------------
def build_h0_model(csv_path: str,
                   model_outfile: str,
                   data_outfile: str,
                   reg_param: float = 1e-6,
                   window_size: str = '5T') -> None:
    t0 = time.time()
    df = read_wadi_csv(csv_path)

    # 1. Automatically identify the columns to be used.
    sensor_cols = auto_select_sensor_columns(df)
    if len(sensor_cols) < MIN_VALID_COLUMNS:
        raise RuntimeError("Too few sensor columns detected.")
    verbose_print(f"Using {len(sensor_cols)} columns for modeling: {sensor_cols} ...")

    # 2. Cleaning & Filling (to obtain clean 1-second data)
    df_clean_1sec = coerce_and_fill(df, sensor_cols)
    final_cols = list(df_clean_1sec.columns)
    M = len(final_cols)
    verbose_print(f"Final M = {M} features used.")

    # 3. [Output 1] Saves clean 1-second data (unscaled) for use by CombGapE.
    verbose_print(f"Saving clean 1-sec (unscaled) data to: {data_outfile}")
    payload_1sec = {
        'df_clean_1sec': df_clean_1sec,
        'sensor_columns': final_cols,
        'sample_freq': SAMPLE_FREQ
    }
    with open(data_outfile, "wb") as f:
        pickle.dump(payload_1sec, f)

    # 4. Fit Scaler on 1-second data
    verbose_print("Fitting RobustScaler on 1-sec data...")
    X_raw_1sec = df_clean_1sec.to_numpy(dtype=float)
    scaler = RobustScaler()
    scaler.fit(X_raw_1sec) # Only fit the model, do not convert immediately.

    # 5. [Windowing] Apply windowing to the 1-second data.
    # Note: We are windowing (mean) on the *unscaled* data.
    # Then, the *window mean* is scaled.
    verbose_print(f"Applying '{window_size}' windowing (mean)...")
    df_windows_raw = df_clean_1sec.resample(window_size).mean()
    df_windows_raw = df_windows_raw.dropna(how='all')
    
    X_windows_raw = df_windows_raw.to_numpy()
    
    # 6. Scale the windowed data.
    verbose_print("Scaling the windowed data...")
    X_windows_scaled = scaler.transform(X_windows_raw)
    
    # 7. [Output 2] Estimate the H0 model on the scaled window data.
    verbose_print("Estimating mu_0 and Sigma on scaled, windowed data...")
    mu_0, Sigma_reg = robust_h0_estimate(X_windows_scaled, reg_param=reg_param)

    # 8. Save the model (including scaler, mu_0, Sigma_reg)
    model_payload = {
        "mu_0": mu_0,
        "Sigma_reg": Sigma_reg,
        "scaler": scaler, # Save the *fitted* scaler.
        "sensor_columns": final_cols,
        "window_size": window_size,
        "sample_freq": SAMPLE_FREQ,
        "reg_param": reg_param,
        "created_at": pd.Timestamp.now().isoformat()
    }
    with open(model_outfile, "wb") as f:
        pickle.dump(model_payload, f)

    verbose_print("\n--- H0 model build complete ---")
    verbose_print(f"Saved 1-sec data to: {data_outfile}")
    verbose_print(f"Saved windowed model to: {model_outfile}")
    verbose_print(f"M = {M}, N_1sec_samples = {X_raw_1sec.shape[0]}")
    verbose_print(f"N_windows = {X_windows_scaled.shape[0]}")
    verbose_print(f"mu_0 shape: {mu_0.shape}, Sigma_reg shape: {Sigma_reg.shape}")
    verbose_print(f"Elapsed: {time.time() - t0:.2f} s")


# -----------------------------
# CLI support
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        NORMAL_DATA_PATH = sys.argv[1]
    verbose_print("WADI H0 builder starting...")
    build_h0_model(
        NORMAL_DATA_PATH, 
        MODEL_WINDOWED_OUTFILE, 
        DATA_1SEC_OUTFILE, 
        REG_PARAM,
        WINDOW_SIZE
    )