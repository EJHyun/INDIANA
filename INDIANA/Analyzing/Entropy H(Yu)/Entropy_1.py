import os
import glob
import math
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd



def detect_separator_and_header(file_path: str, possible_seps=(",", "\t", ";", "|")) -> Tuple[str, bool]:
    """
    Try to detect separator and whether the file has a header row.
    Returns (sep, has_header).
    """
    for sep in possible_seps:
        try:
            # Try with header
            df_head = pd.read_csv(file_path, sep=sep, nrows=5, engine="python")
            if df_head.shape[1] >= 4:
                # Heuristic: if all column names are generic integers, assume no header
                has_header = not all(isinstance(c, int) for c in df_head.columns)
                return sep, has_header
        except Exception:
            continue
    # Fallback
    return ",", True


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to a standard set:
    user_id, poi_id, category_id, timestamp, latitude, longitude
    
    Handles both 6-column and 8-column formats:
    - 6 columns: user_id, poi_id, category_id, timestamp, latitude, longitude
    - 8 columns: user_id, poi_id, category_id, category_name, latitude, longitude, timezone, timestamp
    """
    col_map: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}

    def pick(name_candidates: List[str]) -> str:
        for cand in name_candidates:
            if cand in lower_cols:
                return lower_cols[cand]
        return None

    user_col = pick(["user", "user_id", "uid", "userid"])
    poi_col = pick(["poi", "poi_id", "venueid", "locationid", "place_id"])
    cat_col = pick(["category", "category_id", "cat", "cid"])
    time_col = pick(["timestamp", "time", "datetime", "checkin_time"])
    lat_col = pick(["lat", "latitude"])
    lon_col = pick(["lon", "lng", "longitude", "long"])

    # Handle 8-column format (TSMC dataset): user, poi, category_id, category_name, lat, lon, timezone, timestamp
    if len(df.columns) == 8:
        cols = list(df.columns)
        user_col = user_col or cols[0]
        poi_col = poi_col or cols[1]
        cat_col = cat_col or cols[2]
        # Skip cols[3] (category_name)
        lat_col = lat_col or cols[4]
        lon_col = lon_col or cols[5]
        # Skip cols[6] (timezone)
        time_col = time_col or cols[7]
    # Handle 6-column format
    elif len(df.columns) == 6:
        cols = list(df.columns)
        user_col = user_col or cols[0]
        poi_col = poi_col or cols[1]
        cat_col = cat_col or cols[2]
        time_col = time_col or cols[3]
        lat_col = lat_col or cols[4]
        lon_col = lon_col or cols[5]

    required = {
        "user_id": user_col,
        "poi_id": poi_col,
        "category_id": cat_col,
        "timestamp": time_col,
        "latitude": lat_col,
        "longitude": lon_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Could not identify required columns: {missing}. "
                         f"Detected columns: {list(df.columns)}")

    col_map[required["user_id"]] = "user_id"
    col_map[required["poi_id"]] = "poi_id"
    col_map[required["category_id"]] = "category_id"
    col_map[required["timestamp"]] = "timestamp"
    col_map[required["latitude"]] = "latitude"
    col_map[required["longitude"]] = "longitude"

    df = df.rename(columns=col_map)
    return df[["user_id", "poi_id", "category_id", "timestamp", "latitude", "longitude"]]


def load_one_file(file_path: str) -> pd.DataFrame:
    sep, has_header = detect_separator_and_header(file_path)
    df = pd.read_csv(
        file_path,
        sep=sep,
        header=0 if has_header else None,
        engine="python"
    )
    if not has_header:
        # Temporarily name generic columns
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df = normalize_columns(df)
    return df


def load_all_data(folder: str) -> pd.DataFrame:
    patterns = ["*.txt", "*.csv"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat)))
    
    # Exclude Python script files
    files = [f for f in files if not f.endswith('.py')]
    
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in folder: {folder}")

    dfs = []
    for f in files:
        print(f"Loading {f} ...")
        try:
            df = load_one_file(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {f} due to error: {e}")
    if not dfs:
        raise RuntimeError("No valid files could be loaded.")
    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


# -----------------------
# Entropy Utilities
# -----------------------

def entropy_from_counts(counts: np.ndarray) -> float:
    """
    Shannon entropy (bits) from counts.
    """
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


# -----------------------
# Main Analysis
# -----------------------

def compute_entropies(df: pd.DataFrame, min_checkins: int = 10):
    # Parse timestamp and extract hour-of-day
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Filter users with enough check-ins
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_checkins].index
    df = df[df["user_id"].isin(valid_users)].copy()
    if df.empty:
        raise RuntimeError("No users with sufficient check-ins after filtering.")

    print(f"Total check-ins after filtering: {len(df)}")
    print(f"Number of users with >= {min_checkins} check-ins: {df['user_id'].nunique()}")

    # Baseline: H(Y | u)
    baseline_entropies = []
    for u, g in df.groupby("user_id"):
        poi_counts = g.groupby("poi_id").size().values
        h = entropy_from_counts(poi_counts)
        baseline_entropies.append(h)
    baseline_entropy = float(np.mean(baseline_entropies))

    # Category context: H(Y | u, C)
    cat_entropies = []
    for u, g in df.groupby("user_id"):
        total_u = len(g)
        h_u_c = 0.0
        for (c, gc) in g.groupby("category_id"):
            p_c_given_u = len(gc) / total_u
            poi_counts_c = gc.groupby("poi_id").size().values
            h_y_u_c = entropy_from_counts(poi_counts_c)
            h_u_c += p_c_given_u * h_y_u_c
        cat_entropies.append(h_u_c)
    cat_context_entropy = float(np.mean(cat_entropies))

    # Joint context (Category + Time): H(Y | u, C, T)
    joint_entropies = []
    for u, g in df.groupby("user_id"):
        total_u = len(g)
        h_u_ct = 0.0
        for (c, t), gct in g.groupby(["category_id", "hour"]):
            p_ct_given_u = len(gct) / total_u
            poi_counts_ct = gct.groupby("poi_id").size().values
            h_y_u_ct = entropy_from_counts(poi_counts_ct)
            h_u_ct += p_ct_given_u * h_y_u_ct
        joint_entropies.append(h_u_ct)
    joint_context_entropy = float(np.mean(joint_entropies))

    return baseline_entropy, cat_context_entropy, joint_context_entropy


def print_results(baseline: float, cat_ctx: float, joint_ctx: float):
    def search_space(h: float) -> float:
        # Equivalent search space size: 2^H
        return float(2 ** h)

    baseline_ss = search_space(baseline)
    cat_ss = search_space(cat_ctx)
    joint_ss = search_space(joint_ctx)

    red_cat = 100.0 * (baseline - cat_ctx) / baseline if baseline > 0 else 0.0
    red_joint = 100.0 * (baseline - joint_ctx) / baseline if baseline > 0 else 0.0

    print("\n================ Entropy Reduction Analysis ================")
    print(f"Baseline Entropy H(Y|u): {baseline:.6f} bits")
    print(f"Equivalent search space: {baseline_ss:.4e}")
    print("-----------------------------------------------------------")
    print(f"Category Context Entropy H(Y|u, C): {cat_ctx:.6f} bits")
    print(f"Equivalent search space: {cat_ss:.4e}")
    print(f"Reduction vs Baseline: {red_cat:.2f} %")
    print("-----------------------------------------------------------")
    print(f"Joint Context Entropy H(Y|u, C, T): {joint_ctx:.6f} bits")
    print(f"Equivalent search space: {joint_ss:.4e}")
    print(f"Reduction vs Baseline: {red_joint:.2f} %")
    print("===========================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Entropy Reduction Analysis for LBSN check-in data."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=".",
        help="Folder containing .txt/.csv dataset files."
    )
    parser.add_argument(
        "--min_checkins",
        type=int,
        default=10,
        help="Minimum number of check-ins per user to be included."
    )
    args = parser.parse_args()

    folder = args.data_folder
    min_checkins = args.min_checkins

    print(f"Data folder: {folder}")
    print(f"Minimum check-ins per user: {min_checkins}")

    df = load_all_data(folder)
    print(f"Total raw check-ins loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    baseline, cat_ctx, joint_ctx = compute_entropies(df, min_checkins=min_checkins)
    print_results(baseline, cat_ctx, joint_ctx)


if __name__ == "__main__":
    main()