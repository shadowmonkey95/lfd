"""
Credit Card Fraud Detection - Feature Engineering Preprocessing
Based on janiobachmann's "Credit Fraud || Dealing with Imbalanced Datasets"
https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

Dataset: creditcard.csv
Columns: Time, V1-V28 (already PCA-transformed), Amount, Class
Target:  Class  (1 = fraud, 0 = normal)

Adapted for unsupervised continual anomaly detection (pyclad):
  - No train/test split or SMOTE (pyclad handles concept splitting)
  - Sorts by Time so concept-incremental scenario respects chronological order
  - Keeps Time column for time-based scenario, drops it for others
  - Renames Class -> isFraud for pipeline compatibility

Usage (keep Time for time_based scenario):
    python creditcard_preprocess.py \
        --input   ../../data/raw/credit-card-fraud/creditcard.csv \
        --output  ../../data/preprocessed/credit-card-fraud/creditcard_preprocessed_with_time.csv \
        --keep_time

Usage (drop Time, for non-time-based scenarios):
    python creditcard_preprocess.py \
        --input   ../../data/raw/credit-card-fraud/creditcard.csv \
        --output  ../../data/preprocessed/credit-card-fraud/creditcard_preprocessed.csv
"""

import argparse
import gc
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    print("[1/5] Loading data...")
    df = pd.read_csv(input_path)
    print(f"      Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Null values: {df.isnull().sum().sum()}")
    print(f"      Fraud rate: {df['Class'].mean():.4%}  ({df['Class'].sum()} fraud / {len(df)} total)")
    return df


# ---------------------------------------------------------------------------
# 2. SCALE Amount AND Time  (janiobachmann exact approach)
#    V1-V28 are already PCA-transformed → no scaling needed for them
#    Amount and Time are raw → RobustScaler (robust to outliers)
# ---------------------------------------------------------------------------

def scale_amount_and_time(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Scaling Amount and Time with RobustScaler...")

    rob_scaler = RobustScaler()

    # Scale Amount
    df["scaled_amount"] = rob_scaler.fit_transform(
        df["Amount"].values.reshape(-1, 1)
    ).flatten()

    # Scale Time
    df["scaled_time"] = rob_scaler.fit_transform(
        df["Time"].values.reshape(-1, 1)
    ).flatten()

    # Drop originals (janiobachmann: drop Time and Amount, insert scaled versions)
    # We keep raw Time separately for sorting, then handle it in clean_and_sort
    df = df.drop(columns=["Amount"])
    # Keep raw Time until after sort — will be dropped or kept based on --keep_time flag

    print("      scaled_amount and scaled_time added.")
    return df


# ---------------------------------------------------------------------------
# 3. OUTLIER REMOVAL  (janiobachmann exact logic)
#    Only applied to the 4 most fraud-discriminative V features.
#    Uses IQR * 1.5 cutoff on the FRAUD subset only — we find the fraud
#    distribution boundaries, then remove extreme outliers from the full df.
#    This is janiobachmann's exact method: he inspects boxplots of each V
#    feature split by Class, identifies extreme outlier rows, and removes them.
#    Key features with extreme outliers per his EDA: V10, V12, V14, V3
# ---------------------------------------------------------------------------

# The 4 features janiobachmann explicitly removes outliers from
OUTLIER_FEATURES = ["V10", "V12", "V14", "V3"]


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Removing outliers (IQR method on fraud subset for V10, V12, V14, V3)...")

    fraud_df = df[df["Class"] == 1]
    rows_before = len(df)

    for feature in OUTLIER_FEATURES:
        if feature not in df.columns:
            print(f"      Skipping {feature} (not found)")
            continue

        q25 = fraud_df[feature].quantile(0.25)
        q75 = fraud_df[feature].quantile(0.75)
        iqr  = q75 - q25

        cut_off = iqr * 1.5
        lower   = q25 - cut_off
        upper   = q75 + cut_off

        # Remove rows where the feature falls outside the fraud IQR bounds
        outliers_mask = (df[feature] < lower) | (df[feature] > upper)
        df = df[~outliers_mask]

        removed = outliers_mask.sum()
        print(f"      {feature}: removed {removed} outliers  "
              f"(lower={lower:.3f}, upper={upper:.3f})")

    print(f"      Rows: {rows_before} -> {len(df)}  (removed {rows_before - len(df)} total)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. SORT BY TIME, RENAME TARGET, CLEAN
# ---------------------------------------------------------------------------

def clean_and_sort(df: pd.DataFrame, keep_time: bool) -> pd.DataFrame:
    print("[4/5] Sorting by Time and cleaning...")

    # Sort chronologically using raw Time column (already in the df)
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)

    # Rename target to match pyclad pipeline convention
    df = df.rename(columns={"Class": "isFraud"})

    if keep_time:
        # Keep raw Time for time-based scenario sort_column lookup
        # scaled_time is also present as a feature
        print("      Keeping raw Time column for time-based scenario.")
    else:
        # Drop raw Time — scaled_time is already a feature
        df = df.drop(columns=["Time"], errors="ignore")
        print("      Dropped raw Time column (use scaled_time as feature).")

    # Clean inf/NaN (shouldn't be any, but safety first)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    gc.collect()
    print(f"      Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 5. NORMALISE remaining V features with StandardScaler
#    Amount and Time are already RobustScaled.
#    V1-V28 are PCA-transformed but not unit-scaled — StandardScaler aligns
#    their variance with scaled_amount / scaled_time for anomaly detectors.
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame, keep_time: bool) -> pd.DataFrame:
    print("[5/5] Standard-scaling V features...")

    y = df["isFraud"].copy()

    # Columns to protect from scaling
    protected = ["isFraud"]
    if keep_time:
        protected.append("Time")

    df_features = df.drop(columns=protected)

    scaler = StandardScaler()
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    # Don't re-scale the already-robust-scaled columns
    skip = {"scaled_amount", "scaled_time"}
    cols_to_scale = [c for c in numeric_cols if c not in skip]

    df_features[cols_to_scale] = scaler.fit_transform(
        df_features[cols_to_scale]
    ).astype(np.float32)

    # Reconstruct
    for col in protected:
        df_features[col] = df[col].values

    print(f"      Final shape:   {df_features.shape}")
    print(f"      Fraud rate:    {y.mean():.4%}")
    return df_features


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str, keep_time: bool) -> pd.DataFrame:
    df = load_data(input_path)
    df = scale_amount_and_time(df)
    df = remove_outliers(df)
    df = clean_and_sort(df, keep_time=keep_time)
    df = normalise(df, keep_time=keep_time)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Saved to: {out}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud - janiobachmann preprocessing for pyclad"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed/creditcard_preprocessed_with_time.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--keep_time",
        action="store_true",
        default=False,
        help="Keep raw Time column for time-based scenario (default: drop it)",
    )
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
        keep_time=args.keep_time,
    )