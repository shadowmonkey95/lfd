"""
Credit Card Transactions Fraud Detection - Feature Engineering Preprocessing
Based on marcinrutecki's "Best techniques and metrics for Imbalanced Dataset"
https://www.kaggle.com/code/marcinrutecki/best-techniques-and-metrics-for-imbalanced-dataset

Dataset: fraudTrain.csv + fraudTest.csv  (kartik2112/fraud-detection)
Columns: trans_date_trans_time, cc_num, merchant, category, amt, first, last,
         gender, street, city, state, zip, lat, long, city_pop, job, dob,
         trans_num, unix_time, merch_lat, merch_long, is_fraud
Target:  is_fraud  (1 = fraud, 0 = normal)

Adapted for unsupervised continual anomaly detection (pyclad):
  - No train/test split or SMOTE (pyclad handles concept splitting)
  - Merges fraudTrain + fraudTest into one file, sorts by unix_time
  - Keeps unix_time column for time-based scenario
  - is_fraud column name is already pipeline-compatible (no rename needed)

Usage (keep unix_time for time_based scenario):
    python sparkov_preprocess.py \
        --train   ../../data/raw/4_sparkov/fraudTrain.csv \
        --test    ../../data/raw/4_sparkov/fraudTest.csv \
        --output  ../../data/preprocessed/4_sparkov/4sparkov_preprocessed_with_time.csv \
        --keep_time

Usage (drop unix_time, for non-time-based scenarios):
    python sparkov_preprocess.py \
        --train   ../../data/raw/4_sparkov/fraudTrain.csv \
        --test    ../../data/raw/4_sparkov/fraudTest.csv \
        --output  ../../data/preprocessed/4_sparkov/4sparkov_preprocessed_without_time.csv
"""

import argparse
import gc
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. LOAD & MERGE TRAIN + TEST
# ---------------------------------------------------------------------------

def load_data(train_path: str, test_path: str | None) -> pd.DataFrame:
    print("[1/6] Loading data...")
    df = pd.read_csv(train_path, index_col=0)
    print(f"      Train shape: {df.shape}")

    if test_path and pathlib.Path(test_path).exists():
        df_test = pd.read_csv(test_path, index_col=0)
        print(f"      Test  shape: {df_test.shape}")
        df = pd.concat([df, df_test], ignore_index=True)
        print(f"      Combined shape: {df.shape}")
    else:
        print("      No test file — using train only.")

    print(f"      Null values: {df.isnull().sum().sum()}")
    print(f"      Fraud rate:  {df['is_fraud'].mean():.4%}  "
          f"({df['is_fraud'].sum()} fraud / {len(df)} total)")
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (marcinrutecki approach)
#    Key engineered features:
#      - Datetime parts from trans_date_trans_time (hour, day, month, weekday)
#      - Age from dob
#      - Geographic distance between cardholder and merchant
#      - Transaction amount aggregations per category and per customer
# ---------------------------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Feature engineering...")

    # ---- Datetime features from trans_date_trans_time ----
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"]    = df["trans_date_trans_time"].dt.hour.astype(np.int8)
    df["trans_day"]     = df["trans_date_trans_time"].dt.day.astype(np.int8)
    df["trans_month"]   = df["trans_date_trans_time"].dt.month.astype(np.int8)
    df["trans_weekday"] = df["trans_date_trans_time"].dt.dayofweek.astype(np.int8)
    # Drop the raw datetime string (not useful for anomaly detectors)
    df = df.drop(columns=["trans_date_trans_time"])

    # ---- Age from date of birth ----
    df["dob"] = pd.to_datetime(df["dob"])
    reference_date = pd.Timestamp("2020-12-31")   # end of dataset period
    df["age"] = ((reference_date - df["dob"]).dt.days / 365.25).astype(np.float32)
    df = df.drop(columns=["dob"])

    # ---- Geographic distance: cardholder location vs merchant location ----
    # Simple Euclidean distance in lat/lon degrees (sufficient as a relative feature)
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df["distance_km"] = np.sqrt(
            (df["lat"] - df["merch_lat"]) ** 2 +
            (df["long"] - df["merch_long"]) ** 2
        ).astype(np.float32)
        df = df.drop(columns=["lat", "long", "merch_lat", "merch_long"])

    # ---- Transaction amount aggregations per category ----
    df["amt_mean_category"] = df.groupby("category")["amt"].transform("mean")
    df["amt_std_category"]  = df.groupby("category")["amt"].transform("std").fillna(0)
    df["amt_to_mean_category"] = (df["amt"] / df["amt_mean_category"].replace(0, np.nan))

    # ---- Transaction amount aggregations per customer (cc_num) ----
    df["amt_mean_customer"] = df.groupby("cc_num")["amt"].transform("mean")
    df["amt_std_customer"]  = df.groupby("cc_num")["amt"].transform("std").fillna(0)
    df["amt_to_mean_customer"] = (df["amt"] / df["amt_mean_customer"].replace(0, np.nan))

    print("      Feature engineering done.")
    return df


# ---------------------------------------------------------------------------
# 3. DROP IDENTIFIER & HIGH-CARDINALITY COLUMNS
#    Columns that carry no signal for anomaly detection:
#      - cc_num, trans_num: unique identifiers
#      - first, last, street: PII with too many unique values
#      - merchant, job: very high cardinality strings
#        (their signal is already partially captured via category and amt aggs)
# ---------------------------------------------------------------------------

DROP_COLS = ["cc_num", "trans_num", "first", "last", "street", "merchant", "job"]


def drop_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Dropping identifier and high-cardinality columns...")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"      Dropped {len(cols_to_drop)} columns -> shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. ENCODE CATEGORICAL COLUMNS  (marcinrutecki: LabelEncoder)
#    Remaining categoricals after identifier drop:
#      category, gender, city, state, zip
# ---------------------------------------------------------------------------

CAT_COLS = ["category", "gender", "city", "state", "zip"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Label encoding categorical columns...")
    encoded = 0
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).values)
        encoded += 1
    print(f"      Encoded {encoded} columns.")
    return df


# ---------------------------------------------------------------------------
# 5. SORT BY TIME, CLEAN INF/NAN
# ---------------------------------------------------------------------------

def clean_and_sort(df: pd.DataFrame, keep_time: bool) -> pd.DataFrame:
    print("[5/6] Sorting by unix_time and cleaning...")

    # Sort chronologically — critical for concept-incremental scenario
    if "unix_time" in df.columns:
        df = df.sort_values("unix_time").reset_index(drop=True)

    if keep_time:
        print("      Keeping unix_time column for time-based scenario.")
    else:
        df = df.drop(columns=["unix_time"], errors="ignore")
        print("      Dropped unix_time column.")

    # Clean inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    gc.collect()
    print(f"      Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 6. SCALE  (marcinrutecki: RobustScaler for amt, StandardScaler for rest)
#    - amt and distance_km: RobustScaler (skewed, outlier-prone)
#    - all other numeric features: StandardScaler
#    - is_fraud and unix_time (if kept): untouched
# ---------------------------------------------------------------------------

ROBUST_COLS = ["amt", "distance_km", "city_pop",
               "amt_mean_category", "amt_std_category", "amt_to_mean_category",
               "amt_mean_customer", "amt_std_customer", "amt_to_mean_customer"]


def normalise(df: pd.DataFrame, keep_time: bool) -> pd.DataFrame:
    print("[6/6] Scaling features...")

    y = df["is_fraud"].copy()

    protected = ["is_fraud"]
    if keep_time:
        protected.append("unix_time")

    df_features = df.drop(columns=protected)

    # RobustScaler for skewed/outlier-prone columns
    rob_scaler = RobustScaler()
    robust_cols_present = [c for c in ROBUST_COLS if c in df_features.columns]
    if robust_cols_present:
        df_features[robust_cols_present] = rob_scaler.fit_transform(
            df_features[robust_cols_present]
        ).astype(np.float32)

    # StandardScaler for remaining numeric columns
    std_scaler = StandardScaler()
    remaining_numeric = [
        c for c in df_features.select_dtypes(include=[np.number]).columns
        if c not in robust_cols_present
    ]
    if remaining_numeric:
        df_features[remaining_numeric] = std_scaler.fit_transform(
            df_features[remaining_numeric]
        ).astype(np.float32)

    # Reconstruct protected columns
    for col in protected:
        df_features[col] = df[col].values

    print(f"      Final shape:   {df_features.shape}")
    print(f"      Fraud rate:    {y.mean():.4%}")
    return df_features


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(
    train_path: str,
    test_path: str | None,
    output_path: str,
    keep_time: bool,
) -> pd.DataFrame:
    df = load_data(train_path, test_path)
    df = feature_engineering(df)
    df = drop_identifier_cols(df)
    df = encode_categoricals(df)
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
        description="Fraud Transactions - marcinrutecki preprocessing for pyclad"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to fraudTrain.csv",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Path to fraudTest.csv (optional, will be merged with train)",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed/fraud-transactions/fraud_preprocessed_with_time.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--keep_time",
        action="store_true",
        default=False,
        help="Keep unix_time column for time-based scenario (default: drop it)",
    )
    args = parser.parse_args()

    preprocess(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        keep_time=args.keep_time,
    )