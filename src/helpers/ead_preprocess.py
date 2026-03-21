"""
https://www.kaggle.com/artgor/eda-and-models

Adapted for unsupervised continual anomaly detection:
  - No train/test split (pyclad handles concept splitting)
  - No target encoding (uses label encoding only - no leakage)
  - Sorts by TransactionDT so concept-incremental scenario respects time order
  - Keeps isFraud as the final column

Usage:        
    python src/helpers/ead_preprocess.py \
        --transaction data/raw/ieee-fraud-detection/train_transaction.csv \
        --identity    data/raw/ieee-fraud-detection/train_identity.csv \
        --output      data/preprocessed/ieee-fraud-detection/ead_preprocessed.csv

Output CSV is compatible with load_csv_dataset(file_path, target_col='isFraud').
"""

import argparse
import gc
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. LOAD & MERGE
# ---------------------------------------------------------------------------

def load_and_merge(transaction_path: str, identity_path: str | None) -> pd.DataFrame:
    print("[1/6] Loading data...")
    df = pd.read_csv(transaction_path)
    print(f"      Transaction shape: {df.shape}")

    if identity_path and pathlib.Path(identity_path).exists():
        identity = pd.read_csv(identity_path)
        df = df.merge(identity, on="TransactionID", how="left")
        print(f"      After identity merge: {df.shape}")
    else:
        print("      No identity file - skipping merge.")

    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (artgor exact logic)
# ---------------------------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Feature engineering...")

    # ---- TransactionAmt aggregations (card1, card4) ----
    for grp in ["card1", "card4"]:
        if grp not in df.columns:
            continue
        df[f"TransactionAmt_to_mean_{grp}"] = (
            df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("mean")
        )
        df[f"TransactionAmt_to_std_{grp}"] = (
            df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("std")
        )

    # ---- id_02 aggregations (card1, card4) ----
    if "id_02" in df.columns:
        for grp in ["card1", "card4"]:
            if grp not in df.columns:
                continue
            df[f"id_02_to_mean_{grp}"] = (
                df["id_02"] / df.groupby([grp])["id_02"].transform("mean")
            )
            df[f"id_02_to_std_{grp}"] = (
                df["id_02"] / df.groupby([grp])["id_02"].transform("std")
            )

    # ---- D15 aggregations (card1, card4, addr1, addr2) ----
    if "D15" in df.columns:
        for grp in ["card1", "card4", "addr1", "addr2"]:
            if grp not in df.columns:
                continue
            df[f"D15_to_mean_{grp}"] = (
                df["D15"] / df.groupby([grp])["D15"].transform("mean")
            )
            df[f"D15_to_std_{grp}"] = (
                df["D15"] / df.groupby([grp])["D15"].transform("std")
            )

    # ---- Email domain splits by '.' into 3 parts (artgor exact) ----
    for ecol in ["P_emaildomain", "R_emaildomain"]:
        if ecol not in df.columns:
            continue
        split = df[ecol].str.split(".", expand=True).reindex(columns=[0, 1, 2])
        df[f"{ecol}_1"] = split[0]
        df[f"{ecol}_2"] = split[1]
        df[f"{ecol}_3"] = split[2]

    print("      Feature engineering done.")
    return df


# ---------------------------------------------------------------------------
# 3. DROP LOW-INFORMATION COLUMNS  (artgor exact logic)
#    Drops columns that are:
#      - >90% null
#      - >90% a single value (including NaN counted)
#      - Only 1 unique value
# ---------------------------------------------------------------------------

def drop_low_info_cols(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Dropping low-information columns...")

    many_null_cols = [
        col for col in df.columns
        if df[col].isnull().sum() / len(df) > 0.9
    ]

    big_top_value_cols = [
        col for col in df.columns
        if df[col].value_counts(dropna=False, normalize=True).iloc[0] > 0.9
    ]

    one_value_cols = [
        col for col in df.columns
        if df[col].nunique() <= 1
    ]

    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
    cols_to_drop = [c for c in cols_to_drop if c != "isFraud"]

    df = df.drop(columns=cols_to_drop)
    print(f"      Dropped {len(cols_to_drop)} columns -> shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. LABEL ENCODE CATEGORICALS  (artgor exact cat_cols list)
# ---------------------------------------------------------------------------

CAT_COLS = [
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
    "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
    "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo", "ProductCD",
    "card4", "card6", "M4",
    "P_emaildomain", "R_emaildomain",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9",
    "P_emaildomain_1", "P_emaildomain_2", "P_emaildomain_3",
    "R_emaildomain_1", "R_emaildomain_2", "R_emaildomain_3",
]


def label_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Label encoding categoricals...")
    encoded = 0
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        # artgor fits on full data (no train/test split here)
        df[col] = le.fit_transform(df[col].astype(str).values)
        encoded += 1
    print(f"      Encoded {encoded} columns.")
    return df


# ---------------------------------------------------------------------------
# 5. SORT BY TIME, DROP IDS, CLEAN INF/NAN  (artgor: clean_inf_nan)
# ---------------------------------------------------------------------------

def clean_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Sorting by TransactionDT, cleaning inf/NaN...")

    # Sort by transaction time — CRITICAL for concept-incremental scenario
    # so that concept splits respect the natural time ordering of fraud patterns
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Drop identifier / time columns (artgor drops these before modelling)
    df = df.drop(columns=["TransactionID", "TransactionDT"], errors="ignore")

    # Replace inf with NaN (artgor exact: replace([np.inf, -np.inf], np.nan))
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN with column median
    df = df.fillna(df.median(numeric_only=True))

    gc.collect()
    print(f"      Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 6. NORMALISE  (standard scale all numeric, isFraud stays 0/1)
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/6] Standard-scaling numeric features...")
    y = df["isFraud"].copy()
    df = df.drop(columns=["isFraud"])

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols]).astype(np.float32)

    df["isFraud"] = y.values
    print(f"      Final shape:   {df.shape}")
    print(f"      Fraud rate:    {y.mean():.4%}")
    return df


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(
    transaction_path: str,
    identity_path: str | None,
    output_path: str,
) -> pd.DataFrame:
    df = load_and_merge(transaction_path, identity_path)
    df = feature_engineering(df)
    df = drop_low_info_cols(df)
    df = label_encode_categoricals(df)
    df = clean_and_sort(df)
    df = normalise(df)

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
        description="IEEE Fraud - artgor feature engineering for pyclad"
    )
    parser.add_argument(
        "--transaction",
        required=True,
        help="Path to train_transaction.csv",
    )
    parser.add_argument(
        "--identity",
        default=None,
        help="Path to train_identity.csv (optional but recommended)",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed/both_preprocessed_normalized.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    preprocess(
        transaction_path=args.transaction,
        identity_path=args.identity,
        output_path=args.output,
    )