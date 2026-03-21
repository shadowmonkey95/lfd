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
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Feature engineering...")

    for grp in ["card1", "card4"]:
        if grp not in df.columns:
            continue
        df[f"TransactionAmt_to_mean_{grp}"] = (
            df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("mean")
        )
        df[f"TransactionAmt_to_std_{grp}"] = (
            df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("std")
        )

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

    for ecol in ["P_emaildomain", "R_emaildomain"]:
        if ecol not in df.columns:
            continue
        split = df[ecol].str.split(".", expand=True).reindex(columns=[0, 1, 2])
        df[f"{ecol}_1"] = split[0]
        df[f"{ecol}_2"] = split[1]
        df[f"{ecol}_3"] = split[2]

    return df

# ---------------------------------------------------------------------------
# 3. DROP LOW-INFORMATION COLUMNS
# ---------------------------------------------------------------------------

def drop_low_info_cols(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Dropping low-information columns...")

    many_null_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.9]
    big_top_value_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).iloc[0] > 0.9]
    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]

    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
    # Ensure we don't drop target or the time column we want to keep
    cols_to_drop = [c for c in cols_to_drop if c not in ["isFraud", "TransactionDT"]]

    df = df.drop(columns=cols_to_drop)
    print(f"      Dropped {len(cols_to_drop)} columns -> shape {df.shape}")
    return df

# def drop_low_info_cols(df: pd.DataFrame) -> pd.DataFrame:
#     print("[3/6] Dropping low-information columns...")
    
#     # Identify candidates for dropping
#     many_null_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.9]
#     big_top_value_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).iloc[0] > 0.9]
#     one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]

#     cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
    
#     # FORCE PROTECTION of the columns we need for the scenario
#     protected = ["isFraud", "TransactionDT"]
#     cols_to_drop = [c for c in cols_to_drop if c not in protected]

#     df = df.drop(columns=cols_to_drop)
    
#     # Safety Assertion
#     for col in protected:
#         if col not in df.columns:
#             raise ValueError(f"CRITICAL ERROR: {col} was dropped in Step 3!")
            
#     print(f"      Dropped {len(cols_to_drop)} columns -> shape {df.shape}")
#     return df

# ---------------------------------------------------------------------------
# 4. LABEL ENCODE CATEGORICALS
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
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).values)
    return df

# ---------------------------------------------------------------------------
# 5. SORT BY TIME & CLEAN (TransactionDT PRESERVED)
# ---------------------------------------------------------------------------

def clean_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Sorting by TransactionDT and cleaning...")

    if "TransactionDT" in df.columns:
        # Sort and reset index, keeping TransactionDT as a column
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    # Drop TransactionID but KEEP TransactionDT
    df = df.drop(columns=["TransactionID"], errors="ignore")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    gc.collect()
    return df

# ---------------------------------------------------------------------------
# 6. NORMALISE (Excluding TransactionDT and isFraud)
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/6] Standard-scaling features (excluding Time and Target)...")
    
    y = df["isFraud"].copy()
    time_col = df["TransactionDT"].copy()
    
    # Temporarily drop columns we don't want to scale
    df_to_scale = df.drop(columns=["isFraud", "TransactionDT"])

    scaler = StandardScaler()
    numeric_cols = df_to_scale.select_dtypes(include=[np.number]).columns.tolist()
    df_to_scale[numeric_cols] = scaler.fit_transform(df_to_scale[numeric_cols]).astype(np.float32)

    # Reconstruct the dataframe
    df_to_scale["TransactionDT"] = time_col.values
    df_to_scale["isFraud"] = y.values
    
    print(f"      Final shape: {df_to_scale.shape}")
    return df_to_scale

# # --- UPDATE STEP 6 ---
# def normalise(df: pd.DataFrame) -> pd.DataFrame:
#     print("[6/6] Standard-scaling features...")
    
#     # Instead of dropping and re-adding, we just select columns to scale
#     cols_to_scale = [c for c in df.columns if c not in ["isFraud", "TransactionDT"]]
    
#     scaler = StandardScaler()
#     # Apply scaling only to feature columns
#     df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale]).astype(np.float32)
    
#     # Move TransactionDT and isFraud to the end explicitly to ensure order
#     cols_order = [c for c in df.columns if c not in ["TransactionDT", "isFraud"]] + ["TransactionDT", "isFraud"]
#     df = df[cols_order]
    
#     return df

# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(transaction_path: str, identity_path: str | None, output_path: str):
    df = load_and_merge(transaction_path, identity_path)
    df = feature_engineering(df)
    df = drop_low_info_cols(df)
    df = label_encode_categoricals(df)
    df = clean_and_sort(df)
    df = normalise(df)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved preprocessed data to: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE Fraud Preprocessing - Keep Time for Concepts")
    parser.add_argument("--transaction", required=True, help="Path to train_transaction.csv")
    parser.add_argument("--identity", default=None, help="Path to train_identity.csv")
    parser.add_argument("--output", default="data/preprocessed/ead_preprocessed_with_time.csv", help="Output CSV path")
    
    args = parser.parse_args()
    preprocess(args.transaction, args.identity, args.output)