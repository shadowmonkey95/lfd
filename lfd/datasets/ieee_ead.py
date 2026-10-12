"""IEEE-CIS Fraud Detection — feature engineering from the artgor Kaggle kernel "EDA and models"
(https://www.kaggle.com/artgor/eda-and-models; copy in references/notebooks/ieee-fraud-detection/).

Ported from the original lfd project (tag legacy-v1: src/helpers/ead_preprocess_keepTransactionDT.py).
Same steps and order, but in memory (no intermediate CSV) and parameterised so later prep versions can switch
individual steps without copying the whole file.

Steps
  1 load + left-join identity on TransactionID
  2 feature engineering: TransactionAmt / id_02 / D15 ratios to group mean/std; e-mail domain split
  3 drop low-information columns (>90% null, >90% one value, 1 unique)   — isFraud and TransactionDT protected
  4 label-encode categoricals (CAT_COLS)
  5 sort by TransactionDT, drop TransactionID, inf -> NaN, NaN -> column median
  6 StandardScaler on all features (not TransactionDT / isFraud), float32

Caveat (kept for faithfulness, see FIT_SCOPE): steps 2-6 compute statistics (group means, medians, encoders,
scaler) on the WHOLE dataset, i.e. including the future concepts. That is fine for a static Kaggle model but in a
continual stream it leaks future information into early concepts. A causal variant is a candidate for prep_v02.
"""
from __future__ import annotations

import gc
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

TIME_COL = "TransactionDT"
LABEL_COL = "isFraud"
ID_COL = "TransactionID"

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

PROTECTED = [LABEL_COL, TIME_COL]


def load_and_merge(raw_dir: pathlib.Path, use_identity: bool = True) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = raw_dir / "train_identity.csv"
    if use_identity and ident.exists():
        df = df.merge(pd.read_csv(ident), on=ID_COL, how="left")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    for grp in ["card1", "card4"]:
        if grp in df.columns:
            df[f"TransactionAmt_to_mean_{grp}"] = df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("mean")
            df[f"TransactionAmt_to_std_{grp}"] = df["TransactionAmt"] / df.groupby([grp])["TransactionAmt"].transform("std")
    if "id_02" in df.columns:
        for grp in ["card1", "card4"]:
            if grp in df.columns:
                df[f"id_02_to_mean_{grp}"] = df["id_02"] / df.groupby([grp])["id_02"].transform("mean")
                df[f"id_02_to_std_{grp}"] = df["id_02"] / df.groupby([grp])["id_02"].transform("std")
    if "D15" in df.columns:
        for grp in ["card1", "card4", "addr1", "addr2"]:
            if grp in df.columns:
                df[f"D15_to_mean_{grp}"] = df["D15"] / df.groupby([grp])["D15"].transform("mean")
                df[f"D15_to_std_{grp}"] = df["D15"] / df.groupby([grp])["D15"].transform("std")
    for ecol in ["P_emaildomain", "R_emaildomain"]:
        if ecol in df.columns:
            split = df[ecol].str.split(".", expand=True).reindex(columns=[0, 1, 2])
            df[f"{ecol}_1"], df[f"{ecol}_2"], df[f"{ecol}_3"] = split[0], split[1], split[2]
    return df


def drop_low_info_cols(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    many_null = [c for c in df.columns if df[c].isnull().sum() / len(df) > threshold]
    big_top = [c for c in df.columns if df[c].value_counts(dropna=False, normalize=True).iloc[0] > threshold]
    one_val = [c for c in df.columns if df[c].nunique() <= 1]
    drop = [c for c in set(many_null + big_top + one_val) if c not in PROTECTED]
    return df.drop(columns=drop), sorted(drop)


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str).values)
    return df


def clean_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL in df.columns:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
    df = df.drop(columns=[ID_COL], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    gc.collect()
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    y, t = df[LABEL_COL].copy(), df[TIME_COL].copy()
    x = df.drop(columns=[LABEL_COL, TIME_COL])
    num = x.select_dtypes(include=[np.number]).columns.tolist()
    x[num] = StandardScaler().fit_transform(x[num]).astype(np.float32)
    x[TIME_COL] = t.values
    x[LABEL_COL] = y.values
    return x


def preprocess(raw_dir: pathlib.Path, use_identity: bool = True, low_info_threshold: float = 0.9) -> tuple[pd.DataFrame, dict]:
    """Full artgor/EAD pipeline. Returns (df with features + TransactionDT + isFraud, report dict)."""
    df = load_and_merge(raw_dir, use_identity)
    rows_in, cols_in = df.shape
    df = feature_engineering(df)
    df, dropped = drop_low_info_cols(df, low_info_threshold)
    df = label_encode(df)
    df = clean_and_sort(df)
    df = normalise(df)
    report = {"rows_in": rows_in, "cols_in": cols_in, "dropped_low_info_cols": dropped,
              "shape_out": list(df.shape), "fraud_rate": float(df[LABEL_COL].mean()),
              "fit_scope": "global (all rows, incl. future concepts) — legacy behaviour"}
    return df, report
