"""
Malicious URL Detection - Feature Engineering Preprocessing
Based on hamzamanssor's "Detection Malicious URL Using ML Models"
https://www.kaggle.com/code/hamzamanssor/detection-malicious-url-using-ml-models

Dataset: malicious_phish.csv  (sid321axn/malicious-urls-dataset)
Columns: url, type
Types:   benign (normal) | phishing, defacement, malware (anomaly)

Since ML models cannot use raw URL strings, ALL features are lexically
extracted from the URL string itself. This is the core technique of the notebook.

Target: isFraud  (0 = benign, 1 = any malicious type)

Usage:
    python malicious_url_preprocess.py \
        --input   ../../data/raw/6_malicious_phish/malicious_phish.csv \
        --output  ../../data/preprocessed/6_malicious_phish/malicious_url_preprocessed.csv
"""

import argparse
import gc
import math
import pathlib
import re
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# SUSPICIOUS KEYWORDS  (common in phishing/malware URLs)
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "submit", "checkout",
    "paypal", "ebay", "amazon", "apple", "google", "microsoft",
    "free", "lucky", "bonus", "prize", "click", "redirect",
]


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    print("[1/5] Loading data...")
    df = pd.read_csv(input_path)
    print(f"      Shape:       {df.shape}")
    print(f"      Columns:     {list(df.columns)}")
    print(f"      Null values: {df.isnull().sum().sum()}")
    print(f"      Type distribution:\n{df['type'].value_counts()}")
    return df


# ---------------------------------------------------------------------------
# 2. CREATE BINARY TARGET
#    benign = 0  (normal)
#    phishing + defacement + malware = 1  (anomaly)
# ---------------------------------------------------------------------------

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Creating binary target (isFraud)...")
    df["isFraud"] = (df["type"] != "benign").astype(int)
    print(f"      Anomaly rate: {df['isFraud'].mean():.4%}  "
          f"({df['isFraud'].sum()} malicious / {len(df)} total)")
    df = df.drop(columns=["type"])
    return df


# ---------------------------------------------------------------------------
# 3. LEXICAL FEATURE EXTRACTION  (hamzamanssor notebook approach)
#    Raw URL strings → numeric features
#    Grouped into: length, character counts, structural, entropy, keyword
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string — high entropy = more random = suspicious."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def _has_ip(url: str) -> int:
    """Check if domain is an IP address rather than a hostname."""
    try:
        domain = urlparse(url).netloc.split(":")[0]
        parts = domain.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return 1
    except Exception:
        pass
    return 0


def _count_suspicious_keywords(url: str) -> int:
    url_lower = url.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)


def _safe_urlparse(u: str):
    """urlparse wrapper that silently returns an empty result for malformed URLs
    (e.g. invalid IPv6 like [::1] without brackets handled correctly)."""
    try:
        return urlparse(u) if "://" in u else urlparse("http://" + u)
    except ValueError:
        return urlparse("")   # empty fallback — all fields will be empty strings


def extract_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Extracting lexical features from URL strings...")

    urls = df["url"].astype(str)

    # ---- Parse URL components ----
    parsed = urls.apply(_safe_urlparse)

    scheme  = parsed.apply(lambda p: p.scheme)
    domain  = parsed.apply(lambda p: p.netloc)
    path    = parsed.apply(lambda p: p.path)
    query   = parsed.apply(lambda p: p.query)

    # ---- Length features ----
    df["url_length"]    = urls.str.len()
    df["domain_length"] = domain.str.len()
    df["path_length"]   = path.str.len()
    df["query_length"]  = query.str.len()
    df["tld_length"]    = domain.apply(
        lambda d: len(d.split(".")[-1]) if "." in d else 0
    )

    # ---- Character count features ----
    df["num_dots"]       = urls.str.count(r"\.")
    df["num_hyphens"]    = urls.str.count(r"-")
    df["num_underscores"]= urls.str.count(r"_")
    df["num_slashes"]    = urls.str.count(r"/")
    df["num_digits"]     = urls.str.count(r"\d")
    df["num_special"]    = urls.str.count(r"[@!#$%^&*()+=\[\]{};':\"\\|,<>\?~]")
    df["num_equals"]     = urls.str.count(r"=")
    df["num_ampersands"] = urls.str.count(r"&")
    df["num_question"]   = urls.str.count(r"\?")
    df["num_at"]         = urls.str.count(r"@")   # @ in URL = suspicious redirect

    # ---- Ratio features ----
    df["digit_ratio"]   = df["num_digits"]  / df["url_length"].replace(0, 1)
    df["special_ratio"] = df["num_special"] / df["url_length"].replace(0, 1)

    # ---- Structural features ----
    df["num_subdomains"] = domain.apply(
        lambda d: max(len(d.split(".")) - 2, 0) if d else 0
    )
    df["is_https"]       = (scheme == "https").astype(int)
    df["has_ip"]         = urls.apply(_has_ip)
    df["has_port"]       = domain.str.contains(r":\d+").astype(int)

    # ---- Entropy features ----
    df["url_entropy"]    = urls.apply(_shannon_entropy)
    df["domain_entropy"] = domain.apply(_shannon_entropy)

    # ---- Keyword features ----
    df["suspicious_keyword_count"] = urls.apply(_count_suspicious_keywords)

    # ---- Drop raw URL string (not usable by anomaly detectors) ----
    df = df.drop(columns=["url"])

    print(f"      Extracted {len(df.columns) - 1} features.")   # -1 for isFraud
    return df


# ---------------------------------------------------------------------------
# 4. CLEAN
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/5] Cleaning...")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"      Removed {before - len(df)} duplicate rows.")

    gc.collect()
    print(f"      Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 5. NORMALISE  (StandardScaler on all features, isFraud untouched)
# ---------------------------------------------------------------------------

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/5] Standard-scaling features...")

    y = df["isFraud"].copy()
    df_features = df.drop(columns=["isFraud"])

    scaler = StandardScaler()
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    df_features[numeric_cols] = scaler.fit_transform(
        df_features[numeric_cols]
    ).astype(np.float32)

    df_features["isFraud"] = y.values

    print(f"      Final shape:  {df_features.shape}")
    print(f"      Anomaly rate: {y.mean():.4%}")
    return df_features


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = create_target(df)
    df = extract_lexical_features(df)
    df = clean(df)
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
        description="Malicious URL - hamzamanssor lexical feature preprocessing for pyclad"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to malicious_phish.csv",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed/malicious-url/malicious_url_preprocessed.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    preprocess(
        input_path=args.input,
        output_path=args.output,
    )