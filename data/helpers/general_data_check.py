import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


file_name = "both_preprocessed_normalized.csv"
# target_column = "Class"
# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)  # Auto-detect terminal width
# pd.set_option('display.max_colwidth', None)  # Show full column content

# ====== Load the dataset ======
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
file_path = BASE_DIR / "data/preprocessed/ieee-fraud-detection" / file_name
df = pd.read_csv(file_path)


# ====== Basic Info ======
print("Basic Information:")
print(df.info())
print("\n")

# # ====== Target class distribution ======
# print("Target class distribution")
# print(df[target_column].value_counts())
# print(df[target_column].value_counts(normalize=True))
# print("\n")

# ====== Preview Data ======
print("First 20 rows:")
print(df.head(20))
df.head(20).to_csv('tmp_head.csv', index=False)
print("Saved to tmp_head.csv")
print("\n")

print("Last 20 rows:")
print(df.tail(20))
df.tail(20).to_csv('tmp_tail.csv', index=False)
print("Saved to tmp_tail.csv")
print("\n")

# ====== Summary Statistics ======
print("Summary Statistics (numerical):")
numerical_cols = df.select_dtypes(include=['number']).columns
if len(numerical_cols) > 0:
    print(df.describe())
else:
    print("No numerical columns found.")
print("\n")

print("Summary Statistics (categorical):")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(df.describe(include=['object', 'category']))
else:
    print("No categorical columns found.")
print("\n")

# Show data types
print("Column Data Types:")
print(df.dtypes)
print("\n")

# ====== Check Missing Values ======
print("Missing Values per Column:")
print(df.isna().sum())
print("\n")

# ====== Check Duplicates ======
print("Number of Duplicate Rows:", df.duplicated().sum())
print("\n")

# ====== Column Overview ======
print("Columns:")
print(df.columns.tolist())
print("\n")

print("Dataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n")

# ====== Unique Value Counts (for categorical columns) ======
print("Unique Values per Column:")
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
print("\n")

# ====== Correlation Matrix (numerical only) ======
print("Correlation Matrix:")
print(df.corr(numeric_only=True))
print("\n")

# ====== Quick Value Distributions ======
for col in df.select_dtypes(include=['object', 'category']).columns:
    print(f"Value counts for '{col}':")
    print(df[col].value_counts().head(10))
    print("\n")

print("Done examining dataset.")