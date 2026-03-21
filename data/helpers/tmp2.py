import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv("/home/vu/lfd/data/raw/creditcard_with_time.csv")
df = pd.read_csv("/home/vu/lfd/data/preprocesses/ieee-fraud-detection/both_preprocessed_normalized.csv")
print(f"\nOriginal shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check NaN values
print("\n" + "="*60)
print("CHECKING FOR NaN VALUES")
print("="*60)

nan_counts = df.isnull().sum()
total_nans = nan_counts.sum()

print(f"\nTotal NaN values: {total_nans}")

if total_nans > 0:
    print("\nNaN values found in columns:")
    print(nan_counts[nan_counts > 0])
else:
    print("\n✓ No NaN values found - dataset is clean!")

# Check data types
print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes.value_counts())

# Scale Amount only
print("\n" + "="*60)
print("SCALING AMOUNT COLUMN")
print("="*60)

print("\nBefore scaling:")
print(f"  Amount - Mean: {df['Amount'].mean():.2f}, Std: {df['Amount'].std():.2f}")
print(f"  Amount - Min: {df['Amount'].min():.2f}, Max: {df['Amount'].max():.2f}")

# Initialize scaler
scaler = StandardScaler()

# Scale Amount
df['Amount'] = scaler.fit_transform(df[['Amount']])

print("\nAfter scaling:")
print(f"  Amount - Mean: {df['Amount'].mean():.6f}, Std: {df['Amount'].std():.6f}")
print(f"  Amount - Min: {df['Amount'].min():.2f}, Max: {df['Amount'].max():.2f}")

# Time stays as-is
print("\n" + "="*60)
print("TIME COLUMN (Unchanged)")
print("="*60)
print(f"  Time - Mean: {df['Time'].mean():.2f}, Std: {df['Time'].std():.2f}")
print(f"  Time - Min: {df['Time'].min():.2f}, Max: {df['Time'].max():.2f}")
print("  ✓ Time column kept as original values")

# Save
print("\n" + "="*60)
print("SAVING PREPROCESSED DATA")
print("="*60)

output_path = "/home/vu/lfd/data/preprocessed/creditcard_with_time_preprocessed2.csv"
df.to_csv(output_path, index=False)

print(f"✓ Saved to: {output_path}")
print(f"  Final shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Samples: {len(df):,}")
print(f"✓ Features: {df.shape[1] - 1}")
print(f"✓ Fraud cases: {df['Class'].sum():,}")
print(f"✓ Fraud rate: {df['Class'].mean()*100:.3f}%")
print(f"✓ NaN values: {df.isnull().sum().sum()}")
print(f"✓ Amount column: Scaled (mean=0, std=1)")
print(f"✓ Time column: Original values (unchanged)")
print(f"✓ V1-V28: Original values (already preprocessed)")
print(f"✓ Class: Original values (0 or 1)")

print("\n" + "="*60)
print("✓ PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nOutput file: {output_path}")
print("\nYou can now use this file for training!")