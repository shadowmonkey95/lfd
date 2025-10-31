import pyclad
import numpy as np
import pandas as pd
import pathlib
import os

from data.helpers.data_loader import load_csv_dataset, load_npy_dataset
from src.helpers import clustering

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
dataset_name = 'kaggle_credit_card_transactions_preprocessed'
file_name = "kaggle_credit_card_transactions_preprocessed.csv"
# file_path = os.path.join(BASE_DIR, "../data", file_name)
file_path = BASE_DIR / "data/kaggle_credit_card_transactions_preprocessed.csv"

normal_data, anomaly_data = load_csv_dataset(file_path, target_col="Fraud", exclude_columns=[0, 1, "Time", "Merchant City", "Merchant State", "latitude", "longitude"])

print("Normal Data:", normal_data.shape)
print("Anomaly Data:", anomaly_data.shape)
print("Normal Nan values:", np.isnan(normal_data).sum())
print("Anomaly Nan values:", np.isnan(anomaly_data).sum())