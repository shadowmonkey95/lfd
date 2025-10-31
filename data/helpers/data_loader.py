from typing import Tuple, Optional, Union, List
import numpy as np
import pandas as pd

def load_npy_dataset(
    filepath: str, 
    target_col: Optional[Union[int, str]] = None,
    exclude_columns: Optional[List[Union[int, str]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a .npy dataset and split it into normal and anomaly samples.

    Parameters
    ----------
    filepath : str
        Path to the .npy file.
    target_col : int or str, optional
        Column index or column name of the label.
        If None, assumes the last column is the label.
        If a string is provided, the .npy file must contain a structured array.
    exclude_columns : list of int or str, optional
        List of column indices (int) or names (str) to exclude from features.
        These columns are removed in addition to the target column.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        normal_data, anomaly_data arrays (target and excluded columns are removed).
    """
    data = np.load(filepath, allow_pickle=True)
    # # Drop rows with NaN values
    # data = data[~np.isnan(data).any(axis=1)]
    exclude_columns = exclude_columns or []

    # Handle structured arrays with named columns
    if isinstance(data, np.ndarray) and data.dtype.names:
        # Structured array (columns have names)
        if target_col is None:
            raise ValueError("For structured arrays, please specify the target_col name.")
        
        # Convert all exclude_columns to names (only strings are valid for structured arrays)
        exclude_names = set()
        for col in exclude_columns:
            if isinstance(col, str):
                exclude_names.add(col)
            elif isinstance(col, int):
                if 0 <= col < len(data.dtype.names):
                    exclude_names.add(data.dtype.names[col])
                else:
                    raise ValueError(f"Column index {col} out of range for structured array with {len(data.dtype.names)} columns.")
            else:
                raise TypeError(f"exclude_columns must contain int or str, got {type(col)}")
        
        labels = data[target_col]
        
        # Get feature columns (exclude target and excluded columns)
        feature_cols = [col for col in data.dtype.names 
                       if col != target_col and col not in exclude_names]
        features = np.column_stack([data[col] for col in feature_cols])
    else:
        # Regular numpy array (no named columns)
        target_idx = target_col if isinstance(target_col, int) else -1
        
        # Convert all exclude_columns to indices
        exclude_indices = set()
        for col in exclude_columns:
            if isinstance(col, int):
                # Handle negative indices
                actual_idx = col if col >= 0 else data.shape[1] + col
                if 0 <= actual_idx < data.shape[1]:
                    exclude_indices.add(actual_idx)
                else:
                    raise ValueError(f"Column index {col} out of range for array with {data.shape[1]} columns.")
            elif isinstance(col, str):
                raise TypeError("String column names not supported for regular numpy arrays. Use structured arrays or CSV.")
            else:
                raise TypeError(f"exclude_columns must contain int or str, got {type(col)}")
        
        # Normalize target_idx to positive
        target_idx = target_idx if target_idx >= 0 else data.shape[1] + target_idx
        
        labels = data[:, target_idx]
        
        # Get all columns to keep (exclude target and excluded columns)
        all_indices = set(range(data.shape[1]))
        columns_to_remove = exclude_indices | {target_idx}
        columns_to_keep = sorted(all_indices - columns_to_remove)
        
        features = data[:, columns_to_keep]

    normal_data = features[labels == 0]
    anomaly_data = features[labels == 1]
    return normal_data, anomaly_data


def load_csv_dataset(
    filepath: str, 
    target_col: Optional[Union[int, str]] = None,
    exclude_columns: Optional[List[Union[int, str]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV dataset and split it into normal and anomaly samples.

    Parameters
    ----------
    filepath : str
        Path to the .csv file.
    target_col : str or int, optional
        Name or index of the label column. If None, assumes the last column is the label.
    exclude_columns : list of int or str, optional
        List of column indices (int) or names (str) to exclude from features.
        These columns are removed in addition to the target column.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        normal_data, anomaly_data arrays (target and excluded columns are removed).
    """
    df = pd.read_csv(filepath)
    # Drop rows with NaN values
    df = df.dropna()
    exclude_columns = exclude_columns or []

    # Default: last column as label
    if target_col is None:
        target_col = df.columns[-1]
    elif isinstance(target_col, int):
        # Convert int index to column name
        target_col = df.columns[target_col]

    # Convert all exclude_columns to column names
    exclude_names = set()
    for col in exclude_columns:
        if isinstance(col, str):
            if col in df.columns:
                exclude_names.add(col)
            else:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        elif isinstance(col, int):
            if 0 <= col < len(df.columns):
                exclude_names.add(df.columns[col])
            elif -len(df.columns) <= col < 0:
                exclude_names.add(df.columns[col])  # Handle negative indices
            else:
                raise ValueError(f"Column index {col} out of range for DataFrame with {len(df.columns)} columns.")
        else:
            raise TypeError(f"exclude_columns must contain int or str, got {type(col)}")

    # Columns to drop: target + excluded columns
    columns_to_drop = list(exclude_names | {target_col})

    normal_data = df[df[target_col] == 0].drop(columns=columns_to_drop).to_numpy()
    anomaly_data = df[df[target_col] == 1].drop(columns=columns_to_drop).to_numpy()

    return normal_data, anomaly_data