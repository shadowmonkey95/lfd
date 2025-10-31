import pandas as pd
from pathlib import Path
from typing import Union, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# Type alias for scaler options
ScalerType = Literal['standard', 'minmax', 'robust', 'maxabs']


def preprocess_data(
    filepath: str,
    sort_column: str = None,
    target_column: str = None,
    normalize: Union[int, list, None] = None,
    scaler: ScalerType = None,
    base_dir: Path = None,
):
    """
    Preprocess dataset by sorting and normalizing features, then save to disk.
    
    Parameters
    ----------
    filepath : str
        Path to the input CSV file to preprocess.
    sort_column : str, optional
        Column name to sort the data by (e.g., 'Time' for temporal ordering).
        If None, data is not sorted.
    target_column : str, optional
        Target/label column name (e.g., 'Class'). This column will be excluded 
        from normalization. If None, no column is treated as target.
    normalize : int, list, or None, default=None
        Normalization strategy:
        - None or 0: No normalization applied
        - 1: Normalize all numeric columns (except sort_column and target_column)
        - list: Normalize only the specified column names
    scaler : ScalerType, optional
        Scaler type to use for normalization. Options:
        - 'standard': StandardScaler (mean=0, std=1) - default if normalize != None
        - 'minmax': MinMaxScaler (scale to [0,1])
        - 'robust': RobustScaler (uses median/IQR, robust to outliers)
        - 'maxabs': MaxAbsScaler (scale to [-1,1], preserves sparsity)
        If None and normalize is not None, defaults to 'standard'.
    base_dir : Path, optional
        Base directory path for saving preprocessed data. If None, uses current directory.
    
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with sorted and/or normalized features.
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Sort if sort_column is provided
    if sort_column is not None:
        df = df.sort_values(sort_column).reset_index(drop=True)
        print(f"Data sorted by '{sort_column}'")
    
    # Treat None as 0 for normalize
    if normalize is None:
        normalize = 0
    
    # Normalization logic
    if normalize == 0:
        print("No normalization applied")
    
    else:
        # Auto-select StandardScaler if scaler not specified
        if scaler is None:
            scaler = 'standard'
            print(f"Scaler not specified, defaulting to 'standard'")
        
        # Select scaler based on parameter
        scaler_map = {
            'standard': StandardScaler(),      # Mean=0, Std=1 (best for normal distribution)
            'minmax': MinMaxScaler(),          # Scale to [0,1] (best for bounded features)
            'robust': RobustScaler(),          # Uses median/IQR (best for outliers)
            'maxabs': MaxAbsScaler(),          # Scale to [-1,1] (preserves sparsity)
        }
        
        if scaler not in scaler_map:
            raise ValueError(f"scaler must be one of: {list(scaler_map.keys())}")
        
        scaler_obj = scaler_map[scaler]
        
        if normalize == 1:
            # Normalize all numeric columns except sort_column and target_column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            # Build exclusion list
            exclude_cols = []
            if sort_column is not None:
                exclude_cols.append(sort_column)
            if target_column is not None:
                exclude_cols.append(target_column)
            
            cols_to_normalize = [col for col in numeric_cols 
                                if col not in exclude_cols]
            
            df[cols_to_normalize] = scaler_obj.fit_transform(df[cols_to_normalize])
            print(f"Normalized all columns with '{scaler}': {cols_to_normalize}")
        
        elif isinstance(normalize, list):
            # Normalize only specified columns
            df[normalize] = scaler_obj.fit_transform(df[normalize])
            print(f"Normalized columns with '{scaler}': {normalize}")
        
        else:
            raise ValueError("normalize must be None, 0, 1, or a list of column names")
    
    # Save preprocessed data
    if base_dir is None:
        base_dir = Path.cwd()
    
    output_dir = base_dir / 'data' / 'preprocessed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from filepath and create output path
    original_filename = Path(filepath).stem
    output_path = output_dir / f'{original_filename}_preprocessed.csv'
    
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")
    
    return df

if __name__ == '__main__':
    from pathlib import Path
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    df = preprocess_data(
        filepath='data/raw/creditcard_with_time.csv',
        sort_column='Time',
        target_column='Class',
        normalize=['Amount'],
        scaler=None,
        base_dir=None
    )