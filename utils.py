import pandas as pd
import numpy as np

def calculate_heuristic(fraud_row, nonfraud_rows):
    """
    Simple heuristic: feature with max absolute z-score
    fraud_row: pd.Series of one fraud row
    nonfraud_rows: pd.DataFrame of N non-fraud rows (e.g., 5 rows)
    Returns the column name of the most responsible feature
    """
    mean_vals = nonfraud_rows.mean()
    std_vals = nonfraud_rows.std(ddof=0)
    # Avoid divide by zero
    std_vals = std_vals.replace(0, 1e-6)
    
    z_scores = abs(fraud_row - mean_vals) / std_vals
    most_responsible = z_scores.idxmax()
    return most_responsible
