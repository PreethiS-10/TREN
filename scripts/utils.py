# utils.py
import pandas as pd

def load_preprocessed_master():
    """Load the preprocessed master dataframe from CSV."""
    return pd.read_csv('data/processed_master_df.csv')

def load_exports_full():
    """Load the full detailed exports dataframe from CSV."""
    return pd.read_csv('data/processed_exports_full.csv')

def load_imports_full():
    """Load the full detailed imports dataframe from CSV."""
    return pd.read_csv('data/processed_imports_full.csv')

def load_scenario_forecasts(filepath="data/2030_predictions_by_scenario.csv"):
    """
    Load the 2030 scenario forecast CSV with caching to optimize performance.
    """
    df = pd.read_csv(filepath)
    return df

