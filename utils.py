"""
Utility functions for data processing and model training
"""
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smape_np(y_true, y_pred, eps=1e-8):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)


def log2_price(p: np.ndarray, min_price: float = 1e-6) -> np.ndarray:
    """Convert prices to log2 scale"""
    return np.log2(np.clip(p, min_price, None))


def delog2(x: np.ndarray) -> np.ndarray:
    """Convert log2 values back to original scale"""
    return np.power(2.0, x)


def split_train_val(df: pd.DataFrame, frac_val: float = 0.01, seed: int = 42):
    """Split dataframe into train and validation sets"""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * frac_val)
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_data(train_df: pd.DataFrame, price_col: str = 'price', min_price: float = 1e-6):
    """Clean and prepare training data"""
    df = train_df.copy()
    
    # Clean catalog content
    df['catalog_content'] = df['catalog_content'].fillna("").astype(str).str.strip()
    
    # Clean prices
    df = df.loc[pd.to_numeric(df[price_col], errors="coerce").notnull()].copy()
    df[price_col] = df[price_col].astype(float)
    df = df.loc[df[price_col] >= 0.0].reset_index(drop=True)
    
    return df