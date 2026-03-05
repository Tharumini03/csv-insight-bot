import pandas as pd

def detect_schema(df: pd.DataFrame) -> dict:
    """
    Understand the dataset structure:
    - rows/cols
    - column types
    - missing values
    - basic numeric stats
    """
    rows, cols = df.shape

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_top = missing_by_col[missing_by_col > 0].head(10).to_dict()

    # Basic stats for numeric columns (safe, short)
    numeric_summary = {}
    for c in numeric_cols:
        numeric_summary[c] = {
            "min": None if df[c].dropna().empty else float(df[c].min()),
            "max": None if df[c].dropna().empty else float(df[c].max()),
            "mean": None if df[c].dropna().empty else float(df[c].mean()),
        }

    return {
        "rows": int(rows),
        "cols": int(cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": non_numeric_cols,
        "missing_top": missing_top,
        "total_missing_cells": int(df.isna().sum().sum()),
        "numeric_summary": numeric_summary
    }