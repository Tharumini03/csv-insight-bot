import pandas as pd

def basic_clean(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """
    Basic cleaning:
    - remove fully empty columns
    - fill missing values:
        numeric -> median
        categorical -> mode (most frequent)
    - drop rows where target is missing
    Returns cleaned_df and cleaning_report
    """
    report = {}
    df = df.copy()

    before_shape = df.shape

    # Drop columns that are completely empty
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
    report["dropped_empty_columns"] = empty_cols

    # Drop rows with missing target (can't train without target)
    target_missing_before = int(df[target].isna().sum()) if target in df.columns else 0
    if target in df.columns:
        df = df.dropna(subset=[target])
    report["dropped_rows_missing_target"] = target_missing_before

    # Fill missing values in other columns
    filled = {}
    for col in df.columns:
        if col == target:
            continue

        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            filled[col] = {"missing_filled": missing_count, "method": "median"}
        else:
            mode_series = df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            filled[col] = {"missing_filled": missing_count, "method": "mode/Unknown"}

    report["filled_columns"] = filled
    report["before_shape"] = {"rows": int(before_shape[0]), "cols": int(before_shape[1])}
    report["after_shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report["total_missing_after"] = int(df.isna().sum().sum())

    return df, report