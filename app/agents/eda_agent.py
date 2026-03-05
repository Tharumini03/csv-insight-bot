import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_eda(df: pd.DataFrame, target: str, file_id: str):

    output_dir = f"app/storage/outputs/{file_id}"
    os.makedirs(output_dir, exist_ok=True)

    plots = {}

    # -------------------------
    # Plot 1 — Target distribution
    # -------------------------

    plt.figure(figsize=(6,4))

    if pd.api.types.is_numeric_dtype(df[target]):
        sns.histplot(df[target], kde=False)
    else:
        sns.countplot(x=df[target])

    plt.title("Target Distribution")

    path1 = f"{output_dir}/target_distribution.png"
    plt.savefig(path1)
    plt.close()

    plots["target_plot"] = path1


    # -------------------------
    # Plot 2 — Correlation heatmap
    # -------------------------

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] > 1:

        corr = numeric_df.corr()

        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")

        plt.title("Correlation Heatmap")

        path2 = f"{output_dir}/correlation_heatmap.png"
        plt.savefig(path2)
        plt.close()

        plots["heatmap"] = path2

    else:
        plots["heatmap"] = None

    return plots