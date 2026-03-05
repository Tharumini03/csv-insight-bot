import os
import matplotlib.pyplot as plt

def plot_feature_importance(feature_importance, file_id: str):
    """
    feature_importance: list of (feature_name, importance_value)
    Saves a bar chart image and returns the file path.
    """
    out_dir = f"app/storage/outputs/{file_id}"
    os.makedirs(out_dir, exist_ok=True)

    if not feature_importance:
        return None

    # Take top 10
    top = feature_importance[:10]
    names = [x[0] for x in top][::-1]     # reverse for nicer chart
    values = [x[1] for x in top][::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(names, values)
    plt.title("Top Feature Importances")
    plt.tight_layout()

    path = f"{out_dir}/feature_importance.png"
    plt.savefig(path)
    plt.close()

    return path