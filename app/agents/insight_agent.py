def generate_insight(model_info):

    score = round(model_info["score"], 3)

    top_features = [f for f, _ in model_info["feature_importance"][:3]]

    text = f"""
The model achieved an accuracy of {score}.

The most important features influencing the prediction appear to be
{', '.join(top_features)}.

This suggests these variables have the strongest relationship with the target outcome.
"""

    return text