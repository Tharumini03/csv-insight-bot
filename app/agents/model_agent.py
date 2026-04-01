import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def detect_problem_type(y: pd.Series) -> str:
    """
    Decide classification vs regression.
    - If target is not numeric -> classification
    - If target has small unique values -> classification
    - Else -> regression
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    unique_values = y.nunique(dropna=True)
    if unique_values <= 15:
        return "classification"

    return "regression"


def train_and_evaluate(df: pd.DataFrame, target: str, model_choice: str = "rf") -> dict:    
    """
    Train a baseline model and evaluate.
    Handles categorical features using OneHotEncoder.
    Returns: model info + score.
    """
    X = df.drop(columns=[target])
    y = df[target]

    problem_type = detect_problem_type(y)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    if problem_type == "classification":
        metric_name = "Accuracy"

        if model_choice == "logreg":
            # Logistic Regression needs scaling for numeric features
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ("num", Pipeline(steps=[
                        ("scaler", StandardScaler())
                    ]), numeric_features),
                ]
            )
            model = LogisticRegression(max_iter=2000)
        else:
            # Random Forest does NOT need scaling
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                    ("num", "passthrough", numeric_features),
                ]
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)

    else:
        # Regression case: Logistic Regression is not valid.
        # If user selected logreg, we fallback to RandomForestRegressor.
        metric_name = "R²"

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", "passthrough", numeric_features),
            ]
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    preds = pipeline.predict(X_test)

    # Score
    if problem_type == "classification":
        score = float(accuracy_score(y_test, preds))
    else:
        score = float(r2_score(y_test, preds))


        # Feature importance (only for tree models)
    try:
        importances = pipeline.named_steps["model"].feature_importances_
        feature_names = (
            numeric_features +
            list(pipeline.named_steps["prep"]
                 .named_transformers_["cat"]
                 .get_feature_names_out(categorical_features))
        )

        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]

    except:
        feature_importance = []

    return {
        "problem_type": "Classification" if problem_type == "classification" else "Regression",
        "model_used": model.__class__.__name__,
        "metric": metric_name,
        "score": score,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "num_features": len(numeric_features),
        "cat_features": len(categorical_features),
        "feature_importance": feature_importance
    }