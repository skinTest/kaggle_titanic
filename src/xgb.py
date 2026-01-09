# colsample_bytree to decrease
# learning_rate to decrese
# n_estimator: 500 mid
# "model__max_depth": 2
# subsample: 1.0 why the upper bound
# model__colsample_bytree: 0.6 lower bound
# model__min_child_weight": [3, 8, 15] 8
# "model__reg_lambda": [1, 3, 5], 1 low
# model__reg_alpha 0.5 upper bound [0, 0.2, 0.5],

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump


# -------------------------
#  Paths & constants
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# -------------------------
#  Data loading & FE
# -------------------------
def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Minimal FE, consistent with what we've been using:
      - Age_is_missing
      - Embarked_is_missing
    No manual imputation here to avoid leakage; imputation is done in pipeline.
    """
    for df in (train_df, test_df):
        df["Age_is_missing"] = df["Age"].isna().astype(int)
        df["Embarked_is_missing"] = df["Embarked"].isna().astype(int)

    feature_cols = [
        "Pclass",
        "Sex",
        "SibSp",
        "Parch",
        "Fare",
        "Age",
        "Embarked",
        "Age_is_missing",
        "Embarked_is_missing",
    ]

    X = train_df[feature_cols].copy()
    y = train_df["Survived"].astype(int).copy()
    X_test = test_df[feature_cols].copy()

    return X, y, X_test


# -------------------------
#  Preprocessing pipeline
# -------------------------
def build_preprocessor():
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked",
        "Age_is_missing",
        "Embarked_is_missing",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# -------------------------
#  XGB model (your params)
# -------------------------
def build_xgb_pipeline(preprocessor):
    """
    Use the parameter set you selected:

      colsample_bytree = 0.6
      learning_rate    = 0.03
      max_depth        = 2
      min_child_weight = 8
      n_estimators     = 500
      reg_alpha        = 0.5
      reg_lambda       = 1
      subsample        = 1.0
    """

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=8,
        subsample=1.0,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # keep 1 to avoid Windows/joblib unicode issues; raise if your env is safe
        use_label_encoder=False,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", xgb),
        ]
    )

    return pipe


# -------------------------
#  CV evaluation
# -------------------------
def evaluate_with_cv(pipe, X, y):
    """
    Run Stratified 5-fold CV and compute:
      - mean train score, std
      - mean test score, std
      - overfit gap
      - custom_score (penalize gap & std)
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=1,  # again 1 for safety
    )

    mean_train = cv_results["train_score"].mean()
    std_train = cv_results["train_score"].std()
    mean_test = cv_results["test_score"].mean()
    std_test = cv_results["test_score"].std()

    overfit_gap = mean_train - mean_test

    # You can tweak these weights; for now use 1.0 for both
    alpha_gap = 1.0
    beta_std = 1.0
    custom_score = mean_test - alpha_gap * overfit_gap - beta_std * std_test

    print("=== Cross-validation results (XGB) ===")
    print(f"Train accuracy:  {mean_train:.4f} ± {std_train:.4f}")
    print(f"Test  accuracy:  {mean_test:.4f} ± {std_test:.4f}")
    print(f"Overfit gap:     {overfit_gap:.4f}")
    print(f"Custom score:    {custom_score:.4f}")
    print("=====================================\n")

    return {
        "mean_train": mean_train,
        "std_train": std_train,
        "mean_test": mean_test,
        "std_test": std_test,
        "overfit_gap": overfit_gap,
        "custom_score": custom_score,
    }


# -------------------------
#  Train on full data & predict
# -------------------------
def fit_full_and_predict(pipe, X, y, X_test, test_df):
    """
    Fit on full training set and produce Kaggle submission.
    """
    print("Fitting XGB pipeline on full training data...")
    pipe.fit(X, y)

    # Predict probabilities then threshold at 0.5
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Sanity check on training accuracy (in-sample)
    train_pred = pipe.predict(X)
    train_acc = accuracy_score(y, train_pred)
    print(f"Train accuracy on full data: {train_acc:.4f}")

    # Build submission
    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": y_pred,
        }
    )

    out_path = DATA_DIR / "submission_xgb.csv"
    submission.to_csv(out_path, index=False)
    print(f"✅ Submission saved to {out_path}")

    # Save model
    model_path = MODELS_DIR / "titanic_xgb_model.joblib"
    dump(pipe, model_path)
    print(f"✅ Model saved to {model_path}")


# -------------------------
#  Main orchestration
# -------------------------
def main():
    train_df, test_df = load_data()
    X, y, X_test = feature_engineering(train_df, test_df)
    preprocessor = build_preprocessor()
    pipe = build_xgb_pipeline(preprocessor)

    # 1) CV evaluation (with gap & custom score)
    cv_summary = evaluate_with_cv(pipe, X, y)

    # 2) Fit full model and predict
    fit_full_and_predict(pipe, X, y, X_test, test_df)


if __name__ == "__main__":
    main()
