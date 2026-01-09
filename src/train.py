"""
Titanic Survival Prediction - Full Stacking Pipeline
----------------------------------------------------
Includes:
- Feature engineering
- Preprocessing
- Hyperparameter tuning
- Stacking ensemble (LR + RF + XGB)
Author: [Your Name]
Date: 2025-10
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Config
# --------------------------------------------------
RANDOM_STATE = 42
os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\joblib_temp"
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(os.environ["JOBLIB_TEMP_FOLDER"], exist_ok=True)

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

# --------------------------------------------------
# 2. Feature engineering helper
# --------------------------------------------------
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and handle missing values."""
    df = df.copy()
    df['Age_is_missing'] = df['Age'].isna().astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Age_is_missing']]

# --------------------------------------------------
# 3. Build preprocessing pipeline
# --------------------------------------------------
def build_preprocessor():
    numeric_features = ['SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Age_is_missing']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor

# --------------------------------------------------
# 4. Hyperparameter search for RandomForest
# --------------------------------------------------
def tune_random_forest(preprocessor, X, y, model_dir="models"):
    rf_pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'model__n_estimators': [100, 200, 400, 600],
        'model__max_depth': [3, 5, 7, 9, None],
        'model__min_samples_split': [2, 3, 5, 7],
        'model__min_samples_leaf': [1, 2, 3, 5],
        'model__max_features': ['sqrt', 'log2']
    }

    print("🔍 Tuning RandomForest hyperparameters...")
    search = RandomizedSearchCV(
        rf_pipe, param_distributions=param_grid,
        n_iter=25, cv=5, scoring='accuracy', n_jobs=-1,
        random_state=RANDOM_STATE, verbose=1
    )
    search.fit(X, y)
    print("✅ Best RF params:", search.best_params_)
    print(f"✅ Best CV accuracy: {search.best_score_:.4f}")

    with open(os.path.join(model_dir, "rf_best_params.json"), "w") as f:
        json.dump(search.best_params_, f, indent=2)
    return search.best_params_


def clean_param_keys(params: dict) -> dict:
    """Remove 'model__' prefix from pipeline parameter names."""
    clean = {}
    for k, v in params.items():
        if k.startswith("model__"):
            clean[k.replace("model__", "")] = v
        else:
            clean[k] = v
    return clean


# --------------------------------------------------
# 5. Build stacking model
# --------------------------------------------------
def build_stacking_model(preprocessor, rf_params):
    estimators = [
        ('lr', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(**rf_params, random_state=RANDOM_STATE)),
        ('xgb', XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, eval_metric='logloss'))
    ]

    meta_learner = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    stack_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', stack_clf)
    ])
    return pipe

# --------------------------------------------------
# 6. Train, evaluate, predict
# --------------------------------------------------
def main():
    df = pd.read_csv(TRAIN_PATH)
    t_df = pd.read_csv(TEST_PATH)

    X = prepare_features(df)
    y = df['Survived']
    X_test_final = prepare_features(t_df)

    preprocessor = build_preprocessor()

    # Load or tune RF params
    rf_params_path = "models/rf_best_params.json"
    if os.path.exists(rf_params_path):
        print("⚙️  Loading existing RF best params...")
        with open(rf_params_path) as f:
            rf_params = json.load(f)
        rf_params = clean_param_keys(rf_params)
    else:
        rf_params = tune_random_forest(preprocessor, X, y)
        rf_params = clean_param_keys(rf_params)

    pipe = build_stacking_model(preprocessor, rf_params)

    # Train/val split for evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe.fit(X_train, y_train)
    print(f"Train accuracy: {pipe.score(X_train, y_train):.4f}")
    print(f"Validation accuracy: {pipe.score(X_val, y_val):.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-val mean: {scores.mean():.4f} ± {scores.std():.4f}")

    # Final fit and predict
    pipe.fit(X, y)
    y_pred = pipe.predict(X_test_final)
    submission = pd.DataFrame({
        'PassengerId': t_df['PassengerId'],
        'Survived': y_pred
    })
    submission.to_csv("data/submission.csv", index=False)
    print("✅ Submission saved to data/submission.csv")

    dump(pipe, "models/titanic_stack_model.joblib")
    print("✅ Model saved to models/titanic_stack_model.joblib")

if __name__ == "__main__":
    main()
