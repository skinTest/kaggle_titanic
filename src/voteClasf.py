import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load data
df = pd.read_csv("data/train.csv")
t_df = pd.read_csv("data/test.csv")

# ---- Features ----
for tar in (df, t_df):
    # Age missing flag
    tar['Age_is_missing'] = tar['Age'].isna().astype(int)

    # Fill Age with median
    tar['Age'] = tar['Age'].fillna(tar['Age'].median())

feature_cols = ['Pclass','Sex','SibSp','Parch','Age_is_missing']
X = df[feature_cols].copy()
y = df['Survived']
X_test_final = t_df[feature_cols].copy()

# ---- Define transformers ----
numeric_features = ['SibSp','Parch']   # small integers
categorical_features = ['Pclass','Sex','Age_is_missing']  # treated as categorical

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())   # helps LR
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ---- Base models ----
lr_clf = LogisticRegression(max_iter=2000, C=1.0)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,       # keep it shallow for stability
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

# ---- Voting Classifier ----
voting_clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', VotingClassifier(
        estimators=[('lr', lr_clf), ('rf', rf_clf)],
        voting='soft'
    ))
])

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X, y, cv=cv, scoring='accuracy')

print("Voting Classifier CV Mean:", scores.mean())
print("Voting Classifier CV Std:", scores.std())

# ---- Train final and predict ----
voting_clf.fit(X, y)
pred_test = voting_clf.predict(X_test_final)

submission_df = pd.DataFrame({
    'PassengerId': t_df['PassengerId'],
    'Survived': pred_test.astype(int)
})
submission_df.to_csv("submissions/submission_voting_simple.csv", index=False)
print("Saved submission_voting_simple.csv")
