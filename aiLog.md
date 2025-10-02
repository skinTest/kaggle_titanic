# question Log

## Exploring stage
### self exploring about overfitting
#### model parameter importances
underlying is the importance of my first model(how should I )
Title_Mr: 0.23894263652977044
Sex_male: 0.2314506056252722
Fare: 0.09724493517118882
Pclass: 0.09025198067271385
Title_Mrs: 0.05198441515541184
Age: 0.04465884715115741
Deck_U: 0.04460930366322972
PassengerId: 0.03786516748500242
Title_Miss: 0.03566444878424065
SibSp: 0.02917434232208011
Parch: 0.021186267510462455
Embarked_S: 0.01332046921302208
Title_Master: 0.011461255386404039
IsAlone: 0.00920701449848716
Deck_E: 0.00875636345892931
Age_is_missing: 0.006480449870249144
Deck_D: 0.006024535399863121
Deck_B: 0.004767867266579116
Title_Rev: 0.004550929774606848
Deck_C: 0.004207160176428828
Embarked_Q: 0.0037136373811237564
Deck_F: 0.0016978029971011423
Title_Other: 0.0015716317190939977
Deck_G: 0.0011829421024819574
Deck_T: 2.4990685099755606e-05

Age and Sex, lady and children first.

#### logic regression model
I trained the model with identified feature as the first random tree model do
valid acc: .81
test acc: .53

**problems**
1. I think model feature importances might be helpful, yet I don't know how to standardize my feature and combine their coef to see the result please help me.
2. about SVM, support vector machine. I remember this is the first class of my data science course. yet, I have no impression how to code this. please notify me

#### Cross-validation
it seems weird. I set test_size to even 0.8, the validation rate is "Model Accuracy with training:  0.7952"
I just can't explain this. please 

```python
train_index = df.index
test_index = t_df.index
combined_df = pd.concat([df, t_df], ignore_index=True, sort=False)

### One-Hot Encoding
categorical_cols = ['Sex', 'Embarked', 'Deck', 'Title']
combined_encoded = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

### Re-separating into Training and Test sets
df_encoded = combined_encoded.iloc[train_index]
test_encoded = combined_encoded.iloc[test_index]

### Preparing Data for the Model
# The features for both sets must be the same, so we get the training columns
# and select them from the test set.
features = df_encoded.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1).columns

X = df_encoded[features]
y = df_encoded['Survived']

### Splitting and Training the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Model Accuracy with training: {accuracy: .4f}')
```

### ensemble part
Obviously, I need your help to implement this.

#### logistic coef
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_train)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_scaled, y_train)

# Feature importance = absolute value of coefficients
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

print(importance)
```

 Feature  Coefficient
21        Title_Mr    -0.858865
14          Deck_E     0.856396
6   Age_is_missing    -0.734765
1           Pclass    -0.657479
22       Title_Mrs     0.657146
8         Sex_male    -0.631787
10      Embarked_S    -0.614009
19    Title_Master     0.601956
12          Deck_C    -0.523030
11          Deck_B     0.480003
4            Parch    -0.472030
16          Deck_G    -0.414206
9       Embarked_Q     0.358141
13          Deck_D     0.310313
5             Fare     0.304826
7          IsAlone     0.292316
3            SibSp    -0.283458
23     Title_Other     0.252635
15          Deck_F     0.236761
17          Deck_T    -0.205571
20      Title_Miss     0.197617
0      PassengerId    -0.090901
18          Deck_U     0.074526
2              Age    -0.047505
24       Title_Rev     0.000000


### base builing process
#### logistic 

