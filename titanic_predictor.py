# Titanic Survival Predictor - Ensemble Model

# Combines XGBoost, Random Forest, and Logistic Regression for improved accuracy


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Load data
# ---------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ---------------------------
# Feature Engineering
# ---------------------------
def extract_title(name):
    m = re.search(r",\s*([^\.]+)\.", name)
    return m.group(1).strip() if m else "Unknown"

def cabin_to_deck(cabin):
    if pd.isna(cabin) or cabin == "":
        return "Unknown"
    return str(cabin)[0]

def add_features(df):
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["Deck"] = df["Cabin"].apply(cabin_to_deck)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    df["IsChild"] = (df["Age"] < 12).astype(int)
    df["Sex_Pclass"] = df["Sex"] + "_" + df["Pclass"].astype(str)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 80], labels=False)
    
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Noble", "Countess": "Noble", "Sir": "Noble",
        "Don": "Noble", "Jonkheer": "Noble", "Dona": "Noble",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer",
        "Dr": "Officer", "Rev": "Officer"
    }
    df["Title"] = df["Title"].replace(title_map)
    
    return df

train_fe = add_features(train)
test_fe = add_features(test)

# ---------------------------
# Prepare data
# ---------------------------
target = "Survived"
drop_cols = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]

X = train_fe.drop(columns=drop_cols)
y = train_fe[target]
X_test = test_fe.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

# ---------------------------
# Preprocessing
# ---------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------------------
# Ensemble Models
# ---------------------------
xgb = XGBClassifier(
    n_estimators=450,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

logreg = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)

# Soft voting combines probabilities for better calibration
ensemble = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("logreg", logreg)],
    voting="soft"
)

# Full pipeline
pipe = Pipeline(steps=[("preprocessor", preprocess), ("ensemble", ensemble)])

# ---------------------------
# Cross-validation
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print(f"✅ Mean CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# ---------------------------
# Train full model
# ---------------------------
pipe.fit(X, y)

# Training set check
y_pred_in = pipe.predict(X)
print(f"\nTraining Accuracy: {accuracy_score(y, y_pred_in):.4f}")
print(classification_report(y, y_pred_in, digits=3))

# ---------------------------
# Predict & Save Submission
# ---------------------------
predictions = pipe.predict(X_test).astype(int)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)
print("\n Submission file 'submission.csv' saved successfully!")
