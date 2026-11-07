"""Train five classifiers on the UCI Obesity Levels dataset and save them.

Run:  python train_models.py
Outputs:
    model/<name>.joblib   - full pipeline (preprocessing + classifier)
    test_data.csv         - held-out test split (features + target) for the app
    results/metrics.csv   - metrics of all models on the held-out split
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RAW_CSV = os.path.join("data_raw", "ObesityDataSet_raw_and_data_sinthetic.csv")
TARGET = "NObeyesdad"
SEED = 2026

df = pd.read_csv(RAW_CSV)
print("Dataset shape:", df.shape)

num_cols = df.drop(columns=TARGET).select_dtypes(include="number").columns.tolist()
cat_cols = df.drop(columns=TARGET).select_dtypes(exclude="number").columns.tolist()
print("Numeric features    :", num_cols)
print("Categorical features:", cat_cols)

X = df.drop(columns=TARGET)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


def make_preprocessor():
    # GaussianNB needs dense input, so the encoder returns a dense array
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


classifiers = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "svm": SVC(kernel="rbf", C=5, probability=True, random_state=SEED),
}

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

rows = []
for name, clf in classifiers.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba, multi_class="ovr",
                             average="weighted", labels=pipe.classes_),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted"),
        "F1": f1_score(y_test, pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    joblib.dump(pipe, os.path.join("model", f"{name}.joblib"))
    print("saved", name)

results = pd.DataFrame(rows).round(4)
results.to_csv(os.path.join("results", "metrics.csv"), index=False)
print(results.to_string(index=False))

test_out = X_test.copy()
test_out[TARGET] = y_test
test_out.to_csv("test_data.csv", index=False)
print("test_data.csv rows:", len(test_out))
