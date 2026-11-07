import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

TARGET = "NObeyesdad"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Obesity Level Classifier", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_model(file_name):
    return joblib.load(os.path.join("model", file_name))


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                            labels=pipe.classes_)
    except ValueError:
        auc = float("nan")  # uploaded file may miss some classes
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


st.title("⚖️ Obesity Level Prediction - Model Explorer")
st.caption("UCI 'Estimation of Obesity Levels' dataset | 5 pre-trained classifiers | "
           "no retraining happens here, uploaded data is only scored.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Choose a model", list(MODEL_FILES))
    st.markdown("The CSV must have all 16 feature columns plus the "
                f"target column `{TARGET}`. Try the repo's `test_data.csv`.")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"Column `{TARGET}` not found in the uploaded file.")
    st.stop()

X, y = data.drop(columns=TARGET), data[TARGET]
pipe = load_model(MODEL_FILES[choice])

missing = [c for c in pipe.named_steps["prep"].feature_names_in_ if c not in X.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

st.subheader("Uploaded data preview")
st.write(f"{data.shape[0]} rows x {data.shape[1]} columns")
st.dataframe(data.head(), use_container_width=True)

pred, scores = evaluate(pipe, X, y)

st.subheader(f"Evaluation metrics - {choice}")
cols = st.columns(6)
for col, (name, val) in zip(cols, scores.items()):
    col.metric(name, "n/a" if np.isnan(val) else f"{val:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    labels = list(pipe.classes_)
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("All models on this uploaded data")
table = []
for name, f in MODEL_FILES.items():
    _, s = evaluate(load_model(f), X, y)
    table.append({"Model": name, **s})
table = pd.DataFrame(table).set_index("Model").round(4)
st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf"), use_container_width=True)
