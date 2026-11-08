"""Streamlit interface: evaluate saved classifiers on an uploaded labelled CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")
st.title("Dry Bean Variety Classification")
st.caption("A comparison of five supervised classification algorithms using the UCI Dry Bean dataset.")
st.info("Upload a labelled test CSV containing all 16 input features and the target column. The app loads saved models; it does not retrain them.")

available = {name: path for name, filename in MODEL_FILES.items() if (path := MODEL_DIR / filename).exists()}
if not available:
    st.error("No trained model files found. Run `python train.py` first and commit the generated models/ directory.")
    st.stop()

model_name = st.selectbox("Choose a classifier", list(available))
upload = st.file_uploader("Upload labelled test data (CSV)", type=["csv"])
if upload is None:
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.caption("No file uploaded. You can use the included test_data.csv as a demonstration.")
        if st.button("Load included test data"):
            st.session_state["test_csv_bytes"] = sample.read_bytes()
    else:
        st.warning("Upload a test CSV to calculate evaluation metrics.")
        st.stop()
    raw = st.session_state.get("test_csv_bytes")
    if raw is None:
        st.stop()
    from io import BytesIO
    data = pd.read_csv(BytesIO(raw))
else:
    data = pd.read_csv(upload)

bundle = joblib.load(available[model_name])
model, features, target = bundle["model"], bundle["features"], bundle["target"]
missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
try:
    X = data[features].apply(pd.to_numeric, errors="raise")
    y = data[target].astype(str)
except Exception as exc:
    st.error(f"Could not read feature/target columns: {exc}")
    st.stop()

pred = model.predict(X)
proba = model.predict_proba(X)
classes = model.classes_
try:
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
except ValueError:
    auc = float("nan")
metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC (macro OVR)": auc,
    "Precision (macro)": precision_score(y, pred, average="macro", zero_division=0),
    "Recall (macro)": recall_score(y, pred, average="macro", zero_division=0),
    "F1 (macro)": f1_score(y, pred, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(y, pred),
}
st.subheader(f"Evaluation results — {model_name}")
cols = st.columns(3)
for i, (label, value) in enumerate(metrics.items()):
    cols[i % 3].metric(label, "N/A" if pd.isna(value) else f"{value:.4f}")
left, right = st.columns([1, 1])
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=classes, xticks_rotation=45,
        cmap="Blues", colorbar=False, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
st.subheader("Prediction preview")
preview = data.copy()
preview["Predicted_" + target] = pred
st.dataframe(preview.head(30), use_container_width=True)
st.download_button("Download predictions", preview.to_csv(index=False).encode("utf-8"),
    file_name="dry_bean_predictions.csv", mime="text/csv")
