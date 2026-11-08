"""Train five required classifiers and evaluate on one untouched holdout set.
Run this on BITS Virtual Lab: python train.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report,
    confusion_matrix)
from data_download import load_uci_dry_bean

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def main():
    data, target = load_uci_dry_bean()
    X = data.drop(columns=[target]).apply(pd.to_numeric, errors="raise")
    y = data[target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Keep a labelled test CSV for the Streamlit evaluation interface.
    test = X_test.copy()
    test[target] = y_test.values
    test.to_csv(ROOT / "test_data.csv", index=False)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=3.0, kernel="rbf", probability=True,
                class_weight="balanced", random_state=RANDOM_STATE))
        ]),
    }

    labels = sorted(y.unique())
    rows, reports = [], {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=model.classes_),
            "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        }
        rows.append(metrics)
        reports[name] = {
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
            "classes": model.classes_.tolist(),
        }
        safe = name.lower().replace(" ", "_").replace("-", "")
        if name == "K-Nearest Neighbors":
            safe = "k_nearest_neighbors"
        joblib.dump({"model": model, "features": X.columns.tolist(),
                     "target": target, "classes": model.classes_.tolist()},
                    MODEL_DIR / f"{safe}.joblib")

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(ROOT / "model_comparison.csv", index=False)
    with open(ROOT / "evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    summary = {
        "dataset_rows": int(len(data)), "feature_count": int(X.shape[1]),
        "target_column": target, "class_names": labels,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "winner_by_macro_f1": str(results.iloc[0]["Model"]),
        "metrics": rows,
    }
    with open(ROOT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nEvaluation on the held-out test set:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved models/, test_data.csv, model_comparison.csv, evaluation_details.json and run_summary.json")

if __name__ == "__main__":
    main()
