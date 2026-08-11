"""Optimized LOOCV Evaluation with Robust Scaling, Multiclass L1-Logistic Regression, and SVM."""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")


def main():
    print("1. Loading expression matrix and metadata...", flush=True)
    X = pd.read_csv("data/processed/GSE52194_expression_matrix.csv", index_col=0)
    y_df = pd.read_csv("data/processed/GSE52194_labels.csv")

    # Align samples
    common = X.index.intersection(y_df["sample_id"])
    X = X.loc[common]
    y = y_df.set_index("sample_id").loc[common, "target"]

    # Log2 Transformation
    print("2. Preprocessing & Selecting Top High Variance Genes...", flush=True)
    X_log = np.log2(X + 1.0)
    
    # Select Top 1000 Most Variable Genes
    gene_variances = X_log.var(axis=0)
    top_hvg = gene_variances.nlargest(1000).index
    X_filtered = X_log[top_hvg]

    total_samples = len(X_filtered)
    print(
        f"    Ready: {total_samples} samples x {X_filtered.shape[1]} highly variable genes.\n",
        flush=True,
    )

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    print("3. Running LOOCV Evaluation loop with Multiclass L1-Logistic Regression...", flush=True)

    for step, (train_idx, test_idx) in enumerate(
        loo.split(X_filtered), start=1
    ):
        X_train, X_test = X_filtered.iloc[train_idx], X_filtered.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 2. Select Top 50 features inside CV fold
        k_features = min(50, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)

        # 3. One-vs-Rest L1 Logistic Regression (Handles multiclass for high-dim RNA-Seq)
        base_lr = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.5,
            random_state=42,
            class_weight="balanced"
        )
        clf = OneVsRestClassifier(base_lr)
        clf.fit(X_train_sel, y_train)

        actual = y_test.values[0]
        pred = clf.predict(X_test_sel)[0]

        y_true.append(actual)
        y_pred.append(pred)

        print(
            f"    [Sample {step:02d}/{total_samples:02d}] Actual: {actual:<10} | Predicted: {pred}",
            flush=True,
        )

    print("\n===========================================", flush=True)
    print("    LOOCV Final Performance Summary", flush=True)
    print("===========================================", flush=True)
    print(classification_report(y_true, y_pred, zero_division=0), flush=True)
    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(y_true, y_pred), flush=True)


if __name__ == "__main__":
    main()
