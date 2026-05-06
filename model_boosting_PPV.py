"""Boosting-method exploration on the PPV-bin classification target.

Mirrors cell 45 of model_exploration_genomicPositions_andCryptic_PPV_mutation_cris_run.ipynb:
    AdaBoost / XGBoost / LightGBM / CatBoost evaluated with GroupKFold(5) over
    mutation_key, sample-weighted by MIC evidence
    (has_exact_mic*3 + 1 + has_variant_mic*2), target = 4-class PPV bin.

Same downstream comparison as cell 46 (mean Accuracy / Balanced Accuracy across folds).

Outputs:
    - boosting_PPV_classifier_reports.txt  (per-fold classification reports)
    - boosting_PPV_summary.csv             (per-model mean metrics)
    - boosting_PPV_confusion_matrix.png    (one panel per model)
"""
from encoding import full_data_pipeline

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = "final_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_ppv_bin(data):
    def _bin(p):
        if p >= 0.8:
            return "3"
        if p >= 0.6:
            return "2"
        if p >= 0.2:
            return "1"
        return "0"

    out = data.copy()
    out["ppv_bin"] = out["ppv"].apply(_bin)
    return out


def get_models(random_state=42):
    return {
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state, use_label_encoder=False, eval_metric="mlogloss"
        ),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=random_state, verbose=False),
    }


def evaluate_boosters(data, target_col="ppv_bin", random_state=42):
    print(
        "Running boosting classifiers on the WHO + CRyPTIC dataset with "
        "genomic annotations and GroupKFold(5) by mutation, target = PPV bin..."
    )

    X = data.drop(
        columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"],
        errors="ignore",
    )
    y = data[target_col]
    groups = data["mutation_key"].fillna("unknown_mutation")

    # Match cell 45 NaN handling exactly.
    if "mutation_position" in X.columns:
        X["mutation_position"] = X["mutation_position"].fillna(-1)
    if "del_len" in X.columns:
        X["del_len"] = X["del_len"].fillna(0)
    if "ins_len" in X.columns:
        X["ins_len"] = X["ins_len"].fillna(0)

    nan_counts = X.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print("NaN counts before dropping:")
        print(nan_counts)

    nan_mask = X.isna().any(axis=1)
    if nan_mask.any():
        print(f"Dropping {nan_mask.sum()} rows containing remaining NaN values.")
        X = X[~nan_mask]
        y = y[~nan_mask]
        groups = groups[~nan_mask]

    # ppv_bin is already 0-indexed ("0".."3") — just cast to int for the boosters.
    y = y.astype(int)

    gkf = GroupKFold(n_splits=5)
    models = get_models(random_state)

    summary_rows = []
    fold_predictions = {name: [] for name in models}

    with open(os.path.join(OUTPUT_DIR, "boosting_PPV_classifier_reports.txt"), "w") as f:
        for model_name, clf in models.items():
            header = f"==================== Evaluating {model_name} ===================="
            print(f"\n{header}")
            f.write(f"\n{header}\n")
            fold_metrics = []

            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                sample_weights = (
                    X_train["has_exact_mic"] * 3 + 1 + X_train["has_variant_mic"] * 2
                )

                if model_name == "CatBoost":
                    clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
                else:
                    clf.fit(X_train, y_train, sample_weight=sample_weights)

                y_pred = clf.predict(X_test)
                if y_pred.ndim > 1:
                    y_pred = y_pred.ravel()
                y_pred = y_pred.astype(int)

                acc = accuracy_score(y_test, y_pred)
                bal = balanced_accuracy_score(y_test, y_pred)
                fold_metrics.append((fold, acc, bal))

                fold_predictions[model_name].append(
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "mutation": groups.iloc[test_idx].values,
                            "actual": y_test.to_numpy(),
                            "predicted": y_pred,
                        }
                    )
                )

                report = classification_report(y_test, y_pred, zero_division=0)
                f.write(f"\nFold {fold}\n")
                f.write(report)
                f.write(f"Balanced Accuracy: {bal:.4f}\n")

            mean_acc = np.mean([m[1] for m in fold_metrics])
            mean_bal = np.mean([m[2] for m in fold_metrics])
            print(f"Mean Accuracy: {mean_acc:.4f}")
            print(f"Mean Balanced Accuracy: {mean_bal:.4f}")
            f.write(f"\nMean Accuracy: {mean_acc:.4f}\n")
            f.write(f"Mean Balanced Accuracy: {mean_bal:.4f}\n")

            summary_rows.append(
                {
                    "Model": model_name,
                    "Accuracy": mean_acc,
                    "Balanced_Accuracy": mean_bal,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False)
    return summary_df, fold_predictions


def plot_confusion_matrices(fold_predictions, save_path="boosting_PPV_confusion_matrix.png"):
    risk_labels = ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
    n = len(fold_predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, fold_dfs) in zip(axes, fold_predictions.items()):
        df = pd.concat(fold_dfs, ignore_index=True)
        cm = confusion_matrix(df["actual"], df["predicted"], labels=[0, 1, 2, 3])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=risk_labels, yticklabels=risk_labels, ax=ax,
        )
        bal = balanced_accuracy_score(df["actual"], df["predicted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nBalanced Acc: {bal:.3f}")
    fig.suptitle("Confusion Matrices of PPV Bins — Boosting Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_path), dpi=200, bbox_inches="tight")
    plt.close()


def main():
    _, data_genomic_positions, _ = full_data_pipeline()
    data_with_bin = add_ppv_bin(data_genomic_positions)

    summary_df, fold_predictions = evaluate_boosters(data_with_bin)

    print("\nSummary of Model Performance (Averaged over 5 Folds):")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "boosting_PPV_summary.csv"), index=False)
    print("Saved boosting_PPV_summary.csv and boosting_PPV_classifier_reports.txt")

    plot_confusion_matrices(fold_predictions)
    print("Saved boosting_PPV_confusion_matrix.png")


if __name__ == "__main__":
    main()
