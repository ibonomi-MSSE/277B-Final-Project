from encoding import full_data_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.neural_network import MLPClassifier

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np


def logistic_regression_ppv_classifier(data, target_col="ppv_bin", random_state=42):
    print(f"Running Logistic Regression Classifier on the WHO dataset with genomic annotations and CRyPTIC MIC on 5 groups of unseen mutations with PPV bins...")

    # let's add bins instead of continuous ppv
    def ppv_bin(ppv):
        if ppv >= 0.8:
            return "3"
        elif ppv >= 0.6:
            return "2"
        elif ppv >= 0.2:
            return "1"
        else:
            return "0"

    data["ppv_bin"] = data["ppv"].apply(ppv_bin)


    X = data.drop(columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"], errors="ignore")
    # noticed NA values for ins_len and del_len, will set these to 0
    X['ins_len'] = X['ins_len'].fillna(0)
    X['del_len'] = X['del_len'].fillna(0)
    X = X.fillna(0)  # only 4 rows with nas should be ok to modify

    y = data[target_col]
    
    groups = data["mutation_key"].fillna("unknown_mutation") # it looks like there were 4 unknowns

    gkf = GroupKFold(n_splits=5)
    all_results = []
    metrics = []

    with open("logistic_regression_PPV_classifier_reports.txt", "w") as f:
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

            # write the results to a txt file
            f.write(f"\n=== Fold {fold+1} ===\n")
        
            # unseen mutations in this fold
            test_mutations = groups.iloc[test_idx].unique()
            
            f.write(f"Number of unseen mutations: {len(test_mutations)}\n")
            f.write("Sample mutations:\n")
            f.write("\n".join(test_mutations[:10].astype(str)) + "\n")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
            sample_weights = X_train["has_exact_mic"] * 3 + 1 + X_train["has_variant_mic"] * 2 # add some weights to samples with MIC data

            # scale the data
            scaler = StandardScaler()
            XS_train = scaler.fit_transform(X_train)
            XS_test = scaler.transform(X_test)

            # convert back to dataframe
            XS_train = pd.DataFrame(XS_train, columns=X_train.columns)
            XS_test = pd.DataFrame(XS_test, columns=X_test.columns)

            # Run a logistic regression model as a baseline
            model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=random_state, class_weight='balanced')
            model.fit(XS_train, y_train, sample_weight=sample_weights)
            y_pred = model.predict(XS_test)

            metrics.append({
                "fold": fold + 1,
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred)
            })

            fold_df = pd.DataFrame({
                "fold": fold + 1,
                "mutation": groups.iloc[test_idx].values,
                "actual": y_test.to_numpy(),
                "predicted": y_pred
            })

            all_results.append(fold_df)

            f.write(f"\nFold {fold+1}\n")
            f.write(f"{classification_report(y_test, y_pred)}\n")
            f.write(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}\n\n")

        results_df = pd.concat(all_results, ignore_index=True)
        metrics_df = pd.DataFrame(metrics)

        results_df["correct"] = results_df["actual"] == results_df["predicted"]
        mean_balanced_accuracy = metrics_df['balanced_accuracy'].mean()

        cm = confusion_matrix(results_df["actual"], results_df["predicted"], labels=["0", "1", "2", "3"])
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
                    yticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"])
        
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix of PPV Bins")

        plt.text(0.25, 0.95, f'Balanced Accuracy: {mean_balanced_accuracy:.2f}',
                      verticalalignment='top',
                      horizontalalignment='right',
                      transform=ax.transAxes,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # save the file as a png
        plt.savefig(f"logistic_regression_ppv_classifier.png", dpi=300)
        plt.close()
    
    return metrics_df, results_df


def main():
    data, data_genomic_positions, _ = full_data_pipeline()

    metrics, results = logistic_regression_ppv_classifier(data_genomic_positions)


if __name__ == "__main__":
    data, data_genomic_positions, _ = full_data_pipeline()

    metrics, results = logistic_regression_ppv_classifier(data_genomic_positions)


