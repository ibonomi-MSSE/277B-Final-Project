from encoding import full_data_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np



def baseline_ppv_model(data, target_col="ppv", test_size=0.2, random_state=42):
    print(f"Running RandomForestRegressor baseline model on the WHO dataset and some CRyPTIC MIC values...")

    # let's run a random train test split first
    X = data.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # plot the predicted vs actual PPV and compare to the test set
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # __ Test Plot __
    test_MAE = mean_absolute_error(y_test, y_pred)
    test_R2 = r2_score(y_test, y_pred)
    test_summary = f"MAE Test: {test_MAE:.3f}\nR2: {test_R2:.3f}"

    ax[0].text(0.25, 0.95, test_summary,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax[0].scatter(y_test, y_pred, alpha=0.2)
    ax[0].plot([0, 1], [0, 1], "r--")
    ax[0].set_xlabel("Actual PPV")
    ax[0].set_ylabel("Predicted PPV")
    ax[0].set_title("Predicted vs Actual PPV (Test data)")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)

    # __ Train Plot __
    train_MAE = mean_absolute_error(y_train, y_pred_train)
    train_R2 = r2_score(y_train, y_pred_train)
    train_summary = f"MAE Train: {train_MAE:.3f}\nR2: {train_R2:.3f}"

    ax[1].text(0.95, 0.95, train_summary,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax[1].scatter(y_train, y_pred_train, alpha=0.2)
    ax[1].plot([0, 1], [0, 1], "r--")
    ax[1].set_xlabel("Actual PPV")
    ax[1].set_ylabel("Predicted PPV")
    ax[1].set_title("Predicted vs Actual PPV (Train data)")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)

    fig.suptitle("Baseline PPV Model Random Split (Random Forest Regressor)")

    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"baseline_ppv_random_split.png", dpi=300)
    plt.close()

    return model


def mutation_holdout_regressor(data, target_col="ppv", random_state=42):
    print(f"Running RandomForestRegressor on the WHO dataset with genomic annotations and CRyPTIC MIC on 5 groups of unseen mutations...")

    gene_key = data.filter(like="gene_").idxmax(axis=1)
    data["mutation_key"] = gene_key + "_" + data['position'].astype(str)

    X = data.drop(columns=["ppv", "resistant", "mutation_key", "position", "drug", "drug_norm"], errors="ignore")
    y = data[target_col]
    groups_new = data["mutation_key"]

    gkf = GroupKFold(n_splits=5)
    mae_summary = []
    r2_summary = []
    models = []

    #let's print the models for each group of mutations hidden
    fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups_new)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(n_estimators=200,
                                      random_state=random_state,
                                      n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        summary_text = f"MAE: {mae:.3f}\nR2: {r2:.3f}"

        mae_summary.append(mae)
        r2_summary.append(r2)
        models.append(model)

        # plot the predicted vs actual PPV
        ax[fold].scatter(y_test, y_pred, alpha=0.2)
        ax[fold].plot([0, 1], [0, 1], "r--")
        ax[fold].set_xlabel("Actual PPV")
        ax[fold].set_title(f"Fold {fold+1}")
        ax[fold].set_xlim(0, 1)
        ax[fold].set_ylim(0, 1)
        if fold == 0:
            ax[fold].set_ylabel("Predicted PPV")
        ax[fold].text(0.95, 0.95, summary_text,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        

    fig.suptitle("Mutation Holdout PPV Model: GroupKFold by Mutation")
    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"ppv_mutation_holdout_classifier.png", dpi=300)
    plt.close()

    return models, mae_summary, r2_summary


def mutation_holdout_classifier(data, target_col="ppv_bin", random_state=42):
    print(f"Running RandomForestClassifier on the WHO dataset with genomic annotations and CRyPTIC MIC on 5 groups of unseen mutations with PPV bins...")

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
    y = data[target_col]
    groups = data["mutation_key"]

    gkf = GroupKFold(n_splits=5)
    all_results = []
    metrics = []

    with open("mutation_holdout_classifier_reports.txt", "w") as f:
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

            model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )

            model.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_test)

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

        cm = confusion_matrix(results_df["actual"], results_df["predicted"], labels=["0", "1", "2", "3"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
                    yticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix of PPV Bins")

        # save the file as a png
        plt.savefig(f"ppv_mutation_holdout_classifier.png", dpi=300)
        plt.close()
    
    return metrics_df, results_df


def main():
    data, data_genomic_positions = full_data_pipeline()

    model_baseline_regressor = baseline_ppv_model(data)

    models, mae_summary, r2_summary = mutation_holdout_regressor(data_genomic_positions)

    classifier_metrics, classifier_results = mutation_holdout_classifier(data_genomic_positions)


if __name__ == "__main__":
    data, data_genomic_positions = full_data_pipeline()

    model_baseline_regressor = baseline_ppv_model(data)

    models, mae_summary, r2_summary = mutation_holdout_regressor(data_genomic_positions)

    classifier_metrics, classifier_results = mutation_holdout_classifier(data_genomic_positions)


