from encoding import full_data_pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import os

OUTPUT_DIR = "final_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ANN_model_PPV(data, target_col="ppv"):
    print(f"Running ANN on PPV ......")


    X = data.drop(columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"], errors="ignore")
    y = data[target_col]
    
    groups = data["mutation_key"].fillna("unknown_mutation") # it looks like there were 4 unknowns

    gkf = GroupKFold(n_splits=5)
    mae_summary = []
    r2_summary = []
    models = []

    #let's print the models for each group of mutations hidden
    fig, ax = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)

    with open(os.path.join(OUTPUT_DIR, "ANN_mutation_holdout_reports.txt"), "w") as f:
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
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)

            scaler = StandardScaler()
            XS_train = scaler.fit_transform(X_train)
            XS_test = scaler.transform(X_test)

            # fit Ann
            model = MLPRegressor(hidden_layer_sizes=(64,32),
                                    activation="relu",
                                    alpha=0.001,
                                    max_iter=500,
                                    random_state=42,
                                    early_stopping=True,
                                    validation_fraction=0.1)

            model.fit(XS_train, y_train, sample_weight=sample_weights)
            y_pred = model.predict(XS_test)

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
        

        fig.suptitle("ANN Model (Target PPV): GroupKFold by Mutation")
        plt.tight_layout()

        # save the file as a png
        plt.savefig(os.path.join(OUTPUT_DIR, "ANN_model_PPV.png"), dpi=300)
        plt.close()

    return models, mae_summary, r2_summary


def ANN_model_PPV_bins(data, target_col="ppv_bin"):
    print(f"Running ANN on PPV bins ......")

    def ppv_bin(ppv):
        if ppv >= 0.8:
            return 3
        elif ppv >= 0.6:
            return 2
        elif ppv >= 0.2:
            return 1
        else:
            return 0

    data["ppv_bin"] = data["ppv"].apply(ppv_bin)


    X = data.drop(columns=["ppv", "ppv_bin", "resistant", "mutation_key", "drug", "drug_norm"], errors="ignore")
    y = data[target_col]
    
    groups = data["mutation_key"].fillna("unknown_mutation") # it looks like there were 4 unknowns

    gkf = GroupKFold(n_splits=5)
    all_results = []
    metrics = []

    with open(os.path.join(OUTPUT_DIR, "ANN_model_PPV_bin_reports.txt"), "w") as f:
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
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)

            scaler = StandardScaler()
            XS_train = scaler.fit_transform(X_train)
            XS_test = scaler.transform(X_test)

            # fit Ann
            model = MLPClassifier(hidden_layer_sizes=(64,32),
                                    activation="relu",
                                    alpha=0.001,
                                    max_iter=500,
                                    random_state=42,
                                    early_stopping=True,
                                    validation_fraction=0.1)

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

        cm = confusion_matrix(results_df["actual"], results_df["predicted"], labels=[0, 1, 2, 3])
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
                    yticklabels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("ANN Confusion Matrix of PPV Bins")
        plt.text(0.25, 0.95, f'Balanced Accuracy: {mean_balanced_accuracy:.2f}',
                      verticalalignment='top',
                      horizontalalignment='right',
                      transform=ax.transAxes,
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # save the file as a png
        plt.savefig(os.path.join(OUTPUT_DIR, "ANN_model_PPV_bin.png"), dpi=300)
        plt.close()
    return metrics_df, results_df


def main():
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
    ann_model_ppv = ANN_model_PPV(data_genomic_positions)
    ann_model_ppv_bins = ANN_model_PPV_bins(data_genomic_positions)


if __name__ == "__main__":
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
    ann_model_ppv = ANN_model_PPV(data_genomic_positions)
    ann_model_ppv_bins = ANN_model_PPV_bins(data_genomic_positions)



