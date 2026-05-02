from encoding import main_data_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np



def baseline_ppv_model(data, target_col="ppv", test_size=0.2, random_state=42):
    # let's run a random train test split first
    X_baseline = data.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y_baseline = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=test_size, random_state=random_state)

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
    plt.show()

    return model


# modify 
def modify_features(data):

    gene_key = data.filter(like="gene_").idxmax(axis=1)
    data["mutation_key"] = gene_key + "_" + data['position'].astype(str)

    # features and the target
    X = data.drop(columns=["resistant", "ppv", "mutation_key"])
    y = data["ppv"]
    group = data["mutation_key"]

    # Now we can use groupKFold to split the data
   
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=group)):
        print(f"Fold {fold + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train a simple model (e.g., Random Forest)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
        print(f"R2: {r2_score(y_test, y_pred):.3f}")

def mutation_holdout_model(data, target_col="ppv", test_size=0.2, random_state=42):
    # let's run a random train test split first
    X_baseline = data.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y_baseline = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X_baseline, y_baseline, test_size=test_size, random_state=random_state)

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
    plt.show()

    return model


if __name__ == "__main__":
    data = main_data_pipeline()
    model = baseline_ppv_model(data)


