from encoding import full_data_pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np


def ANN_model(data, drug_lookup, target_col="resistant"):
    print(f"Running ANN on Drug holdout...")

    holdout_drugs = ['Amikacin', 'Kanamycin', 'Capreomycin']
    holdout_codes = [code for code, drug in drug_lookup.items() if drug in holdout_drugs]

    stress_test = data.copy()
    stress_test.replace([np.inf, -np.inf], np.nan, inplace=True) # replace any inf values with nan
    stress_test = stress_test.fillna(-1) # fill any nan values

    train_holdout = stress_test[~stress_test["drug"].isin(holdout_codes)].copy()
    test_holdout = stress_test[stress_test["drug"].isin(holdout_codes)].copy()

    X_train = train_holdout.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y_train = train_holdout[target_col]
    X_test = test_holdout.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y_test = test_holdout[target_col]

    # scale the data
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

    model.fit(XS_train, y_train)
    y_pred = model.predict(XS_test)

    print("Classification Report for ANN:")
    print(classification_report(y_test, y_pred))

    # plot the predicted vs actual PPV and compare to the test set
    fig, ax = plt.subplots(figsize=(10, 10))

    # __ Confusion Matrix __
    who_labels = {
        0: 'NotAssoc w R',
        1: 'NotAssoc w R(interim)',
        2: 'Assoc w R(interim)',
        3: 'Assoc w R'
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3], normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[who_labels[i] for i in [0,1,2,3]])
    disp.plot(ax=ax, values_format="0.2f", cmap=plt.cm.Blues, colorbar=True)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    test_summary = (f"Balanced Accuracy {balanced_accuracy:.3f}")
    plt.text(0.05, 0.95, test_summary,
             transform=ax.transAxes,
             verticalalignment="top",
             horizontalalignment="left",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.title("ANN on 3 Drug Holdout (WHO Grades)", pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"ANN_model.png", dpi=300)
    plt.close()


    plt.plot(model.loss_curve_)
    plt.title("ANN Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"ANN_TrainingLossCurve.png", dpi=300)
    plt.close()

    return model, balanced_accuracy


def main():
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
    ann_model = ANN_model(data, drug_lookup)


if __name__ == "__main__":
    data, data_genomic_positions, drug_lookup = full_data_pipeline()
    ann_model = ANN_model(data, drug_lookup)



