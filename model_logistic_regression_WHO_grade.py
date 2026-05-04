from encoding import full_data_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np



def baseline_logistic_model(data, target_col="resistant"):
    print(f"Running Logistic Regression on a baseline model using the WHO dataset ...")

    data.replace([np.inf, -np.inf], np.nan, inplace=True) # find Nan values
    data = data.fillna(-1)

    # let's run a random train test split first
    X = data.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale the data
    scaler = StandardScaler()
    XS_train = scaler.fit_transform(X_train)
    XS_test = scaler.transform(X_test)

    # convert back to dataframe
    XS_train = pd.DataFrame(XS_train, columns=X.columns)
    XS_test = pd.DataFrame(XS_test, columns=X.columns)

    # Run a logistic regression model as a baseline
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(XS_train, y_train)
    y_pred = model.predict(XS_test)
    print("Classification Report for Baseline Logistic Regression:")
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

    plt.title("Baseline Logistic Regression Model Random Split (WHO Grades)", pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"baseline_LogReg_random_split.png", dpi=300)
    plt.close()

    return model, balanced_accuracy


def drug_holdout_logistic_model(data, drug_lookup, target_col="resistant"):
    print(f"Running Logistic Regression hiding 3 drugs (included are 2 structurally similar) ...")

    holdout_drugs = ['Amikacin', 'Kanamycin', 'Streptomycin']
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

    # convert back to dataframe
    XS_train = pd.DataFrame(XS_train, columns=X_train.columns)
    XS_test = pd.DataFrame(XS_test, columns=X_test.columns)

    # Run a logistic regression model as a baseline
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(XS_train, y_train)
    y_pred = model.predict(XS_test)

    print("Classification Report for Logistic Regression on Holdout Test Set:")
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

    plt.title("Logistic Regression 3 Drug Holdout (WHO Grades)", pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"Logistic_regression_3Drug_holdout.png", dpi=300)
    plt.close()

    return model, balanced_accuracy


def PCA_logistic_model(data, drug_lookup, target_col="resistant"):
    print(f"Running Logistic Regression with PCA ...")

    holdout_drugs = ['Amikacin', 'Kanamycin', 'Streptomycin']
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

    pca = PCA(n_components=0.95, random_state=42)

    # run PCA
    X_PCA_train = pca.fit_transform(XS_train)
    X_PCA_test = pca.transform(XS_test)

    # Run a logistic regression model as a baseline
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_PCA_train, y_train)
    y_pred = model.predict(X_PCA_test)

    print("Classification Report for Logistic Regression with PCA:")
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

    plt.title("PCA and Logistic Regression on 3 Drug Holdout (WHO Grades)", pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # save the file as a png
    plt.savefig(f"PCA_Logistic_regression.png", dpi=300)
    plt.close()

    return model, balanced_accuracy


def ANN_model(data, drug_lookup, target_col="resistant"):
    print(f"Running ANN on Drug holdout...")

    holdout_drugs = ['Amikacin', 'Kanamycin', 'Streptomycin']
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

    return model, balanced_accuracy


def main():
    data, data_genomic_positions, drug_lookup = full_data_pipeline()

    logistic_regression_baseline = baseline_logistic_model(data)
    drug_holdout_model = drug_holdout_logistic_model(data, drug_lookup)
    PCA_model = PCA_logistic_model(data, drug_lookup)
    ann_model = ANN_model(data, drug_lookup)


if __name__ == "__main__":
    data, data_genomic_positions, drug_lookup = full_data_pipeline()

    logistic_regression_baseline = baseline_logistic_model(data)
    drug_holdout_model = drug_holdout_logistic_model(data, drug_lookup)
    PCA_model = PCA_logistic_model(data, drug_lookup)
    ann_model = ANN_model(data, drug_lookup)



