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



def baseline_logistic_model(data, target_col="resistant", test_size=0.2, random_state=42):
    print(f"Running Logistic Regression on a baseline model using the WHO dataset ...")

    data.replace([np.inf, -np.inf], np.nan, inplace=True) # find Nan values
    data = data.fillna(-1)

    # let's run a random train test split first
    X = data.drop(columns=["resistant", "ppv", "drug", "drug_norm"])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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

    cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[who_labels[i] for i in [0,1,2,3]])
    disp.plot(ax=ax, values_format="0.2f", cmap=plt.cm.Blues, colorbar=True)

    test_summary = (f"Balanced Accuracy {balanced_accuracy_score(y_test, y_pred):.3f}")
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

    return model



def main():
    data, data_genomic_positions = full_data_pipeline()

    logistic_regression_baseline = baseline_logistic_model(data)


if __name__ == "__main__":
    data, data_genomic_positions = full_data_pipeline()

    logistic_regression_baseline = baseline_logistic_model(data)



