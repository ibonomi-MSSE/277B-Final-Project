# Makefile for setting up the environment and running peptide visualization
.PHONY: environment TargetWHO Target PPV clean test # .PHONY is added when target dependencies are not files.

ENVIRONMENT=chem277B_final

environment:
	conda env create -f environment.yaml

TargetWHO:
	echo "Running all models on target WHO Resistance Grades..."
	python model_Logistic_Regression_WHO_grade.py
	python model_ANN_WHO_grade.py
	echo "Success!"

TargetPPV:
	echo "Running all models on target PPV..."
	python model_RandomForest_PPV.py
	echo "Success!"

clean:
	echo "Removing images..."
	rm -f ANN_model.png baseline_LogReg_random_split.png Drug_lookup.txt ANN_TrainingLossCurve.png PCA_Logistic_regression.png Logistic_regression_3Drug_holdout.png
	rm -f baseline_ppv_random_split.png mutation_holdout_classifier_reports.txt ppv_mutation_holdout_classifier.png ppv_mutation_holdout_regressor.png
	rm -f logistic_regression_PPV_classifier_reports.txt logistic_regression_ppv_classifier.png
