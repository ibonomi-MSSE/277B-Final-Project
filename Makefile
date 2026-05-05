# Makefile for setting up the environment and running peptide visualization
.PHONY: environment TargetWHO TargetPPV TargetBoosting clean test # .PHONY is added when target dependencies are not files.

ENVIRONMENT=chem277B_final

environment:
	conda env create -f environment.yaml
	conda activate $(ENVIRONMENT)

TargetWHO:
	echo "Running all models on target WHO Resistance Grades..."
	python model_Logistic_Regression_WHO_grade.py
	python model_ANN_WHO_grade.py
	echo "Success!"

TargetPPV:
	echo "Running all models on target PPV..."
	python model_RandomForest_PPV.py
	python model_logistic_regression_ppv.py
	python model_ANN_PPV.py
	echo "Success!"

TargetBoosting:
	echo "Running boosting-method exploration on target PPV bin..."
	python model_boosting_PPV.py
	echo "Success!"

TargetBoosting:
	echo "Running boosting-method exploration on target PPV bin..."
	python model_boosting_PPV.py
	echo "Success!"

clean:
	echo "Removing outputs folder..."
	rm -rf final_model_outputs
	rm -f Drug_lookup.txt
	echo "Removing images..."
	rm -f boosting_PPV_classifier_reports.txt boosting_PPV_summary.csv boosting_PPV_confusion_matrix.png
