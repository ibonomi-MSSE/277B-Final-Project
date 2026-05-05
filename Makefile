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
	python model_logistic_regression_ppv.py
	python model_ANN_PPV.py
	echo "Success!"

clean:
	echo "Removing outputs folder..."
	rm -rf final_model_outputs
	rm -f Drug_lookup.txt
