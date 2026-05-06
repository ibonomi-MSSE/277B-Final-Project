# Makefile for setting up the environment and running peptide visualization
.PHONY: environment Data TargetWHO TargetPPV clean test # .PHONY is added when target dependencies are not files.

ENVIRONMENT=chem277B_final

environment:
	conda env create -f environment.yaml
	conda activate $(ENVIRONMENT)

Data:
	echo "Downloading Cryptic data... give me a sec!"
	python data/cryptic_consortium_data/download_cryptic_dataset.py
	echo "Creating Cryptic data... just a little longer!"
	python data/cryptic_consortium_data/create_cryptic_consortium_data.py
	echo "Querying Cryptic data... almost there!"
	python data/cryptic_consortium_data/query.py
	echo "Transforming Cryptic data... last step!"
	python data/cryptic_consortium_data/transform.py
	echo "Success! You're ready to train some models!"

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
	echo "Running boosting-method exploration on target PPV bin..."
	python model_boosting_PPV.py
	echo "Success!"

clean:
	echo "Removing model outputs and EDA folders..."
	rm -rf final_model_outputs
	rm -rf EDA_outputs
	rm -f Drug_lookup.txt
	rm -rf catboost_info


