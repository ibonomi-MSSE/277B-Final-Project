# 277B-Final-Project
## Structure-Aware Resistance Prediction in Mycobacterium tuberculosis

### Team Members:
Cris Zong, Ethan Chan, Isabella Beatrice Bonomi, Robert Craig Wallace, Sidney Alexa Brooks

### Objective and Goal of the Project
- **Objective:** To develop a machine learning model that predicts M. tuberculosis drug resistance by jointly encoding mutation profiles and drug molecular structure, rather than treating each drug as an independent categorical label.

- **Goal:** To accurately predict resistance confidence levels from mutation loci and Morgan fingerprints, and ultimately generalize to novel mutations not present in existing catalogues.

### Background
Tuberculosis remains a leading cause of infectious disease mortality worldwide (Suvvari, 2025), with multidrug-resistant and extensively drug-resistant strains presenting an escalating threat to global public health. Phenotypic drug susceptibility testing, while considered the gold standard, requires weeks (Mugenyi et al., 2024) to yield results due to the slow growth of Mycobacterium tuberculosis, delaying the initiation of effective treatment. Genotypic approaches based on whole-genome sequencing offer a rapid alternative by identifying mutations associated with drug resistance, but current interpretation frameworks (Hunt et al., 2019) function as static lookup tables that treat each drug as a categorical label, without leveraging the structural chemistry of the drugs themselves. The objective of this project is to develop a model that predicts antimicrobial resistance by jointly encoding a genetic mutation and a molecular representation of the drug, enabling the model to learn relationships between mutational context and drug chemistry and potentially generalize to novel mutations.

### Datasets
Our main dataset is the WHO Catalogue of Mutations in Mycobacterium tuberculosis Complex (2nd edition, 2023), which documents over 30,000 variants and their statistical association with phenotypic resistance across 13 anti-TB medicines derived from over 52,000 clinical isolates. 

To support our structure-aware framework, we integrated both genetic and chemical information. Mutation data was custom encoded into machine-learning-compatible features using a parser to break down protein and RNA/DNA level mutations into their associated mutation types (e.g., frameshift, deletion, nonsense, duplication). We used Mycobrowser to facilitate genomic structure embeddings of the mutations. This allows for our model to understand the importance of the positioning of the mutation(s) and its relationship to resistance.
Drug information was represented using molecular descriptors such as SMILES strings to Morgan fingerprint with each fingerprint bit encoded into its own column. Morgan fingerprints represent specific substructures including rings and functional groups using binary encoding, allowing for very precise structural patterns. ChemBERTa embeddings, open source BERT-style transformers trained on chemical datasets including PubChem, ZINC20, and MoleculeNet, were also used to capture each drug’s global structure using higher-dimensional vector representations. To address the class imbalances, we added supplemental data from the CRyPTIC Consortium which included over 12,000 isolated samples of M. tuberculosis with over half of the samples having confirmed resistance to TB drugs. Similar to the WHO dataset, this dataset included genetic variants and their associated resistance to the 13 main TB drugs.

The original target variable was the WHO's five-tier final confidence grading, ranging from "Associated with resistance" to "Not associated with resistance," framed as a multi-class classification task. Because these grades are ordinal in nature, we began with baseline models — logistic regression, PCA + Logistic Regression, and Boosting Methods — before progressing to neural networks like ANNs with softmax activation and categorical cross-entropy loss. We evaluated model performance using per-class precision, recall, and F1 score, and a confusion matrix analysis to assess whether errors cluster among neighboring grades.

Given the limited number of drugs in the dataset (15 TB drugs) and the ambiguity of WHO grades, we changed our target from WHO grades to PPV (positive predictive values of resistance). By focusing on PPV, we were able to build models that can more accurately generalize to novel mutations and capture biological relationships independent of a specific drug. While the current dataset is still specific to TB, the PPV-based framework could be extended to other bacterial pathogens if similar drug-mutation resistance datasets are available.


### Running the Repository
1. Setup Environment and install dependencies --> run `make environment` in terminal, then `conda activate chem277B_final`
2. To run EDA run `make EDA`.
3. To run all models on the initial target `WHO Resistance Grades` run `make TargetWHO`.
4. To run all models on our final target `PPV` (Positive Predicted Value) run `make TargetPPV`.
Notes: All EDA outputs will be in a folder named `EDA_outputs` and all model outputs will be saved in a folder named `final_model_outputs`
5. After running the full repository, you can clean all outputs generated with `make clean` which will remove all imaged and txt files generated.


### Statistical metrics:

Sens = Sensitivity (true positive rate — how often the mutation correctly predicts resistance)
Spec = Specificity (true negative rate — how often absence of the mutation correctly predicts susceptibility)
PPV = Positive Predictive Value (given the mutation is present, probability the strain is actually resistant)
OR = Odds Ratio (strength of association between mutation and resistance)
lb / ub = Lower bound / Upper bound (confidence interval limits, typically 95%)
k = likely Cohen's kappa (agreement statistic)
FE = Fisher's Exact test
sig = significant (whether the result passes a significance threshold)
pval / pvalue = p-value
SR = Susceptible + Resistant (combined phenotype category)

Isolate: single sample taken from a single patient.

SOLO vs non-SOLO:
SOLO = the mutation appears alone — no other resistance mutations present in that sample. This isolates the effect of just that one mutation.
Non-SOLO = all samples regardless of what other mutations are present
Note that SOLO is for KNOWN mutations. The isolate contains a sample with many bacterial cells, and they could have other mutations between each other
that are not catalogued.
Also Present vs Absent: the mutation is present or absent. Present_SOLO means the mutation is present and there are no other catalogued mutations.
Absent_SOLO means there are no catalogued mutations.


Sensitivity asks: "Of all the resistant strains, how many had this mutation?"
It's about not missing resistance — high sensitivity means few false negatives
Example: if 100 strains are resistant and 80 have the mutation, Sens = 80%

PPV asks: "Of all the strains with this mutation, how many are actually resistant?"
It's about not over-calling resistance — high PPV means few false positives
Example: if 100 strains have the mutation and 90 of those are resistant, PPV = 90%


                  Resistant    Susceptible
Mutation Present:    a              b
Mutation Absent:     c              d
OR (odds ratio) = (a/b) / (c/d)

### References
- WHO Mutation Catalogue GitHub Repository: https://github.com/GTB-tbsequencing/mutation-catalogue-2023
- Aslam, B., et al. "Antimicrobial Resistance: A Growing Serious Threat for Global Public Health." Infection and Drug Resistance, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10340576/
- Anahtar, M. N., et al. "Applications of Machine Learning to the Problem of Antimicrobial Resistance." PLOS Computational Biology, 2024. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012579
- Centers for Disease Control and Prevention. "Drug-Resistant TB." https://www.cdc.gov/tb/about/drug-resistant.html
- World Health Organization. Catalogue of Mutations in Mycobacterium tuberculosis Complex and Their Association with Drug Resistance, 2nd ed. Geneva: WHO, 2023. https://www.who.int/publications/i/item/9789240082410
- Suvvari TK. "The persistent threat of tuberculosis – Why ending TB remains elusive?" J Infect Public Health. 2025. PMC11763180.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11763180/ 
- Mugenyi N, et al. "Innovative laboratory methods for improved tuberculosis diagnosis and drug-susceptibility testing." Frontiers in Tuberculosis, 2024. https://www.frontiersin.org/journals/tuberculosis/articles/10.3389/ftubr.2023.1295979/full
- Hunt M, et al. "Antibiotic resistance prediction for Mycobacterium tuberculosis from genome sequence data with Mykrobe." Wellcome Open Research, 2019. https://wellcomeopenresearch.org/articles/4-191 
- World Health Organization. "Antimicrobial Resistance." https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance 
- Farrell LJ, et al. "Revitalizing the drug pipeline: AntibioticDB, an open access database to aid antibacterial research and development." Journal of Antimicrobial Chemotherapy, 2018. https://antibioticdb.com/ 
- Alcock BP, et al. "CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database." Nucleic Acids Research, 2023. https://card.mcmaster.ca/
- CRyPTIC Consortium. Comprehensive Resistance Prediction for Tuberculosis: an International Consortium — Datasets. https://crypticproject.info/datasets/ 
- Phelan JE, et al. "Integrating informatics tools and portable sequencing technology for rapid detection of resistance to anti-tuberculous drugs." Genome Medicine, 2019. https://tbdr.lshtm.ac.uk/ 
- Feldgarden M, et al. "AMRFinderPlus and the Reference Gene Catalog facilitate examination of the genomic links among antimicrobial resistance, stress response, and virulence." Scientific Reports, 2021. https://github.com/ncbi/amr/wiki 




MTR (Multi-Task Regression) additionally trains the model to predict 200 physicochemical properties computed by RDKit — things like logP, molecular weight, hydrogen bond donors, etc.