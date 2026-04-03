# 277B-Final-Project
## Structure-Aware Resistance Prediction in Mycobacterium tuberculosis via MIC Regression

### Team Members:
Cris Zong, Ethan Chan, Isabella Beatrice Bonomi, Robert Craig Wallace, Sidney Alexa Brooks

### Objective and Goal of the Project
- **Objective:** To develop a machine learning model that predicts M. tuberculosis drug resistance by jointly encoding mutation profiles and drug molecular structure, rather than treating each drug as an independent categorical label.

- **Goal:** To accurately predict resistance confidence levels from mutation loci and Morgan fingerprints, and ultimately generalize to novel anti-TB compounds not present in existing catalogues.

### Background
Tuberculosis remains a leading cause of infectious disease mortality worldwide (Suvvari, 2025), with multidrug-resistant and extensively drug-resistant strains presenting an escalating threat to global public health. Phenotypic drug susceptibility testing, while considered the gold standard, requires weeks (Mugenyi et al., 2024) to yield results due to the slow growth of Mycobacterium tuberculosis, delaying the initiation of effective treatment. Genotypic approaches based on whole-genome sequencing offer a rapid alternative by identifying mutations associated with drug resistance, but current interpretation frameworks (Hunt et al., 2019) function as static lookup tables that treat each drug as a categorical label, without leveraging the structural chemistry of the drugs themselves. The objective of this project is to develop a model that predicts antimicrobial resistance by jointly encoding a genetic mutation and a molecular representation of the drug, enabling the model to learn relationships between mutational context and drug chemistry and potentially generalize to novel or experimental anti-TB compounds.

### Proposal
We will use the WHO Catalogue of Mutations in Mycobacterium tuberculosis Complex (2nd edition, 2023), which documents over 30,000 variants and their statistical association with phenotypic resistance across 13 anti-TB medicines derived from over 52,000 clinical isolates. Each drug will be represented as a Morgan fingerprint derived from its SMILES string, capturing structural features of the compound; if time allows, we may explore a graph neural network to encode molecular structure directly. Each mutation will be encoded as a binary vector over known resistance-associated loci, representing shortened genomic information about the strain. The target variable is the WHO's five-tier final confidence grading, ranging from "Associated with resistance" to "Not associated with resistance," framed as a multi-class classification task. Because these grades are ordinal in nature, we will explore weighted loss functions that penalize distant misclassifications more heavily than adjacent ones. We will begin with baseline models — decision trees and logistic regression — before progressing to neural networks with softmax activation and categorical cross-entropy loss. We will evaluate model performance using per-class precision, recall, and F1 score, macro-averaged AUROC, and a confusion matrix analysis to assess whether errors cluster among neighboring grades.


### References
- WHO Mutation Catalogue GitHub Repository: https://github.com/GTB-tbsequencing/mutation-catalogue-2023
- Aslam, B., et al. "Antimicrobial Resistance: A Growing Serious Threat for Global Public Health." Infection and Drug Resistance, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10340576/
- Anahtar, M. N., et al. "Applications of Machine Learning to the Problem of Antimicrobial Resistance." PLOS Computational Biology, 2024. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012579
- Centers for Disease Control and Prevention. "Drug-Resistant TB." https://www.cdc.gov/tb/about/drug-resistant.html
World Health Organization. Catalogue of Mutations in Mycobacterium tuberculosis Complex and Their Association with Drug Resistance, 2nd ed. Geneva: WHO, 2023. https://www.who.int/publications/i/item/9789240082410
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
