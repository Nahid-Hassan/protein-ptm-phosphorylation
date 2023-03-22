# Protein Post-translational Modification Phosphorylation Site Prediction using ML

In this project we trying to the importance of protein phosphorylation as a post-translational modification and its role in regulating protein functions. Dysregulation of phosphorylation can lead to various diseases, making it necessary to predict phosphorylation sites in uncharacterized amino acid sequences. However, current experimental methods have limitations, and computational methods have been proposed as an alternative. The paper explores different features that can be used for phosphorylation prediction, with protein sequence data being the most commonly used. Machine learning algorithms such as SVM, Random Forest, and XGBoost were used to predict phosphorylation sites with an average accuracy of 85% on an independent test set.

**Table of Contents**

- [Protein Post-translational Modification Phosphorylation Site Prediction using ML](#protein-post-translational-modification-phosphorylation-site-prediction-using-ml)
  - [Design Methodology](#design-methodology)
    - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
    - [Model Generation and Validation](#model-generation-and-validation)
      - [Random Forest](#random-forest)
      - [XGBoost](#xgboost)
      - [Support Vector Machine](#support-vector-machine)
    - [Model Performance Evaluation](#model-performance-evaluation)
      - [Confusion Matrix](#confusion-matrix)
  - [Results](#results)


## Design Methodology

### Data Collection and Preprocessing

The dataset is collected from [`musiteDeep_Web`](https://www.musite.net/). We worked on with phosphorylation site. 
So, we collected phosphorylation `(pS, pY, pT)` dataset. From those dataset we cut 21 residues sequence fragment.

### Model Generation and Validation

#### Random Forest
#### XGBoost

#### Support Vector Machine

### Model Performance Evaluation

#### Confusion Matrix

## Results