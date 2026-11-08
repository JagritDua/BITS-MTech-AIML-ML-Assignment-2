# Dry Bean Variety Classification

## a. Problem statement

In this project, I implemented and compared five supervised machine learning classification algorithms to identify dry bean varieties using geometric measurements extracted from bean images.

The task is a seven-class classification problem. I used the same stratified training and testing split for all five models to ensure a fair comparison.

## b. Dataset description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 602
- **URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Instances:** 13,611
- **Input features:** 16 numerical features
- **Target classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA
- **Problem type:** Multiclass classification

The features describe the geometric properties of dry beans, including area, perimeter, major and minor axis lengths, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and shape factors.

## c. GitHub Repository Link

**Repository:** https://github.com/JagritDua/BITS-MTech-AIML-ML-Assignment-2

## d. Models used and evaluation

I implemented the following five classification algorithms using Python and Scikit-learn:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

I used an 80:20 stratified train-test split with random state 42. Feature scaling was applied within the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree was trained without feature scaling.

I evaluated all models using the same held-out test dataset.

### Evaluation metrics

The following table presents the actual results obtained from my BITS Virtual Lab execution. All values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9166 | 0.9947 | 0.9305 | 0.9309 | 0.9303 | 0.8996 |
| Decision Tree | 0.8920 | 0.9521 | 0.9059 | 0.9080 | 0.9069 | 0.8696 |
| K-Nearest Neighbors | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | 0.9254 | 0.9949 | 0.9369 | 0.9377 | 0.9372 | 0.9100 |

### Observations on model performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 91.66% and a macro F1 score of 0.9303. Its AUC score of 0.9947 indicates excellent class discrimination. The results show that a linear classification approach performs well on this dataset, although it does not achieve the highest overall performance.

**2. Decision Tree**

The Decision Tree classifier achieved an accuracy of 89.20% and a macro F1 score of 0.9069. It provides an interpretable classification approach, but its performance was lower than the other models in this experiment.

**3. K-Nearest Neighbors**

KNN achieved 91.66% accuracy and a macro F1 score of 0.9292. Its performance was close to Logistic Regression. Feature standardization was important because KNN relies on distance calculations between observations.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved 89.79% accuracy and a macro F1 score of 0.9091. Its AUC score was 0.9916, showing strong class discrimination. Its lower accuracy may partly be associated with the conditional independence assumption, which can be restrictive when geometric features are correlated.

**5. Support Vector Machine**

SVM achieved the highest accuracy of 92.54%, macro precision of 0.9369, macro recall of 0.9377 and macro F1 score of 0.9372. It also achieved the highest AUC of 0.9949 and MCC of 0.9100. These results indicate that SVM provided the strongest overall classification performance among the five models in this experiment.

### Overall winner

Based on my experimental results, **Support Vector Machine (SVM)** is the overall winner.

It achieved the highest accuracy, AUC, precision, recall, F1 score and MCC among the five classifiers. Its balanced performance across these metrics makes it the strongest model for this particular dataset and train-test split.

## Streamlit application

I developed an interactive Streamlit application to evaluate the saved classification models.

The application includes:

- CSV upload functionality for labelled test data.
- A dropdown menu to select any of the five classifiers.
- Display of all six evaluation metrics.
- Confusion matrix.
- Classification report.
- Prediction preview and CSV download.
- Loading of saved trained models without retraining on uploaded test data.

**Live Streamlit URL:** To be added after deployment.

## Reproducibility and execution

1. Create or activate a Python environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Train the models:

   `python train.py`

4. Inspect the generated files:
   - `dataset.csv`
   - `test_data.csv`
   - `model_comparison.csv`
   - `evaluation_details.json`
   - `run_summary.json`
   - `models/`

5. Start the Streamlit application:

   `streamlit run app.py`

6. Deploy `app.py` from this GitHub repository using Streamlit Community Cloud.

## Repository contents

- `app.py` - Streamlit evaluation application.
- `train.py` - Model training and evaluation script.
- `data_download.py` - Dataset loading utility.
- `requirements.txt` - Required Python dependencies.
- `dataset.csv` - Dataset used for the experiment.
- `test_data.csv` - Labelled held-out test dataset.
- `models/` - Five saved trained model pipelines.
- `model_comparison.csv` - Evaluation metrics.
- `evaluation_details.json` - Detailed evaluation results.
- `run_summary.json` - Experiment summary.
- `screenshots/` - Assignment execution screenshots.

## Academic integrity and limitations

I conducted the model training and evaluation in BITS Virtual Lab and used the same held-out test dataset to compare all five classifiers. The reported results correspond to this particular dataset split and experimental configuration. I acknowledge the use of AI tools for learning support, debugging and documentation assistance, in accordance with the course academic integrity policy.
