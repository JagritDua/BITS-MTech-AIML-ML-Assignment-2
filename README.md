# Dry Bean Variety Classification

## 1. Problem Statement

In this project, I implemented and compared five supervised machine learning classification algorithms to identify dry bean varieties using geometric measurements extracted from bean images.

The task is a seven-class classification problem. I used the same stratified training and testing split for all five models to ensure a fair comparison.

The objective was to evaluate the performance of different classification algorithms and identify which model provides the most effective classification results on the Dry Bean dataset.

## 2. Dataset Description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 602
- **Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Number of instances:** 13,611
- **Input features:** 16 numerical features
- **Target classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA
- **Problem type:** Multiclass classification

The dataset contains geometric measurements extracted from images of dry beans.

The 16 input features include:

- Area
- Perimeter
- Major Axis Length
- Minor Axis Length
- Aspect Ratio
- Eccentricity
- Convex Area
- Equivalent Diameter
- Extent
- Solidity
- Roundness
- Compactness
- Shape Factor 1
- Shape Factor 2
- Shape Factor 3
- Shape Factor 4

## 3. GitHub Repository

**Repository:** https://github.com/JagritDua/BITS-MTech-AIML-ML-Assignment-2

The repository contains the model training script, Streamlit application, saved models, evaluation results and supporting files.

## 4. Machine Learning Models Used

I implemented and evaluated the following five supervised classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Experimental Configuration

I used the following experimental configuration:

- Train-test split: 80:20
- Stratification: Enabled
- Random state: 42
- Evaluation dataset: Same held-out test dataset for all five models
- Feature scaling: Applied within the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM.
- Decision Tree: Trained without feature scaling.
- Classification type: Seven-class classification.

Using the same training and testing split allowed me to compare the performance of all five models under consistent conditions.

## 5. Evaluation Metrics

I evaluated the models using six performance metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the ability of the classifier to distinguish between classes using macro one-vs-rest ROC AUC.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures the proportion of actual instances correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which considers both correct and incorrect classifications.

Precision, recall and F1-score are reported using macro averaging so that each bean variety receives equal weight.

### Model Performance Comparison

The following results were obtained from my experiment in BITS Virtual Lab.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9166 | 0.9947 | 0.9305 | 0.9309 | 0.9303 | 0.8996 |
| Decision Tree | 0.8920 | 0.9521 | 0.9059 | 0.9080 | 0.9069 | 0.8696 |
| K-Nearest Neighbors | 0.9166 | 0.9860 | 0.9318 | 0.9271 | 0.9292 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9254** | **0.9949** | **0.9369** | **0.9377** | **0.9372** | **0.9100** |

## 6. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9166 and a macro F1-score of 0.9303. Its AUC of 0.9947 indicates strong class discrimination.

The model performed well overall, although its results suggest that a linear decision boundary may not capture every relationship between the geometric features and bean varieties.

### Decision Tree

The Decision Tree achieved an accuracy of 0.8920 and a macro F1-score of 0.9069.

It offers interpretability because its classification decisions can be represented through a series of feature-based splits. However, it produced the lowest overall performance among the five evaluated models.

### K-Nearest Neighbors

KNN achieved an accuracy of 0.9166 and a macro F1-score of 0.9292.

Feature standardization is important for distance-based classification because the input features have different numerical scales. The model achieved competitive results, although its performance was slightly below Logistic Regression and SVM.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8979 and a macro F1-score of 0.9091.

The model assumes conditional independence between input features given the class. Since several geometric measurements may be correlated, this assumption could limit its classification performance.

Despite this limitation, the model achieved a relatively high AUC of 0.9916.

### Support Vector Machine

Support Vector Machine achieved the strongest overall performance, with an accuracy of 0.9254 and a macro F1-score of 0.9372.

It also achieved the highest AUC of 0.9949 and MCC of 0.9100.

The RBF kernel allows the model to capture nonlinear relationships between the geometric measurements and bean varieties, which appears useful for this classification problem.

### Overall Comparison

Based on the measured results, Support Vector Machine performed best among the five evaluated algorithms.

It achieved the highest accuracy, macro precision, macro recall, macro F1-score, AUC and MCC.

KNN and Logistic Regression also produced competitive results, while Decision Tree achieved the lowest overall performance.

The results indicate that the SVM with an RBF kernel was the most effective model among the algorithms evaluated on this particular train-test split.

## 7. Streamlit Application

I developed an interactive Streamlit application to demonstrate and evaluate the trained classification models.

**Live Application:** https://bits-mtech-aiml-ml-assignment-2-ycs4qwjxf7ywxqgzn5vzvc.streamlit.app/

### Application Features

The application provides:

- Selection between five trained classification algorithms.
- Upload of labelled test data in CSV format.
- Loading of the included test dataset for demonstration.
- Evaluation using six performance metrics.
- Confusion matrix visualization.
- Classification report for individual bean varieties.
- Prediction preview.
- CSV download functionality for evaluation results.

The application loads the saved trained models and evaluates uploaded labelled data. It does not retrain the models during prediction or evaluation.

## 8. Repository Structure

```text
BITS-MTech-AIML-ML-Assignment-2/
│
├── app.py
├── train.py
├── data_download.py
├── requirements.txt
├── README.md
│
├── dataset.csv
├── test_data.csv
├── model_comparison.csv
├── evaluation_details.json
├── run_summary.json
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
│
└── screenshots/
```

## 9. Reproducibility and Execution

The following steps can be used to reproduce the experiment.

### Step 1: Clone the repository

```bash
git clone https://github.com/JagritDua/BITS-MTech-AIML-ML-Assignment-2.git

cd BITS-MTech-AIML-ML-Assignment-2
```

### Step 2: Create a Python environment

```bash
conda create -n drybean python=3.11 -y

conda activate drybean
```

Alternatively, a Python virtual environment can be used.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the dataset and train the models

```bash
python train.py
```

The training script retrieves the Dry Bean dataset through the UCI dataset repository interface, trains all five classifiers and generates the evaluation outputs.

### Step 5: Run the Streamlit application

```bash
streamlit run app.py
```

The application can then be accessed through the local URL provided by Streamlit.

### Step 6: Deployment

The application is deployed using Streamlit Community Cloud and is connected to the GitHub repository.

## 10. Academic Integrity and Limitations

I conducted the model training and evaluation using the BITS Virtual Lab environment.

The reported performance values correspond to the specific dataset split and experimental configuration used in this project.

The results may vary if the dataset split, model parameters or preprocessing configuration are changed.

I used AI-based tools for learning support, debugging and documentation assistance, in accordance with the applicable course academic integrity requirements.

The conclusions presented in this project are based on the measured experimental results rather than assumed model performance.

## 11. Conclusion

In this project, I implemented and compared five supervised machine learning algorithms for dry bean variety classification using 16 geometric features.

I evaluated the models using accuracy, AUC, precision, recall, F1-score and Matthews Correlation Coefficient.

Among the five classifiers, Support Vector Machine achieved the best overall results, with an accuracy of 92.54% and a macro F1-score of 93.72%.

The experiment helped me understand the differences between linear, tree-based, distance-based, probabilistic and kernel-based classification algorithms.

The Streamlit application provides an interactive interface to compare the trained models and evaluate labelled test datasets.
