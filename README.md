# Sleep Health and Sleep Disorder Detection

A machine learning mini project that analyzes sleep, health, and
lifestyle information to classify sleep-disorder categories.

## Project Overview

Sleep plays an important role in physical and mental well-being. This
project uses machine learning classification algorithms to identify
patterns associated with sleep disorders based on sleep habits,
lifestyle attributes, demographic information, and health indicators.

The project compares four classification models:

-   Logistic Regression
-   Support Vector Machine (SVM)
-   Random Forest
-   XGBoost

**Best overall model:** Random Forest, selected for its consistent test
and cross-validation performance.


## Objectives

-   Analyze a dataset containing sleep, health, and lifestyle-related
    attributes.
-   Preprocess missing values, categorical features, and data
    inconsistencies.
-   Develop and compare multiple machine learning classification models.
-   Evaluate the models using accuracy, weighted precision, weighted
    recall, weighted F1-score, confusion matrices, and 5-fold
    cross-validation.
-   Identify important features associated with the model's predictions.

## Dataset

-   **Dataset name:** `sleep_health_lifestyle_dataset`
-   **Source:** [Insights into Sleep Patterns and Daily Habits ---
    Kaggle](https://www.kaggle.com/datasets/siamaktahmasbi/insights-into-sleep-patterns-and-daily-habits)
-   **Size:** 400 records and 13 columns
-   **Target variable:** `Sleep Disorder`

The dataset includes demographic, lifestyle, sleep-related, and
health-related attributes.

### Features

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  Person ID                           Unique identifier for each
                                      individual

  Gender                              Gender of the individual

  Age                                 Age in years

  Occupation                          Occupational category

  Sleep Duration                      Average sleep duration in hours

  Quality of Sleep                    Self-reported sleep quality on a
                                      1--10 scale

  Physical Activity Level             Physical activity in minutes per
                                      day

  Stress Level                        Stress level on a 1--10 scale

  BMI Category                        Body Mass Index category

  Blood Pressure                      Blood pressure in
                                      systolic/diastolic format

  Heart Rate                          Heart rate in beats per minute

  Daily Steps                         Number of steps per day

  Sleep Disorder                      Target category; missing values
                                      were represented as `None` during
                                      preprocessing
  -----------------------------------------------------------------------

## Data Preprocessing

The following preprocessing steps were performed:

1.  Inspected the dataset's shape, data types, descriptive statistics,
    missing values, and duplicate rows.
2.  Replaced missing values in the `Sleep Disorder` column with `None`.
3.  Split `Blood Pressure` into two numerical features: `Systolic_BP`
    and `Diastolic_BP`.
4.  Standardized the BMI category value `Normal Weight` to `Normal`.
5.  Handled missing numerical values, where encountered, using the
    median of the corresponding feature.
6.  Checked for duplicate rows; no duplicates were found.
7.  Retained the available observations without applying an explicit
    outlier-removal technique.

### Train-Test Split

-   Training set: 80%
-   Testing set: 20%
-   `random_state=42`
-   Stratified sampling was used to preserve class proportions.

### Feature Scaling

`StandardScaler` was used for numerical feature standardization. It was
fitted on the training data and then applied to both training and
testing data to prevent data leakage.

-   Logistic Regression and SVM used scaled features.
-   Random Forest and XGBoost used unscaled features.

## Machine Learning Models

  ---------------------------------------------------------------------------
  Model                   Description             Configuration noted in the
                                                  report
  ----------------------- ----------------------- ---------------------------
  Logistic Regression     Linear classification   `random_state=42`,
                          baseline                `max_iter=1000`

  Support Vector Machine  RBF-kernel classifier   `kernel='rbf'`,
  (SVM)                   for non-linear decision `probability=True`,
                          boundaries              `random_state=42`

  Random Forest           Ensemble of decision    `n_estimators=100`,
                          trees                   `max_depth=6`,
                                                  `random_state=42`

  XGBoost                 Sequential              `eval_metric='mlogloss'`,
                          gradient-boosted        `random_state=42`
                          decision trees          
  ---------------------------------------------------------------------------

All four models were trained using the same stratified 80/20 split. A
5-fold Stratified Cross-Validation was also performed.

## Evaluation Metrics

The models were evaluated using:

-   **Accuracy:** Proportion of correctly classified samples.
-   **Weighted Precision:** Precision averaged across classes, weighted
    by class support.
-   **Weighted Recall:** Recall averaged across classes, weighted by
    class support.
-   **Weighted F1-score:** Weighted harmonic mean of precision and
    recall.
-   **Confusion Matrix:** Displays correct and incorrect predictions for
    each class.
-   **5-Fold Cross-Validation Accuracy:** Average accuracy across five
    stratified validation folds.

## Results and Model Comparison

### Performance Summary

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | 5-Fold CV Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.50% | 0.5323 | 0.7250 | 0.6139 | 70.31% |
| Support Vector Machine (SVM) | 72.50% | 0.5256 | 0.7250 | 0.6094 | 72.50% |
| Random Forest | 72.50% | 0.5256 | 0.7250 | 0.6094 | 72.50% |
| XGBoost | 70.00% | 0.5957 | 0.7000 | 0.6295 | 63.75% |

### Key Findings

-   Logistic Regression, SVM, and Random Forest achieved the highest
    test accuracy at **72.50%**.
-   XGBoost achieved the highest weighted precision (**0.5957**) and
    weighted F1-score (**0.6295**).
-   SVM and Random Forest achieved the highest 5-fold cross-validation
    accuracy (**72.50%**).
-   Random Forest feature-importance analysis ranked Daily Steps, Sleep
    Duration, and Physical Activity Level among the most important
    features.

### Best Model: Random Forest

Random Forest was selected as the best overall model because it showed
consistent performance across the test set and cross-validation:

| Evaluation Metric | Random Forest Result |
|---|---:|
| Test Accuracy | **72.50%** |
| 5-Fold Cross-Validation Accuracy | **72.50%** |
| Weighted Precision | **0.5256** |
| Weighted Recall | **0.7250** |
| Weighted F1-Score | **0.6094** |

Although XGBoost achieved higher weighted precision and F1-score, its
test accuracy was 70.00% and its cross-validation accuracy was 63.75%.
Random Forest was therefore selected based on its more consistent
overall performance and its feature-importance analysis.

## Feature Importance

The Random Forest feature-importance analysis identified the following
features among the most influential:

| Rank | Feature |
|---:|---|
| 1 | Daily Steps |
| 2 | Sleep Duration |
| 3 | Physical Activity Level |
| 4 | Quality of Sleep |
| 5 | Age |
| 6 | Heart Rate |
| 7 | Stress Level |
| 8 | Diastolic Blood Pressure |
| 9 | Systolic Blood Pressure |
| 10 | Occupation -- Office Worker |

## Advantages

-   Compares four machine learning algorithms using multiple evaluation
    metrics.
-   Uses stratified train-test splitting and 5-fold stratified
    cross-validation.
-   Provides feature-importance information through Random Forest.
-   Reports model performance using more than accuracy alone.

## Limitations

-   The dataset contains only 400 records, which may limit
    generalization.
-   Missing values in the `Sleep Disorder` target column were treated as
    `None`; this assumption may affect results if some missing values
    represent unknown labels.
-   No explicit outlier-removal method was applied.
-   Advanced class-imbalance methods such as SMOTE or class weighting
    were not used.
-   Hyperparameter tuning with GridSearchCV or RandomizedSearchCV was
    not performed.
-   ROC-AUC evaluation was not included.

## Future Scope

-   Use a larger and more diverse sleep-health dataset.
-   Apply class-imbalance techniques such as SMOTE or class weighting.
-   Perform hyperparameter tuning for SVM, Random Forest, and XGBoost.
-   Develop a user-friendly application that accepts sleep and lifestyle
    information and predicts a sleep-disorder category.
-   Explore additional health and lifestyle features where available.

## Project Team

This mini project was completed as part of **BAI702 -- Machine Learning
II**, Seventh Semester, Bachelor of Engineering in Computer Science and
Engineering (Artificial Intelligence and Machine Learning), AMC
Engineering College, Bengaluru, during the academic year 2026--27.

| Team Member |
|---|
| Sandhya D |
| Sanjana M P |
| Shravani V |
| Teja Shree V |

**Project Guide:** Prof. Bhavya K Bharathan, Assistant Professor,
Department of CSE (AIML).
