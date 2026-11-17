# Sleep Health and Sleep Disorder Detection

An end-to-end Machine Learning pipeline to analyze sleep health and predict various sleep disorders (e.g., Insomnia, Sleep Apnea) based on lifestyle, cardiovascular, and daily habit metrics.

## 🚀 Project Features
- **Comprehensive Data Processing**: Cleans and preprocesses numeric and categorical features, handles missing values, and processes systolic/diastolic blood pressure readings.
- **Multiple Classifier Evaluations**: Compares performance across **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost**.
- **Rich Visualizations**: Outputs correlation summaries, model metric comparison bar charts, and confusion matrices.

## 📊 Detailed Model Evaluation Results

Below is the complete classification report and performance breakdown across all models, including both overall metrics and class-specific metrics:

| Model | Class / Metric Type | Accuracy | Precision | Recall | F1-Score | CV Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Overall | 0.725 | 0.5323 | 0.7250 | 0.6139 | 0.7031 |
| | Class: Insomnia | - | 0.0000 | 0.0000 | 0.0000 | - |
| | Class: None | - | 0.7342 | 1.0000 | 0.8467 | - |
| | Class: Sleep Apnea | - | 0.0000 | 0.0000 | 0.0000 | - |
| **SVM** | Overall | 0.725 | 0.5256 | 0.7250 | 0.6094 | 0.7250 |
| | Class: Insomnia | - | 0.0000 | 0.0000 | 0.0000 | - |
| | Class: None | - | 0.7250 | 1.0000 | 0.8406 | - |
| | Class: Sleep Apnea | - | 0.0000 | 0.0000 | 0.0000 | - |
| **Random Forest** | Overall | 0.725 | 0.5256 | 0.7250 | 0.6094 | 0.7250 |
| | Class: Insomnia | - | 0.0000 | 0.0000 | 0.0000 | - |
| | Class: None | - | 0.7250 | 1.0000 | 0.8406 | - |
| | Class: Sleep Apnea | - | 0.0000 | 0.0000 | 0.0000 | - |
| **XGBoost** | Overall | 0.700 | 0.5957 | 0.7000 | 0.6295 | 0.6375 |
| | Class: Insomnia | - | 0.3333 | 0.1250 | 0.1818 | - |
| | Class: None | - | 0.7297 | 0.9310 | 0.8182 | - |
| | Class: Sleep Apnea | - | 0.0000 | 0.0000 | 0.0000 | - |

## 📂 Files inside the Repository
- `sleep_disorder_detection.py`: Complete executable Python pipeline script.
- `detailed_model_results.csv`: Per-class and overall classification reports.
- `model_results.csv`: Summary table of performance metrics across models.
- `model_comparison_chart.png`: Visual comparison of accuracy, precision, recall, and F1-score.
- `confusion_matrices.png`: Heatmaps showing actual vs. predicted classifications.
