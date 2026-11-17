# Sleep Health and Sleep Disorder Detection

An end-to-end Machine Learning pipeline to analyze sleep health and predict various sleep disorders (e.g., Insomnia, Sleep Apnea) based on lifestyle, cardiovascular, and daily habit metrics.

## 🚀 Project Features
- **Comprehensive Data Processing**: Cleans and preprocesses numeric and categorical features, handles missing values, and processes systolic/diastolic blood pressure readings.
- **Multiple Classifier Evaluations**: Compares performance across **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost**.
- **Rich Visualizations**: Outputs correlation summaries, model metric comparison bar charts, and confusion matrices.

## 📊 Model Evaluation Results
The models are cross-validated and tested on standard stratified splits. Detailed classification reports and visual diagrams are generated dynamically.

## 📂 Files inside the Repository
- `sleep_disorder_detection.py`: Complete executable Python pipeline script.
- `detailed_model_results.csv`: Per-class and overall classification reports.
- `model_results.csv`: Summary table of performance metrics across models.
- `model_comparison_chart.png`: Visual comparison of accuracy, precision, recall, and F1-score.
- `confusion_matrices.png`: Heatmaps showing actual vs. predicted classifications.
