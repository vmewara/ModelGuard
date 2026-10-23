
# ModelGuard
## Explainable ML Monitoring and Early-Warning Platform

ModelGuard is a machine-learning monitoring and early-warning platform built using Python and Streamlit. It is designed to help monitor changes in data distributions, evaluate data quality, track monitoring results, and identify potential issues that may affect machine-learning model performance.

The project combines statistical drift detection, data quality checks, monitoring reports, and model artifacts in an interactive dashboard.

## Table of Contents

- Overview
- Features
- Monitoring Techniques
- Machine Learning Models
- Technologies Used
- Project Structure
- Installation
- Running the Application
- Monitoring Workflow
- Project Limitations
- Future Improvements
- Author

## 1. Overview

Machine-learning models may encounter changes in incoming data after deployment. These changes can affect the reliability of predictions and indicate that further investigation may be necessary.

ModelGuard provides a dashboard for examining data drift, checking data quality, viewing monitoring history, and reviewing alerts and incidents.

The project is intended for learning, experimentation, and portfolio demonstration of machine-learning monitoring concepts.

## 2. Features

### Interactive Monitoring Dashboard
- Streamlit-based user interface.
- Centralized view of monitoring results.
- Access to reports and monitoring information.

### Data Drift Detection
- Population Stability Index (PSI) analysis.
- Kolmogorov–Smirnov (KS) analysis.
- Comparison of reference and current data distributions.
- Identification of potentially significant distribution changes.

### Data Quality Monitoring
- Review of data quality reports.
- Support for examining potential data issues.
- Monitoring of incoming data batches.

### Alerts and Incidents
- View recorded alerts.
- Review incident information.
- Track potential issues identified during monitoring.

### Monitoring History
- Store and review monitoring history.
- Examine previously generated monitoring reports.
- Review available historical results.

### Simulated Data
- Work with simulated current-data batches.
- Test monitoring workflows without requiring a live production data source.

### Machine Learning Model Artifacts
- Random Forest pipeline.
- Artificial Neural Network (ANN) model.
- ANN preprocessing artifact.

## 3. Monitoring Techniques

### Population Stability Index (PSI)

PSI is a statistical measure used to compare the distribution of a variable between a reference dataset and a current dataset.

It can help identify changes in the distribution of incoming data that may warrant investigation.

### Kolmogorov–Smirnov (KS) Analysis

The Kolmogorov–Smirnov test compares two distributions to assess whether they differ statistically.

In a monitoring workflow, it can be used to examine distribution changes between reference and current data.

### Data Quality Monitoring

Data quality checks help identify potential issues in incoming data. Depending on the checks implemented in the application, these may include missing values, unexpected values, or other inconsistencies.

### Alerts and Incidents

Monitoring results can be recorded and presented as alerts or incidents so that potential issues can be reviewed.

**Note:** Drift detection identifies changes in data distributions; it does not automatically prove that model accuracy has decreased.

## 4. Machine Learning Models

ModelGuard includes the following model artifacts:

### Random Forest
A saved Scikit-learn pipeline that can include preprocessing and a trained Random Forest estimator.

### Artificial Neural Network (ANN)
A saved neural network model intended for use with the project's TensorFlow/Keras workflow.

### ANN Preprocessor
A saved preprocessing artifact associated with the ANN workflow.

These artifacts are intended to support the project's machine-learning monitoring and experimentation workflow.

## 5. Technologies Used

- **Python** — application programming and data processing.
- **Streamlit** — interactive dashboard.
- **Pandas** — data manipulation and analysis.
- **NumPy** — numerical operations.
- **Matplotlib** — data visualization.
- **Seaborn** — statistical visualization.
- **Scikit-learn** — machine-learning pipelines and statistical utilities.
- **TensorFlow/Keras** — neural network model support.
- **Joblib** — loading and saving Python model artifacts.
- **Git and GitHub** — source control and project hosting.

## 6. Project Structure

```text
ModelGuard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── last_daily_update.json
│   ├── simulated_current.csv
│   └── simulated/
│       └── simulated_current_2026-10-23.csv
├── modelguard_alerts.csv
├── modelguard_ann.keras
├── modelguard_ann_preprocessor.pkl
├── modelguard_data_quality_report.csv
├── modelguard_ks_report.csv
├── modelguard_monitoring_history.csv
├── modelguard_psi_report.csv
└── random_forest_pipeline.pkl
```

The exact files and folders may change as the project develops. The structure above reflects the files used in the current project setup.

## 7. Installation

### Prerequisites

Install Python and Git. Python 3.10 is the version used in the current development environment.

### Step 1: Clone the Repository

```bash
git clone https://github.com/vmewara/ModelGuard.git
```

### Step 2: Open the Project Folder

```bash
cd ModelGuard
```

### Step 3: Create a Virtual Environment

On Windows:

```powershell
py -m venv .venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### Step 5: Install Dependencies

```bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If you are using an activated virtual environment, you can also run:

```bash
python -m pip install -r requirements.txt
```

## 8. Running the Application

From the project directory, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open the displayed URL in your browser to access the dashboard.

## 9. Monitoring Workflow

The intended monitoring workflow is:

1. Load the available reference and current datasets.
2. Examine the current data and available data quality results.
3. Compare reference and current distributions using PSI and KS analysis.
4. Review available monitoring reports.
5. Inspect recorded alerts and incidents.
6. Review monitoring history to understand previously recorded results.

The actual workflow depends on the functionality implemented in `app.py` and the availability of the required data and model files.

## 10. Project Limitations

- The project currently uses simulated data rather than a confirmed live production data feed.
- Drift detection alone does not establish whether prediction accuracy has changed.
- Model performance monitoring requires appropriate ground-truth labels and evaluation metrics when available.
- Automated scheduling and continuous monitoring depend on the runtime environment and additional configuration.
- The saved model artifacts must be compatible with the installed Python libraries.
- The current project should be treated as a prototype, not a fully validated production monitoring service.

## 11. Future Improvements

Possible future enhancements include:

- Automated periodic monitoring.
- Additional data quality checks.
- Prediction distribution monitoring.
- Model performance tracking when actual labels become available.
- More detailed alert severity and incident management.
- Monitoring history visualizations.
- Integration with live data sources.
- Cloud deployment and automated workflows.
- Unit tests and monitoring pipeline validation.

These are potential improvements and should not be interpreted as features that are already implemented.

## 12. Author

**S Vishal**

Machine Learning | Data Science | Python

## 13. Project Status

ModelGuard is a machine-learning monitoring and portfolio project under development. It demonstrates practical concepts in statistical drift detection, data quality monitoring, saved model artifacts, and interactive dashboard development.

---

If you find this project useful, feel free to explore the repository and follow its development.
