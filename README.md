
# ModelGuard

### Explainable ML Monitoring and Early-Warning Platform

ModelGuard is a machine-learning monitoring dashboard built with Python and Streamlit. It helps monitor data drift, data quality, and potential model-related issues.

## Features

- Interactive monitoring dashboard
- Population Stability Index (PSI) drift analysis
- Kolmogorov–Smirnov (KS) drift analysis
- Data quality monitoring
- Alerts and incidents
- Monitoring history
- Simulated current data batches
- Random Forest and Artificial Neural Network (ANN) model files

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Joblib

## Project Structure

```text
ModelGuard/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── models/
├── monitoring/
└── reports/
```

## Installation

Install the required packages:

```bash
py -m pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
py -m streamlit run app.py
```

## Monitoring Capabilities

ModelGuard is designed to help identify changes in data distributions, assess data quality, record monitoring results, and display alerts.

## Project Status

Developed as a machine-learning monitoring and portfolio project.

## Author

Vishal
