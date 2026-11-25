
# 🛡️ ModelGuard AI
### Explainable Machine Learning Monitoring & GenAI-Powered Model Analysis

ModelGuard AI is an intelligent machine learning monitoring platform designed to monitor model-related data changes, detect data drift, evaluate data quality, track model performance, and investigate potential monitoring issues using Generative AI.

The platform combines traditional machine learning monitoring techniques with GenAI-powered explanations to make model monitoring more understandable and accessible.

It provides an interactive Streamlit dashboard with monitoring reports, statistical drift analysis, alerts, model comparisons, and AI-powered assistance.

---

## 🎯 Project Objectives

Machine learning models can experience performance degradation when real-world data changes over time. This phenomenon is commonly associated with data drift and changing data distributions.

ModelGuard AI aims to:

- Detect changes in incoming data distributions.
- Monitor data quality and identify potential issues.
- Generate drift reports using statistical techniques.
- Track monitoring history and alerts.
- Compare machine learning model performance.
- Use Generative AI to explain monitoring findings.
- Provide AI-assisted incident investigation and performance recommendations.
- Package the application using Docker for portability.

---

## 🚀 Key Features

### 1. Monitoring Dashboard
- Interactive Streamlit dashboard.
- Overview of monitoring activities.
- Summary of detected drift.
- Monitoring metrics and visualizations.
- Centralized monitoring information.

### 2. Data Drift Detection
Uses statistical techniques to identify changes between reference and current datasets.

**Population Stability Index (PSI)**
- Measures changes in feature distributions.
- Categorizes drift levels.
- Helps identify features requiring investigation.

**Kolmogorov-Smirnov (KS) Test**
- Compares reference and current distributions.
- Calculates KS statistics and p-values.
- Helps identify statistically significant distribution changes.

### 3. Data Quality Monitoring
- Monitors missing values.
- Identifies data quality issues.
- Generates data quality reports.
- Helps assess dataset reliability.

### 4. Alerts & Incident Management
- Tracks detected drift conditions.
- Generates monitoring alerts.
- Maintains incident information.
- Supports investigation of potential monitoring issues.

### 5. Model Performance Analysis
Compares two machine learning models:

- Random Forest
- Artificial Neural Network (ANN)

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 6. GenAI Assistant
An AI-powered assistant integrated using the Groq API.

Capabilities:
- Answers questions about ModelGuard.
- Explains drift detection results.
- Summarizes monitoring findings.
- Helps users understand model monitoring concepts.

### 7. AI Incident Investigator
- Investigates detected drift findings.
- Explains potentially affected features.
- Provides possible causes and investigation suggestions.
- Helps users understand monitoring incidents.

### 8. AI Performance Advisor
- Compares Random Forest and ANN performance.
- Explains precision and recall trade-offs.
- Provides AI-generated model improvement suggestions.
- Helps interpret evaluation metrics.

### 9. Docker Support
- Containerized Streamlit application.
- Docker image available on Docker Hub.
- Simplified application setup.
- Portable execution across supported Docker environments.

---

## 🏗️ System Architecture

```text
              Reference Dataset
                     |
                     v
              Current Dataset
                     |
                     v
              Data Processing
                     |
                     v
             Data Quality Checks
                     |
                     v
             Drift Detection
                PSI + KS
                     |
                     v
           Monitoring Reports
                     |
                     v
             Alerts & Incidents
                     |
          +----------+----------+
          |          |          |
          v          v          v
       GenAI      Incident   Performance
      Assistant  Investigator  Advisor
          |          |          |
          +----------+----------+
                     |
                     v
              Streamlit Dashboard
                     |
                     v
                Docker Image
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Statistical Analysis | SciPy |
| ML Models | Random Forest, ANN |
| Drift Detection | PSI, KS Test |
| Generative AI | Groq API |
| Containerization | Docker |
| Version Control | Git, GitHub |

---

## 📊 Machine Learning Models

ModelGuard includes model performance comparison using Random Forest and Artificial Neural Network models.

| Metric | Random Forest | ANN |
|---|---:|---:|
| Accuracy | 77.86% | 78.71% |
| Precision | 57.21% | 59.44% |
| Recall | 65.78% | 62.30% |
| F1 Score | 61.19% | 60.84% |
| ROC-AUC | 83.46% | 83.97% |

These results illustrate the trade-offs between the two models. ANN has slightly higher accuracy and ROC-AUC, while Random Forest has higher recall.

---

## 📂 Project Structure

```text
ModelGuard/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── genai/
│   ├── __init__.py
│   ├── llm_engine.py
│   ├── modelguard_assistant.py
│   ├── incident_investigator.py
│   ├── performance_advisor.py
│   └── modelguard_engine.py
│
├── data/
│   └── simulated/
│
├── modelguard_alerts.csv
├── modelguard_data_quality_report.csv
├── modelguard_ks_report.csv
├── modelguard_psi_report.csv
└── modelguard_monitoring_history.csv
```

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip
- Groq API key (for GenAI features)

### Step 1: Clone the Repository

```bash
git clone https://github.com/vmewara/ModelGuard.git
```

### Step 2: Navigate to the Project

```bash
cd ModelGuard
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Groq API Key

Set your API key as an environment variable.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Do not upload your API key to GitHub.

### Step 5: Run the Application

```bash
streamlit run app.py
```

Open the application in your browser:

http://localhost:8501

---

## 🐳 Docker Deployment

ModelGuard AI is available as a Docker image.

### Pull the Docker Image

```bash
docker pull genai001343/modelguard-ai:latest
```

### Run the Container

```bash
docker run -d \
  --name modelguard-container \
  -p 8501:8501 \
  -e GROQ_API_KEY=your_groq_api_key \
  genai001343/modelguard-ai:latest
```

### Access the Application

http://localhost:8501

### Stop the Container

```bash
docker stop modelguard-container
```

### Restart the Container

```bash
docker start modelguard-container
```

### Docker Hub

https://hub.docker.com/r/genai001343/modelguard-ai

---

## 🧠 GenAI Integration

ModelGuard AI integrates Generative AI through the Groq API.

The GenAI layer connects monitoring information with natural-language explanations.

The application contains three specialized AI capabilities:

1. **ModelGuard Assistant:** General questions and monitoring explanations.
2. **Incident Investigator:** AI-assisted analysis of detected drift findings.
3. **Performance Advisor:** Interpretation of model evaluation metrics and recommendations.

The GenAI functionality is designed to assist interpretation. Its explanations should be validated against the underlying monitoring reports and model evaluation results.

---

## 📈 Monitoring Techniques

**Population Stability Index (PSI)**

Used to quantify changes in feature distributions between reference and current datasets.

**Kolmogorov-Smirnov (KS) Test**

Used to compare distributions and assess whether observed differences are statistically significant.

These techniques provide complementary information and help identify features that may require further investigation.

---

## 🔮 Future Enhancements

- Real-time data ingestion.
- Automated model retraining workflows.
- Advanced concept drift detection.
- Integration with cloud monitoring platforms.
- Model explainability using SHAP.
- Advanced monitoring notifications.
- Historical model performance tracking.
- Enhanced GenAI incident root-cause analysis.

---

## ⚠️ Important Disclaimer

ModelGuard AI is an educational and portfolio demonstration project.

The current monitoring workflow uses simulated data to demonstrate drift detection, monitoring reports, and incident analysis. Simulated drift findings should not be interpreted as confirmed production incidents.

GenAI-generated explanations and recommendations are advisory and require validation.

---

## 👨‍💻 Author

**S. Vishal**

BCA – Data Science  
SRM Institute of Science and Technology, Chennai

- GitHub: https://github.com/vmewara
- LinkedIn: https://linkedin.com/in/s-vishal-12ba04375

---

## ⭐ Project

If you find this project interesting, consider giving the repository a star!

**ModelGuard AI – Making Machine Learning Monitoring More Explainable with Generative AI.**
