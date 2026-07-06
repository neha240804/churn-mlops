# 🚀 Customer Churn Prediction - End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-orange.svg)](https://dvc.org/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success.svg)](https://github.com/features/actions)
[![Render](https://img.shields.io/badge/Backend-Render-46E3B7.svg)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

An end-to-end **MLOps project** that predicts customer churn using Machine Learning and modern MLOps practices. The project includes data preprocessing, model training, experiment tracking, REST API development, CI/CD, Docker containerization, data versioning, monitoring, and cloud deployment.

---

## 🌐 Live Demo

### 🖥️ Streamlit Application
🔗 https://churn-mlops-e7t2pa9qxg7i7224uuinbb.streamlit.app/

### ⚡ FastAPI Backend
🔗 https://churn-mlops-7tul.onrender.com

### 📖 Swagger API Documentation
🔗 https://churn-mlops-7tul.onrender.com/docs

---

# 📌 Problem Statement

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers is significantly more expensive than acquiring new ones.

This project predicts whether a telecom customer is likely to churn based on customer demographics, account information, and subscribed services, enabling businesses to take proactive retention actions.

---

# ⭐ Features

- End-to-End MLOps Pipeline
- Automated Data Preprocessing
- Multiple Machine Learning Models
- Hyperparameter Tuning using GridSearchCV
- MLflow Experiment Tracking
- FastAPI REST API
- Streamlit Interactive Web Application
- Docker Containerization
- DVC Data Versioning
- GitHub Actions CI Pipeline
- Evidently AI Data Drift Monitoring
- Cloud Deployment on Render & Streamlit Community Cloud

---

# 🏗️ System Architecture

```text
                     User
                       │
                       ▼
         Streamlit Web Application
                       │
                 REST API Request
                       │
                       ▼
              FastAPI Backend
                       │
        Load Trained XGBoost Model
                       │
                       ▼
              Customer Prediction
                       │
                       ▼
             Prediction Response
```

---

# 🛠️ Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

## MLOps

- MLflow
- FastAPI
- Docker
- DVC
- GitHub Actions
- Evidently AI

## Deployment

- Render
- Streamlit Community Cloud

---

# 📂 Project Structure

```text
churn-mlops/

├── api/
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── config/
│
├── data/
│
├── monitoring/
│
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   ├── logger.py
│   └── exceptions.py
│
├── tests/
│
├── streamlit_app.py
├── Dockerfile
├── pipeline.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Workflow

```text
Dataset
   │
   ▼
Data Validation
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train Multiple Models
   │
   ▼
GridSearchCV
   │
   ▼
MLflow Tracking
   │
   ▼
Best Model Selection
   │
   ▼
FastAPI REST API
   │
   ▼
Docker
   │
   ▼
Render Deployment
   │
   ▼
Streamlit Frontend
```

---

# 🤖 Models Trained

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (Hyperparameter Tuned)

Hyperparameter tuning was performed using **GridSearchCV**.

---

# 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|--------:|
| Logistic Regression | 73.81% | 50.43% | **78.34%** | **61.36%** | 84.13% |
| Random Forest | 79.99% | 65.65% | 51.60% | 57.78% | 83.85% |
| Gradient Boosting | 80.62% | 67.35% | 52.41% | 58.95% | 84.34% |
| XGBoost | **80.91%** | **67.92%** | 53.21% | 59.67% | **84.51%** |

---

# 📈 Experiment Tracking

MLflow was used for:

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Model Artifact Management

---

# 🌐 REST API

FastAPI provides REST endpoints for real-time customer churn prediction.

### Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Home |
| `/health` | Health Check |
| `/docs` | Swagger Documentation |
| `/predict` | Customer Churn Prediction |

---

# 🐳 Docker

The application is containerized using Docker for consistent deployment across different environments.

---

# 📦 Data Versioning

DVC is used to version datasets and ensure reproducible machine learning experiments.

---

# 🔄 CI/CD

GitHub Actions automatically:

- Installs dependencies
- Runs Unit Tests
- Validates the project on every push and pull request

---

# 📉 Data Drift Monitoring

Evidently AI is used to monitor feature drift and generate automated drift reports for incoming data.

---

# 🚀 Deployment

Frontend deployed using **Streamlit Community Cloud**

Backend API deployed using **Render**

---

# 🧪 Unit Testing

Implemented unit tests for:

- Configuration Loading
- Data Ingestion
- Model Artifact Validation

---

# 🔮 Future Improvements

- MLflow Model Registry
- Automated Retraining Pipeline
- Kubernetes Deployment
- Monitoring Dashboard
- Feature Store Integration
- Authentication & User Management

---

# 👩‍💻 Author

**Neha Chaudhary**

Electrical Engineering Undergraduate

Machine Learning • Deep Learning • MLOps • Generative AI

---

⭐ If you found this project helpful, consider giving it a star!
