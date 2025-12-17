# 🛡️ PhishGuard AI - Phishing Website Detection

**Intelligent Machine Learning System for Cybersecurity Applications**

---

## 📋 Problem Statement

Phishing attacks remain one of the most prevalent cybersecurity threats, targeting individuals and organizations through deceptive websites designed to steal sensitive information. Traditional rule-based detection methods often fail against sophisticated phishing attempts.

**Solution:** This project implements an intelligent machine learning-based system that analyzes website characteristics and features to classify URLs as legitimate or phishing with high accuracy using ensemble and deep learning approaches.

---

## 🎯 Project Overview

PhishGuard AI is an end-to-end machine learning pipeline that combines feature extraction, data preprocessing, and dual model architectures (XGBoost + Artificial Neural Networks) to detect phishing websites with enterprise-grade reliability. The system includes a REST API with an interactive web interface for real-time predictions.

---

## 💻 Tech Stack

<table style="width:100%; border-collapse: collapse; background: #1a1a1a; padding: 20px; border-radius: 8px;">
  <tr style="border-bottom: 2px solid #333;">
    <td style="padding: 15px; text-align: center;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Languages</div>
      <span style="display: inline-block; background: #4a90e2; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🐍 PYTHON</span>
      <span style="display: inline-block; background: #2e5a96; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📊 YAML</span>
    </td>
    <td style="padding: 15px; text-align: center; border-left: 2px solid #333;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Frameworks & Libraries</div>
      <span style="display: inline-block; background: #ff6b35; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">⚡ XGBOOST</span>
      <span style="display: inline-block; background: #ff8c42; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🔬 SCIKIT-LEARN</span>
      <span style="display: inline-block; background: #ffa500; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📈 PANDAS</span>
      <span style="display: inline-block; background: #ff7f50; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🔢 NUMPY</span>
    </td>
    <td style="padding: 15px; text-align: center; border-left: 2px solid #333;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Web & API</div>
      <span style="display: inline-block; background: #00a651; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🌐 FASTAPI</span>
      <span style="display: inline-block; background: #009e3d; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">⚙️ UVICORN</span>
      <span style="display: inline-block; background: #006b35; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🎨 JINJA2</span>
    </td>
  </tr>
  <tr>
    <td style="padding: 15px; text-align: center;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Feature Extraction</div>
      <span style="display: inline-block; background: #e74c3c; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🔍 BEAUTIFULSOUP4</span>
      <span style="display: inline-block; background: #c0392b; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🌐 DNSPYTHON</span>
      <span style="display: inline-block; background: #d63031; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📋 WHOIS</span>
    </td>
    <td style="padding: 15px; text-align: center; border-left: 2px solid #333;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Monitoring & Tracking</div>
      <span style="display: inline-block; background: #9b59b6; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📊 MLFLOW</span>
      <span style="display: inline-block; background: #8e44ad; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📈 MATPLOTLIB</span>
      <span style="display: inline-block; background: #7d3c98; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">📊 SEABORN</span>
    </td>
    <td style="padding: 15px; text-align: center; border-left: 2px solid #333;">
      <div style="font-weight: bold; color: #fff; margin-bottom: 10px; font-size: 14px;">Deep Learning</div>
      <span style="display: inline-block; background: #3498db; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">🧠 ANN/MLP</span>
      <span style="display: inline-block; background: #2980b9; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">⚡ SCIPY</span>
      <span style="display: inline-block; background: #1f618d; color: white; padding: 6px 12px; border-radius: 4px; margin: 4px; font-weight: bold; font-size: 12px;">💾 JOBLIB</span>
    </td>
  </tr>
</table>

---

## 📊 Data & Features

### Dataset Information
- **Source:** Phishing dataset with labeled legitimate and phishing websites
- **Features:** 30 engineered features covering URL, domain, SSL, and page-based indicators
- **Train-Test Split:** 80-20 with random seed (42) for reproducibility

### Feature Categories
- **URL-Level Features:** IP address, URL length, URL shortening services, special characters, domain prefixes/suffixes
- **Security Features:** SSL certificate state, HTTPS token, domain registration length
- **Content Features:** Page rank, favicon, DNS records, web traffic, links structure, embedded content analysis

---

## 🔄 Project Flow

```
Raw Dataset (Phishing CSV)
    ↓
Data Loading & Exploration
    ↓
Data Preprocessing (Handle missing values, encoding)
    ↓
Feature Scaling (StandardScaler)
    ↓
Train-Test Split (80-20)
    ↓
┌─────────────────────────────────────┐
├─→ XGBoost Model Training          ├─→ Model Evaluation & Metrics
├─→ ANN (MLPClassifier) Training    ├─→ Cross-validation
└─────────────────────────────────────┘
    ↓
Model Serialization & Artifact Storage
    ↓
REST API Deployment (FastAPI)
    ↓
Real-time Prediction Interface
```

---

## ⚙️ How It Works

### 1. Feature Extraction
The system extracts 30 features from each URL by analyzing:
- URL structure and composition
- Domain registration details via WHOIS lookups
- SSL certificate validity and state
- DNS records and reputation
- HTML content and page structure

### 2. Data Preprocessing
Features are standardized using StandardScaler to normalize values between -1, 0, and 1, ensuring all features contribute equally to model training.

### 3. Dual Model Architecture

**XGBoost (Extreme Gradient Boosting):**
- Ensemble method that builds decision trees sequentially
- Each tree corrects errors from previous ones
- Configuration: 400 estimators, max_depth=7, learning_rate=0.04, regularization (alpha=0.3, lambda=1.0)

**ANN/MLP (Artificial Neural Network):**
- Feedforward neural network with multiple hidden layers
- Architecture: Input → 128 neurons → 64 neurons → 32 neurons → Binary Output
- Features: Early stopping, ReLU activation, Adam optimizer, batch size 64, max iterations 400

### 4. Model Evaluation
Both models are evaluated using:
- **Accuracy:** Overall correctness of predictions
- **Precision:** Of detected phishing sites, how many are true positives
- **Recall:** Of actual phishing sites, how many are detected
- **F1-Score:** Harmonic mean balancing precision and recall

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Models | Dual (XGBoost + ANN) |
| Features | 30 Engineered |
| Train-Test Split | 80-20 |
| Experiment Tracking | MLflow |

**Model Hyperparameters:**
```
XGBoost:
  - Estimators: 400
  - Max Depth: 7
  - Learning Rate: 0.04
  - Regularization (L1/L2): 0.3/1.0

ANN/MLP:
  - Hidden Layers: [128, 64, 32]
  - Activation: ReLU
  - Solver: Adam
  - Early Stopping: Enabled
```

---

## 📚 Essential Definitions

### Machine Learning Concepts

**Supervised Learning:** Training on labeled data (input-output pairs) to learn patterns and make predictions on new data. Here: classifying URLs as phishing or legitimate.

**Classification:** ML task predicting discrete categories (Phishing or Legitimate) rather than continuous values.

**Feature Engineering:** Creating meaningful features from raw data that help models learn patterns. Here: extracting URL characteristics, domain info, and page structure.

### XGBoost (Extreme Gradient Boosting)

An optimized gradient boosting framework that builds ensemble of decision trees. Each tree corrects residuals from previous trees, creating a powerful combined model.

**Advantages:**
- Fast training and prediction
- Handles non-linear relationships well
- Built-in regularization prevents overfitting
- Robust to outliers in data
- Excellent performance on tabular data

### ANN (Artificial Neural Network) / MLP (Multi-Layer Perceptron)

A feedforward neural network with multiple layers of interconnected neurons. Each neuron performs weighted sum of inputs + bias, followed by activation function. Network learns by adjusting weights through backpropagation.

**How It Works:**
- **Input Layer:** Receives 30 features
- **Hidden Layers:** Learn hierarchical representations (128 → 64 → 32 neurons)
- **Output Layer:** Binary classification (Phishing vs Legitimate)
- **Activation:** ReLU introduces non-linearity
- **Backpropagation:** Adjusts weights to minimize prediction error

### Evaluation Metrics

**Accuracy:** (TP + TN) / Total. Percentage of correct predictions.

**Precision:** TP / (TP + FP). Of all URLs flagged as phishing, how many were actually phishing?

**Recall:** TP / (TP + FN). Of actual phishing URLs, how many were detected?

**F1-Score:** 2 × (Precision × Recall) / (Precision + Recall). Best for imbalanced datasets.

---

## 📁 Project Structure

```
phising-website-detection/
├── src/                          # Core ML pipeline
│   ├── data_loader.py            # Load datasets
│   ├── preprocessor.py           # Data cleaning & scaling
│   ├── train_xgb.py              # XGBoost training
│   ├── train_ann.py              # ANN/MLP training
│   ├── pipeline.py               # Orchestrate workflow
│   ├── website_feature_extraction.py  # 30 features
│   ├── config_loader.py          # Load YAML config
│   └── utils.py                  # Utilities
├── api/                          # FastAPI REST endpoint
│   ├── main.py                   # FastAPI app
│   └── templates/
│       └── index.html            # Web UI
├── inference/
│   └── predictor.py              # Real-time prediction
├── notebook/
│   └── EDA.ipynb                 # Exploratory analysis
├── data/
│   └── phising.csv               # Training dataset
├── artifacts/                    # Trained models
├── config.yaml                   # Hyperparameters
└── requirements.txt              # Dependencies
```

---

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Edit `config.yaml` to adjust model hyperparameters, data paths, and artifact locations.

### 3. Train Models
```bash
python run_pipeline.py
```
This trains both XGBoost and ANN models, evaluates them, saves artifacts, and logs to MLflow.

### 4. Run Web API
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Access at `http://localhost:8000`

### 5. Make Predictions
Use the web interface to enter URL features and get real-time predictions.

---

## 🏆 Key Achievements

- **Dual ML Architecture:** Combines gradient boosting and deep learning for robust predictions
- **30 Feature Engineering:** Comprehensive feature extraction covering URL, domain, security, and content aspects
- **Production-Ready API:** REST API with interactive web interface using FastAPI
- **Experiment Tracking:** MLflow integration for reproducible experiments
- **Configuration Management:** YAML-based configuration for easy hyperparameter tuning
- **Data Preprocessing:** Proper train-test split with StandardScaler normalization
- **Scalable Architecture:** Modular design for easy extension and integration

---

## 🌐 Deployment

The system includes:
- **API Layer:** FastAPI for high-performance REST endpoints
- **Web Interface:** Interactive HTML/CSS UI for predictions
- **Model Serialization:** Joblib for efficient model persistence
- **Real-time Predictions:** Sub-second inference latency

---

## 🔮 Future Enhancements

- Ensemble voting combining both model predictions
- Real-time URL feature extraction from live websites
- Advanced hyperparameter tuning (Bayesian optimization)
- Database integration for tracking predictions
- Docker containerization for easy deployment
- Model explainability with SHAP values
- Continuous learning pipeline with new data

---

## 📌 Conclusion

PhishGuard AI demonstrates a complete ML solution for cybersecurity, combining advanced feature engineering, multiple model architectures, and production-ready deployment.

**Showcases proficiency in:**
- End-to-end ML pipeline development
- Multiple ML algorithms (Gradient Boosting & Deep Learning)
- Feature engineering and data preprocessing
- Model evaluation and hyperparameter tuning
- REST API development and deployment
- Experiment tracking and reproducibility
- Professional code organization and best practices

**Perfect for AI/ML Engineer roles** - demonstrates practical ML engineering skills from model development to production deployment, with focus on security applications and real-world impact.
