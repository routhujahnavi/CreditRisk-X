# CreditRisk-X — Explainable Credit Default Risk Prediction & Threshold Optimization Platform

CreditRisk-X is a production-quality, explainable credit default risk prediction and decision boundary optimization platform based on the **Home Credit Default Risk** dataset. It includes a modular Machine Learning pipeline (preprocessing, stratified cross-validation, modeling, and explanation), a FastAPI backend, and an interactive React-based analytics dashboard.

### 🚀 Live Application
The platform is deployed live and can be accessed at: **[https://creditrisk-x.onrender.com](https://creditrisk-x.onrender.com)**

---

## 1. Architecture Overview

The system is architected in two clean, modular layers:
1. **Machine Learning Layer (`ml/`)**: Handles dataset generation/loading, automated data profiling, leakage-free column preprocessing, model cross-validation, and precomputation of metrics/explanations.
2. **Application Layer (`backend/` & `frontend/`)**: Exposes the models via a FastAPI REST service, validates inputs with Pydantic, and provides a polished, interactive React dashboard for business diagnostic and what-if simulations.

### System Flow Diagram
```
  [User UI Form] ──(Input JSON)──> [FastAPI Backend] ──(Pydantic Validation)──> [Prediction Service]
                                                                                       │
  [Interactive] <──(JSON Response)── [Backend API] <──(Prob + Explanations) <─── [Saved Pipeline]
  [Data Charts]
```

---

## 2. Machine Learning Methodology

### Leakage-Safe Preprocessing
To guarantee academic and mathematical integrity, all data transformations are encapsulated inside a scikit-learn `Pipeline` and `ColumnTransformer`:
* **Numerical Features**: Imputed using column **medians** and standardized using `StandardScaler`.
* **Categorical Features**: Imputed using **most frequent** values and converted using `OneHotEncoder(handle_unknown="ignore")`.

**Anti-Leakage Design:** Standard scaling and imputer parameters are never fit on the full dataset. Instead, they are fit *only* on the training split inside the cross-validation folds. This prevents test set statistical parameters (like mean/variance) from leaking into the training pipeline.

### Stratified Cross-Validation
Because defaults are highly imbalanced, we employ a 5-fold **StratifiedKFold** cross-validation scheme. This ensures that every fold maintains a target ratio (~10.6%) identical to the overall training set, preventing optimistic model performance evaluations.

### Evaluated Models
1. **Logistic Regression (Baseline)**: L2 regularized linear model. It provides clear, linear baseline metrics and direct coefficients for global interpretability.
2. **HistGradientBoosting (Nonlinear)**: Tree-based gradient boosting classifier. Designed to handle missing value structures natively and model non-linear boundaries.

---

## 3. Empirical Results (Actual Experiments)

*Metrics are extracted from `models/pipeline_metadata.json` generated from 10,000 simulated applicants (80/20 train-test split):*

| Model Estimator | CV Mean ROC-AUC | CV Standard Deviation | Test Set ROC-AUC | Test Set PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **0.6710** | **± 0.0159** | **0.6492** | **0.2015** |
| **HistGradientBoosting (Nonlinear)** | 0.6353 | ± 0.0180 | 0.6188 | 0.1783 |

*Note: The linear Logistic Regression slightly outperformed the Gradient Boosting model. This indicates the underlying default risk probabilities contain strong linear log-odds characteristics, showcasing the value of establishing simple, interpretable baselines.*

---

## 4. Threshold Optimization Economics

In credit underwriting, using a standard probability threshold of `0.50` is rarely optimal:
* **Low Thresholds (e.g., 0.20)**: Highly conservative. Minimizes default rate (False Negatives), but denies credit to many creditworthy applicants (False Positives), resulting in lost revenue.
* **High Thresholds (e.g., 0.70)**: Aggressive growth. Approves more applicants, but exposes the bank to high credit write-offs from actual defaults (False Negatives).

CreditRisk-X provides an interactive slider to analyze these trade-offs and evaluate Precision, Recall, F1, and simulated denial rates.

---

## 5. API Documentation

### GET `/api/health`
Checks service health status. Returns: `{"status": "healthy"}`.

### GET `/api/model-info`
Returns general training information, the active/best model key, and list of features.

### GET `/api/metrics?model_key=<model_key>`
Returns fold-wise scores and downsampled coordinates for rendering ROC and PR curves.

### GET `/api/threshold-analysis?model_key=<model_key>`
Returns metrics evaluated across thresholds `0.05` to `0.95`.

### GET `/api/feature-importance?model_key=<model_key>`
Returns coefficients (Logistic Regression) or permutation importance values (Gradient Boosting).

### GET `/api/data-profile`
Returns automated data profiling results (missingness, class balance, continuous distributions).

### POST `/api/predict?model_key=<model_key>&threshold=<threshold>`
Evaluates risk for an applicant, returning probability, risk class, and local explanations.

---

## 6. How to Run

### Prerequisite
Make sure you have Python 3 (with venv) and Node.js (with npm) installed.

### Step 1: Backend Setup & Training
```bash
# 1. Navigate to the project root and create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install requirements
pip install -r backend/requirements.txt

# 3. Generate data, run data profiling, and train/save the models
python3 ml/generate_data.py
python3 ml/data_profiling.py
python3 ml/train.py

# 4. Start the FastAPI backend server (runs on port 8000)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Frontend Setup & Run
```bash
# 1. Open a new terminal tab, navigate to the frontend folder
cd frontend

# 2. Install node modules
npm install

# 3. Start the Vite React development server (runs on port 5173 / 3000)
npm run dev
```
Open `http://localhost:5173` (or the terminal-provided URL) in your browser.

---

## 7. Model Limitations & Fairness

* **Class Imbalance**: Default rates are low. Accuracy is a deceptive metric. Models are trained on balanced cross-validation but evaluated on AUC.
* **Demographic Bias**: The model uses variables like Gender (`CODE_GENDER`) or Family Status (`NAME_FAMILY_STATUS`). In real banking, using these features to make lending decisions is illegal under Fair Lending Acts (such as ECOA in the US) as it perpetuates systemic biases. They are included here strictly for educational comparison.
* **Educational Prediction**: The outputs are scorecards representing mathematical estimations and do not constitute actual credit approvals or legal financial advice.
