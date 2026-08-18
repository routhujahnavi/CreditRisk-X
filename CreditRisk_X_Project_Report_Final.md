# An Internship Course Report
# On
# CreditRisk-X: An Explainable and Threshold-Aware Machine Learning Pipeline for Credit Default Risk Assessment

**Anil Neerukonda Institute of Technology and Sciences**  
*(Affiliated to Andhra University) Sangivalasa, Visakhapatnam – 531162*  

**SUBMITTED BY**  
*   **Name:** Routhu Jahnavi Santoshi  
*   **Roll Number:** A23126510113  
*   **IBM Student ID:** IBMQ2DST2091  
*   **Department:** Computer Science and Engineering  
*   **UG - Level:** 3  
*   **Specialization:** Artificial Intelligence & Machine Learning  
*   **Company Name:** IBM  

---

## Abstract
In retail banking, assessing credit risk is a foundational challenge. Traditional credit scoring systems rely on static rules or black-box predictions that fail to explain the drivers of default or optimize classification thresholds under asymmetric business costs (where a False Negative, or default, costs significantly more than a False Positive, or rejected compliant customer). **CreditRisk-X** addresses this dual challenge by building an end-to-end, explainable, and threshold-aware machine learning application based on simulated applicant profiles mimicking the **Home Credit Default Risk** dataset.

The system implements a modular data generation engine producing 10,000 records containing 16 numerical and 10 categorical variables. To ensure mathematical and academic integrity, a leakage-safe preprocessing pipeline is engineered: missing values are imputed, categorical variables are one-hot encoded, and numerical features are standardized using parameters fit strictly inside a Stratified 5-Fold Cross-Validation loop. The system trains and compares a baseline Logistic Regression model and a non-linear HistGradientBoosting classifier. Logistic Regression emerged as the superior estimator, achieving a mean cross-validation ROC-AUC of `0.6710 ± 0.0159` and a test ROC-AUC of `0.6492` (compared to `0.6353 ± 0.0180` and `0.6188` for the boosting model), implying a strong linear correlation between simulated log-odds and default indicators.

To render predictions actionable, a local prediction engine generates SHAP-like directional feature contribution explanations in real time. These are combined with an interactive threshold optimization module. A full-stack web application was built around this pipeline: a FastAPI backend serves the prediction and analytics routes, while a custom, dark-themed React dashboard (Vite, Tailwind-inspired Vanilla CSS, SVG data charts) enables users to profile dataset DNA, evaluate classification thresholds, run what-if simulations, and query local explainability. The completed platform is containerized and deployed publicly, demonstrating how financial institutions can bridge the gap between model training, business utility optimization, and explainable AI.

---

## Table of Contents
1.  **Introduction**
    *   1.1 About the Internship Program
    *   1.2 Objectives of the Project
    *   1.3 Scope of the Project
2.  **Company / Organization Profile**
    *   2.1 About the Organization
    *   2.2 History
    *   2.3 Vision and Mission
    *   2.4 Products and Services
    *   2.5 Organizational Structure
    *   2.6 Department / Domain Worked In
    *   2.7 Relevance of the Domain to the Project
3.  **Tools and Technologies Used**
    *   3.1 Frontend
    *   3.2 Backend
    *   3.3 Machine Learning Engine
    *   3.4 Deployment and Tooling
    *   3.5 Why These Tools Were Chosen
    *   3.6 Development Environment
4.  **Project Details**
    *   4.1 Problem Statement
    *   4.2 System Design and Architecture
    *   4.3 Methodology
    *   4.4 Implementation
    *   4.5 Screenshots
    *   4.6 Results
    *   4.7 Appendix: Sample API Input and Output
5.  **Learning Outcomes**
    *   5.1 Technical Skills Gained
    *   5.2 Soft Skills Gained
6.  **Conclusion**
7.  **Future Scope**
8.  **References**
9.  **Student & Coordinator Details**

---

## 1. Introduction

### 1.1 About the Internship Program
The internship/project programme provides students with an opportunity to gain practical, hands-on exposure to industry-oriented technologies and to apply concepts learned during academic study to a real, self-directed problem. It is designed to bridge the gap between classroom theory and practical implementation by giving students experience in end-to-end project development, problem-solving, independent research, and the use of modern software tools and technologies.

As part of this programme, the project **"CreditRisk-X: An Explainable and Threshold-Aware Machine Learning Pipeline for Credit Default Risk Assessment"** was undertaken to explore the practical application of machine learning to financial credit underwriting. The work involved every stage of a modern applied-ML project, from mock dataset generation and profiling to leakage-safe preprocessing pipelines, model selection, local explanation generation, backend API development, frontend interface development, integration testing, and containerized deployment.

The programme also provided practical experience with a wide range of tools and technologies, including Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, React, Vite, Vanilla CSS, SVG-based charting, Git, GitHub, and Render. The completed project helped develop both technical and professional skills, including designing production-quality software, documenting technical work, and preparing a full-stack machine learning application for real-world hosting.

### 1.2 Objectives of the Project
The specific objectives of the CreditRisk-X project are:
*   To design a dataset generator that simulates 10,000 credit applicant profiles with realistic demographics, asset holdings, income levels, and correlations that mimic the Home Credit Default Risk dataset.
*   To build an automated profiling module that reports data quality, targets class distributions, and analyzes missingness patterns.
*   To implement a leakage-safe preprocessing pipeline using scikit-learn's `ColumnTransformer` to isolate feature imputation, scaling, and categorical encoding within training folds.
*   To evaluate and compare baseline Logistic Regression with a non-linear HistGradientBoosting classifier using Stratified 5-Fold Cross-Validation.
*   To develop a prediction service in FastAPI that returns default probabilities alongside SHAP-like directional feature contributions explaining local predictions.
*   To build an interactive threshold analysis module that calculates business utility (e.g., net revenue, savings, loss prevention) based on adjustable false negative/false positive cost weights.
*   To build a premium, dark-mode React frontend serving custom SVG-based interactive charts (ROC curves, Precision-Recall curves, Threshold trade-off lines, and Feature Contribution charts).
*   To deploy the integrated full-stack application to a cloud hosting platform for public validation.

### 1.3 Scope of the Project
The scope of CreditRisk-X covers the design and development of a complete explainable risk assessment and decision boundary optimization platform, delivered through a deployed full-stack web application. The major areas covered within the scope of the project are described below:

#### Dataset Generation & Profiling
Ingestion of high-fidelity synthetic applicant records. Profiling covers row/column counts, missing cell percentages, duplicated records, target class imbalance ratios, and numerical-to-target correlations.

#### Preprocessing & Model Selection
A reusable scikit-learn preprocessing pipeline. Model selection covers regularized Logistic Regression (L2 penalty) and HistGradientBoosting classifiers, evaluated on ROC-AUC, PR-AUC, and Confusion Matrices.

#### Interactive Decision Engine
Interactive threshold optimization, allowing credit managers to slide probability thresholds and immediately see the impact on Precision, Recall, F1-score, and net business utility (net financial savings).

#### Explainability & Local Explanations
Directional feature contributions that identify which specific parameters (e.g., external credit ratings, income, age) pushed the default probability above or below the baseline.

#### Web Application & Deployment
A FastAPI backend serving API endpoints, mounted with static files of the React/Vite production build, deployed as a single-port service on Render.

---

## 2. Company / Organization Profile

### 2.1 About the Organization
International Business Machines Corporation (IBM) is a global technology and consulting organization headquartered in Armonk, New York. IBM provides technology products, software, infrastructure, consulting, and professional services to organizations across different industries. Its major areas of focus include artificial intelligence, hybrid cloud, enterprise software, IT infrastructure, cybersecurity, data, and business consulting.

IBM has a global presence and works with organizations to modernize their technology infrastructure, manage and analyze data, adopt cloud technologies, and develop AI-based solutions. The company also provides learning and internship opportunities that allow students to gain practical exposure to areas such as software development, artificial intelligence, machine learning, data science, cloud computing, and consulting.

The internship / project programme provides an opportunity to apply technical knowledge to an industry-oriented problem. The project undertaken, "CreditRisk-X: An Explainable and Threshold-Aware Machine Learning Pipeline for Credit Default Risk Assessment," focuses on applying predictive modeling and explainable AI techniques to automated credit risk underwriting.

### 2.2 History
IBM has a long history in the development of computing and information technology. The organization traces its origins to 1911 as the Computing-Tabulating-Recording Company (C-T-R), and was renamed International Business Machines Corporation in 1924. Over the following decades, IBM contributed to major developments in computing, including the IBM System/360 family of mainframe computers and the IBM Personal Computer, which helped establish personal computing as a mainstream technology. In artificial intelligence, IBM gained international recognition through its Deep Blue chess-playing system in the 1990s, and more recently through its Watson and watsonx AI platforms.

### 2.3 Vision and Mission
IBM's vision centers on technological innovation and the practical application of advanced computing to solve complex business and societal challenges. Its mission emphasizes “innovation that matters” — helping organizations use trusted, secure, and responsible technology to improve their operations — closely tied to its work in artificial intelligence, hybrid cloud, enterprise software, infrastructure, cybersecurity, and consulting.

### 2.4 Products and Services
IBM's offerings are broadly grouped into the following categories:
*   **Software:** Enterprise software and AI platforms such as watsonx, along with data, automation, and cybersecurity solutions that help organizations manage data and build AI applications.
*   **Infrastructure:** Enterprise infrastructure technologies such as IBM Z mainframes and IBM Power systems, built for organizations requiring reliable, high-performance computing.
*   **Consulting and Services:** Digital transformation, AI adoption, cloud migration, cybersecurity, and data-analytics consulting that helps organizations implement suitable technology solutions.

### 2.5 Organizational Structure
IBM operates through business-focused divisions — principally Software, Consulting, Infrastructure, and Financing — spanning multiple geographic regions. Research and development activity supports innovation in artificial intelligence, cloud computing, and cybersecurity, with an organizational culture that emphasizes cross-functional collaboration between technical and business teams.

### 2.6 Department / Domain Worked In
The project domain is Machine Learning, Artificial Intelligence, and Applied Data Science, specifically the sub-field of predictive credit underwriting, model evaluation, and Explainable AI (XAI). Within this domain, the project involved data generation, automated dataset profiling, leakage-free cross-validation modeling, and full-stack integration of the resulting underwriting engine into a deployable web application.

### 2.7 Relevance of the Domain to the Project
Working within the Machine Learning and AI domain gave direct exposure to challenges that mirror real financial industry problems: building robust model pipelines, preventing data leakage during preprocessing, optimizing thresholds under highly asymmetric business costs, and providing transparent predictions. These align directly with real-world credit risk underwriting where compliance (e.g., Fair Lending Acts) mandates explanation of loan denials and business viability demands strict loss-minimization.

---

## 3. Tools and Technologies Used

### 3.1 Frontend
*   **React (Vite):** A fast, component-based user interface framework for rapid rendering and state management.
*   **Vanilla CSS:** Curated custom stylesheet representing a sleek dark-mode theme utilizing glassmorphism, responsive grid systems, and smooth transitions.
*   **Custom SVG-based Charting:** Custom inline SVG implementations to draw ROC curves, Precision-Recall curves, and local explanation bar charts, ensuring fast render times without loading heavy external canvas libraries.

### 3.2 Backend
*   **Python 3.14:** The primary programming language for dataset preprocessing, model evaluation, and backend scripting.
*   **FastAPI:** A high-performance, asynchronous REST framework that automatically generates interactive OpenAPI documentation (Swagger UI).
*   **Uvicorn:** A lightning-fast ASGI web server implementation used to host the FastAPI application.

### 3.3 Machine Learning Engine
*   **pandas & NumPy:** Used to generate, load, manipulate, and compute statistical profiles of applicant tabular structures.
*   **scikit-learn:** Provides preprocessing transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`), model estimators (`LogisticRegression`, `HistGradientBoostingClassifier`), validation generators (`StratifiedKFold`), and metrics packages.
*   **joblib:** Used to persist trained estimators and preprocessing pipelines to disk as lightweight binary objects.

### 3.4 Deployment and Tooling
*   **Git & GitHub:** Used for source code version control and remote synchronization.
*   **Render:** The hosting platform used for compiling static React assets and running the FastAPI web server.

### 3.5 Why These Tools Were Chosen
FastAPI was selected over Flask because it handles asynchronous requests natively, provides automatic Pydantic request body validation, and generates interactive Swagger endpoints out of the box. React with Vite was selected for the frontend to enable rapid iteration, fast page reloading, and structured component modularity. Custom SVG charts were implemented rather than importing heavy charting libraries (like Chart.js or Recharts) to provide clean visualizations, precise coordinate mapping, and full CSS hover animations with zero bundle dependencies. Scikit-learn was chosen because its unified API makes wrapping transformers and models inside a pipeline exceptionally clean.

### 3.6 Development Environment
*   **Operating System:** macOS (Local development environment).
*   **Backend Environment:** Python virtual environment (`venv`) for local package isolation.
*   **Frontend Environment:** Node.js runtime environment utilizing npm for building production assets (`npm run build`).
*   **Deployment Configuration:** Single-port service where FastAPI mounts the compiled React frontend static directory (`frontend/dist`) at the root `/` and serves API requests at `/api`.

---

## 4. Project Details

### 4.1 Problem Statement
In automated credit underwriting, models must determine whether an applicant will default on a loan. This task is complicated by three major issues:
1.  **Class Imbalance:** In typical credit datasets, defaults represent a small minority of cases (e.g., ~10%). Standard classifiers trained on imbalanced data tend to predict the majority class (non-default) to maximize accuracy, leading to high default rates.
2.  **Asymmetric Costs:** Rejecting a good customer (False Positive) leads to a small loss in potential interest revenue. Accepting a defaulting customer (False Negative) leads to a massive loss (the unpaid principal). Standard machine learning models default to a probability threshold of 0.5, which is rarely optimal. The threshold must be tuned based on business utility weights.
3.  **Lack of Transparency:** Modern machine learning estimators are complex. Credit compliance regulations require banks to justify why a loan was denied (e.g., "debt ratio is too high"). Black-box models are unacceptable; local decisions must be accompanied by plain-text explanations.

### 4.2 System Design and Architecture
CreditRisk-X uses a modular architecture separating model generation, data validation, REST APIs, and client-side visualization:

```
  [User Input Form] ──(Pydantic JSON)──> [FastAPI Backend] ──> [Prediction Service]
                                                                        │
  [Interactive UI] <──(200 OK + Expl) ── [JSON Response] <── (Local Explainer + Model)
```

The system comprises three main blocks:
*   **Preprocessing Pipeline:** Encapsulates standard scaling, median numerical imputation, and one-hot encoding into a scikit-learn `Pipeline`.
*   **FastAPI Routing:** Exposes endpoints to profile datasets, query model evaluation curves (ROC/PR), simulate threshold metrics, and post applicant variables.
*   **React Client:** Visualizes cumulative metrics and handles what-if variables (sliding thresholds, changing business costs) in an interactive dashboard.

### 4.3 Methodology

#### High-Fidelity Data Generation
A dataset of 10,000 credit applicants is generated. Columns represent real Home Credit Default Risk variables (e.g., `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_BIRTH`, and three external source ratings `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). Correlations are engineered such that lower external ratings, higher debt ratios (`AMT_ANNUITY` / `AMT_INCOME_TOTAL`), and younger ages (`DAYS_BIRTH`) lead to a higher probability of default.

#### Leakage-Safe Cross-Validation
To prevent data leakage, training parameters are never fit on the full dataset:
*   The data is split into Stratified 5-Fold partitions.
*   For each fold, the `ColumnTransformer` fits its imputer and scaler *only* on the training fold, transforming the test fold.
*   This ensures that the test folds remain completely unseen during pipeline fitment, guaranteeing academic validity.

#### Model Comparison
Two pipelines are trained:
*   **Logistic Regression (Baseline):** Regularized L2 solver.
*   **HistGradientBoosting (Nonlinear):** A tree-based ensemble designed for tabular numerical and categorical features.

#### Local Predictor & Explainer
For local predictions, a directional feature contribution engine is built:
*   For Logistic Regression, feature contributions are computed as:
    $$\text{Contribution}_j = w_j \cdot (x_j - \mu_j) / \sigma_j$$
    where $w_j$ is the model coefficient, $x_j$ is the standardized input value, and $\mu_j, \sigma_j$ are the scaler's historical parameters.
*   This represents a local linear SHAP approximation: positive contributions increase the risk of default (rendered in red), while negative contributions decrease the risk (rendered in green).

#### Business Utility Optimization
The business utility metric is computed by assigning financial weights:
*   **True Positive (TP - Correctly Rejected Default):** Prevents a credit loss. Savings = $\text{Cost of Default} \times \text{TP}$.
*   **False Positive (FP - Wrongly Rejected Compliant Client):** Loses potential profit. Cost = $\text{Cost of Missed Opportunity} \times \text{FP}$.
*   **False Negative (FN - Wrongly Accepted Default):** Incurs a default loss. Cost = $\text{Cost of Default} \times \text{FN}$.
*   **True Negative (TN - Correctly Accepted Client):** Earns profit. Revenue = $\text{Opportunity Revenue} \times \text{TN}$.
*   The platform computes the net savings curve across thresholds to identify the mathematically optimal decision boundary.

### 4.4 Implementation
The FastAPI backend exposes the following REST endpoints:
*   `GET /api/health`: Simple status check.
*   `GET /api/model-info`: Returns model summaries, validation scores, and list of features.
*   `GET /api/data-profile`: Returns automated dataset summary, imbalance ratios, and missingness metrics.
*   `GET /api/metrics`: Serves coordinates for ROC and Precision-Recall curves.
*   `GET /api/threshold-analysis`: Calculates metrics (Precision, Recall, F1, TP, FP, TN, FN) across a sequence of thresholds from `0.0` to `1.0`.
*   `POST /api/predict`: Validates inputs with Pydantic, calculates probability of default, and returns directional explanations.

### 4.5 Screenshots
*(Layout placeholders to be populated with web application captures)*

#### Screenshot 1: Platform Ingestion & Data Quality Profiler
`[Placeholder: Data Ingestion Tab showing row count, target imbalance (10.64%), and missingness percentages]`

#### Screenshot 2: Model Performance Dashboard
`[Placeholder: Performance Tab showing ROC Curve, PR Curve, and default metrics comparisons between Logistic Regression and Gradient Boosting]`

#### Screenshot 3: Business Threshold Optimizer
`[Placeholder: Threshold Tab showing interactive sliders for False Negative/False Positive weights, the Net Savings curve, and the optimal threshold line]`

#### Screenshot 4: Local Underwriting Simulation & Explainer
`[Placeholder: Underwriting Tab showing application input forms, predicted risk gauge (e.g., 24.5%), and the diverging green/red local feature contribution bar chart]`

### 4.6 Results

#### Model Performance Comparison
*   **Logistic Regression (Baseline):**
    *   Mean CV ROC-AUC: `0.6710 ± 0.0159`
    *   Test ROC-AUC: `0.6492`
    *   Test PR-AUC: `0.2015`
*   **HistGradientBoosting (Nonlinear):**
    *   Mean CV ROC-AUC: `0.6353 ± 0.0180`
    *   Test ROC-AUC: `0.6188`
    *   Test PR-AUC: `0.1758`

*Analysis:* The linear baseline model outperformed the gradient booster. This indicates that the risk log-odds in the simulated dataset have a strong linear alignment, confirming the utility of regularized linear models on tabular data where feature signals are additive.

#### Threshold Optimization Analysis
Using a standard threshold of `0.5`, the baseline model yields:
*   Precision: `1.0000`, Recall: `0.0047`, F1-score: `0.0093`
*   Confusion Matrix: TP: `1`, FP: `0`, TN: `1787`, FN: `212`

Because the dataset is imbalanced and the model was evaluated under default probability, a threshold of `0.5` causes the model to classify almost all applicants as non-default, which leads to high defaults (212 defaults accepted). 

By using the **Threshold Optimizer** with a Default Cost of $10,000 and an Opportunity Cost of $1,000, the system calculates the optimal threshold to be **`0.108`**. At this threshold:
*   Recall increases to **`68%`** (detecting 144 of the 213 defaults).
*   Net financial savings are maximized by minimizing the expensive False Negatives, demonstrating the practical value of threshold-aware classifiers.

### 4.7 Appendix: Sample API Input and Output

#### Sample Input JSON (`POST /api/predict`):
```json
{
  "NAME_CONTRACT_TYPE": "Cash loans",
  "CODE_GENDER": "F",
  "FLAG_OWN_CAR": "Y",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 0,
  "AMT_INCOME_TOTAL": 180000.0,
  "AMT_CREDIT": 500000.0,
  "AMT_ANNUITY": 25000.0,
  "AMT_GOODS_PRICE": 450000.0,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_HOUSING_TYPE": "House / apartment",
  "DAYS_BIRTH": -15000,
  "DAYS_EMPLOYED": -2000,
  "DAYS_REGISTRATION": -1000,
  "DAYS_ID_PUBLISH": -3000,
  "CNT_FAM_MEMBERS": 2,
  "REGION_RATING_CLIENT": 2,
  "OCCUPATION_TYPE": "Managers",
  "ORGANIZATION_TYPE": "Business Entity Type 3",
  "EXT_SOURCE_1": 0.55,
  "EXT_SOURCE_2": 0.62,
  "EXT_SOURCE_3": 0.48,
  "DAYS_LAST_PHONE_CHANGE": -100,
  "AMT_REQ_CREDIT_BUREAU_YEAR": 1
}
```

#### Sample Output JSON:
```json
{
  "model_key": "logistic_regression",
  "probability": 0.0824,
  "threshold": 0.108,
  "prediction": 0,
  "decision": "Approved",
  "explanations": [
    {"feature": "EXT_SOURCE_2", "value": 0.62, "contribution": -0.412, "effect": "Decreases default risk"},
    {"feature": "EXT_SOURCE_1", "value": 0.55, "contribution": -0.285, "effect": "Decreases default risk"},
    {"feature": "DAYS_BIRTH", "value": -15000, "contribution": 0.184, "effect": "Increases default risk"},
    {"feature": "AMT_ANNUITY", "value": 25000.0, "contribution": 0.115, "effect": "Increases default risk"}
  ]
}
```

---

## 5. Learning Outcomes

### 5.1 Technical Skills Gained
*   **Asymmetric Cost Modeling:** Learned to apply decision theory to classifier outputs, optimizing probability thresholds based on financial utility instead of default mathematical boundaries.
*   **Leakage-Free Validation Engineering:** Mastered the structure of scikit-learn transformers and pipelines to enforce strict statistical isolation between training and testing folds inside cross-validation loops.
*   **Explainable AI (XAI):** Implemented local linear feature attribution algorithms to extract and render directional contributions, transforming black-box numbers into plain-text rationales.
*   **Unified Full-Stack Deployment:** Configured a single-port deployment architecture by mounting React's static production builds directly inside FastAPI's static middleware.
*   **Custom SVG Canvas Rendering:** Gained experience mapping data coordinates directly to SVG paths and rectangles in React, bypassing heavy external charting libraries.

### 5.2 Soft Skills Gained
*   **Translating Tech to Business:** Developed the ability to communicate machine learning performance metrics (AUC, Precision, Recall) in terms of business impact (profit, losses, savings).
*   **Modular Architecture Design:** Learned to break down a complex, ambiguous problem into clean, isolated components (generators, profilers, predict services, routers, interfaces) before coding.
*   **Technical Writing:** Refined the ability to document methodologies, mathematical computations, and engineering workflows in a clear, academic project report format.

---

## 6. Conclusion
The CreditRisk-X project successfully demonstrates an explainable and threshold-aware machine learning pipeline for credit default risk assessment. By generating a tabular dataset of 10,000 applicant profiles, implementing a leakage-safe preprocessing pipeline, and training Logistic Regression and HistGradientBoosting classifiers, the platform establishes a mathematically sound framework for automated underwriting. Logistic Regression achieved a superior cross-validated ROC-AUC of `0.6710 ± 0.0159`, which was successfully served via a FastAPI backend.

Beyond predicting raw probability, the system implements a local explanation engine that identifies key risk drivers and an interactive threshold optimizer that maximizes net financial savings under asymmetric costs. Delivered through a sleek React web dashboard and hosted publicly on Render, the project demonstrates how applied data science can bridge the gap between model training, business decision-making, and transparency.

---

## 7. Future Scope
*   **Real-time SHAP Integration:** Integrate the official `shap` library to calculate Shapley values for the non-linear HistGradientBoosting classifier, accommodating feature interactions.
*   **Challenger Models:** Incorporate advanced gradient boosting libraries like LightGBM, XGBoost, and CatBoost, which natively handle high cardinality and missing values.
*   **Dynamic Data Ingestion:** Enable connection to live databases or CSV uploads, allowing underwriters to profile and run predictions on real datasets.
*   **Interactive Counterfactual Explanations:** Build what-if features enabling loan officers to see which applicant variables (e.g., lower credit request amount) would change a "Denied" decision to "Approved."
*   **Underwriting Audit Log:** Add a database backend (e.g., SQLite or PostgreSQL) to archive underwriting decisions and explanations for regulatory audits.

---

## 8. References
*   *IBM:* Artificial Intelligence and Machine Learning, IBM Official Documentation and Resources.
*   *Pedregosa, F. et al.:* Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, Vol. 12, pp. 2825–2830, 2011.
*   *Lundberg, S. M. & Lee, S.-I.:* A Unified Approach to Interpreting Model Predictions, Advances in Neural Information Processing Systems (NeurIPS), 2017.
*   *FastAPI Documentation:* FastAPI Framework for Building APIs with Python.
*   *React Documentation:* React – A JavaScript Library for Building User Interfaces.
*   *GitHub Repository:* [https://github.com/routhujahnavi/CreditRisk-X](https://github.com/routhujahnavi/CreditRisk-X)
*   *Live Application:* [https://creditrisk-x.onrender.com](https://creditrisk-x.onrender.com)

---

## 9. Student & Coordinator Details

### Student Profile
| Field | Value |
| :--- | :--- |
| **NAME** | Routhu Jahnavi Santoshi |
| **ROLL NUMBER** | A23126510113 |
| **IBM STUDENT ID** | IBMQ2DST2091 |
| **DEPARTMENT** | Computer Science and Engineering |
| **UG - LEVEL** | 3 |
| **COMPANY NAME** | IBM |
| **SPECIALIZATION** | Artificial Intelligence & Machine Learning |

### Review Committee
| Role | Coordinator Name & Department |
| :--- | :--- |
| **Summer Internship Coordinator** | **Mrs. G. Pranitha** (Assistant Professor) |
| **Head of the Department** | **Dr. G. Srinivas** (Computer Science and Engineering) |

### API Route Configuration
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/api/health` | Returns backend service status |
| **GET** | `/api/model-info` | Returns model performance summaries and input columns |
| **GET** | `/api/data-profile` | Returns dataset missingness, shape, and correlations |
| **GET** | `/api/metrics` | Serves ROC and PR curve points |
| **GET** | `/api/threshold-analysis` | Computes prediction metrics across 100 probability thresholds |
| **POST** | `/api/predict` | Validates applicant JSON, runs inference, and returns local explanations |

### Meta-Feature and Data Variables
| Column Name | Data Type | Feature Type | Description |
| :--- | :--- | :--- | :--- |
| `SK_ID_CURR` | Integer | ID | Unique identifier for the applicant |
| `TARGET` | Integer | Binary Label | default indicator (1 = default, 0 = compliant) |
| `NAME_CONTRACT_TYPE` | String | Categorical | Cash loans vs Revolving loans |
| `CODE_GENDER` | String | Categorical | Gender of the applicant |
| `FLAG_OWN_CAR` | String | Categorical | Car ownership flag (Y/N) |
| `FLAG_OWN_REALTY` | String | Categorical | Realty ownership flag (Y/N) |
| `CNT_CHILDREN` | Integer | Numerical | Number of children |
| `AMT_INCOME_TOTAL` | Float | Numerical | Annual income of the applicant |
| `AMT_CREDIT` | Float | Numerical | Loan credit amount |
| `AMT_ANNUITY` | Float | Numerical | Monthly loan payment annuity |
| `AMT_GOODS_PRICE` | Float | Numerical | Purchase price of the goods for loan |
| `NAME_INCOME_TYPE` | String | Categorical | Working, Commercial, Pensioner, State servant |
| `NAME_EDUCATION_TYPE`| String | Categorical | Secondary, Higher, Incomplete higher, etc. |
| `NAME_FAMILY_STATUS` | String | Categorical | Married, Single, Civil marriage, Separated, Widow |
| `NAME_HOUSING_TYPE` | String | Categorical | House/apartment, With parents, Rented, etc. |
| `DAYS_BIRTH` | Integer | Numerical | Age of client in days (negative value) |
| `DAYS_EMPLOYED` | Integer | Numerical | Employment duration in days (negative value) |
| `DAYS_REGISTRATION` | Integer | Numerical | Days since registration change |
| `DAYS_ID_PUBLISH` | Integer | Numerical | Days since ID document change |
| `CNT_FAM_MEMBERS` | Integer | Numerical | Number of family members |
| `REGION_RATING_CLIENT`| Integer | Numerical | Regional rating of client (1, 2, 3) |
| `OCCUPATION_TYPE` | String | Categorical | Job role (Laborer, Manager, Driver, Accountant, etc.)|
| `ORGANIZATION_TYPE` | String | Categorical | Employer business sector type |
| `EXT_SOURCE_1` | Float | Numerical | Credit rating external rating source 1 |
| `EXT_SOURCE_2` | Float | Numerical | Credit rating external rating source 2 |
| `EXT_SOURCE_3` | Float | Numerical | Credit rating external rating source 3 |
| `DAYS_LAST_PHONE_CHANGE`| Integer | Numerical | Days since last mobile change |
| `AMT_REQ_CREDIT_BUREAU_YEAR`| Integer | Numerical | Enquiries to Credit Bureau in last year |
