import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in python path
sys.path.insert(0, '/Users/rjahnavisantoshi/Desktop/CreditRisk')

import joblib
import pandas as pd
import numpy as np
from ml.preprocessing import get_preprocessor
from backend.app.services.prediction import PredictionService
from backend.app.main import app

client = TestClient(app)

@pytest.fixture
def sample_raw_applicant():
    """Mock raw input from frontend (Pydantic schema format)."""
    return {
        "age_years": 32.0,
        "gender": "F",
        "children_count": 2,
        "family_members_count": 4,
        "family_status": "Married",
        "education_type": "Higher education",
        "housing_type": "House / apartment",
        "income_total": 150000.0,
        "credit_amount": 350000.0,
        "annuity_amount": 18000.0,
        "goods_price": 300000.0,
        "employment_status": "Employed",
        "employment_duration_years": 4.5,
        "income_type": "Working",
        "occupation_type": "Core staff",
        "organization_type": "Business Entity Type 3",
        "owns_car": "Y",
        "owns_realty": "Y",
        "years_registration_change": 3.5,
        "years_id_publish": 1.2,
        "years_phone_change": 0.8,
        "credit_bureau_queries_year": 0,
        "region_rating": 2,
        "ext_source_1": 0.6,
        "ext_source_2": 0.7,
        "ext_source_3": 0.5
    }

def test_preprocessor_configuration():
    """Verify preprocessing pipeline has correct steps."""
    numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT']
    categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR']
    preprocessor = get_preprocessor(numerical_cols, categorical_cols, use_scaling=True)
    
    assert preprocessor is not None
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert 'num' in transformer_names
    assert 'cat' in transformer_names
    
    # Check scaling is present in numerical pipeline
    num_tuple = next(t for t in preprocessor.transformers if t[0] == 'num')
    num_pipe = num_tuple[1]
    assert 'scaler' in num_pipe.named_steps
    assert 'imputer' in num_pipe.named_steps

def test_model_loading_and_prediction():
    """Verify saved pipelines load correctly and yield valid probabilities [0, 1]."""
    models_dir = '/Users/rjahnavisantoshi/Desktop/CreditRisk/models'
    lr_path = os.path.join(models_dir, 'logistic_regression_pipeline.joblib')
    gb_path = os.path.join(models_dir, 'gradient_boosting_pipeline.joblib')
    
    assert os.path.exists(lr_path), "Logistic regression pipeline file missing"
    assert os.path.exists(gb_path), "Gradient boosting pipeline file missing"
    
    lr_pipe = joblib.load(lr_path)
    gb_pipe = joblib.load(gb_path)
    
    # Create a mock single-row dataframe matching features format
    mock_df = pd.DataFrame([{
        'NAME_CONTRACT_TYPE': 'Cash loans',
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 1,
        'AMT_INCOME_TOTAL': 150000.0,
        'AMT_CREDIT': 450000.0,
        'AMT_ANNUITY': 22500.0,
        'AMT_GOODS_PRICE': 400000.0,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Secondary / special education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'DAYS_BIRTH': -12000,
        'DAYS_EMPLOYED': -2000,
        'DAYS_REGISTRATION': -1500,
        'DAYS_ID_PUBLISH': -800,
        'OCCUPATION_TYPE': 'Laborers',
        'CNT_FAM_MEMBERS': 3,
        'REGION_RATING_CLIENT': 2,
        'ORGANIZATION_TYPE': 'Business Entity Type 3',
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.6,
        'EXT_SOURCE_3': 0.4,
        'DAYS_LAST_PHONE_CHANGE': -500,
        'AMT_REQ_CREDIT_BUREAU_YEAR': 2
    }])
    
    # Predictions range check
    for pipe in [lr_pipe, gb_pipe]:
        probs = pipe.predict_proba(mock_df)
        assert probs.shape == (1, 2)
        p_default = probs[0, 1]
        assert 0.0 <= p_default <= 1.0, f"Probability {p_default} out of range [0, 1]"

def test_prediction_service(sample_raw_applicant):
    """Verify PredictionService accurately translates variables and classifies risk."""
    service = PredictionService()
    assert service.is_loaded, "PredictionService failed to load model assets"
    
    # Run test for logistic regression
    res_lr = service.predict(sample_raw_applicant, 'logistic_regression', 0.5)
    assert 'default_probability' in res_lr
    assert 'risk_classification' in res_lr
    assert 'explanation' in res_lr
    assert 0.0 <= res_lr['default_probability'] <= 1.0
    
    # Verify classification logic
    prob = res_lr['default_probability']
    expected_class = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    assert res_lr['risk_classification'] == expected_class
    
    # Assert explanation is populated
    assert len(res_lr['explanation']) > 0
    assert 'display_name' in res_lr['explanation'][0]

def test_api_health_endpoint():
    """Test health check route."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "CreditRisk-X API"}

def test_api_model_info_endpoint():
    """Test model-info retrieval."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert "models_summary" in data
    assert "features" in data

def test_api_predict_success(sample_raw_applicant):
    """Test POST predict returns expected predictions."""
    response = client.post("/api/predict?model_key=logistic_regression&threshold=0.4", json=sample_raw_applicant)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_classification" in data
    assert data["decision_threshold"] == 0.4
    assert data["model_used"] == "logistic_regression"

def test_api_predict_invalid_input(sample_raw_applicant):
    """Test POST predict returns 422 Unprocessable Entity when validation fails."""
    # Delete a required key like 'income_total'
    invalid_applicant = sample_raw_applicant.copy()
    del invalid_applicant['income_total']
    
    response = client.post("/api/predict?model_key=logistic_regression&threshold=0.5", json=invalid_applicant)
    assert response.status_code == 422 # Pydantic validation error
