import os
import json
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.applicant import ApplicantInput
from backend.app.services.prediction import PredictionService

router = APIRouter(prefix="/api")

# Instantiate prediction service
prediction_service = PredictionService()

@router.get("/health")
def get_health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CreditRisk-X API"}

@router.get("/model-info")
def get_model_info():
    """Returns details on trained models and features."""
    if not prediction_service.is_loaded:
        prediction_service.load_models()
        if not prediction_service.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded or not trained yet.")
            
    # Return basic info, best model, features
    return {
        "best_model": prediction_service.metadata.get("best_model"),
        "dataset_info": prediction_service.metadata.get("dataset_info"),
        "models_summary": {
            k: {
                "name": v["name"],
                "cv_mean": v["cv_mean"],
                "cv_std": v["cv_std"],
                "test_auc": v["test_auc"],
                "test_pr_auc": v["test_pr_auc"]
            }
            for k, v in prediction_service.metadata.get("models", {}).items()
        },
        "features": prediction_service.metadata.get("feature_info")
    }

@router.get("/metrics")
def get_metrics(model_key: str = Query(..., description="Model key: logistic_regression or gradient_boosting")):
    """Returns detailed cross-validation and evaluation metrics for a specific model."""
    if not prediction_service.is_loaded:
        prediction_service.load_models()
        if not prediction_service.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded or not trained yet.")
            
    models_metadata = prediction_service.metadata.get("models", {})
    if model_key not in models_metadata:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")
        
    m = models_metadata[model_key]
    return {
        "name": m["name"],
        "cv_scores": m["cv_scores"],
        "cv_mean": m["cv_mean"],
        "cv_std": m["cv_std"],
        "test_auc": m["test_auc"],
        "test_pr_auc": m["test_pr_auc"],
        "default_metrics": m["default_metrics"],
        "roc_curve": m["roc_curve"],
        "pr_curve": m["pr_curve"],
        "importance": m["importance"]
    }

@router.get("/threshold-analysis")
def get_threshold_analysis(model_key: str = Query(..., description="Model key")):
    """Returns threshold analysis data (precision, recall, f1 vs threshold)."""
    if not prediction_service.is_loaded:
        prediction_service.load_models()
        if not prediction_service.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded or not trained yet.")
            
    models_metadata = prediction_service.metadata.get("models", {})
    if model_key not in models_metadata:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")
        
    return models_metadata[model_key]["threshold_analysis"]

@router.get("/feature-importance")
def get_feature_importance(model_key: str = Query(..., description="Model key")):
    """Returns global feature importances or coefficients."""
    if not prediction_service.is_loaded:
        prediction_service.load_models()
        if not prediction_service.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded or not trained yet.")
            
    models_metadata = prediction_service.metadata.get("models", {})
    if model_key not in models_metadata:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found.")
        
    return models_metadata[model_key]["importance"]

@router.get("/data-profile")
def get_data_profile():
    """Returns dataset profiling report generated from training data."""
    profile_path = os.path.join(prediction_service.models_dir, 'data_profile.json')
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=503, detail="Dataset profiling report not found. Run data_profiling.py first.")
        
    try:
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset profiling report: {e}")

@router.post("/predict")
def predict_risk(applicant: ApplicantInput, model_key: str = "gradient_boosting", threshold: float = 0.5):
    """Executes credit risk prediction for a single applicant."""
    if not prediction_service.is_loaded:
        prediction_service.load_models()
        if not prediction_service.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded or not trained yet.")
            
    try:
        result = prediction_service.predict(applicant.model_dump(), model_key, threshold)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
