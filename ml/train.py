import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix

from preprocessing import get_preprocessor

def train_and_evaluate():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = '/Users/rjahnavisantoshi/Desktop/CreditRisk/ml/data/application_train.csv'
    models_dir = '/Users/rjahnavisantoshi/Desktop/CreditRisk/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['SK_ID_CURR', 'TARGET'])
    y = df['TARGET']
    
    # Identify numerical and categorical cols
    feature_cols = X.columns
    numerical_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(X[col])]
    
    # 2. Train-Test Split (80/20) - Stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Save test set for potential verification tests
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(models_dir, 'test_set.csv'), index=False)
    
    # 3. Define models
    models = {
        'logistic_regression': {
            'name': 'Logistic Regression (Baseline)',
            'estimator': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
            'preprocessor': get_preprocessor(numerical_cols, categorical_cols, use_scaling=True)
        },
        'gradient_boosting': {
            'name': 'HistGradientBoosting (Nonlinear)',
            # HistGradientBoosting handles missing values natively, but we can also use imputation
            # We will use imputation in preprocessor for consistent feature inputs
            'estimator': HistGradientBoostingClassifier(random_state=42, max_iter=100, learning_rate=0.05),
            'preprocessor': get_preprocessor(numerical_cols, categorical_cols, use_scaling=True)
        }
    }
    
    # Metadata to save
    metadata = {
        "dataset_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(df),
            "default_rate_train": float(y_train.mean()),
            "default_rate_test": float(y_test.mean())
        },
        "models": {}
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Loop over models
    for model_key, model_info in models.items():
        print(f"Training {model_info['name']}...")
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', model_info['preprocessor']),
            ('estimator', model_info['estimator'])
        ])
        
        # A. Stratified Cross-Validation
        cv_scores = []
        oof_preds = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Fit pipeline on training fold only (prevents data leakage)
            pipeline.fit(X_tr, y_tr)
            
            # Predict probability on validation fold
            val_probs = pipeline.predict_proba(X_va)[:, 1]
            oof_preds[val_idx] = val_probs
            
            fold_auc = roc_auc_score(y_va, val_probs)
            cv_scores.append(fold_auc)
            print(f"  Fold {fold+1}: ROC-AUC = {fold_auc:.4f}")
            
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        print(f"  CV Mean ROC-AUC: {cv_mean:.4f} +/- {cv_std:.4f}")
        
        # B. Train on full training set
        pipeline.fit(X_train, y_train)
        
        # C. Evaluate on independent test set
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        
        # Precompute ROC and PR curves to save in metadata
        fpr, tpr, roc_thresholds = roc_curve(y_test, test_probs)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, test_probs)
        
        # We downsample curve points to keep JSON small
        # Select 50 points evenly spaced
        roc_indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        # pr_thresholds is 1 element shorter than precision and recall in scikit-learn
        pr_indices = np.linspace(0, len(pr_thresholds) - 1, 50, dtype=int)
        
        # Calculate PR-AUC
        pr_auc = float(auc(recall, precision))
        
        # Calculate confusion matrix for default 0.5 threshold
        cm = confusion_matrix(y_test, (test_probs >= 0.5).astype(int))
        tn, fp, fn, tp = map(int, cm.ravel())
        
        # Precompute metrics at different thresholds (0.05 to 0.95 in 0.05 increments)
        threshold_analysis = []
        for t in np.arange(0.05, 0.96, 0.05):
            t_round = round(t, 2)
            t_preds = (test_probs >= t_round).astype(int)
            t_cm = confusion_matrix(y_test, t_preds)
            t_tn, t_fp, t_fn, t_tp = map(int, t_cm.ravel())
            
            t_prec = float(t_tp / (t_tp + t_fp)) if (t_tp + t_fp) > 0 else 0.0
            t_rec = float(t_tp / (t_tp + t_fn)) if (t_tp + t_fn) > 0 else 0.0
            t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0
            
            # Rate of predicted high risk
            predicted_high_risk_pct = float(np.mean(t_preds) * 100)
            
            threshold_analysis.append({
                "threshold": t_round,
                "precision": t_prec,
                "recall": t_rec,
                "f1_score": t_f1,
                "tp": t_tp,
                "fp": t_fp,
                "tn": t_tn,
                "fn": t_fn,
                "predicted_high_risk_pct": predicted_high_risk_pct
            })
            
        # Get feature importance / coefficients
        importance_data = {}
        fitted_preprocessor = pipeline.named_steps['preprocessor']
        fitted_estimator = pipeline.named_steps['estimator']
        
        # Get feature names after preprocessing
        # 1. Numerical feature names (remain the same)
        num_features = numerical_cols
        # 2. Categorical feature names (after one-hot encoding)
        cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_ohe = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        
        all_features_transformed = num_features + cat_features_ohe
        
        if model_key == 'logistic_regression':
            coefs = fitted_estimator.coef_[0].tolist()
            importance_data = {
                "type": "coefficients",
                "values": {feat: float(coef) for feat, coef in zip(all_features_transformed, coefs)}
            }
        else:
            # HistGradientBoosting doesn't expose feature_importances_ natively if it's scikit-learn standard
            # but scikit-learn HistGradientBoosting does NOT have feature_importances_ attribute!
            # Wait, let's double check if HistGradientBoostingClassifier has feature_importances_.
            # It does not! Scikit-learn's HistGradientBoostingClassifier does not have feature_importances_.
            # To get feature importances for HistGradientBoosting, we can compute Permutation Importance,
            # or we can use a RandomForestClassifier which does have feature_importances_.
            # Let's check: can we use Permutation Importance on validation set, or can we check if there's a simple way?
            # Yes! Permutation importance is a standard, robust way.
            # Alternatively, since RandomForestClassifier is also a tree model, let's see if we should use RandomForest or HistGradientBoosting.
            # HistGradientBoosting is faster. Let's compute Permutation Importance for HistGradientBoosting,
            # or let's use a RandomForestClassifier if we want standard feature_importances_.
            # Actually, computing permutation importance is very academic and fits the objective.
            # Or we can write a quick custom importance estimation, or use RandomForest.
            # Let's see: how about we write a small permutation importance calculator?
            # Yes! It is simple:
            from sklearn.inspection import permutation_importance
            print("  Computing permutation importance for Gradient Boosting (this may take a few seconds)...")
            perm_imp = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42)
            # Permutation importance works on raw input features (X_test), which is even better because
            # it tells us the importance of the original columns (e.g. EXT_SOURCE_3) rather than the one-hot encoded categories!
            # This is extremely clean and user-friendly for the explainability page!
            importance_data = {
                "type": "permutation_importance",
                "values": {feat: float(imp) for feat, imp in zip(feature_cols, perm_imp.importances_mean)}
            }
            
        # Save pipeline file
        pipeline_file = os.path.join(models_dir, f'{model_key}_pipeline.joblib')
        joblib.dump(pipeline, pipeline_file)
        
        # Populate model metadata
        metadata["models"][model_key] = {
            "name": model_info["name"],
            "cv_scores": [float(s) for s in cv_scores],
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "test_auc": float(test_auc),
            "test_pr_auc": pr_auc,
            "default_metrics": {
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "f1_score": 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn)) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            },
            "roc_curve": {
                "fpr": [float(fpr[i]) for i in roc_indices],
                "tpr": [float(tpr[i]) for i in roc_indices],
                "thresholds": [float(roc_thresholds[i]) if np.isfinite(roc_thresholds[i]) else 1.0 for i in roc_indices]
            },
            "pr_curve": {
                "precision": [float(precision[i]) for i in pr_indices],
                "recall": [float(recall[i]) for i in pr_indices],
                "thresholds": [float(pr_thresholds[i]) for i in pr_indices]
            },
            "threshold_analysis": threshold_analysis,
            "importance": importance_data
        }
        
    # Write metadata json
    metadata["feature_info"] = {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols
    }
    
    # Save best model
    best_model_key = max(models.keys(), key=lambda k: metadata["models"][k]["test_auc"])
    metadata["best_model"] = best_model_key
    print(f"Best model is {best_model_key} with test ROC-AUC: {metadata['models'][best_model_key]['test_auc']:.4f}")
    
    # Save metadata
    metadata_file = os.path.join(models_dir, 'pipeline_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved all models and metadata to {models_dir}")

if __name__ == '__main__':
    train_and_evaluate()
