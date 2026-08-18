import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PredictionService:
    def __init__(self):
        self.models_dir = '/Users/rjahnavisantoshi/Desktop/CreditRisk/models'
        self.metadata_path = os.path.join(self.models_dir, 'pipeline_metadata.json')
        self.pipelines = {}
        self.metadata = {}
        self.is_loaded = False
        
        # Load pipelines and metadata if they exist
        self.load_models()

    def load_models(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load both pipelines
                lr_path = os.path.join(self.models_dir, 'logistic_regression_pipeline.joblib')
                gb_path = os.path.join(self.models_dir, 'gradient_boosting_pipeline.joblib')
                
                if os.path.exists(lr_path):
                    self.pipelines['logistic_regression'] = joblib.load(lr_path)
                if os.path.exists(gb_path):
                    self.pipelines['gradient_boosting'] = joblib.load(gb_path)
                    
                self.is_loaded = True
                print("Models loaded successfully in PredictionService.")
            except Exception as e:
                print(f"Error loading models: {e}")
                self.is_loaded = False
        else:
            print("Model metadata not found. Run train.py first.")
            self.is_loaded = False

    def predict(self, input_data: Dict[str, Any], model_key: str, threshold: float) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()
            if not self.is_loaded:
                raise RuntimeError("Models are not trained or loaded. Please check model files.")
                
        if model_key not in self.pipelines:
            raise ValueError(f"Model key '{model_key}' not found. Available: {list(self.pipelines.keys())}")
            
        pipeline = self.pipelines[model_key]
        
        # 1. Map input_data (Pydantic dict) to raw DataFrame features
        # Convert user-friendly fields to negative days, etc.
        days_birth = -1 * int(round(input_data['age_years'] * 365.25))
        
        if input_data['employment_status'] == 'Pensioner':
            days_employed = 365243
        elif input_data['employment_status'] == 'Unemployed':
            days_employed = -1 * int(round(0)) # 0 days
        else:
            days_employed = -1 * int(round(input_data['employment_duration_years'] * 365.25))
            
        days_registration = -1 * int(round(input_data['years_registration_change'] * 365.25))
        days_id_publish = -1 * int(round(input_data['years_id_publish'] * 365.25))
        days_last_phone_change = -1 * int(round(input_data['years_phone_change'] * 365.25))
        
        # Map back to model expected columns
        raw_features = {
            'NAME_CONTRACT_TYPE': 'Cash loans',  # Default contract type
            'CODE_GENDER': input_data['gender'],
            'FLAG_OWN_CAR': input_data['owns_car'],
            'FLAG_OWN_REALTY': input_data['owns_realty'],
            'CNT_CHILDREN': input_data['children_count'],
            'AMT_INCOME_TOTAL': input_data['income_total'],
            'AMT_CREDIT': input_data['credit_amount'],
            'AMT_ANNUITY': input_data['annuity_amount'],
            'AMT_GOODS_PRICE': input_data['goods_price'],
            'NAME_INCOME_TYPE': input_data['income_type'],
            'NAME_EDUCATION_TYPE': input_data['education_type'],
            'NAME_FAMILY_STATUS': input_data['family_status'],
            'NAME_HOUSING_TYPE': input_data['housing_type'],
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'DAYS_REGISTRATION': days_registration,
            'DAYS_ID_PUBLISH': days_id_publish,
            'OCCUPATION_TYPE': input_data['occupation_type'],
            'CNT_FAM_MEMBERS': input_data['family_members_count'],
            'REGION_RATING_CLIENT': input_data['region_rating'],
            'ORGANIZATION_TYPE': input_data['organization_type'],
            'EXT_SOURCE_1': input_data['ext_source_1'],
            'EXT_SOURCE_2': input_data['ext_source_2'],
            'EXT_SOURCE_3': input_data['ext_source_3'],
            'DAYS_LAST_PHONE_CHANGE': days_last_phone_change,
            'AMT_REQ_CREDIT_BUREAU_YEAR': input_data['credit_bureau_queries_year']
        }
        
        # Create DataFrame (single row)
        df_input = pd.DataFrame([raw_features])
        
        # Ensure correct column order matching X_train
        expected_cols = self.metadata['feature_info']['numerical_cols'] + self.metadata['feature_info']['categorical_cols']
        # Reorder df columns
        df_input = df_input[expected_cols]
        
        # 2. Get prediction probability
        prob = float(pipeline.predict_proba(df_input)[0, 1])
        
        # Determine classification based on threshold
        risk_class = "HIGH RISK" if prob >= threshold else "LOW RISK"
        
        # 3. Local explanations
        explanation = self._explain_prediction(df_input, pipeline, model_key, raw_features)
        
        return {
            "default_probability": prob,
            "risk_classification": risk_class,
            "decision_threshold": threshold,
            "model_used": model_key,
            "explanation": explanation
        }

    def _explain_prediction(self, df_input: pd.DataFrame, pipeline, model_key: str, raw_features: dict) -> List[Dict[str, Any]]:
        explanations = []
        
        preprocessor = pipeline.named_steps['preprocessor']
        estimator = pipeline.named_steps['estimator']
        
        # Get processed features representation
        # Transformed values
        X_trans = preprocessor.transform(df_input)[0]
        
        # Reconstruct feature names (numerical and categorical one-hot encoded)
        numerical_cols = self.metadata['feature_info']['numerical_cols']
        categorical_cols = self.metadata['feature_info']['categorical_cols']
        
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features_ohe = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        
        all_features_transformed = numerical_cols + cat_features_ohe
        
        # Explanation logic
        if model_key == 'logistic_regression':
            # Local Linear explanation: Contribution = Coefficient * Processed Value
            coefs = estimator.coef_[0]
            intercept = float(estimator.intercept_[0])
            
            contributions = {}
            for name, val, coef in zip(all_features_transformed, X_trans, coefs):
                # Only keep non-zero and non-trivial contributions
                contribution = float(val * coef)
                contributions[name] = contribution
                
            # Aggregate one-hot categories back to their original columns for user readability
            raw_contributions = {}
            
            # 1. Numerical contributions
            for col in numerical_cols:
                raw_contributions[col] = contributions.get(col, 0.0)
                
            # 2. Categorical contributions (sum up OHE categories)
            for cat_col in categorical_cols:
                raw_contributions[cat_col] = 0.0
                
            for feat_name, contribution in contributions.items():
                if feat_name not in numerical_cols:
                    # Find which categorical column this OHE feature belongs to
                    for cat_col in categorical_cols:
                        if feat_name.startswith(cat_col + "_"):
                            raw_contributions[cat_col] += contribution
                            break
                            
            # Convert to UI explanation list
            for feature, value in raw_contributions.items():
                # Display name
                display_name = feature.replace('_', ' ').title()
                
                # Check sign
                # Logistic regression link is logit(p). 
                # Positive contribution = increases risk of default.
                # Negative contribution = decreases risk of default.
                influence = "High Risk" if value > 0 else "Low Risk"
                
                # Formulate human-friendly explanation text
                text = self._get_feature_explanation_text(feature, raw_features[feature], value > 0)
                
                explanations.append({
                    "feature": feature,
                    "display_name": display_name,
                    "contribution": value,
                    "influence": influence,
                    "text": text
                })
                
        else:
            # HistGradientBoosting (Nonlinear)
            # We construct surrogate explanations using the applicant's raw values relative to the 
            # median training distribution, weighted by the feature's global Permutation Importance.
            # This represents: "How much this feature deviates from normal, and how important that feature is to the model."
            # Load data profiles to find median values
            profile_path = os.path.join(self.models_dir, 'data_profile.json')
            try:
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                num_profiles = profile['numerical_features']
                cat_profiles = profile['categorical_features']
            except:
                num_profiles = {}
                cat_profiles = {}
                
            gb_importance = self.metadata['models']['gradient_boosting']['importance']['values']
            
            for feature, importance in gb_importance.items():
                if importance <= 0:
                    continue
                    
                val = raw_features[feature]
                display_name = feature.replace('_', ' ').title()
                
                # Assess impact direction based on our known monotonic risk relationships
                # (e.g. low external source = higher risk; young age = higher risk; high DTI = higher risk)
                # Compare value with dataset median/mode
                is_high_risk = False
                diff_desc = ""
                
                if feature in numerical_cols and feature in num_profiles:
                    median_val = num_profiles[feature]['median']
                    if pd.isna(val):
                        diff_desc = "Value is missing (imputed as median)"
                        # Imputed value doesn't deviate from standard
                        continue
                    else:
                        deviation = val - median_val
                        # Let's map deviation to risk direction based on column properties
                        if feature in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
                            # Lower external source means higher risk
                            is_high_risk = deviation < 0
                            diff_desc = f"Below average score ({val:.3f} vs typical {median_val:.3f})" if is_high_risk else f"Above average score ({val:.3f} vs typical {median_val:.3f})"
                        elif feature == 'DAYS_BIRTH':
                            # Negative age, so closer to 0 (larger number) means younger, which means higher risk
                            # Deviation > 0 means younger (e.g., -10000 > -15000)
                            is_high_risk = deviation > 0
                            age_years = -val / 365.25
                            median_age = -median_val / 365.25
                            diff_desc = f"Younger applicant ({age_years:.1f} years vs typical {median_age:.1f})" if is_high_risk else f"Older applicant ({age_years:.1f} years vs typical {median_age:.1f})"
                        elif feature == 'DAYS_EMPLOYED':
                            # Pensioner is 365243 (positive). Unemployed is 0. Employed is negative.
                            # Younger/fewer years of work means higher risk.
                            if val == 365243:
                                is_high_risk = False
                                diff_desc = "Pensioner status"
                            else:
                                work_years = -val / 365.25
                                median_work = -median_val / 365.25 if median_val != 365243 else 5.0
                                is_high_risk = work_years < median_work
                                diff_desc = f"Shorter employment ({work_years:.1f} yrs vs typical {median_work:.1f})" if is_high_risk else f"Longer employment ({work_years:.1f} yrs vs typical {median_work:.1f})"
                        elif feature == 'AMT_CREDIT':
                            is_high_risk = deviation > 0
                            diff_desc = f"Higher credit requested (${val:,.0f} vs typical ${median_val:,.0f})" if is_high_risk else f"Lower credit requested (${val:,.0f} vs typical ${median_val:,.0f})"
                        elif feature == 'AMT_ANNUITY':
                            is_high_risk = deviation > 0
                            diff_desc = f"Higher monthly annuity (${val:,.0f} vs typical ${median_val:,.0f})" if is_high_risk else f"Lower monthly annuity (${val:,.0f} vs typical ${median_val:,.0f})"
                        elif feature == 'AMT_INCOME_TOTAL':
                            # Lower income means higher risk
                            is_high_risk = deviation < 0
                            diff_desc = f"Lower annual income (${val:,.0f} vs typical ${median_val:,.0f})" if is_high_risk else f"Higher annual income (${val:,.0f} vs typical ${median_val:,.0f})"
                        else:
                            # Standard numeric
                            is_high_risk = deviation > 0 if num_profiles[feature]['correlation_with_target'] > 0 else deviation < 0
                            diff_desc = f"Deviates from standard level"
                            
                elif feature in categorical_cols and feature in cat_profiles:
                    # Categorical feature
                    value_counts = cat_profiles[feature]['value_counts']
                    # Most frequent
                    mode_val = max(value_counts.keys(), key=lambda k: value_counts[k])
                    
                    if pd.isna(val) or val == 'None' or val is None:
                        continue
                        
                    is_mode = val == mode_val
                    
                    # Custom mapping for specific critical fields
                    if feature == 'NAME_EDUCATION_TYPE':
                        is_high_risk = val in ['Secondary / special education', 'Lower secondary']
                        diff_desc = f"Education level: {val}"
                    elif feature == 'CODE_GENDER':
                        is_high_risk = val == 'M' # Males statistically default more in this dataset
                        diff_desc = f"Gender: {val}"
                    else:
                        is_high_risk = False
                        diff_desc = f"Value: {val}"
                else:
                    continue
                    
                # Contribution magnitude equals importance * deviation direction multiplier
                contribution_val = importance * (1.0 if is_high_risk else -1.0)
                influence = "High Risk" if is_high_risk else "Low Risk"
                text = f"{diff_desc} ({'increases' if is_high_risk else 'decreases'} default risk)."
                
                explanations.append({
                    "feature": feature,
                    "display_name": display_name,
                    "contribution": contribution_val,
                    "influence": influence,
                    "text": text
                })
                
        # Sort explanations by absolute contribution descending, keep top 8
        explanations = sorted(explanations, key=lambda x: abs(x['contribution']), reverse=True)[:8]
        return explanations

    def _get_feature_explanation_text(self, feature: str, val: Any, is_high_risk: bool) -> str:
        # Generate custom text based on value and risk association
        if feature in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            return f"External credit rating score of {val} is {'low, indicating higher default risk' if is_high_risk else 'strong, lowering default risk'}."
        elif feature == 'DAYS_BIRTH':
            age = -val / 365.25
            return f"Applicant age of {age:.1f} years {'poses a higher default probability' if is_high_risk else 'indicates higher stability and lower risk'}."
        elif feature == 'DAYS_EMPLOYED':
            if val == 365243:
                return "Pensioner status lowers credit default probability."
            years = -val / 365.25
            return f"Employment duration of {years:.1f} years {'is short, which increases default risk' if is_high_risk else 'shows job stability, reducing default risk'}."
        elif feature == 'AMT_INCOME_TOTAL':
            return f"Annual income of ${val:,.0f} {'is lower relative to credit, increasing debt stress' if is_high_risk else 'is strong, supporting repayment ability'}."
        elif feature == 'AMT_CREDIT':
            return f"Requested credit amount of ${val:,.0f} {'is high relative to income' if is_high_risk else 'is conservative, reducing default risk'}."
        elif feature == 'NAME_EDUCATION_TYPE':
            return f"Education level ({val}) {'is associated with higher baseline credit risk' if is_high_risk else 'lowers baseline default risk'}."
        elif feature == 'CODE_GENDER':
            return f"Gender profile ({val}) {'is historically associated with slightly higher risk segments' if is_high_risk else 'is associated with lower risk segments'}."
        else:
            return f"Feature value '{val}' {'increases' if is_high_risk else 'decreases'} predicted credit default risk."
