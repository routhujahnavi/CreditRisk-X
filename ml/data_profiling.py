import os
import json
import pandas as pd
import numpy as np

def run_profiling(data_path, output_json_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 1. Basic stats
    num_rows, num_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    
    # Target distribution
    target_col = 'TARGET'
    target_counts = df[target_col].value_counts().to_dict()
    target_pct = (df[target_col].value_counts(normalize=True) * 100).to_dict()
    
    target_info = {
        "0_count": int(target_counts.get(0, 0)),
        "1_count": int(target_counts.get(1, 0)),
        "0_pct": float(target_pct.get(0, 0)),
        "1_pct": float(target_pct.get(1, 0)),
        "class_imbalance_ratio": float(target_counts.get(0, 0) / max(1, target_counts.get(1, 0)))
    }
    
    # 2. Features classification
    # Exclude SK_ID_CURR and TARGET
    feature_cols = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET']]
    
    numerical_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    # 3. Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = (df.isnull().mean() * 100).to_dict()
    
    missing_report = []
    for col in df.columns:
        missing_report.append({
            "column": col,
            "count": int(missing_counts[col]),
            "pct": float(missing_pcts[col]),
            "type": str(df[col].dtype)
        })
    missing_report = sorted(missing_report, key=lambda x: x['pct'], reverse=True)
    
    # 4. Numerical columns details & Target correlation
    numerical_details = {}
    correlations = df[numerical_cols + ['TARGET']].corr(numeric_only=True)['TARGET'].to_dict()
    
    constant_features = []
    for col in numerical_cols:
        desc = df[col].describe()
        std_val = float(desc.get('std', 0.0))
        mean_val = float(desc.get('mean', 0.0))
        min_val = float(desc.get('min', 0.0))
        max_val = float(desc.get('max', 0.0))
        median_val = float(df[col].median())
        
        # Check constant
        if np.isnan(std_val) or std_val < 1e-6:
            constant_features.append(col)
            
        numerical_details[col] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "median": median_val,
            "correlation_with_target": float(correlations.get(col, 0.0))
        }
        
    # Categorical columns details
    categorical_details = {}
    for col in categorical_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        unique_count = int(df[col].nunique())
        
        # Convert all keys to string (e.g. handle NaN key)
        val_counts_str = {str(k): int(v) for k, v in val_counts.items()}
        
        # Check constant
        if unique_count <= 1:
            constant_features.append(col)
            
        categorical_details[col] = {
            "unique_values_count": unique_count,
            "value_counts": val_counts_str
        }
        
    # Compile full report
    report = {
        "dataset_summary": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "numerical_features_count": len(numerical_cols),
            "categorical_features_count": len(categorical_cols),
            "duplicate_rows": duplicate_rows,
            "constant_features": constant_features
        },
        "target_distribution": target_info,
        "missing_values": missing_report,
        "numerical_features": numerical_details,
        "categorical_features": categorical_details
    }
    
    # Ensure directory for output exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Dataset profiling completed. Report saved to {output_json_path}")
    return report

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(base_dir, 'CreditRisk', 'ml', 'data', 'application_train.csv')
    # If path resolving fails in python run, use hardcoded workspace path
    if not os.path.exists(data_file):
        data_file = '/Users/rjahnavisantoshi/Desktop/CreditRisk/ml/data/application_train.csv'
    output_file = '/Users/rjahnavisantoshi/Desktop/CreditRisk/models/data_profile.json'
    
    run_profiling(data_file, output_file)
