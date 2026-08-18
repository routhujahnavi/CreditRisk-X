from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(numerical_cols, categorical_cols, use_scaling=True):
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    Ensures that transformations are configured but not fit, 
    preventing data leakage when used within cross-validation.
    """
    # 1. Numerical pipeline
    num_steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    if use_scaling:
        num_steps.append(('scaler', StandardScaler()))
        
    num_transformer = Pipeline(steps=num_steps)
    
    # 2. Categorical pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='drop' # Drop columns like SK_ID_CURR and TARGET
    )
    
    return preprocessor
