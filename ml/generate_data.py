import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=10000, seed=42):
    np.random.seed(seed)
    
    # 1. Base applicant profiles
    sk_ids = np.arange(100001, 100001 + num_samples)
    
    # Target variable (imbalance: ~8.1% default rate in the original dataset)
    # We will define a risk score that dictates target probability to create real correlations
    
    # Generate predictive numerical features first
    # EXT_SOURCES: normalized scores between 0 and 1
    ext_source_2 = np.random.beta(5, 5, size=num_samples) # centered around 0.5
    ext_source_3 = np.random.beta(4, 5, size=num_samples) # centered around 0.44
    ext_source_1 = np.random.beta(3, 4, size=num_samples) # centered around 0.43
    
    # Age in days (DAYS_BIRTH: between -25200 (70 years) and -7300 (20 years))
    age_years = np.random.uniform(20, 70, size=num_samples)
    days_birth = -1 * np.round(age_years * 365.25).astype(int)
    
    # Income (AMT_INCOME_TOTAL: lognormal distribution)
    amt_income_total = np.random.lognormal(mean=11.9, sigma=0.5, size=num_samples)
    amt_income_total = np.round(amt_income_total / 1000) * 1000 # round to nearest 1000
    
    # Credit Amount (AMT_CREDIT: typically correlated with income)
    # Let's make credit amount roughly 3x to 6x income, plus some random noise
    amt_credit = amt_income_total * np.random.uniform(2.5, 5.5, size=num_samples)
    amt_credit = np.round(amt_credit / 500) * 500
    
    # Annuity (AMT_ANNUITY: credit / 10 to 20, plus noise)
    amt_annuity = amt_credit / np.random.uniform(10, 25, size=num_samples)
    amt_annuity = np.round(amt_annuity / 10) * 10
    
    # Goods Price (AMT_GOODS_PRICE: close to credit amount for cash/consumer loans)
    amt_goods_price = amt_credit * np.random.uniform(0.85, 1.0, size=num_samples)
    amt_goods_price = np.round(amt_goods_price / 1000) * 1000
    
    # Days Employed: negative number. Pensioners have positive 365243.
    is_pensioner = (age_years > 60) & (np.random.rand(num_samples) > 0.3)
    days_employed = np.zeros(num_samples, dtype=int)
    # Employed: duration is between 0 and age - 18
    for i in range(num_samples):
        if is_pensioner[i]:
            days_employed[i] = 365243
        else:
            max_work_years = age_years[i] - 18
            work_years = np.random.uniform(0.1, max_work_years)
            days_employed[i] = -1 * int(np.round(work_years * 365.25))
            
    days_registration = -1 * np.round(np.random.exponential(scale=10, size=num_samples) * 365.25).astype(int)
    days_id_publish = -1 * np.round(np.random.uniform(0, 15, size=num_samples) * 365.25).astype(int)
    days_last_phone_change = -1 * np.round(np.random.exponential(scale=3, size=num_samples) * 365.25).astype(int)
    
    cnt_children = np.random.choice([0, 1, 2, 3, 4], p=[0.72, 0.18, 0.08, 0.018, 0.002], size=num_samples)
    cnt_fam_members = cnt_children + np.random.choice([1, 2], p=[0.25, 0.75], size=num_samples) # single or partner
    
    region_rating = np.random.choice([1, 2, 3], p=[0.15, 0.70, 0.15], size=num_samples)
    queries_year = np.random.poisson(lam=1.8, size=num_samples)
    
    # Categoricals
    contract_types = np.random.choice(['Cash loans', 'Revolving loans'], p=[0.90, 0.10], size=num_samples)
    genders = np.random.choice(['F', 'M', 'XNA'], p=[0.66, 0.34, 0.00], size=num_samples) # mostly female in dataset
    own_car = np.random.choice(['Y', 'N'], p=[0.34, 0.66], size=num_samples)
    own_realty = np.random.choice(['Y', 'N'], p=[0.69, 0.31], size=num_samples)
    
    income_types = []
    education_types = []
    family_statuses = []
    housing_types = []
    occupation_types = []
    organization_types = []
    
    inc_choices = ['Working', 'Commercial associate', 'Pensioner', 'State servant']
    inc_probs = [0.51, 0.23, 0.18, 0.08]
    
    edu_choices = ['Secondary / special education', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    edu_probs = [0.71, 0.24, 0.03, 0.018, 0.002]
    
    fam_choices = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    fam_probs = [0.64, 0.15, 0.10, 0.06, 0.05]
    
    house_choices = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
    house_probs = [0.88, 0.08, 0.02, 0.015, 0.004, 0.001]
    
    occ_choices = ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff']
    occ_probs = [0.35, 0.20, 0.15, 0.10, 0.08, 0.04, 0.04, 0.03, 0.01]
    
    org_choices = ['Business Entity Type 3', 'Self-employed', 'Other', 'Medicine', 'Government', 'School', 'Trade: type 7', 'Construction', 'Kindergarten']
    org_probs = [0.22, 0.16, 0.15, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03]
    # Add rest to make 1.0
    org_choices.append('Industry: type 9')
    org_probs.append(0.21)
    
    for i in range(num_samples):
        # Income Type matches Pensioner status
        if days_employed[i] == 365243:
            inc = 'Pensioner'
            occ = 'None'
            org = 'XNA'
        else:
            inc = np.random.choice(inc_choices[:2] + [inc_choices[3]], p=[inc_probs[0]/0.82, inc_probs[1]/0.82, inc_probs[3]/0.82])
            occ = np.random.choice(occ_choices, p=occ_probs)
            org = np.random.choice(org_choices, p=org_probs)
            
        income_types.append(inc)
        education_types.append(np.random.choice(edu_choices, p=edu_probs))
        family_statuses.append(np.random.choice(fam_choices, p=fam_probs))
        housing_types.append(np.random.choice(house_choices, p=house_probs))
        occupation_types.append(occ)
        organization_types.append(org)
        
    # Introduce Missing Values in EXT_SOURCES (similar to real data)
    # EXT_SOURCE_1 has high missing rate (~56%)
    # EXT_SOURCE_2 has very low missing rate (~0.2%)
    # EXT_SOURCE_3 has moderate missing rate (~20%)
    ext_source_1[np.random.rand(num_samples) < 0.56] = np.nan
    ext_source_2[np.random.rand(num_samples) < 0.002] = np.nan
    ext_source_3[np.random.rand(num_samples) < 0.20] = np.nan
    
    # OCCUPATION_TYPE has ~31% missing rate in real dataset
    occupation_types = [np.nan if (np.random.rand() < 0.31 and occ != 'None') else occ for occ in occupation_types]
    # Replace 'None' for Pensioners back to NaN as well
    occupation_types = [np.nan if occ == 'None' else occ for occ in occupation_types]
    
    # 2. Risk score & Target generation (Ground Truth Default Probability)
    # Risk calculation based on logical variables (Higher score = Higher risk)
    # Logit model to determine probability of default
    
    # Fill temp values for linear combination calculation
    temp_ext1 = np.nan_to_num(ext_source_1, nan=0.43)
    temp_ext2 = np.nan_to_num(ext_source_2, nan=0.50)
    temp_ext3 = np.nan_to_num(ext_source_3, nan=0.44)
    
    # Young clients have higher risk (convert to positive age, normalize)
    norm_age = -days_birth / 365.25
    # High debt-to-income ratio increases risk
    dti = amt_credit / amt_income_total
    
    # Pensioners have slightly lower risk, Working has higher risk
    is_work = np.array([1.0 if t == 'Working' else 0.0 for t in income_types])
    
    # Education: Higher education has lower risk
    is_high_edu = np.array([1.0 if t == 'Higher education' or t == 'Academic degree' else 0.0 for t in education_types])
    
    # Construct logits: baseline default rate log odds
    logit = (
        0.9
        - 2.8 * temp_ext2 # Strongest negative predictor
        - 2.0 * temp_ext3 # Second strongest negative predictor
        - 1.5 * temp_ext1 # Third strongest negative predictor
        - 0.02 * norm_age # Older client -> lower risk
        + 0.15 * dti      # High credit/income ratio -> higher risk
        + 0.40 * is_work  # Working category -> higher risk
        - 0.60 * is_high_edu # Higher education -> lower risk
        + 0.3 * (region_rating - 2) # Worse region rating -> higher risk
    )
    
    # Convert logits to probability
    prob = 1 / (1 + np.exp(-logit))
    
    # Sample Target (binary classification 0 or 1)
    target = np.random.binomial(n=1, p=prob)
    
    # Construct DataFrame
    df = pd.DataFrame({
        'SK_ID_CURR': sk_ids,
        'TARGET': target,
        'NAME_CONTRACT_TYPE': contract_types,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'AMT_GOODS_PRICE': amt_goods_price,
        'NAME_INCOME_TYPE': income_types,
        'NAME_EDUCATION_TYPE': education_types,
        'NAME_FAMILY_STATUS': family_statuses,
        'NAME_HOUSING_TYPE': housing_types,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'DAYS_REGISTRATION': days_registration,
        'DAYS_ID_PUBLISH': days_id_publish,
        'OCCUPATION_TYPE': occupation_types,
        'CNT_FAM_MEMBERS': cnt_fam_members,
        'REGION_RATING_CLIENT': region_rating,
        'ORGANIZATION_TYPE': organization_types,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'DAYS_LAST_PHONE_CHANGE': days_last_phone_change,
        'AMT_REQ_CREDIT_BUREAU_YEAR': queries_year
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    df = generate_synthetic_data()
    file_path = os.path.join(data_dir, 'application_train.csv')
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic application_train.csv with shape {df.shape} at {file_path}")
    print(f"Default rate: {df['TARGET'].mean() * 100:.2f}%")
