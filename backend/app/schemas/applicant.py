from pydantic import BaseModel, Field
from typing import Optional

class ApplicantInput(BaseModel):
    # Demographics
    age_years: float = Field(..., ge=18, le=100, description="Age of applicant in years")
    gender: str = Field(..., description="Gender: M or F")
    children_count: int = Field(..., ge=0, le=20, description="Number of children")
    family_members_count: int = Field(..., ge=1, le=22, description="Number of family members")
    family_status: str = Field(..., description="Family status: Married, Single / not married, Civil marriage, Separated, Widow")
    education_type: str = Field(..., description="Education: Secondary / special education, Higher education, Incomplete higher, Lower secondary, Academic degree")
    housing_type: str = Field(..., description="Housing situation: House / apartment, With parents, Municipal apartment, Rented apartment, Office apartment, Co-op apartment")
    
    # Financials
    income_total: float = Field(..., ge=0, description="Annual total income")
    credit_amount: float = Field(..., ge=0, description="Loan credit amount requested")
    annuity_amount: float = Field(..., ge=0, description="Loan annuity amount")
    goods_price: float = Field(..., ge=0, description="Price of goods for consumer loan")
    
    # Employment
    employment_status: str = Field(..., description="Employment status: Employed, Pensioner, Unemployed")
    employment_duration_years: float = Field(0.0, ge=0, le=80, description="Duration in current job in years (0 if pensioner/unemployed)")
    income_type: str = Field(..., description="Income category: Working, Commercial associate, Pensioner, State servant")
    occupation_type: Optional[str] = Field(None, description="Occupation category (can be null)")
    organization_type: str = Field(..., description="Organization category")
    
    # Assets
    owns_car: str = Field(..., description="Owns car: Y or N")
    owns_realty: str = Field(..., description="Owns real estate: Y or N")
    
    # Registration / ID
    years_registration_change: float = Field(..., ge=0, le=100, description="Years since registration change")
    years_id_publish: float = Field(..., ge=0, le=100, description="Years since identity document change")
    years_phone_change: float = Field(..., ge=0, le=100, description="Years since last phone change")
    
    # Credit Bureau queries
    credit_bureau_queries_year: int = Field(..., ge=0, le=50, description="Number of Credit Bureau queries in last year")
    region_rating: int = Field(..., ge=1, le=3, description="Region rating: 1, 2, or 3")
    
    # External scores (highly predictive, often missing)
    ext_source_1: Optional[float] = Field(None, ge=0.0, le=1.0, description="External source score 1")
    ext_source_2: Optional[float] = Field(None, ge=0.0, le=1.0, description="External source score 2")
    ext_source_3: Optional[float] = Field(None, ge=0.0, le=1.0, description="External source score 3")

    class Config:
        json_schema_extra = {
            "example": {
                "age_years": 35.0,
                "gender": "F",
                "children_count": 1,
                "family_members_count": 3,
                "family_status": "Married",
                "education_type": "Secondary / special education",
                "housing_type": "House / apartment",
                "income_total": 180000.0,
                "credit_amount": 450000.0,
                "annuity_amount": 22500.0,
                "goods_price": 400000.0,
                "employment_status": "Employed",
                "employment_duration_years": 5.5,
                "income_type": "Working",
                "occupation_type": "Core staff",
                "organization_type": "Business Entity Type 3",
                "owns_car": "N",
                "owns_realty": "Y",
                "years_registration_change": 4.2,
                "years_id_publish": 2.1,
                "years_phone_change": 1.5,
                "credit_bureau_queries_year": 1,
                "region_rating": 2,
                "ext_source_1": 0.45,
                "ext_source_2": 0.52,
                "ext_source_3": 0.41
            }
        }
