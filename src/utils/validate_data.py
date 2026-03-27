import great_expectations as ge
import pandas as pd
from typing import Tuple, List

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn.
    """
    print("🚀Starting data validation with Great Expectations...")

    # Convert to GE Dataset
    ge_df=ge.dataset.PandasDataset(df)

    # 1. SCHEMA & MISSING VALUES
    # ---------------------------------------------------------
    # Note: Using consistent naming (no spaces) to match typical Telco schemas
    critical_columns=[
        "customerID", "gender", "Partner", "Dependents", 
        "PhoneService", "InternetService", "Contract", 
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    
    for col in critical_columns:
        ge_df.expect_column_to_exist(col)
        ge_df.expect_column_values_to_not_be_null(col)

    # 2. CATEGORICAL BUSINESS LOGIC
    # ---------------------------------------------------------
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])

    # 3. NUMERIC VALIDATION
    # ---------------------------------------------------------
    # Tenure: 0 to 10 years
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Charges: Ensuring they are positive
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=250)
    
    # Data Consistency: TotalCharges >= MonthlyCharges
    # We use 'mostly' because first-month partial bills exist
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95 
    )

    # 4. EXECUTION & RESULTS
    # ---------------------------------------------------------
    results=ge_df.validate()
    
    failed_expectations=[
        r["expectation_config"]["expectation_type"] 
        for r in results["results"] if not r["success"]
    ]

    total_checks=len(results["results"])
    passed_checks=total_checks-len(failed_expectations)

    if results["success"]:
        print(f"PASSED:{passed_checks}/{total_checks} successful.")
    else:
        print(f"FAILED:{len(failed_expectations)}/{total_checks} failed.")
        print(f"Specific Failures:{failed_expectations}")

    return results["success"],failed_expectations