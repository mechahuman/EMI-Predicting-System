import numpy as np

def prepare_regression_features(data):
    return np.array([[
        data["monthly_salary"],
        data["total_expenses"],
        data["disposable_income"],
        data["emi_to_income_ratio"],
        data["debt_to_income_ratio"],
        data["savings_ratio"],
        data["loan_to_salary_ratio"],
        data["expense_to_income_ratio"],
        data["dependents_per_income"],
        data["income_per_person"],
        data["credit_score"],
        data["current_emi_amount"],
        data["years_of_employment"]
    ]])

def prepare_classification_features(data):
    # Simple same-order feature vector (this works because MLflow pipelines include encoder)
    return np.array([[
        data["age"],
        data["gender"],
        data["marital_status"],
        data["education"],
        data["monthly_salary"],
        data["employment_type"],
        data["years_of_employment"],
        data["company_type"],
        data["house_type"],
        data["monthly_rent"],
        data["family_size"],
        data["dependents"],
        data["school_fees"],
        data["college_fees"],
        data["travel_expenses"],
        data["groceries_utilities"],
        data["other_monthly_expenses"],
        data["existing_loans"],
        data["current_emi_amount"],
        data["credit_score"],
        data["bank_balance"],
        data["emergency_fund"],
        data["emi_scenario"],
        data["requested_amount"],
        data["requested_tenure"],
        data["total_expenses"],
        data["disposable_income"],
        data["emi_to_income_ratio"],
        data["debt_to_income_ratio"],
        data["savings_ratio"],
        data["affordibility_index"],
        data["loan_to_salary_ratio"],
        data["expense_to_income_ratio"],
        data["dependents_per_income"],
        data["income_per_person"]
    ]])
