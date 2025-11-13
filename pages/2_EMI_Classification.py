import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="EMI Eligibility (Classification)")
st.title("EMI Eligibility — Classification")

# ----------------- Load models -----------------
@st.cache_resource
def load_models():
    clf = joblib.load("artifacts/classification/model.pkl")
    try:
        reg = joblib.load("artifacts/regression/model.pkl")
    except:
        reg = None
    return clf, reg

clf, reg = load_models()

# ----------------- UI Inputs -----------------
st.write("Enter applicant details:")

age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

monthly_salary = st.number_input("Monthly Salary", 0.0, 200000.0, 30000.0, step=500.0)
employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 2.0)
company_type = st.text_input("Company Type", "Pvt")

house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
monthly_rent = st.number_input("Monthly Rent", 0.0, 100000.0, 0.0)
family_size = st.number_input("Family Size", 0, 20, 1)
dependents = st.number_input("Dependents", 0, 20, 0)

school_fees = st.number_input("School Fees", 0.0, 100000.0, 0.0)
college_fees = st.number_input("College Fees", 0.0, 100000.0, 0.0)
travel_expenses = st.number_input("Travel Expenses", 0.0, 50000.0, 1000.0)
groceries_utilities = st.number_input("Groceries + Utilities", 0.0, 50000.0, 3000.0)
other_monthly_expenses = st.number_input("Other Monthly Expenses", 0.0, 50000.0, 500.0)

existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
current_emi_amount = st.number_input("Current EMI Amount", 0.0, 200000.0, 0.0)
credit_score = st.number_input("Credit Score", 300.0, 850.0, 600.0)
bank_balance = st.number_input("Bank Balance", 0.0, 10000000.0, 10000.0)
emergency_fund = st.number_input("Emergency Fund", 0.0, 10000000.0, 5000.0)

emi_scenario = st.selectbox("EMI Scenario", ["Normal", "Critical", "Pre_Approved", "Pre_Qualified", "Other"])
requested_amount = st.number_input("Requested Loan Amount", 0.0, 10000000.0, 50000.0)
requested_tenure = st.number_input("Requested Tenure (months)", 1, 360, 12)

# ----------------- Derived columns -----------------
total_monthly_expenses = (
    monthly_rent + groceries_utilities + travel_expenses +
    school_fees + college_fees + other_monthly_expenses
)
disposable_income = monthly_salary - total_monthly_expenses
emi_to_income_ratio = (current_emi_amount / monthly_salary) if monthly_salary else 0.0
debt_to_income_ratio = (
    current_emi_amount + (requested_amount / (requested_tenure or 1))
) / (monthly_salary or 1)
savings_ratio = (emergency_fund / monthly_salary) if monthly_salary else 0.0
affordibility_index = (disposable_income / requested_amount) if requested_amount else 0.0
loan_to_salary_ratio = (requested_amount / (monthly_salary * 12)) if monthly_salary else 0.0
expense_to_income_ratio = (total_monthly_expenses / monthly_salary) if monthly_salary else 0.0
dependents_per_income = (dependents / monthly_salary) if monthly_salary else 0.0
income_per_person = (monthly_salary / (family_size + 1)) if (family_size + 1) else 0.0

# ----------------- Build input DataFrame -----------------
row = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education": education,
    "monthly_salary": monthly_salary,
    "employment_type": employment_type,
    "years_of_employment": years_of_employment,
    "company_type": company_type,
    "house_type": house_type,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
    "total_monthly_expenses": total_monthly_expenses,
    "disposable_income": disposable_income,
    "emi_to_income_ratio": emi_to_income_ratio,
    "debt_to_income_ratio": debt_to_income_ratio,
    "savings_ratio": savings_ratio,
    "affordibility_index": affordibility_index,
    "loan_to_salary_ratio": loan_to_salary_ratio,
    "expense_to_income_ratio": expense_to_income_ratio,
    "dependents_per_income": dependents_per_income,
    "income_per_person": income_per_person
}

X_df = pd.DataFrame([row])

# ----------------- Compute max_monthly_emi if classifier expects it -----------------
if "max_monthly_emi" in getattr(clf, "feature_names_in_", []):
    if reg is not None:
        reg_input = pd.DataFrame([[
            row["monthly_salary"],
            row["total_monthly_expenses"],
            row["disposable_income"],
            row["emi_to_income_ratio"],
            row["debt_to_income_ratio"],
            row["savings_ratio"],
            row["loan_to_salary_ratio"],
            row["expense_to_income_ratio"],
            row["dependents_per_income"],
            row["income_per_person"],
            row["credit_score"],
            row["current_emi_amount"],
            row["years_of_employment"]
        ]], columns=[
            'monthly_salary','total_monthly_expenses','disposable_income','emi_to_income_ratio',
            'debt_to_income_ratio','savings_ratio','loan_to_salary_ratio','expense_to_income_ratio',
            'dependents_per_income','income_per_person','credit_score','current_emi_amount','years_of_employment'
        ])
        X_df["max_monthly_emi"] = reg.predict(reg_input)[0]
    else:
        st.error("Classifier expects 'max_monthly_emi' but regression model is missing.")
        st.stop()

# ----------------- Label mapping -----------------
label_map = {
    0: "Eligible",
    1: "High_Risk",
    2: "Not_Eligible"
}

# ----------------- Prediction -----------------
if st.button("Predict EMI Eligibility"):
    try:
        pred_raw = clf.predict(X_df)
        pred_int = int(pred_raw[0]) if hasattr(pred_raw, "__len__") else int(pred_raw)
        display_label = label_map.get(pred_int, "Unknown")
        st.success(f"Predicted EMI Eligibility: **{display_label}**")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
