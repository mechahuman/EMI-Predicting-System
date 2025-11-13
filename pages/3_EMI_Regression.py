import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils.load_models import load_regression_model

st.title("Max EMI Amount — Regression")

@st.cache_resource
def get_reg():
    return load_regression_model()

reg = get_reg()

# Provide only the numeric inputs that regression expects (the features list you used to train)
monthly_salary = st.number_input("monthly_salary", min_value=0.0, value=30000.0)
total_monthly_expenses = st.number_input("total_monthly_expenses", min_value=0.0, value=5000.0)
disposable_income = st.number_input("disposable_income", min_value=-1e9, value=25000.0)
emi_to_income_ratio = st.number_input("emi_to_income_ratio", min_value=0.0, value=0.0, format="%.4f")
debt_to_income_ratio = st.number_input("debt_to_income_ratio", min_value=0.0, value=0.0, format="%.4f")
savings_ratio = st.number_input("savings_ratio", min_value=0.0, value=0.1, format="%.4f")
loan_to_salary_ratio = st.number_input("loan_to_salary_ratio", min_value=0.0, value=0.14, format="%.4f")
expense_to_income_ratio = st.number_input("expense_to_income_ratio", min_value=0.0, value=0.16, format="%.4f")
dependents_per_income = st.number_input("dependents_per_income", min_value=0.0, value=0.0, format="%.6f")
income_per_person = st.number_input("income_per_person", min_value=0.0, value=15000.0)
credit_score = st.number_input("credit_score", min_value=0.0, value=600.0)
current_emi_amount = st.number_input("current_emi_amount", min_value=0.0, value=0.0)
years_of_employment = st.number_input("years_of_employment", min_value=0.0, value=2.0)

# Create DataFrame with regression feature order (as trained)
X_reg = pd.DataFrame([[
    monthly_salary,
    total_monthly_expenses,
    disposable_income,
    emi_to_income_ratio,
    debt_to_income_ratio,
    savings_ratio,
    loan_to_salary_ratio,
    expense_to_income_ratio,
    dependents_per_income,
    income_per_person,
    credit_score,
    current_emi_amount,
    years_of_employment
]], columns=[
    'monthly_salary','total_monthly_expenses','disposable_income','emi_to_income_ratio',
    'debt_to_income_ratio','savings_ratio','loan_to_salary_ratio','expense_to_income_ratio',
    'dependents_per_income','income_per_person','credit_score','current_emi_amount','years_of_employment'
])

st.subheader("Regression input preview")
st.dataframe(X_reg.T)

if st.button("Predict Max EMI"):
    try:
        pred = reg.predict(X_reg)
        st.success(f"Predicted Max Monthly EMI: ₹ {pred[0]:,.2f}")
    except Exception as e:
        st.error("Prediction failed. See diagnostic below.")
        st.exception(e)
