import streamlit as st

st.set_page_config(page_title="Loan EMI Prediction", page_icon="💰")

st.title("Loan EMI Prediction System")

st.write("""
Welcome!

Use the left sidebar to:

- Page 1: Explore the dataset  
- Page 2: Predict EMI Eligibility
- Page 3: Predict Maximum EMI Amount  
- Page 4: View model monitoring info  

This app uses MLflow-trained models.
""")
