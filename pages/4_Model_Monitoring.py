import streamlit as st

st.title("Model Monitoring")

st.write("Upload MLflow metrics or model monitoring data.")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    import pandas as pd
    df = pd.read_csv(file)
    st.dataframe(df)
