import streamlit as st
import pandas as pd

st.title("Data Explorer")

file = st.file_uploader("Upload cleaned dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset Loaded Successfully")

    st.subheader("Preview")
    st.dataframe(df)

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Null Values")
    st.write(df.isna().sum())
    
else:
    st.info("Upload the cleaned dataset to explore.")
