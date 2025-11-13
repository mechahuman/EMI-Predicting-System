import joblib
import os

def load_classification_model():
    path = "artifacts/classification/model.pkl"
    return joblib.load(path)

def load_regression_model():
    path = "artifacts/regression/model.pkl"
    return joblib.load(path)
