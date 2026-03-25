import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
@st.cache_resource
def load_model():
    model = joblib.load('knn_model.pkl')
    preprocessors = joblib.load('preprocessors.pkl')
    return model, preprocessors

model, preprocessors = load_model()
city_encoder = preprocessors['city_encoder']
employment_encoder = preprocessors['employment_encoder']

# Page config
st.set_page_config(page_title="KNN Loan Prediction", layout="wide")
st.title("🏦 KNN Regression Loan Prediction App")
st.markdown("Enter customer details to predict the **target value** (likely loan repayment/approval amount)")

# Sidebar inputs
st.sidebar.header("Input Features")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 80, 40)
with col2:
    credit_score = st.slider("Credit Score", 300, 850, 650)

income = st.sidebar.number_input("Income", 0.0, 200000.0, 50000.0)
loan_amount = st.sidebar.number_input("Loan Amount", 0.0, 1000000.0, 200000.0)

city = st.sidebar.selectbox("City", ['Chennai', 'Hyderabad', 'Bangalore', 'Mumbai', 'Other'])
employment_type = st.sidebar.selectbox("Employment Type", ['Salaried', 'Unemployed', 'Self-Employed', 'Other'])

if st.sidebar.button("Predict Target", type="primary"):
    # Prepare input
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score],
        'city': [city],
        'employment_type': [employment_type]
    })
    
    # Transform categoricals
    input_data['city_encoded'] = city_encoder.transform(input_data['city'].fillna('Other'))
    input_data['employment_encoded'] = employment_encoder.transform(input_data['employment_type'].fillna('Other'))
    
    final_features = ['age', 'income', 'loan_amount', 'credit_score', 'city_encoded', 'employment_encoded']
    X_input = input_data[final_features]
    
    # Predict
    prediction = model.predict(X_input)[0]
    
    # Display results
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Target", f"${prediction:,.2f}")
    with col2:
        st.info(f"**Input Summary:**\nAge: {age}\nIncome: ${income:,.0f}\nLoan Amount: ${loan_amount:,.0f}\nCredit Score: {credit_score}\nCity: {city}\nEmployment: {employment_type}")

# Show data info
with st.expander("📊 Dataset Info"):
    df = pd.read_csv('knn_regression_dataset.csv')
    st.dataframe(df.head())
    st.metric("Dataset Size", f"{len(df):,}")
    st.metric("Target Range", f"${df['target'].min():.0f} - ${df['target'].max():.0f}")

st.markdown("---")
st.caption("Powered by KNN Regression (n_neighbors=5) | Train R² & Test R² displayed during training")

