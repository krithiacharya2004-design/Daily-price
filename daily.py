import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
try:
    model = joblib.load('daily_price_model.joblib')
except Exception as e:
    model = None
    st.error(f"Error loading model: {e}")

# Define expected feature columns (must match training data)
feature_columns = ['feature1', 'feature2_cat_A', 'feature2_cat_B']

# Streamlit App Title
st.title('Daily Price Prediction App')

st.write("""
This app predicts the daily price based on the selected features.
""")

# Sidebar for user input
st.sidebar.header('Specify Input Features')

# User input function
def user_input_features():
    data = {}

    # Numerical input
    data['feature1'] = st.sidebar.number_input('Feature 1 (e.g., weight)', min_value=0.0, value=50.0)

    # Categorical input and one-hot encoding
    category = st.sidebar.selectbox('Feature 2 (Category)', ['A', 'B'])
    data['feature2_cat_A'] = 1 if category == 'A' else 0
    data['feature2_cat_B'] = 1 if category == 'B' else 0

    # Convert input to DataFrame
    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# Display the user inputs
st.subheader('User Input Features')
st.write(input_df)

# Make prediction
if model is not None and not input_df.empty and list(input_df.columns) == feature_columns:
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f"Predicted Modal Price: {prediction[0]:.2f}")
else:
    st.warning("Please provide valid inputs and ensure the model and features match.")

    
