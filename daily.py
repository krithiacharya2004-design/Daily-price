import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Make sure the path to your joblib file is correct
try:
    model = joblib.load('/content/daily_price_model.joblib')
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'daily_price_model.joblib' is in the correct path (/content/).")
    st.stop() # Stop the app if the model is not found


try:
    # Assuming X_train is a global variable from the previous notebook state
    feature_columns = X_train.columns.tolist()
    st.success(f"Loaded {len(feature_columns)} feature columns from X_train.")
except NameError:
    st.error("X_train not found. Please run the data preprocessing and splitting steps first.")
    st.stop()


# App title
st.title('Daily Price Prediction App')

st.write("""
This app predicts the daily price based on the selected features.
""")

# Sidebar for user input features
st.sidebar.header('Specify Input Features')

# Function to get user input
def user_input_features():
    
    # Initialize input data with zeros, matching the structure of X_train
    input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    st.sidebar.write("Enter values for the features:")

    st.sidebar.write("Note: For a production app, replace text inputs with appropriate widgets like selectboxes for categorical features.")

   
    st.sidebar.subheader("Example Input (replace with appropriate widgets):")

    st.write("Input feature section needs to be customized based on your data's columns and types.")
    st.write("You need to create input widgets (like st.selectbox for categories and st.number_input for numerical features) that match the one-hot encoded columns your model was trained on.")

    # Return the dataframe with user inputs
    return input_data

# Get user input features
input_df = user_input_features()

# Display the input features (optional)
st.subheader('User Input features')
st.write(input_df)

# Make prediction (only if input_df has the correct structure and model is loaded)
if model is not None and not input_df.empty and len(input_df.columns) == len(feature_columns):
    prediction = model.predict(input_df)

    st.subheader('Prediction')
    st.write(f"Predicted Modal Price: {prediction[0]:.2f}")
else:
    st.write("Please provide input features in the sidebar.")
    if model is None:
        st.warning("Model not loaded.")
    if input_df.empty:
         st.warning("Input features not provided.")
    if len(input_df.columns) != len(feature_columns):
        st.warning(f"Input features ({len(input_df.columns)} columns) do not match model features ({len(feature_columns)} columns).")
