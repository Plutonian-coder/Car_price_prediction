import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained models and scaler
try:
    with open('linear_regression_model.pkl', 'rb') as f:
        linear_reg_model = pickle.load(f)
    with open('lasso_regression_model.pkl', 'rb') as f:
        lasso_reg_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please make sure 'linear_regression_model.pkl', 'lasso_regression_model.pkl', and 'scaler.pkl' are in the same directory.")
    st.stop()

st.title("Car Price Prediction App")

st.sidebar.header("Enter Car Details")

# Get user input
year = st.sidebar.number_input("Year", min_value=1990, max_value=2024, value=2015)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=50000)
owner_type = st.sidebar.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.sidebar.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission_type = st.sidebar.selectbox("Transmission Type", ['Manual', 'Automatic'])

# Create a dictionary from user input
user_data = {
    'year': year,
    'km_driven': km_driven,
    'owner': owner_type,
    'fuel': fuel_type,
    'seller_type': seller_type,
    'transmission': transmission_type
}

# Create a DataFrame from user input
user_df = pd.DataFrame([user_data])

# Apply one-hot encoding to categorical features
categorical_cols = ['owner', 'fuel', 'seller_type', 'transmission']

# Create a DataFrame to hold the one-hot encoded columns
encoded_user_df = pd.DataFrame()

for col in categorical_cols:
    # Get the unique categories from the original training data for each column
    # This assumes you have access to the original training data or its unique values for these columns
    # For simplicity in this example, we'll hardcode some common categories.
    # In a real application, you would load or derive these from your training data.
    if col == 'owner':
        categories = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
    elif col == 'fuel':
        categories = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
    elif col == 'seller_type':
        categories = ['Individual', 'Dealer', 'Trustmark Dealer']
    elif col == 'transmission':
        categories = ['Manual', 'Automatic']

    for category in categories:
        encoded_user_df[f'{col}_{category}'] = (user_df[col] == category).astype(int)

# Drop the original categorical columns from user_df
user_df = user_df.drop(categorical_cols, axis=1)

# Concatenate the original numerical columns and the encoded categorical columns
user_df = pd.concat([user_df, encoded_user_df], axis=1)


# Ensure the order of columns matches the training data features
# This is crucial for correct prediction. You would typically save the column order
# from your training data features (X_train) and use it here.
expected_columns = ['year', 'km_driven', 'owner_First Owner', 'owner_Fourth & Above Owner',
                    'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner',
                    'fuel_CNG', 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',
                    'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
                    'transmission_Automatic', 'transmission_Manual']

# Add missing columns with a value of 0 and ensure the order is correct
for col in expected_columns:
    if col not in user_df.columns:
        user_df[col] = 0

# Reindex the user_df to ensure column order matches expected_columns
user_df = user_df[expected_columns]


# Scale the user_df DataFrame using the fitted scaler
user_df_scaled = scaler.transform(user_df)
user_df_scaled = pd.DataFrame(user_df_scaled, columns=user_df.columns)


if st.sidebar.button("Predict Selling Price"):
    # Make predictions
    linear_reg_prediction = linear_reg_model.predict(user_df_scaled)
    lasso_reg_prediction = lasso_reg_model.predict(user_df_scaled)

    st.subheader("Predicted Selling Price")
    st.write(f"Linear Regression Prediction: ₹{linear_reg_prediction[0]:,.2f}")
    st.write(f"Lasso Regression Prediction: ₹{lasso_reg_prediction[0]:,.2f}")