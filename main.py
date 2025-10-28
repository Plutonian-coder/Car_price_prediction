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


# Ensure the order of columns matches the training data features.
# Prefer to use the feature names recorded on the fitted scaler (if available).
try:
    if hasattr(scaler, 'feature_names_in_'):
        expected_columns = list(scaler.feature_names_in_)
    else:
        raise AttributeError
except Exception:
    # Fallback: this should match the column order used during training. Keep
    # the hardcoded list as a fallback if the scaler wasn't fitted from a
    # DataFrame (older sklearn versions or a plain numpy-based scaler).
    expected_columns = ['year', 'km_driven', 'owner_First Owner', 'owner_Fourth & Above Owner',
                        'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner',
                        'fuel_CNG', 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',
                        'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
                        'transmission_Automatic', 'transmission_Manual']

# Add any missing columns that the scaler expects with a sensible default of 0.
for col in expected_columns:
    if col not in user_df.columns:
        user_df[col] = 0

# Reindex the user_df to ensure column order matches expected_columns
user_df = user_df[expected_columns]


# Scale the user_df DataFrame using the fitted scaler
user_df_scaled = scaler.transform(user_df)
# The scaler may have been fitted including or excluding certain columns (in
# this repo the scaler was fitted including the target 'selling_price'). The
# transformed DataFrame uses the scaler's feature order (we used
# scaler.feature_names_in_ to build `user_df`), so make a DataFrame with those
# column names.
scaler_feature_names = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else list(user_df.columns)
user_df_scaled = pd.DataFrame(user_df_scaled, columns=scaler_feature_names)

# Models expect a specific set of features. Prefer the linear model's
# feature_names_in_ (both models were trained on the same features here).
if hasattr(linear_reg_model, 'feature_names_in_'):
    model_features = list(linear_reg_model.feature_names_in_)
else:
    # Fallback: use the intersection of scaler columns and current user_df
    model_features = [c for c in scaler_feature_names if c in user_df.columns]

# Ensure model input contains exactly the features expected by the model.
# Add any missing model features with zeros, and drop any extras (like
# 'selling_price' which was present for the scaler but not used by the models).
for col in model_features:
    if col not in user_df_scaled.columns:
        user_df_scaled[col] = 0
model_input = user_df_scaled[model_features]


if st.sidebar.button("Predict Selling Price"):
    # Make predictions using only the features the models expect
    linear_reg_prediction = linear_reg_model.predict(model_input)
    lasso_reg_prediction = lasso_reg_model.predict(model_input)
    # The models appear to predict the target in the scaled space because
    # during training the scaler was fit including the target column
    # 'selling_price'. We therefore inverse-transform the model outputs to
    # get prices on the original scale using the scaler's mean and scale for
    # the selling_price feature.
    if hasattr(scaler, 'feature_names_in_') and 'selling_price' in list(scaler.feature_names_in_):
        sell_idx = list(scaler.feature_names_in_).index('selling_price')
        sell_mean = scaler.mean_[sell_idx]
        sell_scale = scaler.scale_[sell_idx]

        linear_actual = linear_reg_prediction[0] * sell_scale + sell_mean
        lasso_actual = lasso_reg_prediction[0] * sell_scale + sell_mean
    else:
        # If we don't have selling_price in the scaler, assume model outputs
        # are already on the original scale.
        linear_actual = linear_reg_prediction[0]
        lasso_actual = lasso_reg_prediction[0]

    # Currency display: convert INR to USD. Provide a sidebar input so the
    # user can adjust the exchange rate; default is 0.012 (approx 1 INR = 0.012 USD).
    inr_to_usd = st.sidebar.number_input('INR → USD rate', value=0.012, format="%.6f")
    linear_usd = linear_actual * inr_to_usd
    lasso_usd = lasso_actual * inr_to_usd

    st.subheader("Predicted Selling Price")
    st.write(f"Linear Regression Prediction: ${linear_usd:,.2f} (≈ ₹{linear_actual:,.0f})")
    st.write(f"Lasso Regression Prediction: ${lasso_usd:,.2f} (≈ ₹{lasso_actual:,.0f})")