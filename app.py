
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('XGBoost.pkl')
scaler = joblib.load('scaler_2.pkl')

# Get the feature names from the scaler
feature_names = scaler.feature_names_in_

st.title('Property Price Predictor')

st.write("""
### Welcome to the Property Price Predictor!
This app uses a Random Forest model to predict property prices based on various features.
Please input the details of the property below.
""")

# Create input fields for each feature
area_type = st.selectbox('Area Type', ['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area'])
availability = st.selectbox('Availability', ['Ready To Move'])
location = st.selectbox('Location', ['Whitefield', 'Sarjapur Road', 'Electronic City', 'Uttarahalli', 'Yelahanka'])
bath = st.number_input('Number of Bathrooms', min_value=1, max_value=6, value=2)
balcony = st.number_input('Number of Balconies', min_value=0, max_value=5, value=1)
total_sqft = st.number_input('Total Square Feet', min_value=200, max_value=4000, value=1000)
house_size = st.number_input('House Size (BHK)', min_value=1, max_value=6, value=2)
house_type = st.selectbox('House Type', ['BHK', 'Bedroom', 'RK'])

# Create a dictionary of inputs
input_dict = {
    'area_type': area_type,
    'availability': availability,
    'location': location,
    'bath': bath,
    'balcony': balcony,
    'total_sqft': total_sqft,
    'house_size': house_size,
    'house_type': house_type
}

# Create a DataFrame from the input
input_df = pd.DataFrame([input_dict])

# Preprocess the input data
input_df['area_type'] = input_df['area_type'].map({'Super built-up Area': 4, 'Built-up Area': 3, 'Plot Area': 2, 'Carpet Area': 1})
input_df['availability'] = input_df['availability'].map({'Ready To Move': 1, 'Not Ready': 0})
input_df['house_type'] = input_df['house_type'].map({'BHK': 0, 'Bedroom': 1, 'RK': 2})

# For location, you might want to use the mean encoding from your training data
# Here, we're using a placeholder value of 0 for simplicity
input_df['location'] = 0

# Ensure the input DataFrame has the same columns in the same order as the training data
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale the input data
input_scaled = scaler.transform(input_df)

if st.button('Predict Price'):
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Since we used log transformation on the price, we need to apply exp to get the actual price
    predicted_price = np.exp(prediction[0])
    
    st.success(f'The predicted price is ₹{predicted_price:,.2f} Lakhs')

st.write("""
### Note:
This model uses historical data and may not account for recent market changes. 
Always consult with a real estate professional for accurate property valuations.
""")