import streamlit as st
import numpy as np
import joblib

# Load the trained model and mapping
model = joblib.load("newrandom_forest_model.pkl")
class_mapping = joblib.load("class_mapping.pkl")

st.title("Dry Beans Classifier 🌱")
st.write("Enter the features of the bean to predict its class.")

# Input sliders
area = st.number_input("Area", value=1.0, format="%.6f")
eccentricity = st.number_input("Eccentricity", value=0.8, format="%.6f")
extent = st.number_input("Extent", value=0.7, format="%.6f")
solidity = st.number_input("Solidity", value=0.9, format="%.6f")
roundness = st.number_input("Roundness", value=0.7, format="%.6f")
shape_factor_4 = st.number_input("Shape Factor 4", value=0.5, format="%.6f")

if st.button("Predict Bean Type"):
    # Create feature array from inputs
    X = np.array([[area, eccentricity, extent, solidity, roundness, shape_factor_4]])
    
    # Predict
    predicted_class_code = model.predict(X)[0]
    predicted_label = class_mapping[predicted_class_code]

    # Display result
    st.success(f"🌱 Predicted Bean Type: {predicted_label}")