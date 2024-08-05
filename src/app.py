import streamlit as st
import pandas as pd
from pickle import load
import os
# Load the pre-trained model
# Obtener el directorio actual del script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construir rutas relativas a partir del directorio actual
model_path = os.path.join(current_dir, "models", "random_forest_regressor_default_42.sav")

# Cargar el modelo y el vectorizador usando rutas relativas
model = load(open(model_path, "rb"))

# Class dictionary for prediction
class_dict = {
    "0": "Negative for diabetes",
    "1": "Positive for diabetes"
}

# Title of the app
st.title("Diabetes Prediction")

# Create sliders for each column in the dataset
pregnancies = st.slider("Number of pregnancies", min_value=0, max_value=20, step=1)
glucose = st.slider("Glucose level", min_value=0, max_value=200, step=1)
insulin = st.slider("Insulin level", min_value=0, max_value=800, step=1)
bmi = st.slider("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.slider("Age", min_value=0, max_value=120, step=1)

# Make prediction when the user clicks the button
if st.button("Predict"):
    # Get the model prediction
    prediction = str(model.predict([[pregnancies, glucose, insulin, bmi, dpf, age]])[0])
    pred_class = class_dict[prediction]
    
    # Display the prediction
    st.write("Prediction:", pred_class)