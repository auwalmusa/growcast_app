import streamlit as st
import pandas as pd
from joblib import load

# Correctly load pre-trained model and scaler with joblib
scaler = load('growcast_scaler.joblib')  # Make sure to convert your scaler to '.joblib' format if not already
model = load('growcast_model.joblib')

# **Data Loading and Preparation**
def get_maize_data():
    data = pd.read_csv("maize_yield_prediction_dataset.csv")
    # Add any potential data cleaning steps here if needed 
    return data

# **Sidebar**
def add_sidebar(data):
    st.sidebar.header("Maize Growth Parameters")
    soil_ph = st.sidebar.slider("Soil pH", min_value=data['soilph'].min(), max_value=data['soilph'].max(), value=data['soilph'].mean())
    p2o5 = st.sidebar.slider("P2O5", min_value=data['p2o5'].min(), max_value=data['p2o5'].max(), value=data['p2o5'].mean())
    k2o = st.sidebar.slider("K2O", min_value=data['k2o'].min(), max_value=data['k2o'].max(), value=data['k2o'].mean())
    zn = st.sidebar.slider("Zinc (Zn)", min_value=data['zn'].min(), max_value=data['zn'].max(), value=data['zn'].mean())
    clay_content = st.sidebar.slider("Clay Content", min_value=data['claycontent'].min(), max_value=data['claycontent'].max(), value=data['claycontent'].mean())
    eca = st.sidebar.slider("Electrical Conductivity (ECa)", min_value=data['eca'].min(), max_value=data['eca'].max(), value=data['eca'].mean())
    draught_force = st.sidebar.slider("Draught Force", min_value=data['draughtforce'].min(), max_value=data['draughtforce'].max(), value=data['draughtforce'].mean())
    cone_index = st.sidebar.slider("Cone Index", min_value=data['coneindex'].min(), max_value=data['coneindex'].max(), value=data['coneindex'].mean())
    precipitation = st.sidebar.slider("Precipitation", min_value=data['precipitation'].min(), max_value=data['precipitation'].max(), value=data['precipitation'].mean())
    temperature = st.sidebar.slider("Temperature", min_value=data['temperature'].min(), max_value=data['temperature'].max(), value=data['temperature'].mean())

    return soil_ph, p2o5, k2o, zn, clay_content, eca, draught_force, cone_index, precipitation, temperature

# **Prediction Function**
def get_prediction(soil_ph, p2o5, k2o, zn, clay_content, eca, draught_force, cone_index, precipitation, temperature): 
    input_data = pd.DataFrame({
        'SoilPH': [soil_ph],
        'P2O5': [p2o5],
        'K2O': [k2o],
        'Zn': [zn],
        'ClayContent': [clay_content],
        'ECa': [eca],
        'DraughtFc': [draught_force],
        'Conelnde': [cone_index],
        'Precipitation': [precipitation],
        'Temperature': [temperature]
    })

    scaled_data = scaler.transform(input_data) 
    prediction = model.predict(scaled_data)[0]
    return prediction

# **Yield Range Logic**
def get_yield_class(predicted_yield):
    if predicted_yield >= 9.0:
        yield_class = "high"
    elif predicted_yield >= 6.0:
        yield_class = "medium"
    else:
        yield_class = "low"
    return yield_class

# **Main Application**
def main():
    st.set_page_config(page_title="GrowCast", page_icon="🌱", layout="wide")
    data = get_maize_data()  # Load the data

    # Sidebar inputs
    soil_ph, p2o5, k2o, zn, clay_content, eca, draught_force, cone_index, precipitation, temperature = add_sidebar(data) 

    st.title("GrowCast: Precision Yield Forecasting")
    st.write("This application forecasts maize yield based on various growth parameters using a precision agriculture model.")

    # You might want to call get_prediction here and display the results
    predicted_yield = get_prediction(soil_ph, p2o5, k
