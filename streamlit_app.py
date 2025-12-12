import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Adaptive Sleep Architect", page_icon="🌙")

# --- TITLE & DESCRIPTION ---
st.title("🌙 Adaptive Sleep Cycle Architect")
st.write("""
This AI tool helps you optimize your sleep schedule based on your daily strain.
It analyzes **physical activity**, **stress levels**, and **biometrics** to recommend the optimal sleep duration.
""")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Ensure the CSV is in the same repository as this script
    try:
        df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'Sleep_health_and_lifestyle_dataset.csv' is in the repository.")
        return None

df = load_data()

if df is not None:
    # --- 2. PREPROCESSING ---
    # Filter: Learn only from "Good" nights (Quality >= 7)
    good_sleep_df = df[df['Quality of Sleep'] >= 7].copy()

    # Encoder for Gender
    le = LabelEncoder()
    good_sleep_df['Gender_Code'] = le.fit_transform(good_sleep_df['Gender'])

    # Feature Selection
    features = ['Age', 'Gender_Code', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
    target = 'Sleep Duration'

    X = good_sleep_df[features]
    y = good_sleep_df[target]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. TRAIN MODEL ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- 4. MODEL PERFORMANCE METRICS ---
    # Predictions for validation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with st.expander("📊 View Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f} hrs", help="Average error in hours")
        col2.metric("MSE (Mean Squared Error)", f"{mse:.2f}", help="Lower is better")
        col3.metric("R² Score (Accuracy)", f"{r2:.2f}", help="1.0 is perfect prediction")
        
        st.write("The model was trained on high-quality sleep data only to ensure optimal recommendations.")

    # --- 5. USER INPUT (SIDEBAR) ---
    st.sidebar.header("📝 Enter Your Daily Stats")

    user_gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    user_age = st.sidebar.slider("Age", 18, 60, 25)
    user_stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
    user_activity = st.sidebar.slider("Physical Activity (mins)", 0, 120, 30)
    user_steps = st.sidebar.number_input("Daily Steps", 0, 20000, 5000)
    user_hr = st.sidebar.number_input("Resting Heart Rate", 40, 100, 70)

    # --- 6. PREDICTION LOGIC ---
    if st.sidebar.button("Calculate Optimal Sleep"):
        # Encode Input
        gender_encoded = le.transform([user_gender])[0]
        
        # Prepare Array
        input_data = np.array([[user_age, gender_encoded, user_activity, user_stress, user_hr, user_steps]])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Formatting Time
        hours = int(prediction)
        minutes = int((prediction - hours) * 60)
        
        # --- DISPLAY RESULTS ---
        st.success(f"🛌 Recommended Sleep: **{hours} hours and {minutes} minutes**")
        
        # Contextual Advice
        if user_stress > 6:
            st.warning("⚠️ High stress detected. Your recommended sleep is higher to allow for mental recovery.")
        if user_activity > 60:
            st.info("💪 High physical activity detected. Deep sleep is crucial for muscle repair tonight.")

    # --- 7. VISUALIZATIONS ---
    st.divider()
    st.subheader("🔍 What impacts your sleep needs?")
    
    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=model.feature_importances_, y=features, palette='viridis', ax=ax)
    ax.set_title("Feature Importance in Sleep Prediction")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

else:
    st.warning("Data could not be loaded. Please check your file structure.")
