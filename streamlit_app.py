

st.title('Machine Learning')

st.write('Mini Hackathon')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- 1. Load Data ---
# Ensure the CSV is in the same folder as this script
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# --- 2. Smart Preprocessing ---
# FILTER: We only want to learn from "Good" nights (Quality >= 7)
# This teaches the AI: "Under these conditions, X hours resulted in good rest."
good_sleep_df = df[df['Quality of Sleep'] >= 7].copy()

# ENCODE: Convert 'Male'/'Female' to numbers (Male=1, Female=0 or vice versa)
le = LabelEncoder()
good_sleep_df['Gender_Code'] = le.fit_transform(good_sleep_df['Gender'])

# SELECT FEATURES: These are the "Inputs" from your daily life
features = ['Age', 'Gender_Code', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']
target = 'Sleep Duration' # The "Output" we want to predict

X = good_sleep_df[features]
y = good_sleep_df[target]

# --- 3. Train the Architect ---
# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Random Forest (Great for detecting non-linear patterns like "High stress + Low activity")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. Evaluate Performance ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Model Accuracy Check:")
print(f"Average Error: {mae:.2f} hours (approx {int(mae*60)} minutes)")
print("-" * 30)

# --- 5. The "Architect" Function (User Interface) ---
def get_optimal_sleep(age, gender, activity_mins, stress_level, heart_rate, steps):
    """
    Input your daily stats -> Get personalized sleep recommendation.
    """
    # Convert gender to code using the same encoder as training
    try:
        gender_code = le.transform([gender])[0]
    except ValueError:
        return "Error: Gender must be 'Male' or 'Female'"

    # Create input array
    user_input = pd.DataFrame([[age, gender_code, activity_mins, stress_level, heart_rate, steps]], columns=features)
    
    # Predict
    recommended_hours = model.predict(user_input)[0]
    
    # Format nice output
    hours = int(recommended_hours)
    minutes = int((recommended_hours - hours) * 60)
    
    return f"Based on your day, aim for: {hours}h {minutes}m of sleep."

# --- 6. Test It! (Sample Scenarios) ---
# Scenario A: High Stress, Low Activity (The "Burnout" Day)
print("\nScenario 1: High Stress (8/10), Low Activity (30 mins)")
print(get_optimal_sleep(age=25, gender='Male', activity_mins=30, stress_level=8, heart_rate=80, steps=3000))

# Scenario B: High Activity, Low Stress (The "Athlete" Day)
print("\nScenario 2: Low Stress (4/10), High Activity (90 mins)")
print(get_optimal_sleep(age=25, gender='Male', activity_mins=90, stress_level=4, heart_rate=65, steps=10000))

# --- 7. (Optional) Visualize What Matters ---
plt.figure(figsize=(10, 5))
sns.barplot(x=model.feature_importances_, y=features, palette='viridis')
plt.title('What Factors Drive Your Sleep Needs?')
plt.xlabel('Importance Score')
plt.show()
