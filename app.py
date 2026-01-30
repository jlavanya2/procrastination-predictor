# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_model():
    df = pd.read_csv("data/procrastination_behavior_dataset.csv")

    df = pd.get_dummies(
        df,
        columns=["task_type", "day_of_week"],
        drop_first=True
    )

    X = df.drop("procrastinated", axis=1)
    y = df["procrastinated"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)
    return model, X.columns

model, model_features = train_model()


# =========================
# APP TITLE
# =========================
st.set_page_config(page_title="Procrastination Predictor", layout="centered")

st.title("🧠 Procrastination Predictor")
st.write("Estimate your procrastination risk based on task context, emotional state, and recent social media usage.")

st.divider()

# =========================
# USER INPUTS
# =========================
task_type = st.selectbox(
    "What type of task are you planning?",
    ["Study", "Work", "Personal", "Fitness", "Creative"]
)

day_of_week = st.selectbox(
    "Day of the week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

task_priority = st.slider("Task Priority (1 = Low, 5 = High)", 1, 5, 3)
mood = st.slider("Current Mood (1 = Low, 5 = High)", 1, 5, 3)
energy = st.slider("Energy Level (1 = Low, 5 = High)", 1, 5, 3)
stress = st.slider("Stress Level (1 = Low, 5 = High)", 1, 5, 3)

used_social_media = st.radio(
    "Did you use social media in the last hour?",
    ["Yes", "No"]
)

post_social_feeling = st.slider(
    "How do you feel after using social media?",
    1, 5, 3
)

# =========================
# FEATURE ENGINEERING
# =========================
used_social_media_num = 1 if used_social_media == "Yes" else 0
is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

emotional_load = stress - energy
mental_readiness = (mood + energy) / 2
social_media_risk = int(used_social_media_num == 1 and post_social_feeling <= 3)
low_energy = int(energy <= 2)
high_stress = int(stress >= 4)

procrastination_risk_index = (
    emotional_load +
    social_media_risk * 2 +
    low_energy +
    high_stress
)

# =========================
# SHOW ENGINEERED SIGNALS
# =========================
st.divider()
st.subheader("🧠 Engineered Behavioral Signals")

st.write("Emotional Load:", emotional_load)
st.write("Mental Readiness:", mental_readiness)
st.write("Social Media Risk:", social_media_risk)
st.write("Procrastination Risk Index:", procrastination_risk_index)

# =========================
# BUILD MODEL INPUT
# =========================
input_data = pd.DataFrame({
    "task_priority": [task_priority],
    "mood": [mood],
    "energy": [energy],
    "stress": [stress],
    "used_social_media": [used_social_media_num],
    "post_social_feeling": [post_social_feeling],
    "emotional_load": [emotional_load],
    "mental_readiness": [mental_readiness],
    "social_media_risk": [social_media_risk],
    "low_energy": [low_energy],
    "high_stress": [high_stress],
    "is_weekend": [is_weekend],
    "procrastination_risk_index": [procrastination_risk_index]
})

# Add one-hot encoded task type & day
for col in model_features:

    if col not in input_data.columns:
        input_data[col] = 0

# Activate correct one-hot columns
task_col = f"task_type_{task_type.lower()}"
day_col = f"day_of_week_{day_of_week.lower()}"

if task_col in input_data.columns:
    input_data[task_col] = 1

if day_col in input_data.columns:
    input_data[day_col] = 1

# Ensure correct column order
input_data = input_data[model_features]


# =========================
# PREDICTION
# =========================
st.divider()

if st.button("🔮 Predict Procrastination"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Procrastination Risk\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Low Procrastination Risk\n\nProbability: {probability:.2f}")

    st.subheader("🔍 Why this prediction?")
    reasons = []

    if social_media_risk:
        reasons.append("Recent social media usage with emotional drain")
    if emotional_load > 1:
        reasons.append("Stress is higher than energy")
    if low_energy:
        reasons.append("Low energy levels")
    if high_stress:
        reasons.append("High stress levels")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• Emotional and behavioral signals appear balanced")

st.caption("Built using behavioral data science & machine learning")
