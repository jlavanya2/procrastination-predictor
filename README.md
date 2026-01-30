# 🧠 Procrastination Predictor – Behavioral ML Project

An end-to-end **data science and machine learning project** that predicts procrastination risk using **emotional state, task context, and recent social media usage**, and delivers results through an interactive **Streamlit web app**.

---

## 📌 Project Motivation

Procrastination is often blamed on laziness or lack of motivation, but research and everyday experience suggest it is deeply influenced by:
- Emotional state (stress, energy, mood)
- Task pressure
- Short-term digital behavior (especially social media usage)

This project explores whether **procrastination can be predicted and explained** using measurable behavioral signals collected just before starting a task.

---

## 🎯 Problem Statement

> Can procrastination be predicted using emotional, behavioral, and digital usage signals?

Key questions explored:
- Does recent social media usage increase procrastination risk?
- Is emotional overload more influential than task priority?
- Can machine learning models explain *why* procrastination happens?

---

## 📊 Dataset Overview

- **Type:** Behavioral & contextual dataset  
- **Source:** Synthetic dataset designed to mirror real Google Form responses  
- **Rows:** 200  
- **Target Variable:** `procrastinated` (Yes / No)

### Key Features

| Category | Features |
|--------|---------|
| Task Context | task_type, task_priority |
| Time Context | day_of_week, is_weekend |
| Emotional State | mood, energy, stress |
| Digital Behavior | used_social_media, post_social_feeling |
| Engineered Signals | emotional_load, mental_readiness, social_media_risk, procrastination_risk_index |

---

## 🧠 Feature Engineering (Core Contribution)

To better model human behavior, raw inputs were transformed into psychologically meaningful features:

- **Emotional Load:** `stress - energy`
- **Mental Readiness:** average of mood and energy
- **Social Media Risk:** social media usage combined with emotional drain
- **Procrastination Risk Index:** composite behavioral risk score

These engineered features significantly improved both **model performance and interpretability**.

---

## 🤖 Machine Learning Models Used

| Model | Purpose |
|-----|--------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Rule-based behavioral explanation |
| Random Forest | High-performance ensemble model |

**Random Forest** achieved the best balance between precision and recall.

---

## 🔍 Key Insights

- Emotional factors (stress, energy) are stronger predictors than task metadata
- Recent social media usage combined with emotional drain significantly increases procrastination risk
- Procrastination follows **predictable behavioral patterns**, not randomness

---

## 🖥️ Interactive Streamlit App

The project includes a **Streamlit web app** that:
- Accepts real-time user inputs
- Performs feature engineering inside the app
- Predicts procrastination risk using a trained ML model
- Explains predictions in human-readable terms

This turns the project into a **usable behavioral analytics product**, not just an offline analysis.

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  
- Git & GitHub  

---

## 📁 Project Structure

