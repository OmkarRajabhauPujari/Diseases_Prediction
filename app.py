import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset to get symptom names
data = pd.read_csv("dataset.csv")
symptoms = list(data.columns[:-1])  # exclude prognosis

st.set_page_config(page_title="Disease Prediction", page_icon="🏥")

st.title("🏥 AI Disease Prediction System")
st.write("Select your symptoms and get prediction")

# Multi-select symptoms
selected_symptoms = st.multiselect("Choose Symptoms:", symptoms)

# Prediction button
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom")
    else:
        input_data = [0] * len(symptoms)

        for symptom in selected_symptoms:
            index = symptoms.index(symptom)
            input_data[index] = 1

        # Prediction
        prediction = model.predict([input_data])[0]

        # Probabilities (Top 3)
        probs = model.predict_proba([input_data])[0]
        top3 = probs.argsort()[-3:][::-1]

        st.success(f"🩺 Predicted Disease: {prediction}")

        st.subheader("📊 Top Predictions:")
        for i in top3:
            st.write(f"{model.classes_[i]}: {round(probs[i]*100,2)}%")

        # Basic precautions
        st.subheader("💡 Precautions:")
        st.write("- Consult a doctor")
        st.write("- Take proper rest")
        st.write("- Stay hydrated")