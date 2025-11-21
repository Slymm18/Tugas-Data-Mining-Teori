import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =============================================
# 1. Konfigurasi Halaman
# =============================================
st.set_page_config(
    page_title="Anime Popularity Prediction",
    page_icon="🎌",
    layout="centered"
)

st.title("🎌 Anime Popularity Prediction (Ensemble Model)")
st.write("Gunakan aplikasi ini untuk memprediksi apakah suatu anime termasuk **Populer** atau **Tidak Populer** berdasarkan model Ensemble (Naive Bayes + Random Forest).")


# =============================================
# 2. Load Model .pkl
# =============================================
@st.cache_resource
def load_models():
    try:
        model_nb = joblib.load("model_naive_bayes.pkl")
        model_rf = joblib.load("model_random_forest.pkl")
        ensemble = joblib.load("model_ensemble_voting(1).pkl")
        features = joblib.load("model_features.pkl")
        return model_nb, model_rf, ensemble, features
    except:
        return None, None, None, None


model_nb, model_rf, ensemble_model, features = load_models()

if not all([model_nb, model_rf, ensemble_model, features]):
    st.error("⚠️ Model belum ditemukan! Upload file .pkl ke folder yang sama dengan main.py.")
    st.stop()


# =============================================
# 3. Sidebar – Tampilkan Akurasi Model
# (Opsional: ganti angka sesuai hasil training)
# =============================================
st.sidebar.header("📊 Model Accuracy")

st.sidebar.success("Naive Bayes: 0.86")
st.sidebar.success("Random Forest: 0.94")
st.sidebar.success("Ensemble Voting: 0.96")


# =============================================
# 4. Input Form untuk User
# =============================================
st.subheader("⬇️ Masukkan Data Anime")

members = st.number_input("Jumlah Members (MyAnimeList)", min_value=0, value=500000, step=1000)
score = st.number_input("Score", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
position = st.number_input("Ranking Position", min_value=1, value=300, step=1)
popularity = st.number_input("Popularity Index", min_value=1, value=200, step=1)

input_data = pd.DataFrame(
    [[members, score, position, popularity]],
    columns=features
)


# =============================================
# 5. Predict Button
# =============================================
st.write("---")

if st.button("🔮 Prediksi Popularitas Anime"):

    try:
        pred = ensemble_model.predict(input_data)[0]

        label = "Populer" if pred == 1 else "Tidak Populer"
        color = "🟢" if pred == 1 else "🔴"

        st.subheader("Hasil Prediksi:")
        st.success(f"{color} Anime ini diprediksi: **{label}**")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

