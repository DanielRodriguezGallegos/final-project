
import streamlit as st
import pickle

# Cargar modelo y vectorizador
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Interfaz de usuario
st.title("📰 Fake News Detector")
st.write("Pega una noticia o artículo para predecir si es real o falsa.")

text = st.text_area("Texto de la noticia:")

if st.button("Analizar"):
    if not text.strip():
        st.warning("⚠️ Por favor, escribe algo.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()

        st.write(f"🔍 Confianza del modelo: **{prob:.2%}**")
        if pred == 1:
            st.success("✅ Esta noticia parece **REAL**.")
        else:
            st.error("🚨 Esta noticia parece **FALSA**.")
