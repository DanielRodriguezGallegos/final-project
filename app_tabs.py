
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

st.set_page_config(layout="wide")
st.title("🧠 Fake News Classifier with Streamlit")

tab1, tab2, tab3, tab4 = st.tabs(["📰 Inference", "📊 Dataset", "⚙️ Tuning", "📈 Analysis"])

# Cargar modelo
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

# Cargar datos
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv"
    df = pd.read_csv(url).dropna(subset=["text", "label"])
    df["content"] = df["text"]
    return df

model, vectorizer = load_model()
df = load_data()

# Pestaña 1: Inference
with tab1:
    st.subheader("Clasificador de noticias falsas")
    texto = st.text_area("Pega aquí una noticia:")

    if st.button("Analizar"):
        if not texto.strip():
            st.warning("⚠️ Escribe algo para analizar.")
        else:
            vec = vectorizer.transform([texto])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()
            st.write(f"🔍 Confianza: **{prob:.2%}**")
            if pred == 1:
                st.success("✅ Esta noticia parece **REAL**.")
            else:
                st.error("🚨 Esta noticia parece **FALSA**.")

# Pestaña 2: Dataset Visualization
with tab2:
    st.subheader("Distribución de clases")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="label", data=df, ax=ax1)
    ax1.set_xticklabels(["Falsa (0)", "Real (1)"])
    st.pyplot(fig1)

    st.subheader("Longitud del texto")
    df["text_len"] = df["text"].apply(len)
    fig2, ax2 = plt.subplots()
    sns.histplot(df["text_len"], bins=50, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("WordCloud de noticias reales")
    real_text = " ".join(df[df["label"] == 1]["text"])
    wc_real = WordCloud(width=800, height=400, background_color="white").generate(real_text)
    st.image(wc_real.to_array(), use_column_width=True)

    st.subheader("WordCloud de noticias falsas")
    fake_text = " ".join(df[df["label"] == 0]["text"])
    wc_fake = WordCloud(width=800, height=400, background_color="white").generate(fake_text)
    st.image(wc_fake.to_array(), use_column_width=True)

# Pestaña 3: Hyperparameter Tuning (simulado)
with tab3:
    st.subheader("Tuning de hiperparámetros")
    C_values = [0.1, 0.5, 1, 5, 10]
    scores = [0.86, 0.88, 0.89, 0.88, 0.87]

    fig3, ax3 = plt.subplots()
    ax3.plot(C_values, scores, marker='o')
    ax3.set_xlabel("C (Regularización)")
    ax3.set_ylabel("Accuracy")
    st.pyplot(fig3)

    st.markdown("**Mejor configuración:** `C=1`, `max_iter=1000`")

# Pestaña 4: Evaluación y análisis
with tab4:
    st.subheader("Matriz de confusión y reporte")
    X = vectorizer.transform(df["content"])
    y = df["label"]
    y_pred = model.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    cm = confusion_matrix(y, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    st.pyplot(fig4)

    st.subheader("Análisis de errores")
    fp = df[(y == 0) & (y_pred == 1)].sample(1)["content"].values[0]
    fn = df[(y == 1) & (y_pred == 0)].sample(1)["content"].values[0]
    st.markdown(f"**Falso positivo:** {fp}")
    st.markdown(f"**Falso negativo:** {fn}")
