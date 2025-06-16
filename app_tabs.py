import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Cargar modelo y vectorizador
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Cargar datos para análisis
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("Análisis del Modelo")
st.write("Matriz de confusión y reporte de clasificación")

# Preprocesamiento
X = vectorizer.transform(df["text"].astype(str))
y_true = df["label"]
y_pred = model.predict(X)

# Reporte de clasificación
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.subheader("Reporte de clasificación")
st.dataframe(df_report)

# Matriz de confusión
st.subheader("Matriz de confusión")
conf_matrix = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_xticklabels(['Falso (0)', 'Real (1)'])
ax.set_yticklabels(['Falso (0)', 'Real (1)'])
st.pyplot(fig)
