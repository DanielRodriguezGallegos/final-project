
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Cargar modelo y vectorizador
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Cargar datos para métricas
df = pd.read_csv("https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv")
df = df.dropna()
X = vectorizer.transform(df["text"])
y = df["label"]
y_pred = model.predict(X)

# Métricas y visualización
st.title("Análisis del modelo")

st.header("Reporte de clasificación")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.header("Matriz de confusión")
conf_matrix = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Matriz de confusión")
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['Falso', 'Real'])
ax.set_yticklabels(['Falso', 'Real'])
st.pyplot(fig)
