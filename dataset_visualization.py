
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Cargar dataset
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv"
    df = pd.read_csv(url)
    return df.dropna(subset=["text", "label"])

df = load_data()

# Título
st.title("📊 Dataset Visualization - Fake News")

# Distribución de clases
st.subheader("Distribución de clases")
fig1, ax1 = plt.subplots()
sns.countplot(x="label", data=df, ax=ax1)
ax1.set_xticklabels(["Falsa (0)", "Real (1)"])
ax1.set_ylabel("Número de muestras")
st.pyplot(fig1)

# Longitud del texto
st.subheader("Distribución de longitud del texto")
df["text_len"] = df["text"].apply(len)
fig2, ax2 = plt.subplots()
sns.histplot(df["text_len"], bins=50, kde=True, ax=ax2)
ax2.set_xlabel("Número de caracteres")
st.pyplot(fig2)

# WordCloud
st.subheader("Nube de palabras de noticias reales")
real_text = " ".join(df[df["label"] == 1]["text"].dropna())
wordcloud_real = WordCloud(width=800, height=400, background_color="white").generate(real_text)
st.image(wordcloud_real.to_array(), caption="Noticias reales", use_column_width=True)

st.subheader("Nube de palabras de noticias falsas")
fake_text = " ".join(df[df["label"] == 0]["text"].dropna())
wordcloud_fake = WordCloud(width=800, height=400, background_color="white").generate(fake_text)
st.image(wordcloud_fake.to_array(), caption="Noticias falsas", use_column_width=True)
