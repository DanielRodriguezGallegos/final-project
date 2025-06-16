
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=["text", "label"])
    df["content"] = df["text"]
    return df

df = load_data()
X = vectorizer.transform(df["content"])
y = df["label"]
y_pred = model.predict(X)

st.title("📈 Model Analysis and Justification")

st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.subheader("Error Analysis")
false_positives = df[(y == 0) & (y_pred == 1)].sample(1)["content"].values[0]
false_negatives = df[(y == 1) & (y_pred == 0)].sample(1)["content"].values[0]
st.markdown(f"**Ejemplo Falso Positivo:** {false_positives}")
st.markdown(f"**Ejemplo Falso Negativo:** {false_negatives}")
