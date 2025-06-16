
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Cargar el dataset desde Hugging Face (versión directa)
df = pd.read_csv("https://huggingface.co/datasets/mrm8488/fake-news/resolve/main/fake_news.csv")

# Preprocesar: usar solo la columna 'text'
df["content"] = df["text"].fillna("")
df = df[["content", "label"]].dropna()

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)

# Vectorizar texto
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluar modelo
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Guardar modelo y vectorizador
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
