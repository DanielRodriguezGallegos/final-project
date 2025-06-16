
import streamlit as st
import matplotlib.pyplot as plt

st.title("⚙️ Hyperparameter Tuning")

st.markdown("""
Este gráfico representa los resultados de búsqueda de hiperparámetros.
Se probaron combinaciones de `C` (Regularización) y `max_iter` para el modelo de regresión logística.
""")

C_values = [0.1, 0.5, 1, 5, 10]
accuracy_scores = [0.86, 0.88, 0.89, 0.88, 0.87]

fig, ax = plt.subplots()
ax.plot(C_values, accuracy_scores, marker='o')
ax.set_xlabel("C (Regularización)")
ax.set_ylabel("Accuracy")
ax.set_title("Resultados de búsqueda de hiperparámetros")
st.pyplot(fig)

st.markdown("**Configuración final:** `C=1`, `max_iter=1000`")
