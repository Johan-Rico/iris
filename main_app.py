import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(page_title="Iris Classifier Edu", layout="wide")

# --- ENCABEZADO ---
st.title("🌿 Clasificador Dinámico: Dataset Iris")
st.markdown("""
Esta aplicación es una herramienta pedagógica para entender cómo funciona un modelo de **Machine Learning** (Random Forest) 
utilizando el famoso conjunto de datos de flores Iris.
""")

# --- CARGA DE DATOS ---
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# --- SIDEBAR: CONFIGURACIÓN DINÁMICA ---
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.info("Ajusta los hiperparámetros para ver cómo cambia el rendimiento.")

test_size = st.sidebar.slider("Tamaño del set de prueba (%)", 10, 50, 20) / 100
n_estimators = st.sidebar.slider("Número de árboles (n_estimators)", 10, 200, 100)
max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)

# --- ENTRENAMIENTO ---
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# --- INTERFAZ PRINCIPAL ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Análisis del Dataset")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Gráfico PCA para visualización pedagógica
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig_pca = px.scatter(
        components, x=0, y=1, color=df.target.astype(str),
        labels={'0': 'Componente 1', '1': 'Componente 2', 'color': 'Especie'},
        title="Reducción de dimensionalidad (PCA)",
        color_discrete_map={'0': 'setosa', '1': 'versicolor', '2': 'virginica'}
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with col2:
    st.subheader("📈 Desempeño del Modelo")
    st.metric("Precisión (Accuracy)", f"{accuracy:.2%}")
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    st.pyplot(fig_cm)
    
    # Importancia de características
    st.write("**Importancia de las variables:**")
    feat_importances = pd.Series(clf.feature_importances_, index=iris.feature_names)
    st.bar_chart(feat_importances)

# --- VALIDACIÓN DE ENTRADA CONTRA PREDICCIÓN ---
st.divider()
st.subheader("🔍 Probador de Predicciones")
st.write("Mueve los sliders para simular una flor y ver la predicción en tiempo real.")

c1, c2, c3, c4 = st.columns(4)
with c1: s_len = st.slider("Sepal Length", 4.3, 7.9, 5.4)
with c2: s_wid = st.slider("Sepal Width", 2.0, 4.4, 3.4)
with c3: p_len = st.slider("Petal Length", 1.0, 6.9, 1.3)
with c4: p_wid = st.slider("Petal Width", 0.1, 2.5, 0.2)

user_input = np.array([[s_len, s_wid, p_len, p_wid]])
prediction = clf.predict(user_input)
prediction_proba = clf.predict_proba(user_input)

# Resultado de la predicción
st.write(f"### Resultado: La flor es una **Iris-{target_names[prediction[0]].upper()}**")

# Probabilidades en barra
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write("Confianza del modelo por clase:")
st.bar_chart(prob_df.T)

st.markdown("""
---
**Nota Pedagógica:** * **Accuracy:** Indica qué tan bien clasifica el modelo en general.
* **Matriz de Confusión:** Permite ver en qué especies específicas se está confundiendo el modelo.
* **Importancia de Variables:** Muestra que, usualmente, las medidas del pétalo son más determinantes que las del sépalo.
""")
