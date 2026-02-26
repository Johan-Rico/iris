import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Iris Classifier - EAFIT AI", layout="wide")

st.title("Classification Playground: Dataset Iris")
st.markdown("""
Esta herramienta permite explorar dinámicamente cómo diferentes modelos de Machine Learning 
clasifican las especies de flores basándose en sus medidas morfológicas.
""")

# --- SIDEBAR: PARÁMETROS DEL MODELO ---
st.sidebar.header("Configuración del Modelo")
classifier_name = st.sidebar.selectbox("Selecciona el Clasificador", ("SVM", "Random Forest"))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C (Regularización)", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth (Profundidad)", 2, 15)
        n_estimators = st.sidebar.slider("N Estimators (Árboles)", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

# --- CARGA DE DATOS ---
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
df = X.copy()
df['species'] = [iris.target_names[i] for i in y]

# --- ENTRENAMIENTO ---
def get_classifier(clf_name, params):
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                    max_depth=params["max_depth"], random_state=123)
    return clf

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
clf = get_classifier(classifier_name, params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# --- LAYOUT PRINCIPAL: VISUALIZACIÓN ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Visualización de Datos")
    fig = px.scatter(df, x=iris.feature_names[0], y=iris.feature_names[1], 
                     color='species', title="Sépalos: Largo vs Ancho")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Métricas de Desempeño")
    st.text(f"Modelo seleccionado: {classifier_name}")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig_cm)

# --- INTERFAZ DE VALIDACIÓN (PREDICCIÓN) ---
st.divider()
st.subheader("Validación de Entrada vs Predicción")
st.write("Mueve los sliders para simular una nueva flor y ver la predicción del modelo en tiempo real.")

i_col1, i_col2, i_col3, i_col4 = st.columns(4)
s_length = i_col1.slider("Sepal Length", 4.0, 8.0, 5.4)
s_width = i_col2.slider("Sepal Width", 2.0, 4.5, 3.4)
p_length = i_col3.slider("Petal Length", 1.0, 7.0, 1.3)
p_width = i_col4.slider("Petal Width", 0.1, 2.5, 0.2)

# Predicción individual
input_data = [[s_length, s_width, p_length, p_width]]
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data) if hasattr(clf, "predict_proba") else None

st.info(f"**Resultado de la Predicción:** La flor es una **{iris.target_names[prediction[0]]}**")

if prediction_proba is not None:
    prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.bar_chart(prob_df.T)
