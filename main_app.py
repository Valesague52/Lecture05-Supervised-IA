import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(page_title="Clasificación MNIST", layout="wide")

st.title("🔢 Clasificación de Dígitos - MNIST")
st.markdown(
    "Comparación de modelos con métricas, curvas ROC y frontera de decisión usando PCA."
)

# =====================================================
# CARGA DEL DATASET
# =====================================================
@st.cache_data
def load_data():
    with st.spinner('Cargando dataset MNIST...'):
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
    return X, y

X_full, y_full = load_data()

# =====================================================
# SIDEBAR - CONFIGURACIÓN
# =====================================================
st.sidebar.header("⚙ Configuración")

# Configuración del dataset
st.sidebar.subheader("📊 Dataset")
sample_size = st.sidebar.slider(
    "Cantidad de muestras",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=500
)

# Tomar una muestra del dataset
indices = np.random.choice(len(X_full), sample_size, replace=False)
X = X_full[indices]
y = y_full[indices]

# Configuración del modelo
st.sidebar.subheader("🤖 Modelo")
modelo_nombre = st.sidebar.selectbox(
    "Selecciona el modelo",
    ("Regresión Logística", "KNN", "SVM", "Árbol de Decisión")
)

# Parámetros específicos del modelo
if modelo_nombre == "KNN":
    k = st.sidebar.slider("Número de vecinos (K)", 1, 15, 5)
elif modelo_nombre == "SVM":
    C = st.sidebar.slider("Parámetro C", 0.1, 5.0, 1.0, 0.1)
elif modelo_nombre == "Árbol de Decisión":
    depth = st.sidebar.slider("Profundidad máxima", 2, 20, 10)

# Configuración del split
test_size = st.sidebar.slider(
    "Tamaño del test (%)",
    10, 40, 20
) / 100

# =====================================================
# DIVISIÓN Y ESCALADO
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# SELECCIÓN DEL MODELO
# =====================================================
if modelo_nombre == "Regresión Logística":
    model = LogisticRegression(max_iter=1000, random_state=42)
elif modelo_nombre == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)
elif modelo_nombre == "SVM":
    model = SVC(C=C, probability=True, random_state=42)
elif modelo_nombre == "Árbol de Decisión":
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

# =====================================================
# ENTRENAMIENTO
# =====================================================
with st.spinner('Entrenando modelo...'):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Probabilidades para ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)
    else:
        y_score = None

# =====================================================
# MÉTRICAS PRINCIPALES
# =====================================================
st.header("📊 Métricas de Desempeño")

col1, col2, col3, col4 = st.columns(4)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1-score", f"{f1:.3f}")

# =====================================================
# MATRIZ DE CONFUSIÓN
# =====================================================
st.header("🔍 Matriz de Confusión")

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm, 
            xticklabels=range(10), yticklabels=range(10))
ax_cm.set_xlabel('Predicción')
ax_cm.set_ylabel('Valor Real')
ax_cm.set_title('Matriz de Confusión')
st.pyplot(fig_cm)
plt.close()

# =====================================================
# CURVAS ROC
# =====================================================
if y_score is not None:
    st.header("📈 Curvas ROC (One-vs-Rest)")
    
    y_test_bin = label_binarize(y_test, classes=range(10))
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, label=f'Clase {i} (AUC = {roc_auc:.2f})')
    
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador aleatorio')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax_roc.set_title('Curvas ROC por Clase')
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)
    st.pyplot(fig_roc)
    plt.close()

# =====================================================
# FRONTERA DE DECISIÓN CON PCA
# =====================================================
st.header("🧠 Frontera de Decisión (PCA 2D)")

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Entrenar modelo con datos reducidos
model_pca = type(model)(**model.get_params())
if hasattr(model_pca, 'probability'):
    model_pca.set_params(probability=True)
model_pca.fit(X_train_pca, y_train)

# Crear malla para visualización
h = 0.5
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Predecir en la malla
Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualizar
fig_pca, ax_pca = plt.subplots(figsize=(12, 10))

# Frontera de decisión
ax_pca.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')

# Puntos de datos
scatter = ax_pca.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                        c=y_train, cmap='tab10', s=10, alpha=0.8)

ax_pca.set_xlabel('Primer Componente Principal')
ax_pca.set_ylabel('Segundo Componente Principal')
ax_pca.set_title(f'Frontera de Decisión con PCA - {modelo_nombre}')
plt.colorbar(scatter, ax=ax_pca, label='Dígito')
st.pyplot(fig_pca)
plt.close()

# =====================================================
# VISUALIZACIÓN DE IMÁGENES
# =====================================================
st.header("🖼 Ejemplos de Imágenes MNIST")

n_images = st.slider("Número de imágenes a mostrar", 5, 25, 10)

# Crear figura con subplots
fig_img, axes = plt.subplots(2, (n_images + 1) // 2, figsize=(15, 6))
axes = axes.flatten()

for i in range(n_images):
    # Tomar una imagen aleatoria del conjunto original
    idx = np.random.randint(0, len(X))
    img = X[idx].reshape(28, 28)
    
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f'Dígito: {y[idx]}')
    axes[i].axis("off")

# Ocultar axes vacíos
for i in range(n_images, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
st.pyplot(fig_img)
plt.close()

# =====================================================
# INFORMACIÓN ADICIONAL
# =====================================================
with st.expander("ℹ️ Información del Dataset MNIST"):
    st.markdown("""
    ### Sobre MNIST
    - **Descripción**: Base de datos de dígitos escritos a mano
    - **Imágenes**: 70,000 imágenes en escala de grises de 28x28 píxeles
    - **Clases**: 10 dígitos (0-9)
    - **Características**: 784 píxeles por imagen (28x28)
    
    ### Modelos disponibles
    - **Regresión Logística**: Modelo lineal con regularización
    - **KNN**: Clasificación por vecinos cercanos
    - **SVM**: Máquinas de vectores de soporte
    - **Árbol de Decisión**: Estructura jerárquica de decisiones
    
    ### Métricas mostradas
    - **Accuracy**: Proporción de predicciones correctas
    - **Precision**: Precisión ponderada por clase
    - **Recall**: Sensibilidad ponderada por clase
    - **F1-score**: Media armónica de precision y recall
    """)
