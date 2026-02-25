import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# CARGA DEL DATASET (CONVERSIÓN SEGURA)
# =====================================================
@st.cache_data
def load_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    return X, y

X_full, y_full = load_data()

# =====================================================
# SIDEBAR DATASET
# =====================================================
st.sidebar.header("⚙ Configuración del Dataset")

sample_size = st.sidebar.slider(
    "Cantidad de muestras",
    2000,
    10000,
    5000,
    step=500
)

# AHORA sí recortamos correctamente
X = X_full[:sample_size]
y = y_full[:sample_size]

# =====================================================
# SIDEBAR MODELO
# =====================================================
st.sidebar.header("🤖 Configuración del Modelo")

modelo_nombre = st.sidebar.selectbox(
    "Selecciona el modelo",
    ("Logistic Regression", "KNN", "SVM", "Decision Tree")
)

test_size = st.sidebar.slider(
    "Tamaño del test (%)",
    10,
    40,
    20
) / 100

# =====================================================
# DIVISIÓN
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# MODELO
# =====================================================
if modelo_nombre == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif modelo_nombre == "KNN":
    k = st.sidebar.slider("Número de vecinos (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

elif modelo_nombre == "SVM":
    C = st.sidebar.slider("Parámetro C", 0.1, 5.0, 1.0)
    model = SVC(C=C, probability=True)

elif modelo_nombre == "Decision Tree":
    depth = st.sidebar.slider("Profundidad máxima", 2, 20, 5)
    model = DecisionTreeClassifier(max_depth=depth)

# =====================================================
# ENTRENAMIENTO
# =====================================================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =====================================================
# MÉTRICAS
# =====================================================
st.subheader("📊 Métricas")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

# =====================================================
# MATRIZ DE CONFUSIÓN
# =====================================================
st.subheader("🔍 Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, cmap="Blues", ax=ax_cm)
st.pyplot(fig_cm)

# =====================================================
# ROC MULTICLASE
# =====================================================
st.subheader("📈 Curvas ROC")

y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = model.predict_proba(X_test)

fig_roc, ax_roc = plt.subplots()

for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"Clase {i} (AUC={roc_auc:.2f})")

ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.legend(fontsize=7)
st.pyplot(fig_roc)

# =====================================================
# FRONTERA DECISIÓN PCA
# =====================================================
st.subheader("🧠 Frontera de decisión (PCA 2D)")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

model_pca = type(model)(**model.get_params())
model_pca.fit(X_train_pca, y_train)

h = 0.5
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, s=10)
st.pyplot(fig)

# =====================================================
# VISUALIZACIÓN DE IMÁGENES (SIN KEYERROR)
# =====================================================
st.subheader("🖼 Ejemplos de imágenes")

n_images = st.slider("Cantidad de imágenes a mostrar", 5, 25, 10)

fig_img, axes = plt.subplots(1, n_images, figsize=(15, 3))

for i in range(n_images):
    axes[i].imshow(X[i, :].reshape(28, 28), cmap="gray")
    axes[i].set_title(str(y[i]))
    axes[i].axis("off")

st.pyplot(fig_img)
