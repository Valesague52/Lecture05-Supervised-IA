import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(page_title="Clasificación Iris", layout="wide")

st.title("🌸 Clasificación con Iris Dataset")
st.markdown("Comparación de modelos de Machine Learning con visualización de métricas y fronteras de decisión.")

# ==========================
# CARGAR DATASET
# ==========================
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# ==========================
# SIDEBAR - CONFIGURACIÓN
# ==========================
st.sidebar.header("⚙ Configuración")

modelo_nombre = st.sidebar.selectbox(
    "Selecciona el modelo",
    ("Logistic Regression", "KNN", "SVM", "Decision Tree")
)

test_size = st.sidebar.slider("Tamaño del test (%)", 10, 50, 30) / 100

# ==========================
# SELECCIÓN DE FEATURES PARA FRONTERA
# ==========================
st.sidebar.header("Visualización de frontera")

feature1 = st.sidebar.selectbox("Feature 1", feature_names, index=0)
feature2 = st.sidebar.selectbox("Feature 2", feature_names, index=1)

idx1 = feature_names.index(feature1)
idx2 = feature_names.index(feature2)

X_selected = X[:, [idx1, idx2]]

# ==========================
# DIVISIÓN Y ESCALADO
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# CREACIÓN DEL MODELO
# ==========================
if modelo_nombre == "Logistic Regression":
    model = LogisticRegression()
elif modelo_nombre == "KNN":
    k = st.sidebar.slider("Número de vecinos (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
elif modelo_nombre == "SVM":
    C = st.sidebar.slider("Parámetro C", 0.1, 10.0, 1.0)
    model = SVC(C=C, probability=True)
elif modelo_nombre == "Decision Tree":
    depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=depth)

# ==========================
# ENTRENAMIENTO
# ==========================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================
# MÉTRICAS
# ==========================
st.subheader("📊 Métricas de desempeño")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1-score", f"{f1:.3f}")

# ==========================
# MATRIZ DE CONFUSIÓN
# ==========================
st.subheader("🔍 Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel("Real")
plt.xlabel("Predicho")
st.pyplot(fig_cm)

# ==========================
# CURVA ROC (One-vs-Rest)
# ==========================
st.subheader("📈 Curva ROC (One-vs-Rest)")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

fig_roc, ax_roc = plt.subplots()

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# ==========================
# FRONTERA DE DECISIÓN
# ==========================
st.subheader("🧠 Frontera de decisión")

h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k')
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
st.pyplot(fig)
