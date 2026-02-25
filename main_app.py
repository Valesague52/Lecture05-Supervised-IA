import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificador Iris Dataset",
    page_icon="🌸",
    layout="wide"
)

# Título y descripción
st.title("🌸 Clasificador de Flores Iris")
st.markdown("""
Esta aplicación permite clasificar el dataset Iris utilizando diferentes algoritmos de machine learning.
Puedes seleccionar el modelo, ajustar parámetros y visualizar diferentes métricas de desempeño.
""")

# Cargar datos
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris

df, iris = load_data()

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Modelo")

# Selección de características
st.sidebar.subheader("Selección de Características")
feature_names = iris.feature_names
selected_features = st.sidebar.multiselect(
    "Características a utilizar:",
    feature_names,
    default=feature_names
)

if len(selected_features) < 2:
    st.sidebar.warning("Selecciona al menos 2 características")
    selected_features = feature_names[:2]

# Selección de modelo
st.sidebar.subheader("Selección de Modelo")
models = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

selected_model = st.sidebar.selectbox(
    "Selecciona un modelo:",
    list(models.keys())
)

# Parámetros específicos del modelo
st.sidebar.subheader("Parámetros del Modelo")
params = {}

if selected_model == "Regresión Logística":
    params['C'] = st.sidebar.slider("C (Regularización)", 0.01, 10.0, 1.0, 0.1)
    params['solver'] = st.sidebar.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg'])

elif selected_model == "Árbol de Decisión":
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    params['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 20, 2)

elif selected_model == "Random Forest":
    params['n_estimators'] = st.sidebar.slider("N Estimators", 10, 200, 100, 10)
    params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)

elif selected_model == "SVM":
    params['C'] = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)
    params['kernel'] = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly'])

elif selected_model == "K-Nearest Neighbors":
    params['n_neighbors'] = st.sidebar.slider("N Neighbors", 1, 20, 5)
    params['weights'] = st.sidebar.selectbox("Weights", ['uniform', 'distance'])

elif selected_model == "Gradient Boosting":
    params['n_estimators'] = st.sidebar.slider("N Estimators", 10, 200, 100, 10)
    params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)

# Configuración de entrenamiento
st.sidebar.subheader("Configuración de Entrenamiento")
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20, 5) / 100
random_state = st.sidebar.number_input("Random State", 0, 100, 42)
use_scaler = st.sidebar.checkbox("Estandarizar características", True)

# Tabs para organización
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Métricas", "🎯 Visualizaciones"])

with tab1:
    st.header("Exploración de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(10))
        
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df[feature_names].describe())
    
    with col2:
        st.subheader("Distribución de clases")
        fig, ax = plt.subplots(figsize=(8, 6))
        df['target_names'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_xlabel('Clase')
        ax.set_ylabel('Cantidad')
        ax.set_title('Distribución de especies Iris')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        plt.close()

with tab2:
    st.header("Entrenamiento del Modelo")
    
    # Preparar datos
    X = df[selected_features].values
    y = df['target'].values
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Escalar si es necesario
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
    
    # Crear y entrenar modelo con parámetros
    model = models[selected_model]
    if params:
        model.set_params(**params)
    
    with st.spinner('Entrenando modelo...'):
        model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")
    
    # Validación cruzada
    st.subheader("Validación Cruzada (5 folds)")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 6), cv_scores, color='skyblue', edgecolor='navy')
    ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Media: {cv_scores.mean():.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Resultados de Validación Cruzada')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tab3:
    st.header("Métricas de Desempeño Detalladas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=iris.target_names, 
                   yticklabels=iris.target_names, ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Valor Real')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Reporte de Clasificación")
        report = classification_report(y_test, y_pred, 
                                     target_names=iris.target_names, 
                                     output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # Curvas ROC (solo para clasificación binaria o con one-vs-rest)
    if y_pred_proba is not None and len(np.unique(y)) == 2:
        st.subheader("Curva ROC")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Curva ROC')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

with tab4:
    st.header("Visualizaciones Avanzadas")
    
    if len(selected_features) == 2:
        st.subheader("Frontera de Decisión")
        
        # Crear malla para visualización
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        # Preparar datos de malla
        mesh_data = np.c_[xx.ravel(), yy.ravel()]
        
        if len(selected_features) > 2:
            # Si hay más de 2 características, necesitamos completar con medias
            mesh_data_full = np.zeros((mesh_data.shape[0], len(selected_features)))
            mesh_data_full[:, :2] = mesh_data
            for i in range(2, len(selected_features)):
                mesh_data_full[:, i] = X[:, i].mean()
        else:
            mesh_data_full = mesh_data
        
        # Escalar si es necesario
        if use_scaler and scaler is not None:
            mesh_data_scaled = scaler.transform(mesh_data_full)
        else:
            mesh_data_scaled = mesh_data_full
        
        # Predecir en malla
        Z = model.predict(mesh_data_scaled)
        Z = Z.reshape(xx.shape)
        
        # Visualizar
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Contorno de la frontera de decisión
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')
        
        # Scatter plot de los datos
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', 
                           edgecolor='black', s=100)
        
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title(f'Frontera de Decisión - {selected_model}')
        
        # Leyenda
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                     label=name, markersize=10)
                          for i, name in enumerate(iris.target_names)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        st.pyplot(fig)
        plt.close()
        
    else:
        st.info("Selecciona exactamente 2 características en el panel lateral para visualizar la frontera de decisión.")
    
    # Visualización 3D interactiva
    if len(selected_features) >= 3:
        st.subheader("Visualización 3D Interactiva")
        
        fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1], 
                           z=selected_features[2], color='target_names',
                           title='Visualización 3D del Dataset Iris',
                           labels={'target_names': 'Especie'})
        
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)

# Información adicional
with st.expander("ℹ️ Información sobre los modelos"):
    st.markdown("""
    ### Modelos disponibles:
    - **Regresión Logística**: Modelo lineal para clasificación binaria y multiclase
    - **Árbol de Decisión**: Estructura de árbol para toma de decisiones
    - **Random Forest**: Conjunto de árboles de decisión
    - **SVM**: Máquinas de vectores de soporte
    - **K-Nearest Neighbors**: Clasificación basada en vecinos cercanos
    - **Gradient Boosting**: Ensamble de modelos secuenciales
    - **Naive Bayes**: Clasificador probabilístico basado en teorema de Bayes
    
    ### Métricas:
    - **Accuracy**: Proporción de predicciones correctas
    - **Precision**: Proporción de predicciones positivas correctas
    - **Recall**: Proporción de verdaderos positivos identificados
    - **F1-Score**: Media armónica de precision y recall
    """)

# Footer
st.markdown("---")
st.markdown("Creado con ❤️ usando Streamlit | Dataset Iris")
