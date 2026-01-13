import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.figure_factory as ff
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

st.set_page_config(page_title="Cliente Perfeito", layout="wide", page_icon="👔")
st.title("🎯 Cliente Perfeito")
st.markdown("Sistema de Machine Learning para identificar padrões de navegação e prever conversão.\nUse gráficos, métricas e relatórios automáticos.")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Painel de Configurações")
test_size = st.sidebar.slider("Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", value=True)

# =========================
# UPLOAD DE ARQUIVOS
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

@st.cache_data
def carregar_dados(file):
    try:
        if hasattr(file, "name"):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                return pd.read_excel(file)
        else:
            if str(file).endswith(".csv"):
                return pd.read_csv(file)
            elif str(file).endswith(".xlsx"):
                return pd.read_excel(file)
        st.error("Formato de arquivo inválido!")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        st.stop()

if uploaded_file:
    df = carregar_dados(uploaded_file)
    st.success("Arquivo carregado com sucesso!")
else:
    st.warning("Faça upload de um CSV ou Excel para continuar.")
    st.stop()

# =========================
# SELEÇÃO COLUNA TARGET
# =========================
target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", df.columns)

# Tentativa de conversão segura para numérico
try:
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().any():
        st.warning("Coluna target possui valores não numéricos, convertendo NaN para 0")
        y = y.fillna(0).astype(int)
    else:
        y = y.astype(int)
except Exception as e:
    st.error(f"Erro ao processar coluna target: {e}")
    st.stop()

X = df.drop(columns=[target_col])

# =========================
# PREPROCESSAMENTO
# =========================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# =========================
# SPLIT COM SEGURANÇA
# =========================
# Se stratify der erro (classe com única amostra), usamos split sem stratify
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
except ValueError:
    st.warning("Estratificação não possível, dividindo sem stratify")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

if usar_smote:
    smote = SMOTE(random_state=random_state)
    X_train_p, y_train = smote.fit_resample(X_train_p, y_train)

# =========================
# TREINAMENTO DE MODELOS
# =========================
@st.cache_resource
def treinar_modelos(X, y, rs):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs)
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=rs,
        eval_metric="logloss",
        n_jobs=-1
    )
    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return log_reg, rf, xgb

log_reg, rf, xgb = treinar_modelos(X_train_p, y_train, random_state)

# =========================
# MÉTRICAS
# =========================
y_pred = xgb.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# =========================
# DASHBOARD
# =========================
st.subheader("Métricas de Desempenho")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{acc:.2%}")
col2.metric("Precisão", f"{prec:.2%}")
col3.metric("Recall", f"{rec:.2%}")
col4.metric("F1-score", f"{f1:.2%}")

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Matriz de Confusão")
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(cm, x=list(set(y)), y=list(set(y)), colorscale="Blues", showscale=True)
st.plotly_chart(fig_cm, use_container_width=True)
