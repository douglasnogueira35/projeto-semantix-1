import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="🎯 Cliente Perfeito", layout="wide")
st.title("🎯 Cliente Perfeito - Painel do Analista")
st.write("Sistema de Machine Learning para identificar padrões de navegação com maior probabilidade de compra. Use os gráficos interativos, métricas e relatórios automáticos.")

# --- Arquivo padrão ---
arquivo_padrao = r"C:\Users\dougl\Downloads\projeto semantix 1\online_shoppers_intention.csv"

# --- Upload de arquivo ---
uploaded_file = st.file_uploader("📂 Carregar CSV ou Excel", type=["csv", "xls", "xlsx"], help="Arraste e solte seu arquivo ou clique para selecionar (máx 200MB).")

@st.cache_data
def carregar_dados(file=None):
    if file is None:
        if os.path.exists(arquivo_padrao):
            file = arquivo_padrao
        else:
            st.error("Nenhum arquivo carregado ou arquivo padrão não encontrado!")
            return pd.DataFrame()
    try:
        if str(file).lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif str(file).lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            st.error("Formato inválido! Use CSV ou Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()
    return df

df = carregar_dados(uploaded_file)
if df.empty:
    st.warning("Nenhum dado disponível. Faça upload de um arquivo válido.")
    st.stop()
else:
    st.success(f"Arquivo carregado com sucesso! Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
    st.dataframe(df.head())

# --- Seleção da coluna target ---
target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Ajusta tipo do target para classificação ---
if y.dtype == object:
    y = pd.factorize(y)[0]

# --- Seleção de colunas ---
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# --- Pré-processamento ---
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# --- Divisão treino/teste ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # se não houver estratificação possível
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    st.warning("Estratificação não possível, dividindo sem stratify")

# --- Modelos ---
def treinar_modelos(X, y):
    log_reg = Pipeline([("prep", preprocessor), ("lr", LogisticRegression(max_iter=500))])
    rf = Pipeline([("prep", preprocessor), ("rf", RandomForestClassifier(n_estimators=200))])
    xgb = Pipeline([("prep", preprocessor), ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])

    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return log_reg, rf, xgb

log_reg, rf, xgb = treinar_modelos(X_train, y_train)

# --- Métricas ---
def calcular_metricas(modelo, X, y):
    y_pred = modelo.predict(X)
    return {
        "Acurácia": accuracy_score(y, y_pred),
        "Precisão": precision_score(y, y_pred, average="weighted"),
        "Recall": recall_score(y, y_pred, average="weighted"),
        "F1-score": f1_score(y, y_pred, average="weighted")
    }

st.subheader("📊 Métricas - Conjunto de Teste")
metricas_log = calcular_metricas(log_reg, X_test, y_test)
metricas_rf = calcular_metricas(rf, X_test, y_test)
metricas_xgb = calcular_metricas(xgb, X_test, y_test)

st.write("**Regressão Logística**", metricas_log)
st.write("**Random Forest**", metricas_rf)
st.write("**XGBoost**", metricas_xgb)

# --- SHAP explicativo para XGBoost ---
explainer = shap.Explainer(xgb.named_steps["xgb"], preprocessor.fit_transform(X_train))
shap_values = explainer(preprocessor.transform(X_test))

st.subheader("🔍 Explicação SHAP - XGBoost")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, features=preprocessor.transform(X_test), feature_names=np.array(preprocessor.get_feature_names_out()), show=False)
st.pyplot(fig)

# --- Download de relatório ---
st.subheader("📄 Relatório Executivo")
relatorio_texto = f"""
Relatório do Cliente Perfeito
Número de linhas: {df.shape[0]}
Número de colunas: {df.shape[1]}
Modelos treinados: Regressão Logística, Random Forest, XGBoost
Métricas XGBoost: {metricas_xgb}
"""
st.text_area("📋 Relatório TXT", relatorio_texto, height=200)

# Download PDF
buffer = BytesIO()
doc = SimpleDocTemplate(buffer)
styles = getSampleStyleSheet()
story = [Paragraph(relatorio_texto, styles["Normal"])]
doc.build(story)
st.download_button("⬇️ Baixar Relatório PDF", data=buffer.getvalue(), file_name="relatorio_cliente_perfeito.pdf")
