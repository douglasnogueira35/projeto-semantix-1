# ========================================================
# APP STREAMLIT – CLIENTE PERFEITO
# Sistema de ML para prever intenção de compra
# ========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

st.set_page_config(page_title="Cliente Perfeito", layout="wide")

# =========================
# FUNÇÃO PARA CARREGAR DADOS
# =========================
@st.cache_data
def carregar_dados(file):
    try:
        if file is None:
            st.warning("Nenhum arquivo carregado. Faça upload de um CSV ou Excel.")
            return pd.DataFrame()
        if str(file).lower().endswith(".csv"):
            df = pd.read_csv(file)
        elif str(file).lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            st.error("Formato inválido! Use CSV ou Excel.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

# =========================
# TÍTULO E INSTRUÇÕES
# =========================
st.title("🎯 Cliente Perfeito")
st.write("""
Sistema de Machine Learning para identificar padrões de navegação associados à maior probabilidade de compra.
Use os gráficos interativos, métricas e relatórios automáticos.
""")

# =========================
# UPLOAD DE ARQUIVO
# =========================
uploaded_file = st.file_uploader("📂 Carregar CSV ou Excel", type=['csv', 'xls', 'xlsx'], help="Tamanho máximo 200MB")
df = carregar_dados(uploaded_file)

if df.empty:
    st.stop()  # Para o app se não houver dados

st.success("Arquivo carregado com sucesso!")
st.dataframe(df.head())

# =========================
# SELEÇÃO DE COLUNA TARGET
# =========================
target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", df.columns)

# =========================
# VARIÁVEIS E PREPROCESSAMENTO
# =========================
X = df.drop(columns=[target_col])
y = df[target_col]

# Detectar colunas categóricas e numéricas
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# =========================
# DIVISÃO TREINO/TESTE
# =========================
test_size = st.slider("📏 Proporção do conjunto de teste", min_value=0.1, max_value=0.5, value=0.3)
random_state = st.number_input("🔁 Random State", min_value=0, value=42)

# Transformar em arrays
try:
    X_p = preprocessor.fit_transform(X)
except Exception as e:
    st.error(f"Erro no preprocessamento: {e}")
    st.stop()

# Checar se target é numérico para stratify
if y.nunique() < 2:
    stratify = None
    st.warning("Estratificação não possível, dividindo sem stratify")
else:
    stratify = y

X_train, X_test, y_train, y_test = train_test_split(
    X_p, y, test_size=test_size, random_state=random_state, stratify=stratify
)

# =========================
# BALANCEAMENTO SEGURO
# =========================
usar_smote = st.checkbox("Aplicar SMOTE para balanceamento de classes")
if usar_smote:
    try:
        min_count = y_train.value_counts().min()
        if min_count <= 1:
            st.warning(f"SMOTE não será aplicado: classe minoritária com {min_count} amostra(s)")
        else:
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.success("SMOTE aplicado com sucesso!")
    except Exception as e:
        st.warning(f"SMOTE não pôde ser aplicado: {e}")

# =========================
# TREINAMENTO DE MODELOS
# =========================
st.subheader("🏋️‍♂️ Treinamento de Modelos")

log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

modelos = {"Regressão Logística": log_reg, "Random Forest": rf, "XGBoost": xgb}

resultados = {}

for nome, model in modelos.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    except:
        auc = np.nan
    resultados[nome] = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "ROC AUC": auc
    }

# =========================
# EXIBIR RESULTADOS
# =========================
st.subheader("📊 Métricas dos Modelos")
for nome, metricas in resultados.items():
    st.markdown(f"### {nome}")
    st.write(metricas)

# =========================
# GRÁFICOS INTERATIVOS
# =========================
st.subheader("📈 Gráficos Interativos")
# Exemplo: Importância de variáveis no RandomForest
importances = rf.feature_importances_
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = num_cols + cat_cols
df_importance = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fig = px.bar(df_importance.sort_values("Importance", ascending=False), x="Feature", y="Importance",
             title="Importância de Variáveis - Random Forest")
st.plotly_chart(fig)

# =========================
# RELATÓRIO AUTOMÁTICO
# =========================
st.subheader("📄 Relatório Executivo")
relatorio_texto = "Relatório gerado automaticamente:\n\n"
for nome, metricas in resultados.items():
    relatorio_texto += f"{nome}:\n"
    for met, val in metricas.items():
        relatorio_texto += f"  {met}: {val:.3f}\n"
    relatorio_texto += "\n"

st.text_area("📝 Relatório na Tela", relatorio_texto, height=250)

# Gerar PDF
pdf_buffer = BytesIO()
doc = SimpleDocTemplate(pdf_buffer)
styles = getSampleStyleSheet()
story = [Paragraph("Relatório Executivo - Cliente Perfeito", styles['Title']), Spacer(1,12)]
for nome, metricas in resultados.items():
    story.append(Paragraph(nome, styles['Heading2']))
    for met, val in metricas.items():
        story.append(Paragraph(f"{met}: {val:.3f}", styles['Normal']))
    story.append(Spacer(1,12))
doc.build(story)
pdf_buffer.seek(0)
st.download_button("⬇️ Baixar PDF", data=pdf_buffer, file_name="relatorio_cliente_perfeito.pdf", mime="application/pdf")
