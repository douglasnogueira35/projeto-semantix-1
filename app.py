import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Cliente Perfeito", layout="wide")

st.title("🎯 Cliente Perfeito")
st.markdown("Sistema de Machine Learning para identificar padrões de navegação associados à maior probabilidade de compra. Use os gráficos interativos, métricas e relatórios automáticos.")

# =========================
# FUNÇÃO SEGURA PARA CARREGAR CSV/EXCEL
# =========================
@st.cache_data
def carregar_dados(file):
    if file is None:
        return pd.DataFrame()
    try:
        nome_arquivo = file.name.lower()
        if nome_arquivo.endswith(".csv"):
            df = pd.read_csv(file)
        elif nome_arquivo.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            st.error("Formato inválido! Use CSV ou Excel.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

# =========================
# UPLOAD DO ARQUIVO
# =========================
uploaded_file = st.file_uploader("📂 Carregar CSV ou Excel", type=['csv', 'xls', 'xlsx'], help="Tamanho máximo 200MB")
df = carregar_dados(uploaded_file)

if df.empty:
    st.warning("Nenhum arquivo carregado ou arquivo inválido. Faça upload de um CSV ou Excel.")
    st.stop()

st.success(f"Arquivo carregado com sucesso! ({df.shape[0]} linhas x {df.shape[1]} colunas)")

# =========================
# SELEÇÃO DA COLUNA TARGET
# =========================
target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", df.columns)

# =========================
# SEPARAÇÃO DE VARIÁVEIS
# =========================
X = df.drop(columns=[target_col])
y = df[target_col]

# Se target não for numérico, tentar codificar
if y.dtype == "object":
    y = pd.factorize(y)[0]

# =========================
# OPÇÃO DE BALANCEAMENTO
# =========================
balancear = st.checkbox("Aplicar SMOTE (balanceamento de classes)")

# =========================
# SEPARAR CONJUNTO DE TREINO/TESTE
# =========================
test_size = st.slider("Proporção do conjunto de teste", min_value=0.1, max_value=0.5, value=0.3)
random_state = st.number_input("Random State", min_value=0, value=42)

if len(np.unique(y)) > 1:
    stratify = y
else:
    stratify = None
    st.warning("Estratificação não possível (apenas uma classe presente).")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify
)

# =========================
# DETECTAR COLUNAS NUMÉRICAS E CATEGÓRICAS
# =========================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# =========================
# PIPELINE
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# =========================
# MODELAGEM
# =========================
log_reg = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=100))])
xgb = Pipeline([("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))])

modelos = {"Regressão Logística": log_reg, "Random Forest": rf, "XGBoost": xgb}

# =========================
# TREINAR MODELOS
# =========================
metricas = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas[nome] = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1": f1_score(y_test, y_pred, average='macro', zero_division=0),
    }

# =========================
# BALANCEAMENTO
# =========================
if balancear:
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# EXIBIR MÉTRICAS
# =========================
st.subheader("📊 Métricas dos Modelos")
for nome, met in metricas.items():
    st.markdown(f"### {nome}")
    st.write(pd.DataFrame([met]))

# =========================
# EXIBIR IMPORTÂNCIA DE VARIÁVEIS COM SHAP (XGBoost)
# =========================
st.subheader("🧠 Explicabilidade do Modelo (XGBoost)")

explainer = shap.TreeExplainer(xgb.named_steps['clf'])
X_test_transformed = preprocessor.transform(X_test)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(shap_values, X_test_transformed, feature_names=preprocessor.get_feature_names_out(), show=False)
st.pyplot(bbox_inches='tight')
plt.clf()

# =========================
# GERAR RELATÓRIO TXT E PDF
# =========================
def gerar_relatorio(metricas):
    texto = "Relatório de Métricas\n\n"
    for nome, met in metricas.items():
        texto += f"{nome}:\n"
        for k, v in met.items():
            texto += f"  {k}: {v:.4f}\n"
        texto += "\n"
    return texto

relatorio_txt = gerar_relatorio(metricas)
st.subheader("📄 Relatório Final")
st.text_area("Relatório", relatorio_txt, height=300)

# Botão para download TXT
st.download_button("📥 Baixar Relatório TXT", relatorio_txt.encode("utf-8"), "relatorio.txt")

# Gerar PDF
pdf_buffer = BytesIO()
c = canvas.Canvas(pdf_buffer, pagesize=letter)
c.drawString(50, 750, "Relatório de Métricas")
y_pos = 720
for linha in relatorio_txt.split("\n"):
    c.drawString(50, y_pos, linha)
    y_pos -= 15
c.save()
st.download_button("📥 Baixar Relatório PDF", pdf_buffer.getvalue(), "relatorio.pdf")
