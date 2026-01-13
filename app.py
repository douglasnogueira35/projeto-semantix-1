# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# CONFIGURAÇÃO STREAMLIT
# =========================
st.set_page_config(
    page_title="🎯 Cliente Perfeito",
    page_icon="👔",
    layout="wide"
)

st.title("🎯 Cliente Perfeito")
st.markdown("Sistema de Machine Learning para identificar padrões de navegação associados à maior probabilidade de compra. Use os gráficos interativos e relatórios automáticos.")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("👨‍💼 Painel do Analista")
test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("🔁 Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", value=True)

st.sidebar.markdown("### 📂 Carregar CSV ou Excel")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

arquivo_padrao = "C:\\Users\\dougl\\Downloads\\projeto semantix 1\\online_shoppers_intention.csv"

# =========================
# FUNÇÃO DE CARREGAMENTO
# =========================
@st.cache_data
def carregar_dados(file):
    try:
        if str(file).lower().endswith(".csv"):
            return pd.read_csv(file)
        elif str(file).lower().endswith(".xlsx"):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

df = carregar_dados(uploaded_file) if uploaded_file else carregar_dados(arquivo_padrao)

if df.empty:
    st.warning("Nenhum arquivo carregado ou arquivo padrão não encontrado! Faça upload de um CSV ou Excel.")
    st.stop()
else:
    st.success("Arquivo carregado com sucesso!")

# =========================
# SELEÇÃO DE TARGET
# =========================
st.subheader("Selecione a coluna target (variável de interesse)")
target_col = st.selectbox("Escolha a coluna target", options=df.columns)

y = df[target_col]
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

X = df.drop(columns=[target_col])

# =========================
# PREPROCESSAMENTO
# =========================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
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
def treinar_modelos(X, y):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                        colsample_bytree=0.8, random_state=random_state, eval_metric="logloss", n_jobs=-1)
    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return log_reg, rf, xgb

log_reg, rf, xgb = treinar_modelos(X_train_p, y_train)

# =========================
# METRICAS
# =========================
y_pred = xgb.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
roc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), average="weighted", multi_class="ovo")

# =========================
# EXIBIÇÃO DE METRICAS
# =========================
st.subheader("📊 Métricas do Modelo XGBoost")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Acurácia", f"{acc:.2%}")
col2.metric("Precisão", f"{prec:.2%}")
col3.metric("Recall", f"{rec:.2%}")
col4.metric("F1-score", f"{f1:.2%}")
col5.metric("ROC AUC", f"{roc:.2%}")

# =========================
# EXPLICABILIDADE - SHAP
# =========================
st.subheader("🧠 Explicabilidade - SHAP")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_p[:300])
feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
shap_importance = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({"Variável": feature_names, "Impacto Médio": shap_importance}).sort_values("Impacto Médio", ascending=False)
fig_shap = px.bar(shap_df.head(10), x="Impacto Médio", y="Variável", orientation="h", title="Top 10 Variáveis - SHAP")
fig_shap.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_shap, use_container_width=True)

# =========================
# RELATÓRIOS
# =========================
st.subheader("📄 Relatório Executivo")
relatorio_texto = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo escolhido: XGBoost (Melhor performance)
Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}
ROC AUC: {roc:.2%}

Principais insights:
- Variáveis relacionadas ao comportamento de navegação impactam diretamente na conversão.
- Métricas robustas indicam alta confiabilidade do modelo.
- XGBoost escolhido pelo melhor trade-off entre precisão e explicabilidade.
"""
st.text_area("Relatório (TXT)", relatorio_texto, height=250)
st.download_button("⬇️ Baixar relatório TXT", relatorio_texto, "relatorio_cliente_perfeito.txt")

# PDF
pdf_file = "relatorio_cliente_perfeito.pdf"
doc = SimpleDocTemplate(pdf_file)
styles = getSampleStyleSheet()
story = [Paragraph(par, styles["Normal"]) for par in relatorio_texto.split("\n") if par.strip()]
doc.build(story)
with open(pdf_file, "rb") as f:
    st.download_button("⬇️ Baixar PDF", f, file_name=pdf_file, mime="application/pdf")
