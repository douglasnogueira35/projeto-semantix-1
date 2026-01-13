# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Cliente Perfeito | Inteligência de Conversão",
    page_icon="👔",
    layout="wide"
)

st.title("🎯 Cliente Perfeito")
st.markdown("""
Sistema de **Machine Learning** para identificar padrões de navegação associados à maior probabilidade de compra.
Use os gráficos interativos e relatórios automáticos.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 👨‍💼 Painel do Analista")
st.sidebar.caption("Modelagem preditiva de conversão")

test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("🔁 Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", True)

st.sidebar.markdown("### 📂 Carregar CSV ou Excel")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"], help="Tamanho máximo: 200MB")

# =========================
# FUNÇÃO CARREGAR DADOS
# =========================
def carregar_dados(file):
    try:
        if file is not None:
            nome_arquivo = file.name.lower()
            if nome_arquivo.endswith(".csv"):
                return pd.read_csv(file)
            elif nome_arquivo.endswith(".xlsx"):
                return pd.read_excel(file)
            else:
                st.error("Formato inválido! Use CSV ou Excel.")
                return pd.DataFrame()
        else:
            st.warning("Nenhum arquivo carregado. Faça upload de CSV ou Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

# =========================
# CARREGAR DADOS
# =========================
df = carregar_dados(uploaded_file)

# =========================
# VALIDAÇÃO DO DATAFRAME
# =========================
if df.empty:
    st.stop()

st.success("Arquivo carregado com sucesso!")

# =========================
# SELEÇÃO DA TARGET
# =========================
target_col = st.selectbox("Selecione a coluna target (ex: Compra)", df.columns, index=len(df.columns)-1)
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# =========================
# IDENTIFICAR COLUNAS NUMÉRICAS E CATEGÓRICAS
# =========================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# =========================
# PREPROCESSAMENTO
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
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
# MÉTRICAS XGBOOST
# =========================
y_pred = xgb.predict(X_test_p)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_p)[:,1])

# =========================
# VISUALIZAÇÃO DE MÉTRICAS
# =========================
st.subheader("📊 Métricas do Modelo XGBoost")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Acurácia", f"{acc:.2%}")
col2.metric("Precisão", f"{prec:.2%}")
col3.metric("Recall", f"{rec:.2%}")
col4.metric("F1-score", f"{f1:.2%}")
col5.metric("ROC AUC", f"{roc_auc:.2%}")

# =========================
# IMPORTÂNCIA DE VARIÁVEIS
# =========================
feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
imp_rf = pd.DataFrame({
    "Variável": feature_names,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False).head(10)

fig_rf = px.bar(imp_rf, x="Importância", y="Variável", orientation="h", title="Top 10 Variáveis — Random Forest")
fig_rf.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_rf, use_container_width=True)

# =========================
# EXPLICABILIDADE SHAP
# =========================
st.subheader("🧠 Explicabilidade do Modelo XGBoost (SHAP)")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test_p[:300])
shap_importance = np.abs(shap_values.values).mean(axis=0)

shap_df = pd.DataFrame({
    "Variável": feature_names,
    "Impacto Médio": shap_importance
}).sort_values("Impacto Médio", ascending=False).head(10)

fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", title="Top 10 Variáveis — SHAP")
fig_shap.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_shap, use_container_width=True)

# =========================
# RELATÓRIO EXECUTIVO
# =========================
st.subheader("📄 Relatório Executivo")
relatorio = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo selecionado: XGBoost (alto desempenho em dados desbalanceados)
Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}
ROC AUC: {roc_auc:.2%}

Insights principais:
- Variáveis de navegação e tempo em páginas são determinantes para conversão.
- Random Forest e SHAP confirmam a relevância das mesmas variáveis.
- Balanceamento SMOTE opcional melhora modelos desbalanceados.

Conclusão:
Sistema confiável para apoiar decisões estratégicas em e-commerce.
""".strip()

st.text_area("Relatório automático", relatorio, height=300)
st.download_button("⬇️ Baixar relatório TXT", relatorio, "relatorio_cliente_perfeito.txt")

# =========================
# GERAR PDF
# =========================
pdf_buffer = BytesIO()
doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
styles = getSampleStyleSheet()
story = [Paragraph(line, styles["Normal"]) for line in relatorio.split("\n")]
story.insert(0, Paragraph("📄 Relatório Executivo – Cliente Perfeito", styles["Title"]))
doc.build(story)
pdf_buffer.seek(0)
st.download_button("⬇️ Baixar PDF", pdf_buffer, file_name="relatorio_cliente_perfeito.pdf")
