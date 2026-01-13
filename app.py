# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Cliente Perfeito | Dashboard Preditivo",
    page_icon="👔",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 👨‍💼 Painel do Analista")
st.sidebar.caption("Modelagem preditiva de conversão")

test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("🔁 Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", True)

st.sidebar.markdown("### 📂 Carregar CSV ou Excel")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

# =========================
# CARREGAMENTO DE DADOS
# =========================
@st.cache_data
def carregar_dados(file):
    if file is None:
        return None
    try:
        if str(file).lower().endswith(".csv"):
            return pd.read_csv(file)
        elif str(file).lower().endswith(".xlsx"):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Caminho relativo para arquivo padrão sem acentos
arquivo_padrao = os.path.join(os.path.dirname(__file__), "intencao_de_compradores_online.csv")
df = carregar_dados(uploaded_file) if uploaded_file else carregar_dados(arquivo_padrao)

if df is None:
    st.warning("Arquivo padrão não encontrado. Faça upload de um CSV ou Excel.")
    st.stop()
else:
    st.success("Arquivo carregado com sucesso!")

# =========================
# TRADUÇÃO DAS COLUNAS
# =========================
traducao = {
    "Administrative": "Paginas_Administrativas",
    "Administrative_Duration": "Tempo_Administrativo",
    "Informational": "Paginas_Informativas",
    "Informational_Duration": "Tempo_Informativo",
    "ProductRelated": "Paginas_Produto",
    "ProductRelated_Duration": "Tempo_Produto",
    "BounceRates": "Taxa_Rejeicao",
    "ExitRates": "Taxa_Saida",
    "PageValues": "Valor_Pagina",
    "SpecialDay": "Dia_Especial",
    "Month": "Mes",
    "OperatingSystems": "Sistema_Operacional",
    "Browser": "Navegador",
    "Region": "Regiao",
    "TrafficType": "Tipo_Trafego",
    "VisitorType": "Tipo_Visitante",
    "Weekend": "Fim_de_Semana",
    "Revenue": "Compra"
}

df = df.rename(columns=traducao)

# =========================
# PREPARAÇÃO DOS DADOS
# =========================
target_col = "Compra"

y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

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
# TREINAMENTO DOS MODELOS
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
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# =========================
# DASHBOARD
# =========================
st.title("🎯 Cliente Perfeito")
st.markdown(
    """
Sistema de **Machine Learning** para identificar padrões de navegação
associados à maior probabilidade de compra. Use os gráficos interativos e relatórios automáticos.
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{acc:.2%}")
col2.metric("Precisão", f"{prec:.2%}")
col3.metric("Recall", f"{rec:.2%}")
col4.metric("F1-score", f"{f1:.2%}")

# =========================
# IMPORTÂNCIA DAS VARIÁVEIS
# =========================
feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))

imp_rf = pd.DataFrame({
    "Variável": feature_names,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False).head(10)

fig_rf = px.bar(
    imp_rf, x="Importância", y="Variável",
    orientation="h", title="Top 10 Variáveis — Random Forest"
)
fig_rf.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_rf, use_container_width=True)

# =========================
# SHAP
# =========================
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_p[:300])
shap_importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Variável": feature_names,
    "Impacto Médio": shap_importance
}).sort_values("Impacto Médio", ascending=False).head(10)

fig_shap = px.bar(
    shap_df, x="Impacto Médio", y="Variável",
    orientation="h", title="Top 10 Variáveis — SHAP"
)
fig_shap.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_shap, use_container_width=True)

# =========================
# RELATÓRIO AUTOMÁTICO
# =========================
texto_relatorio = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo escolhido: XGBoost
Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}

Principais Insights:
- Variáveis relacionadas ao comportamento de navegação são decisivas.
- Tempo e valor das páginas impactam diretamente a conversão.
- Modelo XGBoost oferece melhor trade-off entre performance e interpretabilidade.

Conclusão:
Sistema confiável para apoio à tomada de decisão em e-commerce.
"""

st.text_area("📄 Relatório automático (pronto para análise)", texto_relatorio, height=300)

# =========================
# DOWNLOAD PDF
# =========================
def gerar_pdf(texto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(linha, styles["Normal"]) for linha in texto.split("\n")]
    story.insert(0, Paragraph("RELATÓRIO EXECUTIVO – CLIENTE PERFEITO", styles["Title"]))
    story.insert(1, Spacer(1, 12))
    doc.build(story)
    buffer.seek(0)
    return buffer

st.download_button("⬇️ Baixar Relatório PDF", data=gerar_pdf(texto_relatorio),
                   file_name="relatorio_cliente_perfeito.pdf", mime="application/pdf")
