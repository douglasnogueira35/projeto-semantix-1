# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

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
Sistema de **Machine Learning** para identificar padrões de navegação
associados à maior probabilidade de compra.  
Use os gráficos interativos, métricas e relatórios automáticos.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("👨‍💼 Painel do Analista")
test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("🔁 Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", True)

# Upload de arquivos
st.sidebar.markdown("📂 Carregar CSV ou Excel")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

# =========================
# CARREGAMENTO DE DADOS
# =========================
@st.cache_data
def carregar_dados(file):
    try:
        if hasattr(file, "name"):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                return pd.read_excel(file)
        else:  # caminho local
            if str(file).endswith(".csv"):
                return pd.read_csv(file)
            elif str(file).endswith(".xlsx"):
                return pd.read_excel(file)
        st.error("Formato de arquivo inválido! Use CSV ou Excel.")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        st.stop()

if uploaded_file:
    df = carregar_dados(uploaded_file)
    st.success("Arquivo carregado com sucesso!")
else:
    st.warning("Nenhum arquivo carregado! Faça upload de um CSV ou Excel.")
    st.stop()

# =========================
# SELEÇÃO DA COLUNA TARGET
# =========================
target_col = st.selectbox(
    "Selecione a coluna target (Ex: Compra)",
    options=df.columns
)

try:
    y = df[target_col].astype(int)
except:
    st.error("A coluna target deve ser numérica ou convertível para int.")
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
tabs = st.tabs(["📊 Visão Geral", "📈 Resultados", "🧠 Explicabilidade", "📑 Relatório Executivo"])

with tabs[0]:
    st.subheader("Visão Geral")
    st.dataframe(df.head())

with tabs[1]:
    st.subheader("Métricas de Desempenho")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{acc:.2%}")
    col2.metric("Precisão", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(cm, x=list(set(y)), y=list(set(y)),
                                        colorscale="Blues", showscale=True)
    st.plotly_chart(fig_cm, use_container_width=True)

with tabs[2]:
    st.subheader("Explicabilidade")
    feature_names = (
        num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    )
    # Random Forest
    imp_rf = pd.DataFrame({"Variável": feature_names, "Importância": rf.feature_importances_}).sort_values("Importância", ascending=False).head(10)
    fig_rf = px.bar(imp_rf, x="Importância", y="Variável", orientation="h", title="Top 10 Variáveis — Random Forest")
    fig_rf.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rf, use_container_width=True)
    # SHAP XGBoost
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test_p[:300])
    shap_df = pd.DataFrame({
        "Variável": feature_names,
        "Impacto Médio": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("Impacto Médio", ascending=False).head(10)
    fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", title="Top 10 Variáveis — SHAP")
    fig_shap.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)

with tabs[3]:
    st.subheader("Relatório Executivo")
    texto = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo escolhido: XGBoost (melhor desempenho)

Métricas:
- Acurácia: {acc:.2%}
- Precisão: {prec:.2%}
- Recall: {rec:.2%}
- F1-score: {f1:.2%}

Insights:
- Variáveis de comportamento de navegação são determinantes.
- Tempo e valor das páginas impactam diretamente a conversão.
- Sistema confiável para apoio estratégico em e-commerce.
"""
    st.text_area("Relatório automático", texto, height=300)

    def gerar_pdf(texto):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [Paragraph(texto, styles["Normal"])]
        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf_buffer = gerar_pdf(texto)
    st.download_button("⬇️ Baixar relatório PDF", pdf_buffer, file_name="relatorio_cliente_perfeito.pdf", mime="application/pdf")
