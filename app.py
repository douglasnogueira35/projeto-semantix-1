# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Cliente Perfeito | Inteligência de Conversão",
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
arquivo_padrao = "online_shoppers_intention.csv"  # deve estar no mesmo diretório do app.py

# =========================
# FUNÇÃO DE CARREGAMENTO
# =========================
@st.cache_data
def carregar_dados(file):
    try:
        if hasattr(file, "name"):
            nome = file.name.lower()
            if nome.endswith(".csv"):
                return pd.read_csv(file)
            elif nome.endswith(".xlsx"):
                return pd.read_excel(file)
        elif isinstance(file, str):
            if file.lower().endswith(".csv"):
                return pd.read_csv(file)
            elif file.lower().endswith(".xlsx"):
                return pd.read_excel(file)
        st.error("Formato inválido. Use CSV ou Excel.")
        return None
    except FileNotFoundError:
        st.error(f"Arquivo '{file}' não encontrado. Faça upload de um CSV ou Excel.")
        return None

df = carregar_dados(uploaded_file) if uploaded_file else carregar_dados(arquivo_padrao)
if df is None:
    st.stop()
st.success("Arquivo carregado com sucesso!")

# =========================
# TRADUÇÃO DAS COLUNAS
# =========================
traducao = {
    "Administrative":"Paginas_Administrativas",
    "Administrative_Duration":"Tempo_Administrativo",
    "Informational":"Paginas_Informativas",
    "Informational_Duration":"Tempo_Informativo",
    "ProductRelated":"Paginas_Produto",
    "ProductRelated_Duration":"Tempo_Produto",
    "BounceRates":"Taxa_Rejeicao",
    "ExitRates":"Taxa_Saida",
    "PageValues":"Valor_Pagina",
    "SpecialDay":"Dia_Especial",
    "Month":"Mes",
    "OperatingSystems":"Sistema_Operacional",
    "Browser":"Navegador",
    "Region":"Regiao",
    "TrafficType":"Tipo_Trafego",
    "VisitorType":"Tipo_Visitante",
    "Weekend":"Fim_de_Semana",
    "Revenue":"Compra"
}
df.rename(columns=traducao, inplace=True)

# =========================
# SELEÇÃO DE TARGET
# =========================
target_col = st.selectbox("Selecione a coluna target", df.columns, index=df.columns.get_loc("Compra"))
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# =========================
# IDENTIFICAR COLUNAS
# =========================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# =========================
# PREPROCESSAMENTO + FEATURE SELECTION
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
selector = SelectKBest(score_func=f_classif, k="all")  # mantém todas

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", selector)
])

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

X_train_sel = pipeline.fit_transform(X_train, y_train)
X_test_sel = pipeline.transform(X_test)

if usar_smote:
    smote = SMOTE(random_state=random_state)
    X_train_sel, y_train = smote.fit_resample(X_train_sel, y_train)

# =========================
# TREINAMENTO
# =========================
@st.cache_resource
def treinar_modelos(X, y):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1
    )
    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return log_reg, rf, xgb

log_reg, rf, xgb = treinar_modelos(X_train_sel, y_train)

# =========================
# MÉTRICAS
# =========================
y_pred_xgb = xgb.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred_xgb)
prec = precision_score(y_test, y_pred_xgb)
rec = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)
roc_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_sel)[:,1])

# =========================
# DASHBOARD
# =========================
tabs = st.tabs(["📊 Visão Geral", "📈 Resultados", "🧠 Explicabilidade", "📑 Relatório Executivo"])

# -------------------------
# VISÃO GERAL
# -------------------------
with tabs[0]:
    st.title("🎯 Cliente Perfeito")
    st.markdown("""
Sistema de **Machine Learning** para identificar padrões de navegação associados à maior probabilidade de compra.
Use os gráficos interativos e relatórios automáticos.
""")

# -------------------------
# RESULTADOS
# -------------------------
with tabs[1]:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2%}")
    col2.metric("Precisão", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")
    col5.metric("ROC AUC", f"{roc_auc:.2%}")
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, xgb.predict_proba(X_test_sel)[:,1])
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'Curva ROC - AUC={roc_auc:.3f}',
        labels=dict(x="False Positive Rate", y="True Positive Rate")
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

# -------------------------
# EXPLICABILIDADE
# -------------------------
with tabs[2]:
    st.subheader("🧠 Explicabilidade do Modelo XGBoost")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test_sel[:300])
    
    feature_names = num_cols + list(pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"Variável": feature_names, "Impacto Médio": shap_importance}).sort_values("Impacto Médio", ascending=False).head(10)
    
    fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", title="Top 10 Features - SHAP")
    fig_shap.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)

# -------------------------
# RELATÓRIO EXECUTIVO
# -------------------------
with tabs[3]:
    texto_relatorio = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo escolhido: XGBoost
Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}
ROC AUC: {roc_auc:.2%}

Insights:
- Tempo e valor das páginas impactam diretamente a conversão.
- Variáveis relacionadas ao comportamento de navegação são decisivas.
- Sistema confiável para apoio à tomada de decisão estratégica em e-commerce.
"""
    st.text_area("Relatório Automático", texto_relatorio, height=320)
    st.download_button("⬇️ Baixar relatório TXT", texto_relatorio, "relatorio_cliente_perfeito.txt")

    # Gerar PDF
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(linha, styles['Normal']) for linha in texto_relatorio.split("\n")]
    story.insert(0, Paragraph("📄 Relatório Executivo – Cliente Perfeito", styles['Title']))
    story.insert(1, Spacer(1,12))
    doc.build(story)
    st.download_button("⬇️ Baixar relatório PDF", pdf_buffer.getvalue(), "relatorio_cliente_perfeito.pdf")
