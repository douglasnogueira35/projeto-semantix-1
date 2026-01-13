# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from io import BytesIO

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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

st.sidebar.markdown("### 📂 Carregar Dados")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv","xlsx"])
arquivo_padrao = r"C:\Users\dougl\Downloads\projeto semantix 1\online_shoppers_intention.csv"

# =========================
# FUNÇÃO DE CARREGAMENTO
# =========================
@st.cache_data
def carregar_dados(file):
    if hasattr(file, 'name'):
        nome = file.name.lower()
    else:
        nome = str(file).lower()
    if nome.endswith(".csv"):
        return pd.read_csv(file)
    elif nome.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Formato inválido. Use CSV ou Excel.")
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
# PREPROCESSAMENTO
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# =========================
# FEATURE SELECTION AUTOMÁTICA
# =========================
selector = SelectKBest(score_func=f_classif, k="all")  # Mantém todas, mas pode ajustar k

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# =========================
# PIPELINE COM PREPROCESSAMENTO
# =========================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", selector)
])

X_train_sel = pipeline.fit_transform(X_train, y_train)
X_test_sel = pipeline.transform(X_test)

if usar_smote:
    smote = SMOTE(random_state=random_state)
    X_train_sel, y_train = smote.fit_resample(X_train_sel, y_train)

# =========================
# TREINAMENTO DE MODELOS COM TUNING
# =========================
@st.cache_resource
def treinar_modelos(X, y):
    # Random Forest com tuning rápido
    rf_param = {"n_estimators":[100,300], "max_depth":[5,10,None]}
    rf = RandomizedSearchCV(RandomForestClassifier(random_state=random_state), rf_param, n_iter=3, cv=3)
    
    xgb_param = {"max_depth":[3,5], "learning_rate":[0.05,0.1], "n_estimators":[100,300]}
    xgb = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state), xgb_param, n_iter=3, cv=3)
    
    log_reg = LogisticRegression(max_iter=1000)
    
    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    
    return log_reg, rf.best_estimator_, xgb.best_estimator_

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
Use os gráficos interativos para análise e explore relatórios automáticos.
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
    
    # Top 10 features
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
