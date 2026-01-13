# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
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
from io import BytesIO

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

# =========================
# CARREGAMENTO DE DADOS
# =========================
@st.cache_data
def carregar_dados(file):
    if file is None:
        return pd.DataFrame()
    
    # Detecta tipo do arquivo corretamente
    if hasattr(file, "name"):
        nome = file.name.lower()
    else:
        nome = str(file).lower()
    
    if nome.endswith(".csv"):
        return pd.read_csv(file)
    elif nome.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.warning("Formato de arquivo inválido. Envie CSV ou Excel.")
        return pd.DataFrame()

# Caminho relativo para arquivo padrão no repositório
arquivo_padrao = "intenção_de_compradores_online.csv"

if uploaded_file:
    df = carregar_dados(uploaded_file)
else:
    try:
        df = carregar_dados(arquivo_padrao)
    except FileNotFoundError:
        st.warning("Arquivo padrão não encontrado. Faça upload de um CSV ou Excel.")
        df = pd.DataFrame()

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

if not df.empty:
    df.rename(columns=traducao, inplace=True)

# =========================
# SELEÇÃO DA TARGET
# =========================
if not df.empty:
    target_col = st.selectbox("Selecione a coluna target", df.columns, index=len(df.columns)-1)
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # =========================
    # PREPARAÇÃO DOS DADOS
    # =========================
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
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
            n_jobs=-1
        )
        log_reg.fit(X, y)
        rf.fit(X, y)
        xgb.fit(X, y)
        return log_reg, rf, xgb

    log_reg, rf, xgb = treinar_modelos(X_train_p, y_train)

    # =========================
    # MÉTRICAS E PREVISÕES
    # =========================
    y_pred = xgb.predict(X_test_p)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_p)[:, 1])

    # =========================
    # DASHBOARD
    # =========================
    st.title("🎯 Cliente Perfeito")
    st.markdown("Sistema de **Machine Learning** para identificar padrões de navegação associados à maior probabilidade de compra.")

    # Métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Acurácia", f"{acc:.2%}")
    col2.metric("Precisão", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")
    col5.metric("ROC AUC", f"{roc_auc:.2%}")

    # Feature importance Random Forest
    feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    imp_rf = pd.DataFrame({"Variável": feature_names, "Importância": rf.feature_importances_}).sort_values("Importância", ascending=False).head(10)
    fig_rf = px.bar(imp_rf, x="Importância", y="Variável", orientation="h", title="Top 10 Variáveis — Random Forest")
    fig_rf.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rf, use_container_width=True)

    # SHAP explicabilidade
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test_p[:300])
    shap_df = pd.DataFrame({
        "Variável": feature_names,
        "Impacto Médio": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("Impacto Médio", ascending=False).head(10)
    fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", title="Top 10 Variáveis — SHAP")
    fig_shap.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)

    # =========================
    # RELATÓRIO EXECUTIVO TXT
    # =========================
    relatorio_txt = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo: XGBoost

Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}
ROC AUC: {roc_auc:.2%}

Principais Insights:
- Variáveis relacionadas ao comportamento de navegação são decisivas.
- Tempo e valor das páginas impactam diretamente a conversão.
- Modelo robusto para tomada de decisão estratégica.

Conclusão:
Sistema confiável para apoio à tomada de decisão em e-commerce.
""".strip()

    st.text_area("📄 Relatório automático (TXT)", relatorio_txt, height=320)
    st.download_button("⬇️ Baixar relatório TXT", relatorio_txt, "relatorio_cliente_perfeito.txt")

    # =========================
    # RELATÓRIO PDF
    # =========================
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()
    story = []

    for line in relatorio_txt.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    st.download_button("⬇️ Baixar relatório PDF", pdf_buffer.getvalue(), "relatorio_cliente_perfeito.pdf")
