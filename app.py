# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
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

# Parâmetros
test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("🔁 Random State", value=42, step=1)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (SMOTE)", True)

# Upload de dados
st.sidebar.markdown("### 📂 Carregar CSV ou Excel")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

arquivo_padrao = r"C:\Users\dougl\Downloads\projeto semantix 1\online_shoppers_intention.csv"

@st.cache_data
def carregar_dados(file):
    try:
        if file is None:
            return pd.read_csv(arquivo_padrao)
        if str(file).lower().endswith(".csv"):
            return pd.read_csv(file)
        elif str(file).lower().endswith(".xlsx"):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

# Carrega dataset
df = carregar_dados(uploaded_file)
if df.empty:
    st.warning("Nenhum arquivo carregado ou arquivo padrão não encontrado!")
else:
    st.sidebar.success("Arquivo carregado com sucesso!")

# =========================
# SELEÇÃO DA COLUNA TARGET
# =========================
if not df.empty:
    target_col = st.sidebar.selectbox("Selecione a coluna target", df.columns)
    y = df[target_col]
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

    X_p = preprocessor.fit_transform(X)

    # Seleção de features automática
    selector = SelectKBest(score_func=f_classif, k='all')
    X_sel = selector.fit_transform(X_p, y)
    feature_names = np.array(num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)))
    selected_features = feature_names[selector.get_support(indices=True)]

    # =========================
    # DIVISÃO TREINO/TESTE
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # SMOTE opcional
    if usar_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

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

    log_reg, rf, xgb = treinar_modelos(X_train, y_train, random_state)

    # =========================
    # MÉTRICAS
    # =========================
    y_pred = xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]) if len(np.unique(y)) == 2 else np.nan

    # =========================
    # DASHBOARD
    # =========================
    st.title("🎯 Cliente Perfeito")
    st.markdown("Sistema de Machine Learning para identificar padrões de navegação associados à maior probabilidade de compra. Use gráficos interativos e relatórios automáticos.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{acc:.2%}")
    col2.metric("Precisão", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")

    if not np.isnan(roc):
        st.metric("ROC AUC", f"{roc:.2%}")

    # =========================
    # EXPLICABILIDADE
    # =========================
    st.subheader("🧠 Importância de Variáveis — Random Forest")
    imp_rf = pd.DataFrame({
        "Variável": selected_features,
        "Importância": rf.feature_importances_
    }).sort_values("Importância", ascending=False).head(10)

    fig_rf = px.bar(imp_rf, x="Importância", y="Variável", orientation="h")
    fig_rf.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_rf, use_container_width=True)

    st.subheader("🧠 Explicabilidade XGBoost (SHAP)")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test[:300])
    shap_df = pd.DataFrame({
        "Variável": selected_features,
        "Impacto Médio": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("Impacto Médio", ascending=False).head(10)

    fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h")
    fig_shap.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)

    # =========================
    # RELATÓRIO TXT
    # =========================
    st.subheader("📄 Relatório Executivo")
    relatorio = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo: XGBoost

Acurácia: {acc:.2%}
Precisão: {prec:.2%}
Recall: {rec:.2%}
F1-score: {f1:.2%}
ROC AUC: {roc:.2%}

Principais Insights:
- Variáveis relacionadas ao comportamento de navegação são decisivas.
- Tempo e valor das páginas impactam diretamente a conversão.
- O modelo apresenta alto potencial de uso estratégico.

Conclusão:
Sistema confiável para apoio à tomada de decisão em e-commerce.
""".strip()

    st.text_area("📌 Relatório automático (visualize antes de baixar)", relatorio, height=320)
    st.download_button("⬇️ Baixar relatório TXT", relatorio, "relatorio_cliente_perfeito.txt")

    # =========================
    # RELATÓRIO PDF
    # =========================
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("RELATÓRIO EXECUTIVO – CLIENTE PERFEITO", styles["Title"]), Spacer(1,12)]
    for line in relatorio.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1,6))
    doc.build(story)
    pdf_buffer.seek(0)
    st.download_button("⬇️ Baixar relatório PDF", pdf_buffer, "relatorio_cliente_perfeito.pdf", mime="application/pdf")
