# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import shap

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Cliente Perfeito | Dashboard",
    page_icon="🚀",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🏢 Cliente Perfeito")
uploaded_file = st.sidebar.file_uploader("📂 Carregar CSV ou Excel", type=["csv","xlsx"])
test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (binário)", True)
mostrar_shap = st.sidebar.checkbox("🧠 Mostrar SHAP", True)

# =========================
# FUNÇÃO CARREGAR DADOS
# =========================
@st.cache_data
def carregar_dados(file):
    if hasattr(file,"name"):
        if file.name.lower().endswith(".csv"): return pd.read_csv(file)
        elif file.name.lower().endswith(".xlsx"): return pd.read_excel(file)
    else:
        if str(file).lower().endswith(".csv"): return pd.read_csv(file)
        elif str(file).lower().endswith(".xlsx"): return pd.read_excel(file)
    return pd.DataFrame()

# =========================
# ARQUIVO PADRÃO
# =========================
arquivo_padrao = r"C:\Users\dougl\Downloads\projeto semantix 1\online_shoppers_intention.csv"

try:
    df = carregar_dados(uploaded_file) if uploaded_file else carregar_dados(arquivo_padrao)
except FileNotFoundError:
    st.warning(f"Arquivo padrão '{arquivo_padrao}' não encontrado. Faça upload de um CSV ou Excel.")
    st.stop()

# =========================
# VISUALIZAÇÃO INICIAL
# =========================
st.subheader("📊 Visualização Inicial")
st.dataframe(df.head(10))

# =========================
# SELEÇÃO TARGET
# =========================
target_col = st.selectbox("Selecione a coluna TARGET", df.columns, index=0)
y_raw = df[target_col]
X = df.drop(columns=[target_col])

# =========================
# DETECTAR PROBLEMA
# =========================
if y_raw.nunique() == 2:
    problem_type = "binario"
    y = y_raw.apply(lambda x: 1 if x in [1,"TRUE","True","true"] else 0)
else:
    problem_type = "multiclasse"
    # Reindexar para classes consecutivas
    unique_vals = np.sort(y_raw.unique())
    mapping = {old:new for new,old in enumerate(unique_vals)}
    y = y_raw.map(mapping)

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
    X, y, test_size=test_size, random_state=42,
    stratify=y if problem_type=="binario" else None
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

if problem_type=="binario" and usar_smote:
    smote = SMOTE(random_state=42)
    X_train_p, y_train = smote.fit_resample(X_train_p, y_train)

# =========================
# FEATURE SELECTION
# =========================
fs_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
fs_model.fit(X_train_p, y_train)
selector = SelectFromModel(fs_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_p)
X_test_sel = selector.transform(X_test_p)
selected_features = np.array(num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)))[selector.get_support()]

# =========================
# TREINAMENTO DOS MODELOS
# =========================
@st.cache_resource
def treinar_modelos(X, y):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, eval_metric="mlogloss" if problem_type!="binario" else "logloss",
                        use_label_encoder=False, n_jobs=-1)
    log_reg.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return log_reg, rf, xgb

log_reg, rf, xgb = treinar_modelos(X_train_sel, y_train)

# =========================
# PREDIÇÕES
# =========================
y_pred_log = log_reg.predict(X_test_sel)
y_pred_rf = rf.predict(X_test_sel)
y_pred_xgb = xgb.predict(X_test_sel)

y_prob_log = log_reg.predict_proba(X_test_sel)[:,1] if problem_type=="binario" else None
y_prob_rf = rf.predict_proba(X_test_sel)[:,1] if problem_type=="binario" else None
y_prob_xgb = xgb.predict_proba(X_test_sel)[:,1] if problem_type=="binario" else None

# =========================
# MÉTRICAS
# =========================
st.subheader("📈 Métricas Comparativas")
metrics = {}
for name, y_pred, y_prob in [("Regressão Logística", y_pred_log, y_prob_log),
                              ("Random Forest", y_pred_rf, y_prob_rf),
                              ("XGBoost", y_pred_xgb, y_prob_xgb)]:
    if problem_type=="binario":
        metrics[name] = {
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        }
df_metrics = pd.DataFrame(metrics).T
st.dataframe(df_metrics.style.format("{:.2%}"))

# =========================
# CURVA ROC PARA TODOS OS MODELOS
# =========================
if problem_type=="binario":
    fig_roc = go.Figure()
    for name, y_prob in [("Regressão Logística", y_prob_log),
                         ("Random Forest", y_prob_rf),
                         ("XGBoost", y_prob_xgb)]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
    fig_roc.update_layout(title='Curva ROC Comparativa', template='plotly_white')
    st.plotly_chart(fig_roc, use_container_width=True)

# =========================
# FEATURE IMPORTANCE (RF e XGB)
# =========================
imp_rf = pd.DataFrame({
    "Variável": selected_features,
    "Importância": rf.feature_importances_
}).sort_values("Importância", ascending=False)

imp_xgb = pd.DataFrame({
    "Variável": selected_features,
    "Importância": xgb.feature_importances_
}).sort_values("Importância", ascending=False)

st.subheader("🔹 Top 10 Features — Random Forest")
fig_rf = px.bar(imp_rf.head(10), x="Importância", y="Variável", orientation="h", template="plotly_white")
fig_rf.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_rf, use_container_width=True)

st.subheader("🔹 Top 10 Features — XGBoost")
fig_xgb = px.bar(imp_xgb.head(10), x="Importância", y="Variável", orientation="h", template="plotly_white")
fig_xgb.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_xgb, use_container_width=True)

# =========================
# COEFICIENTES REGRESSÃO LOGÍSTICA
# =========================
st.subheader("🔹 Coeficientes — Regressão Logística")
coef_df = pd.DataFrame({
    "Variável": selected_features,
    "Coeficiente": log_reg.coef_[0]
}).sort_values("Coeficiente", ascending=False)
st.dataframe(coef_df.head(10))

# =========================
# SHAP (XGBoost)
# =========================
if mostrar_shap:
    try:
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer(X_test_sel[:100])
        shap_imp = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Variável": selected_features,
            "Impacto Médio": shap_imp
        }).sort_values("Impacto Médio", ascending=False)
        fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", template="plotly_white")
        fig_shap.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)
    except: st.warning("SHAP indisponível para este modelo.")

# =========================
# RELATÓRIO FINAL TXT/PDF
# =========================
relatorio_texto = "📄 Relatório Cliente Perfeito\n\n"
relatorio_texto += "Melhor modelo: XGBoost (baseado em ROC AUC e F1-score)\n\nTop Features:\n"
for i,row in imp_xgb.head(10).iterrows():
    relatorio_texto += f"{row['Variável']}: {row['Importância']:.4f}\n"

st.subheader("📄 Relatório")
st.text_area("Relatório Completo", relatorio_texto, height=400)
st.download_button("⬇️ Baixar Relatório TXT", relatorio_texto, file_name="relatorio_cliente_perfeito.txt", mime="text/plain")

def gerar_pdf(texto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(texto.replace("\n","<br/>"), styles["Normal"])]
    story.append(Spacer(1,12))
    doc.build(story)
    buffer.seek(0)
    return buffer

pdf_buffer = gerar_pdf(relatorio_texto)
st.download_button("⬇️ Baixar Relatório PDF", pdf_buffer, file_name="relatorio_cliente_perfeito.pdf", mime="application/pdf")
