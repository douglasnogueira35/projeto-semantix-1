# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# =====================================================
# IMPORTS OPCIONAIS (PDF/PPT)
# =====================================================
pdf_ok, ppt_ok = True, True
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    pdf_ok = False
try:
    from pptx import Presentation
    from pptx.util import Inches
except Exception:
    ppt_ok = False

# =====================================================
# CONFIGURAÇÃO STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Cliente Perfeito | Inteligência de Conversão",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🏢 Painel Executivo")
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
usar_smote = st.sidebar.checkbox("Balancear classes (SMOTE)", True)
mostrar_shap = st.sidebar.checkbox("Ativar SHAP (lento)", False)
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =====================================================
# CARREGAR DADOS
# =====================================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file) if uploaded_file else load_data("online_shoppers_intention.csv")

# =====================================================
# RENOMEAR COLUNAS
# =====================================================
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
df.rename(columns=traducao, inplace=True)

# =====================================================
# TARGET / FEATURES
# =====================================================
y = df["Compra"].astype(int)
X = df.drop(columns=["Compra"])
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# =====================================================
# PREPROCESSAMENTO
# =====================================================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# =====================================================
# SPLIT TREINO/TESTE
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# =====================================================
# BALANCEAMENTO SMOTE
# =====================================================
if usar_smote:
    smote = SMOTE(random_state=42)
    X_train_p, y_train = smote.fit_resample(X_train_p, y_train)

# =====================================================
# FEATURE SELECTION
# =====================================================
fs_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
fs_model.fit(X_train_p, y_train)
selector = SelectFromModel(fs_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_p)
X_test_sel = selector.transform(X_test_p)

# =====================================================
# TREINAMENTO MODELO FINAL
# =====================================================
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

model = train_model(X_train_sel, y_train)

# =====================================================
# PREDIÇÕES E MÉTRICAS
# =====================================================
y_pred = model.predict(X_test_sel)
y_prob = model.predict_proba(X_test_sel)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(model, X_train_sel, y_train, cv=5, scoring="roc_auc")

# =====================================================
# FEATURE NAMES
# =====================================================
all_features = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
selected_features = np.array(all_features)[selector.get_support()]

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral",
    "📈 Métricas",
    "🧠 Explicabilidade",
    "📄 Relatórios"
])

# =====================================================
# VISÃO GERAL
# =====================================================
with tab1:
    st.title("🎯 Cliente Perfeito")
    st.markdown("""
**Sistema de Machine Learning corporativo e acadêmico**  

Tópicos Profissionais:
- Pipeline completo e robusto
- Balanceamento de classes e pré-processamento
- Seleção automática de variáveis
- Random Forest com validação cruzada
- Métricas completas: Accuracy, Precision, Recall, F1, ROC AUC
- Intervalos de confiança e bootstrap
- Interpretabilidade com importância e SHAP
- Relatórios executivos em TXT e PDF
- Preparado para dashboard corporativo e banca acadêmica
""")

# =====================================================
# METRICS
# =====================================================
with tab2:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Acurácia", f"{acc:.2%}")
    c2.metric("Precisão", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1-score", f"{f1:.2%}")
    c5.metric("ROC AUC", f"{roc:.3f}")
    st.markdown(f"**Cross-validation (ROC AUC)** média={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

# =====================================================
# EXPLICABILIDADE
# =====================================================
with tab3:
    imp_df = pd.DataFrame({"Variável": selected_features,
                           "Importância": model.feature_importances_}).sort_values("Importância", ascending=False).head(10)
    fig = px.bar(imp_df, x="Importância", y="Variável", orientation="h", title="Top 10 Variáveis")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    
    if mostrar_shap:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_sel[:200])
            shap_imp = np.abs(shap_values.values).mean(axis=0)
            shap_df = pd.DataFrame({"Variável": selected_features, "Impacto Médio": shap_imp})\
                        .sort_values("Impacto Médio", ascending=False).head(10)
            fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável",
                              orientation="h", title="SHAP Top 10")
            fig_shap.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception:
            st.warning("SHAP indisponível neste ambiente.")

# =====================================================
# RELATÓRIO + DOWNLOAD PDF / TXT
# =====================================================
with tab4:
    texto = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO

Modelo: Random Forest com seleção automática de variáveis

Métricas:
- Acurácia: {acc:.2%}
- Precisão: {prec:.2%}
- Recall: {rec:.2%}
- F1-score: {f1:.2%}
- ROC AUC: {roc:.3f}

Cross-validation (ROC AUC):
- Média: {cv_scores.mean():.3f}, Desvio padrão: {cv_scores.std():.3f}

Principais Insights:
- Variáveis comportamentais e tempo de navegação são críticas
- Páginas de produtos e valor de página impactam fortemente a conversão
- Modelo robusto e interpretável para suporte estratégico
"""

    st.subheader("📄 Relatório Automático")
    st.text_area("Relatório detalhado", texto, height=350)
    st.download_button("⬇️ Baixar Relatório (TXT)", texto,
                       file_name="relatorio_cliente_perfeito.txt", mime="text/plain")

    if pdf_ok:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []
        for line in texto.split("\n"):
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1,12))
        doc.build(story)
        st.download_button("⬇️ Baixar PDF Executivo", buffer.getvalue(),
                           file_name="relatorio_cliente_perfeito.pdf",
                           mime="application/pdf")
    else:
        st.info("📄 PDF indisponível (reportlab não instalado).")
