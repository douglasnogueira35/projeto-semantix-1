import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Cliente Perfeito", layout="wide")

st.title("🎯 Cliente Perfeito")
st.write("Sistema de Machine Learning para identificar padrões de navegação associados à maior probabilidade de compra. Use os gráficos interativos, métricas e relatórios automáticos.")

# ---------------------------
# UPLOAD DE ARQUIVO
# ---------------------------
uploaded_file = st.file_uploader("📂 Carregar CSV ou Excel", type=["csv", "xlsx"])
arquivo_padrao = r"C:\Users\dougl\Downloads\projeto semantix 1\online_shoppers_intention.csv"

def carregar_dados(file):
    try:
        if file is None:
            df = pd.read_csv(arquivo_padrao)
        else:
            if str(file).lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        st.success("Arquivo carregado com sucesso!")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()

df = carregar_dados(uploaded_file)
if df.empty:
    st.stop()

# ---------------------------
# SELEÇÃO DA COLUNA TARGET
# ---------------------------
target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# Se coluna target for categórica, codificar para int
if y.dtype == "object":
    y = pd.factorize(y)[0]

# ---------------------------
# PADRÃO TREINO/TESTE
# ---------------------------
test_size = st.slider("📏 Proporção do conjunto de teste", 0.1, 0.5, 0.3)
random_state = st.number_input("🔁 Random State", value=42)

# Verificar se estratificação é possível
min_por_classe = pd.Series(y).value_counts().min()
if len(np.unique(y)) == 1 or min_por_classe < 2:
    stratify = None
    st.warning("Estratificação não possível (uma classe ou poucas amostras). Dividindo sem stratify.")
else:
    stratify = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify
)

# ---------------------------
# PIPELINE PREPROCESSAMENTO
# ---------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ---------------------------
# TREINAMENTO DE MODELOS
# ---------------------------
@st.cache_data
def treinar_modelos(X_train, y_train):
    # Pipeline Regressão Logística
    pipe_lr = Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=1000))])
    pipe_lr.fit(X_train, y_train)
    
    # Random Forest
    pipe_rf = Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=random_state))])
    pipe_rf.fit(X_train, y_train)
    
    # XGBoost
    pipe_xgb = Pipeline([("pre", preprocessor), ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state))])
    pipe_xgb.fit(X_train, y_train)
    
    return pipe_lr, pipe_rf, pipe_xgb

log_reg, rf, xgb = treinar_modelos(X_train, y_train)

# ---------------------------
# MÉTRICAS
# ---------------------------
def calcular_metricas(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

st.subheader("📊 Métricas")
for nome, modelo in zip(["Regressão Logística", "Random Forest", "XGBoost"], [log_reg, rf, xgb]):
    st.write(f"**{nome}**")
    metricas = calcular_metricas(modelo, X_test, y_test)
    st.json(metricas)

# ---------------------------
# GRÁFICOS
# ---------------------------
st.subheader("📈 Gráficos de análise")

# Importância de variáveis do RF
importancia = rf.named_steps["model"].feature_importances_
feat_names = rf.named_steps["pre"].transformers_[1][1].get_feature_names_out(cat_cols).tolist() + num_cols
feat_importance = pd.Series(importancia, index=feat_names).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax)
st.pyplot(fig)

# ---------------------------
# RELATÓRIOS
# ---------------------------
st.subheader("📄 Relatórios")
relatorio_txt = f"""
Relatório Cliente Perfeito

Modelo escolhido: XGBoost (maior performance)
Número de registros: {len(df)}
Coluna target: {target_col}
Principais métricas:
- Acurácia: {calcular_metricas(xgb, X_test, y_test)['Acurácia']:.2f}
- Precisão: {calcular_metricas(xgb, X_test, y_test)['Precisão']:.2f}
- Recall: {calcular_metricas(xgb, X_test, y_test)['Recall']:.2f}
- F1-score: {calcular_metricas(xgb, X_test, y_test)['F1-score']:.2f}
"""
st.text_area("📋 Relatório TXT", relatorio_txt, height=200)

# Download TXT
st.download_button("⬇️ Baixar Relatório TXT", relatorio_txt, file_name="relatorio_cliente_perfeito.txt")

# Download PDF
pdf_buffer = BytesIO()
doc = SimpleDocTemplate(pdf_buffer)
styles = getSampleStyleSheet()
story = [Paragraph(line, styles["Normal"]) for line in relatorio_txt.split("\n")]
doc.build(story)
st.download_button("⬇️ Baixar Relatório PDF", pdf_buffer.getvalue(), file_name="relatorio_cliente_perfeito.pdf")
