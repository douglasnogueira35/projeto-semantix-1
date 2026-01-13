# app.py - Cliente Perfeito
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import unicodedata
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import base64

st.set_page_config(page_title="🎯 Cliente Perfeito", layout="wide")

# Função para normalizar nomes de arquivos
def normalizar_nome_arquivo(file):
    if hasattr(file, 'name'):
        nome = file.name
    else:
        import os
        nome = os.path.basename(file)
    nome_base, extensao = os.path.splitext(nome)
    nome_base = unicodedata.normalize('NFKD', nome_base).encode('ASCII', 'ignore').decode()
    nome_base = nome_base.replace(' ', '_').lower()
    return f"{nome_base}{extensao.lower()}"

# Função para carregar CSV/Excel
@st.cache_data
def carregar_dados(file):
    nome_normalizado = normalizar_nome_arquivo(file)
    try:
        if nome_normalizado.endswith(".csv"):
            df = pd.read_csv(file)
        elif nome_normalizado.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Formato inválido! Use CSV ou Excel.")
            return None
        st.success(f"Arquivo '{nome_normalizado}' carregado com sucesso!")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

st.title("🎯 Cliente Perfeito - Painel do Analista")
st.write("Sistema de Machine Learning para identificar padrões de navegação com maior probabilidade de compra.")

# Upload de arquivo
uploaded_file = st.file_uploader("📂 Carregar CSV ou Excel", type=["csv", "xls", "xlsx"])
df = carregar_dados(uploaded_file) if uploaded_file else None

if df is not None:
    st.dataframe(df.head())
    colunas = df.columns.tolist()
    
    target_col = st.selectbox("Selecione a coluna target (Ex: Compra)", colunas)
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Se target for string, codifica como número
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
    except Exception as e:
        st.error(f"Erro ao definir X e y: {e}")
        st.stop()
    
    # Pré-processamento
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    
    # Split train/test
    test_size = st.slider("📏 Proporção do conjunto de teste", 0.1, 0.5, 0.2)
    random_state = st.number_input("🔁 Random State", value=42)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)
    except Exception as e:
        st.warning("Estratificação não possível, dividindo sem stratify")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # SMOTE opcional
    smote = SMOTE(random_state=random_state)
    if st.checkbox("Aplicar SMOTE para balanceamento de classes"):
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Treinamento de modelos
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    resultados = {}
    for nome, model in modelos.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        resultados[nome] = {
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-score": f1_score(y_test, y_pred, average='weighted')
        }
    
    # Exibir métricas
    st.subheader("📊 Métricas de Avaliação")
    for nome, metricas in resultados.items():
        st.markdown(f"**{nome}**")
        for met, val in metricas.items():
            st.write(f"{met}: {val*100:.2f}%")
    
    # Confusion Matrix Random Forest
    st.subheader("🧮 Matriz de Confusão - Random Forest")
    model_rf = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())])
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)
    
    # Exportar relatório TXT
    relatorio_txt = f"Relatório - Cliente Perfeito\n\nMétricas:\n"
    for nome, metricas in resultados.items():
        relatorio_txt += f"{nome}:\n"
        for met, val in metricas.items():
            relatorio_txt += f"  {met}: {val*100:.2f}%\n"
        relatorio_txt += "\n"
    
    st.download_button("📥 Baixar relatório TXT", relatorio_txt, file_name="relatorio_cliente_perfeito.txt")
    
    # Exportar PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Relatório - Cliente Perfeito", styles['Title']), Spacer(1,12)]
    for nome, metricas in resultados.items():
        elements.append(Paragraph(nome, styles['Heading2']))
        for met, val in metricas.items():
            elements.append(Paragraph(f"{met}: {val*100:.2f}%", styles['Normal']))
    doc.build(elements)
    st.download_button("📥 Baixar relatório PDF", buffer.getvalue(), file_name="relatorio_cliente_perfeito.pdf")
    
else:
    st.warning("Nenhum dado disponível. Faça upload de um arquivo válido.")
