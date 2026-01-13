# 🎯 Cliente Perfeito – Dashboard de Análise Preditiva

**Cliente Perfeito** é um sistema de **Machine Learning** desenvolvido para identificar padrões de navegação de usuários em e-commerce que indicam maior probabilidade de compra. Este dashboard interativo permite analisar métricas, gerar gráficos, relatórios e PDFs de forma automática e intuitiva.

---

## 🛠 Funcionalidades

- Upload de dados **CSV** ou **Excel** (até 200MB) diretamente pelo app.
- Dataset padrão incluído: `intenção_de_compradores_online.csv`.
- Pré-processamento automático:
  - Normalização de colunas numéricas
  - One-Hot Encoding de colunas categóricas
- Treinamento de múltiplos modelos:
  - **Regressão Logística**
  - **Random Forest**
  - **XGBoost**
- Métricas de avaliação:
  - Acurácia, Precisão, Recall, F1-score, ROC AUC
- **Visualização interativa de gráficos**:
  - Importância de variáveis (Random Forest)
  - SHAP para explicabilidade do XGBoost
- **Relatórios automáticos**:
  - TXT exibido na tela e disponível para download
  - PDF pronto para download
- Balanceamento de classes opcional com **SMOTE**
- Interface moderna e rápida, totalmente em português

---

## 📂 Estrutura do Repositório

