# Comparativo OLS, WLS e Inflação Global 2024

Este projeto apresenta uma análise comparativa entre dois modelos de regressão — OLS (Ordinary Least Squares) e WLS (Weighted Least Squares) — com visualização gráfica, além de um painel com as taxas de inflação de diversas economias em 2024.

## 📌 Objetivos

- Demonstrar o impacto da heterocedasticidade na estimação de modelos de regressão.
- Comparar os resultados dos modelos OLS e WLS em uma simulação.
- Visualizar e comparar as taxas de inflação do Brasil e de outros países no cenário econômico de 2024.

## 📈 Tecnologias e bibliotecas utilizadas

- Python 3
- NumPy
- Statsmodels
- SciPy
- Matplotlib

## 📊 Conteúdo dos gráficos

1. **OLS vs WLS:** Comparação entre as curvas ajustadas dos dois modelos em um conjunto de dados simulado com heterocedasticidade.
2. **Inflação 2024:** Gráfico de barras com as taxas de inflação anual de 10 países, incluindo o Brasil.

## 🧠 Conclusão

A modelagem com WLS se mostra mais eficaz em capturar a variabilidade dos dados em presença de heterocedasticidade, gerando previsões mais confiáveis. A análise de inflação evidencia a posição do Brasil em um cenário global, com inflação moderada em relação a economias vizinhas como a Argentina.

## 📎 Arquivos

- `comparativo_ols_wls_inflacao2024.py` — Código-fonte do projeto.
- `comparativo_ols_wls_inflacao2024.png` — Imagem gerada com os gráficos comparativos.

## 🚀 Como executar

1. Clone o repositório:
   ```
   git clone https://github.com/SEU_USUARIO/comparativo-ols-wls-inflacao2024.git
   cd comparativo-ols-wls-inflacao2024
   ```

2. Instale as dependências:
   ```
   pip install numpy matplotlib scipy statsmodels
   ```

3. Execute o script:
   ```
   python comparativo_ols_wls_inflacao2024.py
   ```

---

**Criado por Walter Salles — Economista e Cientista de Dados em formação.**
