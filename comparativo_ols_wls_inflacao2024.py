import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.iolib.table import SimpleTable, default_txt_fmt

# Semente aleatória para reprodutibilidade
np.random.seed(1024)

# Simulação de dados
nsample = 50
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, (x - 5) ** 2))
X = sm.add_constant(X)
beta = [5.0, 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
w[nsample * 6 // 10 :] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e
X = X[:, [0, 1]]

# WLS - Weighted Least Squares
mod_wls = sm.WLS(y, X, weights=1.0 / (w ** 2))
res_wls = mod_wls.fit()

# OLS - Ordinary Least Squares
res_ols = sm.OLS(y, X).fit()

# Intervalos de previsão
pred_ols = res_ols.get_prediction()
iv_l_ols = pred_ols.summary_frame()["obs_ci_lower"]
iv_u_ols = pred_ols.summary_frame()["obs_ci_upper"]
pred_wls = res_wls.get_prediction()
iv_l = pred_wls.summary_frame()["obs_ci_lower"]
iv_u = pred_wls.summary_frame()["obs_ci_upper"]

# Dados de inflação 2024
inflation_data = {
    "Brasil": 4.5,
    "EUA": 3.2,
    "Zona do Euro": 2.6,
    "Reino Unido": 3.8,
    "Argentina": 276.0,
    "China": 0.7,
    "Japão": 2.8,
    "África do Sul": 5.3,
    "Índia": 4.8,
    "México": 4.6
}
countries = list(inflation_data.keys())
inflation_rates = list(inflation_data.values())

# Figura com dois gráficos
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1 - OLS vs WLS
ax1.plot(x, y, "o", label="Data")
ax1.plot(x, y_true, "b-", label="True")
ax1.plot(x, res_ols.fittedvalues, "r--")
ax1.plot(x, iv_u_ols, "r--", label="OLS")
ax1.plot(x, iv_l_ols, "r--")
ax1.plot(x, res_wls.fittedvalues, "g--.")
ax1.plot(x, iv_u, "g--", label="WLS")
ax1.plot(x, iv_l, "g--")
ax1.set_title("Comparação entre OLS e WLS")
ax1.legend(loc="best")

# Subplot 2 - Inflação 2024
bars = ax2.bar(countries, inflation_rates, color="skyblue", edgecolor="black")
ax2.axhline(y=4.5, color="red", linestyle="--", label="Brasil (4.5%)")
ax2.set_title("Taxa de Inflação (%) - Comparativo Internacional 2024")
ax2.set_ylabel("Inflação (%)")
ax2.set_xticks(range(len(countries)))
ax2.set_xticklabels(countries, rotation=45)
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=9)
ax2.legend()

plt.tight_layout()
plt.savefig("comparativo_ols_wls_inflacao2024.png")
plt.show()
