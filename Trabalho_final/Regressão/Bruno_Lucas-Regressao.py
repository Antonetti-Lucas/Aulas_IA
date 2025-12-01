import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def analise_regressao():

    # -------------- Gerador de dados fictícios (MUAC e BMI) -------------------
    np.random.seed(42)

    # Simula MUAC entre 18 e 38 cm
    X = np.random.normal(loc=27, scale=4, size=1373)
    X = np.clip(X, 18, 38)

    # Modelo aproximado do paper, BMI = -0.042 + 0.972·MUAC + erro
    erro = np.random.normal(0, 2.1, size=1373)
    Y = -0.042 + 0.972 * X + erro

    X = X.round(2)
    Y = Y.round(2)

    b0 = -0.042
    b1 = 0.972
    Y_hat = b0 + b1 * X

    # Resíduos e métricas
    residuos = Y - Y_hat
    abs_res = np.abs(residuos)
    sq_res = residuos**2

    MAE = abs_res.mean()
    MSE = sq_res.mean()
    RMSE = sqrt(MSE)

    Y_bar = Y.mean()
    SSE = sq_res.sum()
    SST = ((Y - Y_bar)**2).sum()
    R2 = 1 - SSE / SST

    # Métricas finais
    metrics = {
        "MAE": round(MAE, 4),
        "MSE": round(MSE, 4),
        "RMSE": round(RMSE, 4),
        "R2": round(R2, 4),
        "SSE (∑Erro²)": round(SSE, 4),
        "SST (∑(Y - Ȳ)²)": round(SST, 4),
        "Ȳ": round(Y_bar, 4)
    }

    print("\n----- Métricas de Erro -----")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # -------------------- PLOTS ---------------------------
    # Gráfico de dispersão e reta de regressão
    plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, color="blue", label="Valores Reais")
    plt.plot(X, Y_hat, color="red", label="Reta de Regressão (Ŷ)")
    plt.xlabel("MUAC - Circunferência do Braço (cm)")
    plt.ylabel("BMI - Índice de Massa Corporal (kg/m²)")
    plt.title("Regressão Linear - Defensivo")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de resíduos
    plt.figure(figsize=(8, 5))
    plt.scatter(X, residuos, color="purple")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("MUAC - Circunferência do Braço (cm)")
    plt.ylabel("Resíduos (BMI Real - BMI Predito)")
    plt.title("Análise de Resíduos")
    plt.grid(True)
    plt.show()

    return metrics


if __name__ == "__main__":
    analise_regressao()
