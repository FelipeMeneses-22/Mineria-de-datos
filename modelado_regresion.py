"""modelado_regresion.py
Predecir un NÚMERO (el costo) en insurance y Medicalpremium, comparando
regresión lineal, árbol de decisión y Random Forest.
Es el "siguiente paso" que propone el informe EDA de insurance.

  - insurance       -> charges      (costo médico)
  - Medicalpremium  -> PremiumPrice (precio de la prima)

Métricas (cuanto MENOR el error, mejor):
  MAE  = error promedio, en las mismas unidades del costo
  RMSE = como el MAE, pero castiga más los errores grandes
  R2   = qué parte de la variación del costo explica el modelo (1 = todo, 0 = nada)

Validación cruzada de 5 particiones: cada número es un promedio ± su variación.
El escalador va dentro de un Pipeline (se ajusta solo con el entrenamiento de cada partición).

Ejecutar desde la raíz del proyecto:  python modelado_regresion.py
Salida: data/resultados/modelado_regresion.csv y graficas/modelado_regresion.png
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import LIMPIO, RAIZ, SEMILLA

DATASETS = {"insurance": "charges", "Medicalpremium": "PremiumPrice"}
MUESTRA_INSURANCE = 50000      # insurance tiene 500.000 filas; con 50.000 el Random Forest corre en segundos
CARPETA_RESULTADOS = RAIZ / "data" / "resultados"
CARPETA_GRAFICAS = RAIZ / "graficas"


def main() -> None:
    metricas = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    validacion = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas = []

    for nombre, objetivo in DATASETS.items():
        df = pd.read_csv(LIMPIO / f"{nombre}_limpio.csv")
        if nombre == "insurance" and len(df) > MUESTRA_INSURANCE:
            df = df.sample(MUESTRA_INSURANCE, random_state=SEMILLA)
        y = df[objetivo]
        X = pd.get_dummies(df.drop(columns=objetivo), drop_first=True, dtype=int)
        print(f"\n{nombre}: {len(X)} filas | {X.shape[1]} variables | objetivo: {objetivo}")

        modelos = {
            "Siempre el promedio (referencia)": DummyRegressor(strategy="mean"),
            "Regresión lineal": LinearRegression(),
            "Árbol (prof. 4)": DecisionTreeRegressor(max_depth=4, random_state=SEMILLA),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8,
                                                   random_state=SEMILLA, n_jobs=-1),
        }
        for modelo_nombre, modelo in modelos.items():
            tuberia = make_pipeline(StandardScaler(), modelo)
            r = cross_validate(tuberia, X, y, cv=validacion, scoring=metricas)
            fila = {"dataset": nombre, "modelo": modelo_nombre}
            for m in metricas:
                signo = -1 if m in ("mae", "rmse") else 1       # sklearn devuelve el error en negativo
                fila[m] = signo * r["test_" + m].mean()
                fila[m + "_desv"] = r["test_" + m].std()
            filas.append(fila)
            print(f"  {modelo_nombre:<34} MAE {fila['mae']:>9,.0f} | RMSE {fila['rmse']:>9,.0f} | "
                  f"R2 {fila['r2']:.3f} ± {fila['r2_desv']:.3f}")

    tabla = pd.DataFrame(filas)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(CARPETA_RESULTADOS / "modelado_regresion.csv", index=False)

    # ---------- Gráfica: una fila por dataset, error (MAE) y R2 ----------
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
    for fila_ejes, (nombre, objetivo) in zip(ejes, DATASETS.items()):
        t = tabla[(tabla["dataset"] == nombre) & (~tabla["modelo"].str.startswith("Siempre"))].set_index("modelo")
        fila_ejes[0].barh(t.index, t["mae"], xerr=t["mae_desv"], color="#e45756", capsize=4)
        fila_ejes[0].set_title(f"{nombre}: error promedio (MAE) en {objetivo}\n(menos es mejor)")
        fila_ejes[1].barh(t.index, t["r2"], xerr=t["r2_desv"], color="#4c78a8", capsize=4)
        fila_ejes[1].set_title(f"{nombre}: R2 (más es mejor)")
        fila_ejes[1].set_xlim(0, 1)
        for eje in fila_ejes:
            eje.grid(axis="x", alpha=0.3)
    plt.suptitle("Comparación de modelos de regresión (promedio de 5 particiones)")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICAS / "modelado_regresion.png", dpi=150)
    plt.close()
    print("\nGuardado:", (CARPETA_RESULTADOS / "modelado_regresion.csv").relative_to(RAIZ),
          "y", (CARPETA_GRAFICAS / "modelado_regresion.png").relative_to(RAIZ))


if __name__ == "__main__":
    main()
