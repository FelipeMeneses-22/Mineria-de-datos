"""modelado_justo.py
Comparación de clasificadores sobre adultos_salud (objetivo: HeartDisease)
EN IGUALDAD DE CONDICIONES. Mejora la comparación de modelado.py:

  1. Todos los modelos reciben el mismo trato frente al desbalance
     (solo el 9 % de las personas tiene la enfermedad):
       - Árbol y regresión logística: class_weight="balanced".
       - Naive Bayes: priors=[0.5, 0.5] (las dos clases pesan igual).
       - KNN no tiene esa opción: queda como viene (se avisa en el resumen).
  2. Validación cruzada de 5 particiones: cada número es un promedio ± su
     variación, no una sola prueba.
  3. Se miden recall y F1 (no solo los aciertos), y además ROC AUC y AP, que no
     dependen del umbral de decisión.
  4. El escalador va DENTRO de un Pipeline: en cada partición se ajusta solo con
     los datos de entrenamiento (así no se filtra información de la prueba).

Ejecutar desde la raíz del proyecto:  python modelado_justo.py
Salida: data/resultados/modelado_justo.csv y graficas/modelado_justo.png
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import LIMPIO, RAIZ, SEMILLA

MUESTRA = 40000          # igual que modelado.py; None = todas las filas (KNN tarda muchísimo)
CARPETA_RESULTADOS = RAIZ / "data" / "resultados"
CARPETA_GRAFICAS = RAIZ / "graficas"


def main() -> None:
    df = pd.read_csv(LIMPIO / "adultos_salud_limpio.csv")
    if MUESTRA is not None and len(df) > MUESTRA:
        df = df.sample(MUESTRA, random_state=SEMILLA)
    y = (df["HeartDisease"] == "Yes").astype(int)
    X = pd.get_dummies(df.drop(columns="HeartDisease"), drop_first=True, dtype=int)
    print(f"Personas: {len(X)} | variables: {X.shape[1]} | con enfermedad: {y.mean():.1%}")

    modelos = {
        "Siempre 'No' (referencia)": DummyClassifier(strategy="most_frequent"),
        "Árbol (prof. 4, balanced)": DecisionTreeClassifier(max_depth=4, random_state=SEMILLA,
                                                            class_weight="balanced"),
        "Naive Bayes (50/50)": GaussianNB(priors=[0.5, 0.5]),
        "Regresión logística (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "KNN (k=31)": KNeighborsClassifier(n_neighbors=31),
    }
    # zero_division=0: el modelo de referencia nunca predice "enfermo" y no debe dar avisos
    metricas = {"aciertos": "accuracy", "recall": "recall",
                "precision": make_scorer(precision_score, zero_division=0),
                "f1": make_scorer(f1_score, zero_division=0),
                "roc_auc": "roc_auc", "ap": "average_precision"}
    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    filas = []
    for nombre, modelo in modelos.items():
        # El escalador va dentro del Pipeline: se ajusta solo con el entrenamiento de cada partición
        tuberia = make_pipeline(StandardScaler(), modelo)
        r = cross_validate(tuberia, X, y, cv=validacion, scoring=metricas, n_jobs=-1)
        fila = {"modelo": nombre}
        for m in metricas:
            fila[m] = r["test_" + m].mean()
            fila[m + "_desv"] = r["test_" + m].std()
        filas.append(fila)
        print(f"  listo: {nombre}")

    tabla = pd.DataFrame(filas)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(CARPETA_RESULTADOS / "modelado_justo.csv", index=False)

    print("\n=== Validación cruzada de 5 particiones: promedio ± desviación ===")
    for _, f in tabla.iterrows():
        print(f"{f['modelo']:<32} aciertos {f['aciertos']:.3f} | "
              f"recall {f['recall']:.3f} ± {f['recall_desv']:.3f} | "
              f"precisión {f['precision']:.3f} | F1 {f['f1']:.3f} ± {f['f1_desv']:.3f} | "
              f"ROC AUC {f['roc_auc']:.3f} ± {f['roc_auc_desv']:.3f}")
    print("\nLectura: recall = de los enfermos reales, cuántos detecta; ROC AUC = qué tan bien")
    print("ordena a los enfermos por encima de los sanos (0,5 = al azar, 1 = perfecto).")
    print("Aviso: KNN no puede balancear las clases, por eso su recall no es comparable de forma exacta.")

    # ---------- Gráfica ----------
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    real = tabla[tabla["modelo"] != "Siempre 'No' (referencia)"].set_index("modelo")
    fig, ejes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for eje, (col, titulo) in zip(ejes, [("recall", "Enfermos detectados (recall)"),
                                         ("f1", "F1"), ("roc_auc", "ROC AUC")]):
        eje.barh(real.index, real[col], xerr=real[col + "_desv"], color="#4c78a8", capsize=4)
        eje.set_title(titulo)
        eje.set_xlim(0, 1)
        eje.grid(axis="x", alpha=0.3)
    ejes[1].set_xlabel("Promedio de 5 particiones (la barra negra es la variación)")
    plt.suptitle("Comparación de modelos en igualdad de condiciones (adultos_salud)")
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICAS / "modelado_justo.png", dpi=150)
    plt.close()
    print("\nGuardado:", (CARPETA_RESULTADOS / "modelado_justo.csv").relative_to(RAIZ),
          "y", (CARPETA_GRAFICAS / "modelado_justo.png").relative_to(RAIZ))


if __name__ == "__main__":
    main()
