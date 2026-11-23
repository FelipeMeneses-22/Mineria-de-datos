"""modelado_unificado.py
Compara árbol de decisión, Naive Bayes y KNN sobre la MISMA tabla unificada
y la MISMA partición entrenamiento/prueba que usa arbol_general.py.

Por qué existe: modelado.py compara los tres modelos solo sobre adultos_salud
(con una muestra de 40.000 filas y un árbol sin class_weight). Aquí la
comparación es "en igualdad de condiciones" con el árbol único del proyecto
(sesión 8: mismos datos, misma partición, misma semilla).

Ejecutar desde la raíz del proyecto:  python modelado_unificado.py
Salida: data/resultados/comparacion_unificada.csv y graficas/modelado/
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from arbol_general import SEMILLA, VARIABLES, construir_tabla_unificada

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "data" / "resultados"
GRAFICAS = RAIZ / "graficas" / "modelado"
SALIDA.mkdir(parents=True, exist_ok=True)
GRAFICAS.mkdir(parents=True, exist_ok=True)


def metricas(modelo, X_train, y_train, X_test, y_test, segundos):
    """Una fila de la tabla comparativa."""
    pred = modelo.predict(X_test)
    return {
        "entrena_s": round(segundos, 4),
        "acc_train": round(modelo.score(X_train, y_train), 3),
        "acc_test": round(modelo.score(X_test, y_test), 3),
        "precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
    }


def main() -> None:
    tabla, tasa, _, _ = construir_tabla_unificada()
    X, y = tabla[VARIABLES], tabla["riesgo_alto"]

    # Misma partición que arbol_general.py (mezcla de origen + clase)
    grupo = tabla["origen"] + "_" + y.astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEMILLA, stratify=grupo)
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    print(f"Aciertos de un modelo que siempre dice 'riesgo normal': "
          f"{1 - y_test.mean():.3f}  <- hay que superar esto\n")

    modelos = {
        "Árbol (prof=4, balanced)": DecisionTreeClassifier(
            max_depth=4, random_state=SEMILLA, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        # Equivale al class_weight="balanced" del árbol: la clase de riesgo alto
        # es solo el 9 %, y con ese a priori Naive Bayes casi nunca la predice.
        "Naive Bayes (priors 50/50)": GaussianNB(priors=[0.5, 0.5]),
        "KNN (k=5)": make_pipeline(StandardScaler(),
                                   KNeighborsClassifier(n_neighbors=5)),
    }

    filas = []
    for nombre, modelo in modelos.items():
        t0 = time.perf_counter()
        modelo.fit(X_train, y_train)
        seg = time.perf_counter() - t0
        filas.append({"modelo": nombre,
                      **metricas(modelo, X_train, y_train, X_test, y_test, seg)})
    comparacion = pd.DataFrame(filas)
    print("=== Comparación (datos de prueba) ===")
    print(comparacion.to_string(index=False))
    comparacion.to_csv(SALIDA / "comparacion_unificada.csv", index=False)

    # ---- KNN con y sin escalar --------------------------------------
    crudo = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    print(f"\nKNN k=5 SIN escalar -> aciertos prueba: {crudo.score(X_test, y_test):.3f}"
          f" | recall {recall_score(y_test, crudo.predict(X_test)):.3f}")

    # ---- Efecto de k -------------------------------------------------
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
    ks = [1, 3, 5, 11, 31, 51]
    tabla_k = []
    print("\n   k   acc_train   acc_test   recall   f1")
    for k in ks:
        knn.set_params(kneighborsclassifier__n_neighbors=k).fit(X_train, y_train)
        pred = knn.predict(X_test)
        fila = {"k": k, "acc_train": knn.score(X_train, y_train),
                "acc_test": knn.score(X_test, y_test),
                "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred)}
        tabla_k.append(fila)
        print(f"{k:>4}   {fila['acc_train']:>9.3f}   {fila['acc_test']:>8.3f}"
              f"   {fila['recall']:>6.3f}   {fila['f1']:.3f}")
    tabla_k = pd.DataFrame(tabla_k)
    tabla_k.round(3).to_csv(SALIDA / "knn_efecto_k.csv", index=False)

    # ---- Recall por dataset de origen (como en arbol_general.py) ----
    origen_test = tabla.loc[X_test.index, "origen"]
    por_origen = {}
    for nombre, modelo in modelos.items():
        pred = pd.Series(modelo.predict(X_test), index=X_test.index)
        por_origen[nombre] = {
            o: recall_score(y_test[origen_test == o], pred[origen_test == o])
            for o in ["adultos_salud", "insurance", "Medicalpremium"]}
    por_origen = pd.DataFrame(por_origen)
    print("\nRiesgo alto detectado (recall) por dataset de origen:")
    print(por_origen.round(3).to_string())

    # ---- Gráfica 1: métricas de los tres modelos ---------------------
    comparacion.set_index("modelo")[["precision", "recall", "f1"]].plot(
        kind="bar", figsize=(9, 5), color=["#4c78a8", "#e45756", "#54a24b"])
    plt.title("Árbol, Naive Bayes y KNN sobre la misma partición")
    plt.ylabel("Valor (más alto = mejor)")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(GRAFICAS / "1_comparacion_modelos.png", dpi=150)
    plt.close()

    # ---- Gráfica 2: efecto de k --------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(tabla_k["k"], tabla_k["acc_train"], marker="o", label="Entrenamiento")
    plt.plot(tabla_k["k"], tabla_k["acc_test"], marker="o", label="Prueba")
    plt.plot(tabla_k["k"], tabla_k["recall"], marker="s", linestyle="--",
             label="Riesgo alto detectado (prueba)")
    plt.xscale("log")
    plt.xticks(ks, ks)
    plt.xlabel("k (cantidad de vecinos, escala logarítmica)")
    plt.ylabel("Proporción")
    plt.title("KNN: entre memorizar (k=1) y diluirse (k grande)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFICAS / "2_efecto_de_k.png", dpi=150)
    plt.close()

    print("\nGuardado en:", SALIDA.relative_to(RAIZ), "y", GRAFICAS.relative_to(RAIZ))


if __name__ == "__main__":
    main()
