"""
modelado.py
Comparacion de tres clasificadores sobre adultos_salud.csv.
Objetivo: predecir si una persona tiene enfermedad cardiaca (HeartDisease).

Modelos:
  - Arbol de decision (max_depth=4)
  - Naive Bayes (GaussianNB)
  - K-Vecinos (k=5) con StandardScaler dentro de un Pipeline

Ubicacion: en la raiz del proyecto, al lado de main.py y transformaciones.py.

    proyecto_limpieza/
    |-- main.py
    |-- transformaciones.py
    |-- modelado.py              <- este archivo
    `-- data/
        |-- raw/adultos_salud.csv
        |-- limpio/adultos_salud_limpio.csv   (si existe, se usa este)
        `-- resultados/                       (se crea solo)

Ejecutar:  python modelado.py
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Rutas (todas salen de la carpeta donde esta este archivo)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
RUTA_LIMPIO = RAIZ / "data" / "limpio" / "adultos_salud_limpio.csv"
RUTA_RAW = RAIZ / "data" / "raw" / "adultos_salud.csv"
CARPETA_SALIDA = RAIZ / "data" / "resultados"

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------
OBJETIVO = "HeartDisease"   # columna a predecir (Yes / No)
POSITIVA = "Yes"            # la clase que nos interesa detectar
# El dataset tiene ~320.000 filas y KNN seria muy lento con todas.
# Con una muestra de 40.000 todo corre en poco tiempo.
# Ponga None para usar todas las filas (KNN puede tardar muchos minutos).
MUESTRA = 40000


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV limpio; si todavia no existe, usa el crudo."""
    if RUTA_LIMPIO.exists():
        print("Leyendo:", RUTA_LIMPIO.relative_to(RAIZ))
        return pd.read_csv(RUTA_LIMPIO)

    if RUTA_RAW.exists():
        print("AVISO: no existe adultos_salud_limpio.csv, se usa el crudo")
        print("       con una limpieza minima (duplicados y nulos).")
        df = pd.read_csv(RUTA_RAW)
        return df.drop_duplicates().dropna().reset_index(drop=True)

    raise FileNotFoundError(
        "No se encontro adultos_salud.csv. Copielo en data/raw/ "
        "(o el limpio en data/limpio/)."
    )


def preparar_x_y(df: pd.DataFrame):
    """Separa las variables predictoras (X) del objetivo (y)."""
    if MUESTRA is not None and len(df) > MUESTRA:
        df = df.sample(MUESTRA, random_state=42)
        print(f"Se usa una muestra de {MUESTRA} filas")

    y = df[OBJETIVO]
    # get_dummies convierte el texto (Yes/No, Sex, Race...) en columnas 0/1
    X = pd.get_dummies(df.drop(columns=OBJETIVO), drop_first=True, dtype=int)
    return X, y


def main() -> None:
    df = cargar_datos()
    print("Filas:", len(df), "| Columnas:", df.shape[1])
    print("\nDistribucion del objetivo:")
    print(df[OBJETIVO].value_counts(normalize=True).round(3))

    X, y = preparar_x_y(df)

    # stratify=y mantiene el mismo % de enfermos en train y en test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]} "
          f"| Variables en X: {X_train.shape[1]}")

    # Accuracy de un "modelo tonto" que siempre responde la clase mayoritaria
    base = y_test.value_counts(normalize=True).max()
    print(f"Accuracy si siempre se dice 'No': {base:.3f}  <- hay que superar esto")

    modelos = {
        "Arbol (prof=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN (k=5)": make_pipeline(StandardScaler(),
                                   KNeighborsClassifier(n_neighbors=5)),
    }

    # ------------------------------------------------------------------
    # Entrenar y evaluar cada modelo
    # ------------------------------------------------------------------
    filas = []
    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        duracion = time.perf_counter() - inicio

        pred = modelo.predict(X_test)
        filas.append({
            "modelo": nombre,
            "entrena_s": round(duracion, 4),
            "acc_train": round(modelo.score(X_train, y_train), 3),
            "acc_test": round(modelo.score(X_test, y_test), 3),
            "precision": round(precision_score(y_test, pred, pos_label=POSITIVA,
                                               zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, pos_label=POSITIVA), 3),
            "f1": round(f1_score(y_test, pred, pos_label=POSITIVA), 3),
        })

    resultados = pd.DataFrame(filas)
    print("\n=== Comparacion de modelos (test) ===")
    print(resultados.to_string(index=False))
    print("\nLectura: recall = de los enfermos reales, cuantos detecto el modelo.")
    print("Con clases desbalanceadas el accuracy solo no basta: mire recall y f1.")

    # ------------------------------------------------------------------
    # KNN con y sin escalar: por que la escala importa
    # ------------------------------------------------------------------
    knn_crudo = KNeighborsClassifier(n_neighbors=5)
    knn_crudo.fit(X_train, y_train)
    print(f"\nKNN k=5 SIN escalar -> accuracy test: "
          f"{knn_crudo.score(X_test, y_test):.3f}")
    print("(SleepTime, BMI y los dias de salud pesan mas que las columnas 0/1)")

    # ------------------------------------------------------------------
    # Efecto del valor de k
    # ------------------------------------------------------------------
    knn_k = make_pipeline(StandardScaler(), KNeighborsClassifier())
    print("\n   k   acc_train   acc_test   recall_test")
    for k in [1, 3, 5, 11, 31, 101]:
        knn_k.set_params(kneighborsclassifier__n_neighbors=k)
        knn_k.fit(X_train, y_train)
        rec = recall_score(y_test, knn_k.predict(X_test), pos_label=POSITIVA)
        print(f"{k:>4}   {knn_k.score(X_train, y_train):>9.3f}   "
              f"{knn_k.score(X_test, y_test):>8.3f}   {rec:>11.3f}")

    # ------------------------------------------------------------------
    # Guardar resultados: tabla CSV y matrices de confusion en PNG
    # ------------------------------------------------------------------
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(CARPETA_SALIDA / "comparacion_modelos.csv", index=False)

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    for eje, (nombre, modelo) in zip(ejes, modelos.items()):
        ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test,
                                              ax=eje, colorbar=False)
        eje.set_title(nombre)
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / "matrices_confusion.png", dpi=150)
    plt.close()

    print("\nArchivos guardados en:", CARPETA_SALIDA.relative_to(RAIZ))
    print(" - comparacion_modelos.csv")
    print(" - matrices_confusion.png")


if __name__ == "__main__":
    main()
