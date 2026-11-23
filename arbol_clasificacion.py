"""arbol_clasificacion.py
Árbol de decisión para predecir HeartDisease (Yes/No) en adultos_salud.
Es el mismo flujo de arbol_fuga.py (X/y, split, fit, score) más 5 gráficas.

Ejecutar desde la raíz del proyecto:  python arbol_clasificacion.py
Las imágenes quedan en graficas/adultos_salud/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

RAIZ = Path(__file__).resolve().parent
RUTA = RAIZ / "data" / "limpio" / "adultos_salud_limpio.csv"
CARPETA = RAIZ / "graficas" / "adultos_salud"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---------- 1. Datos: X (características) e y (lo que se predice) ----------
    df = pd.read_csv(RUTA)
    y = df["HeartDisease"]
    # get_dummies convierte el texto (Sex, Race, ...) en columnas 0/1
    X = pd.get_dummies(df.drop(columns="HeartDisease"))
    print("Filas x columnas de X:", X.shape)

    # ---------- 2. Separar train/test ANTES de entrenar ----------
    # stratify=y mantiene la misma proporción de Yes/No en los dos lados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # Si el modelo dijera siempre "No", ¿cuánto acertaría? (línea base)
    base = (y_test == "No").mean()
    print(f"Línea base (siempre 'No'): {base:.3f}")

    # ---------- Gráfica 1: distribución de la variable objetivo ----------
    conteo = y.value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(conteo.index, conteo.values, color=["#4c78a8", "#e45756"])
    for i, v in enumerate(conteo.values):
        plt.text(i, v, f"{v / len(y):.1%}", ha="center", va="bottom")
    plt.title("HeartDisease: muy pocas personas con enfermedad (clases desbalanceadas)")
    plt.ylabel("Cantidad de personas")
    plt.tight_layout()
    plt.savefig(CARPETA / "1_distribucion_objetivo.png", dpi=130)
    plt.close()

    # ---------- 3. Varias profundidades: ¿hay sobreajuste? ----------
    profundidades = [2, 3, 4, 5, 8, 12, None]
    etiquetas, acc_train, acc_test = [], [], []
    print("\nprofundidad  acc_train  acc_test  recall_Yes_test")
    for prof in profundidades:
        modelo = DecisionTreeClassifier(max_depth=prof, random_state=42)
        modelo.fit(X_train, y_train)
        a_tr = modelo.score(X_train, y_train)
        a_te = modelo.score(X_test, y_test)
        rec = recall_score(y_test, modelo.predict(X_test), pos_label="Yes")
        nombre = str(prof) if prof is not None else "sin límite"
        etiquetas.append(nombre)
        acc_train.append(a_tr)
        acc_test.append(a_te)
        print(f"{nombre:>10}  {a_tr:>9.3f}  {a_te:>8.3f}  {rec:>14.3f}")

    # ---------- Gráfica 2: train vs test ----------
    plt.figure(figsize=(8, 5))
    plt.plot(etiquetas, acc_train, marker="o", label="Datos de entrenamiento")
    plt.plot(etiquetas, acc_test, marker="o", label="Datos de prueba")
    plt.axhline(base, color="gray", linestyle="--", label="Responder siempre 'No'")
    plt.xlabel("Profundidad del árbol (preguntas seguidas)")
    plt.ylabel("Aciertos totales (accuracy)")
    plt.title("¿Aprende o memoriza? Entrenamiento sube, prueba no")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CARPETA / "2_sobreajuste.png", dpi=130)
    plt.close()

    # ---------- 4. Árbol final legible (profundidad 3 para que se lea) ----------
    arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    arbol.fit(X_train, y_train)
    print(f"\nÁrbol final (max_depth=3): train {arbol.score(X_train, y_train):.3f}"
          f" | test {arbol.score(X_test, y_test):.3f}")

    # ---------- Gráfica 3: el árbol dibujado ----------
    plt.figure(figsize=(22, 9))
    plot_tree(arbol, feature_names=list(X.columns),
              class_names=list(arbol.classes_),   # orden: ['No', 'Yes']
              filled=True, rounded=True, fontsize=9)
    plt.title("Árbol de decisión para HeartDisease (max_depth=3)")
    plt.savefig(CARPETA / "3_arbol.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---------- Gráfica 4: importancia de variables ----------
    imp = pd.Series(arbol.feature_importances_, index=X.columns)
    imp = imp[imp > 0].sort_values()              # solo las que el árbol usó
    plt.figure(figsize=(9, 5))
    plt.barh(imp.index, imp.values)
    plt.xlabel("Importancia (qué tanto usó el árbol esta variable)")
    plt.title("Qué variables usó el árbol")
    plt.tight_layout()
    plt.savefig(CARPETA / "4_importancia.png", dpi=130)
    plt.close()

    # ---------- Gráfica 5: matriz de confusión ----------
    # Se compara el árbol normal contra uno con class_weight="balanced",
    # que le da más peso a la clase rara (Yes).
    balanceado = DecisionTreeClassifier(max_depth=3, random_state=42,
                                        class_weight="balanced")
    balanceado.fit(X_train, y_train)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5.5))
    for eje, modelo, titulo in [(ejes[0], arbol, "Árbol normal"),
                                (ejes[1], balanceado, "Árbol balanceado")]:
        ConfusionMatrixDisplay.from_estimator(
            modelo, X_test, y_test, normalize="true",   # % por fila real
            values_format=".0%", ax=eje, colorbar=False,
            display_labels=["Sin enfermedad", "Con enfermedad"])
        rec = recall_score(y_test, modelo.predict(X_test), pos_label="Yes")
        eje.set_title(f"{titulo}\naciertos totales {modelo.score(X_test, y_test):.1%}"
                      f" | enfermos detectados {rec:.0%}", fontsize=12)
        eje.set_xlabel("Lo que predijo el árbol")
        eje.set_ylabel("Lo que realmente ocurrió")
        print(f"{titulo}: accuracy {modelo.score(X_test, y_test):.3f}"
              f" | recall Yes {rec:.3f}")
    plt.tight_layout()
    plt.savefig(CARPETA / "5_matriz_confusion.png", dpi=130)
    plt.close()

    print("\nGráficas guardadas en:", CARPETA)


if __name__ == "__main__":
    main()
