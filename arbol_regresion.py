"""arbol_regresion.py
Árboles de decisión de REGRESIÓN: el objetivo es un número (no Yes/No).
  - insurance.csv        -> predecir charges
  - Medicalpremium.csv   -> predecir PremiumPrice
Mismo flujo que arbol_fuga.py, pero con DecisionTreeRegressor y R2 en vez
de accuracy.

Ejecutar desde la raíz del proyecto:  python arbol_regresion.py
Las imágenes quedan en graficas/insurance/ y graficas/Medicalpremium/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

RAIZ = Path(__file__).resolve().parent

DATASETS = {
    "insurance": "charges",
    "Medicalpremium": "PremiumPrice",
}


def analizar(nombre: str, objetivo: str) -> None:
    carpeta = RAIZ / "graficas" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 55}\n{nombre}  (objetivo: {objetivo})\n{'=' * 55}")

    # ---------- 1. X e y ----------
    df = pd.read_csv(RAIZ / "data" / "limpio" / f"{nombre}_limpio.csv")
    y = df[objetivo]
    X = pd.get_dummies(df.drop(columns=objetivo))   # texto -> columnas 0/1
    print("Filas x columnas de X:", X.shape)

    # ---------- 2. Separar train/test ANTES de entrenar ----------
    # (en regresión no hay clases, así que no se usa stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # ---------- 3. Varias profundidades ----------
    profundidades = [2, 3, 4, 5, 6, 8, 12, None]
    etiquetas, r2_train, r2_test = [], [], []
    print("\nprofundidad   R2_train   R2_test")
    for prof in profundidades:
        modelo = DecisionTreeRegressor(max_depth=prof, random_state=42)
        modelo.fit(X_train, y_train)
        etiqueta = str(prof) if prof is not None else "sin límite"
        etiquetas.append(etiqueta)
        r2_train.append(modelo.score(X_train, y_train))   # score() = R2
        r2_test.append(modelo.score(X_test, y_test))
        print(f"{etiqueta:>10}   {r2_train[-1]:>8.3f}   {r2_test[-1]:>7.3f}")

    # ---------- Gráfica 1: ¿aprende o memoriza? ----------
    mejor = r2_test.index(max(r2_test))             # posición del mejor R2 en prueba
    plt.figure(figsize=(9, 5.5))
    plt.plot(etiquetas, r2_train, marker="o", linewidth=2.5,
             label="Datos de entrenamiento (los que el árbol ya vio)")
    plt.plot(etiquetas, r2_test, marker="o", linewidth=2.5,
             label="Datos de prueba (datos nuevos)")
    plt.fill_between(range(mejor, len(etiquetas)), r2_test[mejor:], r2_train[mejor:],
                     color="red", alpha=0.10, label="Zona donde memoriza")
    plt.annotate(f"Mejor punto: profundidad {etiquetas[mejor]}",
                 xy=(mejor, r2_test[mejor]), xytext=(mejor - 1.0, r2_test[mejor] - 0.24),
                 arrowprops={"arrowstyle": "->"}, fontsize=12)
    plt.xlabel("Profundidad del árbol (cantidad de preguntas seguidas)", fontsize=12)
    plt.ylabel("Qué tan bien acierta (R²: 1 = perfecto)", fontsize=12)
    plt.title(f"{nombre}: ¿el árbol aprende o memoriza?", fontsize=14)
    plt.ylim(top=1.2)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(carpeta / "1_sobreajuste.png", dpi=150)
    plt.close()

    # ---------- 4. Árbol final (profundidad 3 para que se lea) ----------
    arbol = DecisionTreeRegressor(max_depth=3, random_state=42)
    arbol.fit(X_train, y_train)
    print(f"\nÁrbol final (max_depth=3): R2 train {arbol.score(X_train, y_train):.3f}"
          f" | R2 test {arbol.score(X_test, y_test):.3f}")

    # ---------- Gráfica 2: el árbol dibujado ----------
    plt.figure(figsize=(22, 9))
    plot_tree(arbol, feature_names=list(X.columns), filled=True,
              rounded=True, fontsize=9, precision=0)
    plt.title(f"Árbol de regresión: {objetivo} (max_depth=3)")
    plt.savefig(carpeta / "2_arbol.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---------- Gráfica 3: importancia de variables ----------
    imp = pd.Series(arbol.feature_importances_, index=X.columns)
    imp = imp[imp > 0].sort_values()
    plt.figure(figsize=(8, 4))
    plt.barh(imp.index, imp.values)
    plt.xlabel("Importancia (qué tanto usó el árbol esta variable)")
    plt.title(f"{nombre}: qué variables usó el árbol")
    plt.tight_layout()
    plt.savefig(carpeta / "3_importancia.png", dpi=130)
    plt.close()

    # ---------- Gráfica 4: costo real vs costo que predice el árbol ----------
    pred = arbol.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, pred, s=6, alpha=0.3)
    limites = [y_test.min(), y_test.max()]
    plt.plot(limites, limites, color="red", linestyle="--", linewidth=2,
             label="Predicción perfecta")
    plt.text(0.03, 0.72,
             f"Cada fila de puntos es una hoja del árbol:\n"
             f"{arbol.get_n_leaves()} hojas = solo {arbol.get_n_leaves()} valores posibles",
             transform=plt.gca().transAxes, ha="left", fontsize=11,
             bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "gray"})
    miles = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    plt.gca().xaxis.set_major_formatter(miles)
    plt.gca().yaxis.set_major_formatter(miles)
    plt.xlabel(f"{objetivo} real (lo que ocurrió)", fontsize=12)
    plt.ylabel(f"{objetivo} predicho (lo que dice el árbol)", fontsize=12)
    plt.title(f"{nombre}: valor real vs valor predicho", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(carpeta / "4_real_vs_predicho.png", dpi=150)
    plt.close()


def main() -> None:
    for nombre, objetivo in DATASETS.items():
        analizar(nombre, objetivo)
    print("\nListo. Revisa la carpeta graficas/")


if __name__ == "__main__":
    main()
