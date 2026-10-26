"""Sesion 7 · Primer arbol de decision para predecir la fuga en TelecomUNO.

Entrena DecisionTreeClassifier sobre telecom_clientes_limpio.csv:
  1. Prepara X e y (codifica tipo_plan y pago_automatico con get_dummies).
  2. Separa train/test ANTES de ajustar (regla de oro).
  3. Compara varias profundidades para ver el sobreajuste.
  4. Imprime las reglas del arbol final con export_text.
  5. Muestra la importancia de las variables.

Ejecutar:  python3 arbol_fuga.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

RUTA = Path(__file__).resolve().parents[2] / "datasets_base" / "telecom_clientes_limpio.csv"


def main() -> None:
    df = pd.read_csv(RUTA)

    # --- Chequeo de calidad ANTES de modelar -----------------------------
    print("Filas x columnas:", df.shape)
    print("Nulos totales:", int(df.isna().sum().sum()))
    print("Duplicados:", int(df.duplicated().sum()))

    # --- X / y ------------------------------------------------------------
    columnas_numero = ["edad", "facturacion_mensual", "antiguedad_meses",
                       "minutos_llamada", "gb_datos", "num_quejas"]
    X = pd.get_dummies(
        df[columnas_numero + ["tipo_plan", "pago_automatico"]],
        columns=["tipo_plan", "pago_automatico"],
    )
    y = df["fuga"]
    print("\nColumnas de X:", list(X.columns))

    # --- Train / test ANTES de fit ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} clientes | Test: {X_test.shape[0]} clientes")

    # --- Varias profundidades: efecto del sobreajuste --------------------
    print("\nprofundidad   accuracy_train   accuracy_test")
    for prof in [2, 3, 4, 5, 8, None]:
        arbol = DecisionTreeClassifier(max_depth=prof, random_state=42)
        arbol.fit(X_train, y_train)
        acc_tr = arbol.score(X_train, y_train)
        acc_te = arbol.score(X_test, y_test)
        etiqueta = str(prof) if prof is not None else "sin limite"
        print(f"{etiqueta:>10}   {acc_tr:>14.3f}   {acc_te:>13.3f}")

    # --- Arbol final legible + reglas ------------------------------------
    arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
    arbol.fit(X_train, y_train)
    print(f"\nArbol final (max_depth=4): train {arbol.score(X_train, y_train):.3f}"
          f" | test {arbol.score(X_test, y_test):.3f}")
    print("\nReglas del arbol:")
    print(export_text(arbol, feature_names=list(X.columns)))

    # --- Importancia de variables ----------------------------------------
    importancias = pd.Series(arbol.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False)
    print("Importancia de variables (top 6):")
    for nombre, valor in importancias.head(6).items():
        print(f"  {nombre:<28} {valor:.3f}")


if __name__ == "__main__":
    main()
