"""arbol_general.py
UN SOLO árbol de decisión que junta los tres datasets del proyecto:
adultos_salud, insurance y Medicalpremium.

Problema: cada dataset predice algo distinto y tiene columnas distintas.
Solución (cada decisión está explicada en el cuaderno proyecto_mineria.ipynb):
  1. Solo se usan variables que existen (o se pueden calcular) en al menos
     dos datasets: edad, imc, hombre, fuma, diabetes.
  2. Se crea un objetivo común, "riesgo_alto" (1 = sí, 0 = no):
       adultos_salud  -> tiene enfermedad cardíaca
       insurance      -> costo médico (charges) entre los más altos
       Medicalpremium -> prima (PremiumPrice) entre las más altas
     "Los más altos" se define con el MISMO porcentaje que tiene la
     enfermedad en adultos_salud (9 %), para que el árbol no pueda acertar
     solo adivinando de qué dataset viene cada fila.
  3. Se toma una muestra del mismo tamaño de cada dataset.

Ejecutar desde la raíz del proyecto:  python arbol_general.py
Las imágenes quedan en graficas/arbol_general/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from config import FILAS_POR_DATASET, LIMPIO, RAIZ, SEMILLA, VARIABLES_ARBOL as VARIABLES

CARPETA = RAIZ / "graficas" / "arbol_general"
CARPETA.mkdir(parents=True, exist_ok=True)
CARPETA_RESULTADOS = RAIZ / "data" / "resultados"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)


def armar_tablas():
    adultos = pd.read_csv(LIMPIO / "adultos_salud_limpio.csv")
    insurance = pd.read_csv(LIMPIO / "insurance_limpio.csv")
    medical = pd.read_csv(LIMPIO / "Medicalpremium_limpio.csv")

    # Porcentaje de "riesgo alto" que se usará en los tres datasets
    tasa = (adultos["HeartDisease"] == "Yes").mean()

    # adultos_salud trae la edad en rangos ("18-24"): se usa el punto medio
    punto_medio = {rango: (int(rango[:2]) + int(rango[3:5])) / 2
                   for rango in adultos["AgeCategory"].unique() if rango[0] != "8"}
    punto_medio["80 or older"] = 82

    t_adultos = pd.DataFrame({
        "origen": "adultos_salud",
        "edad": adultos["AgeCategory"].map(punto_medio),
        "imc": adultos["BMI"],
        "hombre": (adultos["Sex"] == "Male").astype(int),
        "fuma": (adultos["Smoking"] == "Yes").astype(int),
        "diabetes": adultos["Diabetic"].str.startswith("Yes").astype(int),
        "riesgo_alto": (adultos["HeartDisease"] == "Yes").astype(int),
    })
    umbral_costo = insurance["charges"].quantile(1 - tasa)
    t_insurance = pd.DataFrame({
        "origen": "insurance",
        "edad": insurance["age"],
        "imc": insurance["bmi"],
        "hombre": (insurance["sex"] == "male").astype(int),
        "fuma": (insurance["smoker"] == "yes").astype(int),
        "diabetes": -1,                                   # -1 = dato no disponible
        "riesgo_alto": (insurance["charges"] >= umbral_costo).astype(int),
    })
    umbral_prima = medical["PremiumPrice"].quantile(1 - tasa)
    t_medical = pd.DataFrame({
        "origen": "Medicalpremium",
        "edad": medical["Age"],
        "imc": medical["Weight"] / (medical["Height"] / 100) ** 2,   # IMC calculado
        "hombre": -1,                                     # no disponible
        "fuma": -1,                                       # no disponible
        "diabetes": medical["Diabetes"],
        "riesgo_alto": (medical["PremiumPrice"] >= umbral_prima).astype(int),
    })

    return (t_adultos, t_insurance, t_medical), tasa, umbral_costo, umbral_prima


def construir_tabla_unificada():
    """Devuelve (tabla ya muestreada con la semilla 42, tasa, umbral_costo, umbral_prima).
    Es la función que usan otros scripts (por ejemplo modelado_unificado.py)."""
    tablas, tasa, umbral_costo, umbral_prima = armar_tablas()
    return muestrear(tablas), tasa, umbral_costo, umbral_prima


def muestrear(tablas, semilla=SEMILLA):
    """Muestra del mismo tamaño de cada dataset para que ninguno domine."""
    n = min(FILAS_POR_DATASET, *[len(t) for t in tablas])
    return pd.concat([t.sample(n=n, random_state=semilla) for t in tablas], ignore_index=True)


def evaluar_estabilidad(tablas, repeticiones=20):
    """Repite TODO (muestra, separación y árbol) con semillas distintas.
    Sirve para dar un rango y no un solo número: la prueba es pequeña."""
    filas = []
    for semilla in range(repeticiones):
        tabla = muestrear(tablas, semilla)
        X, y = tabla[VARIABLES], tabla["riesgo_alto"]
        grupo = tabla["origen"] + "_" + y.astype(str)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                  random_state=semilla, stratify=grupo)
        arbol = DecisionTreeClassifier(max_depth=4, random_state=SEMILLA,
                                       class_weight="balanced").fit(X_tr, y_tr)
        pred = arbol.predict(X_te)
        origen = tabla.loc[X_te.index, "origen"]
        fila = {"semilla": semilla,
                "aciertos": arbol.score(X_te, y_te),
                "riesgo_alto_detectado": recall_score(y_te, pred),
                "precision": precision_score(y_te, pred, zero_division=0),
                "f1": f1_score(y_te, pred)}
        for nombre in ["adultos_salud", "insurance", "Medicalpremium"]:
            m = origen == nombre
            fila["detectado_" + nombre] = recall_score(y_te[m], pred[m])
        filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    tablas, tasa, umbral_costo, umbral_prima = armar_tablas()
    tabla = muestrear(tablas)
    print(f"Porcentaje de riesgo alto usado en los tres datasets: {tasa:.1%}")
    print(f"  insurance: charges >= {umbral_costo:,.0f} | Medicalpremium: prima >= {umbral_prima:,.0f}")
    print("\nTabla unificada:", tabla.shape)
    print(tabla.groupby("origen")["riesgo_alto"].agg(filas="count", riesgo_alto="sum"))

    # ---------- Gráfica 1: composición de la tabla unificada ----------
    resumen = tabla.groupby(["origen", "riesgo_alto"]).size().unstack(fill_value=0)
    resumen.columns = ["Riesgo normal", "Riesgo alto"]
    resumen.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#f58518", "#4c78a8"])
    plt.title("Tabla unificada: la misma cantidad de filas de cada dataset")
    plt.xlabel("Dataset de origen")
    plt.ylabel("Cantidad de personas")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CARPETA / "1_composicion.png", dpi=150)
    plt.close()

    # ---------- X / y y separación entrenamiento / prueba ----------
    X = tabla[VARIABLES]
    y = tabla["riesgo_alto"]
    grupo = tabla["origen"] + "_" + y.astype(str)       # asegura mezcla en ambos lados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEMILLA, stratify=grupo)
    print(f"\nEntrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    # ---------- Elegir la profundidad (solo con datos de entrenamiento) ----------
    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    profundidades = [2, 3, 4, 5, 6, 8]
    medias, desvios = [], []
    print("\nprofundidad   F1 (validación cruzada)")
    for prof in profundidades:
        modelo = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA,
                                        class_weight="balanced")
        puntajes = cross_val_score(modelo, X_train, y_train, cv=validacion, scoring="f1")
        medias.append(puntajes.mean())
        desvios.append(puntajes.std())
        print(f"{prof:>10}   {puntajes.mean():.3f} ± {puntajes.std():.3f}")

    # ---------- Gráfica 2: elegir la profundidad ----------
    plt.figure(figsize=(8, 5))
    plt.errorbar(profundidades, medias, yerr=desvios, marker="o", linewidth=2.5, capsize=4)
    plt.xlabel("Profundidad del árbol (cantidad de preguntas seguidas)")
    plt.ylabel("F1 en validación cruzada (más alto = mejor)")
    plt.title("¿Cuántas preguntas seguidas conviene hacer?")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CARPETA / "2_elegir_profundidad.png", dpi=150)
    plt.close()

    # ---------- EL árbol ----------
    arbol = DecisionTreeClassifier(max_depth=4, random_state=SEMILLA, class_weight="balanced")
    arbol.fit(X_train, y_train)
    prediccion = arbol.predict(X_test)
    print(f"\nÁrbol final (4 preguntas seguidas): aciertos prueba {arbol.score(X_test, y_test):.3f}"
          f" | riesgo alto detectado {recall_score(y_test, prediccion):.1%}"
          f" | precisión {precision_score(y_test, prediccion):.1%}"
          f" | F1 {f1_score(y_test, prediccion):.3f}")
    print("\nReglas del árbol:")
    print(export_text(arbol, feature_names=VARIABLES))

    # ---------- Gráfica 3: el árbol dibujado ----------
    plt.figure(figsize=(26, 11))
    plot_tree(arbol, feature_names=VARIABLES, class_names=["Riesgo normal", "Riesgo alto"],
              filled=True, rounded=True, fontsize=9, precision=1)
    plt.title("Árbol de decisión único del proyecto (4 preguntas seguidas)")
    plt.savefig(CARPETA / "3_arbol_unico.png", dpi=130, bbox_inches="tight")
    plt.close()

    # ---------- Gráfica 4: importancia de variables ----------
    importancia = pd.Series(arbol.feature_importances_, index=VARIABLES).sort_values()
    plt.figure(figsize=(8, 4))
    plt.barh(importancia.index, importancia.values, color="#4c78a8")
    plt.xlabel("Importancia (qué tanto usó el árbol esta variable)")
    plt.title("Variables que usó el árbol")
    plt.tight_layout()
    plt.savefig(CARPETA / "4_importancia.png", dpi=150)
    plt.close()

    # ---------- Gráfica 5: matriz de confusión ----------
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediccion, normalize="true", values_format=".0%", colorbar=False,
        display_labels=["Riesgo normal", "Riesgo alto"])
    plt.xlabel("Lo que predijo el árbol")
    plt.ylabel("Lo que realmente ocurrió")
    plt.title("Matriz de confusión (datos de prueba)")
    plt.savefig(CARPETA / "5_matriz_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- Gráfica 6: ¿funciona igual en los tres datasets? ----------
    origen_prueba = tabla.loc[X_test.index, "origen"]
    filas = []
    for nombre in ["adultos_salud", "insurance", "Medicalpremium"]:
        s = origen_prueba == nombre
        filas.append({"Dataset": nombre,
                      "Casos de riesgo alto en la prueba": int(y_test[s].sum()),
                      "Riesgo alto detectado": recall_score(y_test[s], prediccion[s]),
                      "Aciertos totales": (y_test[s] == prediccion[s]).mean()})
    por_dataset = pd.DataFrame(filas).set_index("Dataset")
    print("\nDesempeño por dataset (datos de prueba):")
    print(por_dataset.round(3))
    por_dataset[["Riesgo alto detectado", "Aciertos totales"]].plot(
        kind="bar", figsize=(8, 5), color=["#e45756", "#4c78a8"])
    plt.ylim(0, 1.05)
    plt.title("¿El árbol funciona igual en los tres datasets?")
    plt.ylabel("Proporción")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(CARPETA / "6_por_dataset.png", dpi=150)
    plt.close()

    # ---------- Estabilidad: ¿cuánto cambian los resultados con otra muestra? ----------
    est = evaluar_estabilidad(tablas)
    est.to_csv(CARPETA_RESULTADOS / "estabilidad_arbol_unico.csv", index=False)
    print("\nEstabilidad (20 repeticiones con semillas 0 a 19): promedio ± desviación [mín - máx]")
    for col in ["aciertos", "riesgo_alto_detectado", "precision", "f1",
                "detectado_adultos_salud", "detectado_insurance", "detectado_Medicalpremium"]:
        v = est[col]
        print(f"  {col:<26} {v.mean():.3f} ± {v.std():.3f}  [{v.min():.3f} - {v.max():.3f}]")

    # ---------- Gráfica 7: estabilidad ----------
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    est[["riesgo_alto_detectado", "precision", "f1"]].rename(columns={
        "riesgo_alto_detectado": "Riesgo alto\ndetectado", "precision": "Precisión", "f1": "F1"}
    ).plot(kind="box", ax=ejes[0])
    ejes[0].set_title("20 repeticiones: ¿cuánto varía el resultado?")
    ejes[0].set_ylabel("Valor")
    est[["detectado_adultos_salud", "detectado_insurance", "detectado_Medicalpremium"]].rename(columns={
        "detectado_adultos_salud": "adultos_salud", "detectado_insurance": "insurance",
        "detectado_Medicalpremium": "Medicalpremium"}).plot(kind="box", ax=ejes[1])
    ejes[1].set_title("Riesgo alto detectado, por dataset")
    ejes[1].set_ylabel("Proporción detectada")
    plt.tight_layout()
    plt.savefig(CARPETA / "7_estabilidad.png", dpi=150)
    plt.close()

    print("\nGráficas guardadas en:", CARPETA)


if __name__ == "__main__":
    main()
