"""clustering.py
Segmentación (aprendizaje NO supervisado) de adultos_salud e insurance, como en
la sesión 9: escalar -> elegir k (codo + silueta) -> K-means final ->
perfilar con groupby -> contrastar con un dendrograma jerárquico.

Regla de oro (sesión 9, error 4): la variable objetivo (HeartDisease, charges)
NO entra a X. Se usa DESPUÉS, solo para perfilar los segmentos encontrados.

Sobre el escalado: aquí el escalador se ajusta con todos los datos porque
clustering no tiene partición de prueba que proteger (no se evalúa contra
etiquetas). En los modelos supervisados SÍ va dentro de un Pipeline.

Ejecutar desde la raíz del proyecto:
    python clustering.py                  (los dos datasets)
    python clustering.py insurance        (solo uno)
Salida: data/resultados/clustering_<dataset>_*.csv y graficas/clustering/<dataset>/
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent
LIMPIO = RAIZ / "data" / "limpio"
SALIDA = RAIZ / "data" / "resultados"

SEMILLA = 42
RANGO_K = range(2, 9)
MUESTRA_SILUETA = 10000     # la silueta es O(n^2): se calcula sobre una muestra
MUESTRA_DENDROGRAMA = 60    # el jerárquico es O(n^2): solo una muestra pequeña


# ----------------------------------------------------------------------
# Un cargador por dataset: devuelve una tabla con las columnas de VARIABLES
# y una columna "resultado" (lo que se usa SOLO para perfilar).
# ----------------------------------------------------------------------
def cargar_adultos() -> pd.DataFrame:
    df = pd.read_csv(LIMPIO / "adultos_salud_limpio.csv")
    punto_medio = {r: (int(r[:2]) + int(r[3:5])) / 2
                   for r in df["AgeCategory"].unique() if r[0] != "8"}
    punto_medio["80 or older"] = 82
    df["edad"] = df["AgeCategory"].map(punto_medio)
    df["resultado"] = (df["HeartDisease"] == "Yes").astype(int) * 100   # % enfermos
    return df


def extras_adultos(df: pd.DataFrame) -> pd.DataFrame:
    g = df["cluster"]
    return pd.DataFrame({
        "% fuma": (df["Smoking"] == "Yes").groupby(g).mean() * 100,
        "% dificultad caminar": (df["DiffWalking"] == "Yes").groupby(g).mean() * 100,
        "% diabetes": df["Diabetic"].str.startswith("Yes").groupby(g).mean() * 100,
    }).round(1)


def cargar_insurance() -> pd.DataFrame:
    df = pd.read_csv(LIMPIO / "insurance_limpio.csv")
    df["fuma"] = (df["smoker"] == "yes").astype(int)
    df["resultado"] = df["charges"]                                      # costo medio
    return df


def extras_insurance(df: pd.DataFrame) -> pd.DataFrame:
    g = df["cluster"]
    return pd.DataFrame({
        "% hombres": (df["sex"] == "male").groupby(g).mean() * 100,
        "costo mediano": df.groupby("cluster")["charges"].median(),
        "% región southeast": (df["region"] == "southeast").groupby(g).mean() * 100,
    }).round(1)


CONFIGS = {
    "adultos_salud": dict(
        cargar=cargar_adultos, extras=extras_adultos,
        variables=["edad", "BMI", "PhysicalHealth", "MentalHealth", "SleepTime"],
        k=4, ordenar_por="edad",
        nombres={0: "Jóvenes sanos", 1: "Adultos con malestar emocional",
                 2: "Mayores con mala salud física", 3: "Mayores estables"},
        etiqueta="% con enfermedad cardíaca", formato="{:.1f} %",
        titulo_barras="La enfermedad no se usó para agrupar, pero separa a los grupos",
        # Por qué k=4 (decisión documentada, no automática):
        #  - k=2 tiene la silueta más alta (0,41) pero solo separa "buena salud" de
        #    "muchos días de mala salud" (variables 0/30): demasiado grueso.
        #  - De k=3 a k=8 la silueta es casi igual (0,23-0,26) y el codo no es nítido.
        #  - k=4 da grupos distintos por edad y salud, con riesgo muy diferente.
    ),
    "insurance": dict(
        cargar=cargar_insurance, extras=extras_insurance,
        variables=["age", "bmi", "children", "fuma"],
        k=4, ordenar_por="age",
        nombres={0: "Jóvenes no fumadores", 1: "Fumadores",
                 2: "Familias con hijos", 3: "No fumadores maduros"},
        etiqueta="Costo médico medio (charges)", formato="{:,.0f}",
        titulo_barras="El costo no se usó para agrupar, pero separa a los grupos",
        # Por qué k=4 (decisión documentada, no automática):
        #  - k=2 tiene la silueta más alta (0,345) pero solo separa fumadores de
        #    no fumadores, algo que el EDA ya mostraba.
        #  - k=4 (silueta 0,302, la segunda mejor) mantiene a los fumadores como un
        #    grupo y parte a los no fumadores por edad e hijos: ahí está lo nuevo.
        #  - Los fumadores quedan en UN grupo con cualquier k: al escalar, la
        #    variable 0/1 pesa mucho. Es un hallazgo y una limitación.
    ),
}


def segmentar(clave: str) -> None:
    cfg = CONFIGS[clave]
    V, K, nombres = cfg["variables"], cfg["k"], cfg["nombres"]
    carpeta = RAIZ / "graficas" / "clustering" / clave
    carpeta.mkdir(parents=True, exist_ok=True)
    SALIDA.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nCLUSTERING DE {clave.upper()}\n{'=' * 70}")
    df = cfg["cargar"]()
    print(f"Personas: {len(df)} | Variables para agrupar: {V}")
    X = StandardScaler().fit_transform(df[V])           # sin escalar, una variable dominaría

    # ---------- 1. Elegir k: codo (inercia) + silueta ----------
    filas = []
    for k in RANGO_K:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=MUESTRA_SILUETA, random_state=SEMILLA)
        filas.append({"k": k, "inercia": round(km.inertia_), "silueta": round(sil, 3)})
    tabla_k = pd.DataFrame(filas)
    print("\n=== Codo y silueta ===")
    print(tabla_k.to_string(index=False))
    mejor = tabla_k.loc[tabla_k["silueta"].idxmax(), "k"]
    print(f"\nSilueta máxima en k={mejor}. k elegido para el informe: {K}")
    tabla_k.to_csv(SALIDA / f"clustering_{clave}_codo_silueta.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].plot(tabla_k["k"], tabla_k["inercia"], marker="o", linewidth=2.5)
    ejes[0].set(xlabel="k (cantidad de grupos)", ylabel="Inercia (menos = grupos más compactos)",
                title="Método del codo")
    ejes[1].plot(tabla_k["k"], tabla_k["silueta"], marker="o", linewidth=2.5, color="#e45756")
    ejes[1].axvline(K, color="gray", linestyle="--", label=f"k elegido = {K}")
    ejes[1].set(xlabel="k (cantidad de grupos)", ylabel="Silueta promedio (más alto = mejor)",
                title="Coeficiente de silueta")
    ejes[1].legend()
    for e in ejes:
        e.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(carpeta / "1_codo_y_silueta.png", dpi=150)
    plt.close()

    # ---------- 2. K-means final (grupos renumerados de menor a mayor edad) ----------
    km = KMeans(n_clusters=K, n_init=10, random_state=SEMILLA).fit(X)
    df["cluster"] = km.labels_
    orden = df.groupby("cluster")[cfg["ordenar_por"]].mean().sort_values().index
    df["cluster"] = df["cluster"].map({viejo: nuevo for nuevo, viejo in enumerate(orden)})

    # ---------- 3. Perfilamiento con groupby (valores reales) ----------
    perfil = df.groupby("cluster")[V].mean().round(2 if clave == "insurance" else 1)
    perfil.insert(0, "segmento", [nombres.get(i, f"Grupo {i}") for i in perfil.index])
    perfil.insert(1, "personas", df["cluster"].value_counts().sort_index())
    perfil.insert(2, "% del total", (perfil["personas"] / len(df) * 100).round(1))
    perfil[cfg["etiqueta"]] = df.groupby("cluster")["resultado"].mean().round(1 if clave == "adultos_salud" else 0)
    print("\n=== Perfil de cada grupo (medias reales) ===")
    print(perfil.to_string())
    print(f"\nPromedio general ({cfg['etiqueta']}): {df['resultado'].mean():,.1f}")
    perfil.to_csv(SALIDA / f"clustering_{clave}_perfiles.csv")

    extra = cfg["extras"](df)
    print("\nVariables que NO entraron al clustering (sirven para verificar los perfiles):")
    print(extra.to_string())
    extra.to_csv(SALIDA / f"clustering_{clave}_perfiles_extra.csv")

    # ---------- Gráfica 2: mapa de calor de los perfiles (z-scores) ----------
    z = pd.DataFrame(km.cluster_centers_, columns=V)
    z.index = [orden.get_loc(i) for i in range(K)]
    z = z.sort_index()
    plt.figure(figsize=(10, 4.5))
    plt.imshow(z.values, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    plt.colorbar(label="Desviaciones estándar sobre el promedio")
    plt.xticks(range(len(V)), V, rotation=20)
    plt.yticks(range(K), [f"{i} · {nombres.get(i, '')}" for i in z.index])
    for i in range(K):
        for j in range(len(V)):
            plt.text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=10)
    plt.title("Qué distingue a cada grupo (rojo = más alto que el promedio)")
    plt.tight_layout()
    plt.savefig(carpeta / "2_perfiles_mapa_calor.png", dpi=150)
    plt.close()

    # ---------- Gráfica 3: resultado por grupo ----------
    plt.figure(figsize=(9, 5))
    etiquetas = [nombres.get(i, f"Grupo {i}").replace(" con ", "\ncon ").replace(" no ", "\nno ")
                 .replace("Mayores ", "Mayores\n") for i in perfil.index]
    valores = perfil[cfg["etiqueta"]]
    barras = plt.bar(etiquetas, valores, color="#e45756")
    plt.axhline(df["resultado"].mean(), color="gray", linestyle="--", label="Promedio general")
    for b, v in zip(barras, valores):
        plt.text(b.get_x() + b.get_width() / 2, v * 1.01, cfg["formato"].format(v), ha="center")
    plt.ylabel(cfg["etiqueta"])
    plt.title(cfg["titulo_barras"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta / "3_resultado_por_grupo.png", dpi=150)
    plt.close()

    # ---------- Gráfica 4: los grupos en 2 dimensiones (PCA) ----------
    muestra = df.sample(8000, random_state=SEMILLA)
    pca = PCA(n_components=2, random_state=SEMILLA)
    pts = pca.fit_transform(StandardScaler().fit_transform(muestra[V]))
    plt.figure(figsize=(8, 6))
    for g in range(K):
        m = (muestra["cluster"] == g).to_numpy()
        plt.scatter(pts[m, 0], pts[m, 1], s=6, alpha=0.5, label=nombres.get(g, f"Grupo {g}"))
    plt.xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0]:.0%} de la variación)")
    plt.ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1]:.0%} de la variación)")
    plt.title("Los grupos en 2 dimensiones (muestra de 8.000 personas)")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(carpeta / "4_grupos_pca.png", dpi=150)
    plt.close()

    # ---------- 4. Jerárquico sobre una muestra (contraste) ----------
    mini = df[V].sample(n=MUESTRA_DENDROGRAMA, random_state=SEMILLA)
    Z = linkage(StandardScaler().fit_transform(mini), method="ward")
    print(f"\n=== Dendrograma (muestra de {MUESTRA_DENDROGRAMA}) ===")
    print("Primeras fusiones (distancia):", Z[:5, 2].round(1))
    print("Últimas 4 fusiones (distancia):", Z[-4:, 2].round(1))
    for k in (2, 3, 4, 5):
        tam = np.bincount(fcluster(Z, t=k, criterion="maxclust"))[1:]
        print(f"  corte en {k} grupos -> tamaños {sorted(tam.tolist(), reverse=True)}")
    plt.figure(figsize=(13, 5))
    corte = (Z[-K, 2] + Z[-(K - 1), 2]) / 2
    dendrogram(Z, color_threshold=corte, no_labels=True)
    plt.axhline(corte, color="gray", linestyle="--", label=f"Corte en {K} ramas")
    plt.ylabel("Distancia (Ward)")
    plt.title(f"Dendrograma de {MUESTRA_DENDROGRAMA} registros de {clave}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(carpeta / "5_dendrograma.png", dpi=150)
    plt.close()
    print("\nGuardado en:", SALIDA.relative_to(RAIZ), "y", carpeta.relative_to(RAIZ))


def main() -> None:
    claves = sys.argv[1:] or list(CONFIGS)
    for c in claves:
        segmentar(c)


if __name__ == "__main__":
    main()
