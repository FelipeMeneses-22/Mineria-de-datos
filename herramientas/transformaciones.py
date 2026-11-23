
import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from herramientas.io_datos import guardar_csv, leer_csv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RAIZ = Path(__file__).resolve().parent.parent   # raíz del proyecto (este archivo está en herramientas/)


def distancia(a: pd.Series, b: pd.Series) -> float:
    """Distancia euclidiana entre dos filas."""
    return float(((a - b) ** 2).sum() ** 0.5)


class DataTransformer:
    """Pipeline genérico de transformación y reducción de datos limpios."""

    # Cada subclase declara sus columnas.
    NOMBRE: str = ""
    OBJETIVO: str = ""                 # variable a predecir (no se escala)
    NUMERICAS: list[str] = []          # se escalan y entran al PCA
    CATEGORICAS: list[str] = []        # texto -> one-hot
    BINARIAS: list[str] = []           # ya son 0/1: pasan tal cual
    COLUMNA_DISCRETIZAR: str = ""      # numérica que se agrupa en rangos

    def __init__(self, ruta_limpio: Path, ruta_salida: Path):
        if not self.NUMERICAS or not self.OBJETIVO:
            raise NotImplementedError(
                "La subclase debe definir OBJETIVO y NUMERICAS.")
        self.ruta_limpio = ruta_limpio
        self.ruta_salida = ruta_salida
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.X_std: pd.DataFrame | None = None
        self.X_cat: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Método plantilla: orden fijo de la transformación.
    # ------------------------------------------------------------------
    def ejecutar(self) -> pd.DataFrame:
        print(f"\n{'=' * 60}\nTRANSFORMACIÓN: {self.NOMBRE}\n{'=' * 60}")
        self.df = leer_csv(str(self.ruta_limpio))
        self.X = self.df[self.NUMERICAS]
        print("Dataset limpio:", self.df.shape,
              "| matriz numérica:", self.X.shape,
              "| objetivo:", self.OBJETIVO)

        self._por_que_escalar()          # Paso 1
        self._escaladores()              # Paso 2
        self._codificar_categoricas()    # Paso 3
        self._discretizar()              # Paso 4
        self._pca()                      # Paso 5
        matriz = self._matriz_final()    # Paso 6

        self.ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        guardar_csv(matriz, str(self.ruta_salida))
        print(f"\nArchivo exportado: {self.ruta_salida.relative_to(RAIZ)}")
        return matriz

    # ------------------------------------------------------------------
    # 1 · Por qué escalar: la distancia la domina la variable con
    #     unidades grandes.
    # ------------------------------------------------------------------
    def _por_que_escalar(self) -> None:
        c1, c2 = self.X.iloc[0], self.X.iloc[1]
        print("\n--- Distancia entre dos clientes SIN escalar ---")
        for col in self.NUMERICAS:
            print(f"  {col:<24} aporta {abs(c1[col] - c2[col]):>10.1f}")
        print("Distancia total sin escalar :", round(distancia(c1, c2), 1))

        X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(self.X),
                                columns=self.NUMERICAS)
        print("Distancia total con MinMax  :",
              round(distancia(X_minmax.iloc[0], X_minmax.iloc[1]), 3))

    # ------------------------------------------------------------------
    # 2 · Los tres escaladores.
    #     Nota: aquí se ajustan sobre todo el dataset porque es una
    #     demostración. En Modelado, dividir train/test PRIMERO y hacer
    #     fit solo con train (evita fuga de información).
    # ------------------------------------------------------------------
    def _escaladores(self) -> None:
        X = self.X
        print("\n--- MinMaxScaler (todo a [0, 1]) ---")
        X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X),
                                columns=self.NUMERICAS)
        print("Antes  : min", X.min().round(1).tolist())
        print("Después: min", X_minmax.min().round(3).tolist(),
              "max", X_minmax.max().round(3).tolist())

        print("\n--- StandardScaler (media=0, desviación=1) ---")
        self.X_std = pd.DataFrame(StandardScaler().fit_transform(X),
                                  columns=self.NUMERICAS)
        print("Medias tras escalar:", self.X_std.mean().round(3).tolist())
        print("Desv. estándar     :", self.X_std.std().round(3).tolist())

        print("\n--- RobustScaler (mediana e IQR, resiste outliers) ---")
        X_rob = pd.DataFrame(RobustScaler().fit_transform(X),
                             columns=self.NUMERICAS)
        print("Mediana tras escalar:", X_rob.median().round(3).tolist())
        # Se usa StandardScaler para la matriz final. RobustScaler queda
        # como alternativa si en Modelado los outliers (bmi, Weight) pesan
        # demasiado; recordar que se conservaron por justificación de negocio.

    # ------------------------------------------------------------------
    # 3 · Codificación de categóricas (one-hot).
    #     drop_first=True evita columnas redundantes (trampa de la variable
    #     ficticia), importante para la regresión lineal de la fase 4.
    # ------------------------------------------------------------------
    def _codificar_categoricas(self) -> None:
        if not self.CATEGORICAS:
            print("\n--- One-hot ---")
            print("Sin columnas de texto: las categóricas de este dataset "
                  "ya son binarias 0/1:", self.BINARIAS)
            self.X_cat = pd.DataFrame(index=self.df.index)
            return
        print("\n--- One-hot con get_dummies ---")
        self.X_cat = pd.get_dummies(self.df[self.CATEGORICAS],
                                    prefix=self.CATEGORICAS,
                                    drop_first=True, dtype=int)
        print("Columnas nuevas:", self.X_cat.columns.tolist())
        print(self.X_cat.head(3).to_string())

    # ------------------------------------------------------------------
    # 4 · Discretización en 4 rangos por cuantiles.
    #     Sirve para explorar/interpretar por grupos; no se agrega a la
    #     matriz final porque duplicaría la información de la columna
    #     numérica ya escalada.
    # ------------------------------------------------------------------
    def _discretizar(self) -> None:
        col = self.COLUMNA_DISCRETIZAR
        print(f"\n--- KBinsDiscretizer: {col} -> 4 categorías ---")
        disc = KBinsDiscretizer(n_bins=4, encode="ordinal",
                                strategy="quantile")
        bins = disc.fit_transform(self.df[[col]]).ravel()
        print("Bordes de los rangos:", disc.bin_edges_[0].round(0))
        print(pd.Series(bins).value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # 5 · PCA sobre las numéricas estandarizadas.
    # ------------------------------------------------------------------
    def _pca(self) -> None:
        print("\n--- PCA: ¿cuántas dimensiones resumen los datos? ---")
        pca = PCA()
        pca.fit(self.X_std)
        razones = pca.explained_variance_ratio_
        acumulada = razones.cumsum().round(3)
        print("Varianza explicada por componente :", razones.round(3))
        print("Varianza acumulada                :", acumulada)
        para_80 = int((acumulada < 0.80).sum()) + 1
        para_90 = int((acumulada < 0.90).sum()) + 1
        print(f"Componentes para explicar >=80%: {para_80} | "
              f"para >=90%: {para_90} (de {len(self.NUMERICAS)})")

    # ------------------------------------------------------------------
    # 6 · Matriz de features lista para modelar (+ objetivo sin escalar).
    # ------------------------------------------------------------------
    def _matriz_final(self) -> pd.DataFrame:
        partes = [self.X_std, self.X_cat, self.df[self.BINARIAS],
                  self.df[[self.OBJETIVO]]]
        matriz = pd.concat(partes, axis=1)
        print("\n--- Matriz final para modelado ---")
        print("Forma:", matriz.shape)
        print("Features:", matriz.drop(columns=self.OBJETIVO).columns.tolist())
        print("Objetivo (sin escalar):", self.OBJETIVO)
        return matriz


class InsuranceTransformer(DataTransformer):
    """Transformación de insurance_limpio.csv."""

    NOMBRE = "insurance_limpio.csv"
    OBJETIVO = "charges"
    NUMERICAS = ["age", "bmi", "children"]
    CATEGORICAS = ["sex", "smoker", "region"]
    COLUMNA_DISCRETIZAR = "age"


class MedicalPremiumTransformer(DataTransformer):
    """Transformación de Medicalpremium_limpio.csv."""

    NOMBRE = "Medicalpremium_limpio.csv"
    OBJETIVO = "PremiumPrice"
    NUMERICAS = ["Age", "Height", "Weight", "NumberOfMajorSurgeries"]
    BINARIAS = ["Diabetes", "BloodPressureProblems", "AnyTransplants",
                "AnyChronicDiseases", "KnownAllergies",
                "HistoryOfCancerInFamily"]
    COLUMNA_DISCRETIZAR = "Age"


def main() -> None:
    limpio = RAIZ / "data" / "limpio"
    salida = RAIZ / "data" / "transformado"
    pipelines = [
        InsuranceTransformer(limpio / "insurance_limpio.csv",
                             salida / "insurance_transformado.csv"),
        MedicalPremiumTransformer(limpio / "Medicalpremium_limpio.csv",
                                  salida / "Medicalpremium_transformado.csv"),
    ]
    for pipeline in pipelines:
        pipeline.ejecutar()


if __name__ == "__main__":
    main()
