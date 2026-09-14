"""
insurance_cleaner.py
Limpieza específica de insurance.csv (Predicción de Costos Médicos).

Solo contiene lo que es propio de ESTE dataset: qué columnas
categóricas revisar y cómo tratar sus outliers. El resto del pipeline
(duplicados, nulos, diagnóstico, validación) lo hereda de DataCleaner.
"""

from limpieza.base_cleaner import DataCleaner


class InsuranceCleaner(DataCleaner):
    """Pipeline de limpieza para insurance.csv."""

    COLUMNAS_CATEGORICAS = ["sex", "smoker", "region"]

    def __init__(self, ruta_raw: str, ruta_limpio: str):
        super().__init__(ruta_raw, ruta_limpio, nombre_dataset="insurance.csv")

    def _mostrar_duplicados(self, mascara) -> None:
        # A diferencia de Medicalpremium, aquí sí interesa ver la(s)
        # fila(s) duplicada(s) completas para justificar el patrón MCAR.
        print("Fila(s) duplicada(s) encontrada(s):")
        print(self.df[mascara])

    def verificar_categorias(self) -> None:
        print("Verificación de categorías:")
        for col in self.COLUMNAS_CATEGORICAS:
            print(col, "->", sorted(self.df[col].unique()))
        print()
        # Hallazgo: sex, smoker y region ya están normalizadas (sin
        # espacios, tildes ni mayúsculas mezcladas). No se requiere
        # strip()/lower()/replace().

    def tratar_outliers(self) -> None:
        self._iqr("bmi", "bmi")
        lim_charges = self._iqr("charges", "charges")
        self._iqr("age", "age")
        print()

        # age: 0 outliers -> rango 18-64 es plausible para una póliza
        # de salud.

        # bmi: 9 outliers (valores entre ~47 y ~53) -> obesidad severa
        # real, no error de digitación. Se CONSERVAN.

        # charges: 139 outliers. Antes de decidir, se cruza con smoker
        # para entender el origen del valor extremo.
        _, _, outliers_charges = lim_charges
        fumadores_en_outliers = self.df.loc[outliers_charges.index, "smoker"]
        print("Outliers de charges por condición de fumador:")
        print(fumadores_en_outliers.value_counts())
        pct_fumadores = (fumadores_en_outliers == "yes").mean()
        print(f"{len(outliers_charges)} outliers en total, "
              f"{pct_fumadores:.1%} son fumadores\n")
        # Decisión: CONSERVAR. El ~98% de los outliers de charges son
        # personas fumadoras: no es ruido, es la señal central del
        # proyecto (fumar dispara los costos médicos). Winsorizar o
        # eliminar estas filas borraría esa relación y sesgaría el
        # modelo justo en el segmento de mayor riesgo. Los 9 outliers
        # de bmi se conservan por la misma razón.

    def resumen_especifico(self) -> None:
        print("- Categorías verificadas: sex, smoker, region sin inconsistencias")
        print("- Outliers tratados     : 0 (bmi y charges se conservan "
              "por justificación de negocio)")
