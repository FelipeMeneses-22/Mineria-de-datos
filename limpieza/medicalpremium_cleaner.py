"""
medicalpremium_cleaner.py
Limpieza específica de Medicalpremium.csv (perfil clínico: condiciones
preexistentes, cirugías, altura/peso).

Igual que InsuranceCleaner, solo contiene lo propio de este dataset;
el pipeline común vive en DataCleaner.
"""

from limpieza.base_cleaner import DataCleaner


class MedicalPremiumCleaner(DataCleaner):
    """Pipeline de limpieza para Medicalpremium.csv."""

    COLUMNAS_BINARIAS = [
        "Diabetes", "BloodPressureProblems", "AnyTransplants",
        "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily",
    ]

    def __init__(self, ruta_raw: str, ruta_limpio: str):
        super().__init__(ruta_raw, ruta_limpio, nombre_dataset="Medicalpremium.csv")

    def verificar_categorias(self) -> None:
        # Aquí las "categóricas" son en realidad binarias (0/1) en vez
        # de texto (yes/no como en insurance.csv), así que el chequeo
        # es de rango válido, no de grafías.
        print("Verificación de columnas binarias (deben ser solo 0/1):")
        for col in self.COLUMNAS_BINARIAS:
            print(col, "->", sorted(self.df[col].unique()))
        print()
        # Hallazgo: las 6 columnas binarias solo contienen {0, 1}, sin
        # códigos espurios (p. ej. -1, 2, "Y"/"N" mezclado). No se
        # requiere corrección.

    def tratar_outliers(self) -> None:
        self._iqr("Age", "Age")
        self._iqr("Height", "Height")
        lim_weight = self._iqr("Weight", "Weight")
        self._iqr("PremiumPrice", "PremiumPrice")
        self._iqr("NumberOfMajorSurgeries", "NumberOfMajorSurgeries")
        print()

        # Age y Height: 0 outliers -> rangos plausibles para adultos
        # asegurados.

        # Weight: 16 outliers (118-132 kg). Cruzados con Height,
        # corresponden a IMC ~43-52 (obesidad severa real), no errores
        # de digitación. Se CONSERVAN por la misma razón que el bmi de
        # insurance.csv.
        _, _, outliers_weight = lim_weight
        print("Detalle de outliers en Weight (peso real, no error):")
        print(self.df.loc[outliers_weight.index,
                           ["Age", "Height", "Weight", "AnyChronicDiseases"]]
              .sort_values("Weight"))
        print()

        # PremiumPrice: 6 outliers, todos en el techo de la tarifa
        # (39000-40000). Es la variable objetivo del dataset: son primas
        # reales del segmento de mayor riesgo (más condiciones/
        # cirugías), no ruido de captura. Se CONSERVAN.

        # NumberOfMajorSurgeries: la regla IQR marca 16 "outliers", pero
        # es una variable discreta de solo 4 valores posibles (0,1,2,3)
        # con distribución angosta -> el criterio de Tukey sobre-marca
        # outliers en conteos con rango tan corto. El valor 3 es una
        # cirugía mayor perfectamente válida, no un error. Se CONSERVA y
        # se documenta que IQR no es el criterio adecuado para variables
        # discretas de bajo rango.
        print("Distribución completa de NumberOfMajorSurgeries "
              "(variable discreta, IQR no es el criterio ideal aquí):")
        print(self.df["NumberOfMajorSurgeries"].value_counts().sort_index())
        print()

    def resumen_especifico(self) -> None:
        print("- Columnas binarias verificadas: solo valores {0,1}")
        print("- Outliers tratados     : 0 (Weight, PremiumPrice y "
              "NumberOfMajorSurgeries se conservan por justificación de negocio)")
