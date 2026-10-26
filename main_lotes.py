"""
main_lotes.py
Corre limpieza y transformación POR LOTES para los tres datasets.
No reemplaza a main.py: escribe en carpetas aparte (data/limpio_lotes y
data/transformado_lotes) para no pisar los resultados convencionales.

Uso:  python main_lotes.py            (lotes de 10000 filas)
      python main_lotes.py 5000       (lotes de 5000 filas)
"""

import sys
from pathlib import Path

from herramientas.transformaciones_lotes import transformar_por_lotes
from limpieza.lotes_cleaner import BatchCleaner

RAIZ = Path(__file__).resolve().parent

# Cada dataset tiene SUS columnas y reglas: no se asume que sean iguales.
DATASETS = {
    "insurance": {
        "raw": "insurance.csv",
        "verificar": ["sex", "smoker", "region"],
        "iqr": ["bmi", "charges", "age"],
        "transf": {"numericas": ["age", "bmi", "children"],
                   "categoricas": ["sex", "smoker", "region"],
                   "binarias": [], "objetivo": "charges"},
    },
    "Medicalpremium": {
        "raw": "Medicalpremium.csv",
        "verificar": ["Diabetes", "BloodPressureProblems", "AnyTransplants",
                      "AnyChronicDiseases", "KnownAllergies",
                      "HistoryOfCancerInFamily"],
        "iqr": ["Age", "Height", "Weight", "PremiumPrice",
                "NumberOfMajorSurgeries"],
        "transf": {"numericas": ["Age", "Height", "Weight",
                                 "NumberOfMajorSurgeries"],
                   "categoricas": [],
                   "binarias": ["Diabetes", "BloodPressureProblems",
                                "AnyTransplants", "AnyChronicDiseases",
                                "KnownAllergies", "HistoryOfCancerInFamily"],
                   "objetivo": "PremiumPrice"},
    },
    "adultos_salud": {
        "raw": "adultos_salud.csv",
        "verificar": ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                      "Sex", "AgeCategory", "Race", "Diabetic",
                      "PhysicalActivity", "GenHealth", "Asthma",
                      "KidneyDisease", "SkinCancer"],
        "iqr": ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"],
        "transf": {"numericas": ["BMI", "PhysicalHealth", "MentalHealth",
                                 "SleepTime"],
                   "categoricas": ["Smoking", "AlcoholDrinking", "Stroke",
                                   "DiffWalking", "Sex", "AgeCategory",
                                   "Race", "Diabetic", "PhysicalActivity",
                                   "GenHealth", "Asthma", "KidneyDisease",
                                   "SkinCancer"],
                   "binarias": [], "objetivo": "HeartDisease"},
    },
}


def main(tamano_lote: int = 10000) -> None:
    for nombre, cfg in DATASETS.items():
        ruta_raw = RAIZ / "data" / "raw" / cfg["raw"]
        if not ruta_raw.exists():
            print(f"\n[omitido] No existe {ruta_raw}")
            continue
        ruta_limpio = RAIZ / "data" / "limpio_lotes" / f"{nombre}_limpio.csv"
        ruta_transf = RAIZ / "data" / "transformado_lotes" / f"{nombre}_transformado.csv"

        BatchCleaner(ruta_raw, ruta_limpio, cfg["raw"], cfg["verificar"],
                     cfg["iqr"], tamano_lote).ejecutar()

        ruta_transf.parent.mkdir(parents=True, exist_ok=True)
        filas = transformar_por_lotes(ruta_limpio, ruta_transf,
                                      tamano_lote=tamano_lote, **cfg["transf"])
        print(f"\nTransformación por lotes: {filas} filas -> {ruta_transf}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10000)
