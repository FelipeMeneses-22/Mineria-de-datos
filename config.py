"""
config.py
Configuración ÚNICA del proyecto: rutas y columnas de cada dataset.
Si cambia una columna, se cambia aquí y todos los scripts la ven
(main_lotes.py, benchmark_lotes.py, arbol_general.py, modelado_justo.py,
modelado_regresion.py).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RAW = RAIZ / "data" / "raw"
LIMPIO = RAIZ / "data" / "limpio"

SEMILLA = 42                      # misma semilla en todo el proyecto (reproducible)

# ---- Árbol único (arbol_general.py): variables compartidas entre datasets ----
VARIABLES_ARBOL = ["edad", "imc", "hombre", "fuma", "diabetes"]
FILAS_POR_DATASET = 986           # tamaño del dataset más pequeño (Medicalpremium)

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
