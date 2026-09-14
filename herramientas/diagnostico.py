"""
diagnostico.py
Función de diagnóstico general, usada al inicio y al final del pipeline
de limpieza para ambos datasets. Vive aparte porque no es responsabilidad
de ninguna clase de limpieza en particular (SRP): cualquier DataFrame,
sea cual sea su origen, puede diagnosticarse igual.
"""

import pandas as pd


def diagnostico(data: pd.DataFrame, etiqueta: str = "") -> None:
    """Imprime el estado de salud del dataset: filas, nulos y duplicados."""
    print(f"--- Diagnóstico {etiqueta} ---")
    print("Filas:", len(data))
    print("Nulos por columna:")
    print(data.isna().sum())
    print("Duplicados exactos:", data.duplicated().sum())
    print()
