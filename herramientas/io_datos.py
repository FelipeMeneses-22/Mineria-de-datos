"""
io_datos.py
Lectura y escritura de los datasets. Aísla la dependencia de pandas.io
en un solo lugar: si mañana cambia el formato de entrada/salida (por
ejemplo, a Parquet o a una base de datos), solo se toca este archivo y
ninguna clase de limpieza se entera del cambio (SRP + DIP: el resto del
proyecto depende de estas dos funciones, no directamente de pandas).
"""

import pandas as pd


def leer_csv(ruta: str) -> pd.DataFrame:
    """Lee un CSV crudo desde disco."""
    return pd.read_csv(ruta)


def guardar_csv(df: pd.DataFrame, ruta: str) -> None:
    """Guarda un DataFrame ya limpio en disco."""
    df.to_csv(ruta, index=False)
