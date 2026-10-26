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


def leer_csv_por_lotes(ruta: str, tamano_lote: int = 10000):
    """Lee un CSV grande por bloques (Batch Processing).

    No devuelve un DataFrame: devuelve un iterador que entrega un
    DataFrame de `tamano_lote` filas cada vez, así nunca se carga el
    archivo completo en memoria.
    """
    return pd.read_csv(ruta, chunksize=tamano_lote)


def guardar_csv_por_lotes(df: pd.DataFrame, ruta: str, primer_lote: bool) -> None:
    """Escribe un lote al final del archivo de salida.

    El primer lote crea el archivo (con encabezado); los siguientes se
    agregan debajo (modo 'a', sin encabezado).
    """
    df.to_csv(ruta, mode="w" if primer_lote else "a",
              header=primer_lote, index=False)
