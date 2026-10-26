"""
transformaciones_lotes.py
Versión por lotes de la matriz final de transformaciones.py
(StandardScaler en numéricas + one-hot en categóricas + binarias + objetivo).

Problema: estas transformaciones dependen de estadísticas GLOBALES.
- StandardScaler necesita la media y la desviación de TODA la columna.
- get_dummies necesita conocer TODAS las categorías: si un lote no
  contiene una categoría, saldrían columnas distintas en cada lote.

Solución: dos pasadas sobre el archivo.
  Pasada 1: recorrer los lotes solo para APRENDER (partial_fit del
            escalador y conjunto de categorías). No se escribe nada.
  Pasada 2: recorrer los lotes de nuevo y TRANSFORMAR cada uno con lo
            aprendido, escribiéndolo de inmediato.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from herramientas.io_datos import guardar_csv_por_lotes, leer_csv_por_lotes


def transformar_por_lotes(ruta_limpio, ruta_salida, numericas, categoricas,
                          binarias, objetivo, tamano_lote=10000) -> int:
    """Devuelve la cantidad de filas escritas."""
    # ---------- Pasada 1: aprender estadísticas globales ----------
    escalador = StandardScaler()
    categorias = {c: set() for c in categoricas}
    for lote in leer_csv_por_lotes(ruta_limpio, tamano_lote):
        escalador.partial_fit(lote[numericas])          # acumula media y varianza
        for col in categoricas:
            categorias[col].update(lote[col].unique())

    # ---------- Pasada 2: transformar y guardar lote a lote ----------
    primer_lote = True
    filas = 0
    for lote in leer_csv_por_lotes(ruta_limpio, tamano_lote):
        lote = lote.reset_index(drop=True)              # el índice del lote no empieza en 0

        X_std = pd.DataFrame(escalador.transform(lote[numericas]),
                             columns=numericas)
        partes = [X_std]

        for col in categoricas:
            # Se fijan TODAS las categorías para que cada lote genere las mismas columnas
            lote[col] = pd.Categorical(lote[col], categories=sorted(categorias[col]))
            partes.append(pd.get_dummies(lote[[col]], prefix=col,
                                         drop_first=True, dtype=int))

        partes.append(lote[binarias])
        partes.append(lote[[objetivo]])
        matriz = pd.concat(partes, axis=1)

        guardar_csv_por_lotes(matriz, ruta_salida, primer_lote)
        primer_lote = False
        filas += len(matriz)
    return filas
