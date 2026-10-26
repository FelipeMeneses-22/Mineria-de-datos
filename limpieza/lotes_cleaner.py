"""
lotes_cleaner.py
Limpieza por lotes (Batch Processing) para CSV grandes.

Hace el MISMO pipeline de DataCleaner
    Duplicados -> Categorías -> Nulos -> Outliers -> Validación
pero leyendo el archivo en bloques de filas (chunksize) en vez de
cargarlo completo en memoria.

Qué cambia y qué no:
- Duplicados: es una regla GLOBAL (una fila puede repetirse en lotes
  distintos). Se guarda una "huella" (hash) de cada fila ya escrita y
  se descarta la fila si su huella ya existe. Se conserva la primera
  aparición, igual que drop_duplicates().
- Categorías y nulos: se acumulan lote a lote (unión de valores,
  suma de nulos). El resultado es igual al de cargar todo.
- Outliers (IQR): los cuartiles son estadísticas GLOBALES; calcularlos
  por lote daría límites distintos en cada bloque. Como en estos
  datasets los outliers solo se reportan (se conservan todos), el
  archivo de salida no depende de ellos: el reporte se calcula al
  final leyendo SOLO las columnas numéricas del archivo ya limpio.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from herramientas.estadisticas import resumen_iqr
from herramientas.io_datos import guardar_csv_por_lotes, leer_csv_por_lotes


class BatchCleaner:
    """Pipeline de limpieza por lotes. Una instancia = un dataset."""

    def __init__(self, ruta_raw, ruta_limpio, nombre_dataset,
                 columnas_a_verificar, columnas_iqr, tamano_lote=10000):
        # Seguridad: nunca escribir encima del archivo original.
        if Path(ruta_raw).resolve() == Path(ruta_limpio).resolve():
            raise ValueError("La ruta de salida no puede ser la del archivo original.")
        self.ruta_raw = ruta_raw
        self.ruta_limpio = ruta_limpio
        self.nombre_dataset = nombre_dataset
        self.columnas_a_verificar = columnas_a_verificar   # categóricas / binarias
        self.columnas_iqr = columnas_iqr                    # numéricas a revisar
        self.tamano_lote = tamano_lote

    def ejecutar(self) -> dict:
        print(f"\n{'=' * 60}\nLIMPIEZA POR LOTES: {self.nombre_dataset} "
              f"(lotes de {self.tamano_lote} filas)\n{'=' * 60}")
        Path(self.ruta_limpio).parent.mkdir(parents=True, exist_ok=True)

        # Contadores del registro de procesamiento
        leidos = escritos = duplicados = filas_error = 0
        lotes = lotes_error = 0
        nulos = None                                    # se acumula
        valores = {c: set() for c in self.columnas_a_verificar}
        vistos = np.array([], dtype=np.uint64)          # huellas de filas ya escritas
        archivo_creado = False

        inicio = time.perf_counter()
        for lote in leer_csv_por_lotes(self.ruta_raw, self.tamano_lote):
            lotes += 1
            leidos += len(lote)
            try:
                # Paso 1 · Duplicados (dentro del lote y contra lotes anteriores)
                huellas = pd.util.hash_pandas_object(lote, index=False)
                es_nueva = ~huellas.duplicated() & ~np.isin(huellas.values, vistos)
                limpio = lote[es_nueva]

                # Paso 2 · Categorías: se acumulan los valores distintos
                for col in self.columnas_a_verificar:
                    valores[col].update(limpio[col].unique())

                # Paso 3 · Nulos: se suman por columna
                nulos_lote = limpio.isna().sum()
                nulos = nulos_lote if nulos is None else nulos + nulos_lote

                # Guardado progresivo: el lote se escribe y se descarta
                guardar_csv_por_lotes(limpio, self.ruta_limpio, not archivo_creado)
                archivo_creado = True
                vistos = np.concatenate([vistos, huellas[es_nueva].values])

                escritos += len(limpio)
                duplicados += int((~es_nueva).sum())
            except Exception as error:                  # un lote malo no detiene todo
                lotes_error += 1
                filas_error += len(lote)
                print(f"  ERROR en el lote {lotes}: {error}")

        # Reporte (mismo contenido que el pipeline convencional)
        print(f"Duplicados exactos eliminados: {duplicados}\n")
        print("Verificación de categorías:")
        for col, vals in valores.items():
            print(col, "->", sorted(vals))
        print("\nNulos por columna:")
        print(nulos)
        print("Total de nulos:", nulos.sum() if nulos is not None else 0, "\n")

        # Paso 4 · Outliers (IQR global): solo se leen las columnas necesarias
        if self.columnas_iqr and archivo_creado:
            numericas = pd.read_csv(self.ruta_limpio, usecols=self.columnas_iqr)
            for col in self.columnas_iqr:
                resumen_iqr(numericas[col], col)

        segundos = time.perf_counter() - inicio

        # Paso 5 · Registro / validación final
        registro = {
            "dataset": self.nombre_dataset, "tamano_lote": self.tamano_lote,
            "lotes": lotes, "registros_leidos": leidos,
            "registros_escritos": escritos, "duplicados_eliminados": duplicados,
            "lotes_con_error": lotes_error, "filas_en_lotes_con_error": filas_error,
            "segundos": round(segundos, 2),
        }
        print("\nRegistro del procesamiento:")
        for clave, valor in registro.items():
            print(f"- {clave:<26}: {valor}")
        assert leidos == escritos + duplicados + filas_error, "Las cuentas no cuadran"
        if archivo_creado:
            print(f"\nArchivo exportado: {self.ruta_limpio}")
        else:
            print("\nNo se exportó ningún archivo (todos los lotes fallaron).")
        return registro
