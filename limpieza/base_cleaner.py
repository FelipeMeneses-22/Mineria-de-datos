"""
base_cleaner.py
Clase base abstracta para los pipelines de limpieza de datos.

Aplica el patrón "Template Method": el método `ejecutar()` fija el
orden del pipeline visto en clase

    Diagnóstico -> Duplicados -> Categorías -> Nulos -> Outliers -> Validación

y cada dataset concreto (una subclase) solo implementa los pasos que le
son propios, sin poder alterar ese orden.

Cómo se aplican los principios SOLID en este archivo:

- SRP (responsabilidad única): esta clase SOLO orquesta el pipeline.
  Leer/escribir CSV vive en utils/io_datos.py, el diagnóstico en
  utils/diagnostico.py y el cálculo IQR en utils/estadisticas.py.
- OCP (abierto/cerrado): para limpiar un dataset nuevo se crea una
  subclase nueva (ver insurance_cleaner.py / medicalpremium_cleaner.py);
  no hace falta modificar esta clase ni las demás ya existentes.
- LSP (sustitución de Liskov): cualquier subclase de DataCleaner puede
  usarse donde se espere un DataCleaner (main.py no conoce las clases
  concretas, solo llama a `.ejecutar()`).
- ISP (segregación de interfaces): los métodos abstractos están
  separados por paso (categorías, outliers, resumen) en vez de exigir
  un único método gigante; cada subclase implementa solo lo suyo.
- DIP (inversión de dependencias): esta clase depende de las funciones
  de utils/ (abstracciones simples), no de detalles concretos de un
  dataset en particular.
"""

from abc import ABC, abstractmethod

import pandas as pd

from herramientas.diagnostico import diagnostico
from herramientas.estadisticas import resumen_iqr
from herramientas.io_datos import guardar_csv, leer_csv


class DataCleaner(ABC):
    """Pipeline genérico de limpieza de datos tabulares."""

    def __init__(self, ruta_raw: str, ruta_limpio: str, nombre_dataset: str):
        self.ruta_raw = ruta_raw
        self.ruta_limpio = ruta_limpio
        self.nombre_dataset = nombre_dataset
        self.df: pd.DataFrame | None = None
        self.filas_originales: int = 0

    # ------------------------------------------------------------------
    # Método plantilla: define el orden fijo del pipeline.
    # ------------------------------------------------------------------
    def ejecutar(self) -> pd.DataFrame:
        """Corre el pipeline completo en el orden fijo visto en clase."""
        print(f"\n{'=' * 60}\nLIMPIEZA: {self.nombre_dataset}\n{'=' * 60}")
        self.df = leer_csv(self.ruta_raw)
        self.filas_originales = len(self.df)

        diagnostico(self.df, "ANTES")           # Paso 0
        self._eliminar_duplicados()              # Paso 1
        self.verificar_categorias()               # Paso 2 (propio de cada dataset)
        self._revisar_nulos()                     # Paso 3
        self.tratar_outliers()                     # Paso 4 (propio de cada dataset)
        self._validacion_final()                   # Paso 5

        guardar_csv(self.df, self.ruta_limpio)
        print(f"\nArchivo exportado: {self.ruta_limpio}")
        return self.df

    # ------------------------------------------------------------------
    # Pasos comunes a ambos datasets (implementados una sola vez aquí).
    # ------------------------------------------------------------------
    def _eliminar_duplicados(self) -> None:
        """Paso 1 · Duplicados.

        Se eliminan antes que cualquier otro paso para que no contaminen
        medianas/modas calculadas más adelante. La detección es igual
        para ambos datasets; el detalle que se imprime varía, así que se
        delega en el hook `_mostrar_duplicados`.
        """
        print("Antes :", len(self.df), "filas")
        mascara_duplicados = self.df.duplicated(keep=False)
        self._mostrar_duplicados(mascara_duplicados)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print("Después:", len(self.df), "filas\n")

    def _mostrar_duplicados(self, mascara: pd.Series) -> None:
        """Hook con comportamiento por defecto (solo cuenta duplicados).
        InsuranceCleaner lo sobreescribe para mostrar las filas completas.
        """
        print("Duplicados exactos encontrados:", self.df.duplicated().sum())

    def _revisar_nulos(self) -> None:
        """Paso 3 · Nulos (MCAR/MAR/MNAR). Igual para ambos datasets."""
        nulos = self.df.isna().sum()
        print("Nulos por columna:")
        print(nulos)
        print("Total de nulos:", nulos.sum(), "\n")

    def _validacion_final(self) -> None:
        """Paso 5 · Validación final. El resumen específico lo arma cada
        subclase en `resumen_especifico`."""
        diagnostico(self.df, "DESPUÉS")
        print("Resumen de la limpieza:")
        print(f"- Filas originales      : {self.filas_originales}")
        print(f"- Filas tras deduplicar : {len(self.df)}")
        print(f"- Nulos restantes       : {self.df.isna().sum().sum()}")
        self.resumen_especifico()

    def _iqr(self, columna: str, nombre: str):
        """Atajo para no repetir `self.df[...]` en cada subclase."""
        return resumen_iqr(self.df[columna], nombre)

    # ------------------------------------------------------------------
    # Pasos que cada dataset debe implementar a su manera.
    # ------------------------------------------------------------------
    @abstractmethod
    def verificar_categorias(self) -> None:
        """Paso 2 · Consistencia de categorías (texto o binarias 0/1)."""
        raise NotImplementedError

    @abstractmethod
    def tratar_outliers(self) -> None:
        """Paso 4 · Outliers (regla IQR) y su justificación de negocio."""
        raise NotImplementedError

    @abstractmethod
    def resumen_especifico(self) -> None:
        """Líneas de resumen propias del dataset (categorías/outliers)."""
        raise NotImplementedError
