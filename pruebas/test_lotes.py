"""
pruebas/test_lotes.py
Pruebas automáticas del procesamiento por lotes.
Comprueban que hacerlo por bloques da EL MISMO resultado que hacerlo todo junto.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s pruebas -v
(no necesita instalar nada extra: usa la librería estándar)
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))                     # para poder importar el proyecto

from herramientas.transformaciones_lotes import transformar_por_lotes   # noqa: E402
from limpieza.lotes_cleaner import BatchCleaner                         # noqa: E402


def silencio():
    """BatchCleaner imprime mucho; en las pruebas se oculta."""
    return contextlib.redirect_stdout(io.StringIO())


def tabla_de_prueba() -> pd.DataFrame:
    """200 filas con filas repetidas en bloques DISTINTOS y categorías ordenadas."""
    generador = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": generador.normal(50, 10, 200).round(2),
        "z": generador.integers(0, 100, 200),
        "cat": ["a"] * 70 + ["b"] * 70 + ["c"] * 60,   # el primer bloque solo tiene 'a'
        "y": generador.integers(0, 2, 200),
    })
    df.loc[150] = df.loc[3]          # repetida lejos de su original (bloques distintos)
    df.loc[199] = df.loc[3]
    df.loc[40] = df.loc[39]          # repetida dentro del mismo bloque
    return df


class PruebasLimpiezaPorLotes(unittest.TestCase):
    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.dir = Path(self.carpeta.name)
        self.df = tabla_de_prueba()
        self.raw = self.dir / "raw.csv"
        self.df.to_csv(self.raw, index=False)

    def tearDown(self):
        self.carpeta.cleanup()

    def test_resultado_igual_al_normal(self):
        salida = self.dir / "limpio.csv"
        with silencio():
            registro = BatchCleaner(self.raw, salida, "prueba", ["cat"], ["x"],
                                    tamano_lote=30).ejecutar()
        esperado = self.df.drop_duplicates().reset_index(drop=True)
        obtenido = pd.read_csv(salida)
        pd.testing.assert_frame_equal(obtenido, esperado)
        self.assertEqual(registro["duplicados_eliminados"], 3)
        self.assertEqual(registro["lotes_con_error"], 0)

    def test_las_cuentas_cuadran(self):
        with silencio():
            r = BatchCleaner(self.raw, self.dir / "l.csv", "prueba", ["cat"], ["x"],
                             tamano_lote=30).ejecutar()
        self.assertEqual(r["registros_leidos"],
                         r["registros_escritos"] + r["duplicados_eliminados"]
                         + r["filas_en_lotes_con_error"])

    def test_no_escribe_sobre_el_original(self):
        with self.assertRaises(ValueError):
            BatchCleaner(self.raw, self.raw, "prueba", ["cat"], ["x"])

    def test_un_bloque_con_error_se_cuenta_y_no_detiene_todo(self):
        with silencio():
            r = BatchCleaner(self.raw, self.dir / "l.csv", "prueba", ["columna_que_no_existe"],
                             [], tamano_lote=50).ejecutar()
        self.assertEqual(r["lotes_con_error"], r["lotes"])
        self.assertEqual(r["registros_escritos"], 0)


class PruebasTransformacionPorLotes(unittest.TestCase):
    def test_igual_a_la_transformacion_normal(self):
        df = tabla_de_prueba().drop_duplicates().reset_index(drop=True)
        with tempfile.TemporaryDirectory() as carpeta:
            entrada, salida = Path(carpeta) / "e.csv", Path(carpeta) / "s.csv"
            df.to_csv(entrada, index=False)
            transformar_por_lotes(entrada, salida, numericas=["x", "z"], categoricas=["cat"],
                                  binarias=[], objetivo="y", tamano_lote=25)
            lotes = pd.read_csv(salida)

        normal = pd.concat([
            pd.DataFrame(StandardScaler().fit_transform(df[["x", "z"]]), columns=["x", "z"]),
            pd.get_dummies(df[["cat"]], prefix="cat", drop_first=True, dtype=int),
            df[["y"]],
        ], axis=1)
        pd.testing.assert_frame_equal(lotes, normal, check_exact=False, rtol=1e-9, atol=1e-9)

    def test_todos_los_bloques_tienen_las_mismas_columnas(self):
        """Aunque el primer bloque solo tenga la categoría 'a', salen las columnas cat_b y cat_c."""
        df = tabla_de_prueba().drop_duplicates().reset_index(drop=True)
        with tempfile.TemporaryDirectory() as carpeta:
            entrada, salida = Path(carpeta) / "e.csv", Path(carpeta) / "s.csv"
            df.to_csv(entrada, index=False)
            transformar_por_lotes(entrada, salida, ["x"], ["cat"], [], "y", tamano_lote=20)
            columnas = list(pd.read_csv(salida).columns)
        self.assertEqual(columnas, ["x", "cat_b", "cat_c", "y"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
