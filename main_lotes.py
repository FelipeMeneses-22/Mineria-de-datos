"""
main_lotes.py
Corre limpieza y transformación POR LOTES para los tres datasets.
No reemplaza a main.py: escribe en carpetas aparte (data/limpio_lotes y
data/transformado_lotes) para no pisar los resultados convencionales.

Uso:  python main_lotes.py            (lotes de 10000 filas)
      python main_lotes.py 5000       (lotes de 5000 filas)
"""

import sys
from herramientas.transformaciones_lotes import transformar_por_lotes
from limpieza.lotes_cleaner import BatchCleaner
from config import DATASETS, RAIZ

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
