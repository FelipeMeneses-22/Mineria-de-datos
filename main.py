"""
main.py
Punto de entrada ÚNICO del proyecto: desde aquí se ejecuta cualquier paso.

Uso (desde la raíz del proyecto, con el entorno activo):

    python main.py                       muestra los pasos disponibles
    python main.py limpieza              limpieza convencional de insurance y Medicalpremium
    python main.py limpieza --lotes      la misma limpieza pero por bloques (los 3 datasets)
    python main.py lotes                 igual que lo anterior (limpieza + transformación por bloques)
    python main.py transformar           escalado, one-hot y PCA (pipeline convencional)
    python main.py arbol                 el árbol de decisión único
    python main.py modelos               árbol, Naive Bayes y KNN sobre la tabla unificada
    python main.py comparar              comparación con validación cruzada (clasificación y costos)
    python main.py clustering            K-means y dendrograma (o: clustering insurance)
    python main.py pruebas               pruebas automáticas del procesamiento por lotes
    python main.py todo                  todos los pasos, en ese orden (tarda varios minutos)

Opciones:
    --lotes             usar el procesamiento por bloques en el paso 'limpieza'
    --tamano-lote N     filas por bloque (por defecto 10000)
    --sobrescribir      permitir que 'limpieza' reescriba archivos de data/limpio/

Por seguridad, 'limpieza' NO reescribe un archivo que ya existe en data/limpio/
(esos archivos son la entrada de los modelos). Use --sobrescribir si es lo que quiere.
"""
import argparse
import importlib
import sys
import time
import unittest

from config import LIMPIO, RAIZ, RAW

sys.path.insert(0, str(RAIZ))        # para importar los módulos del proyecto desde cualquier lugar


# ----------------------------------------------------------------------
# Pasos
# ----------------------------------------------------------------------
def paso_limpieza(args) -> None:
    """Limpieza convencional (todo el archivo en memoria) con las clases de limpieza/."""
    if args.lotes:
        return paso_lotes(args)

    from limpieza.insurance_cleaner import InsuranceCleaner
    from limpieza.medicalpremium_cleaner import MedicalPremiumCleaner

    # Para agregar un dataset nuevo: crear su clase en limpieza/ y añadirla a esta lista.
    limpiadores = [("insurance", InsuranceCleaner), ("Medicalpremium", MedicalPremiumCleaner)]
    for nombre, clase in limpiadores:
        raw = RAW / f"{nombre}.csv"
        limpio = LIMPIO / f"{nombre}_limpio.csv"
        if not raw.exists():
            print(f"[omitido] {nombre}: no existe {raw.relative_to(RAIZ)}")
            continue
        if limpio.exists() and not args.sobrescribir:
            print(f"[omitido] {nombre}: ya existe {limpio.relative_to(RAIZ)} "
                  f"(use --sobrescribir para reescribirlo)")
            continue
        limpio.parent.mkdir(parents=True, exist_ok=True)
        clase(str(raw), str(limpio)).ejecutar()      # cada limpiador se usa igual (DataCleaner)


def paso_lotes(args) -> None:
    importlib.import_module("main_lotes").main(args.tamano_lote)


def paso_transformar(args) -> None:
    importlib.import_module("herramientas.transformaciones").main()


def paso_arbol(args) -> None:
    importlib.import_module("arbol_general").main()


def paso_modelos(args) -> None:
    importlib.import_module("modelado_unificado").main()


def paso_comparar(args) -> None:
    importlib.import_module("modelado_justo").main()
    importlib.import_module("modelado_regresion").main()


def paso_clustering(args) -> None:
    clustering = importlib.import_module("clustering")
    claves = args.datos or list(clustering.CONFIGS)
    for clave in claves:
        if clave not in clustering.CONFIGS:
            raise SystemExit(f"Dataset desconocido para clustering: {clave}. "
                             f"Opciones: {', '.join(clustering.CONFIGS)}")
        clustering.segmentar(clave)


def paso_pruebas(args) -> None:
    suite = unittest.defaultTestLoader.discover(str(RAIZ / "pruebas"))
    resultado = unittest.TextTestRunner(verbosity=1).run(suite)
    if not resultado.wasSuccessful():
        raise RuntimeError("Fallaron pruebas automáticas")


PASOS = {                                   # el orden de este diccionario es el orden de 'todo'
    "limpieza": paso_limpieza,
    "lotes": paso_lotes,
    "transformar": paso_transformar,
    "arbol": paso_arbol,
    "modelos": paso_modelos,
    "comparar": paso_comparar,
    "clustering": paso_clustering,
    "pruebas": paso_pruebas,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Punto de entrada del proyecto de Minería de Datos",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("paso", nargs="?", choices=list(PASOS) + ["todo"],
                        help="paso a ejecutar (sin paso se muestra la ayuda)")
    parser.add_argument("datos", nargs="*", help="solo para 'clustering': adultos_salud y/o insurance")
    parser.add_argument("--lotes", action="store_true", help="limpieza por bloques")
    parser.add_argument("--tamano-lote", type=int, default=10000, help="filas por bloque")
    parser.add_argument("--sobrescribir", action="store_true", help="reescribir data/limpio/")
    args = parser.parse_args(argv)

    if args.paso is None:
        parser.print_help()
        return 0

    pasos = list(PASOS) if args.paso == "todo" else [args.paso]
    resumen = []
    for nombre in pasos:
        print(f"\n{'#' * 70}\n# PASO: {nombre}\n{'#' * 70}")
        inicio = time.perf_counter()
        try:
            PASOS[nombre](args)
            estado = "OK"
        except (Exception, SystemExit) as error:        # un paso que falla no oculta a los demás
            estado = f"FALLÓ: {error}"
            print(f"\n*** El paso '{nombre}' falló: {error}")
        resumen.append((nombre, estado, time.perf_counter() - inicio))

    if len(pasos) > 1:
        print(f"\n{'=' * 70}\nRESUMEN\n{'=' * 70}")
        for nombre, estado, segundos in resumen:
            print(f"{nombre:<12} {estado:<40} {segundos:>7.1f} s")
    return 0 if all(e == "OK" for _, e, _ in resumen) else 1


if __name__ == "__main__":
    sys.exit(main())
