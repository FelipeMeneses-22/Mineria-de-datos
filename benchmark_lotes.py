"""
benchmark_lotes.py
Compara procesamiento CONVENCIONAL (todo en memoria) vs POR LOTES,
midiendo tiempo y memoria con los datos reales de data/raw/.

Cada medición corre en un proceso Python NUEVO: así el pico de memoria
de una prueba no contamina a la siguiente.

Uso:  python benchmark_lotes.py                    (dataset adultos_salud)
      python benchmark_lotes.py insurance          (otro dataset de DATASETS)
      python benchmark_lotes.py adultos_salud 3    (3 repeticiones; mediana)

Memoria: usa el módulo 'resource' (Linux/Mac). En Windows requiere
`pip install psutil`.
Se reporta:  pico_mb = pico total del proceso
             extra_mb = pico - memoria base (ya con pandas/sklearn importados)
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from herramientas.estadisticas import resumen_iqr
from herramientas.transformaciones_lotes import transformar_por_lotes
from limpieza.lotes_cleaner import BatchCleaner
from main_lotes import DATASETS, RAIZ

TAMANOS = [5000, 10000, 50000]
CARPETA = RAIZ / "data" / "benchmark"


def memoria_pico_mb() -> float:
    try:
        import resource
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return pico / 1e6 if sys.platform == "darwin" else pico / 1024
    except ImportError:                              # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e6


# ----------------------------------------------------------------------
# Versiones CONVENCIONALES (lo mismo que hace el pipeline actual:
# cargar todo el DataFrame en memoria).
# ----------------------------------------------------------------------
def limpieza_convencional(cfg, ruta_raw, ruta_salida):
    df = pd.read_csv(ruta_raw)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in cfg["verificar"]:
        sorted(df[col].unique())
    df.isna().sum()
    for col in cfg["iqr"]:
        resumen_iqr(df[col], col)
    df.to_csv(ruta_salida, index=False)


def transformacion_convencional(cfg, ruta_limpio, ruta_salida):
    t = cfg["transf"]
    df = pd.read_csv(ruta_limpio)
    X_std = pd.DataFrame(StandardScaler().fit_transform(df[t["numericas"]]),
                         columns=t["numericas"])
    partes = [X_std]
    if t["categoricas"]:
        partes.append(pd.get_dummies(df[t["categoricas"]],
                                     prefix=t["categoricas"],
                                     drop_first=True, dtype=int))
    partes += [df[t["binarias"]], df[[t["objetivo"]]]]
    pd.concat(partes, axis=1).to_csv(ruta_salida, index=False)


# ----------------------------------------------------------------------
# Proceso hijo: ejecuta UNA prueba y escribe su resultado como JSON.
# ----------------------------------------------------------------------
def correr_hijo(nombre, fase, modo, tamano):
    cfg = DATASETS[nombre]
    ruta_raw = RAIZ / "data" / "raw" / cfg["raw"]
    base_limpio = CARPETA / f"{nombre}_limpieza_convencional.csv"
    sufijo = "convencional" if modo == "convencional" else f"lotes_{tamano}"
    salida = CARPETA / f"{nombre}_{fase}_{sufijo}.csv"

    base = memoria_pico_mb()
    inicio = time.perf_counter()
    if fase == "limpieza" and modo == "convencional":
        limpieza_convencional(cfg, ruta_raw, salida)
    elif fase == "limpieza":
        BatchCleaner(ruta_raw, salida, cfg["raw"], cfg["verificar"],
                     cfg["iqr"], tamano).ejecutar()
    elif modo == "convencional":
        transformacion_convencional(cfg, base_limpio, salida)
    else:
        transformar_por_lotes(base_limpio, salida, tamano_lote=tamano,
                              **cfg["transf"])
    segundos = time.perf_counter() - inicio
    pico = memoria_pico_mb()
    print("RESULTADO_JSON " + json.dumps(
        {"segundos": segundos, "pico_mb": pico, "extra_mb": pico - base}))


def medir(nombre, fase, modo, tamano, repeticiones):
    corridas = []
    for _ in range(repeticiones):
        p = subprocess.run(
            [sys.executable, __file__, "--hijo", nombre, fase, modo, str(tamano)],
            capture_output=True, text=True, cwd=RAIZ)
        linea = [l for l in p.stdout.splitlines() if l.startswith("RESULTADO_JSON")]
        if p.returncode != 0 or not linea:
            raise RuntimeError(p.stderr[-800:])
        corridas.append(json.loads(linea[0].split(" ", 1)[1]))
    tabla = pd.DataFrame(corridas)
    return tabla.median().round(2).to_dict()       # mediana de las repeticiones


def md5(ruta):
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def main(nombre="adultos_salud", repeticiones=3):
    CARPETA.mkdir(parents=True, exist_ok=True)
    raw = RAIZ / "data" / "raw" / DATASETS[nombre]["raw"]
    filas = []
    for fase in ["limpieza", "transformacion"]:
        for modo, tamano in [("convencional", 0)] + [("lotes", t) for t in TAMANOS]:
            r = medir(nombre, fase, modo, tamano, repeticiones)
            etiqueta = "convencional" if modo == "convencional" else f"lotes {tamano}"
            filas.append({"fase": fase, "modo": etiqueta, **r})
            print(f"{fase:<15} {etiqueta:<14} {r['segundos']:>7.2f} s  "
                  f"pico {r['pico_mb']:>7.1f} MB  extra {r['extra_mb']:>7.1f} MB")

    resultados = pd.DataFrame(filas)
    resultados.to_csv(CARPETA / f"resultados_{nombre}.csv", index=False)

    # ---- Verificación: ¿los dos métodos producen los mismos datos? ----
    print("\nVerificación de resultados idénticos")
    conv = CARPETA / f"{nombre}_limpieza_convencional.csv"
    for t in TAMANOS:
        igual = md5(conv) == md5(CARPETA / f"{nombre}_limpieza_lotes_{t}.csv")
        print(f"  limpieza      lotes {t:>6}: archivo idéntico al convencional -> {igual}")
    conv_t = pd.read_csv(CARPETA / f"{nombre}_transformacion_convencional.csv")
    for t in TAMANOS:
        lot_t = pd.read_csv(CARPETA / f"{nombre}_transformacion_lotes_{t}.csv")
        try:
            pd.testing.assert_frame_equal(conv_t, lot_t, check_exact=False,
                                          rtol=1e-9, atol=1e-9)
            igual = True
        except AssertionError:
            igual = False
        print(f"  transformacion lotes {t:>6}: igual (tolerancia 1e-9) -> {igual}")
    print(f"\nTamaño del archivo original: {raw.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--hijo":
        correr_hijo(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "adultos_salud",
             int(sys.argv[2]) if len(sys.argv) > 2 else 3)
