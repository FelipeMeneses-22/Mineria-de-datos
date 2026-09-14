"""
estadisticas.py
Cálculos estadísticos reutilizables. Separado de las clases de limpieza
porque es lógica genérica (regla de Tukey) que no depende de ningún
dataset en particular: se le puede pasar CUALQUIER columna numérica.
"""

import pandas as pd


def resumen_iqr(serie: pd.Series, nombre: str):
    """Calcula Q1, Q3, IQR y límites de Tukey; reporta cuántos valores
    quedan fuera de esos límites. Devuelve (lim_inf, lim_sup, outliers)."""
    q1, q3 = serie.quantile([.25, .75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    print(f"{nombre}: Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
          f"limites=[{lim_inf:.2f}, {lim_sup:.2f}]  outliers={len(outliers)}")
    return lim_inf, lim_sup, outliers
