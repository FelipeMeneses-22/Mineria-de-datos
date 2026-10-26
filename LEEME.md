# Proyecto Minería de Datos

## Instalación (una sola vez)
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Linux/Mac)
    pip install -r requirements.txt

## Ejecutar (desde esta carpeta, con el entorno activo)
    python main_lotes.py                      # limpieza + transformación por lotes
    python benchmark_lotes.py adultos_salud 3 # convencional vs lotes (tiempo y memoria)
    python arbol_clasificacion.py             # árbol sobre adultos_salud -> graficas/adultos_salud/
    python arbol_regresion.py                 # árboles sobre insurance y Medicalpremium -> graficas/

## Carpetas
- data/raw/      originales sin limpiar (aquí solo adultos_salud.csv; agrega insurance.csv y Medicalpremium.csv si los tienes)
- data/limpio/   CSV limpios que leen los árboles
- herramientas/, limpieza/   funciones y clases del pipeline (incluye el procesamiento por lotes)
- graficas/      imágenes ya generadas

## Nota
No incluí main.py, formatos_datos/ ni CRISP-DM_proyecto.md: no los recibí. Conserva los tuyos.
