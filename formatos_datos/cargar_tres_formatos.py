"""
cargar_tres_formatos.py
Sesión: Espectro estructurado / formatos de datos / APIs REST

Objetivo 4: Cargar un mismo conjunto de datos (insurance_muestra) desde
tres formatos distintos con pandas, interpretando los tipos que infiere.

Los tres archivos (.csv, .json, .db) contienen EXACTAMENTE los mismos
20 registros, solo cambia el formato de almacenamiento.
"""

import sqlite3
import pandas as pd

# ---------------------------------------------------------------------
# 1) CSV — el formato más simple: texto plano separado por comas.
#    pandas infiere el tipo de cada columna mirando los valores.
# ---------------------------------------------------------------------
df_csv = pd.read_csv("insurance_muestra.csv")
print("=== Desde CSV ===")
print(df_csv.head(3))
print(df_csv.dtypes)
print()

# ---------------------------------------------------------------------
# 2) JSON — formato semiestructurado: lista de objetos clave-valor.
#    Es el mismo formato en el que responden la mayoría de APIs REST.
# ---------------------------------------------------------------------
df_json = pd.read_json("insurance_muestra.json")
print("=== Desde JSON ===")
print(df_json.head(3))
print(df_json.dtypes)
print()

# ---------------------------------------------------------------------
# 3) SQL — formato estructurado por excelencia: tabla con esquema fijo
#    dentro de una base de datos relacional (aquí, SQLite).
# ---------------------------------------------------------------------
conexion = sqlite3.connect("insurance_muestra.db")
df_sql = pd.read_sql("SELECT * FROM insurance", conexion)
conexion.close()
print("=== Desde SQL (SQLite) ===")
print(df_sql.head(3))
print(df_sql.dtypes)
print()

# ---------------------------------------------------------------------
# Comprobación: los tres DataFrames deben tener los mismos datos.
# ---------------------------------------------------------------------
print("¿CSV y JSON son iguales?", df_csv.equals(df_json))
print("¿CSV y SQL son iguales? ", df_csv.equals(df_sql))

# Nota para el informe/discusión:
# - CSV y SQL suelen inferir los tipos numéricos correctamente sin ayuda.
# - JSON puede diferir ligeramente en tipos (por ejemplo, enteros que
#   quedan como int64 igual, pero conviene revisar con .dtypes siempre
#   que la fuente sea una API externa, ya que el JSON no obliga a
#   mantener un tipo fijo por columna como sí lo hace una tabla SQL.
