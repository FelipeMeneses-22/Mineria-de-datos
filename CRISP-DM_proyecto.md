# Metodología CRISP-DM aplicada al proyecto

### Predicción de Costos Médicos

Sesión 3 · Minería de Datos (NRC: 95400) - UNIMINUTO
Integrantes: Jose Manuel Salgado, Juan Felipe Meneses Rodríguez

---

## 1. ¿Qué es CRISP-DM?

CRISP-DM (Cross Industry Standard Process for Data Mining) es la metodología estándar de la industria para desarrollar proyectos de minería y ciencia de datos. Divide el trabajo en 6 fases secuenciales, aunque en la práctica es un ciclo iterativo: es normal volver a una fase anterior cuando la evaluación revela un problema en los datos o en el planteamiento del negocio.

Las 6 fases son: Comprensión del negocio, Comprensión de los datos, Preparación de los datos, Modelado, Evaluación y Despliegue. Este documento explica cómo se aplicó cada una en el proyecto, con énfasis en las fases 1 a 3, que son las desarrolladas hasta el momento.

## 2. Las 6 fases aplicadas al proyecto

| Fase CRISP-DM | Aplicación en el proyecto | Detalle / técnica usada |
| --- | --- | --- |
| **1. Comprensión del negocio** | Predecir costos médicos (insurance.csv) y primas de salud (Medicalpremium.csv) a partir de variables demográficas y clínicas, para apoyar decisiones de tarificación de seguros. | ¿Qué variables explican mejor el costo/la prima? ¿Fumar y la obesidad disparan el costo tanto como se cree? |
| **2. Comprensión de los datos** | Exploración inicial de ambos CSV: número de filas, tipos de columna, valores únicos en categóricas y estadísticos descriptivos. | Diagnóstico inicial con pandas: `.shape`, `.dtypes`, `.isna().sum()`, `.duplicated().sum()`. |
| **3. Preparación de los datos** | Pipeline de limpieza aplicado en el mismo orden a los dos datasets: Diagnóstico -> Duplicados -> Categorías -> Nulos -> Outliers -> Validación. | Se eliminó 1 duplicado exacto en insurance.csv. Se verificaron categorías (sex, smoker, region) y columnas binarias (Diabetes, AnyTransplants, etc.). Se aplicó la regla IQR/Tukey para outliers, cruzando charges con smoker y Weight con Height para decidir si conservarlos. |
| **4. Modelado** | Etapa siguiente del proyecto (no desarrollada en esta sesión): entrenar un modelo de regresión que prediga charges o PremiumPrice. | Pendiente: codificar variables categóricas, dividir train/test, probar regresión lineal o árboles de decisión. |
| **5. Evaluación** | Etapa siguiente: medir el desempeño del modelo (error, R²) y confirmar si responde la pregunta de negocio. | Pendiente: métricas como RMSE o MAE, y validación de que las variables clave (fumar, IMC) tengan el peso esperado. |
| **6. Despliegue** | Etapa siguiente: dejar el modelo disponible para estimar costos/primas de nuevos asegurados. | Pendiente: definir cómo se usará el modelo (script, API, dashboard). |

## 3. Detalle de la fase de Preparación de los datos

Esta es la fase más desarrollada hasta ahora. Se construyó un pipeline reutilizable (con clases en Python, aplicando el patrón Template Method) que ejecuta los siguientes pasos, en este orden, sobre ambos datasets:

- **Diagnóstico inicial:** número de filas, nulos por columna y duplicados exactos.
- **Eliminación de duplicados:** se detectó y eliminó 1 fila duplicada exacta en insurance.csv (patrón MCAR); Medicalpremium.csv no tenía duplicados.
- **Verificación de categorías:** columnas de texto (sex, smoker, region) en insurance.csv, y columnas binarias (Diabetes, AnyTransplants, etc.) en Medicalpremium.csv.
- **Revisión de nulos:** 0 valores faltantes en ambos datasets, por tratarse de datasets ya curados.
- **Tratamiento de outliers:** regla IQR/Tukey, cruzando los valores extremos con otras columnas para decidir si son errores o datos reales.
- **Validación final:** comparación de filas y nulos antes/después de la limpieza.

### Decisiones sobre outliers

El hallazgo metodológico más importante del proyecto: los valores marcados como "outliers" por la regla estadística no eran errores de captura, sino la señal central del problema de negocio. Por eso, en los dos datasets, se decidió conservar todos los outliers detectados:

| Variable | Decisión | Justificación |
| --- | --- | --- |
| **bmi / Weight** | Se conservan | Valores altos corresponden a obesidad severa real, no a errores de digitación (verificado cruzando Weight con Height). |
| **charges** | Se conservan | 98% de los outliers son personas fumadoras: es la señal central del proyecto, no ruido. |
| **PremiumPrice** | Se conservan | Es la variable objetivo del dataset; los valores altos son primas reales del segmento de mayor riesgo. |
| **NumberOfMajorSurgeries** | Se conserva (no aplica IQR) | Variable discreta de solo 4 valores posibles; la regla de Tukey sobre-marca outliers en conteos de rango tan corto. |

## 4. Próximos pasos (fases 4 a 6)

Con los datos ya limpios y validados (`insurance_limpio.csv` y `Medicalpremium_limpio.csv`), el proyecto continúa con:

- **Modelado:** codificar las variables categóricas (sex, smoker, region) y entrenar un modelo de regresión que prediga charges o PremiumPrice.
- **Evaluación:** medir el error del modelo (por ejemplo RMSE o MAE) y confirmar si las variables de mayor peso (fumar, IMC, cirugías) coinciden con lo esperado.
- **Despliegue:** definir cómo se pondrá a disposición el modelo final (script de predicción, API o dashboard).

## 5. Conclusión

Seguir CRISP-DM permitió documentar cada decisión de limpieza con evidencia y no solo con intuición, y dejar el proyecto en un punto reproducible: cualquier persona puede correr el mismo pipeline sobre los datos crudos y llegar exactamente a los mismos datasets limpios. La fase de preparación confirmó además una idea central para las fases siguientes: las variables asociadas a hábitos de riesgo (fumar) y condiciones de salud (obesidad, cirugías, enfermedades crónicas) no son ruido a eliminar, sino los predictores más importantes que el modelo deberá capturar.
