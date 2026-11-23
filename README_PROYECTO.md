# Proyecto de Minería de Datos: del dato crudo a los modelos (árbol, Naive Bayes, KNN y clustering)

**Versión actual: 1.14 (22 de noviembre de 2026)**

Proyecto con tres conjuntos de datos (`adultos_salud`, `insurance` y `Medicalpremium`) que recorre las fases de CRISP-DM: entender los datos, prepararlos (limpieza, transformación y procesamiento por lotes) y modelarlos con **un árbol de decisión único**, compararlo con **Naive Bayes y KNN** y segmentar a las personas con **clustering (K-means y jerárquico)**.

### Resumen de resultados

| Qué | Resultado clave | Dónde |
|---|---|---|
| Árbol único (3 datasets, `riesgo_alto`) | Detecta en promedio el 75 % del riesgo alto (entre 57 % y 85 % según la muestra; 61,5 % con la semilla 42); rinde distinto en cada dataset y el `-1` delata el origen | Sección 6 |
| Árbol vs Naive Bayes vs KNN (misma partición) | Ninguno gana en todo: árbol detecta más (0,615), KNN tiene mejor F1 (0,473), Naive Bayes solo funciona con `priors` balanceados | Sección 7b |
| Clustering de `adultos_salud` (k = 4) | 4 segmentos con enfermedad cardíaca de 1,7 % a 23,0 % | Sección 7c |
| Clustering de `insurance` (k = 4) | Fumadores = un grupo aparte (costo medio 32.230 vs 4.201-11.683); los no fumadores se ordenan por edad e hijos | Sección 7c |
| Clasificadores con validación cruzada (`adultos_salud`) | La regresión logística separa mejor que el árbol (ROC AUC 0,83 frente a 0,72) | Sección 7d |
| Predecir costos (`insurance`, `Medicalpremium`) | Random Forest es el mejor: R² 0,88 y 0,79 | Sección 7e |

> Este README se actualiza en cada entrega. Al final está el **historial de cambios**: cada vez que se agregue o modifique algo, sube la versión y se añade una línea.

---

## 1. Cómo ejecutarlo

### Instalación (una sola vez)

```bash
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux / Mac
pip install -r requirements.txt
```

### Ejecutar (desde la carpeta raíz del proyecto, con el entorno activo)

| Qué quieres hacer | Comando |
|---|---|
| **Ver todos los pasos disponibles** | `python main.py` |
| **Ejecutar un paso desde un solo punto** (`limpieza`, `lotes`, `transformar`, `arbol`, `modelos`, `comparar`, `clustering`, `pruebas`) | `python main.py arbol` |
| **Ejecutar todos los pasos en orden** (unos 2 minutos) | `python main.py todo` |
| Ver todo el proyecto paso a paso, con gráficos | `jupyter notebook proyecto_mineria.ipynb` |
| Generar el **árbol único** de los tres datasets | `python arbol_general.py` |
| Limpiar y transformar por lotes los tres datasets | `python main_lotes.py` |
| Comparar procesamiento normal vs por lotes (tarda unos minutos) | `python benchmark_lotes.py adultos_salud 3` |
| Comparar árbol, Naive Bayes y KNN sobre `adultos_salud` | `python modelado.py` |
| Comparar árbol, Naive Bayes y KNN en igualdad de condiciones (misma tabla y partición del árbol único) | `python modelado_unificado.py` |
| Segmentar `adultos_salud` e `insurance` con K-means y dendrograma | `python clustering.py` (o `python clustering.py insurance` para uno solo) |
| Comparar clasificadores con validación cruzada y el mismo trato al desbalance | `python modelado_justo.py` |
| Comparar modelos para predecir costos (`insurance` y `Medicalpremium`) | `python modelado_regresion.py` |
| Correr las pruebas automáticas del procesamiento por lotes | `python -m unittest discover -s pruebas -v` |
| Pipeline de transformación convencional (escalado, one-hot, PCA) | `python -m herramientas.transformaciones` |
| Árboles anteriores, uno por dataset (ya no son necesarios) | `python arbol_clasificacion.py` y `python arbol_regresion.py` |

En el cuaderno, ejecuta las celdas en orden con `Shift + Enter`.

**Sobre `main.py`.** Es el punto de entrada único: cada paso llama al script correspondiente (por ejemplo, `python main.py arbol` hace lo mismo que `python arbol_general.py`). Opciones: `--lotes` (limpieza por bloques), `--tamano-lote N` y `--sobrescribir`. Por seguridad, el paso `limpieza` **no reescribe** archivos que ya existen en `data/limpio/` (son la entrada de los modelos); hay que pedirlo con `--sobrescribir`. En `todo`, un paso que falla no oculta a los demás y al final se muestra un resumen con el estado y el tiempo de cada uno.

---

## 2. Estructura del proyecto

```
proyecto_mineria/
├── main.py                   punto de entrada único: python main.py <paso>
├── config.py                 columnas y rutas de cada dataset (un solo lugar)
├── data/
│   ├── raw/                  originales sin limpiar (aquí solo adultos_salud.csv)
│   ├── limpio/               adultos_salud, insurance y Medicalpremium ya limpios
│   ├── benchmark/            resultados medidos del procesamiento por lotes
│   └── resultados/           salidas de arbol_general.py, modelado*.py y clustering.py (CSV y matrices de confusión)
├── herramientas/             funciones reutilizables
│   ├── diagnostico.py        revisar filas, nulos y duplicados
│   ├── estadisticas.py       regla IQR para valores extremos
│   ├── io_datos.py           leer/guardar CSV (incluye lectura por lotes)
│   ├── transformaciones.py   escalado, one-hot, PCA (pipeline convencional)
│   └── transformaciones_lotes.py   escalado y one-hot por lotes, en dos pasadas
├── limpieza/                 clases de limpieza (POO)
│   ├── base_cleaner.py       DataCleaner: orden fijo del pipeline
│   ├── insurance_cleaner.py  lo propio de insurance
│   ├── medicalpremium_cleaner.py   lo propio de Medicalpremium
│   └── lotes_cleaner.py      BatchCleaner: la misma limpieza, por bloques
├── pruebas/                  pruebas automáticas del procesamiento por lotes
├── graficas/                 imágenes generadas (arbol_general, cuaderno, modelado, clustering/<dataset> y por dataset)
├── presentacion/             dos presentaciones HTML (ver sección 9)
├── docs/                     decisiones_del_codigo.docx (ver sección 9)
├── informes/                 EDA_insurance.html: informe exploratorio de insurance (ver sección 10)
├── proyecto_mineria.ipynb    cuaderno de Jupyter con todo el proyecto
├── arbol_general.py          el árbol único (los 3 datasets)
├── modelado.py               compara árbol, Naive Bayes y KNN sobre adultos_salud
├── modelado_unificado.py     compara árbol, Naive Bayes y KNN sobre la tabla unificada (sección 7b)
├── clustering.py             K-means y dendrograma sobre adultos_salud e insurance (sección 7c)
├── modelado_justo.py         clasificadores con validación cruzada y el mismo trato al desbalance (sección 7d)
├── modelado_regresion.py     regresión lineal, árbol y Random Forest para predecir costos (sección 7e)
├── main_lotes.py             limpieza y transformación por lotes
├── benchmark_lotes.py        comparación de tiempo y memoria
├── arbol_clasificacion.py    árbol solo de adultos_salud (anterior)
├── arbol_regresion.py        árboles de insurance y Medicalpremium (anteriores)
├── ejemplo_arbol_fuga_clase.py   ejemplo de clase (TelecomUNO), sin cambios
├── requirements.txt
└── README_PROYECTO.md
```

**`main.py` es una reconstrucción (v1.14).** El `main.py` original no se recibió; este se escribió a partir de su descripción (correr los limpiadores de `insurance` y `Medicalpremium` de forma intercambiable) y se le sumaron los demás pasos. **Si ya tienes tu `main.py`, no lo pises sin revisar:** si hacía algo más, súbelo y se fusiona.

**Archivos tuyos que NO vinieron en este zip:** `formatos_datos/` y `CRISP-DM_proyecto.md`. Si ya los tienes, copia esta carpeta encima: no los pisa. Este README reemplaza al anterior (`README_PROYECTO.md`, que describía carpetas con nombres antiguos).

---

## 3. Los tres conjuntos de datos

| Datos | Filas (limpias) | Qué predice | Tipo |
|---|---|---|---|
| `adultos_salud` | 301.717 (de 319.795 originales) | `HeartDisease` (Sí/No) | Clasificación |
| `insurance` | 500.000 | `charges` (costo médico) | Regresión |
| `Medicalpremium` | 986 | `PremiumPrice` (prima) | Regresión |

De `insurance` y `Medicalpremium` solo se dispone de las versiones limpias; los originales no están en `data/raw/`.

---

## 4. Diseño de la limpieza: patrón Template Method y SOLID

`DataCleaner` (`limpieza/base_cleaner.py`) fija el orden del pipeline en un único método `ejecutar()`:

```
Diagnóstico -> Duplicados -> Categorías -> Nulos -> Outliers -> Validación
```

Los pasos iguales para todos (duplicados, nulos, diagnóstico) están una sola vez en la clase base. Los pasos propios de cada dataset (categorías y valores extremos) son métodos abstractos que implementa cada cleaner.

| Principio | Dónde se ve |
|---|---|
| **S** (responsabilidad única) | `base_cleaner.py` solo orquesta; leer/escribir vive en `io_datos.py`, el diagnóstico en `diagnostico.py` y el IQR en `estadisticas.py`. |
| **O** (abierto/cerrado) | Para un dataset nuevo se crea una clase nueva en `limpieza/`, sin tocar las existentes. |
| **L** (Liskov) | Cualquier cleaner se usa como un `DataCleaner`: se llama `.ejecutar()` sin saber cuál es. |
| **I** (segregación de interfaces) | La clase base exige tres métodos abstractos pequeños (`verificar_categorias`, `tratar_outliers`, `resumen_especifico`). |
| **D** (inversión de dependencias) | `DataCleaner` depende de funciones de `herramientas/`, no de detalles de un dataset. |

**Decisión sobre valores extremos:** se detectan con la regla IQR pero **no se borran**: son casos reales (fumadores con costos altos, obesidad severa, primas altas) y borrarlos sesgaría el modelo justo en el segmento de mayor riesgo.

---

## 5. Procesamiento por lotes (Batch Processing)

Único método de escalabilidad del proyecto: se lee el archivo en bloques de filas (`chunksize`), se procesa cada bloque y se guarda antes de leer el siguiente. Usa solo Python y pandas.

Puntos delicados que se resolvieron:

- **Duplicados entre bloques:** se guarda una huella (hash) de cada fila ya escrita para descartar repetidas aunque estén en bloques distintos.
- **Valores extremos (IQR):** los cuartiles son globales; el reporte se calcula al final leyendo solo las columnas numéricas.
- **Escalar y codificar:** necesita dos pasadas (primero aprender promedio, desviación y categorías; luego transformar bloque a bloque).
- **Seguridad:** nunca escribe sobre el archivo original; si un bloque falla, lo cuenta y sigue; verifica que `leídas = escritas + duplicadas + con error`.

**Medición real** con `adultos_salud` (25 MB, bloques de 10.000 filas, mediana de 3 repeticiones):

| Fase | Normal | Por lotes |
|---|---|---|
| Memoria extra al limpiar | 110 MB | 36 MB |
| Memoria extra al preparar | 157 MB | 17 MB |
| Tiempo al limpiar | 2,4 s | 4,5 s |
| Tiempo al preparar | 4,1 s | 5,5 s |
| Resultados | — | idénticos |

Se gana memoria y se pierde algo de tiempo. Con archivos de este tamaño no hace falta; la ventaja real aparece cuando el archivo se acerca al tamaño de la memoria disponible. Los resultados por lotes se guardan en `data/limpio_lotes/` y `data/transformado_lotes/` para no pisar los archivos limpios.

---

## 6. El árbol de decisión único (`arbol_general.py` y cuaderno, partes 7 y 8)

Un solo árbol que junta los tres datasets. Como tienen columnas y objetivos distintos, hubo que decidir:

| Problema | Decisión |
|---|---|
| Columnas distintas | Solo las variables compartidas por al menos dos datasets: `edad`, `imc`, `hombre`, `fuma`, `diabetes`. Lo que no existe en un dataset se marca `-1`. |
| Objetivos distintos | Objetivo común `riesgo_alto`: enfermedad cardíaca (`adultos_salud`) o el 9 % más alto en costo o prima (`insurance`, `Medicalpremium`). Se usa el mismo 9 % en los tres. |
| Tamaños distintos | Muestra de 986 filas de cada dataset (el tamaño del más pequeño). |
| Profundidad | 4 preguntas seguidas, elegida por validación cruzada solo con datos de entrenamiento. |
| Pocos casos de riesgo alto | `class_weight="balanced"`. |

**Resultados.** La prueba tiene solo 740 personas, así que el resultado de una sola muestra puede engañar. Por eso `arbol_general.py` repite todo el proceso (muestra, separación y árbol) con 20 semillas distintas y se debe **citar el rango**:

| Métrica (datos de prueba) | Una muestra (semilla 42) | Promedio de 20 muestras (mín. – máx.) |
|---|---|---|
| Riesgo alto detectado | 61,5 % | 75 % (57 % – 85 %) |
| Precisión | 31 % | 26 % (19 % – 41 %) |
| Aciertos totales | 84,6 % | 77 % (67 % – 88 %) |
| Detectado en `insurance` | 100 % | 96 % (88 % – 100 %) |
| Detectado en `adultos_salud` | 52 % | 74 % (41 % – 100 %) |
| Detectado en `Medicalpremium` | 32 % | 54 % (32 % – 73 %) |

- La muestra de semilla 42 fue de las más bajas: el árbol detecta bastante riesgo alto, pero con muchas falsas alarmas (de cada 4 casos que marca, solo 1 lo es).
- **No funciona igual en los tres:** casi siempre acierta en `insurance`; en los otros dos varía mucho.
- Variables más usadas (semilla 42): IMC 30 %, edad 29 %, diabetes 23 %, fumar 17 %.
- Los resultados de la sección 7b usan la partición de la semilla 42; deben leerse con este rango en mente.

**Límites conocidos:**

- El riesgo alto significa cosas distintas en cada dataset (enfermedad, costo, prima).
- **El `-1` delata el dataset de origen.** Cada dataset tiene un patrón de `-1` único (`insurance` no tiene diabetes; `Medicalpremium` no tiene sexo ni fumar; `adultos_salud` tiene todo). Un árbol que solo ve dónde hay `-1` adivina el dataset de origen con 100 % de acierto (cuaderno 8.9b), y el árbol real lo usa: sus reglas incluyen `diabetes <= -0,50` (aísla `insurance`) y `fuma <= -0,50` (aísla `Medicalpremium`). En la práctica son tres modelos pegados en uno, no un modelo único de riesgo, y por eso rinde distinto en cada dataset. Además `-1` es un número falso: `fuma <= 0,5` mezcla "no fuma" con "no se sabe".
- La edad de `adultos_salud` es el punto medio de cada rango.
- La prueba es pequeña (por eso se da un rango) y la profundidad 4 se eligió con una sola muestra.

---

## 7. Comparación de modelos (`modelado.py`)

Hecho en otra conversación (sesión 8) e integrado en esta versión. Compara tres clasificadores sobre `adultos_salud` para predecir `HeartDisease`: árbol de decisión (profundidad 4), Naive Bayes (`GaussianNB`) y KNN (k = 5, con `StandardScaler`).

- Lee `data/limpio/adultos_salud_limpio.csv` (si no existe, usa el original con una limpieza mínima).
- Usa una muestra de 40.000 filas (`MUESTRA`; con `None` usa todas, pero KNN tarda muchos minutos).
- Separa 75 % entrenamiento y 25 % prueba (`random_state=42`) y guarda los resultados en `data/resultados/`.

**Resultados** (ejecutados en este proyecto; responder siempre "No" acierta 0,911):

| Modelo | Aciertos (prueba) | Precisión | Enfermos detectados (recall) | F1 |
|---|---|---|---|---|
| Árbol (prof. 4) | 0,912 | 0,561 | 0,062 | 0,112 |
| Naive Bayes | 0,703 | 0,200 | 0,780 | 0,318 |
| KNN (k = 5) | 0,903 | 0,372 | 0,133 | 0,196 |

**Lectura:**

- Ningún modelo mejora de forma apreciable los aciertos de responder siempre "No" (el árbol llega a 0,912).
- Naive Bayes es el que detecta más enfermos (78 %) pero con muchas falsas alarmas (precisión 20 %); por eso tiene el mejor F1.
- KNN sin escalar da 0,904 y escalado 0,903: en estos datos el escalado casi no cambió los aciertos.
- Con k = 1, KNN memoriza (entrenamiento 1,000, prueba 0,866); con k = 101 acierta como el modelo "siempre No" y detecta apenas el 2 % de los enfermos.

**Aclaraciones:**

- La comparación en igualdad de condiciones con el árbol único está en la sección 7b.
- El árbol de esta comparación (profundidad 4, sin `class_weight`) es distinto del árbol único del proyecto (`arbol_general.py`, sección 6): aquí sirve solo para comparar modelos.
- El texto inicial del archivo habla de una carpeta `proyecto_limpieza/` con `transformaciones.py` en la raíz; en este proyecto está en `herramientas/`. El archivo no se modificó.

---

## 7b. Naive Bayes y KNN frente al árbol único (`modelado_unificado.py` y cuaderno, parte 9)

Sesión 8. A diferencia de `modelado.py`, aquí los modelos compiten sobre **la misma tabla unificada y la misma partición** del árbol único (`arbol_general.py`), de modo que la comparación es en igualdad de condiciones (esto cierra un pendiente de la sección 11).

**Resultados** (prueba: 740 personas, 65 de riesgo alto; responder siempre "riesgo normal" acierta 0,912):

| Modelo | Aciertos (prueba) | Precisión | Riesgo alto detectado (recall) | F1 |
|---|---|---|---|---|
| Árbol (prof. 4, `balanced`) | 0,846 | 0,310 | **0,615** | 0,412 |
| Naive Bayes | 0,909 | 0,333 | 0,031 | 0,056 |
| Naive Bayes (`priors` 50/50) | 0,708 | 0,184 | 0,677 | 0,289 |
| KNN (k = 5, escalado) | 0,934 | **0,786** | 0,338 | **0,473** |

**Lectura:**

- **Ningún modelo gana en todo.** El árbol detecta más y es el único explicable; KNN tiene mejor F1 y pocas falsas alarmas; Naive Bayes solo sirve si se corrige su *a priori*.
- **Naive Bayes** casi no detecta nada con su configuración normal porque solo el 9 % es riesgo alto. Se comprobó que **no** era culpa de los `-1` ni de las variables 0/1 (con solo `edad` e `imc` el recall fue 0,015). Con `priors=[0.5, 0.5]` (equivalente al `class_weight="balanced"` del árbol) sube a 0,677 a cambio de muchas falsas alarmas.
- **KNN necesita escalar:** sin escalar detecta 0,062; escalado, 0,338.
- **Efecto de k:** con k = 1 memoriza (entrenamiento 0,999, prueba 0,900); con k = 31 y 51 el riesgo alto detectado baja a 0,20-0,23.
- **Por dataset:** KNN detecta 0 % del riesgo alto en `adultos_salud` y 90,9 % en `insurance` (mismo sesgo que el árbol: aprende donde las 5 variables explican el riesgo). Naive Bayes con `priors` balanceados es el que más detecta en `adultos_salud` (85,7 %).
- La prueba es pequeña: diferencias como F1 0,412 vs 0,473 son orientativas.

## 7c. Clustering de `adultos_salud` e `insurance` (`clustering.py` y cuaderno, parte 10)

Sesión 9 (aprendizaje **no supervisado**). Se agrupa con 5 variables escaladas (`edad` aproximada, `BMI`, `PhysicalHealth`, `MentalHealth`, `SleepTime`). `HeartDisease` **no entra** al clustering: se usa después solo para perfilar los grupos.

**Elección de k:** la silueta es máxima en k = 2 (0,409), pero ese corte solo separa "buena salud" de "muchos días de mala salud" (demasiado grueso). De k = 3 a 8 la silueta es casi igual (0,23-0,26) y el codo no es nítido. Se eligió **k = 4** por interpretabilidad y porque da grupos con riesgo muy distinto. Es una decisión razonada, no un hallazgo.

| Segmento | Personas | Edad media | Enfermedad cardíaca |
|---|---|---|---|
| Jóvenes sanos | 106.521 (35 %) | 37 | 1,7 % |
| Adultos con malestar emocional (~23 días de mala salud mental al mes) | 27.596 (9 %) | 45 | 6,9 % |
| Mayores con mala salud física (~27 días; 59 % con dificultad para caminar) | 27.171 (9 %) | 62 | **23,0 %** |
| Mayores estables | 140.429 (47 %) | 68 | 12,3 % |

Promedio general: 9,0 %. Los grupos separan el riesgo **sin haber usado la enfermedad**, pero es asociación, no causa.

**Contraste jerárquico:** dendrograma (`linkage="ward"`) sobre una muestra de 60 personas; un corte en 4 ramas es compatible con k = 4, pero con 60 personas lo apoya y no lo demuestra.

### Clustering de `insurance` (cuaderno 10.7)

500.000 personas, agrupadas con `age`, `bmi`, `children` y `fuma` (0/1); `charges` queda fuera y se usa solo para perfilar. La silueta es máxima en k = 2 (0,345), pero solo separa fumadores de no fumadores (algo que el EDA ya mostraba); se eligió **k = 4** (silueta 0,302, la segunda mejor) porque parte a los no fumadores por edad e hijos.

| Segmento | Personas | Edad media | Costo medio |
|---|---|---|---|
| Jóvenes no fumadores | 145.264 (29 %) | 26 | 4.201 |
| Fumadores | 102.862 (21 %) | 38 | **32.230** |
| Familias con hijos (2,7 hijos en promedio) | 114.332 (23 %) | 41 | 9.451 |
| No fumadores maduros | 137.542 (28 %) | 53 | 11.683 |

Costo medio general: 13.226. **Los fumadores forman un solo grupo con cualquier k**, porque al escalar una variable 0/1 pesa mucho en la distancia: confirma que fumar domina el costo, pero el clustering no logra partir a los fumadores. El IMC casi no distingue grupos (coherente con que la obesidad solo encarece junto con fumar). Los archivos quedan en `graficas/clustering/insurance/` y `data/resultados/clustering_insurance_*.csv`.

**Escalado en clustering:** el escalador se ajusta con todos los datos porque aquí no hay partición de prueba que proteger (no se evalúa contra etiquetas). En los modelos supervisados sí va dentro de un `Pipeline`.

**Límites (`adultos_salud`):** silueta baja (~0,25); `PhysicalHealth`/`MentalHealth` valen casi siempre 0 o 30; K-means asume grupos redondeados y es sensible a extremos. Los nombres de los segmentos dependen de estos datos y de la semilla (los grupos se renumeran por edad media para que sean estables): si cambian las variables o k, hay que revisarlos.

## 7d. Clasificadores con validación cruzada (`modelado_justo.py`)

`modelado.py` (sección 7) y `modelado_unificado.py` (7b) comparan sobre **una sola partición**. `modelado_justo.py` repite la comparación sobre `adultos_salud` con **validación cruzada de 5 particiones**, de modo que cada número es un promedio y no una sola prueba:

- **Mismo trato frente al desbalance:** el árbol y la regresión logística usan `class_weight="balanced"` y Naive Bayes usa `priors=[0.5, 0.5]`. KNN no tiene esa opción, así que su recall no es del todo comparable.
- **El escalador va dentro de un `Pipeline`:** se ajusta solo con el entrenamiento de cada partición (sin fuga de información).
- Métricas: enfermos detectados (recall), precisión, F1 y ROC AUC (que no depende del umbral).

Resultados (40.000 personas de `adultos_salud`; responder siempre "No" da recall 0 y ROC AUC 0,50):

| Modelo | Enfermos detectados | Precisión | F1 | ROC AUC |
|---|---|---|---|---|
| Árbol (prof. 4, balanced) | 0,68 | 0,19 | 0,30 | 0,72 |
| Naive Bayes (50/50) | 0,85 | 0,17 | 0,28 | 0,80 |
| **Regresión logística (balanced)** | 0,77 | 0,22 | **0,34** | **0,83** |
| KNN (k=5) | 0,13 | 0,36 | 0,19 | 0,70 |
| KNN (k=31) | 0,04 | 0,54 | 0,07 | 0,80 |

**Lectura:** la regresión logística separa mejor a los enfermos que el árbol (ROC AUC 0,83 frente a 0,72); el árbol conserva su ventaja de ser fácil de explicar. Es coherente con 7b: ningún modelo gana en todo. Salida: `data/resultados/modelado_justo.csv` y `graficas/modelado_justo.png`.

## 7e. Predecir costos (`modelado_regresion.py`)

Para `insurance` y `Medicalpremium` el objetivo es un número, así que se comparan regresión lineal, árbol (prof. 4) y Random Forest con las métricas que corresponden: MAE (error promedio), RMSE y R², con validación cruzada de 5 particiones. Es el "siguiente paso" que propone el informe EDA. Para `insurance` se usa una muestra de 50.000 filas.

| Modelo | `insurance` (error · R²) | `Medicalpremium` (error · R²) |
|---|---|---|
| Siempre el promedio (referencia) | 8.987 · 0,00 | 4.953 · 0,00 |
| Regresión lineal | 3.955 · 0,76 | 2.721 · 0,62 |
| Árbol (prof. 4) | 2.425 · 0,87 | 1.835 · 0,73 |
| **Random Forest** | **2.203 · 0,88** | **1.156 · 0,79** |

**Lectura:** el Random Forest es el mejor en los dos y el árbol supera a la regresión lineal, coherente con el informe EDA (la obesidad solo encarece cuando se combina con fumar, y un modelo lineal no captura esa interacción). Salida: `data/resultados/modelado_regresion.csv` y `graficas/modelado_regresion.png`.

---

## 8. Cuaderno de Jupyter

`proyecto_mineria.ipynb` corre el proyecto celda por celda y genera cada gráfico:

1. Configuración y carga de datos
2. Calidad de los datos (vacíos, repetidas, categorías, extremos)
3. Limpieza
4. Procesamiento por lotes
5. Gráficos exploratorios
6. Escalar y codificar
7. Unir los tres datasets
8. El árbol único (incluye 8.9b, la limitación del `-1`, y 8.11, la estabilidad con 20 muestras)
9. Naive Bayes y KNN frente al árbol
10. Clustering (K-means y jerárquico)
11. Conclusiones

Debe estar en la raíz del proyecto (junto a `data/`). Guarda sus resultados en `data/cuaderno/` y `graficas/cuaderno/` (las gráficas nuevas son la 19 a la 27). Necesita `scipy` (ya está en `requirements.txt`). **Sin fuga de datos:** en la parte 6 se separa entrenamiento/prueba *antes* y el escalador se ajusta solo con entrenamiento (en la parte 9, KNN usa un `Pipeline` que hace lo mismo).

---

## 9. Presentaciones y documento

- `presentacion/presentacion_decisiones.html` y `presentacion/presentacion_proyecto_completo.html` (7 diapositivas cada una; flechas para navegar, `F` pantalla completa, `Ctrl+P` para PDF).
- `docs/decisiones_del_codigo.docx` (partes críticas del código y por qué se tomaron esas decisiones).

> **Actualizados parcialmente (v1.13):** las dos presentaciones y el Word cuentan el árbol único con sus rangos y la comparación con validación cruzada (7d y 7e). **Aún no incluyen** Naive Bayes y KNN sobre la tabla unificada (7b) ni el clustering (7c).

---

## 10. Trabajo de otras conversaciones

| Elemento | Qué es | Estado |
|---|---|---|
| `modelado.py` | Comparación de árbol, Naive Bayes y KNN sobre `adultos_salud`. | **Integrado** en la versión 1.9 (sección 7). |
| `informes/EDA_insurance.html` | Informe exploratorio (EDA) de `insurance`, con la estructura oficial de 6 secciones: portada y contexto, calidad de datos, univariado, bivariado/multivariado, hallazgos y recomendaciones, y anexos (el código está en las celdas del propio notebook). | **Integrado** en la versión 1.10, sin modificar. |

### Qué concluye el informe EDA de `insurance`

1. **Fumar es el factor que más separa los costos:** un fumador cuesta en promedio unas 4 veces más que un no fumador, y `smoker` tiene correlación 0,79 con `charges`.
2. **La obesidad solo se vuelve cara combinada con fumar:** en no fumadores sube el costo medio un 11 %; en fumadores, un 95 %.
3. **Los costos muy altos no son errores:** el 97,8 % de los valores extremos de `charges` son fumadores, por eso se conservan.

### Ojo: dos versiones de `insurance`

El informe se hizo con una versión de **1.337 filas** (el conjunto original de Kaggle, tras quitar 1 fila repetida). El archivo `data/limpio/insurance_limpio.csv` de este proyecto tiene **500.000 filas**. Las cifras son muy parecidas, pero no idénticas:

| Cifra | Informe (1.337 filas) | `data/limpio` (500.000 filas) |
|---|---|---|
| Porcentaje de fumadores | 20,5 % | 20,6 % |
| Costo medio (`charges`) | 13.279 | 13.226 |
| Correlación fumar - costo | 0,79 | 0,79 |
| Valores extremos de `charges` | 10,4 % | 11,2 % |

Si el informe se vuelve a ejecutar sobre el archivo de 500.000 filas, las cifras exactas del texto cambiarán. Hay que decidir con cuál de las dos versiones se entrega el proyecto.

---

## 11. Historial de cambios

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0 | — | Proyecto base: limpieza con `DataCleaner` y cleaners por dataset, `herramientas/`, `transformaciones.py`. |
| 1.1 | 25 oct 2026 | **Procesamiento por lotes:** `limpieza/lotes_cleaner.py`, `herramientas/transformaciones_lotes.py`, dos funciones nuevas en `io_datos.py`, `main_lotes.py` y `benchmark_lotes.py`. |
| 1.2 | 25-26 oct 2026 | **Árboles por dataset con gráficas:** `arbol_clasificacion.py` y `arbol_regresion.py`; gráficas con etiquetas en español. |
| 1.3 | 27 oct 2026 | **Presentaciones HTML** (decisiones y proyecto completo), con lenguaje sencillo. |
| 1.4 | 27 oct 2026 | **Cuaderno de Jupyter** `proyecto_mineria.ipynb` (con un solo árbol, sobre `adultos_salud`). |
| 1.5 | 29 oct 2026 | **Documento Word** `docs/decisiones_del_codigo.docx`. |
| 1.6 | 17 nov 2026 | **Árbol único sobre los tres datasets:** `arbol_general.py`; el cuaderno se amplía con las partes 7 a 9 (unir datasets, árbol único, conclusiones). |
| 1.7 | 21 nov 2026 | **Este paquete:** `README_PROYECTO.md` actualizado (reemplaza a `LEEME.md` y al README anterior), con estructura real, resultados e historial de cambios. |
| 1.8 | 21 nov 2026 | **README:** se registra el trabajo de otras conversaciones (`modelado.py` e informe EDA), cuyos archivos no están en este zip (sección 10). Sin cambios de código. |
| 1.9 | 21 nov 2026 | **Integración de `modelado.py`** (comparación de árbol, Naive Bayes y KNN) y de su salida en `data/resultados/`; nueva sección 7 con sus resultados. Sin cambios en el resto del código. |
| 1.10 | 21 nov 2026 | **Integración del informe EDA de `insurance`** (`informes/EDA_insurance.html`, sin modificar); sección 10 con sus conclusiones y la advertencia sobre las dos versiones de `insurance`. Corrección de dos referencias de sección en la estructura. Sin cambios de código. |
| 1.11 | 21 nov 2026 | **Sesiones 8 y 9 integradas al proyecto:** `modelado_unificado.py` (árbol, Naive Bayes y KNN sobre la misma partición del árbol único), `clustering.py` (K-means, codo, silueta, perfiles y dendrograma sobre `adultos_salud`), partes 9 y 10 nuevas en el cuaderno (las conclusiones pasan a ser la parte 11), secciones 7b y 7c, y `scipy` en `requirements.txt`. El resto del código no cambió. |
| 1.12 | 21 nov 2026 | **Escalado sin fuga de datos** en la parte 6 del cuaderno (se separa antes y el escalador se ajusta solo con entrenamiento); **demostración de la limitación del `-1`** (cuaderno 8.9b); **clustering de `insurance`** (`clustering.py` ahora procesa los dos datasets; sus salidas pasan a `graficas/clustering/<dataset>/` y `data/resultados/clustering_<dataset>_*.csv`) y parte 10.7 del cuaderno. |
| 1.13 | 22 nov 2026 | **Fusión de dos líneas de trabajo.** Se integran a la v1.12 (sesiones 8 y 9) las mejoras hechas en paralelo: `config.py` (columnas y rutas en un solo lugar); `pruebas/` (6 pruebas automáticas de los lotes); `modelado_justo.py` y `modelado_regresion.py` (secciones 7d y 7e); estabilidad del árbol único con 20 muestras (cuaderno 8.11, gráfica 27; el resultado se cita como rango); corrección de la carpeta raíz de `transformaciones.py`; presentaciones y Word actualizados al árbol único. `arbol_general.py` conserva `construir_tabla_unificada()` (la usa `modelado_unificado.py`) y agrega `armar_tablas()` y `muestrear()`. |
| 1.14 | 22 nov 2026 | **`main.py`:** punto de entrada único con los pasos `limpieza`, `lotes`, `transformar`, `arbol`, `modelos`, `comparar`, `clustering`, `pruebas` y `todo`. Es una reconstrucción (el original no se recibió). Probado: cada paso por separado y `todo` terminan sin errores; la limpieza no reescribe `data/limpio/` sin `--sobrescribir`. Presentación y Word actualizados. |

### Pendientes

- Decidir qué versión de `insurance` usa el proyecto: el informe EDA usa 1.337 filas y `data/limpio/insurance_limpio.csv` tiene 500.000 (ver sección 10).
- Si se necesita un informe EDA por dataset, faltan los de `adultos_salud` y `Medicalpremium` (el cuaderno cubre parte de esa exploración).
- Actualizar las presentaciones HTML y el Word para incluir Naive Bayes y KNN sobre la tabla unificada (7b) y el clustering (7c).
- Probar el `cluster` como variable de entrada del modelo supervisado (idea de la sesión 9; no se probó).
- `herramientas/transformaciones.py` ajusta sus escaladores con todos los datos (`fit_transform` sobre toda la matriz): sirve para demostrar, pero para modelar habría que ajustarlos solo con entrenamiento. `transformaciones_lotes.py` aprende la media y la desviación de toda la columna por diseño. Solo se corrigió su carpeta raíz (v1.13); su escalado no se modificó.
- Comparar este `main.py` con el original del proyecto y fusionar lo que falte (hace falta subir el archivo).
- Probar el procesamiento por lotes con `insurance` y `Medicalpremium` originales (solo se ejecutó con datos reales en `adultos_salud`).
- Elegir la profundidad del árbol único con validación cruzada repetida (hoy se eligió con una sola muestra).
- `modelado.py` (original) sigue usando un árbol sin `class_weight`; para comparar modelos se recomienda `modelado_unificado.py` (7b) o `modelado_justo.py` (7d).
