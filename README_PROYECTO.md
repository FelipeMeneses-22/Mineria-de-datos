# Proyecto: Predicción de Costos Médicos — Limpieza de datos

Reorganización de `limpieza_insurance.py` y `limpieza_medicalpremium.py`
en un proyecto con POO y principios SOLID. La lógica y los resultados
son **exactamente los mismos** que en los scripts originales; solo
cambió cómo está organizado el código.

## Estructura

```
proyecto_limpieza/
├── main.py                          # Punto de entrada: corre ambos pipelines
├── utils/                           # Funciones puras y reutilizables
│   ├── diagnostico.py               #   -> diagnostico()
│   ├── estadisticas.py              #   -> resumen_iqr()
│   └── io_datos.py                  #   -> leer_csv() / guardar_csv()
├── cleaners/                        # Clases (POO)
│   ├── base_cleaner.py              #   -> DataCleaner (clase abstracta)
│   ├── insurance_cleaner.py         #   -> InsuranceCleaner
│   └── medicalpremium_cleaner.py    #   -> MedicalPremiumCleaner
└── data/
    ├── raw/                         # Aquí van insurance.csv y Medicalpremium.csv
    └── limpio/                      # Aquí se exportan los *_limpio.csv
```

## Cómo correrlo

1. Copia `insurance.csv` y `Medicalpremium.csv` dentro de `data/raw/`.
2. Desde la carpeta `proyecto_limpieza/`, ejecuta:

```bash
python main.py
```

Esto imprime el mismo diagnóstico paso a paso que los scripts
originales y genera `data/limpio/insurance_limpio.csv` y
`data/limpio/Medicalpremium_limpio.csv`.

## Diseño: patrón "Template Method"

`DataCleaner` (en `base_cleaner.py`) define el orden fijo del pipeline
visto en clase:

```
Diagnóstico -> Duplicados -> Categorías -> Nulos -> Outliers -> Validación
```

en un único método `ejecutar()`. Los pasos que son **iguales** para
cualquier dataset (duplicados, nulos, diagnóstico) están implementados
una sola vez en la clase base. Los pasos que **dependen del dataset**
(qué columnas categóricas revisar, cómo tratar cada outlier) son
métodos abstractos que `InsuranceCleaner` y `MedicalPremiumCleaner`
implementan cada uno a su manera.

## Cómo se aplica cada principio SOLID

| Principio | Dónde se ve |
|---|---|
| **S** — Responsabilidad única | `base_cleaner.py` solo orquesta el pipeline; leer/escribir CSV vive en `io_datos.py`; el diagnóstico en `diagnostico.py`; el cálculo IQR en `estadisticas.py`. Cada archivo tiene una sola razón para cambiar. |
| **O** — Abierto/cerrado | Para limpiar un tercer dataset se crea una clase nueva en `cleaners/` (heredando de `DataCleaner`) y se agrega a la lista en `main.py`. No hay que tocar `base_cleaner.py` ni las clases existentes. |
| **L** — Sustitución de Liskov | `main.py` trata a `InsuranceCleaner` y `MedicalPremiumCleaner` como simples `DataCleaner`: llama `.ejecutar()` sin saber cuál es cuál. Cualquier subclase nueva puede usarse igual. |
| **I** — Segregación de interfaces | La clase base exige tres métodos abstractos pequeños y específicos (`verificar_categorias`, `tratar_outliers`, `resumen_especifico`) en lugar de un único método gigante que mezcle todo. |
| **D** — Inversión de dependencias | `DataCleaner` depende de funciones de `utils/` (abstracciones simples: "leer un csv", "calcular IQR"), no de detalles concretos de un dataset. |

## Diferencias intencionales que se conservaron

- En `insurance.csv` se muestran las filas duplicadas completas (útil
  para justificar el patrón MCAR); en `Medicalpremium.csv` solo se
  cuenta cuántos duplicados hay. Esto se resolvió con un método "hook"
  (`_mostrar_duplicados`) que `InsuranceCleaner` sobreescribe y que
  `MedicalPremiumCleaner` deja con el comportamiento por defecto.
- Todos los comentarios de justificación de negocio (por qué se
  conservan los outliers de `bmi`, `charges`, `Weight`, `PremiumPrice`,
  etc.) se conservaron tal cual, ahora dentro del método
  `tratar_outliers` de cada clase.
