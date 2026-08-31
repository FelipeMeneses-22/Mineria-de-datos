# 🏥 Análisis Predictivo de Costos Médicos (Medical Cost Personal Datasets)

**Asignatura:** Minería de datos (NRC: 95400)  
**Docente:** Esteban Ernesto Morales Castro  
**Programa:** Ingeniería en Sistemas  
**Institución:** Corporación Universitaria Minuto de Dios  
**Fecha:** 31 de Agosto de 2024

---

## 👥 Integrantes

- Jose Manuel Salgado
- Juan Felipe Meneses Rodríguez

---

## 📖 Introducción

En el sector de los seguros de salud, la determinación precisa de las primas de seguro representa un desafío crítico de gestión de riesgos y equidad financiera. Las compañías aseguradoras deben establecer tarifas que cubran los costos esperados de atención médica sin perder competitividad en el mercado. Para lograrlo, la minería de datos y el aprendizaje automático ofrecen herramientas robustas capaces de descubrir patrones no evidentes en los perfiles de los asegurados.

El dataset _Medical Cost Personal Datasets_ proporciona información individualizada sobre características demográficas (edad, sexo, número de hijos, región de residencia) y hábitos de vida (consumo de tabaco e Índice de Masa Corporal). El presente trabajo aborda la problemática de predecir la variable continua `charges` (gastos médicos facturados) a partir de estas variables explicativas. Mediante la aplicación del proceso de descubrimiento de conocimiento en bases de datos (KDD), que abarca desde la limpieza y preparación de datos hasta el modelado predictivo y evaluación, se busca entender cómo interactúan los factores de riesgo de un individuo para determinar su costo médico final.

---

## 🎯 Objetivos

### Objetivo General

Desarrollar un modelo predictivo mediante técnicas de minería de datos y aprendizaje supervisado para determinar los costos médicos individuales (`charges`) a partir de las características demográficas y hábitos de vida de los asegurados, identificando las variables de mayor impacto en la tarificación de seguros médicos.

### Objetivos Específicos

1. **Efectuar el procesamiento y análisis exploratorio de datos (EDA):** Identificar distribuciones, valores atípicos, correlaciones e interacciones entre las variables del dataset (`insurance.csv`).
2. **Determinar las variables críticas de mayor sesgo económico:** Evaluar el impacto relativo del consumo de tabaco, el Índice de Masa Corporal y la edad sobre el monto final de las primas médicas.
3. **Entrenar y comparar modelos de regresión:** Implementar y evaluar algoritmos de regresión (como Regresión Lineal Múltiple, Decision Tree Regressor y Random Forest Regressor) utilizando métricas de desempeño como el Error Cuadrático Medio, el Error Absoluto Medio y el Coeficiente de Determinación.

---

## 📊 Dataset a Emplear

El conjunto de datos utilizado fue obtenido de Kaggle: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance).

### Columnas y Atributos:

- `age`: Edad del asegurado.
- `sex`: Género (`female`, `male`).
- `bmi`: Índice de masa corporal ($kg/m^2$).
- `children`: Número de hijos/dependientes cubiertos.
- `smoker`: Si el asegurado fuma o no (`yes`, `no`).
- `region`: Zonas geográficas de EE. UU. (`northeast`, `southeast`, `southwest`, `northwest`).
- `charges`: Costo médico individual facturado por la aseguradora (Variable a predecir).

---

## 💡 Conclusiones

1. **Fumar es el factor que dispara los costos:** Lo que más influyó en el aumento de los cargos médicos fue sin duda si la persona fuma o no. En las gráficas y análisis se nota de inmediato una brecha enorme entre fumadores y no fumadores, siendo este hábito el responsable de los cobros más altos de todo el dataset.
2. **La combinación peligrosa entre obesidad y cigarrillo:** Notamos algo muy interesante al cruzar el $BMI$ (índice de masa corporal) con el hábito de fumar. En las personas que no fuman, tener un $BMI$ alto sube un poco el costo, pero no de forma drástica. Sin embargo, en los fumadores con obesidad los costos se disparan por completo. Esta relación demuestra que los factores de riesgo no actúan solos, sino que se potencian entre sí.
3. **El modelo funciona muy bien para predecir tarifas:** Al probar diferentes algoritmos, los modelos basados en árboles dieron mejores resultados que la regresión lineal básica, logrando predecir más del 80% de la variabilidad de los costos. Esto nos demuestra que aplicar minería de datos en este tipo de información sí es útil para que una aseguradora estime los costos de un cliente de forma rápida y precisa según su perfil.
