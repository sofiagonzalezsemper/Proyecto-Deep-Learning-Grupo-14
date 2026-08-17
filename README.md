# Asistente de Sentimiento Financiero  
## FinBERT + Human-in-the-Loop

Proyecto grupal del curso de **Deep Learning**: un asistente que clasifica el sentimiento —positivo, negativo o neutral— de noticias y frases financieras, combinando un modelo de Deep Learning de última generación con supervisión humana antes de que cualquier predicción sea utilizada para tomar decisiones.

### Equipo

- Santiago Vero
- Sofía González
- [Nombre del tercer integrante]

---

## Índice

1. [Problema y motivación](#problema-y-motivación)
2. [Arquetipo de producto ML](#arquetipo-de-producto-ml)
3. [Dataset](#dataset)
4. [Ciclo de vida del proyecto](#ciclo-de-vida-del-proyecto)
5. [Modelos evaluados](#modelos-evaluados)
6. [Arquitectura final: FinBERT](#arquitectura-final-finbert)
7. [Explicabilidad con LIME](#explicabilidad-con-lime)
8. [Prototipo en Streamlit](#prototipo-en-streamlit)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Instalación y uso](#instalación-y-uso)
11. [Desafíos y aprendizajes](#desafíos-y-aprendizajes)
12. [Consideraciones de producción](#consideraciones-de-producción)
13. [Video de presentación](#video-de-presentación)
14. [Licencia](#licencia)

---

## Problema y motivación

Los analistas financieros y equipos de research necesitan leer y clasificar constantemente noticias y reportes para tomar decisiones de inversión:

> ¿Esta frase indica que la empresa va bien, mal o presenta información neutral?

Hacer esta clasificación manualmente a gran escala es lento y difícil de escalar. Sin embargo, confiar ciegamente en un modelo automático sin supervisión también es riesgoso, ya que un error de clasificación puede derivar en una mala decisión financiera.

Antes de optar por Deep Learning, se evaluaron alternativas más simples:

- Reglas basadas en diccionarios de sentimiento financiero.
- Modelos clásicos de Machine Learning sobre características de texto.
- Representaciones mediante TF-IDF combinadas con una red neuronal MLP.

Estas alternativas funcionan razonablemente bien, pero el lenguaje financiero contiene matices, contexto y negaciones que una bolsa de palabras no siempre puede capturar correctamente.

Por ejemplo:

- “Las ganancias aumentaron”.
- “Las ganancias no aumentaron”.

Por eso, una arquitectura capaz de comprender el contexto —Deep Learning y, particularmente, los Transformers— resulta adecuada para este problema.

En términos de impacto frente a costo, el proyecto presenta:

- **Alto impacto:** ahorra tiempo de análisis y permite estandarizar criterios.
- **Bajo costo de construcción:** utiliza un dataset público, Financial PhraseBank, y un modelo preentrenado especializado en finanzas, FinBERT, en lugar de entrenar toda la arquitectura desde cero.

---

## Arquetipo de producto ML

El arquetipo seleccionado es **Human-in-the-Loop**.

El modelo:

1. Predice el sentimiento de una frase financiera.
2. Informa el nivel de confianza de la predicción.
3. Explica localmente el resultado mediante LIME.
4. Permite que un analista humano confirme o corrija la clasificación.

La predicción no se utiliza para tomar decisiones hasta que es revisada por una persona.

Se eligió este arquetipo, en lugar de un sistema completamente autónomo, porque en el dominio financiero los errores de clasificación pueden tener consecuencias reales. Además, el modelo puede equivocarse ante frases ambiguas o situaciones fuera de su distribución de entrenamiento.

La IA busca acelerar y asistir el trabajo del analista, no reemplazarlo sin supervisión.

---

## Dataset

Se utilizó **Financial PhraseBank**, un dataset de frases extraídas de noticias financieras en inglés.

### Características principales

- **Registros originales:** 2.264 frases.
- **Registros después de eliminar duplicados:** 2.259.
- **Clases:** Negativo, Neutral y Positivo.
- **Partición:** 70% entrenamiento, 15% validación y 15% prueba.
- **Estrategia de partición:** estratificada.
- **Tratamiento del desbalance:** pesos de clase durante el entrenamiento.

### Distribución de clases

| Clase | Proporción |
|---|---:|
| Neutral | 61% |
| Positivo | 25% |
| Negativo | 13% |

El dataset presenta un desbalance importante, especialmente en la clase negativa. Por este motivo, se utilizaron pesos de clase para evitar que el modelo favoreciera excesivamente la clase mayoritaria.

---

## Ciclo de vida del proyecto

### 1. Planificación

Se reutilizó el problema trabajado en prácticas individuales anteriores: la clasificación de sentimiento financiero.

Esto permitió aprovechar:

- Un dataset previamente validado.
- El conocimiento adquirido en dos iteraciones anteriores.
- Resultados de modelos de distinta complejidad.
- Una base de comparación consistente.

Desde el inicio se definieron:

- El arquetipo **Human-in-the-Loop**.
- El desarrollo de un prototipo en Streamlit.
- La distribución de roles dentro del equipo.

### 2. Recolección y preparación de datos

Las principales tareas fueron:

- Carga del dataset Financial PhraseBank.
- Limpieza de datos.
- Análisis exploratorio.
- Eliminación de registros duplicados.
- Análisis del desbalance de clases.
- Partición estratificada en entrenamiento, validación y prueba.

### 3. Exploración y refinamiento del modelo

Se evaluaron tres arquitecturas, avanzando desde una solución clásica hacia modelos de mayor complejidad y adecuación al dominio:

1. MLP con TF-IDF.
2. Transformer entrenado desde cero.
3. FinBERT mediante transfer learning.

### 4. Evaluación

Todos los modelos se compararon utilizando:

- El mismo conjunto de prueba.
- La misma semilla aleatoria.
- Accuracy.
- F1 macro.
- F1 weighted.

Esto permitió realizar comparaciones consistentes y justas entre las distintas iteraciones.

---

## Modelos evaluados

| Métrica | MLP + TF-IDF | Transformer desde cero | FinBERT |
|---|---:|---:|---:|
| Accuracy | 87,32% | 85,25% | **99,12%** |
| F1 Macro | 83,35% | 79,87% | **98,62%** |
| F1 Weighted | 87,40% | 85,03% | **99,12%** |

### MLP + TF-IDF

Se utilizó como baseline clásico y sirvió como piso de comparación para las arquitecturas posteriores.

A pesar de su menor complejidad, obtuvo resultados sólidos.

### Transformer entrenado desde cero

El Transformer entrenado desde cero no logró superar al baseline.

Con aproximadamente 1.580 frases de entrenamiento, el volumen de datos no fue suficiente para que el modelo aprendiera representaciones y embeddings propios de manera robusta. Como resultado, el modelo presentó sobreajuste y una menor capacidad de generalización.

### FinBERT mediante transfer learning

FinBERT superó ampliamente a las otras dos alternativas.

Partir de un modelo que ya conoce el lenguaje financiero y ajustarlo con el dataset específico del proyecto resultó mucho más efectivo que entrenar una arquitectura compleja desde cero.

Los resultados confirman que, cuando se trabaja con un dataset reducido, el transfer learning desde un modelo especializado en el dominio puede ofrecer una mejora sustancial.

---

## Arquitectura final: FinBERT

Se seleccionó [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert), un modelo basado en BERT y preentrenado específicamente con lenguaje financiero.

BERT utiliza bloques Transformer encoder con mecanismos de self-attention bidireccional, lo que le permite analizar cada palabra considerando tanto el contexto anterior como el posterior.

Sobre este modelo se realizó un fine-tuning supervisado completo con el dataset del proyecto.

### Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Modelo base | `ProsusAI/finbert` |
| Learning rate | `2e-5` |
| Optimizador | AdamW |
| Épocas máximas | 8 |
| Early stopping | Sí |
| Weight decay | Sí |
| Función de pérdida | Ponderada por clase |
| Longitud máxima | `MAX_LENGTH = 58` |

### Mapeo de clases

| Valor | Clase |
|---:|---|
| 0 | Positivo |
| 1 | Negativo |
| 2 | Neutral |

El mapeo de clases requirió especial atención porque el orden de las etiquetas del dataset propio no coincidía necesariamente con el orden interno utilizado por FinBERT.

---

## Explicabilidad con LIME

Para que el flujo Human-in-the-Loop resulte útil, el modelo no solo debe predecir: también debe explicar su predicción.

Se utilizó **LIME —Local Interpretable Model-Agnostic Explanations—** como técnica de explicabilidad local.

LIME genera variaciones de una frase, perturbando o eliminando palabras, y observa cómo estos cambios afectan la predicción. De esta manera, puede identificar qué palabras empujaron al modelo hacia cada clase.

La explicación permite que el analista:

- Identifique las palabras más influyentes.
- Evalúe si la lógica del modelo parece razonable.
- Detecte explicaciones dudosas o basadas en términos irrelevantes.
- Decida si confirma o corrige la predicción.

La explicabilidad es una parte central del producto, ya que permite que el usuario confíe —o desconfíe— de una predicción puntual en lugar de aceptarla a ciegas.

---

## Prototipo en Streamlit

El prototipo implementa el siguiente flujo:

```text
Input de texto
      ↓
Predicción de sentimiento
      ↓
Nivel de confianza
      ↓
Explicación mediante LIME
      ↓
Confirmación o corrección humana
      ↓
Registro en historial exportable
```

La aplicación contiene tres pestañas principales.

### 1. Frase individual

El usuario ingresa una frase financiera.

La aplicación muestra:

- Sentimiento predicho.
- Nivel de confianza.
- Probabilidades asociadas a cada clase.
- Palabras más influyentes según LIME.
- Opción para confirmar o corregir la clasificación.

### 2. Lote CSV

El usuario puede cargar un archivo CSV con varias frases.

El modelo procesa y clasifica todas las frases, facilitando su uso en escenarios de mayor volumen.

### 3. Historial de decisiones

La aplicación registra las predicciones revisadas por el analista, incluyendo las confirmaciones y correcciones realizadas.

El historial puede exportarse a CSV. En una implementación futura, estos datos podrían utilizarse para:

- Analizar los errores frecuentes del modelo.
- Detectar cambios en la distribución de los datos.
- Construir un nuevo dataset supervisado.
- Reentrenar y mejorar el modelo.

---

## Estructura del repositorio

```text
.
├── notebooks/
│   ├── practica1_deep_learning.ipynb
│   ├── practica2_transformer.ipynb
│   └── practica3_finbert_transfer_learning.ipynb
│
├── prototipo/
│   ├── app.py
│   ├── model_utils.py
│   ├── explain_utils.py
│   ├── modelo/
│   │   └── finbert_sentimiento_financiero_practica3/
│   ├── ejemplos.csv
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

### Descripción de los archivos principales

| Archivo | Descripción |
|---|---|
| `practica1_deep_learning.ipynb` | Baseline con MLP y TF-IDF |
| `practica2_transformer.ipynb` | Transformer entrenado desde cero |
| `practica3_finbert_transfer_learning.ipynb` | Fine-tuning de FinBERT y explicabilidad |
| `app.py` | Interfaz principal de Streamlit |
| `model_utils.py` | Carga del modelo y función de predicción |
| `explain_utils.py` | Generación de explicaciones con LIME |
| `ejemplos.csv` | Frases de ejemplo para el procesamiento en lote |
| `requirements.txt` | Dependencias del prototipo |

> **Importante:** antes de publicar el repositorio, se deben ajustar los nombres de carpetas y archivos de esta sección para que coincidan exactamente con la estructura real.

---

## Instalación y uso

### Requisitos previos

- Python 3.9 o superior.
- Git.
- Jupyter Notebook o JupyterLab.
- Conexión a internet para descargar el modelo base la primera vez, en caso de no contar con el checkpoint local.

### Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd NOMBRE_DEL_REPOSITORIO
```

Reemplazar `URL_DEL_REPOSITORIO` y `NOMBRE_DEL_REPOSITORIO` por los valores correspondientes.

### Ejecución de los notebooks

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Iniciar Jupyter Notebook:

```bash
jupyter notebook
```

Luego, abrir el notebook correspondiente a cada práctica para reproducir el entrenamiento y la evaluación de cada modelo.

### Ejecución del prototipo

Ingresar a la carpeta del prototipo:

```bash
cd prototipo
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Copiar el checkpoint de FinBERT entrenado en la Práctica 3 —generado mediante `trainer.save_model()` y `tokenizer.save_pretrained()`— dentro de:

```text
prototipo/modelo/finbert_sentimiento_financiero_practica3/
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

Si el checkpoint entrenado no está disponible, la aplicación utiliza automáticamente el modelo base de FinBERT descargado desde Hugging Face Hub. Esto requiere conexión a internet durante la primera ejecución.

---

## Desafíos y aprendizajes

### Falta de datos para entrenar un Transformer desde cero

El Transformer entrenado desde cero no logró superar al baseline simple debido al tamaño reducido del dataset.

Esto obligó a pivotar hacia transfer learning, aprovechando las representaciones previamente aprendidas por un modelo especializado.

### Mapeo de etiquetas

Fue necesario resolver cuidadosamente el mapeo entre:

- Las etiquetas utilizadas en el dataset propio.
- El orden interno de clases utilizado por FinBERT.
- Las etiquetas mostradas en el prototipo.

Un mapeo incorrecto habría producido resultados aparentemente válidos, pero semánticamente equivocados.

### Desarrollo del prototipo

El prototipo se desarrolló y probó end-to-end utilizando un modelo de reemplazo o dummy en un entorno sin acceso directo a Hugging Face Hub.

Se dejó documentado el procedimiento para reemplazar ese checkpoint por el modelo real antes de realizar una demostración o despliegue.

### Aprendizaje principal

> Una arquitectura más compleja no garantiza mejores resultados si no existe un volumen suficiente de datos para entrenarla correctamente.

Además, la explicabilidad no debe considerarse un accesorio. En un producto Human-in-the-Loop, es una pieza central para que la persona pueda evaluar la confiabilidad de cada predicción.

---

## Consideraciones de producción

Para llevar el proyecto más allá del contexto académico deberían considerarse los siguientes aspectos.

### Monitoreo

Sería necesario monitorear continuamente:

- Distribución de las predicciones.
- Confianza del modelo.
- Correcciones realizadas por los analistas.
- Rendimiento por clase.
- Cambios en el lenguaje utilizado en las noticias.

Esto permitiría detectar **data drift** o **concept drift**, ya que el lenguaje financiero y los temas relevantes pueden cambiar con el tiempo.

### Deuda técnica

Actualmente, la aplicación puede depender de la descarga del modelo base desde Hugging Face Hub.

En un entorno productivo sería conveniente:

- Almacenar un artefacto propio y versionado.
- Registrar las versiones del modelo, tokenizer y dataset.
- Mantener trazabilidad entre cada modelo y sus métricas.
- Evitar dependencias externas durante la inferencia.

### Escalabilidad

Si el volumen de frases aumentara, podrían incorporarse:

- Procesamiento asíncrono.
- Clasificación en lote.
- Una API de inferencia.
- Infraestructura dedicada para servir el modelo.
- Almacenamiento persistente del historial.
- Colas de revisión para los analistas.

### Mejora continua

Las correcciones humanas podrían transformarse en nuevos ejemplos etiquetados para:

- Reentrenar periódicamente el modelo.
- Aplicar active learning.
- Detectar casos ambiguos.
- Mejorar el desempeño sobre datos reales.

---

## Video de presentación

Agregar aquí el enlace al video de presentación del proyecto:

```text
[Ver video de presentación](URL_DEL_VIDEO)
```

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT.

Consultar el archivo [`LICENSE`](LICENSE) para más información.
